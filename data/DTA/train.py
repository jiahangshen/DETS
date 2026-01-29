import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool
from ogb.utils import smiles2graph
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from rdkit import RDLogger

# 屏蔽 RDKit 警告
RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. Dataset & Collate (显存优化版)
# ==========================================

AA_VOCAB = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","X","-","?"]
AA_TO_ID = {aa:i for i,aa in enumerate(AA_VOCAB)}

class MolProtDataset(Dataset):
    def __init__(self, csv_path, max_len=1024):
        self.df = pd.read_csv(csv_path)
        
        # 自动适配列名
        if "COMPOUND_SMILES" in self.df.columns:
            self.smiles = self.df["COMPOUND_SMILES"].tolist()
            self.seqs = self.df["PROTEIN_SEQUENCE"].tolist()
            self.labels = self.df["REG_LABEL"].astype(float).tolist()
        elif "smiles" in self.df.columns:
            self.smiles = self.df["smiles"].tolist()
            self.seqs = self.df["sequence"].tolist()
            self.labels = self.df["label"].astype(float).tolist()
        else:
            raise ValueError(f"Unknown column format in {csv_path}")
            
        self.max_len = max_len
        # 预探测图维度
        try:
            g = smiles2graph("CC")
            self.graph_in_dim = g["node_feat"].shape[1]
        except:
            self.graph_in_dim = 9 # Fallback

    def encode_seq(self, seq):
        # 只做映射和截断，不做填充！填充交给 collate_fn
        ids = [AA_TO_ID.get(ch, AA_TO_ID["?"]) for ch in seq]
        if len(ids) > self.max_len: 
            ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self): return len(self.smiles)

    def __getitem__(self, idx):
        # 1. Graph
        gdict = smiles2graph(self.smiles[idx])
        graph = Data(x=torch.tensor(gdict["node_feat"],dtype=torch.float),
                     edge_index=torch.tensor(gdict["edge_index"],dtype=torch.long),
                     edge_attr=torch.tensor(gdict["edge_feat"],dtype=torch.float) if gdict["edge_feat"] is not None else None)
        
        # 2. Sequence (Variable Length)
        input_ids = self.encode_seq(self.seqs[idx])
        
        # 3. Label
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return graph, input_ids, label

def collate_fn(batch):
    """
    动态填充 (Dynamic Padding) - 极大节省显存
    """
    graphs, input_ids_list, labels = zip(*batch)
    
    # 1. 图打包
    batched_graph = Batch.from_data_list(graphs)
    
    # 2. 序列动态填充
    batch_size = len(input_ids_list)
    # 找出当前 batch 中最长的序列
    max_batch_len = max([len(x) for x in input_ids_list])
    
    padded_ids = torch.zeros((batch_size, max_batch_len), dtype=torch.long) # 默认 0 填充
    attention_mask = torch.zeros((batch_size, max_batch_len), dtype=torch.long) # 0 表示 padding
    
    for i, seq in enumerate(input_ids_list):
        end = len(seq)
        padded_ids[i, :end] = seq
        attention_mask[i, :end] = 1 # 有内容的部分设为 1
        
    labels = torch.stack(labels)
    
    return batched_graph, padded_ids, attention_mask, labels

# ==========================================
# 2. Model
# ==========================================

class GraphEncoder(nn.Module):
    def __init__(self,in_dim,hidden_dim=128,num_layers=3,dropout=0.1):
        super().__init__()
        self.layers=nn.ModuleList()
        for i in range(num_layers):
            mlp=nn.Sequential(nn.Linear(in_dim if i==0 else hidden_dim,hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,hidden_dim))
            self.layers.append(GINConv(mlp))
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        for conv in self.layers:
            x=self.dropout(conv(x, edge_index))
        return global_mean_pool(x, batch)

class ProteinEncoder(nn.Module):
    def __init__(self,vocab_size=30,emb_dim=128,n_heads=4,n_layers=2,proj_dim=256,dropout=0.1,max_len=1024):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        self.pos_embedding=nn.Embedding(max_len,emb_dim)
        layer=nn.TransformerEncoderLayer(d_model=emb_dim,nhead=n_heads,dropout=dropout,dim_feedforward=emb_dim*4,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=n_layers)
        self.proj=nn.Sequential(nn.Linear(emb_dim,proj_dim),nn.ReLU(),nn.Dropout(dropout))

    @staticmethod
    def masked_mean(x,mask):
        mask=mask.unsqueeze(-1).type_as(x)
        return (x*mask).sum(1)/mask.sum(1).clamp(min=1e-6)

    def forward(self,input_ids,attention_mask):
        bsz,seq_len=input_ids.size()
        pos=torch.arange(seq_len,device=input_ids.device).unsqueeze(0).expand(bsz,seq_len)
        x=self.embedding(input_ids)+self.pos_embedding(pos)
        x=self.encoder(x,src_key_padding_mask=(attention_mask==0))
        return self.proj(self.masked_mean(x,attention_mask))

class MLP(nn.Module):
    def __init__(self,in_dim,hiddens,out_dim,dropout=0.1):
        super().__init__()
        layers=[]
        last=in_dim
        for h in hiddens:
            layers+=[nn.Linear(last,h),nn.ReLU(),nn.Dropout(dropout)]
            last=h
        layers.append(nn.Linear(last,out_dim))
        self.net=nn.Sequential(*layers)

    def forward(self,x):return self.net(x)

class MolProtRegressor(nn.Module):
    def __init__(self,graph_in_dim,g_hidden=128,g_layers=3,seq_vocab_size=30,seq_emb_dim=128,seq_heads=4,seq_layers=2,seq_proj_dim=256,mlp_hidden=(256,128),dropout=0.1):
        super().__init__()
        self.graph_encoder=GraphEncoder(graph_in_dim,g_hidden,g_layers,dropout)
        self.protein_encoder=ProteinEncoder(vocab_size=seq_vocab_size,emb_dim=seq_emb_dim,n_heads=seq_heads,n_layers=seq_layers,proj_dim=seq_proj_dim,dropout=dropout)
        fusion_in=g_hidden+seq_proj_dim
        self.head=MLP(fusion_in,mlp_hidden,1,dropout)

    def forward(self,graph,input_ids,attention_mask):
        g_emb=self.graph_encoder(graph)
        p_emb=self.protein_encoder(input_ids,attention_mask)
        fused = torch.cat([g_emb,p_emb],-1)
        return self.head(fused).squeeze(-1), fused

# ==========================================
# 3. Utils & Metrics
# ==========================================

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def concordance_index(y_true, y_pred):
    n = 0; n_concordant = 0; n_tied = 0
    for i in range(len(y_true)):
        for j in range(i+1, len(y_true)):
            if y_true[i] == y_true[j]: continue
            n += 1
            diff_true = y_true[i] - y_true[j]
            diff_pred = y_pred[i] - y_pred[j]
            if diff_true * diff_pred > 0: n_concordant += 1
            elif diff_pred == 0: n_tied += 1
    return (n_concordant + 0.5 * n_tied) / n if n > 0 else 0

def calc_rm2(y_true, y_pred):
    y_true, y_pred = np.array(y_true).reshape(-1,1), np.array(y_pred).reshape(-1,1)
    try:
        lr = LinearRegression().fit(y_pred, y_true)
        r2 = lr.score(y_pred, y_true)
        lr0 = LinearRegression(fit_intercept=False).fit(y_pred, y_true)
        r2_0 = lr0.score(y_pred, y_true)
        return r2 * (1 - np.sqrt(abs(r2 - r2_0)))
    except:
        return 0.0

def train_epoch(model, loader, optimizer, criterion, device):
    model.train(); losses=[]
    for graphs, input_ids, attn_mask, labels in loader:
        graphs, input_ids, attn_mask, labels = graphs.to(device), input_ids.to(device), attn_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        out, emb = model(graphs, input_ids, attn_mask)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval(); losses=[]; preds=[]; trues=[]
    for graphs, input_ids, attn_mask, labels in loader:
        graphs, input_ids, attn_mask, labels = graphs.to(device), input_ids.to(device), attn_mask.to(device), labels.to(device)
        out, emb = model(graphs, input_ids, attn_mask)
        loss = criterion(out, labels)
        losses.append(loss.item())
        preds.extend(out.cpu().numpy())
        trues.extend(labels.cpu().numpy())
        
    preds, trues = np.array(preds), np.array(trues)
    
    # 1. 基础指标
    mae = mean_absolute_error(trues, preds)
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    
    # 2. DTA 指标
    ci = concordance_index(trues, preds)
    rm2 = calc_rm2(trues, preds)
    
    # 3. [新增] 相对误差指标 (MAPE & Acc)
    relative_errors = np.abs((trues - preds) / np.maximum(np.abs(trues),1))
    mape = np.mean(relative_errors) * 100 
    hits = (relative_errors <= 0.10)
    acc_10 = np.mean(hits) * 100 
    
    return np.mean(losses), mae, rmse, ci, rm2, mape, acc_10

# ==========================================
# 4. Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Baseline: Standard Transfer Learning")
    parser.add_argument("--csv", type=str, default="./drug-target/Davis.csv")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    # [优化] 默认 batch_size 调大一点，因为我们用了动态填充
    parser.add_argument("--batch_size", type=int, default=32) 
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    # [优化] 默认模型变小一点，防止 OOM
    parser.add_argument("--g_hidden", type=int, default=128)
    parser.add_argument("--g_layers", type=int, default=3)
    parser.add_argument("--seq_emb_dim", type=int, default=128)
    parser.add_argument("--seq_layers", type=int, default=2)
    parser.add_argument("--seq_proj_dim", type=int, default=256)
    parser.add_argument("--mlp_hidden", type=str, default="256,128")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="DTA_checkpoints_baseline")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--pretrained", type=str, default="/data/wsh/bindingdb/checkpoints/Davis80_best_model.pt")
    
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    mlp_hidden = tuple(int(x) for x in args.mlp_hidden.split(","))

    # 1. Dataset
    print(f"Loading dataset: {args.csv}")
    dataset = MolProtDataset(args.csv, max_len=args.max_seq_len)
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 2. Model
    print(f"Initializing model on {args.device}...")
    model = MolProtRegressor(
        graph_in_dim=dataset.graph_in_dim,
        g_hidden=args.g_hidden, g_layers=args.g_layers,
        seq_vocab_size=30, seq_emb_dim=args.seq_emb_dim,
        seq_heads=4, seq_layers=args.seq_layers, seq_proj_dim=args.seq_proj_dim,
        mlp_hidden=mlp_hidden, dropout=args.dropout
    ).to(args.device)

    # 4. Optimizer
    # Baseline 策略：冻结 backbone，只微调 head，或者全量微调
    # 这里采用全量微调 (Full Fine-tuning) 以获得最佳 Baseline 性能
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_path = os.path.join(args.save_dir, "baseline_best.pt")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, mae, rmse, ci, rm2, mape, acc = eval_epoch(model, val_loader, criterion, args.device)
        
        print(f"Epoch {epoch}: Loss {tr_loss:.4f} | Val MAE {mae:.4f} CI {ci:.4f} | MAPE {mape:.2f}% Acc {acc:.1f}%")

        if mae < best_val_mae:
            best_val_mae = mae
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    # 5. Final Test
    print("\nLoading best model for final evaluation...")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=args.device))
        test_loss, test_mae, test_rmse, test_ci, test_rm2, test_mape, test_acc = eval_epoch(model, test_loader, criterion, args.device)
        
        print("-" * 60)
        print(f"[BASELINE RESULT] Test Set:")
        print(f"   MAE:  {test_mae:.4f}")
        print(f"   RMSE: {test_rmse:.4f}")
        print(f"   CI:   {test_ci:.4f}")
        print(f"   RM2:  {test_rm2:.4f}")
        print(f"   MAPE: {test_mape:.2f} %")
        print(f"   Acc:  {test_acc:.2f} %")
        print("-" * 60)
    else:
        print("Training failed or no model saved.")

if __name__=="__main__":
    main()