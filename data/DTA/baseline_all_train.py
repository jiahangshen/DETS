import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool
from ogb.utils import smiles2graph
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from rdkit import RDLogger
import os
import shutil

# [新增] 引入适配器
from sampler import CoresetSamplerDTA

RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. Dataset & Collate (复用你的代码)
# ==========================================
AA_VOCAB = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","X","-","?"]
AA_TO_ID = {aa:i for i,aa in enumerate(AA_VOCAB)}

class MolProtDataset(Dataset):
    def __init__(self, csv_path, max_len=1024):
        self.df = pd.read_csv(csv_path)
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
        try:
            g = smiles2graph("CC")
            self.graph_in_dim = g["node_feat"].shape[1]
        except:
            self.graph_in_dim = 9

    def encode_seq(self, seq):
        ids = [AA_TO_ID.get(ch, AA_TO_ID["?"]) for ch in seq]
        if len(ids) > self.max_len: ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self): return len(self.smiles)

    def __getitem__(self, idx):
        gdict = smiles2graph(self.smiles[idx])
        graph = Data(x=torch.tensor(gdict["node_feat"],dtype=torch.float),
                     edge_index=torch.tensor(gdict["edge_index"],dtype=torch.long),
                     edge_attr=torch.tensor(gdict["edge_feat"],dtype=torch.float) if gdict["edge_feat"] is not None else None)
        input_ids = self.encode_seq(self.seqs[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        mol_index = torch.tensor([idx], dtype=torch.long) # Return index for sampler
        return graph, input_ids, label, mol_index

def collate_fn(batch):
    graphs, input_ids_list, labels, indices = zip(*batch)
    batched_graph = Batch.from_data_list(graphs)
    batch_size = len(input_ids_list)
    max_batch_len = max([len(x) for x in input_ids_list])
    padded_ids = torch.zeros((batch_size, max_batch_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_batch_len), dtype=torch.long)
    for i, seq in enumerate(input_ids_list):
        end = len(seq)
        padded_ids[i, :end] = seq
        attention_mask[i, :end] = 1
    labels = torch.stack(labels)
    indices = torch.stack(indices)
    return batched_graph, padded_ids, attention_mask, labels, indices

# ==========================================
# 2. Model (添加 forward_with_features)
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
        if x.shape[1] < mask.shape[1]: mask = mask[:, :x.shape[1]]
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
        # [修改] 默认 forward 兼容性
        out, _ = self.forward_with_features(graph,input_ids,attention_mask)
        return out, _

    # [新增] 接口
    def forward_with_features(self,graph,input_ids,attention_mask):
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
    lr = LinearRegression().fit(y_pred, y_true)
    r2 = lr.score(y_pred, y_true)
    lr0 = LinearRegression(fit_intercept=False).fit(y_pred, y_true)
    r2_0 = lr0.score(y_pred, y_true)
    return r2 * (1 - np.sqrt(abs(r2 - r2_0)))

def train_epoch(model, loader, optimizer, criterion, device):
    model.train(); losses=[]
    for graphs, input_ids, attn_mask, labels, _ in loader:
        graphs, input_ids, attn_mask, labels = graphs.to(device), input_ids.to(device), attn_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        out, _ = model.forward_with_features(graphs, input_ids, attn_mask)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, abs_tol=0.1):
    model.eval(); losses=[]; preds=[]; trues=[]
    for graphs, input_ids, attn_mask, labels, _ in loader:
        graphs, input_ids, attn_mask, labels = graphs.to(device), input_ids.to(device), attn_mask.to(device), labels.to(device)
        out, _ = model.forward_with_features(graphs, input_ids, attn_mask)
        loss = criterion(out, labels)
        losses.append(loss.item())
        preds.extend(out.cpu().numpy())
        trues.extend(labels.cpu().numpy())
        
    preds, trues = np.array(preds), np.array(trues)
    
    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    ci = concordance_index(trues, preds)
    rm2 = calc_rm2(trues, preds)
    
    # 混合容忍度
    abs_diff = np.abs(trues - preds)
    tol_threshold = np.maximum(0.10 * np.abs(trues), abs_tol)
    acc_10 = np.mean(abs_diff <= tol_threshold) * 100
    
    mape = np.mean(abs_diff / np.maximum(np.abs(trues), 1.0)) * 100 
    
    return np.mean(losses), mae, rmse, ci, rm2, mape, acc_10

# ==========================================
# 4. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="DTA Baseline with Sampling")
    # Data Paths
    parser.add_argument("--source_csv", type=str, default="./drug-target/Davis.csv")
    parser.add_argument("--target_csv", type=str, default="./drug-target/CASF2016_graph_valid.csv")
    
    # Sampling Params
    parser.add_argument("--sampling_method", type=str, default="Full", 
                        choices=["Full", "Hard Random", "Herding", "K-Means", 
                                 "Entropy", "Least Confidence", "GraNd-20", "Glister", "Influence", 
                                 "EL2N-2", "DP", "CSIE",
                                 "Soft Random", "InfoBatch", "MolPeg", "epsilon-greedy", "UCB"])
    parser.add_argument("--sampling_ratio", type=float, default=0.2)
    
    # Train Params
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32) 
    parser.add_argument("--epochs_p1", type=int, default=200)
    parser.add_argument("--epochs_p2", type=int, default=200) # Fine-tuning epochs
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    parser.add_argument("--g_hidden", type=int, default=128)
    parser.add_argument("--g_layers", type=int, default=3)
    parser.add_argument("--seq_emb_dim", type=int, default=128)
    parser.add_argument("--seq_layers", type=int, default=2)
    parser.add_argument("--seq_proj_dim", type=int, default=256)
    parser.add_argument("--mlp_hidden", type=str, default="256,128")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="DTA_checkpoints_baseline")
    parser.add_argument("--patience", type=int, default=50) # for Phase 2
    
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    mlp_hidden = tuple(int(x) for x in args.mlp_hidden.split(","))

    print(f"--- DTA Baseline: {args.sampling_method} (Ratio: {args.sampling_ratio}) ---")

    # 1. Dataset
    target_ds = MolProtDataset(args.target_csv, max_len=args.max_seq_len)
    source_ds = MolProtDataset(args.source_csv, max_len=args.max_seq_len)
    
    # Split Target (Fixed Seed Generator)
    n_total = len(target_ds)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    split_generator = torch.Generator().manual_seed(args.seed)
    target_train, target_val, target_test = random_split(target_ds, [n_train, n_val, n_total-n_train-n_val], generator=split_generator)

    target_train_loader = DataLoader(target_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    target_val_loader = DataLoader(target_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    target_test_loader = DataLoader(target_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 2. Init Model
    model = MolProtRegressor(
        graph_in_dim=target_ds.graph_in_dim,
        g_hidden=args.g_hidden, g_layers=args.g_layers,
        seq_vocab_size=30, seq_emb_dim=args.seq_emb_dim,
        seq_heads=4, seq_layers=args.seq_layers, seq_proj_dim=args.seq_proj_dim,
        mlp_hidden=mlp_hidden, dropout=args.dropout
    ).to(args.device)
    
    # Init Weights (Xavier)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    # ==========================================
    # Stage 0: Coreset Selection
    # ==========================================
    train_source_dataset = source_ds
    
    if args.sampling_method != 'Full':
        print(f"\n>>> [Stage 0] Sampling...")
        # 注意：这里需要传入 collate_fn
        sampler = CoresetSamplerDTA(model, source_ds, target_train, args.device, collate_fn=collate_fn)
        indices = sampler.select(args.sampling_method, args.sampling_ratio)
        train_source_dataset = Subset(source_ds, indices)
        print(f"-> Source reduced: {len(source_ds)} -> {len(train_source_dataset)}")

    # ==========================================
    # Stage 1: Source Pre-training
    # ==========================================
    print("\n=== Phase 1: Source Pre-training ===")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    best_p1_mae = float("inf")
    
    for epoch in range(1, args.epochs_p1 + 1):
        tr_loss = train_epoch(model, source_loader, optimizer, criterion, args.device)
        
        # Phase 1 没必要每次都测，每10轮测一次
        if epoch % 1 == 0:
            val_loss, mae, _, _, _, _, _ = eval_epoch(model, target_val_loader, criterion, args.device)
            print(f"[P1] Ep {epoch}: Loss {tr_loss:.4f} | Val MAE {mae:.4f}")
            if mae < best_p1_mae:
                best_p1_mae = mae
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_p1.pt"))

    # ==========================================
    # Stage 2: Target Fine-tuning (Frozen Body)
    # ==========================================
    print("\n=== Phase 2: Target Fine-tuning ===")
    if os.path.exists(os.path.join(args.save_dir, "best_p1.pt")):
        model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_p1.pt")))
        
    print("Freezing backbone layers...")
    for param in model.graph_encoder.parameters():
        param.requires_grad = False
    for param in model.protein_encoder.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    
    # 降低学习率微调 (或保持，视数据量而定)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    
    best_ft_mae = float("inf")
    patience_counter = 0
    final_path = os.path.join(args.save_dir, "final_model.pt")
    
    for epoch in range(1, args.epochs_p2 + 1):
        tr_loss = train_epoch(model, target_train_loader, optimizer, criterion, args.device)
        val_loss, mae, rmse, ci, rm2, mape, acc = eval_epoch(model, target_val_loader, criterion, args.device)
        
        print(f"[FT] Ep {epoch}: Loss {tr_loss:.4f} | Val MAE {mae:.4f} CI {ci:.4f}")
            
        if mae < best_ft_mae:
            best_ft_mae = mae
            torch.save(model.state_dict(), final_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience: 
                print("Early stopping triggered.")
                break

    # Final Test
    print("\n>>> Final Test Evaluation")
    if os.path.exists(final_path):
        model.load_state_dict(torch.load(final_path))
        test_loss, test_mae, test_rmse, test_ci, test_rm2, test_mape, test_acc = eval_epoch(model, target_test_loader, criterion, args.device)
        
        print("-" * 60)
        print(f"[BASELINE RESULT] Method: {args.sampling_method}")
        print(f"   MAE:  {test_mae:.4f}")
        print(f"   RMSE: {test_rmse:.4f}")
        print(f"   CI:   {test_ci:.4f}")
        print(f"   RM2:  {test_rm2:.4f}")
        print(f"   MAPE: {test_mape:.2f} %")
        print(f"   Acc:  {test_acc:.2f} %")
        print("-" * 60)
    else:
        print("Training failed.")

if __name__=="__main__":
    main()