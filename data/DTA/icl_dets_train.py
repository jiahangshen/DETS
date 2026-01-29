import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split, Subset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool
from ogb.utils import smiles2graph
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import shutil
import math

# ==========================================
# PART 0: 基础设置
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- Dataset (需根据实际环境调整) ---
AA_VOCAB = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","X","-","?"]
AA_TO_ID = {aa:i for i,aa in enumerate(AA_VOCAB)}

class MolProtDataset(Dataset):
    def __init__(self, csv_path, max_len=1024):
        self.df = pd.read_csv(csv_path)
        # 兼容不同的列名
        if "COMPOUND_SMILES" in self.df.columns:
            self.smiles = self.df["COMPOUND_SMILES"].tolist()
            self.seqs = self.df["PROTEIN_SEQUENCE"].tolist()
            self.labels = self.df["REG_LABEL"].astype(float).tolist()
        else:
            self.smiles = self.df["smiles"].tolist()
            self.seqs = self.df["sequence"].tolist()
            self.labels = self.df["label"].astype(float).tolist()
            
        self.max_len = max_len
        # Dummy graph for dim check
        g = smiles2graph("CC")
        self.graph_in_dim = g["node_feat"].shape[1]

    def encode_seq(self, seq):
        ids = [AA_TO_ID.get(ch, AA_TO_ID["?"]) for ch in seq]
        if len(ids) > self.max_len: ids = ids[:self.max_len]
        attn_mask = [1]*len(ids)
        while len(ids)<self.max_len: ids.append(AA_TO_ID["-"]); attn_mask.append(0)
        return torch.tensor(ids), torch.tensor(attn_mask)

    def __len__(self): return len(self.smiles)

    def __getitem__(self, idx):
        gdict = smiles2graph(self.smiles[idx])
        graph = Data(x=torch.tensor(gdict["node_feat"],dtype=torch.float),
                     edge_index=torch.tensor(gdict["edge_index"],dtype=torch.long),
                     edge_attr=torch.tensor(gdict["edge_feat"],dtype=torch.float) if gdict["edge_feat"] is not None else None)
        input_ids, attn_mask = self.encode_seq(self.seqs[idx])
        label = torch.tensor(self.labels[idx],dtype=torch.float)
        mol_index = torch.tensor([idx], dtype=torch.long)
        return graph, input_ids, attn_mask, label, mol_index

def collate_fn(batch):
    graphs, ids, masks, labels, indices = zip(*batch)
    batched_graph = Batch.from_data_list(graphs)
    return batched_graph, torch.stack(ids), torch.stack(masks), torch.stack(labels), torch.stack(indices)

# --- Models ---

class GraphEncoder(nn.Module):
    def __init__(self,in_dim,hidden_dim=128,num_layers=3,dropout=0.1):
        super().__init__()
        self.layers=nn.ModuleList()
        for i in range(num_layers):
            mlp=nn.Sequential(nn.Linear(in_dim if i==0 else hidden_dim,hidden_dim),
                              nn.ReLU(),nn.Dropout(dropout),
                              nn.Linear(hidden_dim,hidden_dim))
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

    def forward(self,input_ids,attention_mask):
        bsz,seq_len=input_ids.size()
        pos=torch.arange(seq_len,device=input_ids.device).unsqueeze(0).expand(bsz,seq_len)
        x=self.embedding(input_ids)+self.pos_embedding(pos)
        x=self.encoder(x,src_key_padding_mask=(attention_mask==0))
        
        # Mean Pooling with Mask
        mask_float = attention_mask.float().unsqueeze(-1)
        # Fix dimension mismatch if transformer truncates
        if x.shape[1] < mask_float.shape[1]:
            mask_float = mask_float[:, :x.shape[1], :]
            
        sum_embeddings = torch.sum(x * mask_float, dim=1)
        sum_mask = torch.clamp(mask_float.sum(dim=1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        return self.proj(mean_embeddings)

class MolProtRegressor(nn.Module):
    def __init__(self,graph_in_dim,g_hidden=128,g_layers=3,seq_vocab_size=30,seq_emb_dim=128,seq_heads=4,seq_layers=2,seq_proj_dim=256,mlp_hidden=(256,128),dropout=0.1):
        super().__init__()
        self.graph_encoder=GraphEncoder(graph_in_dim,g_hidden,g_layers,dropout)
        self.protein_encoder=ProteinEncoder(vocab_size=seq_vocab_size,emb_dim=seq_emb_dim,n_heads=seq_heads,n_layers=seq_layers,proj_dim=seq_proj_dim,dropout=dropout)
        
        fusion_in = g_hidden + seq_proj_dim
        
        # DETS Projector
        self.proj = nn.Sequential(
            nn.Linear(fusion_in, fusion_in // 2),
            nn.ReLU(),
            nn.Linear(fusion_in // 2, 64), # Dim 64 for clustering
            nn.Tanh() # Use Tanh for bounded embeddings (-1, 1)
        )
        
        # EDL Head
        layers=[]
        last=fusion_in
        for h in mlp_hidden:
            layers+=[nn.Linear(last,h),nn.ReLU(),nn.Dropout(dropout)]
            last=h
        self.mlp_body = nn.Sequential(*layers)
        self.edl_head = nn.Linear(last, 4)

    def forward(self, graph, input_ids, attention_mask):
        g_emb = self.graph_encoder(graph)
        p_emb = self.protein_encoder(input_ids, attention_mask)
        fused = torch.cat([g_emb, p_emb], -1)
        
        z = self.proj(fused)
        
        feat = self.mlp_body(fused)
        outputs = self.edl_head(feat)
        
        gamma = outputs[:, 0]
        nu    = F.softplus(outputs[:, 1]) + 1e-6
        alpha = F.softplus(outputs[:, 2]) + 1.0 + 1e-6
        beta  = F.softplus(outputs[:, 3]) + 1e-6
        
        # Return stacked [Batch, 4]
        edl_outputs = torch.stack([gamma, nu, alpha, beta], dim=1)
        return edl_outputs, z, fused

# ==========================================
# PART 1: ICL Loss (Gauss-Legendre)
# ==========================================
class IntervalCensoredLoss(nn.Module):
    def __init__(self, tol_percent=0.05, abs_tol=0.1, lambda_reg=0.01):
        super().__init__()
        self.tol_percent = tol_percent
        self.abs_tol = abs_tol
        self.lambda_reg = lambda_reg
        
        _nodes = [-0.9931286, -0.9639719, -0.9122344, -0.8391170, -0.7463319,
                  -0.6360537, -0.5108670, -0.3737061, -0.2277858, -0.0765265,
                   0.0765265,  0.2277858,  0.3737061,  0.5108670,  0.6360537,
                   0.7463319,  0.8391170,  0.9122344,  0.9639719,  0.9931286]
        _weights = [0.0176140, 0.0406014, 0.0626720, 0.0832767, 0.1019301,
                    0.1181945, 0.1316886, 0.1420961, 0.1491730, 0.1527534,
                    0.1527534, 0.1491730, 0.1420961, 0.1316886, 0.1181945,
                    0.1019301, 0.0832767, 0.0626720, 0.0406014, 0.0176140]
        
        self.register_buffer('nodes', torch.tensor(_nodes).view(1, -1))
        self.register_buffer('weights', torch.tensor(_weights).view(1, -1))

    def forward(self, outputs, targets_norm, targets_physical, y_std, epoch=None, total_epochs=None):
        gamma = outputs[:, 0].view(-1, 1)
        nu    = outputs[:, 1].view(-1, 1)
        alpha = outputs[:, 2].view(-1, 1)
        beta  = outputs[:, 3].view(-1, 1)
        
        y_norm = targets_norm.view(-1, 1)
        y_phys = targets_physical.view(-1, 1)
        
        # Hybrid Tolerance: max(relative, absolute)
        # 对于 DTA 任务 (pKd ~ 5-10)，abs_tol=0.1 是合理的
        rel_tol = torch.abs(y_phys) * self.tol_percent
        abs_tol_t = torch.tensor(self.abs_tol, device=y_phys.device)
        delta_phys = torch.maximum(rel_tol, abs_tol_t)
        
        delta_norm = delta_phys / (y_std + 1e-8)

        df = 2 * alpha
        scale = torch.sqrt((beta * (1 + nu)) / (nu * alpha + 1e-8))
        
        # Integration Bounds
        lower = y_norm - delta_norm
        upper = y_norm + delta_norm
        center = (upper + lower) / 2.0
        half_width = (upper - lower) / 2.0
        
        # Device Check
        if self.nodes.device != center.device:
            self.nodes = self.nodes.to(center.device)
            self.weights = self.weights.to(center.device)

        x_points = center + half_width * self.nodes 
        
        df_exp = df.expand_as(x_points)
        scale_exp = scale.expand_as(x_points)
        gamma_exp = gamma.expand_as(x_points)
        
        z = (x_points - gamma_exp) / (scale_exp + 1e-8)
        
        # Log PDF Calculation
        log_prob = (torch.lgamma((df_exp + 1) / 2) 
                    - torch.lgamma(df_exp / 2) 
                    - 0.5 * torch.log(math.pi * df_exp) 
                    - torch.log(scale_exp) 
                    - (df_exp + 1) / 2 * torch.log(1 + z**2 / df_exp))
        
        pdf_vals = torch.exp(log_prob)
        weighted_sum = torch.sum(pdf_vals * self.weights, dim=1, keepdim=True)
        prob_mass = half_width * weighted_sum
        
        prob_mass = torch.clamp(prob_mass, min=1e-6, max=1.0 - 1e-6)
        nll = -torch.log(prob_mass)
        
        # Regularization
        raw_error = torch.abs(y_norm - gamma)
        effective_error = F.softplus(raw_error - delta_norm, beta=5.0) 
        reg_loss = effective_error * (2 * nu + alpha)
        
        anneal = 1.0
        if epoch is not None and total_epochs is not None:
            anneal = min(1.0, epoch / max(1, total_epochs // 5))
            
        total_loss = nll + self.lambda_reg * anneal * reg_loss
        return total_loss

# ==========================================
# PART 2: DETS Weighting Logic
# ==========================================

def soft_tanimoto_sim(z1, z2):
    # z1: [N, D], z2: [K, D]
    z1 = z1.unsqueeze(1)
    z2 = z2.unsqueeze(0)
    dot = (z1 * z2).sum(dim=2)
    norm1 = (z1**2).sum(dim=2)
    norm2 = (z2**2).sum(dim=2)
    return dot / (norm1 + norm2 - dot + 1e-8)

def find_prototype_match(z_batch, prototypes, y_consensuses):
    # Use Soft Tanimoto for continuous embeddings
    sim_matrix = soft_tanimoto_sim(z_batch, prototypes)
    max_sim, nearest_proto_idx = torch.max(sim_matrix, dim=1)
    # Clamp for safety
    max_sim = torch.clamp(max_sim, 0.0, 1.0)
    nearest_y_consensus = y_consensuses[nearest_proto_idx]
    return max_sim, nearest_y_consensus, nearest_proto_idx

def calculate_weights(E_stat, Delta_L, args):
    # Thermo-Dynamic Weighting
    sigma_eff = args.sigma_0 * (1.0 + args.gamma * Delta_L)
    energy = (1.0 - E_stat)**2
    temperature = 2 * sigma_eff**2 + 1e-8
    
    w_raw = args.alpha * torch.exp( - energy / temperature )
    
    # Soft Limit instead of Hard Cutoff
    weights = torch.clamp(w_raw, min=args.w_min, max=1.0)
    return weights.double()

def pre_calculate_prototypes_on_target(model, target_loader, device, num_prototypes):
    print(f"-> Extracting Anchors from Target...")
    model.eval()
    all_z = []; all_y = []
    with torch.no_grad():
        for batch in target_loader:
            graphs, ids, masks, labels, _ = batch
            graphs, ids, masks = graphs.to(device), ids.to(device), masks.to(device)
            _, z, _ = model(graphs, ids, masks)
            all_z.append(z.cpu())
            all_y.append(labels)
            
    z_matrix = torch.cat(all_z).numpy()
    y_array = torch.cat(all_y).numpy()
    
    kmeans = KMeans(n_clusters=num_prototypes, random_state=42, n_init=10).fit(z_matrix)
    prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)
    
    y_cons = []
    for k in range(num_prototypes):
        mask = (kmeans.labels_ == k)
        if np.any(mask): y_cons.append(np.mean(y_array[mask]))
        else: y_cons.append(0.0)
    y_cons = torch.tensor(y_cons, dtype=torch.float).to(device)
    
    print(f"-> Anchors Established: {len(prototypes)} prototypes.")
    return prototypes, y_cons

def calculate_weights_for_source(model, source_loader, prototypes, y_cons, device, args):
    model.eval()
    temp_unc = []
    temp_E_stat = []
    
    with torch.no_grad():
        for batch in source_loader:
            graphs, ids, masks, labels, _ = batch
            graphs, ids, masks = graphs.to(device), ids.to(device), masks.to(device)
            outputs, z, _ = model(graphs, ids, masks)
            
            nu, alpha, beta = outputs[:, 1], outputs[:, 2], outputs[:, 3]
            epistemic_unc = beta / (nu * (alpha - 1) + 1e-8)
            E_stat, _, _ = find_prototype_match(z, prototypes, y_cons)
            
            temp_unc.append(epistemic_unc)
            temp_E_stat.append(E_stat)
            
    cat_unc = torch.cat(temp_unc)       
    cat_E_stat = torch.cat(temp_E_stat)
    
    u_min, u_max = cat_unc.min(), cat_unc.max()
    delta_L = (cat_unc - u_min) / (u_max - u_min + 1e-8)
    
    weights = calculate_weights(cat_E_stat, delta_L, args).cpu()
    
    # Soft Filter: Keep high weights, reduce low weights to w_min
    final_weights = weights
    
    print(f"   [Weights] Min: {final_weights.min():.4f}, Max: {final_weights.max():.4f}, Mean: {final_weights.mean():.4f}")
    return final_weights

# ==========================================
# PART 3: Metrics & Training
# ==========================================

def concordance_index(y_true, y_pred):
    # 简化的 C-Index 计算
    try:
        from lifelines.utils import concordance_index as ci
        return ci(y_true, y_pred)
    except ImportError:
        # Fallback 手写
        n = len(y_true)
        if n < 2: return 0
        order = np.argsort(y_true)
        y_true = y_true[order]
        y_pred = y_pred[order]
        c = 0
        for i in range(n):
            for j in range(i+1, n):
                if y_true[i] != y_true[j]:
                    if y_pred[i] < y_pred[j]: c += 1
                    elif y_pred[i] == y_pred[j]: c += 0.5
        return c / (n*(n-1)/2)

def calc_rm2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true) < 2: return 0.0
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    return r2

def evaluate(model, loader, device, y_mean, y_std, abs_tol=0.1):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            graphs, ids, masks, labels, _ = batch
            graphs, ids, masks = graphs.to(device), ids.to(device), masks.to(device)
            outputs, _, _ = model(graphs, ids, masks)
            gamma = outputs[:, 0]
            
            # Anti-Normalize
            pred = gamma * y_std + y_mean
            
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.numpy())
            
    preds, trues = np.array(preds), np.array(trues)
    
    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    ci = concordance_index(trues, preds)
    rm2 = calc_rm2(trues, preds)
    
    abs_diff = np.abs(trues - preds)
    # Acc@10 logic
    tol_threshold = np.maximum(0.10 * np.abs(trues), abs_tol)
    acc_10 = np.mean(abs_diff <= tol_threshold) * 100
    
    # MAPE logic
    mape = np.mean(abs_diff / np.maximum(np.abs(trues), 1)) * 100
    
    return mae, rmse, ci, rm2, mape, acc_10

class EarlyStopping:
    def __init__(self, patience=50, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_mae, model):
        score = -val_mae
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), self.path)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), self.path)
            self.counter = 0

def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--source_csv", type=str, default="../drug-target/Davis.csv")
    parser.add_argument("--target_csv", type=str, default="../drug-target/CASF2016_graph_valid.csv")
    parser.add_argument("--pretrained", type=str, default="")
    
    # Hyperparams
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000, help="Phase 1 Epochs")
    parser.add_argument("--ft_epochs", type=int, default=500, help="Phase 2 Epochs")
    parser.add_argument("--lr", type=float, default=1e-3) # Base LR
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="DETS_DTA")
    
    # DETS
    parser.add_argument("--num_prototypes", type=int, default=3)
    parser.add_argument("--tau_high", type=float, default=0.9)
    parser.add_argument("--sigma_0", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.5) # Alpha Scale
    parser.add_argument("--w_min", type=float, default=0.1)
    
    # ICL
    parser.add_argument("--edl_lambda", type=float, default=0.01)
    parser.add_argument("--tol_percent", type=float, default=0.05)
    parser.add_argument("--abs_tol", type=float, default=0.1)
    
    parser.add_argument("--mlp_hidden", type=str, default="256,128")
    
    args = parser.parse_args()
    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    mlp_hidden = tuple(int(x) for x in args.mlp_hidden.split(","))

    # 1. Load Data
    print("Loading datasets...")
    target_ds = MolProtDataset(args.target_csv, max_len=args.max_seq_len)
    source_ds = MolProtDataset(args.source_csv, max_len=args.max_seq_len)
    
    n_total = len(target_ds)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    target_train, target_val, target_test = random_split(target_ds, [n_train, n_val, n_total-n_train-n_val])
    
    target_train_loader = DataLoader(target_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
    target_val_loader = DataLoader(target_val, batch_size=32, shuffle=False, collate_fn=collate_fn)
    target_test_loader = DataLoader(target_test, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Stats
    all_train_labels = [target_ds.labels[i] for i in target_train.indices]
    y_tensor = torch.tensor(all_train_labels, dtype=torch.float)
    y_mean = y_tensor.mean().to(args.device)
    y_std = y_tensor.std().to(args.device)
    print(f"Target Stats: Mean={y_mean.item():.4f}, Std={y_std.item():.4f}")

    # Model
    model = MolProtRegressor(
        graph_in_dim=target_ds.graph_in_dim,
        mlp_hidden=mlp_hidden, dropout=args.dropout
    ).to(args.device)
    
    # Init Weights
    torch.nn.init.xavier_uniform_(model.edl_head.weight)
    
    # Loss
    loss_fn_source = IntervalCensoredLoss(tol_percent=0.0, abs_tol=args.abs_tol, lambda_reg=args.edl_lambda).to(args.device)
    loss_fn_target = IntervalCensoredLoss(tol_percent=args.tol_percent, abs_tol=args.abs_tol, lambda_reg=args.edl_lambda).to(args.device)

    # ------------------------------------
    # PHASE 0: Warm-up
    # ------------------------------------
    print("\n>>> Phase 0: Warm-up on Target...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(20):
        model.train()
        for batch in target_train_loader:
            graphs, ids, masks, labels, _ = batch
            graphs, ids, masks, labels = graphs.to(args.device), ids.to(args.device), masks.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs, _, _ = model(graphs, ids, masks)
            y_norm = (labels - y_mean) / y_std
            loss = loss_fn_target(outputs, y_norm, labels, y_std, epoch, 20)
            loss.mean().backward()
            optimizer.step()
            
    proto_loader = DataLoader(target_train, batch_size=32, shuffle=False, collate_fn=collate_fn)
    prototypes, y_cons = pre_calculate_prototypes_on_target(model, proto_loader, args.device, args.num_prototypes)

    # ------------------------------------
    # PHASE 1: DETS Pre-training
    # ------------------------------------
    print("\n=== PHASE 1: DETS Source Training ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    stopper = EarlyStopping(patience=100, path=os.path.join(args.save_dir, "best_p1.pt"))
    
    for epoch in range(1, args.epochs+1):
        temp_src_loader = DataLoader(source_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
        weights = calculate_weights_for_source(model, temp_src_loader, prototypes, y_cons, args.device, args)
        
        n_active = int(len(source_ds) * 0.5)
        sampler = WeightedRandomSampler(weights, num_samples=n_active, replacement=False)
        source_loader = DataLoader(source_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
        
        model.train()
        epoch_loss = 0
        for batch in source_loader:
            graphs, ids, masks, labels, _ = batch
            graphs, ids, masks, labels = graphs.to(args.device), ids.to(args.device), masks.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs, _, _ = model(graphs, ids, masks)
            y_norm = (labels - y_mean) / y_std
            loss = loss_fn_source(outputs, y_norm, labels, y_std, epoch, args.epochs).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        
        mae, rmse, ci, rm2, mape, acc = evaluate(model, target_val_loader, args.device, y_mean, y_std)
        if epoch % 10 == 0:
            print(f"[P1] Ep {epoch} | Loss {epoch_loss/len(source_loader):.4f} | Val MAE {mae:.4f} Acc {acc:.1f}%")
            
        stopper(mae, model)
        if stopper.early_stop: break

    # ------------------------------------
    # PHASE 2: Target Fine-tuning
    # ------------------------------------
    print("\n=== PHASE 2: Target Fine-tuning ===")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_p1.pt")))
    
    # Stage 2.1: Head
    print(">>> Stage 2.1: Head Alignment...")
    for param in model.parameters(): param.requires_grad = False
    for param in model.edl_head.parameters(): param.requires_grad = True
    
    opt_head = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr*0.05)
    stopper_head = EarlyStopping(patience=50, path=os.path.join(args.save_dir, "best_head.pt"))
    
    for epoch in range(1, 201):
        model.train()
        for batch in target_train_loader:
            graphs, ids, masks, labels, _ = batch
            graphs, ids, masks, labels = graphs.to(args.device), ids.to(args.device), masks.to(args.device), labels.to(args.device)
            opt_head.zero_grad()
            outputs, _, _ = model(graphs, ids, masks)
            y_norm = (labels - y_mean) / y_std
            loss = loss_fn_target(outputs, y_norm, labels, y_std, epoch, 200).mean()
            loss.backward()
            opt_head.step()
            
        mae, _, _, _, _, _ = evaluate(model, target_val_loader, args.device, y_mean, y_std)
        if epoch % 20 == 0: print(f"[Head] Ep {epoch} MAE {mae:.4f}")
        stopper_head(mae, model)
        if stopper_head.early_stop: break

    # Stage 2.2: Full
    print(">>> Stage 2.2: Full Fine-tuning...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_head.pt")))
    for param in model.parameters(): param.requires_grad = True
    
    # 激进策略: LR=5e-4, Restart
    opt_full = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=args.weight_decay)
    sched_full = CosineAnnealingWarmRestarts(opt_full, T_0=50, T_mult=2)
    stopper_full = EarlyStopping(patience=100, path=os.path.join(args.save_dir, "final_model.pt"))
    
    for epoch in range(1, args.ft_epochs+1):
        model.train()
        for batch in target_train_loader:
            graphs, ids, masks, labels, _ = batch
            graphs, ids, masks, labels = graphs.to(args.device), ids.to(args.device), masks.to(args.device), labels.to(args.device)
            opt_full.zero_grad()
            outputs, _, _ = model(graphs, ids, masks)
            y_norm = (labels - y_mean) / y_std
            loss = loss_fn_target(outputs, y_norm, labels, y_std, epoch, args.ft_epochs).mean()
            loss.backward()
            opt_full.step()
            
        sched_full.step()
        
        mae, rmse, ci, rm2, mape, acc = evaluate(model, target_val_loader, args.device, y_mean, y_std)
        if epoch % 10 == 0: print(f"[Full] Ep {epoch} MAE {mae:.4f} Acc {acc:.1f}%")
        
        stopper_full(mae, model)
        if stopper_full.early_stop: break

    # Final Test
    print("\n>>> Testing Final Model...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "final_model.pt")))
    mae, rmse, ci, rm2, mape, acc = evaluate(model, target_test_loader, args.device, y_mean, y_std)
    print("-" * 60)
    print(f"[FINAL RESULT] Test Set:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   CI:   {ci:.4f}")
    print(f"   RM2:  {rm2:.4f}")
    print(f"   MAPE: {mape:.2f} %")
    print(f"   Acc:  {acc:.2f} %")
    print("-" * 60)

if __name__=="__main__":
    main()