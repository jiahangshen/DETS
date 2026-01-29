import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool
from ogb.utils import smiles2graph
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans
import os
import shutil

# ==========================================
# PART 0: 基础设置与类定义
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- 1. Dataset ---
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

# --- 2. Models ---

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

class MolProtRegressor(nn.Module):
    def __init__(self,graph_in_dim,g_hidden=128,g_layers=3,seq_vocab_size=30,seq_emb_dim=128,seq_heads=4,seq_layers=2,seq_proj_dim=256,mlp_hidden=(256,128),dropout=0.1):
        super().__init__()
        self.graph_encoder=GraphEncoder(graph_in_dim,g_hidden,g_layers,dropout)
        self.protein_encoder=ProteinEncoder(vocab_size=seq_vocab_size,emb_dim=seq_emb_dim,n_heads=seq_heads,n_layers=seq_layers,proj_dim=seq_proj_dim,dropout=dropout)
        
        fusion_in = g_hidden + seq_proj_dim
        
        # --- DETS Projection Head ---
        self.proj = nn.Sequential(
            nn.Linear(fusion_in, fusion_in // 2),
            nn.ReLU(),
            nn.Linear(fusion_in // 2, 128), 
            nn.Sigmoid() 
        )
        
        # --- EDL Head ---
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
        
        edl_outputs = torch.stack([gamma, nu, alpha, beta], dim=1)
        return edl_outputs, z, fused

# ==========================================
# PART 1: DETS & EDL Logic
# ==========================================

def soft_tanimoto_kernel(z_a, z_b):
    z_a_expanded = z_a.unsqueeze(1)
    z_b_expanded = z_b.unsqueeze(0)
    dot_product = torch.sum(z_a_expanded * z_b_expanded, dim=2)
    norm_a_sq = torch.sum(z_a_expanded * z_a_expanded, dim=2)
    norm_b_sq = torch.sum(z_b_expanded * z_b_expanded, dim=2)
    return dot_product / (norm_a_sq + norm_b_sq - dot_product + 1e-8)

def find_prototype_match(z_batch, prototypes, y_consensuses):
    tanimoto_matrix = soft_tanimoto_kernel(z_batch, prototypes)
    max_tanimoto, nearest_proto_idx = torch.max(tanimoto_matrix, dim=1)
    nearest_y_consensus = y_consensuses[nearest_proto_idx]
    return max_tanimoto, nearest_y_consensus, nearest_proto_idx

def calculate_weights(E_stat, Delta_L, args):
    weights = torch.ones_like(E_stat)
    hard_anchor_mask = E_stat >= args.tau_high
    E_soft = E_stat[~hard_anchor_mask]
    
    sigma_eff = args.sigma_0 * (1.0 + args.gamma * Delta_L[~hard_anchor_mask])
    D_struct = 1.0 - E_soft
    w_soft = args.alpha * torch.exp(- (D_struct**2) / (2 * sigma_eff**2 + 1e-8))
    
    weights[~hard_anchor_mask] = torch.clamp(w_soft, min=args.w_min, max=args.alpha)
    weights[hard_anchor_mask] = 1.0
    return weights.double()

def edl_loss_per_sample(outputs, targets, epoch, total_epochs, lambda_coef=0.01):
    gamma, nu, alpha, beta = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
    y = targets
    
    two_beta_lambda = 2 * beta * (1 + nu)
    nll = 0.5 * torch.log(3.14159 / nu) \
        - alpha * torch.log(two_beta_lambda) \
        + (alpha + 0.5) * torch.log(nu * (y - gamma)**2 + two_beta_lambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

    error = torch.abs(y - gamma)
    reg = error * (2 * nu + alpha)
    
    annealing_coef = min(1.0, epoch / 10.0) 
    return nll + lambda_coef * annealing_coef * reg

# --- DETS Helpers ---

def pre_calculate_prototypes_on_target(model, target_loader, device, num_prototypes):
    print(f"-> Extracting Anchors from Target (CASF)...")
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
    
    print(f"   Running K-Means (K={num_prototypes})...")
    kmeans = KMeans(n_clusters=num_prototypes, random_state=42, n_init=10).fit(z_matrix)
    
    prototypes_indices = []
    for k in range(num_prototypes):
        cluster_mask = (kmeans.labels_ == k)
        cluster_points = z_matrix[cluster_mask]
        if len(cluster_points) > 0:
            dists = cdist(cluster_points, kmeans.cluster_centers_[k].reshape(1,-1))
            original_idx = np.where(cluster_mask)[0][np.argmin(dists)]
            prototypes_indices.append(original_idx)
            
    prototypes_z = torch.tensor(z_matrix[prototypes_indices], dtype=torch.float).to(device)
    
    y_consensuses = []
    for k in range(num_prototypes):
        mask = (kmeans.labels_ == k)
        if np.any(mask):
            y_consensuses.append(np.mean(y_array[mask]))
        else:
            y_consensuses.append(0.0)
    prototypes_y = torch.tensor(y_consensuses, dtype=torch.float).to(device)
    
    print(f"-> Anchors Established: {len(prototypes_z)} prototypes.")
    return prototypes_z, prototypes_y

# ==========================================
# [关键修改] 基于 Epistemic Uncertainty 的权重计算
# ==========================================
def calculate_weights_for_source(model, source_loader, prototypes, y_cons, device, args):
    """
    修改说明：
    1. 移除了基于数值偏差(Diff)的 delta_L 计算。
    2. 新增了基于 Epistemic Uncertainty 的 delta_L 计算。
    """
    model.eval()
    
    # 临时列表用于存储中间结果
    temp_unc = []
    temp_E_stat = []
    temp_labels = []
    
    # 硬截断范围 (DTA 任务常见范围)
    min_accept, max_accept = 2.0, 14.0 
    
    with torch.no_grad():
        for batch in source_loader:
            graphs, ids, masks, labels, _ = batch
            graphs, ids, masks = graphs.to(device), ids.to(device), masks.to(device)
            
            # 1. 获取模型输出
            outputs, z, _ = model(graphs, ids, masks)
            
            # 2. 解包 EDL 参数
            nu    = outputs[:, 1]
            alpha = outputs[:, 2]
            beta  = outputs[:, 3]
            
            # 3. [核心逻辑] 计算认知不确定性
            # Epistemic Uncertainty = beta / (nu * (alpha - 1))
            epistemic_unc = beta / (nu * (alpha - 1) + 1e-8)
            
            # 原型匹配
            E_stat, _, _ = find_prototype_match(z, prototypes, y_cons)
            
            # 收集
            temp_unc.append(epistemic_unc)
            temp_E_stat.append(E_stat)
            temp_labels.append(labels.to(device))
            
    # 拼接所有 Batch
    cat_unc = torch.cat(temp_unc)       # (N, )
    cat_E_stat = torch.cat(temp_E_stat)
    cat_labels = torch.cat(temp_labels)
    
    # 4. [核心逻辑] 归一化不确定性到 [0, 1] 得到 delta_L
    u_min = cat_unc.min()
    u_max = cat_unc.max()
    delta_L = (cat_unc - u_min) / (u_max - u_min + 1e-8)
    
    # 5. 计算权重 (传入 Uncertainty based delta_L)
    # 不确定性高 -> delta_L 高 -> sigma_eff 大 -> 筛选变宽
    w_global = calculate_weights(cat_E_stat, delta_L, args).cpu()
    
    # 6. 硬截断过滤 (安全网)
    range_mask = (cat_labels.cpu() >= min_accept) & (cat_labels.cpu() <= max_accept)
    raw_weights = w_global * range_mask.float()
    
    # 7. Top-K 保留 (防止引入过多噪音)
    keep_ratio = 0.30
    k = int(len(raw_weights) * keep_ratio)
    
    if k > 0:
        values, indices = torch.topk(raw_weights, k)
        new_weights = torch.zeros_like(raw_weights)
        new_weights[indices] = values
        if new_weights.max() > 0:
            new_weights /= new_weights.max()
        raw_weights = new_weights    

    # [Debug] 打印权重统计信息
    w_mean = raw_weights.mean().item()
    n_zeros = (raw_weights == 0).sum().item()
    
    print(f"   [Weight Stats] Mean: {w_mean:.4f} | Zero Count: {n_zeros}/{len(raw_weights)} ({(n_zeros/len(raw_weights))*100:.1f}%)")
    
    return raw_weights

# ==========================================
# 3. Metrics
# ==========================================

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

def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            graphs, ids, masks, labels, _ = batch
            graphs, ids, masks = graphs.to(device), ids.to(device), masks.to(device)
            outputs, _, _ = model(graphs, ids, masks)
            gamma = outputs[:, 0]
            preds.extend(gamma.cpu().numpy())
            trues.extend(labels.numpy())
            
    preds, trues = np.array(preds), np.array(trues)
    
    mae = mean_absolute_error(trues, preds)
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    ci = concordance_index(trues, preds)
    rm2 = calc_rm2(trues, preds)
    
    relative_errors = np.abs((trues - preds) / (np.abs(trues) + 1e-8))
    mape = np.mean(relative_errors) * 100 
    hits = (relative_errors <= 0.10)
    acc_10 = np.mean(hits) * 100
    
    return mae, rmse, ci, rm2, mape, acc_10

def train_phase1(model, loader, optimizer, device, prototypes, y_cons, args, epoch):
    model.train()
    losses = []
    for batch in loader:
        graphs, ids, masks, labels, _ = batch
        graphs, ids, masks, labels = graphs.to(device), ids.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # 1. 前向传播
        outputs, z, _ = model(graphs, ids, masks)
        
        # 2. 动态权重 (这里简化为 Batch 内计算或沿用全局权重)
        # 为了梯度连贯性，这里我们复用 calculate_weights 的逻辑，但在 Batch 内做
        with torch.no_grad():
            nu, alpha, beta = outputs[:, 1], outputs[:, 2], outputs[:, 3]
            batch_unc = beta / (nu * (alpha - 1) + 1e-8)
            
            # Batch 内归一化 (近似)
            b_min, b_max = batch_unc.min(), batch_unc.max()
            batch_delta_L = (batch_unc - b_min) / (b_max - b_min + 1e-8)
            
            E_stat, _, _ = find_prototype_match(z, prototypes, y_cons)
            
            weights = calculate_weights(E_stat, batch_delta_L, args)
            
        # 3. 计算 Loss
        loss_vec = edl_loss_per_sample(outputs, labels, epoch, args.epochs, args.edl_lambda)
        loss = torch.mean(loss_vec * weights)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        
    return np.mean(losses)

def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--source_csv", type=str, default="../drug-target/Davis.csv")
    parser.add_argument("--target_csv", type=str, default="../drug-target/CASF2016_graph_valid.csv")
    parser.add_argument("--pretrained", type=str, default="/Davis80_best_model.pt")
    
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200) # Phase 1
    parser.add_argument("--ft_epochs", type=int, default=150) # Phase 2
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="DETS_DTA")
    
    parser.add_argument("--num_prototypes", type=int, default=5)
    parser.add_argument("--tau_high", type=float, default=0.9)
    parser.add_argument("--sigma_0", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=10)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--w_min", type=float, default=1e-4)
    parser.add_argument("--edl_lambda", type=float, default=0.01)
    parser.add_argument("--mlp_hidden", type=str, default="256,128")
    
    args = parser.parse_args()
    print(args)
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
    
    # 2. Init Model
    model = MolProtRegressor(
        graph_in_dim=target_ds.graph_in_dim,
        mlp_hidden=mlp_hidden, dropout=args.dropout
    ).to(args.device)
    
    if os.path.exists(args.pretrained):
        print(f"Loading pretrained backbone from {args.pretrained}")
        state = torch.load(args.pretrained, map_location=args.device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state.items() if k in model_dict and 'head' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
    print("Initializing DETS Projection Head (High Variance)...")
    for layer in model.proj:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0) 
            if layer.bias is not None: 
                torch.nn.init.constant_(layer.bias, 0.0) 
    
    with torch.no_grad():
        model.graph_encoder.layers[-1].nn[0].weight.add_(torch.randn_like(model.graph_encoder.layers[-1].nn[0].weight) * 0.1)
        model.protein_encoder.proj[0].weight.add_(torch.randn_like(model.protein_encoder.proj[0].weight) * 0.1)
        
    torch.nn.init.xavier_uniform_(model.edl_head.weight)
    run_name = f"n{args.num_prototypes}_tau{args.tau_high}_sig{args.sigma_0}"
    args.save_dir = os.path.join("DETS_DTA", run_name)
    os.makedirs(args.save_dir, exist_ok=True)

    # ==========================================
    # PHASE 1: DETS Guided Pre-training
    # ==========================================
    print("\n=== PHASE 1: DETS Source Training ===")
    
    proto_loader = DataLoader(target_train, batch_size=32, shuffle=False, collate_fn=collate_fn)
    prototypes, y_cons = pre_calculate_prototypes_on_target(model, proto_loader, args.device, args.num_prototypes)
    
    optimizer = torch.optim.Adam([
        {"params": model.proj.parameters(), "lr": args.lr* 10},       
        {"params": model.mlp_body.parameters(), "lr": args.lr},
        {"params": model.edl_head.parameters(), "lr": args.lr},
        {"params": model.graph_encoder.parameters(), "lr": args.lr * 0.1},
        {"params": model.protein_encoder.parameters(), "lr": args.lr * 0.1},
    ], weight_decay=args.weight_decay)
    
    best_p1_loss = float('inf')
    patience = 0
    
    for epoch in range(1, args.epochs+1):
        temp_src_loader = DataLoader(source_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
        source_weights = calculate_weights_for_source(model, temp_src_loader, prototypes, y_cons, args.device, args)
        
        n_active = (source_weights > 0).sum().item()
        print(f"-> [Ep {epoch}] Active Samples: {n_active}/{len(source_ds)}")
        
        if n_active < 10: n_active = 100
        sampler = WeightedRandomSampler(source_weights, num_samples=n_active, replacement=True)
        source_loader = DataLoader(source_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
        
        loss = train_phase1(model, source_loader, optimizer, args.device, prototypes, y_cons, args, epoch)
        
        mae, rmse, ci, rm2, mape, acc = evaluate(model, target_val_loader, args.device)
        print(f"[P1] Ep {epoch} | Loss {loss:.4f} | Val MAE {mae:.4f} CI {ci:.4f} Acc {acc:.1f}%")
        
        if loss < best_p1_loss:
            best_p1_loss = loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_p1.pt"))
            patience = 0
        else:
            patience += 1
        if patience >= 50: break

    # ==========================================
    # PHASE 2: Target Fine-tuning
    # ==========================================
    print("\n=== PHASE 2: Target Fine-tuning ===")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_p1.pt")))
    
    # Stage 2.1
    print(">>> Stage 2.1: Head Alignment...")
    for param in model.parameters(): param.requires_grad = False
    for param in model.mlp_body.parameters(): param.requires_grad = True
    for param in model.edl_head.parameters(): param.requires_grad = True
    
    optimizer_head = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    
    best_ft_mae = float('inf')
    
    for epoch in range(1, 151):
        model.train()
        for batch in target_train_loader:
            graphs, ids, masks, labels, _ = batch
            graphs, ids, masks, labels = graphs.to(args.device), ids.to(args.device), masks.to(args.device), labels.to(args.device)
            optimizer_head.zero_grad()
            outputs, _, _ = model(graphs, ids, masks)
            loss_vec = edl_loss_per_sample(outputs, labels, epoch, 500)
            loss = loss_vec.mean()
            loss.backward()
            optimizer_head.step()
            
        mae, rmse, ci, rm2, mape, acc = evaluate(model, target_val_loader, args.device)
        if epoch % 10 == 0: print(f"[Head] Ep {epoch} MAE {mae:.4f} CI {ci:.4f} Acc {acc:.1f}%")
        if mae < best_ft_mae:
            best_ft_mae = mae
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_head.pt"))

    # Stage 2.2
    print(">>> Stage 2.2: Full Fine-tuning...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_head.pt")))
    for param in model.parameters(): param.requires_grad = True
    
    optimizer_all = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
    patience = 0
    best_ft_mae = float('inf')
    
    for epoch in range(1, args.ft_epochs + 1):
        model.train()
        for batch in target_train_loader:
            graphs, ids, masks, labels, _ = batch
            graphs, ids, masks, labels = graphs.to(args.device), ids.to(args.device), masks.to(args.device), labels.to(args.device)
            optimizer_all.zero_grad()
            outputs, _, _ = model(graphs, ids, masks)
            loss_vec = edl_loss_per_sample(outputs, labels, epoch, args.ft_epochs)
            loss = loss_vec.mean()
            loss.backward()
            optimizer_all.step()
            
        mae, rmse, ci, rm2, mape, acc = evaluate(model, target_val_loader, args.device)
        if epoch % 10 == 0: print(f"[Full] Ep {epoch} MAE {mae:.4f} CI {ci:.4f} RM2 {rm2:.4f} Acc {acc:.1f}%")
        
        if mae < best_ft_mae:
            best_ft_mae = mae
            torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pt"))
            patience = 0
        else:
            patience += 1
        if patience >= 50: break

    # Final Test
    print("\n>>> Testing Final Model...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "final_model.pt")))
    mae, rmse, ci, rm2, mape, acc = evaluate(model, target_test_loader, args.device)
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