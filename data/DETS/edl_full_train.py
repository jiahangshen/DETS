# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans
import os
import shutil
import math
from dataset import ATCTDataset 

RDLogger.DisableLog('rdApp.*')

if os.path.exists('enthalpy/processed'):
    shutil.rmtree('enthalpy/processed', ignore_errors=True)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ==========================================
# PART 0: 模型定义 (完全复刻 EDL 版本)
# ==========================================
class GNNRegressor(nn.Module):
    def __init__(self, num_node_features, num_edge_features,
                 hidden_dim=256, num_layers=4, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # [复刻] 原始 GINE 结构，无 LayerNorm
        for i in range(num_layers):
            in_dim = num_node_features if i == 0 else hidden_dim
            nn_lin = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINEConv(nn_lin, edge_dim=num_edge_features)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim)) # 恢复 BatchNorm

        self.dropout = dropout
        
        # [新增] DETS 投影头 (这是唯一新增的结构，用于计算相似度)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64) 
        )
        
        # [复刻] 原始 EDL 的预测头逻辑
        # 动态初始化，将在 forward 中根据 global_feat 维度决定输入
        self.fc_final = None 
        self.edl_head = None
        self.hidden_dim = hidden_dim

    def _init_heads(self, global_feat_dim):
        # 拼接结构特征 + 物理特征
        input_dim = self.hidden_dim + global_feat_dim
        
        self.fc_final = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim), 
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU()
        )
        self.edl_head = nn.Linear(self.hidden_dim // 2, 4)
        
        # 初始化权重
        torch.nn.init.xavier_uniform_(self.edl_head.weight)
        self.to(next(self.parameters()).device)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 第一次运行时初始化 Head
        if self.fc_final is None:
            global_feat_dim = data.global_feat.shape[1]
            self._init_heads(global_feat_dim)

        # GNN Backbone
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global Pooling
        g = global_add_pool(x, batch) 
        
        # DETS Projection (用于采样，不参与最终预测)
        z = self.proj(g)
        z = F.normalize(z, p=2, dim=1)
        
        # [复刻] Feature Injection
        phys_feat = data.global_feat.view(g.shape[0], -1) 
        g_concat = torch.cat([g, phys_feat], dim=1) 
        
        # Prediction
        g_final = self.fc_final(g_concat)
        outputs = self.edl_head(g_final) 
        
        gamma = outputs[:, 0]
        nu    = F.softplus(outputs[:, 1]) + 1e-6
        alpha = F.softplus(outputs[:, 2]) + 1.0 + 1e-6
        beta  = F.softplus(outputs[:, 3]) + 1e-6
        
        edl_outputs = torch.stack([gamma, nu, alpha, beta], dim=1)
        
        return edl_outputs, z, g

# ==========================================
# PART 1: Loss Functions
# ==========================================

# 1. 标准 EDL Loss (用于 Phase 2，保持原汁原味)
def edl_loss_per_sample(outputs, targets, epoch, total_epochs, lambda_coef=0.01):
    gamma = outputs[:, 0]
    nu    = outputs[:, 1]
    alpha = outputs[:, 2]
    beta  = outputs[:, 3]
    y     = targets
    
    two_beta_lambda = 2 * beta * (1 + nu)
    nll = 0.5 * torch.log(3.14159 / nu) \
        - alpha * torch.log(two_beta_lambda) \
        + (alpha + 0.5) * torch.log(nu * (y - gamma)**2 + two_beta_lambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

    error = torch.abs(y - gamma)
    reg = error * (2 * nu + alpha)
    
    annealing = min(1.0, epoch / 10.0) 
    return nll + lambda_coef * annealing * reg

# 2. ICL Loss (仅用于 Phase 1 抗噪，保留高精度积分版)
class IntervalCensoredLoss(nn.Module):
    def __init__(self, tol_percent=0.05, abs_tol=1.0, lambda_reg=0.01):
        super().__init__()
        self.tol_percent = tol_percent
        self.abs_tol = abs_tol
        self.lambda_reg = lambda_reg
        # Gauss-Legendre Nodes
        _nodes = [-0.9931286, -0.9639719, -0.9122344, -0.8391170, -0.7463319, -0.6360537, -0.5108670, -0.3737061, -0.2277858, -0.0765265, 0.0765265, 0.2277858, 0.3737061, 0.5108670, 0.6360537, 0.7463319, 0.8391170, 0.9122344, 0.9639719, 0.9931286]
        _weights = [0.0176140, 0.0406014, 0.0626720, 0.0832767, 0.1019301, 0.1181945, 0.1316886, 0.1420961, 0.1491730, 0.1527534, 0.1527534, 0.1491730, 0.1420961, 0.1316886, 0.1181945, 0.1019301, 0.0832767, 0.0626720, 0.0406014, 0.0176140]
        self.register_buffer('nodes', torch.tensor(_nodes).view(1, -1))
        self.register_buffer('weights', torch.tensor(_weights).view(1, -1))

    def forward(self, outputs, target_norm, target_phys, y_std, epoch, total_epochs):
        gamma, nu, alpha, beta = outputs[:,0].view(-1,1), outputs[:,1].view(-1,1), outputs[:,2].view(-1,1), outputs[:,3].view(-1,1)
        y_norm, y_phys = target_norm.view(-1,1), target_phys.view(-1,1)
        
        rel_tol = torch.abs(y_phys) * self.tol_percent
        delta_phys = torch.maximum(rel_tol, torch.tensor(self.abs_tol, device=y_phys.device))
        delta_norm = delta_phys / (y_std + 1e-8)

        df = 2 * alpha
        scale = torch.sqrt((beta * (1 + nu)) / (nu * alpha + 1e-8))
        
        lower = y_norm - delta_norm
        upper = y_norm + delta_norm
        center = (upper + lower) / 2.0
        half_width = (upper - lower) / 2.0
        
        if self.nodes.device != center.device:
            self.nodes = self.nodes.to(center.device); self.weights = self.weights.to(center.device)
            
        x_points = center + half_width * self.nodes 
        df_exp = df.expand_as(x_points); scale_exp = scale.expand_as(x_points); gamma_exp = gamma.expand_as(x_points)
        z = (x_points - gamma_exp) / (scale_exp + 1e-8)
        
        log_prob = (torch.lgamma((df_exp + 1) / 2) - torch.lgamma(df_exp / 2) - 0.5 * torch.log(math.pi * df_exp) - torch.log(scale_exp) - (df_exp + 1) / 2 * torch.log(1 + z**2 / df_exp))
        pdf_vals = torch.exp(log_prob)
        prob_mass = half_width * torch.sum(pdf_vals * self.weights, dim=1, keepdim=True)
        prob_mass = torch.clamp(prob_mass, min=1e-6, max=1.0-1e-6)
        nll = -torch.log(prob_mass)
        
        raw_error = torch.abs(y_norm - gamma)
        effective_error = F.softplus(raw_error - delta_norm, beta=5.0) 
        reg_loss = effective_error * (2 * nu + alpha)
        
        anneal = min(1.0, epoch / max(1, total_epochs // 5)) 
        return (nll + self.lambda_reg * anneal * reg_loss).squeeze()

# ==========================================
# PART 2: DETS Helpers (保持不变)
# ==========================================
def soft_tanimoto_kernel(z_a, z_b):
    sim = torch.mm(z_a, z_b.t())
    return torch.clamp(sim, 0.0, 1.0) 

def find_prototype_match(z_batch, prototypes):
    sim_matrix = torch.mm(z_batch, prototypes.t())
    max_sim, _ = torch.max(sim_matrix, dim=1)
    return torch.clamp(max_sim, 0.0, 1.0)

def calculate_weights(E_stat, Delta_L, args):
    sigma_eff = args.sigma_0 * (1.0 + args.gamma * Delta_L) 
    D_struct = 1.0 - E_stat
    w_raw = args.alpha * torch.exp(- (D_struct**2) / (2 * sigma_eff**2 + 1e-8))
    weights = torch.clamp(w_raw, min=args.w_min, max=1.0)
    return weights.double()

def get_n_atoms(batch):
    if hasattr(batch, 'batch') and batch.batch is not None:
        return torch.bincount(batch.batch).view(-1, 1).float()
    return torch.tensor([batch.x.shape[0]], device=batch.x.device).view(-1, 1).float()

def pre_calculate_prototypes_on_target(model, target_dataset, device, num_prototypes):
    print(f"-> Extracting Anchors...")
    model.eval()
    model.to(device)
    loader = DataLoader(target_dataset, batch_size=256, shuffle=False)
    # 初始化 Head
    with torch.no_grad(): model(next(iter(loader)).to(device))
    
    all_z = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, z, _ = model(batch)
            all_z.append(z.cpu())
    z_matrix = torch.cat(all_z).numpy()
    kmeans = KMeans(n_clusters=num_prototypes, random_state=42, n_init=10).fit(z_matrix)
    prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    return prototypes

def calculate_weights_for_source(model, source_dataset, prototypes, args):
    model.eval()
    device = args.device
    loader = DataLoader(source_dataset, batch_size=1024, shuffle=False)
    
    all_unc = []
    all_E = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs, z, _ = model(batch)
            nu, alpha, beta = outputs[:, 1], outputs[:, 2], outputs[:, 3]
            unc = beta / (nu * (alpha - 1) + 1e-8)
            E_stat = find_prototype_match(z, prototypes)
            all_unc.append(unc)
            all_E.append(E_stat)
            
    cat_unc = torch.cat(all_unc)
    cat_E = torch.cat(all_E)
    u_min, u_max = cat_unc.min(), cat_unc.max()
    delta_L = (cat_unc - u_min) / (u_max - u_min + 1e-8)
    
    weights = calculate_weights(cat_E, delta_L, args).cpu()
    
    k = int(len(weights) * 0.3)
    top_val, _ = torch.topk(weights, k)
    threshold = top_val[-1]
    final_weights = torch.where(weights >= threshold, weights, torch.tensor(args.w_min))
    print(f"   [Weight Stats] Min: {final_weights.min():.4f}, Mean: {final_weights.mean():.4f}")
    return final_weights

# ==========================================
# PART 3: 评估函数
# ==========================================
def evaluate(model, loader, device, y_mean_atom, y_std_atom, return_preds=False, abs_tol_total=2.0):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs, _, _ = model(batch) 
            gamma_per_atom = outputs[:, 0].view(-1, 1)
            # 反归一化
            pred_atom = gamma_per_atom * y_std_atom + y_mean_atom
            n_atoms = get_n_atoms(batch).to(device)
            pred_total = pred_atom * n_atoms
            preds.extend(pred_total.cpu().numpy().flatten())
            ys.extend(batch.y.view(-1).cpu().numpy())
    
    y_true = np.array(ys)
    y_pred = np.array(preds)
    abs_diff = np.abs(y_true - y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    safe_y = np.maximum(np.abs(y_true), 1e-6)
    mape = np.mean(abs_diff / safe_y) * 100 
    hits = (abs_diff <= np.maximum(0.10 * np.abs(y_true), abs_tol_total))
    acc_10 = np.mean(hits) * 100 
    
    if return_preds: return mae, rmse, mape, acc_10, y_pred, y_true
    return mae, rmse, mape, acc_10

# ==========================================
# PART 4: 主程序
# ==========================================
def main(args):
    print(f"--- Hybrid GNN-DETS Pipeline (Standard EDL Fine-tuning) ---")
    set_seed(42)
    device = torch.device(args.device)

    target_full = ATCTDataset(args.target_csv)
    idx_target = range(len(target_full))
    tr_idx, test_idx = train_test_split(idx_target, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(tr_idx, test_size=0.1, random_state=42)
    
    target_train = target_full.index_select(train_idx)
    target_val = target_full.index_select(val_idx)
    target_test = target_full.index_select(test_idx)
    source_dataset = ATCTDataset(args.source_csv)
    
    # Per-Atom Stats
    loader_stat = DataLoader(target_train, batch_size=256, shuffle=False)
    y_atom_list = []
    for b in loader_stat:
        n = get_n_atoms(b)
        y_atom_list.append(b.y.view(-1, 1) / torch.clamp(n, min=1.0))
    y_atom_tensor = torch.cat(y_atom_list)
    y_mean_atom = y_atom_tensor.mean().to(device)
    y_std_atom = y_atom_tensor.std().to(device)
    print(f"Target Atom Stats: Mean={y_mean_atom.item():.4f}, Std={y_std_atom.item():.4f}")

    # Model
    model = GNNRegressor(target_full[0].x.shape[1], 
                         target_full[0].edge_attr.shape[1] if target_full[0].edge_attr is not None else 0,
                         hidden_dim=args.hidden_dim, num_layers=args.layers, dropout=args.dropout).to(device)
    
    # Losses
    # Phase 1: ICL Loss (抗噪)
    loss_fn_p1 = IntervalCensoredLoss(tol_percent=0.0, abs_tol=args.abs_tol/15.0, lambda_reg=args.edl_lambda).to(device)
    
    # --- Phase 0: Warmup (Target, ICL) ---
    print("\n>>> Phase 0: Warm-up")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(target_train, batch_size=32, shuffle=True)
    for epoch in range(20):
        model.train()
        for b in loader:
            b = b.to(device)
            optimizer.zero_grad()
            out, _, _ = model(b)
            n = get_n_atoms(b).to(device)
            y_atom = b.y.view(-1,1)/n
            y_norm = (y_atom - y_mean_atom)/y_std_atom
            loss = loss_fn_p1(out, y_norm, y_atom, y_std_atom, epoch, 20).mean()
            loss.backward()
            optimizer.step()
            
    prototypes = pre_calculate_prototypes_on_target(model, target_train, device, args.num_prototypes)

    # --- Phase 1: DETS Source (ICL Loss + DETS Sampling) ---
    print("\n=== PHASE 1: DETS Source Training ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_p1_mae = float('inf')
    target_val_loader = DataLoader(target_val, batch_size=128)
    
    for epoch in range(1, args.epochs + 1):
        weights = calculate_weights_for_source(model, source_dataset, prototypes, args)
        n_active = int(len(source_dataset) * 0.5)
        sampler = WeightedRandomSampler(weights, num_samples=n_active, replacement=False)
        src_loader = DataLoader(source_dataset, batch_size=args.batch_size, sampler=sampler)
        
        model.train()
        epoch_loss = 0
        for b in src_loader:
            b = b.to(device)
            optimizer.zero_grad()
            out, z, _ = model(b)
            n = get_n_atoms(b).to(device)
            y_atom = b.y.view(-1,1)/n
            y_norm = (y_atom - y_mean_atom)/y_std_atom
            
            # DETS Weighting Calculation for Loss
            with torch.no_grad():
                nu, alpha, beta = out[:,1], out[:,2], out[:,3]
                unc = beta/(nu*(alpha-1)+1e-8)
                d_l = (unc - unc.min())/(unc.max()-unc.min()+1e-8)
                e_s = find_prototype_match(z, prototypes)
                w_b = calculate_weights(e_s, d_l, args)
                
            loss_vec = loss_fn_p1(out, y_norm, y_atom, y_std_atom, epoch, args.epochs)
            loss = (loss_vec * w_b).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            val_mae, _, _, _ = evaluate(model, target_val_loader, device, y_mean_atom, y_std_atom, abs_tol_total=args.abs_tol)
            print(f"[P1] Ep {epoch} | Loss {epoch_loss/len(src_loader):.4f} | Val MAE {val_mae:.4f}")
            if val_mae < best_p1_mae:
                best_p1_mae = val_mae
                torch.save(model.state_dict(), "DETS/best_p1.pt")

    # --- Phase 2: Target Fine-tuning (完全复刻 EDL Phase 2) ---
    print("\n=== PHASE 2: Target Fine-tuning ===")
    model.load_state_dict(torch.load("DETS/best_p1.pt"))
    
    # 2.1 Head Alignment (冻结其他，只训 Head)
    for param in model.parameters(): param.requires_grad = False
    for param in model.edl_head.parameters(): param.requires_grad = True
    for param in model.fc_final.parameters(): param.requires_grad = True
    
    # [复刻] 使用 args.lr
    opt_head = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    ft_loader = DataLoader(target_train, batch_size=16, shuffle=True)
    
    for epoch in range(1, 101):
        model.train()
        for b in ft_loader:
            b = b.to(device)
            opt_head.zero_grad()
            out, _, _ = model(b)
            n = get_n_atoms(b).to(device)
            y_atom = b.y.view(-1,1)/n
            y_norm = (y_atom - y_mean_atom)/y_std_atom
            
            # [复刻] 使用 Standard EDL Loss
            loss = edl_loss_per_sample(out, y_norm, epoch, 100, lambda_coef=args.edl_lambda).mean()
            loss.backward()
            opt_head.step()
            
        if epoch % 20 == 0:
            val_mae, _, _, _ = evaluate(model, target_val_loader, device, y_mean_atom, y_std_atom, abs_tol_total=args.abs_tol)
            print(f"[Head] Ep {epoch} MAE {val_mae:.4f}")
    
    torch.save(model.state_dict(), "DETS/best_head.pt")
    
    # 2.2 Full Fine-tuning (解冻所有，LR * 0.05)
    for param in model.parameters(): param.requires_grad = True
    
    # [复刻] 使用 0.05 倍 LR
    opt_full = torch.optim.Adam(model.parameters(), lr=args.lr * 0.05) 
    
    best_ft_mae = float('inf')
    patience = 0
    for epoch in range(1, args.ft_epochs + 1):
        model.train()
        for b in ft_loader:
            b = b.to(device)
            opt_full.zero_grad()
            out, _, _ = model(b)
            n = get_n_atoms(b).to(device)
            y_atom = b.y.view(-1,1)/n
            y_norm = (y_atom - y_mean_atom)/y_std_atom
            
            # [复刻] 使用 Standard EDL Loss
            loss = edl_loss_per_sample(out, y_norm, epoch, args.ft_epochs, lambda_coef=args.edl_lambda).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_full.step()
            
        val_mae, _, _, _ = evaluate(model, target_val_loader, device, y_mean_atom, y_std_atom, abs_tol_total=args.abs_tol)
        if epoch % 10 == 0: print(f"[Full] Ep {epoch} MAE {val_mae:.4f}")
        
        if val_mae < best_ft_mae:
            best_ft_mae = val_mae
            torch.save(model.state_dict(), "DETS/final_model.pt")
            patience = 0
        else:
            patience += 1
            if patience >= 100: break

    # Final Test
    model.load_state_dict(torch.load("DETS/final_model.pt"))
    test_mae, test_rmse, test_mape, test_acc = evaluate(model, DataLoader(target_test, batch_size=128), device, y_mean_atom, y_std_atom, abs_tol_total=args.abs_tol)
    print(f"Final Test: MAE {test_mae:.4f} | Acc {test_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_csv", type=str, default="./enthalpy/atct.csv")
    parser.add_argument("--source_csv", type=str, default="./enthalpy/wudily_cho.csv")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--is_all_train", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--ft_epochs", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    parser.add_argument("--num_prototypes", type=int, default=10)
    parser.add_argument("--tau_high", type=float, default=0.90)
    parser.add_argument("--sigma_0", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--w_min", type=float, default=0.01)
    parser.add_argument("--edl_lambda", type=float, default=0.01)
    parser.add_argument("--tol_percent", type=float, default=0.05)
    parser.add_argument("--abs_tol", type=float, default=2.0)
    
    args = parser.parse_args()
    main(args)