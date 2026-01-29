import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler, Subset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans
import os
import shutil
import time
import math

# --- 屏蔽警告 ---
from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore")

# --- [关键] 保留原有 Import ---
from dataset import SolvationDataset
from model import GNNRegressor

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class SilentTqdm:
    def __init__(self, iterable, *args, **kwargs): self.iterable = iterable
    def __iter__(self): return iter(self.iterable)
    def set_postfix(self, *args, **kwargs): pass
    def update(self, *args, **kwargs): pass
tqdm = SilentTqdm

# ==========================================
# PART 1: 核心数学组件 (ICL Loss)
# ==========================================
class IntervalCensoredLoss(nn.Module):
    def __init__(self, tol_percent=0.05, abs_tol=0.5, lambda_reg=0.01):
        super().__init__()
        self.tol_percent = tol_percent
        self.abs_tol = abs_tol
        self.lambda_reg = lambda_reg
        # Gauss-Legendre Nodes (20 points)
        _nodes = [-0.9931286, -0.9639719, -0.9122344, -0.8391170, -0.7463319, -0.6360537, -0.5108670, -0.3737061, -0.2277858, -0.0765265, 0.0765265, 0.2277858, 0.3737061, 0.5108670, 0.6360537, 0.7463319, 0.8391170, 0.9122344, 0.9639719, 0.9931286]
        _weights = [0.0176140, 0.0406014, 0.0626720, 0.0832767, 0.1019301, 0.1181945, 0.1316886, 0.1420961, 0.1491730, 0.1527534, 0.1527534, 0.1491730, 0.1420961, 0.1316886, 0.1181945, 0.1019301, 0.0832767, 0.0626720, 0.0406014, 0.0176140]
        self.register_buffer('nodes', torch.tensor(_nodes).view(1, -1))
        self.register_buffer('weights', torch.tensor(_weights).view(1, -1))

    def forward(self, outputs, target_norm, target_phys, y_std, epoch=None, total_epochs=None):
        gamma, nu, alpha, beta = outputs[:,0].view(-1,1), outputs[:,1].view(-1,1), outputs[:,2].view(-1,1), outputs[:,3].view(-1,1)
        y_norm, y_phys = target_norm.view(-1,1), target_phys.view(-1,1)
        
        # Hybrid Tolerance
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
        
        anneal = min(1.0, epoch / max(1, total_epochs // 5)) if (epoch and total_epochs) else 1.0
        return (nll + self.lambda_reg * anneal * reg_loss).squeeze()

# ==========================================
# PART 2: DETS 权重计算逻辑
# ==========================================
def soft_tanimoto_kernel(z_a, z_b):
    z_a = z_a.unsqueeze(1); z_b = z_b.unsqueeze(0)
    dot = (z_a * z_b).sum(2)
    norm_a = (z_a**2).sum(2); norm_b = (z_b**2).sum(2)
    return dot / (norm_a + norm_b - dot + 1e-8)

def find_prototype_match(z_batch, prototypes, y_consensuses):
    # Prototypes [K, D], z_batch [N, D]
    sim = soft_tanimoto_kernel(z_batch, prototypes)
    max_sim, idx = torch.max(sim, dim=1)
    y_cons = y_consensuses[idx]
    return torch.clamp(max_sim, 0.0, 1.0), y_cons, idx

def calculate_weights(E_stat, Delta_L, args):
    sigma_eff = args.sigma_0 * (1.0 + args.gamma * Delta_L)
    w_raw = args.alpha * torch.exp(- ((1.0 - E_stat)**2) / (2 * sigma_eff**2 + 1e-8))
    weights = torch.clamp(w_raw, min=args.w_min, max=1.0)
    # Hard anchor mask logic if needed
    mask = E_stat >= args.tau_high
    weights[mask] = 1.0
    return weights.double()

def pre_calculate_prototypes_on_target(model, target_dataset, device, num_prototypes, num_workers):
    print(f"-> Extracting Anchors...")
    model.eval()
    model.to(device)
    loader = DataLoader(target_dataset, batch_size=1024, shuffle=False, follow_batch=['x_solvent', 'x_solute'], num_workers=num_workers)
    
    all_z, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, z, _ = model(batch)
            all_z.append(z.cpu())
            all_y.append(batch.y.view(-1).cpu())
            
    z_mat = torch.cat(all_z).numpy()
    y_arr = torch.cat(all_y).numpy()
    
    kmeans = KMeans(n_clusters=num_prototypes, random_state=42, n_init=10).fit(z_mat)
    
    # 找最近的真实点作为 Prototype
    proto_idx = []
    for k in range(num_prototypes):
        mask = (kmeans.labels_ == k)
        pts = z_mat[mask]
        if len(pts) > 0:
            dists = cdist(pts, kmeans.cluster_centers_[k].reshape(1,-1))
            proto_idx.append(np.where(mask)[0][np.argmin(dists)])
            
    proto_z = torch.tensor(z_mat[proto_idx], dtype=torch.float).to(device)
    proto_y = []
    for k in range(num_prototypes):
        mask = (kmeans.labels_ == k)
        proto_y.append(np.mean(y_arr[mask]) if np.any(mask) else 0.0)
    proto_y = torch.tensor(proto_y, dtype=torch.float).to(device)
    
    print(f"-> Established {len(proto_z)} Anchors.")
    return proto_z, proto_y

def calculate_weights_for_source(model, source_dataset, prototypes, y_cons, y_mean, y_std, args, num_workers):
    model.eval()
    device = args.device
    loader = DataLoader(source_dataset, batch_size=4096, shuffle=False, follow_batch=['x_solvent', 'x_solute'], num_workers=num_workers)
    
    all_weights = []
    t_mean, t_std = y_mean.item(), y_std.item()
    min_accept, max_accept = t_mean - 3*t_std, t_mean + 3*t_std
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Calc Weights"):
            batch = batch.to(device)
            pred, z, _ = model(batch)
            nu, alpha, beta = pred[:,1], pred[:,2], pred[:,3]
            
            unc = beta/(nu*(alpha-1)+1e-8)
            E_stat, _, _ = find_prototype_match(z, prototypes, y_cons)
            
            # Batch-wise Norm
            d_l = (unc - unc.min())/(unc.max()-unc.min()+1e-8)
            w = calculate_weights(E_stat, d_l, args)
            
            # Range Filter
            y_b = batch.y.view(-1)
            mask = (y_b >= min_accept) & (y_b <= max_accept)
            # Size Filter (Solute atoms <= 100)
            size_mask = (batch.global_feat[:, 2] * 30.0 <= 100.0)
            
            w_final = w * mask.float() * size_mask.float()
            all_weights.append(w_final.cpu())
            
    raw = torch.cat(all_weights)
    
    # Top-K Guarantee
    k = int(len(raw) * args.keep_ratio) # e.g. 0.3 or 0.5
    if k > 0:
        thresh = torch.topk(raw, k)[0][-1]
        raw = torch.where(raw >= thresh, raw, torch.tensor(0.0)) # Or w_min
        if raw.max() > 0: raw /= raw.max()
        
    return raw

def train_epoch_weighted(model, loader, optimizer, device, y_mean, y_std, prototypes, y_cons, args, epoch, loss_fn):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Train Ep {epoch}")
    
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        out, z, _ = model(batch)
        
        y_phys = batch.y.view(-1).to(device)
        y_norm = (y_phys - y_mean) / y_std
        
        # Recalc weights for gradient
        with torch.no_grad():
            nu, alpha, beta = out[:,1], out[:,2], out[:,3]
            unc = beta/(nu*(alpha-1)+1e-8)
            d_l = (unc - unc.min())/(unc.max()-unc.min()+1e-8)
            e_s, _, _ = find_prototype_match(z, prototypes, y_cons)
            w = calculate_weights(e_s, d_l, args)
            
        loss_vec = loss_fn(out, y_norm, y_phys, y_std, epoch, args.epochs)
        loss = (loss_vec * w).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, device, y_mean, y_std, return_preds=False, abs_tol=0.5):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, _, _ = model(batch)
            # 反归一化
            pred = out[:,0] * y_std + y_mean
            preds.extend(pred.cpu().numpy())
            ys.extend(batch.y.view(-1).cpu().numpy())
            
    y_t = np.array(ys); y_p = np.array(preds)
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    
    abs_diff = np.abs(y_t - y_p)
    safe_y = np.maximum(np.abs(y_t), 1.0) 
    mape = np.mean(abs_diff / safe_y) * 100 
    
    hits = abs_diff <= np.maximum(0.10 * np.abs(y_t), abs_tol)
    acc = np.mean(hits) * 100
    
    if return_preds: return mae, rmse, mape, acc, y_p, y_t
    return mae, rmse, mape, acc

# ==========================================
# PART 3: Main Pipeline
# ==========================================
def main(args):
    print(f"--- DETS Solvation Pipeline (Minimal Integration) ---")
    set_seed(42)
    device = torch.device(args.device)
    num_workers = 0 

    # 1. Data
    target_full = SolvationDataset(root='./solvation_cache', csv_file=args.target_csv)
    source_dataset = SolvationDataset(root='./solvation_cache', csv_file=args.source_csv)
    
    idx_target = list(range(len(target_full)))
    tr_idx, test_idx = train_test_split(idx_target, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(tr_idx, test_size=(0.1/0.9), random_state=42)
    
    target_train = Subset(target_full, train_idx)
    target_val = Subset(target_full, val_idx)
    target_test = Subset(target_full, test_idx)
    
    y_raw = torch.tensor([target_full.targets[i] for i in train_idx])
    y_mean = y_raw.mean().to(device)
    y_std = y_raw.std().to(device)
    print(f"Target Stats: Mean={y_mean:.4f}, Std={y_std:.4f}")

    # 2. Model (Imported)
    sample = target_full[0]
    model = GNNRegressor(
        num_node_features=sample.x_solvent.shape[1],
        num_edge_features=sample.edge_attr_solvent.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(device)
    
    # Init DETS specific weights
    for layer in model.proj:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=3.0)
    torch.nn.init.xavier_uniform_(model.edl_head.weight)
    
    # Losses
    loss_fn_p1 = IntervalCensoredLoss(tol_percent=0.0, abs_tol=args.abs_tol, lambda_reg=args.edl_lambda).to(device)
    loss_fn_p2 = IntervalCensoredLoss(tol_percent=args.tol_percent, abs_tol=args.abs_tol, lambda_reg=args.edl_lambda).to(device)

    # --- Phase 0: Warmup ---
    print("\n>>> Phase 0: Warm-up")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(target_train, batch_size=32, shuffle=True, follow_batch=['x_solvent', 'x_solute'])
    for epoch in range(20):
        model.train()
        for b in loader:
            b = b.to(device)
            optimizer.zero_grad()
            out, _, _ = model(b)
            y_norm = (b.y.view(-1).to(device) - y_mean) / y_std
            loss = loss_fn_p2(out, y_norm, b.y.view(-1).to(device), y_std, epoch, 20).mean()
            loss.backward()
            optimizer.step()
            
    prototypes, y_cons = pre_calculate_prototypes_on_target(model, target_train, device, args.num_prototypes, num_workers)

    # --- Phase 1: DETS Source ---
    print("\n=== PHASE 1: DETS Source Training ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50)
    
    best_p1_rmse = float('inf')
    target_val_loader = DataLoader(target_val, batch_size=256, follow_batch=['x_solvent', 'x_solute'])
    
    for epoch in range(1, args.epochs + 1):
        # 每 5 轮更新一次权重
        if epoch == 1 or epoch % 1 == 0:
            print(f"-> [Ep {epoch}] Updating DETS Weights...")
            weights = calculate_weights_for_source(model, source_dataset, prototypes, y_cons, y_mean, y_std, args, num_workers)
            print(f"Weights Stats: Min={weights.min():.4f}, Max={weights.max():.4f}, Mean={weights.mean():.4f}")
        n_active = int(len(source_dataset) * 0.1) # 10% Subsample
        sampler = WeightedRandomSampler(weights, num_samples=n_active, replacement=False)
        src_loader = DataLoader(source_dataset, batch_size=args.batch_size, sampler=sampler, follow_batch=['x_solvent', 'x_solute'])
        
        loss = train_epoch_weighted(model, src_loader, optimizer, device, y_mean, y_std, prototypes, y_cons, args, epoch, loss_fn_p1)
        scheduler.step()
        
        if epoch % 1 == 0:
            mae, rmse, mape, acc = evaluate(model, target_val_loader, device, y_mean, y_std, abs_tol=args.abs_tol)
            print(f"[P1] Ep {epoch} | Loss {loss:.4f} | Val RMSE {rmse:.4f} MAE {mae:.4f} MAPE {mape:.4f} Acc {acc:.1f}%")
            if rmse < best_p1_rmse:
                best_p1_rmse = rmse
                torch.save(model.state_dict(), "DETS_solv_p1.pt")

    # --- Phase 2: Target Fine-tuning ---
    print("\n=== PHASE 2: Target Fine-tuning ===")
    model.load_state_dict(torch.load("DETS_solv_p1.pt"))
    
    # 2.1 Head
    for param in model.parameters(): param.requires_grad = False
    for param in model.edl_head.parameters(): param.requires_grad = True
    for param in model.fc_final.parameters(): param.requires_grad = True
    
    opt_head = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    ft_loader = DataLoader(target_train, batch_size=64, shuffle=True, follow_batch=['x_solvent', 'x_solute'])
    
    best_head_rmse = float('inf')
    for epoch in range(1, 101):
        model.train()
        for b in ft_loader:
            b = b.to(device)
            opt_head.zero_grad()
            out, _, _ = model(b)
            y_norm = (b.y.view(-1).to(device) - y_mean) / y_std
            loss = loss_fn_p2(out, y_norm, b.y.view(-1).to(device), y_std, epoch, 100).mean()
            loss.backward()
            opt_head.step()
        
        if epoch % 10 == 0:
            mae, rmse, mape, acc = evaluate(model, target_val_loader, device, y_mean, y_std, abs_tol=args.abs_tol)
            print(f"[Head] Ep {epoch} Val RMSE {rmse:.4f} MAE {mae:.4f} MAPE {mape:.4f} Acc {acc:.1f}")
            if rmse < best_head_rmse:
                best_head_rmse = rmse
                torch.save(model.state_dict(), "DETS_solv_head.pt")
                
    # 2.2 Full
    model.load_state_dict(torch.load("DETS_solv_head.pt"))
    for param in model.parameters(): param.requires_grad = True
    opt_full = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
    sched_full = CosineAnnealingWarmRestarts(opt_full, T_0=20)
    
    best_ft_rmse = float('inf')
    patience = 0
    
    for epoch in range(1, args.ft_epochs + 1):
        model.train()
        for b in ft_loader:
            b = b.to(device)
            opt_full.zero_grad()
            out, _, _ = model(b)
            y_norm = (b.y.view(-1).to(device) - y_mean) / y_std
            loss = loss_fn_p2(out, y_norm, b.y.view(-1).to(device), y_std, epoch, args.ft_epochs).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt_full.step()
        sched_full.step()
        
        if epoch % 10 == 0:
            mae, rmse, _, acc = evaluate(model, target_val_loader, device, y_mean, y_std, abs_tol=args.abs_tol)
            print(f"[Full] Ep {epoch}  Val RMSE {rmse:.4f} MAE {mae:.4f} MAPE {mape:.4f} Acc {acc:.1f}%")
            if rmse < best_ft_rmse:
                best_ft_rmse = rmse
                torch.save(model.state_dict(), "DETS_solv_final.pt")
                patience = 0
            else:
                patience += 1
            if patience >= 50: break

    # Final Test
    model.load_state_dict(torch.load("DETS_solv_final.pt"))
    test_loader = DataLoader(target_test, batch_size=128, follow_batch=['x_solvent', 'x_solute'])
    mae, rmse, mape, acc = evaluate(model, test_loader, device, y_mean, y_std, abs_tol=args.abs_tol)
    
    print("=" * 60)
    print(f"[FINAL RESULT] Solvation:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Acc:  {acc:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_csv", type=str, default="./solute-solvent/exp_data.csv")
    parser.add_argument("--source_csv", type=str, default="./solute-solvent/qm.csv")
    parser.add_argument("--pretrained", type=str, default="")
    
    parser.add_argument("--batch_size", type=int, default=256) # Phase 1
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ft_epochs", type=int, default=200)
    
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    parser.add_argument("--num_prototypes", type=int, default=20)
    parser.add_argument("--tau_high", type=float, default=0.90)
    parser.add_argument("--sigma_0", type=float, default=0.2) 
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--w_min", type=float, default=0.01)
    parser.add_argument("--keep_ratio", type=float, default=0.3)
    parser.add_argument("--edl_lambda", type=float, default=0.01)
    parser.add_argument("--tol_percent", type=float, default=0.05)
    parser.add_argument("--abs_tol", type=float, default=0.5)
    
    args = parser.parse_args()
    
    os.makedirs("DETS_experiments", exist_ok=True)
    
    main(args)