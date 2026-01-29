import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans
import os
import shutil
import time
class SilentTqdm:
    def __init__(self, iterable, *args, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        return iter(self.iterable)
    def set_postfix(self, *args, **kwargs):
        pass
    def update(self, *args, **kwargs):
        pass

# 将 tqdm 指向这个哑巴类
tqdm = SilentTqdm
# --- 屏蔽警告 ---
from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore")
# ------------------

from dataset import SolvationDataset
from model import GNNRegressor

# ==========================================
# PART 1: 核心数学与辅助函数 (保持不变)
# ==========================================

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
    annealing_coef = min(1.0, epoch / 10.0) 
    return nll + lambda_coef * annealing_coef * reg

def soft_tanimoto_kernel(z_a, z_b):
    z_a_expanded = z_a.unsqueeze(1)
    z_b_expanded = z_b.unsqueeze(0)
    dot_product = torch.sum(z_a_expanded * z_b_expanded, dim=2)
    norm_a_sq = torch.sum(z_a_expanded * z_a_expanded, dim=2)
    norm_b_sq = torch.sum(z_b_expanded * z_b_expanded, dim=2)
    numerator = dot_product
    denominator = norm_a_sq + norm_b_sq - dot_product + 1e-8
    return numerator / denominator

def find_prototype_match(z_batch, prototypes, y_consensuses):
    tanimoto_matrix = soft_tanimoto_kernel(z_batch, prototypes)
    max_tanimoto, nearest_proto_idx = torch.max(tanimoto_matrix, dim=1)
    nearest_y_consensus = y_consensuses[nearest_proto_idx]
    return max_tanimoto, nearest_y_consensus, nearest_proto_idx

def calculate_weights(E_stat, Delta_L, args):
    weights = torch.ones_like(E_stat)
    hard_anchor_mask = E_stat >= args.tau_high
    E_soft = E_stat[~hard_anchor_mask]
    Delta_L_soft = Delta_L[~hard_anchor_mask]
    D_struct = 1.0 - E_soft 
    
    sigma_eff = args.sigma_0 * (1.0 + args.gamma * Delta_L_soft)
    w_soft = args.alpha * torch.exp(- (D_struct**2) / (2 * sigma_eff**2 + 1e-8))
    
    weights[~hard_anchor_mask] = torch.clamp(w_soft, min=args.w_min, max=args.alpha)
    weights[hard_anchor_mask] = 1.0
    return weights.double()

def pre_calculate_prototypes_on_target(model, target_dataset, device, num_prototypes, num_workers):
    print(f"-> Extracting Anchors (Solvation Free Energy)...")
    model.eval()
    model.to(device)
    loader = DataLoader(target_dataset, batch_size=1024, shuffle=False, 
                        follow_batch=['x_solvent', 'x_solute'],
                        num_workers=num_workers, pin_memory=True)
    
    all_z_feats = []
    all_y = [] 
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, z_proj, _ = model(batch)
            all_z_feats.append(z_proj.cpu())
            y = batch.y.view(-1)
            all_y.append(y.cpu())
            
    z_matrix = torch.cat(all_z_feats).numpy()
    y_array = torch.cat(all_y).numpy()
    
    kmeans = KMeans(n_clusters=num_prototypes, random_state=42, n_init=10).fit(z_matrix)
    
    prototypes_indices = []
    for k in range(num_prototypes):
        cluster_mask = (kmeans.labels_ == k)
        cluster_points = z_matrix[cluster_mask]
        if len(cluster_points) > 0:
            distances_to_center = cdist(cluster_points, kmeans.cluster_centers_[k].reshape(1, -1))
            nearest_idx_in_cluster = np.argmin(distances_to_center)
            original_indices = np.where(cluster_mask)[0]
            prototypes_indices.append(original_indices[nearest_idx_in_cluster])
        
    prototypes_z = torch.tensor(z_matrix[prototypes_indices], dtype=torch.float).to(device)
    
    y_consensuses = []
    for k in range(num_prototypes):
        cluster_mask = (kmeans.labels_ == k)
        if np.any(cluster_mask):
            consensus_val = np.mean(y_array[cluster_mask])
        else:
            consensus_val = 0.0
        y_consensuses.append(consensus_val)
        
    prototypes_y_consensus = torch.tensor(y_consensuses, dtype=torch.float).to(device)
    print(f"-> Anchors Established: {len(prototypes_z)} prototypes.")
    return prototypes_z, prototypes_y_consensus

def calculate_weights_for_source(model, source_dataset, prototypes, y_consensuses, y_mean, y_std, args, num_workers, return_details=False):
    model.eval()
    device = args.device
    
    inference_batch_size = 16384  
    
    loader = DataLoader(source_dataset, batch_size=inference_batch_size, shuffle=False,
                        follow_batch=['x_solvent', 'x_solute'],
                        num_workers=num_workers, 
                        pin_memory=True, 
                        persistent_workers=(num_workers > 0),
                        prefetch_factor=2 if num_workers > 0 else None) 
    
    all_weights = []
    save_details = return_details 
    all_E_stat = [] if save_details else None
    all_Delta_L = [] if save_details else None
    
    t_mean = y_mean.item()
    t_std = y_std.item()
    
    min_accept = t_mean - 3.0 * t_std
    max_accept = t_mean + 3.0 * t_std
    count_filtered = 0

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True): 
            for batch in tqdm(loader, desc="Calc Weights (Fast)", leave=False, ncols=100):
                batch = batch.to(device, non_blocking=True)
                
                pred, z_proj, _ = model(batch)
                gamma = pred[:, 0]
                batch_y = batch.y.view(-1).to(device)
                
                pred_physical = gamma * t_std + t_mean
                E_stat, y_consensus, _ = find_prototype_match(z_proj, prototypes, y_consensuses)
                
                raw_diff = pred_physical.view(-1) - y_consensus
                systematic_bias = torch.median(raw_diff) 
                delta_L = torch.abs(raw_diff - systematic_bias) 
                
                w_local = calculate_weights(E_stat, delta_L, args)
                
                range_mask = (batch_y >= min_accept) & (batch_y <= max_accept)
                solute_atoms = batch.global_feat[:, 2] * 30.0
                size_mask = (solute_atoms <= 100.0)
                final_mask = range_mask & size_mask
                
                w_final = w_local * final_mask.float()
                
                if save_details:
                    all_E_stat.append(E_stat.cpu())
                    all_Delta_L.append(delta_L.cpu())
                
                all_weights.append(w_final.cpu())
                count_filtered += (~final_mask).sum().item()

    raw_weights = torch.cat(all_weights)
    
    keep_ratio = args.keep_ratio
    top_k = int(len(raw_weights) * keep_ratio)
    if top_k > 0:
        threshold_val = torch.kthvalue(raw_weights, len(raw_weights) - top_k + 1).values
        mask = raw_weights >= threshold_val
        raw_weights[~mask] = 0.0
        max_val = raw_weights.max()
        if max_val > 0:
            raw_weights = raw_weights / max_val
            
    if save_details:
        return raw_weights, torch.cat(all_E_stat), torch.cat(all_Delta_L)
        
    return raw_weights, torch.tensor([]), torch.tensor([])

def train_epoch_source_weighted(model, loader, optimizer, device, y_mean, y_std, prototypes, y_consensuses, args, epoch, total_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Train Ep {epoch}", leave=False, ncols=100)
    
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        outputs, z_proj, _ = model(batch)
        gamma = outputs[:, 0]
        
        y = batch.y.view(-1).to(device)
        y_norm = (y - y_mean) / y_std 
        
        with torch.no_grad():
            E_stat, y_consensus, _ = find_prototype_match(z_proj, prototypes, y_consensuses)
            pred_physical = gamma * y_std + y_mean
            raw_diff = pred_physical.view(-1) - y_consensus
            batch_bias = torch.median(raw_diff) 
            delta_L = torch.abs(raw_diff - batch_bias)
            weights = calculate_weights(E_stat, delta_L, args)
        
        loss_vector = edl_loss_per_sample(outputs, y_norm, epoch, total_epochs, lambda_coef=args.edl_lambda)
        weighted_loss = torch.mean(loss_vector * weights.to(device)) 
        
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        current_loss = weighted_loss.item()
        total_loss += current_loss
        pbar.set_postfix({'loss': f"{current_loss:.4f}"})
        
    return total_loss / len(loader)

def evaluate(model, loader, device, y_mean, y_std, return_preds=False):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, _, _ = model(batch) 
            gamma = pred[:, 0]
            pred_val = gamma * y_std + y_mean
            y = batch.y.view(-1).cpu().numpy()
            preds.extend(pred_val.cpu().numpy())
            ys.extend(y)
    
    y_true = np.array(ys)
    y_pred = np.array(preds)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    relative_errors = np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))
    mape = np.mean(relative_errors) * 100 
    
    hits_10 = (relative_errors <= 0.10)
    acc_10 = np.mean(hits_10) * 100 
    hits_20 = (relative_errors <= 0.20)
    acc_20 = np.mean(hits_20) * 100
    
    if return_preds:
        return mae, rmse, mape, acc_10, acc_20, y_pred, y_true
    return mae, rmse, mape, acc_10, acc_20

# ==========================================
# PART 2: 主程序
# ==========================================

visualization_logs = {'epoch_stats': [], 'final_z': None, 'test_predictions': None}

def main(args):
    # --- [关键修改] 1. 生成唯一的实验 ID ---
    # 根据所有参与网格搜索的超参数生成一个字符串
    run_id = (f"N{args.num_prototypes}_KR{args.keep_ratio}_"
              f"Tau{args.tau_high}_Sig{args.sigma_0}_"
              f"W{args.w_min}_Lam{args.edl_lambda}")
    
    # --- [关键修改] 2. 创建独立的实验目录 ---
    experiment_dir = os.path.join("SSEP_experiments", run_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"--- DETS Full Pipeline (Accelerated) ---")
    print(f"Run ID: {run_id}")
    print(f"Output Directory: {experiment_dir}")
    print(f"Device: {args.device}")

    num_workers = 0
    print(f"Using num_workers: {num_workers}")

    # 1. 数据加载
    print("Loading Target Dataset...")
    target_full = SolvationDataset(root='./solvation_cache', csv_file=args.target_csv)
    
    print("Loading Source Dataset...")
    source_dataset = SolvationDataset(root='./solvation_cache', csv_file=args.source_csv)
    
    idx_target = list(range(len(target_full)))
    tr_idx, test_idx = train_test_split(idx_target, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(tr_idx, test_size=(0.1/0.9), random_state=42)
    
    from torch.utils.data import Subset
    target_train = Subset(target_full, train_idx)
    phase2_target_train = Subset(target_full, train_idx)
    target_val = Subset(target_full, val_idx)
    target_test = Subset(target_full, test_idx)
    
    y_target_raw = torch.tensor([target_full.targets[i] for i in train_idx], dtype=torch.float)
    y_mean = y_target_raw.mean().to(args.device)
    y_std = y_target_raw.std().to(args.device)
    
    sample = target_full[0]
    num_node_feat = sample.x_solvent.shape[1]
    num_edge_feat = sample.edge_attr_solvent.shape[1]
    
    model = GNNRegressor(
        num_node_features=num_node_feat,
        num_edge_features=num_edge_feat,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(args.device)
    
    if hasattr(torch, 'compile'):
        print("-> Using torch.compile for acceleration...")
        model = torch.compile(model)
    
    if os.path.exists(args.pretrained):
        print(f"-> Loading pretrained: {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)

    for layer in model.proj:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=3.0)
    torch.nn.init.xavier_uniform_(model.edl_head.weight)

    # ------------------------------------
    # PHASE 1
    # ------------------------------------
    print("\n=== PHASE 1: DETS Source Training ===")
    
    prototypes, y_consensuses = pre_calculate_prototypes_on_target(
        model, target_train, args.device, args.num_prototypes, num_workers
    )

    for param in model.parameters(): param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_phase1_mae = float('inf')
    patience_counter = 0
    
    target_val_loader = DataLoader(target_val, batch_size=args.batch_size, shuffle=False, 
                                   follow_batch=['x_solvent', 'x_solute'],
                                   num_workers=num_workers, pin_memory=True)
    target_test_loader = DataLoader(target_test, batch_size=args.batch_size, shuffle=False,
                                    follow_batch=['x_solvent', 'x_solute'],
                                    num_workers=num_workers, pin_memory=True)

    source_weights, E_stat, Delta_L = None, None, None
    weight_update_freq = 1 
    
    # [关键修改] 路径指向 experiment_dir
    p1_model_path = os.path.join(experiment_dir, "best_phase1_model.pt")

    for epoch in range(1, args.epochs + 1):
        
        if epoch == 1 or epoch % weight_update_freq == 0:
            print(f"-> [Ep {epoch}] Recalculating DETS weights on Source (1M)...")
            source_weights, E_stat, Delta_L = calculate_weights_for_source(
                model, source_dataset, prototypes, y_consensuses, y_mean, y_std, args, num_workers, return_details=True
            )
            
            visualization_logs['epoch_stats'].append({
                'epoch': epoch,
                'weights': source_weights.cpu().numpy()
            })
        
        n_active = (source_weights > 0).sum().item()
        
        if n_active == 0: continue

        current_batch_size = min(args.batch_size, n_active)
        if current_batch_size < 4: current_batch_size = 4
            
        sampler = WeightedRandomSampler(source_weights, num_samples=n_active, replacement=True)
        
        source_loader = DataLoader(source_dataset, batch_size=current_batch_size, sampler=sampler, 
                                   drop_last=False, follow_batch=['x_solvent', 'x_solute'],
                                   num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
        
        t0 = time.time()
        loss = train_epoch_source_weighted(
            model, source_loader, optimizer, args.device, 
            y_mean, y_std, prototypes, y_consensuses, args,
            epoch, args.epochs
        )
        train_time = time.time() - t0
        
        val_mae, val_rmse, val_mape, val_acc_10, val_acc_20 = evaluate(model, target_val_loader, args.device, y_mean, y_std)
        
        print(f"[P1] Ep {epoch} ({train_time:.1f}s) | Loss {loss:.4f} |Val MAE {val_mae:.4f} | RMSE {val_rmse:.4f} | MAPE {val_mape:.2f}% | Acc@10 {val_acc_10:.1f}% | Acc@20 {val_acc_20:.1f}%")
        
        if val_mae < best_phase1_mae:
            best_phase1_mae = val_mae
            # [关键修改] 保存路径
            torch.save(model.state_dict(), p1_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Phase 1 Early stopping.")
            break
            
    print("-> Extracting features for visualization...")
    def extract_full_z(loader):
        model.eval()
        z_list = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(args.device)
                _, z, _ = model(batch)
                z_list.append(z.cpu())
        return torch.cat(z_list)

    viz_loader_source = DataLoader(source_dataset, batch_size=1024, shuffle=False, 
                                   follow_batch=['x_solvent', 'x_solute'], num_workers=num_workers)
    viz_loader_target = DataLoader(target_train, batch_size=1024, shuffle=False, 
                                   follow_batch=['x_solvent', 'x_solute'], num_workers=num_workers)
    
    # visualization_logs['final_z'] = ...

    # ------------------------------------
    # PHASE 2
    # ------------------------------------
    print("\n=== PHASE 2: Target Fine-tuning ===")
    
    # [关键修改] 加载路径
    if os.path.exists(p1_model_path):
        model.load_state_dict(torch.load(p1_model_path))
    else:
        print(f"Warning: Phase 1 model not found at {p1_model_path}")
    
    # Stage 2.1
    print(">>> Stage 2.1: Head Alignment...")
    for param in model.parameters(): param.requires_grad = False
    for param in model.fc_final.parameters(): param.requires_grad = True
    for param in model.edl_head.parameters(): param.requires_grad = True
    
    ft_loader = DataLoader(phase2_target_train, batch_size=64, shuffle=True, 
                           follow_batch=['x_solvent', 'x_solute'],
                           num_workers=num_workers, pin_memory=True)
    
    optimizer_head = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    best_head_mae = float('inf')
    patience = 0
    # [关键修改] 路径指向 experiment_dir
    head_aligned_path = os.path.join(experiment_dir, "best_head_aligned.pt")
    
    for ep in range(1, 201):
        model.train()
        total_loss = 0
        for batch in ft_loader:
            batch = batch.to(args.device)
            optimizer_head.zero_grad()
            outputs, _, _ = model(batch)
            y = batch.y.view(-1).to(args.device)
            y_norm = (y - y_mean) / y_std
            loss_vector = edl_loss_per_sample(outputs, y_norm, ep, epoch + args.ft_epochs, lambda_coef=args.edl_lambda)
            loss = loss_vector.mean()
            loss.backward()
            optimizer_head.step()
            total_loss += loss.item()
            
        val_mae, val_rmse, val_mape, val_acc_10, val_acc_20 = evaluate(model, target_val_loader, args.device, y_mean, y_std)
        if val_mae < best_head_mae:
            best_head_mae = val_mae
            patience = 0
            # [关键修改] 保存
            torch.save(model.state_dict(), head_aligned_path)
        else:
            patience += 1
        if patience >= 50: break
        if ep % 20 == 0: print(f"[Head] Ep {ep} MAE {val_mae:.4f} MAPE {val_mape:.1f}%")

    # Stage 2.2
    print(">>> Stage 2.2: Full Fine-tuning...")
    # [关键修改] 加载
    if os.path.exists(head_aligned_path):
        model.load_state_dict(torch.load(head_aligned_path))
        
    for param in model.parameters(): param.requires_grad = True
    optimizer_all = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=args.wd)
    
    best_ft_mae = float('inf')
    patience = 0
    # [关键修改] 路径指向 experiment_dir
    final_model_path = os.path.join(experiment_dir, "final_model.pt")

    for ep in range(1, args.ft_epochs + 1):
        model.train()
        total_loss = 0
        for batch in ft_loader:
            batch = batch.to(args.device)
            optimizer_all.zero_grad()
            outputs, _, _ = model(batch)
            y = batch.y.view(-1).to(args.device)
            y_norm = (y - y_mean) / y_std
            loss_vector = edl_loss_per_sample(outputs, y_norm, ep, args.ft_epochs, lambda_coef=args.edl_lambda)
            loss = loss_vector.mean()
            loss.backward()
            optimizer_all.step()
            total_loss += loss.item()
            
        val_mae, val_rmse, val_mape, val_acc_10, val_acc_20 = evaluate(model, target_val_loader, args.device, y_mean, y_std)
        if ep % 10 == 0: print(f"[Full] Ep {ep} MAE {val_mae:.4f} MAPE {val_mape:.1f}%")
        
        if val_mae < best_ft_mae:
            best_ft_mae = val_mae
            patience = 0
            # [关键修改] 保存
            torch.save(model.state_dict(), final_model_path)
        else:
            patience += 1
        if patience >= 50: break

    print("\n>>> Testing Final Model...")
    # [关键修改] 加载
    if os.path.exists(final_model_path):
        model.load_state_dict(torch.load(final_model_path))
    else:
        print(f"Warning: Final model not found at {final_model_path}")

    test_mae, test_rmse, test_mape, test_acc_10, test_acc_20, preds, targets = evaluate(
        model, target_test_loader, args.device, y_mean, y_std, return_preds=True
    )
    visualization_logs['test_predictions'] = {'pred': preds, 'true': targets}
    
    print("-" * 60)
    print(f"[FINAL RESULT] Solvation Free Energy:")
    print(f"   MAE:      {test_mae:.4f} kcal/mol")
    print(f"   RMSE:     {test_rmse:.4f} kcal/mol")
    print(f"   MAPE:     {test_mape:.2f} %")
    print(f"   Acc@10%:  {test_acc_10:.2f} %")
    print(f"   Acc@20%:  {test_acc_20:.2f} %")
    print("-" * 60)
    
    # [关键修改] 保存可视化数据
    viz_path = os.path.join(experiment_dir, "viz_data.pt")
    torch.save(visualization_logs, viz_path)
    print(f"-> Visualization data saved to {viz_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_csv", type=str, default="../solute-solvent/exp_data.csv")
    parser.add_argument("--source_csv", type=str, default="../solute-solvent/qm.csv")
    parser.add_argument("--pretrained", type=str, default="")
    
    parser.add_argument("--batch_size", type=int, default=4096) 
    parser.add_argument("--epochs", type=int, default=200)                                                      
    parser.add_argument("--ft_epochs", type=int, default=200)
    
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)  
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)  
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    
    parser.add_argument("--keep_ratio", type=float, default=0.8)
    parser.add_argument("--num_prototypes", type=int, default=5)
    parser.add_argument("--tau_high", type=float, default=0.8)
    parser.add_argument("--sigma_0", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--w_min", type=float, default=1e-4)
    parser.add_argument("--edl_lambda", type=float, default=0.05)
    
    args = parser.parse_args()
    
    # [重要] 移除这里的全局清理，防止并行搜索时互相删除缓存
    # if os.path.exists("solvation_processed"): 
    #    shutil.rmtree("solvation_processed", ignore_errors=True)
    
    # 确保基础实验目录存在
    os.makedirs("DETS_experiments", exist_ok=True)
    
    main(args)