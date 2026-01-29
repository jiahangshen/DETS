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
from rdkit.Chem import Descriptors
from ogb.utils import smiles2graph
from rdkit import RDLogger
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans
import os
import shutil
from dataset import ATCTDataset
from model import GNNRegressor
# 屏蔽 RDKit 警告
RDLogger.DisableLog('rdApp.*')

# 清理缓存防止索引错误
if os.path.exists('enthalpy/processed'):
    shutil.rmtree('enthalpy/processed', ignore_errors=True)
    print("!!! PROCESSED CACHE DELETED !!!")

# ==========================================

# ==========================================
# PART 1: 核心数学与辅助函数
# ==========================================

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
    
    # 动态证据: 重新启用，因为 Delta_L 已经归一化且去除了系统偏差
    sigma_eff = args.sigma_0 * (1.0 + args.gamma * Delta_L_soft)
    w_soft = args.alpha * torch.exp(- (D_struct**2) / (2 * sigma_eff**2 + 1e-8))
    
    weights[~hard_anchor_mask] = torch.clamp(w_soft, min=args.w_min, max=args.alpha)
    weights[hard_anchor_mask] = 1.0
    return weights.double()

def pre_calculate_prototypes_on_target(model, target_dataset, device, num_prototypes):
    print(f"-> Extracting Anchors (Intensive Property)...")
    model.eval()
    model.to(device)
    loader = DataLoader(target_dataset, batch_size=256, shuffle=False)
    
    all_z_feats = []
    all_y_per_atom = [] 
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, z_proj, _ = model(batch)
            all_z_feats.append(z_proj.cpu())
            
            # 还原真实原子数，计算强度性质
            n_atoms = batch.global_feat[:, 0] * 30.0
            n_atoms = torch.clamp(n_atoms, min=1.0)
            y = batch.y.view(-1)
            y_per_atom = y / n_atoms
            all_y_per_atom.append(y_per_atom.cpu())
            
    z_matrix = torch.cat(all_z_feats).numpy()
    y_per_atom_array = torch.cat(all_y_per_atom).numpy()
    
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
            consensus_val = np.mean(y_per_atom_array[cluster_mask])
        else:
            consensus_val = 0.0
        y_consensuses.append(consensus_val)
        
    prototypes_y_consensus = torch.tensor(y_consensuses, dtype=torch.float).to(device)
    print(f"-> Anchors Established: {len(prototypes_z)} prototypes.")
    return prototypes_z, prototypes_y_consensus

def calculate_weights_for_source(model, source_dataset, prototypes, y_consensuses, y_mean, y_std, args, return_details=False):
    """
    修改版：支持 return_details，用于可视化数据埋点
    """
    model.eval()
    device = args.device
    loader = DataLoader(source_dataset, batch_size=512, shuffle=False)
    
    all_weights = []
    all_E_stat = []
    all_Delta_L = []
    
    t_mean = y_mean.item()
    t_std = y_std.item()
    min_accept = t_mean - 2.5 * t_std
    max_accept = t_mean + 2.5 * t_std
    
    count_range_filtered = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, z_proj, _ = model(batch)
            batch_y = batch.y.view(-1)
            
            # 1. 物理计算
            pred_physical = pred * y_std + y_mean
            n_atoms = batch.global_feat[:, 0] * 30.0
            n_atoms = torch.clamp(n_atoms, min=1.0).to(device)
            pred_per_atom = pred_physical / n_atoms
            
            # 2. 原型匹配
            E_stat, y_consensus_per_atom, _ = find_prototype_match(z_proj, prototypes, y_consensuses)
            
            # 3. 差分对齐 (Differential Alignment)
            raw_diff = pred_per_atom.view(-1) - y_consensus_per_atom
            systematic_bias = torch.median(raw_diff) # 估计系统偏差
            delta_L = torch.abs(raw_diff - systematic_bias) # 提取特异性
            
            w_local = calculate_weights(E_stat, delta_L, args).cpu()
            
            # 4. 范围过滤 & 尺寸过滤
            range_mask = (batch_y.cpu() >= min_accept) & (batch_y.cpu() <= max_accept)
            size_mask = (n_atoms.cpu().view(-1) <= 20.0) # 只允许原子数<=20
            final_mask = range_mask & size_mask
            
            w_final = w_local * final_mask.float()
            
            all_weights.append(w_final)
            if return_details:
                all_E_stat.append(E_stat.cpu())
                all_Delta_L.append(delta_L.cpu())
            
            count_range_filtered += (~final_mask).sum().item()

    raw_weights = torch.cat(all_weights)
    
    # 5. Hard Cutoff
    keep_ratio = 0.30
    top_k = int(len(raw_weights) * keep_ratio)
    if top_k > 0:
        threshold_val = torch.topk(raw_weights, top_k)[0][-1]
        mask = raw_weights >= threshold_val
        raw_weights[~mask] = 0.0
        max_val = raw_weights.max()
        if max_val > 0:
            raw_weights = raw_weights / max_val
            
    print(f"   [Debug] Filtered {count_range_filtered} mols via Range/Size. Active Mean: {raw_weights[raw_weights>0].mean():.4f}")

    if return_details:
        return raw_weights, torch.cat(all_E_stat), torch.cat(all_Delta_L)
    return raw_weights

def train_epoch_source_weighted(model, loader, optimizer, device, y_mean, y_std, prototypes, y_consensuses, args):
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred, z_proj, _ = model(batch)
        y = batch.y.view(-1).to(device)
        y_norm = (y - y_mean) / y_std 
        
        with torch.no_grad():
            # 训练时实时计算权重 (为了梯度缩放)
            # 注意：为了速度，这里简化计算，不进行复杂的差分对齐，或者沿用之前计算好的权重
            # 最好的方式是直接信赖 Sampler，这里只做简单的 E_stat 检查
            E_stat, y_consensus, _ = find_prototype_match(z_proj, prototypes, y_consensuses)
            # 简单回退到纯结构权重，避免复杂计算
            weights = E_stat 
        
        mse_loss = F.mse_loss(pred, y_norm, reduction='none') 
        # Double Down: 采样已经选了好的，Loss 再加权一次结构相似度
        weighted_loss = torch.mean(mse_loss * weights) 
        
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        total_loss += weighted_loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, device, y_mean, y_std, return_preds=False):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, _, _ = model(batch) 
            pred = pred * y_std + y_mean
            y = batch.y.view(-1).cpu().numpy()
            preds.extend(pred.cpu().numpy())
            ys.extend(y)
    
    y_true = np.array(ys)
    y_pred = np.array(preds)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    relative_errors = np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))
    mape = np.mean(relative_errors) * 100 
    hits = (relative_errors <= 0.10)
    acc_10 = np.mean(hits) * 100 
    
    if return_preds:
        return mae, rmse, mape, acc_10, y_pred, y_true
    return mae, rmse, mape, acc_10

# ==========================================
# PART 2: 主程序 (Full Pipeline + Visualization Logs)
# ==========================================

# 全局日志字典 (用于生成12张图)
visualization_logs = {
    'epoch_stats': [],       # 方案 1, 4, 10
    'final_z': None,         # 方案 7
    'test_predictions': None,# 方案 3, 6, 12
    'source_smiles': None,   # 方案 11
    'target_smiles': None,
    'source_targets': None,  # 方案 6 (误差 vs 大小)
    'source_sizes': None     # 方案 6
}

def main(args):
    print(f"--- DETS Full Pipeline with Visualization Hooks ---")
    print(f"Device: {args.device}")

    # 1. 数据加载
    target_full = ATCTDataset(args.target_csv)
    idx_target = range(len(target_full))
    tr_idx, test_idx = train_test_split(idx_target, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(tr_idx, test_size=(0.1/0.9), random_state=42)
    
    if args.is_all_train:
        target_train = target_full
    else:
        target_train = target_full.index_select(train_idx)
    phase2_target_train = target_full.index_select(train_idx) # Phase 2 用纯 Train
    target_val = target_full.index_select(val_idx)
    target_test = target_full.index_select(test_idx)
    
    source_dataset = ATCTDataset(args.source_csv)
    
    # 2. 统计量
    y_target_raw = torch.tensor([target_full.targets[i] for i in train_idx], dtype=torch.float)
    y_mean = y_target_raw.mean().to(args.device)
    y_std = y_target_raw.std().to(args.device)
    
    # 3. 模型初始化
    sample = target_full[0]
    model = GNNRegressor(
        num_node_features=sample.x.shape[1],
        num_edge_features=sample.edge_attr.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(args.device)
    
    if os.path.exists(args.pretrained):
        print(f"-> Loading pretrained: {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("-> [Warning] Scratch Training.")

    # 稀疏初始化 (解决特征坍塌)
    print("-> Applying Sparsity Init...")
    for layer in model.proj:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=3.0)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, -1.0)
    for layer in model.fc_final:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.xavier_uniform_(model.mu_head.weight)

    # ------------------------------------
    # PHASE 1: DETS Guided Pre-training
    # ------------------------------------
    print("\n=== PHASE 1: DETS Source Training ===")
    
    prototypes, y_consensuses = pre_calculate_prototypes_on_target(
        model, target_train, args.device, num_prototypes=args.num_prototypes
    )

    for param in model.parameters(): param.requires_grad = True
    optimizer = torch.optim.Adam([
        {"params": model.proj.parameters(), "lr": args.lr},       
        {"params": model.fc_final.parameters(), "lr": args.lr},   
        {"params": model.mu_head.parameters(), "lr": args.lr},
        {"params": model.convs.parameters(), "lr": args.lr * 0.1},
        {"params": model.bns.parameters(), "lr": args.lr * 0.1},
    ], weight_decay=args.wd)

    best_phase1_mae = float('inf')
    patience_counter = 0
    target_val_loader = DataLoader(target_val, batch_size=args.batch_size, shuffle=False)
    target_test_loader = DataLoader(target_test, batch_size=args.batch_size, shuffle=False)

    for epoch in range(1, args.epochs + 1):
        # 计算权重并获取详细信息 (用于可视化)
        source_weights, E_stat, Delta_L = calculate_weights_for_source(
            model, source_dataset, prototypes, y_consensuses, y_mean, y_std, args, return_details=True
        )
        
        # [Log] 记录本 Epoch 状态 (方案 1, 4, 10)
        epoch_log = {
            'epoch': epoch,
            'weights': source_weights.cpu().numpy(),
            'E_stat': E_stat.cpu().numpy(),
            'Delta_L': Delta_L.cpu().numpy(),
            # 记录系统温度
            'avg_sigma': (args.sigma_0 * (1 + args.gamma * Delta_L)).mean().item()
        }
        visualization_logs['epoch_stats'].append(epoch_log)
        
        # 打印审计
        n_active = (source_weights > 0).sum().item()
        print(f"-> [Ep {epoch}] Active Samples: {n_active}/{len(source_dataset)}")
        
        sampler = WeightedRandomSampler(source_weights, num_samples=n_active, replacement=True)
        # 防止 n_active 太小导致 batch 报错
        if n_active < args.batch_size:
            current_batch_size = max(4, n_active)
        else:
            current_batch_size = args.batch_size
            
        source_loader = DataLoader(source_dataset, batch_size=current_batch_size, sampler=sampler, drop_last=False)
        
        loss = train_epoch_source_weighted(
            model, source_loader, optimizer, args.device, 
            y_mean, y_std, prototypes, y_consensuses, args
        )
        
        val_mae, val_rmse, val_mape, val_acc = evaluate(model, target_val_loader, args.device, y_mean, y_std)
        print(f"[P1] Loss {loss:.4f} | Val MAE {val_mae:.4f} | Acc {val_acc:.1f}%")

        if val_mae < best_phase1_mae:
            best_phase1_mae = val_mae
            torch.save(model.state_dict(), "DETS/best_phase1_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Phase 1 Early stopping.")
            break
            
    # [Log] Phase 1 结束，保存中间数据
    print("-> Extracting features for visualization (Full Z vectors)...")
    
    # --- [定义提取函数] ---
    def extract_full_z(loader):
        model.eval()
        z_list = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(args.device)
                _, z, _ = model(batch)
                z_list.append(z.cpu())
        return torch.cat(z_list)
    # ----------------------

    # 创建临时的 Loader (无序，大 Batch)
    source_viz_loader = DataLoader(source_dataset, batch_size=512, shuffle=False)
    target_viz_loader = DataLoader(target_train, batch_size=512, shuffle=False)
    
    # 提取全量 Z
    z_source_full = extract_full_z(source_viz_loader)
    z_target_full = extract_full_z(target_viz_loader)
    
    # 保存到 logs
    visualization_logs['final_z'] = {
        'source': z_source_full.numpy(), 
        'target': z_target_full.numpy()
    }
    
    # 验证一下长度
    print(f"   Extracted: Source={len(z_source_full)}, Target={len(z_target_full)}")
    # 记录 SMILES 和 属性 (方案 6, 11)
    visualization_logs['source_smiles'] = source_dataset.smiles_list
    visualization_logs['target_smiles'] = [target_full.smiles_list[i] for i in train_idx]
    
    # 获取原子数用于方案 6 (MW vs Error)
    source_loader_viz = DataLoader(source_dataset, batch_size=512, shuffle=False)
    sizes = []
    for b in source_loader_viz: sizes.append(b.global_feat[:,0].numpy() * 30.0)
    visualization_logs['source_sizes'] = np.concatenate(sizes)
    visualization_logs['source_targets'] = source_dataset.targets

    # ------------------------------------
    # PHASE 2: Target Fine-tuning
    # ------------------------------------
    print("\n=== PHASE 2: Target Fine-tuning ===")
    model.load_state_dict(torch.load("DETS/best_phase1_model.pt"))
    
    # Stage 2.1: Head Align
    print(">>> Stage 2.1: Head Alignment...")
    for param in model.parameters(): param.requires_grad = False
    for param in model.fc_final.parameters(): param.requires_grad = True
    for param in model.mu_head.parameters(): param.requires_grad = True
    
    ft_loader = DataLoader(phase2_target_train, batch_size=32, shuffle=True)
    optimizer_head = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    
    best_head_mae = float('inf')
    patience = 0
    for epoch in range(1, 501):
        model.train()
        for batch in ft_loader:
            batch = batch.to(args.device)
            optimizer_head.zero_grad()
            pred, _, _ = model(batch) 
            y = batch.y.view(-1).to(args.device)
            y_norm = (y - y_mean) / y_std
            loss = F.mse_loss(pred, y_norm)
            loss.backward()
            optimizer_head.step()
            
        val_mae, _, _, val_acc = evaluate(model, target_val_loader, args.device, y_mean, y_std)
        if val_mae < best_head_mae:
            best_head_mae = val_mae
            patience = 0
            torch.save(model.state_dict(), "DETS/best_head_aligned.pt")
        else:
            patience += 1
        if patience >= 250: break
        if epoch % 10 == 0: print(f"[Head] Ep {epoch} MAE {val_mae:.4f} Acc {val_acc:.1f}%")

    # Stage 2.2: Full Fine-tune
    print(">>> Stage 2.2: Full Fine-tuning...")
    model.load_state_dict(torch.load("DETS/best_head_aligned.pt"))
    for param in model.parameters(): param.requires_grad = True
    optimizer_all = torch.optim.Adam(model.parameters(), lr=args.lr * 0.05, weight_decay=args.wd)
    
    best_ft_mae = float('inf')
    patience = 0
    for epoch in range(1, args.ft_epochs + 1):
        model.train()
        for batch in ft_loader:
            batch = batch.to(args.device)
            optimizer_all.zero_grad()
            pred, _, _ = model(batch) 
            y = batch.y.view(-1).to(args.device)
            y_norm = (y - y_mean) / y_std
            loss = F.mse_loss(pred, y_norm)
            loss.backward()
            optimizer_all.step()
            
        val_mae, _, _, val_acc = evaluate(model, target_val_loader, args.device, y_mean, y_std)
        if epoch % 10 == 0: print(f"[Full] Ep {epoch} MAE {val_mae:.4f} Acc {val_acc:.1f}%")
        
        if val_mae < best_ft_mae:
            best_ft_mae = val_mae
            patience = 0
            torch.save(model.state_dict(), "DETS/final_model.pt")
        else:
            patience += 1
        if patience >= 100: break

    # ------------------------------------
    # FINAL & LOGGING
    # ------------------------------------
    print("\n>>> Testing Final Model...")
    model.load_state_dict(torch.load("DETS/final_model.pt"))
    # [Log] 获取 Test Set 预测结果 (方案 3, 12)
    test_mae, test_rmse, test_mape, test_acc, preds, targets = evaluate(
        model, target_test_loader, args.device, y_mean, y_std, return_preds=True
    )
    visualization_logs['test_predictions'] = {'pred': preds, 'true': targets}
    
    print("-" * 60)
    print(f"[FINAL RESULT] Performance on Test Set:")
    print(f"   MAE:      {test_mae:.4f} kcal/mol")
    print(f"   RMSE:     {test_rmse:.4f} kcal/mol")
    print(f"   MAPE:     {test_mape:.2f} %")
    print(f"   Acc(10%): {test_acc:.2f} %")
    print("-" * 60)
    
    # 保存可视化数据
    torch.save(visualization_logs, 'DETS/viz_data.pt')
    print("-> Visualization data saved to 'DETS/viz_data.pt'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_csv", type=str, default="enthalpy/atct.csv")
    parser.add_argument("--source_csv", type=str, default="enthalpy/wudily_cho.csv")
    parser.add_argument("--pretrained", type=str, default="DETS1/new_df_core.pt")
    parser.add_argument("--is_all_train", type=bool, default=False)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500, help="Phase 1 Epochs")                                                      
    parser.add_argument("--ft_epochs", type=int, default=250, help="Phase 2 Epochs")
    
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4)  
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=150)  
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    
    # DETS 核心参数
    parser.add_argument("--num_prototypes", type=int, default=5)
    parser.add_argument("--tau_high", type=float, default=0.80)
    parser.add_argument("--sigma_0", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--w_min", type=float, default=1e-4)

    args = parser.parse_args()
    print(args)
    main(args)