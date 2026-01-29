# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
import shutil
import time

from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore")

from dataset import SolvationDataset
from sampler import CoresetSampler 
class SilentTqdm:
    def __init__(self, iterable, *args, **kwargs): self.iterable = iterable
    def __iter__(self): return iter(self.iterable)
    def set_postfix(self, *args, **kwargs): pass
    def update(self, *args, **kwargs): pass
tqdm = SilentTqdm
# ==========================================
# PART 0: 适配 Baseline 的 Sampler 子类
# ==========================================
class BaselineSampler(CoresetSampler):
    """
    专门适配 Baseline 模型的采样器。
    重写了 _extract_metrics 以处理没有 EDL 参数(nu/alpha/beta)的情况。
    """
    # [修改] 将函数名改为 _extract_metrics
    def _extract_metrics(self, dataset, desc="Extracting Info"):
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, 
                            follow_batch=['x_solvent', 'x_solute'], 
                            num_workers=self.num_workers, pin_memory=True)
        
        all_z, all_y, all_preds, all_uncs = [], [], [], []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                batch = batch.to(self.device)
                
                # Baseline 模型返回 (pred, z, _)
                outputs, z, _ = self.model(batch)
                
                pred = outputs.view(-1)
                
                # Baseline 默认为 0 不确定性
                # 如果是 Entropy/LeastConf 采样，这里会自动使用 Loss 作为代理(因为 unc 全0会导致 fallback 或排序无效)
                # 更好的做法：对于 Baseline，sampler.py 的逻辑中 Entropy 应该 fallback 到 Loss。
                # 检查 sampler.py:
                # if method in ["Entropy", "Least Confidence"]: scores = src_data["unc"]
                # 如果 unc 全 0，这俩方法会失效（变成随机）。
                # 为了 Baseline 公平性，我们可以把 unc 设为 Loss (Error)
                # 因为在确定性回归中，Loss 是唯一的“不确定性”代理
                
                # 计算 Loss 作为不确定性代理
                y = batch.y.view(-1)
                error = torch.abs(y - pred)
                unc = error # 让 Entropy/LeastConf 实际上去选 Loss 最大的样本 (即 Hard Mining)
                
                all_z.append(z.cpu())
                all_y.append(y.cpu())
                all_preds.append(pred.cpu())
                all_uncs.append(unc.cpu())
                
        return {
            "z": torch.cat(all_z).numpy(),
            "y": torch.cat(all_y).numpy(),
            "pred": torch.cat(all_preds).numpy(),
            "unc": torch.cat(all_uncs).numpy()
        }
# ==========================================
# PART 1: Baseline 模型 (接口适配版)
# ==========================================
class BaselineGNNRegressor(nn.Module):
    def __init__(self, num_node_features, num_edge_features,
                 hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.gnn_solvent = self._build_gnn_backbone(num_node_features, num_edge_features, hidden_dim, num_layers)
        self.gnn_solute = self._build_gnn_backbone(num_node_features, num_edge_features, hidden_dim, num_layers)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 128), nn.Sigmoid()
        )
        input_dim_final = hidden_dim * 2 + 4 
        self.fc_final = nn.Sequential(
            nn.Linear(input_dim_final, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.output_head = nn.Linear(hidden_dim, 1) 
        self.global_pool = global_add_pool 

    def _build_gnn_backbone(self, num_node_features, num_edge_features, hidden_dim, num_layers):
        convs = nn.ModuleList()
        bns = nn.ModuleList()
        for i in range(num_layers):
            in_dim = num_node_features if i == 0 else hidden_dim
            nn_lin = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINEConv(nn_lin, edge_dim=num_edge_features)
            convs.append(conv)
            bns.append(nn.BatchNorm1d(hidden_dim))
        return nn.ModuleDict({'convs': convs, 'bns': bns})

    def forward(self, data):
        # [修改] 返回 (pred, emb, None) 以匹配 Sampler 接口
        prediction, emb = self.forward_with_embeddings(data)
        return prediction, emb, None

    def forward_with_embeddings(self, data):
        x_solvent = data.x_solvent
        h_solvent = self._forward_gnn(self.gnn_solvent, x_solvent, data.edge_index_solvent, data.edge_attr_solvent)
        g_solvent = global_add_pool(h_solvent, data.x_solvent_batch, size=data.num_graphs)
        
        x_solute = data.x_solute
        h_solute = self._forward_gnn(self.gnn_solute, x_solute, data.edge_index_solute, data.edge_attr_solute)
        g_solute = global_add_pool(h_solute, data.x_solute_batch, size=data.num_graphs)

        phys_feat = data.global_feat.view(g_solvent.shape[0], -1) 
        g_concat = torch.cat([g_solvent, g_solute, phys_feat], dim=1) 
        
        g_final = self.fc_final(g_concat)
        prediction = self.output_head(g_final) 
        return prediction, g_concat

    def _forward_gnn(self, gnn_module, x, edge_index, edge_attr):
        for conv, bn in zip(gnn_module['convs'], gnn_module['bns']):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# ==========================================
# PART 2: 评估函数
# ==========================================
def evaluate(model, loader, device, y_mean, y_std, return_preds=False):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # [修改] 解包 (pred, _, _)
            out, _, _ = model(batch)
            pred_val = out.view(-1) * y_std + y_mean
            y = batch.y.view(-1).cpu().numpy()
            preds.extend(pred_val.cpu().numpy())
            ys.extend(y)
    
    y_true = np.array(ys)
    y_pred = np.array(preds)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    abs_diff = np.abs(y_true - y_pred)
    hits_10 = abs_diff <= np.maximum(0.10 * np.abs(y_true), 0.5) 
    acc_10 = np.mean(hits_10) * 100 
    
    mape = np.mean(abs_diff / np.maximum(np.abs(y_true), 1.0)) * 100 
    
    if return_preds: return mae, rmse, mape, acc_10, y_pred, y_true
    return mae, rmse, mape, acc_10

# ==========================================
# PART 3: 主程序
# ==========================================
def main(args):
    print(f"--- Baseline Pipeline (Method: {args.sampling_method}) ---")
    
    DYNAMIC_METHODS = ["Soft Random", "InfoBatch", "MolPeg", "epsilon-greedy", "UCB"]
    IS_DYNAMIC = args.sampling_method in DYNAMIC_METHODS
    
    if IS_DYNAMIC:
        print("-> Mode: Dynamic Sampling (Update every {} epochs)".format(args.update_freq))
    else:
        print("-> Mode: Static Sampling (One-shot selection)")
    
    num_workers = 8

    # 1. 数据加载
    target_full = SolvationDataset(root='./solvation_cache', csv_file=args.target_csv)
    source_dataset = SolvationDataset(root='./solvation_cache', csv_file=args.source_csv)
    
    idx_target = list(range(len(target_full)))
    tr_idx, test_idx = train_test_split(idx_target, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(tr_idx, test_size=(0.1/0.9), random_state=42)
    
    from torch.utils.data import Subset
    target_train = Subset(target_full, train_idx)
    target_val = Subset(target_full, val_idx)
    target_test = Subset(target_full, test_idx)
    
    y_raw = torch.tensor([target_full.targets[i] for i in train_idx], dtype=torch.float)
    y_mean = y_raw.mean().to(args.device)
    y_std = y_raw.std().to(args.device)
    
    # 2. 模型初始化
    model = BaselineGNNRegressor(
        num_node_features=target_full[0].x_solvent.shape[1],
        num_edge_features=target_full[0].edge_attr_solvent.shape[1],
        hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(args.device)
    
    if hasattr(torch, 'compile'): model = torch.compile(model)

    # 3. [修改] 使用 BaselineSampler
    sampler = BaselineSampler(model, source_dataset, target_train, args.device, num_workers=num_workers)
    
    # Stage 0: Initial Selection
    train_source_dataset = source_dataset 
    if args.sampling_method != 'Full':
        print(f"\n>>> [Stage 0] Initial Coreset Selection: {args.sampling_method}")
        indices = sampler.select(args.sampling_method, args.sampling_ratio)
        train_source_dataset = Subset(source_dataset, indices)
        print(f"-> Initial Size: {len(train_source_dataset)}")

    # Stage 1: Pre-training
    print(f"\n=== Stage 1: Pre-training ===")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    
    target_val_loader = DataLoader(target_val, batch_size=args.batch_size, shuffle=False, 
                                   follow_batch=['x_solvent', 'x_solute'], num_workers=num_workers, pin_memory=True)

    source_loader = DataLoader(train_source_dataset, batch_size=args.source_batch_size, shuffle=True, 
                               follow_batch=['x_solvent', 'x_solute'], num_workers=num_workers, pin_memory=True)

    best_pre_rmse = float('inf')
    
    for epoch in range(1, args.pre_epochs + 1):
        # 动态采样
        if IS_DYNAMIC and epoch > 1 and (epoch % args.update_freq == 0):
            print(f"\n-> [Ep {epoch}] Re-sampling Coreset via {args.sampling_method}...")
            new_indices = sampler.select(args.sampling_method, args.sampling_ratio)
            train_source_dataset = Subset(source_dataset, new_indices)
            source_loader = DataLoader(train_source_dataset, batch_size=args.source_batch_size, shuffle=True, 
                                       follow_batch=['x_solvent', 'x_solute'], num_workers=num_workers, pin_memory=True)
            print(f"-> Coreset Updated. Size: {len(train_source_dataset)}")

        model.train()
        total_loss = 0
        pbar = tqdm(source_loader, desc=f"Ep {epoch}", leave=False, ncols=100)
        
        for batch in pbar:
            batch = batch.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # [修改] 解包 (pred, _, _)
                pred, _, _ = model(batch)
                pred = pred.view(-1)
                y = batch.y.view(-1).to(args.device)
                y_norm = (y - y_mean) / y_std 
                loss = criterion(pred, y_norm)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        if epoch % 1 == 0:
            val_mae, val_rmse, val_mape, val_acc = evaluate(model, target_val_loader, args.device, y_mean, y_std)
            print(f"[P1] Ep {epoch} | Loss {total_loss/len(source_loader):.4f} | "
                  f"RMSE {val_rmse:.4f} | MAE {val_mae:.4f} | Acc {val_acc:.1f}%")
            
            if val_rmse < best_pre_rmse:
                best_pre_rmse = val_rmse
                torch.save(model.state_dict(), "baseline_pretrain.pt")

    # Stage 2: Target Fine-tuning
    print("\n=== Stage 2: Fine-tuning ===")
    if os.path.exists("baseline_pretrain.pt") and args.pre_epochs > 0:
        model.load_state_dict(torch.load("baseline_pretrain.pt"))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.5, weight_decay=args.wd)
    target_train_loader = DataLoader(target_train, batch_size=args.batch_size, shuffle=True, 
                                     follow_batch=['x_solvent', 'x_solute'], num_workers=num_workers, pin_memory=True)
    
    best_ft_rmse = float('inf')
    patience = 0
    
    for epoch in range(1, args.ft_epochs + 1):
        model.train()
        total_loss = 0
        for batch in target_train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()
            # [修改] 解包
            pred, _, _ = model(batch)
            pred = pred.view(-1)
            y = batch.y.view(-1).to(args.device)
            y_norm = (y - y_mean) / y_std 
            loss = criterion(pred, y_norm)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        val_mae, val_rmse, val_mape, val_acc = evaluate(model, target_val_loader, args.device, y_mean, y_std)
        
        if epoch % 10 == 0:
            print(f"[FT] Ep {epoch} | Val RMSE {val_rmse:.4f} | MAE {val_mae:.4f}")
            
        if val_rmse < best_ft_rmse:
            best_ft_rmse = val_rmse
            torch.save(model.state_dict(), "baseline_final.pt")
            patience = 0
        else:
            patience += 1
            if patience >= args.patience: break

    # Final Test
    print("\n>>> Testing Final Model...")
    model.load_state_dict(torch.load("baseline_final.pt"))
    test_loader = DataLoader(target_test, batch_size=args.batch_size, shuffle=False, 
                             follow_batch=['x_solvent', 'x_solute'], num_workers=num_workers)
    
    test_mae, test_rmse, test_mape, test_acc = evaluate(model, test_loader, args.device, y_mean, y_std)
    print("-" * 60)
    print(f"[BASELINE RESULT] Method: {args.sampling_method}")
    print(f"MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | MAPE: {test_mape:.2f}% | Acc: {test_acc:.2f}%")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_csv", type=str, default="./solute-solvent/exp_data.csv")
    parser.add_argument("--source_csv", type=str, default="./solute-solvent/qm.csv")
    
    parser.add_argument("--sampling_method", type=str, default="Full", 
                        choices=["Full", 
                                 "Hard Random", "Herding", "K-Means", 
                                 "Entropy", "Least Confidence", "GraNd-20", "Glister", "Influence", 
                                 "EL2N-2", "DP", "CSIE",
                                 "Soft Random", "InfoBatch", "MolPeg", "epsilon-greedy", "UCB"],
                        help="Baseline selection method")
    parser.add_argument("--sampling_ratio", type=float, default=0.1)
    parser.add_argument("--update_freq", type=int, default=5)

    parser.add_argument("--source_batch_size", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pre_epochs", type=int, default=100)                                                      
    parser.add_argument("--ft_epochs", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)  
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)  
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    if os.path.exists("solvation_processed"): 
        shutil.rmtree("solvation_processed", ignore_errors=True)
        
    main(args)