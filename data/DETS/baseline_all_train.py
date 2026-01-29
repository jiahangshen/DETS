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
from tqdm import tqdm

from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore")

from dataset import ATCTDataset
from sampler import CoresetSamplerGNN

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ==========================================
# PART 1: Baseline 模型 (修复版)
# ==========================================
class BaselineGNNRegressor(nn.Module):
    def __init__(self, num_node_features, num_edge_features,
                 hidden_dim=256, num_layers=4, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.input_proj = nn.Linear(num_node_features, hidden_dim)

        # GNN Backbone
        for i in range(num_layers):
            nn_lin = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINEConv(nn_lin, edge_dim=num_edge_features)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout
        
        # 投影头 (用于采样器计算相似度)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64) 
        )
        
        # [核心修复] 预测头 (Standard Regression Head, Output Dim=1)
        # 动态初始化，将在 forward 中根据 global_feat 维度决定输入
        self.fc_final = None 
        self.hidden_dim = hidden_dim

    def _init_heads(self, global_feat_dim):
        # 拼接结构特征(hidden_dim) + 物理特征(global_feat_dim)
        input_dim = self.hidden_dim + global_feat_dim
        
        self.fc_final = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim), 
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1) # [修改] 输出 1 维标量
        )
        
        # 初始化权重
        for m in self.fc_final.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        self.to(next(self.parameters()).device)

    def forward(self, data):
        # 默认返回预测值
        pred, _ = self.forward_with_features(data)
        return pred

    def forward_with_features(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 第一次运行时动态初始化 Head
        if self.fc_final is None:
            global_feat_dim = data.global_feat.shape[1]
            self._init_heads(global_feat_dim)

        x = self.input_proj(x)

        # GNN Backbone
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(x, edge_index, edge_attr)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h 

        # Global Pooling
        g = global_add_pool(x, batch) 
        
        # Projection Feature (用于采样)
        z = self.proj(g)
        z = F.normalize(z, p=2, dim=1)
        
        # Feature Injection (拼接物理特征)
        phys_feat = data.global_feat.view(g.shape[0], -1) 
        g_concat = torch.cat([g, phys_feat], dim=1) 
        
        # Prediction
        # [修复] 这里调用 self.fc_final 而不是 self.fc
        pred = self.fc_final(g_concat)
        
        return pred, z

# ==========================================
# PART 2: 辅助函数
# ==========================================
def train_epoch(model, loader, optimizer, device, y_mean, y_std):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # [修改] 获取预测值 (pred: [Batch, 1])
        pred, _ = model.forward_with_features(batch)
        pred = pred.view(-1)
        
        # 使用 Total Enthalpy (标准化)
        y = batch.y.view(-1).to(device)
        y_norm = (y - y_mean) / y_std
        
        loss = F.mse_loss(pred, y_norm)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)

def evaluate(model, loader, device, y_mean, y_std, return_arrays=False, abs_tol=2.0):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # 获取预测值
            pred, _ = model.forward_with_features(batch)
            pred = pred.view(-1)
            
            # 反归一化
            pred_real = pred * y_std + y_mean
            
            y = batch.y.view(-1).cpu().numpy()
            preds.extend(pred_real.cpu().numpy())
            ys.extend(y)
    
    y_true = np.array(ys)
    y_pred = np.array(preds)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 混合容忍度 Acc
    abs_diff = np.abs(y_true - y_pred)
    tol_threshold = np.maximum(0.10 * np.abs(y_true), abs_tol)
    acc_10 = np.mean(abs_diff <= tol_threshold) * 100
    
    # MAPE
    mape = np.mean(abs_diff / np.maximum(np.abs(y_true), 1e-6)) * 100
    
    if return_arrays:
        return mae, rmse, mape, acc_10, y_pred, y_true
    return mae, rmse, mape, acc_10

# ==========================================
# PART 3: 主程序
# ==========================================
import random
def main(args):
    print(f"--- Enthalpy Baseline: {args.sampling_method} (Ratio: {args.sampling_ratio}) ---")
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Data
    print("Loading datasets...")
    target_ds = ATCTDataset(args.target_csv)
    source_ds = ATCTDataset(args.source_csv) # Wudily_Cho
    
    # Split Target
    idx_target = range(len(target_ds))
    tr_idx, test_idx = train_test_split(idx_target, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(tr_idx, test_size=0.1, random_state=42)
    
    target_train = target_ds.index_select(train_idx)
    target_val = target_ds.index_select(val_idx)
    target_test = target_ds.index_select(test_idx)
    
    target_val_loader = DataLoader(target_val, batch_size=args.batch_size, shuffle=False)
    target_test_loader = DataLoader(target_test, batch_size=args.batch_size, shuffle=False)

    # 统计量
    ys = torch.cat([d.y for d in target_train])
    y_mean, y_std = ys.mean().to(args.device), ys.std().to(args.device)
    print(f"Target Stats: Mean={y_mean.item():.2f}, Std={y_std.item():.2f}")

    # 2. Init Model
    sample = target_ds[0]
    # [修改] 使用修正后的 BaselineGNNRegressor
    model = BaselineGNNRegressor(
        num_node_features=sample.x.shape[1],
        num_edge_features=sample.edge_attr.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(args.device)
    
    # Init Weights
    if os.path.exists(args.pretrained):
        print(f"Loading pretrained backbone from {args.pretrained}")
        state = torch.load(args.pretrained, map_location=args.device)
        model.load_state_dict(state, strict=False)

    # ==========================================
    # Stage 0: Coreset Selection
    # ==========================================
    train_source_dataset = source_ds
    
    if args.sampling_method != 'Full':
        print(f"\n>>> [Stage 0] Sampling...")
        sampler = CoresetSamplerGNN(model, source_ds, target_train, args.device)
        # 传入 y_mean, y_std
        indices = sampler.select(args.sampling_method, args.sampling_ratio, y_mean, y_std)
        train_source_dataset = source_ds.index_select(indices) # PyG Subset
        print(f"-> Source reduced: {len(source_ds)} -> {len(train_source_dataset)}")

    # ==========================================
    # Stage 1: Source Pre-training
    # ==========================================
    print("\n=== Phase 1: Source Pre-training ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True)
    best_p1_mae = float("inf")
    
    # 动态采样逻辑所需变量
    DYNAMIC_METHODS = ["Soft Random", "InfoBatch", "MolPeg", "epsilon-greedy", "UCB"]
    IS_DYNAMIC = args.sampling_method in DYNAMIC_METHODS
    
    for epoch in range(1, args.epochs + 1):
        # 动态采样更新
        if IS_DYNAMIC and epoch > 1 and (epoch % args.update_freq == 0):
            print(f"\n-> [Ep {epoch}] Re-sampling Coreset via {args.sampling_method}...")
            sampler = CoresetSamplerGNN(model, source_ds, target_train, args.device)
            indices = sampler.select(args.sampling_method, args.sampling_ratio, y_mean, y_std)
            train_source_dataset = source_ds.index_select(indices)
            source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            print(f"-> Coreset Updated. Size: {len(train_source_dataset)}")

        loss = train_epoch(model, source_loader, optimizer, args.device, y_mean, y_std)
        
        if epoch % 10 == 0:
            val_mae, _, _, val_acc = evaluate(model, target_val_loader, args.device, y_mean, y_std, abs_tol=args.abs_tol)
            print(f"[P1] Ep {epoch}: Loss {loss:.4f} | Val MAE {val_mae:.4f}")
            if val_mae < best_p1_mae:
                best_p1_mae = val_mae
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_p1.pt"))

    # ==========================================
    # Stage 2: Target Fine-tuning
    # ==========================================
    print("\n=== Phase 2: Target Fine-tuning ===")
    if os.path.exists(os.path.join(args.save_dir, "best_p1.pt")):
        model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_p1.pt")))
    
    # Freeze logic (Baseline standard)
    for param in model.convs.parameters(): param.requires_grad = False
    for param in model.bns.parameters(): param.requires_grad = False
    
    # Unfreeze Last Layer & FC
    if model.fc_final is not None:
        for param in model.fc_final.parameters(): param.requires_grad = True
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    
    target_train_loader = DataLoader(target_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    best_ft_mae = float('inf')
    patience = 0
    final_path = os.path.join(args.save_dir, "final_model.pt")
    
    for epoch in range(1, args.ft_epochs + 1):
        loss = train_epoch(model, target_train_loader, optimizer, args.device, y_mean, y_std)
        val_mae, _, _, _ = evaluate(model, target_val_loader, args.device, y_mean, y_std, abs_tol=args.abs_tol)
        
        if epoch % 10 == 0:
            print(f"[FT] Ep {epoch}: Loss {loss:.4f} | Val MAE {val_mae:.4f}")
            
        if val_mae < best_ft_mae:
            best_ft_mae = val_mae
            torch.save(model.state_dict(), final_path)
            patience = 0
        else:
            patience += 1
            if patience >= args.patience: 
                print("Early stopping triggered.")
                break

    # Final Test
    print("\n>>> Testing Final Model...")
    if os.path.exists(final_path):
        model.load_state_dict(torch.load(final_path))
        test_mae, test_rmse, test_mape, test_acc, y_pred, y_true = evaluate(
            model, target_test_loader, args.device, y_mean, y_std, return_arrays=True, abs_tol=args.abs_tol
        )
        
        print("-" * 60)
        print(f"[BASELINE RESULT] Method: {args.sampling_method}")
        print(f"   MAE:  {test_mae:.4f}")
        print(f"   RMSE: {test_rmse:.4f}")
        print(f"   MAPE: {test_mape:.2f} %")
        print(f"   Acc:  {test_acc:.2f} %")
        print("-" * 60)
        
        # Save results
        np.save(os.path.join(args.save_dir, f"preds_{args.sampling_method}.npy"), {
            'pred': y_pred, 'true': y_true
        })
    else:
        print("Training failed.")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", type=str, default="./enthalpy/wudily_cho.csv")
    parser.add_argument("--target_csv", type=str, default="./enthalpy/atct.csv")
    parser.add_argument("--pretrained", type=str, default="") 
    
    parser.add_argument("--sampling_method", type=str, default="Full", 
                        choices=["Full", "Hard Random", "Herding", "K-Means", 
                                 "Entropy", "Least Confidence", "GraNd-20", "Glister", "Influence", 
                                 "EL2N-2", "DP", "CSIE",
                                 "Soft Random", "InfoBatch", "MolPeg", "epsilon-greedy", "UCB"],
                        help="Baseline selection method")
    parser.add_argument("--sampling_ratio", type=float, default=0.2)
    parser.add_argument("--update_freq", type=int, default=10) # 动态采样频率
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500) 
    parser.add_argument("--ft_epochs", type=int, default=1000) 
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4) # 保持和 DETS 一致
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=100)  
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="Enthalpy_checkpoints_baseline")
    parser.add_argument("--abs_tol", type=float, default=2.0) 
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)