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
from tqdm import tqdm  # 引入 tqdm

# --- 屏蔽警告 ---
from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore")
# ------------------

from dataset import SolvationDataset

# ==========================================
# PART 1: Baseline 模型
# ==========================================
class BaselineGNNRegressor(nn.Module):
    def __init__(self, num_node_features, num_edge_features,
                 hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.dropout = dropout
        
        self.gnn_solvent = self._build_gnn_backbone(num_node_features, num_edge_features, hidden_dim, num_layers)
        self.gnn_solute = self._build_gnn_backbone(num_node_features, num_edge_features, hidden_dim, num_layers)

        # 投影头保留但不使用，保持参数一致性
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 128), 
            nn.Sigmoid()
        )
        
        input_dim_final = hidden_dim * 2 + 4 
        
        self.fc_final = nn.Sequential(
            nn.Linear(input_dim_final, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出 1 维预测值
        self.output_head = nn.Linear(hidden_dim, 1) 

    def _build_gnn_backbone(self, num_node_features, num_edge_features, hidden_dim, num_layers):
        convs = nn.ModuleList()
        bns = nn.ModuleList()
        for i in range(num_layers):
            in_dim = num_node_features if i == 0 else hidden_dim
            nn_lin = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINEConv(nn_lin, edge_dim=num_edge_features)
            convs.append(conv)
            bns.append(nn.BatchNorm1d(hidden_dim))
        return nn.ModuleDict({'convs': convs, 'bns': bns})

    def forward(self, data):
        # 溶剂 GNN
        x_solvent = data.x_solvent
        batch_solvent = data.x_solvent_batch 
        h_solvent = self._forward_gnn(self.gnn_solvent, x_solvent, data.edge_index_solvent, data.edge_attr_solvent)
        # 显式传入 size 优化 compile 性能
        g_solvent = global_add_pool(h_solvent, batch_solvent, size=data.num_graphs)
        
        # 溶质 GNN
        x_solute = data.x_solute
        batch_solute = data.x_solute_batch 
        h_solute = self._forward_gnn(self.gnn_solute, x_solute, data.edge_index_solute, data.edge_attr_solute)
        g_solute = global_add_pool(h_solute, batch_solute, size=data.num_graphs)

        # 特征拼接
        phys_feat = data.global_feat.view(g_solvent.shape[0], -1) 
        g_concat = torch.cat([g_solvent, g_solute, phys_feat], dim=1) 
        
        # 预测
        g_final = self.fc_final(g_concat)
        prediction = self.output_head(g_final) 
        
        return prediction

    def _forward_gnn(self, gnn_module, x, edge_index, edge_attr):
        for conv, bn in zip(gnn_module['convs'], gnn_module['bns']):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# ==========================================
# PART 2: 辅助函数 (评估)
# ==========================================

def evaluate(model, loader, device, y_mean, y_std, return_preds=False):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred_norm = out.view(-1)
            pred_val = pred_norm * y_std + y_mean
            
            y = batch.y.view(-1).cpu().numpy()
            preds.extend(pred_val.cpu().numpy())
            ys.extend(y)
    
    y_true = np.array(ys)
    y_pred = np.array(preds)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    relative_errors = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true) , 1))
    mape = np.mean(relative_errors) * 100 
    
    hits_10 = (relative_errors <= 0.10)
    acc_10 = np.mean(hits_10) * 100 
    hits_20 = (relative_errors <= 0.20)
    acc_20 = np.mean(hits_20) * 100
    
    if return_preds:
        return mae, rmse, mape, acc_10, acc_20, y_pred, y_true
    return mae, rmse, mape, acc_10, acc_20

# ==========================================
# PART 3: 主训练逻辑 (加速版)
# ==========================================

def main(args):
    print(f"--- Baseline: Accelerated Training (Memory Dataset + AMP + Compile) ---")
    print(f"Device: {args.device}")

    # [优化] Worker 设置
    num_workers = 8
    print(f"Using num_workers: {num_workers}")

    # 1. 数据加载 (使用缓存)
    # 注意：root 参数必须指定，以便复用 dataset.py 中生成的 .pt 文件
    print("Loading Target Dataset...")
    target_full = SolvationDataset(root='./solvation_cache', csv_file=args.target_csv)
    
    print("Loading Source Dataset (Fast Load)...")
    source_dataset = SolvationDataset(root='./solvation_cache', csv_file=args.source_csv)
    
    # 划分 Target 数据集
    idx_target = list(range(len(target_full)))
    tr_idx, test_idx = train_test_split(idx_target, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(tr_idx, test_size=(0.1/0.9), random_state=42)
    
    from torch.utils.data import Subset
    target_train = Subset(target_full, train_idx)
    target_val = Subset(target_full, val_idx)
    target_test = Subset(target_full, test_idx)
    
    # 计算统计量
    y_target_raw = torch.tensor([target_full.targets[i] for i in train_idx], dtype=torch.float)
    y_mean = y_target_raw.mean().to(args.device)
    y_std = y_target_raw.std().to(args.device)
    print(f"Target Stats: Mean {y_mean.item():.2f}, Std {y_std.item():.2f}")
    
    # 初始化模型
    sample = target_full[0]
    num_node_feat = sample.x_solvent.shape[1]
    num_edge_feat = sample.edge_attr_solvent.shape[1]
    
    model = BaselineGNNRegressor(
        num_node_features=num_node_feat,
        num_edge_features=num_edge_feat,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(args.device)
    
    # [优化] 使用 torch.compile 加速
    if hasattr(torch, 'compile'):
        print("-> Using torch.compile for acceleration...")
        model = torch.compile(model)

    # ------------------------------------
    # Stage 1: Source Pre-training
    # ------------------------------------
    print("\n=== Stage 1: Pre-training on Source Data ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.MSELoss()
    
    # [优化] 增大 Source Batch Size, 开启 Persistent Workers
    source_loader = DataLoader(source_dataset, batch_size=args.source_batch_size, shuffle=True, 
                               follow_batch=['x_solvent', 'x_solute'],
                               num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    target_val_loader = DataLoader(target_val, batch_size=args.batch_size, shuffle=False, 
                                   follow_batch=['x_solvent', 'x_solute'],
                                   num_workers=num_workers, pin_memory=True)

    best_pre_mae = float('inf')
    scaler = torch.cuda.amp.GradScaler() # [优化] 混合精度 Scaler
    
    for epoch in range(1, args.pre_epochs + 1):
        model.train()
        total_loss = 0
        
        # [优化] 使用 tqdm 显示训练进度
        pbar = tqdm(source_loader, desc=f"Pretrain Ep {epoch}", leave=False, ncols=100)
        
        for batch in pbar:
            batch = batch.to(args.device, non_blocking=True) # 异步传输
            optimizer.zero_grad()
            
            # [优化] 混合精度训练 (AMP)
            with torch.cuda.amp.autocast():
                pred = model(batch).view(-1)
                y = batch.y.view(-1).to(args.device)
                y_norm = (y - y_mean) / y_std 
                loss = criterion(pred, y_norm)
            
            # Scaler Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_val = loss.item()
            total_loss += loss_val
            pbar.set_postfix({'loss': f"{loss_val:.4f}"})
            
        val_mae, val_rmse, val_mape, val_acc_10, val_acc_20 = evaluate(model, target_val_loader, args.device, y_mean, y_std)
        
        if epoch % 1 == 0:
            print(f"[Pre-train] Ep {epoch} | Loss {total_loss/len(source_loader):.4f} | "
                  f"Val MAE {val_mae:.4f} | RMSE {val_rmse:.4f} | "
                  f"MAPE {val_mape:.2f}% | Acc@10 {val_acc_10:.1f}% | Acc@20 {val_acc_20:.1f}%")
            
        if val_mae < best_pre_mae:
            best_pre_mae = val_mae
            torch.save(model.state_dict(), "baseline_pretrain.pt")

    # ------------------------------------
    # Stage 2: Target Fine-tuning
    # ------------------------------------
    print("\n=== Stage 2: Fine-tuning on Target Data ===")
    if os.path.exists("baseline_pretrain.pt") and args.pre_epochs > 0:
        print("-> Loading pretrained model...")
        model.load_state_dict(torch.load("baseline_pretrain.pt"))
    else:
        print("-> Training from scratch (No pre-training)...")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.5, weight_decay=args.wd)
    
    # Target 数据量小，使用常规 Batch Size
    target_train_loader = DataLoader(target_train, batch_size=args.batch_size, shuffle=True, 
                                     follow_batch=['x_solvent', 'x_solute'],
                                     num_workers=num_workers, pin_memory=True)
    target_test_loader = DataLoader(target_test, batch_size=args.batch_size, shuffle=False, 
                                    follow_batch=['x_solvent', 'x_solute'],
                                    num_workers=num_workers, pin_memory=True)
    
    best_ft_mae = float('inf')
    patience = 0
    
    for epoch in range(1, args.ft_epochs + 1):
        model.train()
        total_loss = 0
        
        # Target 数据少，这里不一定要用 tqdm，或者用 epoch 级的 tqdm
        for batch in target_train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()
            
            # Target 微调通常不需要 AMP，数据少且为了更高精度
            pred = model(batch).view(-1)
            y = batch.y.view(-1).to(args.device)
            y_norm = (y - y_mean) / y_std 
            
            loss = criterion(pred, y_norm)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        val_mae, val_rmse, val_mape, val_acc_10, val_acc_20 = evaluate(model, target_val_loader, args.device, y_mean, y_std)
        
        if epoch % 1 == 0:
            # [修正] 修复了原代码中 Loss 分母错误和文案错误
            print(f"[Fine-tune] Ep {epoch} | Loss {total_loss/len(target_train_loader):.4f} | "
                  f"Val MAE {val_mae:.4f} | RMSE {val_rmse:.4f} | "
                  f"MAPE {val_mape:.2f}% | Acc@10 {val_acc_10:.1f}% | Acc@20 {val_acc_20:.1f}%")
            
        if val_mae < best_ft_mae:
            best_ft_mae = val_mae
            torch.save(model.state_dict(), "baseline_final.pt")
            patience = 0
        else:
            patience += 1
            
        if patience >= args.patience:
            print("Early stopping triggered.")
            break

    # ------------------------------------
    # Final Testing
    # ------------------------------------
    print("\n>>> Testing Final Baseline Model...")
    model.load_state_dict(torch.load("baseline_final.pt"))
    
    test_mae, test_rmse, test_mape, test_acc_10, test_acc_20, _, _ = evaluate(
        model, target_test_loader, args.device, y_mean, y_std, return_preds=True
    )
    
    print("-" * 60)
    print(f"[BASELINE RESULT] Solvation Free Energy:")
    print(f"   MAE:      {test_mae:.4f} kcal/mol")
    print(f"   RMSE:     {test_rmse:.4f} kcal/mol")
    print(f"   MAPE:     {test_mape:.2f} %")
    print(f"   Acc@10%:  {test_acc_10:.2f} %")
    print(f"   Acc@20%:  {test_acc_20:.2f} %")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_csv", type=str, default="./solute-solvent/exp_data.csv")
    parser.add_argument("--source_csv", type=str, default="./solute-solvent/qm.csv")
    
    # [新增] source_batch_size 参数，默认 1024
    parser.add_argument("--source_batch_size", type=int, default=4096, help="Batch size for source pre-training")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for target fine-tuning/val/test")
    
    parser.add_argument("--pre_epochs", type=int, default=200, help="Epochs for source pre-training")                                                      
    parser.add_argument("--ft_epochs", type=int, default=200, help="Epochs for target fine-tuning")
    
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