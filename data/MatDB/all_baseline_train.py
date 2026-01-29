import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import shutil
import random

# [新增] 引入适配 MLP 的采样器
# 确保 sampler_mlp.py 在同一目录下
from sampler import CoresetSamplerMLP
class SilentTqdm:
    def __init__(self, iterable, *args, **kwargs): self.iterable = iterable
    def __iter__(self): return iter(self.iterable)
    def set_postfix(self, *args, **kwargs): pass
    def update(self, *args, **kwargs): pass
tqdm = SilentTqdm
# ==========================================
# 1. 辅助工具
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    def __init__(self, patience=50, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

def calculate_metrics(y_true, y_pred, abs_tol=0.1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    abs_diff = np.abs(y_true - y_pred)
    
    # [优化] 混合容忍度 Acc
    # 误差 < 10% 或 误差 < 0.1 (对于BandGap)
    tol_threshold = np.maximum(0.10 * np.abs(y_true), abs_tol)
    acc_10 = np.mean(abs_diff <= tol_threshold) * 100
    
    tol_threshold_20 = np.maximum(0.20 * np.abs(y_true), abs_tol)
    acc_20 = np.mean(abs_diff <= tol_threshold_20) * 100
    
    # [优化] MAPE 分母保护
    safe_y_true = np.maximum(np.abs(y_true), 1e-6)
    mape = np.mean(abs_diff / safe_y_true) * 100
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Acc_10": acc_10, "Acc_20": acc_20}

# ==========================================
# 2. 数据处理
# ==========================================
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).view(-1, 1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], idx 

def prepare_data(theory_path, exp_path):
    print(f"Loading data from:\n  Theory: {theory_path}\n  Exp: {exp_path}")
    try:
        df_theo = pd.read_csv(theory_path)
        df_exp = pd.read_csv(exp_path)
    except FileNotFoundError:
        print("Error: CSV files not found.")
        raise

    meta_cols = ['formula', 'target', 'SMILES', 'composition', 'id'] 
    feat_cols = [c for c in df_theo.columns if c not in meta_cols and c in df_exp.columns]
    
    for df in [df_theo, df_exp]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
    X_theo_raw = df_theo[feat_cols].values.astype(np.float32)
    y_theo = df_theo['target'].values.astype(np.float32)
    
    X_exp_raw = df_exp[feat_cols].values.astype(np.float32)
    y_exp = df_exp['target'].values.astype(np.float32)
    
    indices = np.arange(len(X_exp_raw))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)
    
    scaler = StandardScaler()
    X_fit = np.concatenate([X_theo_raw, X_exp_raw[train_idx]], axis=0)
    scaler.fit(X_fit)
    
    X_theo = scaler.transform(X_theo_raw)
    X_exp = scaler.transform(X_exp_raw)
    
    ds_source = SimpleDataset(X_theo, y_theo)
    ds_train = SimpleDataset(X_exp[train_idx], y_exp[train_idx])
    ds_val = SimpleDataset(X_exp[val_idx], y_exp[val_idx])
    ds_test = SimpleDataset(X_exp[test_idx], y_exp[test_idx])
    
    return ds_source, ds_train, ds_val, ds_test, len(feat_cols)

# ==========================================
# 3. 升级版模型 (ResNet MLP)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class BaselineResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_blocks=3, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # 默认 forward 只返回预测值
        out, _ = self.forward_with_features(x)
        return out
        
    # [新增] 供 Sampler 使用，返回 (Prediction, Embedding)
    def forward_with_features(self, x):
        feat = self.input_proj(x)
        for block in self.blocks:
            feat = block(feat)
        # feat 是倒数第二层特征
        out = self.head(feat)
        return out, feat

# ==========================================
# 4. 主程序
# ==========================================
def main(args):
    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    
    print(f"--- Baseline MLP Pipeline (Method: {args.sampling_method}) ---")
    
    # 1. Load Data
    ds_source, ds_train, ds_val, ds_test, input_dim = prepare_data(args.theory_path, args.exp_path)
    print(f"Full Source Data: {len(ds_source)} | Target Train: {len(ds_train)}")
    
    # 2. Model Init
    print(f"Initializing ResNet Baseline (Dim={input_dim})...")
    model = BaselineResMLP(input_dim=input_dim, dropout=args.dropout).to(device)
    
    # Init Weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

    # ==========================================
    # Stage 0: Coreset Selection
    # ==========================================
    if args.sampling_method != 'Full':
        print(f"\n=== Stage 0: Coreset Selection [{args.sampling_method}] ===")
        print(f"   Target Ratio: {args.sampling_ratio * 100:.1f}%")
        
        # 初始化采样器 (MLP版)
        # 注意：这里我们用 ds_train 作为 target_dataset，用于跨域方法 (Glister/Influence)
        sampler = CoresetSamplerMLP(model, ds_source, ds_train, device)
        
        # 执行采样
        indices = sampler.select(args.sampling_method, args.sampling_ratio)
        
        # 创建 Subset
        train_source_dataset = Subset(ds_source, indices)
        print(f"-> Selected {len(train_source_dataset)} samples.")
    else:
        print("\n=== Stage 0: Using Full Dataset ===")
        train_source_dataset = ds_source

    # Loaders
    source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    # --- PHASE 1: Source Pre-training ---
    print("\n=== PHASE 1: Source Pre-training ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    p1_stopper = EarlyStopping(patience=50, path=os.path.join(args.save_dir, 'best_p1.pt'))
    criterion = nn.MSELoss()
    
    for epoch in range(1, args.epochs_p1 + 1):
        model.train()
        train_loss = 0
        for bx, by, _ in source_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(source_loader)
        
        # Val (Optional for P1, usually check loss)
        if epoch % 20 == 0:
            print(f"[P1] Ep {epoch}: Loss {train_loss:.4f}")
        
        # 保存最新的
        p1_stopper(train_loss, model) # 使用 Loss 早停，因为是 Pre-train
        if p1_stopper.early_stop:
            print("Phase 1 Early Stopping")
            break

    # --- PHASE 2.1: Head Alignment ---
    print("\n=== PHASE 2.1: Head Alignment ===")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_p1.pt')))
    
    # Freeze Body
    for name, param in model.named_parameters():
        if "head" in name: param.requires_grad = True
        else: param.requires_grad = False
            
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    p21_stopper = EarlyStopping(patience=50, path=os.path.join(args.save_dir, 'best_head.pt'))
    
    for epoch in range(1, args.epochs_p2_1 + 1):
        model.train()
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for bx, by, _ in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_mae += mean_absolute_error(by.cpu().numpy(), pred.cpu().numpy())
        val_mae /= len(val_loader)
        
        if epoch % 50 == 0: print(f"[P2.1] Ep {epoch} MAE {val_mae:.4f}")
        p21_stopper(val_mae, model)
        if p21_stopper.early_stop: break

    # --- PHASE 2.2: Full Fine-tuning ---
    print("\n=== PHASE 2.2: Full Fine-tuning ===")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_head.pt')))
    for p in model.parameters(): p.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
    p22_stopper = EarlyStopping(patience=50, path=os.path.join(args.save_dir, 'final_model.pt'))
    
    for epoch in range(1, args.epochs_p2_2 + 1):
        model.train()
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for bx, by, _ in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_preds.extend(pred.cpu().numpy())
                val_trues.extend(by.cpu().numpy())
        
        met = calculate_metrics(val_trues, val_preds)
        if epoch % 20 == 0: print(f"[P2.2] Ep {epoch} MAE {met['MAE']:.4f}")
        p22_stopper(met['MAE'], model)
        if p22_stopper.early_stop: break

    # --- FINAL TEST ---
    print("\n>>> Final Test Evaluation")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'final_model.pt')))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by, _ in test_loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            preds.extend(pred.cpu().numpy())
            trues.extend(by.cpu().numpy())
            
    # Bandgap 绝对容忍度建议 0.1 eV
    metrics = calculate_metrics(trues, preds, abs_tol=0.1)
    print("-" * 60)
    print(f"Method: {args.sampling_method} (Ratio={args.sampling_ratio})")
    for k, v in metrics.items(): print(f"{k}: {v:.4f}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径
    parser.add_argument("--theory_path", type=str, default="bandgap/theory_features_clean.csv")
    parser.add_argument("--exp_path", type=str, default="bandgap/exp_features_clean.csv")
    
    # 采样参数
    parser.add_argument("--sampling_method", type=str, default="InfoBatch", 
                        choices=[
                            "Full", 
                            "Hard Random", "Herding", "K-Means", 
                            "Entropy", "Least Confidence", "GraNd-20", "Glister", "Influence", 
                            "EL2N-2", "DP", "CSIE",
                            "Soft Random", "InfoBatch", "MolPeg", "epsilon-greedy", "UCB"
                        ],
                        help="Baseline selection method")
    parser.add_argument("--sampling_ratio", type=float, default=0.2)
    
    # 训练超参 (默认值保持与 CONFIG 一致)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs_p1", type=int, default=1000)
    parser.add_argument("--epochs_p2_1", type=int, default=5000)
    parser.add_argument("--epochs_p2_2", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default='./checkpoints_baseline/MAT/')
    parser.add_argument("--device", type=str, default='cuda:0')
    
    args = parser.parse_args()
    main(args)