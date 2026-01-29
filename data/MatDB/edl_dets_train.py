import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import shutil
import random
import argparse 

# ==========================================
# 0. 配置参数 & 命令行解析
# ==========================================
def get_config():
    parser = argparse.ArgumentParser()
    
    # 路径 & 基础
    parser.add_argument("--theory_path", type=str, default="../bandgap/theory_features_clean.csv")
    parser.add_argument("--exp_path", type=str, default="../bandgap/exp_features_clean.csv")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_bandgap_small_res") 
    parser.add_argument("--device", type=str, default="cuda")
    
    # DETS 核心参数
    parser.add_argument("--num_prototypes", type=int, default=5)
    parser.add_argument("--tau_high", type=float, default=0.90)
    parser.add_argument("--sigma_0", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--w_min", type=float, default=0.1)
    
    # 训练超参
    parser.add_argument("--epochs_p1", type=int, default=1000)
    parser.add_argument("--epochs_p2_1", type=int, default=2000)
    parser.add_argument("--epochs_p2_2", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    args = parser.parse_args()
    config = vars(args)
    return config

# 获取配置
CONFIG = get_config()


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
    def __init__(self, patience=50, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
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

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    safe_y_true = np.where(np.abs(y_true) < 1e-6, 1e-6, y_true)
    rel_err = np.abs((y_true - y_pred) / safe_y_true)
    
    mape = np.mean(rel_err) * 100
    acc_10 = np.mean(rel_err <= 0.10) * 100
    acc_20 = np.mean(rel_err <= 0.20) * 100
    
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

def prepare_data():
    print(f"-> Loading CSVs...")
    try:
        df_theo = pd.read_csv(CONFIG['theory_path'])
        df_exp = pd.read_csv(CONFIG['exp_path'])
    except FileNotFoundError:
        print("Error: CSV files not found.")
        raise

    meta_cols = ['formula', 'target', 'SMILES', 'composition', 'id'] 
    feat_cols = [c for c in df_theo.columns if c not in meta_cols and c in df_exp.columns]
    
    print(f"-> Selected {len(feat_cols)} features.")
    
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
    
    y_target_train = y_exp[train_idx]
    target_stats = {
        'mean': np.mean(y_target_train),
        'std': np.std(y_target_train)
    }
    print(f"-> Target Stats: Mean={target_stats['mean']:.4f}, Std={target_stats['std']:.4f}")
    
    return ds_source, ds_train, ds_val, ds_test, len(feat_cols), target_stats

# ==========================================
# 3. 模型 (ResNet 架构)
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

class ResEvidentialMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_blocks=3, dropout=0.1):
        super(ResEvidentialMLP, self).__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        # Head A: EDL Regressor
        self.edl_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4) 
        )
        
        # Head B: DETS Projector
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64) 
        )

    def forward(self, x):
        feat = self.input_proj(x)
        for block in self.blocks:
            feat = block(feat)
            
        z = torch.sigmoid(self.proj_head(feat))
        out = self.edl_head(feat)
        
        gamma = out[:, 0].view(-1, 1) 
        v     = F.softplus(out[:, 1].view(-1, 1)) + 1e-6
        alpha = F.softplus(out[:, 2].view(-1, 1)) + 1.0 + 1e-6
        beta  = F.softplus(out[:, 3].view(-1, 1)) + 1e-6
        
        return gamma, v, alpha, beta, z

def edl_loss(y, gamma, v, alpha, beta, epoch_ratio, lambda_coef=0.01):
    two_beta_lambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
        - alpha * torch.log(two_beta_lambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_beta_lambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    
    error = torch.abs(y - gamma)
    reg = error * (2 * v + alpha)
    
    annealing = min(1.0, epoch_ratio)
    return nll + lambda_coef * annealing * reg

# ==========================================
# 4. DETS 核心逻辑
# ==========================================
def get_prototypes(model, target_loader, device, num_prototypes):
    print("-> Extracting Prototypes...")
    model.eval()
    all_z, all_y = [], []
    with torch.no_grad():
        for bx, by, _ in target_loader:
            bx = bx.to(device)
            _, _, _, _, z = model(bx)
            all_z.append(z.cpu())
            all_y.append(by)
            
    Z = torch.cat(all_z, dim=0).numpy()
    Y = torch.cat(all_y, dim=0).numpy().flatten()
    
    if len(Z) < num_prototypes: num_prototypes = max(1, len(Z))
    
    kmeans = KMeans(n_clusters=num_prototypes, n_init=10, random_state=42).fit(Z)
    
    proto_indices = []
    for k in range(num_prototypes):
        mask = (kmeans.labels_ == k)
        points = Z[mask]
        if len(points) > 0:
            center = kmeans.cluster_centers_[k].reshape(1, -1)
            dists = np.linalg.norm(points - center, axis=1)
            local_idx = np.argmin(dists)
            global_idx = np.where(mask)[0][local_idx]
            proto_indices.append(global_idx)
            
    prototypes = torch.tensor(Z[proto_indices], dtype=torch.float).to(device)
    
    y_cons = []
    for k in range(num_prototypes):
        mask = (kmeans.labels_ == k)
        if np.any(mask): y_cons.append(np.mean(Y[mask]))
        else: y_cons.append(0.0)
    y_cons = torch.tensor(y_cons, dtype=torch.float).to(device)
            
    return prototypes, y_cons

def soft_tanimoto_sim(z1, z2):
    z1 = z1.unsqueeze(1); z2 = z2.unsqueeze(0)
    dot = (z1 * z2).sum(dim=2)
    norm1 = (z1**2).sum(dim=2); norm2 = (z2**2).sum(dim=2)
    return dot / (norm1 + norm2 - dot + 1e-8)

# ======================================================================
# [关键修改] calculate_weights: 替换为基于 Epistemic Uncertainty 的逻辑
# ======================================================================
def calculate_weights(model, source_loader, prototypes, y_cons, target_stats, device):
    """
    修改说明：
    1. 移除 'raw_diff = gamma - y_c' (数值偏差) 逻辑。
    2. 新增 Epistemic Uncertainty 计算: beta / (nu * (alpha-1))。
    3. 新增 不确定性归一化 -> Delta_L。
    4. 保留物理范围过滤 (Hard Cutoff) 和 Soft Floor。
    """
    model.eval()
    
    all_E_stat = []
    all_unc = []
    all_labels = []
    
    min_accept = target_stats['mean'] - 3.0 * target_stats['std']
    max_accept = target_stats['mean'] + 3.0 * target_stats['std']
    
    # 1. 收集所有样本的 E_stat 和 Uncertainty
    with torch.no_grad():
        for bx, by, _ in source_loader:
            bx = bx.to(device); by = by.to(device)
            
            # 解包 EDL 参数
            gamma, v, alpha, beta, z = model(bx)
            
            # 计算认知不确定性 (Epistemic Uncertainty)
            # Var[mu] = beta / (v * (alpha - 1))
            unc = beta / (v * (alpha - 1) + 1e-8)
            
            # 计算相似度
            sim_matrix = soft_tanimoto_sim(z, prototypes)
            E_stat, _ = sim_matrix.max(dim=1)
            
            all_unc.append(unc.squeeze())
            all_E_stat.append(E_stat)
            all_labels.append(by.squeeze())
            
    # 拼接
    cat_unc = torch.cat(all_unc)
    cat_E_stat = torch.cat(all_E_stat)
    cat_labels = torch.cat(all_labels)
    
    # 2. 全局归一化不确定性 -> Delta_L
    # 这样 Delta_L 就在 [0, 1] 之间，可以作为 sigma_eff 的开关
    u_min = cat_unc.min()
    u_max = cat_unc.max()
    Delta_L = (cat_unc - u_min) / (u_max - u_min + 1e-8)
    
    # 3. 计算权重
    # 不确定性高 -> Delta_L 高 -> Sigma变大 -> 筛选变宽 (Exploration)
    sigma_eff = CONFIG['sigma_0'] * (1.0 + CONFIG['gamma'] * Delta_L)
    D_struct = 1.0 - cat_E_stat
    w_raw = CONFIG['alpha'] * torch.exp( - (D_struct**2) / (2 * sigma_eff**2 + 1e-8) )
    
    # 4. 物理范围过滤 (Hard Cutoff)
    range_mask = (cat_labels >= min_accept) & (cat_labels <= max_accept)
    w_final = w_raw * range_mask.float()
    
    # 5. 归一化权重到 [0, 1]
    if w_final.max() > 0:
        w_final /= w_final.max()
        
    # 6. Soft Floor (保底权重，防止除零或完全不采样)
    w_final = 0.9 * w_final + CONFIG['w_min']
    
    return w_final.cpu()

# ==========================================
# 5. 主程序
# ==========================================
def main():
    set_seed(42)
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device(CONFIG['device'])
    
    # 1. Load Data
    ds_source, ds_train, ds_val, ds_test, input_dim, target_stats = prepare_data()
    print(f"Source Data: {len(ds_source)} | Target Train: {len(ds_train)}")
    
    source_loader_full = DataLoader(ds_source, batch_size=256, shuffle=False)
    train_loader = DataLoader(ds_train, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 2. Model Init
    print(f"Initializing ResNet DETS Model (Dim={input_dim})...")
    model = ResEvidentialMLP(input_dim=input_dim, hidden_dim=256, num_blocks=3, dropout=CONFIG['dropout']).to(device)
    
    print("-> Applying Sparsity Init...")
    proj_last = model.proj_head[-1]
    nn.init.normal_(proj_last.weight, mean=0.0, std=3.0)
    nn.init.constant_(proj_last.bias, -1.0)
    
    edl_last = model.edl_head[-1]
    nn.init.xavier_uniform_(edl_last.weight)

    # --- PHASE 1 ---
    print("\n=== PHASE 1: Source Pre-training (Soft DETS) ===")
    prototypes, y_cons = get_prototypes(model, train_loader, device, CONFIG['num_prototypes'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    for epoch in range(1, CONFIG['epochs_p1'] + 1):
        # [关键] 调用修改后的权重计算函数 (基于不确定性)
        weights = calculate_weights(model, source_loader_full, prototypes, y_cons, target_stats, device)
        
        n_active = len(ds_source)
        sampler = WeightedRandomSampler(weights, num_samples=n_active, replacement=True)
        source_loader = DataLoader(ds_source, batch_size=CONFIG['batch_size'], sampler=sampler)
        
        model.train()
        train_loss = 0
        for bx, by, _ in source_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            gamma, v, a, b, z = model(bx)
            
            with torch.no_grad():
                sim_matrix = soft_tanimoto_sim(z, prototypes)
                E_stat, _ = sim_matrix.max(dim=1)
                batch_w = E_stat 
            
            loss_vec = edl_loss(by, gamma, v, a, b, epoch/CONFIG['epochs_p1'])
            loss = torch.mean(loss_vec.squeeze() * batch_w) 
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(source_loader)
        
        if epoch % 20 == 0:
            model.eval()
            val_mae = 0
            with torch.no_grad():
                for bx, by, _ in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    gamma, _, _, _, _ = model(bx)
                    val_mae += mean_absolute_error(by.cpu().numpy(), gamma.cpu().numpy())
            val_mae /= len(val_loader)
            print(f"[P1] Ep {epoch}: Loss {train_loss:.4f} | Val MAE {val_mae:.4f}")
        
        torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'best_p1.pt'))

    # --- PHASE 2.1 ---
    print("\n=== PHASE 2.1: Head Alignment ===")
    model.load_state_dict(torch.load(os.path.join(CONFIG['save_dir'], 'best_p1.pt')))
    
    # Freeze Body
    for name, param in model.named_parameters():
        if "edl_head" in name: param.requires_grad = True
        else: param.requires_grad = False
            
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'])
    p21_stopper = EarlyStopping(patience=150, verbose=False, path=os.path.join(CONFIG['save_dir'], 'best_head.pt'))
    
    for epoch in range(1, CONFIG['epochs_p2_1'] + 1):
        model.train()
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            gamma, v, a, b, _ = model(bx)
            loss = edl_loss(by, gamma, v, a, b, 1.0).mean()
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for bx, by, _ in val_loader:
                bx, by = bx.to(device), by.to(device)
                gamma, _, _, _, _ = model(bx)
                val_mae += mean_absolute_error(by.cpu().numpy(), gamma.cpu().numpy())
        val_mae /= len(val_loader)
        
        if epoch % 20 == 0: print(f"[P2.1] Ep {epoch} MAE {val_mae:.4f}")
        p21_stopper(val_mae, model)
        if p21_stopper.early_stop: break

    # --- PHASE 2.2 ---
    print("\n=== PHASE 2.2: Full Fine-tuning ===")
    model.load_state_dict(torch.load(os.path.join(CONFIG['save_dir'], 'best_head.pt')))
    for p in model.parameters(): p.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'] * 0.1)
    p22_stopper = EarlyStopping(patience=50, verbose=False, path=os.path.join(CONFIG['save_dir'], 'final_model.pt'))
    
    for epoch in range(1, CONFIG['epochs_p2_2'] + 1):
        model.train()
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            gamma, v, a, b, _ = model(bx)
            loss = edl_loss(by, gamma, v, a, b, 1.0).mean()
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for bx, by, _ in val_loader:
                bx, by = bx.to(device), by.to(device)
                gamma, _, _, _, _ = model(bx)
                val_preds.extend(gamma.cpu().numpy())
                val_trues.extend(by.cpu().numpy())
        
        met = calculate_metrics(val_trues, val_preds)
        if epoch % 20 == 0: print(f"[P2.2] Ep {epoch} MAE {met['MAE']:.4f}")
        p22_stopper(met['MAE'], model)
        if p22_stopper.early_stop: break

    # --- FINAL TEST ---
    print("\n>>> Final Test Evaluation")
    model.load_state_dict(torch.load(os.path.join(CONFIG['save_dir'], 'final_model.pt')))
    model.eval()
    preds, trues, uncs = [], [], []
    with torch.no_grad():
        for bx, by, _ in test_loader:
            bx, by = bx.to(device), by.to(device)
            g, v, a, b, _ = model(bx)
            preds.extend(g.cpu().numpy())
            trues.extend(by.cpu().numpy())
            unc = b / (v * (a - 1)); uncs.extend(unc.cpu().numpy())
            
    metrics = calculate_metrics(trues, preds)
    print("-" * 60)
    for k, v in metrics.items(): print(f"{k}: {v:.4f}")
    print("-" * 60)

if __name__ == "__main__":
    CONFIG = get_config()
    main()