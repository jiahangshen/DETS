import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import shutil
import random
import argparse 
import math

# ==========================================
# 0. 配置参数
# ==========================================
def get_config():
    parser = argparse.ArgumentParser()
    
    # 基础配置
    parser.add_argument("--theory_path", type=str, default="./bandgap/theory_features_clean.csv")
    parser.add_argument("--exp_path", type=str, default="./bandgap/exp_features_clean.csv")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_bandgap_fixed") 
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # DETS 参数
    parser.add_argument("--num_prototypes", type=int, default=10)
    parser.add_argument("--sigma_0", type=float, default=0.5)
    parser.add_argument("--gamma_temp", type=float, default=2.0) 
    parser.add_argument("--alpha_scale", type=float, default=1.0)
    parser.add_argument("--w_min", type=float, default=0.01)
    
    # 训练超参
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--epochs_p1", type=int, default=2000)
    
    # [新增] Phase 2 分两阶段的 Epoch 设置
    parser.add_argument("--epochs_p2_1", type=int, default=2000, help="Head alignment epochs")
    parser.add_argument("--epochs_p2_2", type=int, default=2000, help="Full fine-tuning epochs")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3) 
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # 早停参数
    parser.add_argument("--patience_p1", type=int, default=500)
    parser.add_argument("--patience_p2_1", type=int, default=200) # Head 阶段耐心
    parser.add_argument("--patience_p2_2", type=int, default=200) # Full 阶段耐心
    parser.add_argument("--min_lr", type=float, default=1e-7)
    
    # ICL Loss & Metric 参数
    parser.add_argument("--tol_percent", type=float, default=0.10) 
    parser.add_argument("--abs_tol", type=float, default=0.1) 
    parser.add_argument("--edl_lambda_max", type=float, default=0.01) 
    
    args = parser.parse_args()
    return vars(args)

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

def calculate_metrics(y_true, y_pred, abs_tol=0.1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    abs_diff = np.abs(y_true - y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    tol_threshold_10 = np.maximum(0.10 * np.abs(y_true), abs_tol)
    acc_10 = np.mean(abs_diff <= tol_threshold_10) * 100
    
    tol_threshold_20 = np.maximum(0.20 * np.abs(y_true), abs_tol)
    acc_20 = np.mean(abs_diff <= tol_threshold_20) * 100
    
    safe_y_true = np.maximum(np.abs(y_true), 1)
    mape = np.mean(abs_diff / safe_y_true) * 100
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Acc_10": acc_10, "Acc_20": acc_20}

class EarlyStopping:
    def __init__(self, patience=20, delta=0, path='checkpoint.pt', verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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
    print(f"-> Loading Data...")
    if not os.path.exists(CONFIG['theory_path']):
        print("Warning: CSV not found, generating dummy data.")
        X_theo = np.random.randn(1000, 100).astype(np.float32)
        y_theo = np.random.randn(1000).astype(np.float32)
        X_exp = np.random.randn(100, 100).astype(np.float32)
        y_exp = np.random.randn(100).astype(np.float32)
    else:
        df_theo = pd.read_csv(CONFIG['theory_path'])
        df_exp = pd.read_csv(CONFIG['exp_path'])
        feat_cols = [c for c in df_theo.columns if c not in ['target', 'id', 'formula']]
        X_theo = df_theo[feat_cols].values.astype(np.float32)
        y_theo = df_theo['target'].values.astype(np.float32)
        X_exp = df_exp[feat_cols].values.astype(np.float32)
        y_exp = df_exp['target'].values.astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X_exp, y_exp, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    scaler = StandardScaler()
    scaler.fit(np.concatenate([X_theo, X_train], axis=0))
    
    X_theo = scaler.transform(X_theo)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-6
    
    return (SimpleDataset(X_theo, y_theo), 
            SimpleDataset(X_train, y_train), 
            SimpleDataset(X_val, y_val), 
            SimpleDataset(X_test, y_test), 
            X_train.shape[1], 
            {'mean': y_mean, 'std': y_std})

# ==========================================
# 3. 模型
# ==========================================
class ResEvidentialMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.edl_head = nn.Linear(hidden_dim, 4)
        
    def forward(self, x):
        z = self.encoder(x)
        out = self.edl_head(z)
        gamma = out[:, 0:1] 
        v     = F.softplus(out[:, 1:2]) + 1e-6
        alpha = F.softplus(out[:, 2:3]) + 1.0 + 1e-6
        beta  = F.softplus(out[:, 3:4]) + 1e-6
        return torch.cat([gamma, v, alpha, beta], dim=1), z

# ==========================================
# 4. ICL Loss (Gauss-Legendre)
# ==========================================
class IntervalCensoredLoss(nn.Module):
    def __init__(self, tol_percent=0.10, abs_tol=0.1, lambda_reg_max=0.1):
        super().__init__()
        self.tol_percent = tol_percent
        self.abs_tol = abs_tol 
        self.lambda_reg_max = lambda_reg_max
        
        _nodes = [
            -0.9931285991850949, -0.9639719272779138, -0.9122344282513259, -0.8391169718222188, -0.7463319064601508,
            -0.6360536807265150, -0.5108670019508932, -0.3737060887154195, -0.2277858511416451, -0.0765265211334973,
             0.0765265211334973,  0.2277858511416451,  0.3737060887154195,  0.5108670019508932,  0.6360536807265150,
             0.7463319064601508,  0.8391169718222188,  0.9122344282513259,  0.9639719272779138,  0.9931285991850949
        ]
        _weights = [
            0.0176140071391521, 0.0406014298003869, 0.0626720483341091, 0.0832767415767048, 0.1019301198172404,
            0.1181945376687263, 0.1316886384491733, 0.1420961093183820, 0.1491729864726037, 0.1527533871307258,
            0.1527533871307258, 0.1491729864726037, 0.1420961093183820, 0.1316886384491733, 0.1181945376687263,
            0.1019301198172404, 0.0832767415767048, 0.0626720483341091, 0.0406014298003869, 0.0176140071391521
        ]
        
        self.register_buffer('nodes', torch.tensor(_nodes, dtype=torch.float32).view(1, -1))
        self.register_buffer('weights', torch.tensor(_weights, dtype=torch.float32).view(1, -1))

    def forward(self, outputs, target_norm, target_phys, y_std, epoch, total_epochs):
        gamma = outputs[:, 0:1]
        v     = outputs[:, 1:2]
        alpha = outputs[:, 2:3]
        beta  = outputs[:, 3:4]
        
        y = target_norm.view(-1, 1)
        y_phys = target_phys.view(-1, 1)
        
        rel_tol = torch.abs(y_phys) * self.tol_percent
        abs_tol_t = torch.tensor(self.abs_tol, device=y.device)
        delta_phys = torch.maximum(rel_tol, abs_tol_t)
        
        delta_norm = delta_phys / (y_std + 1e-8)
        
        df = 2 * alpha
        scale = torch.sqrt((beta * (1 + v)) / (v * alpha + 1e-8))
        
        lower = y - delta_norm
        upper = y + delta_norm
        center = (upper + lower) / 2.0
        half_width = (upper - lower) / 2.0
        
        if self.nodes.device != center.device:
            self.nodes = self.nodes.to(center.device)
            self.weights = self.weights.to(center.device)

        x_points = center + half_width * self.nodes
        
        df_exp = df.expand_as(x_points)
        scale_exp = scale.expand_as(x_points)
        gamma_exp = gamma.expand_as(x_points)
        
        z = (x_points - gamma_exp) / (scale_exp + 1e-8)
        
        log_prob = (torch.lgamma((df_exp + 1) / 2) 
                    - torch.lgamma(df_exp / 2) 
                    - 0.5 * torch.log(math.pi * df_exp) 
                    - torch.log(scale_exp) 
                    - (df_exp + 1) / 2 * torch.log(1 + z**2 / df_exp))
        
        pdf_vals = torch.exp(log_prob)
        weighted_sum = torch.sum(pdf_vals * self.weights, dim=1, keepdim=True)
        prob_mass = half_width * weighted_sum
        prob_mass = torch.clamp(prob_mass, min=1e-8, max=1.0-1e-8)
        nll = -torch.log(prob_mass)
        
        raw_error = torch.abs(y - gamma)
        rectified_error = F.softplus(raw_error - delta_norm)
        evidence = 2 * v + alpha
        reg_loss = rectified_error * evidence
        
        if total_epochs > 0:
            anneal = min(1.0, epoch / max(1, total_epochs // 2))
        else:
            anneal = 1.0
            
        total_loss = nll + (self.lambda_reg_max * anneal) * reg_loss
        return total_loss.mean()

# ==========================================
# 5. DETS Weight (Soft Tanimoto)
# ==========================================
def soft_tanimoto_sim(z1, z2):
    z1 = z1.unsqueeze(1); z2 = z2.unsqueeze(0)
    dot = (z1 * z2).sum(dim=2)
    norm1 = (z1**2).sum(dim=2); norm2 = (z2**2).sum(dim=2)
    denominator = norm1 + norm2 - dot
    score = dot / (denominator + 1e-8)
    return torch.clamp(score, 0.0, 1.0)

def compute_dets_weights(model, loader, prototypes, y_std_val, device):
    model.eval()
    all_z = []
    all_unc = []
    
    with torch.no_grad():
        for bx, _, _ in loader:
            bx = bx.to(device)
            out, z = model(bx)
            v = out[:, 1:2]
            alpha = out[:, 2:3]
            beta = out[:, 3:4]
            epistemic = beta / (v * (alpha - 1) + 1e-8)
            all_z.append(z)
            all_unc.append(epistemic)
            
    Z = torch.cat(all_z, dim=0)
    Unc = torch.cat(all_unc, dim=0).squeeze()
    
    sim_matrix = soft_tanimoto_sim(Z, prototypes)
    E_stat, _ = sim_matrix.max(dim=1)
    
    u_min = Unc.min()
    u_max = Unc.max()
    Delta_L = (Unc - u_min) / (u_max - u_min + 1e-8)
    
    sigma_eff = CONFIG['sigma_0'] * (1.0 + CONFIG['gamma_temp'] * Delta_L)
    energy = (1.0 - E_stat)**2
    temperature = 2 * sigma_eff**2
    weights = CONFIG['alpha_scale'] * torch.exp( - energy / (temperature + 1e-8) )
    weights = torch.clamp(weights, min=CONFIG['w_min'], max=1.0)
    return weights.cpu().numpy()

# ==========================================
# 6. Main Pipeline (With Two-Stage Fine-tuning)
# ==========================================
def main():
    set_seed(42)
    device = torch.device(CONFIG['device'])
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    ds_source, ds_train, ds_val, ds_test, dim, stats = prepare_data()
    y_std_val = stats['std']
    y_mean_val = stats['mean']
    
    train_loader = DataLoader(ds_train, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=256, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=256, shuffle=False)
    source_loader_eval = DataLoader(ds_source, batch_size=256, shuffle=False)
    
    model = ResEvidentialMLP(dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    criterion = IntervalCensoredLoss(tol_percent=CONFIG['tol_percent'], abs_tol=CONFIG['abs_tol'], lambda_reg_max=CONFIG['edl_lambda_max']).to(device)
    
    # === PHASE 0: WARM-UP ===
    print("\n>>> Phase 0: Warm-up on Target Data")
    for epoch in range(CONFIG['warmup_epochs']):
        model.train()
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out, _ = model(bx)
            y_phys = by
            y_norm = (by - y_mean_val) / y_std_val
            loss = criterion(out, y_norm, y_phys, y_std_val, epoch, CONFIG['warmup_epochs'])
            loss.backward()
            optimizer.step()
            
    print(">>> Extracting Prototypes...")
    model.eval()
    z_list = []
    with torch.no_grad():
        for bx, _, _ in train_loader:
            _, z = model(bx.to(device))
            z_list.append(z.cpu())
    Z_target = torch.cat(z_list, dim=0).numpy()
    kmeans = KMeans(n_clusters=CONFIG['num_prototypes'], random_state=42).fit(Z_target)
    prototypes = torch.tensor(kmeans.cluster_centers_, device=device, dtype=torch.float)
    
    # === PHASE 1: DETS Pre-training ===
    print(f"\n>>> Phase 1: DETS Pre-training")
    
    scheduler_p1 = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs_p1'], eta_min=CONFIG['min_lr'])
    stopper_p1 = EarlyStopping(patience=CONFIG['patience_p1'], path=os.path.join(CONFIG['save_dir'], 'best_p1.pt'))
    
    for epoch in range(1, CONFIG['epochs_p1'] + 1):
        weights = compute_dets_weights(model, source_loader_eval, prototypes, y_std_val, device)
        if epoch % 10==0:
            print(f"Weights Stats: Min={weights.min():.4f}, Max={weights.max():.4f}, Mean={weights.mean():.4f}")
        
        num_samples = int(len(ds_source) * 0.5)
        sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=False)
        p1_loader = DataLoader(ds_source, batch_size=CONFIG['batch_size'], sampler=sampler)
        
        model.train()
        for bx, by, _ in p1_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out, _ = model(bx)
            y_phys = by
            y_norm = (by - y_mean_val) / y_std_val
            loss = criterion(out, y_norm, y_phys, y_std_val, epoch, CONFIG['epochs_p1'])
            loss.backward()
            optimizer.step()
            
        scheduler_p1.step()
            
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for bx, by, _ in val_loader:
                bx, by = bx.to(device), by.to(device)
                out, _ = model(bx)
                gamma = out[:, 0:1]
                pred = gamma * y_std_val + y_mean_val
                val_mae += mean_absolute_error(by.cpu().numpy(), pred.cpu().numpy())
        val_mae /= len(val_loader)
        
        stopper_p1(val_mae, model)
        
        if epoch % 10 == 0:
            print(f"[P1] Ep {epoch} | Val MAE: {val_mae:.4f} | LR: {scheduler_p1.get_last_lr()[0]:.2e}")
            
        if stopper_p1.early_stop:
            print("Early stopping in Phase 1")
            break

    # === PHASE 2.1: Head Alignment (Freeze Encoder) ===
    print(f"\n>>> Phase 2.1: Head Alignment")
    
    # Reload best from Phase 1
    model.load_state_dict(torch.load(os.path.join(CONFIG['save_dir'], 'best_p1.pt')))
    
    # Freeze Encoder
    for name, param in model.named_parameters():
        if "edl_head" in name: param.requires_grad = True
        else: param.requires_grad = False
    
    # Optimizer for Head only
    optimizer_head = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'])
    stopper_p2_1 = EarlyStopping(patience=CONFIG['patience_p2_1'], path=os.path.join(CONFIG['save_dir'], 'best_head.pt'))
    
    for epoch in range(1, CONFIG['epochs_p2_1'] + 1):
        model.train()
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer_head.zero_grad()
            out, _ = model(bx)
            y_phys = by
            y_norm = (by - y_mean_val) / y_std_val
            loss = criterion(out, y_norm, y_phys, y_std_val, epoch, CONFIG['epochs_p2_1'])
            loss.backward()
            optimizer_head.step()
            
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for bx, by, _ in val_loader:
                bx, by = bx.to(device), by.to(device)
                out, _ = model(bx)
                gamma = out[:, 0:1]
                pred = gamma * y_std_val + y_mean_val
                val_mae += mean_absolute_error(by.cpu().numpy(), pred.cpu().numpy())
        val_mae /= len(val_loader)
        
        stopper_p2_1(val_mae, model)
        if epoch % 10 == 0:
            print(f"[P2.1] Ep {epoch} | Val MAE: {val_mae:.4f}")
        if stopper_p2_1.early_stop:
            print("Early stopping in Phase 2.1")
            break

    # === PHASE 2.2: Full Fine-tuning ===
    print(f"\n>>> Phase 2.2: Full Fine-tuning")
    
    # Reload best from 2.1
    model.load_state_dict(torch.load(os.path.join(CONFIG['save_dir'], 'best_head.pt')))
    
    # Unfreeze all
    for param in model.parameters(): param.requires_grad = True
        
    optimizer_full = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'] * 0.5)
    scheduler_full = CosineAnnealingLR(optimizer_full, T_max=CONFIG['epochs_p2_2'], eta_min=CONFIG['min_lr'])
    stopper_p2_2 = EarlyStopping(patience=CONFIG['patience_p2_2'], path=os.path.join(CONFIG['save_dir'], 'final_model.pt'))
    
    for epoch in range(1, CONFIG['epochs_p2_2'] + 1):
        model.train()
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer_full.zero_grad()
            out, _ = model(bx)
            y_phys = by
            y_norm = (by - y_mean_val) / y_std_val
            loss = criterion(out, y_norm, y_phys, y_std_val, epoch, CONFIG['epochs_p2_2'])
            loss.backward()
            optimizer_full.step()
            
        scheduler_full.step()
            
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for bx, by, _ in val_loader:
                bx, by = bx.to(device), by.to(device)
                out, _ = model(bx)
                gamma = out[:, 0:1]
                pred = gamma * y_std_val + y_mean_val
                val_mae += mean_absolute_error(by.cpu().numpy(), pred.cpu().numpy())
        val_mae /= len(val_loader)
        
        stopper_p2_2(val_mae, model)
        
        if epoch % 10 == 0:
            print(f"[P2.2] Ep {epoch} | Val MAE: {val_mae:.4f} | LR: {scheduler_full.get_last_lr()[0]:.2e}")
            
        if stopper_p2_2.early_stop:
            print("Early stopping in Phase 2.2")
            break

    # === FINAL TEST ===
    model.load_state_dict(torch.load(os.path.join(CONFIG['save_dir'], 'final_model.pt')))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by, _ in test_loader:
            bx, by = bx.to(device), by.to(device)
            out, _ = model(bx)
            gamma = out[:, 0:1]
            pred = gamma * y_std_val + y_mean_val
            preds.extend(pred.cpu().numpy().flatten())
            trues.extend(by.cpu().numpy().flatten())
            
    res = calculate_metrics(trues, preds, abs_tol=CONFIG['abs_tol'])
    print("\nFinal Results:")
    print(res)

if __name__ == "__main__":
    main()