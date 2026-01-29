import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from ogb.utils import smiles2graph
from rdkit import RDLogger
import os

RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. 核心类定义 (保持与 DETS 一致)
# ==========================================

class ATCTDataset(Dataset):
    def __init__(self, csv_file, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.df = pd.read_csv(csv_file)
        
        if 'smiles' not in self.df.columns:
            raise ValueError(f"CSV {csv_file} missing 'smiles' column")
        target_col = None
        for col in ['ΔfH°(298.15 K)', 'Hf298', 'target', 'value']:
            if col in self.df.columns:
                target_col = col
                break
        if target_col is None:
            raise ValueError(f"CSV {csv_file} missing target column")
            
        self.smiles_list = []
        self.targets = []
        
        print(f"Processing {csv_file}...")
        for i, row in self.df.iterrows():
            smi = row['smiles']
            val = row[target_col]
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    g = smiles2graph(smi)
                    if g['node_feat'].shape[0] > 0:
                        canon_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
                        self.smiles_list.append(canon_smi) 
                        self.targets.append(float(val))
                except:
                    pass
        self.targets = np.array(self.targets)
        
    def len(self):
        return len(self.smiles_list)

    def get(self, idx):
        smiles = self.smiles_list[idx]
        y = self.targets[idx]
        gdict = smiles2graph(smiles)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            n_atoms = mol.GetNumHeavyAtoms() / 30.0 
            mw = Descriptors.MolWt(mol) / 400.0
        else:
            n_atoms, mw = 0.0, 0.0
        
        global_feat = torch.tensor([n_atoms, mw], dtype=torch.float).unsqueeze(0) 
        mol_index = torch.tensor([idx], dtype=torch.long)

        data = Data(
            x=torch.tensor(gdict['node_feat'], dtype=torch.float),
            edge_index=torch.tensor(gdict['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(gdict['edge_feat'], dtype=torch.float),
            y=torch.tensor([y], dtype=torch.float),
            mol_index=mol_index, 
            global_feat=global_feat 
        )
        return data

class GNNRegressor(nn.Module):
    def __init__(self, num_node_features, num_edge_features,
                 hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = num_node_features if i == 0 else hidden_dim
            nn_lin = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINEConv(nn_lin, edge_dim=num_edge_features)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout
        
        # 投影头
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 128), 
            nn.Sigmoid() 
        )
        
        # 回归头 (EDL Head)
        self.fc_final = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 输出 4 个参数 [gamma, nu, alpha, beta]
        self.edl_head = nn.Linear(hidden_dim, 4) 

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_add_pool(x, batch)  
        z = self.proj(g)
        
        phys_feat = data.global_feat.view(g.shape[0], -1) 
        g_concat = torch.cat([g, phys_feat], dim=1) 
        
        g_final = self.fc_final(g_concat)
        outputs = self.edl_head(g_final)
        
        # 激活函数约束
        gamma = outputs[:, 0]
        nu = F.softplus(outputs[:, 1]) + 1e-6
        alpha = F.softplus(outputs[:, 2]) + 1.0 + 1e-6
        beta = F.softplus(outputs[:, 3]) + 1e-6
        
        return torch.stack([gamma, nu, alpha, beta], dim=1), z, g

# ==========================================
# 2. EDL Loss 函数
# ==========================================

def edl_loss(outputs, targets, epoch, total_epochs, lambda_coef=0.01):
    gamma, nu, alpha, beta = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
    y = targets.view(-1)
    
    two_beta_lambda = 2 * beta * (1 + nu)
    nll = 0.5 * torch.log(3.14159 / nu) \
        - alpha * torch.log(two_beta_lambda) \
        + (alpha + 0.5) * torch.log(nu * (y - gamma)**2 + two_beta_lambda) \
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        
    error = torch.abs(y - gamma)
    reg = error * (2 * nu + alpha)
    
    annealing_coef = min(1.0, epoch / 10.0) 
    return nll.mean() + lambda_coef * annealing_coef * reg.mean()

# ==========================================
# 3. 训练与评估流程
# ==========================================

def train_epoch(model, loader, optimizer, device, y_mean, y_std, epoch, args):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 1. 获取 EDL 输出
        outputs, _, _ = model(batch) 
        
        # 2. 归一化标签 (NIG Loss 对数值范围敏感，必须归一化)
        y = batch.y.view(-1).to(device)
        y_norm = (y - y_mean) / y_std
        
        # 3. 计算 EDL Loss
        loss = edl_loss(outputs, y_norm, epoch, args.epochs)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)

def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs, _, _ = model(batch)
            
            # 取 gamma 作为预测值
            gamma = outputs[:, 0]
            
            # 反归一化，恢复物理数值
            pred = gamma * y_std + y_mean
            
            y = batch.y.view(-1).cpu().numpy()
            preds.extend(pred.cpu().numpy())
            ys.extend(y)
    
    mae = mean_absolute_error(ys, preds)
    rmse = np.sqrt(mean_squared_error(ys, preds))
    return mae, rmse

def main(args):
    print(f"--- Pre-training with EDL Loss ---")
    dataset = ATCTDataset(args.csv)
    
    idx_all = range(len(dataset))
    train_idx, test_idx = train_test_split(idx_all, test_size=0.1, random_state=42)
    train_dataset = dataset.index_select(train_idx)
    test_dataset = dataset.index_select(test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # 计算统计量 (非常重要！)
    y_train = torch.tensor([dataset.targets[i] for i in train_idx], dtype=torch.float)
    y_mean = y_train.mean().to(args.device)
    y_std = y_train.std().to(args.device)
    print(f"Stats: Mean={y_mean:.4f}, Std={y_std:.4f}")

    sample = dataset[0]
    model = GNNRegressor(
        num_node_features=sample.x.shape[1],
        num_edge_features=sample.edge_attr.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(args.device)

    # 稀疏初始化 (即使是预训练，初始化好一点也没坏处)
    for layer in model.proj:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=3.0)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, -1.0)
    # 关键：初始化 EDL Head
    torch.nn.init.xavier_uniform_(model.edl_head.weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_mae = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, args.device, y_mean, y_std, epoch, args)
        
        # 使用测试集作为验证
        val_mae, val_rmse = evaluate(model, test_loader, args.device, y_mean, y_std)
        print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val MAE {val_mae:.4f} | RMSE {val_rmse:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            # 保存为 new_df_core.pt，供 DETS 使用
            torch.save(model.state_dict(), "DETS/edl_df_core.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Pre-training finished. Best model saved to 'DETS/edl_df_core.pt'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="enthalpy/wudily_cho.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=50)  
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # 创建保存目录
    if not os.path.exists('DETS'):
        os.makedirs('DETS')
        
    main(args)