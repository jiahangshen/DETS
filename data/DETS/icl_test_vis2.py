# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np
import os
from torch_geometric.loader import DataLoader
from dataset import ATCTDataset
import argparse
# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, LayerNorm
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans
import os
import shutil
import math
from dataset import ATCTDataset 
class GNNRegressor(nn.Module):
    def __init__(self, num_node_features, num_edge_features,
                 hidden_dim=256, num_layers=4, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.input_proj = nn.Linear(num_node_features, hidden_dim)

        # [复刻] 原始 GINE 结构，无 LayerNorm，使用 BatchNorm
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
        
        # [新增] DETS 投影头
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64) 
        )
        
        # [复刻] 原始 EDL 的预测头逻辑
        self.fc_final = None 
        self.edl_head = None
        self.hidden_dim = hidden_dim

    def _init_heads(self, global_feat_dim):
        input_dim = self.hidden_dim + global_feat_dim
        
        self.fc_final = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim), 
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU()
        )
        self.edl_head = nn.Linear(self.hidden_dim // 2, 4)
        
        torch.nn.init.xavier_uniform_(self.edl_head.weight)
        self.to(next(self.parameters()).device)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        if self.fc_final is None:
            global_feat_dim = data.global_feat.shape[1]
            self._init_heads(global_feat_dim)

        x = self.input_proj(x)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(x, edge_index, edge_attr)
            h = bn(h) # BatchNorm 在 batch_size=1 时会报错，必须在 DataLoader 设 drop_last=True
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h 

        g = global_add_pool(x, batch) 
        z = self.proj(g)
        z = F.normalize(z, p=2, dim=1)
        
        phys_feat = data.global_feat.view(g.shape[0], -1) 
        g_concat = torch.cat([g, phys_feat], dim=1) 
        
        g_final = self.fc_final(g_concat)
        outputs = self.edl_head(g_final) 
        
        gamma = outputs[:, 0]
        nu    = F.softplus(outputs[:, 1]) + 1e-6
        alpha = F.softplus(outputs[:, 2]) + 1.0 + 1e-6
        beta  = F.softplus(outputs[:, 3]) + 1e-6
        
        edl_outputs = torch.stack([gamma, nu, alpha, beta], dim=1)
        return edl_outputs, z, g

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="DETS/final_model.pt")
    parser.add_argument("--target_csv", type=str, default="./enthalpy/atct.csv")
    parser.add_argument("--source_csv", type=str, default="./enthalpy/wudily_cho.csv")
    parser.add_argument("--device", type=str, default="cuda:0")
    # 这里的参数必须和训练时完全一致
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--num_prototypes", type=int, default=10)
    parser.add_argument("--sigma_0", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--w_min", type=float, default=0.01)
    return parser.parse_args()

@torch.no_grad()
def extract():
    args = get_args()
    device = torch.device(args.device)

    # 1. 加载数据
    print("-> Loading datasets...")
    target_dataset = ATCTDataset(args.target_csv)
    source_dataset = ATCTDataset(args.source_csv)
    
    # 2. 初始化并加载模型
    print(f"-> Loading model from {args.model_path}...")
    # 注意：需要先实例化模型并进行一次 dummy forward 来初始化内部的 fc_final
    model = GNNRegressor(
        target_dataset[0].x.shape[1], 
        target_dataset[0].edge_attr.shape[1] if target_dataset[0].edge_attr is not None else 0,
        hidden_dim=args.hidden_dim, num_layers=args.layers
    ).to(device)
    model.eval()
    # Dummy forward 初始化 Head
    sample_batch = next(iter(DataLoader(target_dataset, batch_size=1))).to(device)
    model(sample_batch)
    
    # 加载权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 准备存储字典
    viz_logs = {}

    # 3. 提取特征 (用于 t-SNE)
    print("-> Extracting latent features (z)...")
    def get_z(ds):
        ld = DataLoader(ds, batch_size=512, shuffle=False)
        zs = []
        for b in ld:
            _, z, _ = model(b.to(device))
            zs.append(z.cpu())
        return torch.cat(zs).numpy()

    viz_logs['final_z'] = {
        'source': get_z(source_dataset),
        'target': get_z(target_dataset)
    }

    # 4. 计算最终权重的分布 (模拟训练结束时的状态)
    print("-> Calculating final weight distribution...")
    # 重新建立原型 (Anchors)
    from sklearn.cluster import KMeans
    import torch.nn.functional as F
    z_target = viz_logs['final_z']['target']
    kmeans = KMeans(n_clusters=args.num_prototypes, random_state=42, n_init=10).fit(z_target)
    prototypes = F.normalize(torch.tensor(kmeans.cluster_centers_, dtype=torch.float), p=2, dim=1).to(device)

    # 计算源域权重
    all_weights, all_E, all_Delta = [], [], []
    source_loader = DataLoader(source_dataset, batch_size=1024, shuffle=False)
    for b in source_loader:
        out, z, _ = model(b.to(device))
        nu, alpha, beta = out[:, 1], out[:, 2], out[:, 3]
        unc = beta / (nu * (alpha - 1) + 1e-8)
        
        # Similarity
        sim_matrix = torch.mm(z, prototypes.t())
        E_stat, _ = torch.max(sim_matrix, dim=1)
        E_stat = torch.clamp(E_stat, 0.0, 1.0)
        
        all_E.append(E_stat.cpu())
        all_Delta.append(unc.cpu()) # 这里存原始不确定性，稍后统一归一化

    cat_E = torch.cat(all_E)
    cat_unc = torch.cat(all_Delta)
    delta_L = (cat_unc - cat_unc.min()) / (cat_unc.max() - cat_unc.min() + 1e-8)
    
    # 使用热力学公式计算权重
    sigma_eff = args.sigma_0 * (1.0 + args.gamma * delta_L)
    w_raw = args.alpha * torch.exp(- ((1.0 - cat_E)**2) / (2 * sigma_eff**2 + 1e-8))
    final_weights = torch.clamp(w_raw, min=args.w_min, max=1.0)

    viz_logs['epoch_stats'] = [{
        'epoch': 'Final',
        'weights': final_weights.numpy(),
        'E_stat': cat_E.numpy(),
        'Delta_L': delta_L.numpy()
    }]

    # 5. 提取测试集预测与校准数据 (Calibration)
    print("-> Extracting calibration data on target...")
    # 我们直接用全部 target 数据来展示校准效果
    target_loader = DataLoader(target_dataset, batch_size=256, shuffle=False)
    preds, trues, uncs = [], [], []
    for b in target_loader:
        out, _, _ = model(b.to(device))
        gamma, nu, alpha, beta = out[:,0], out[:,1], out[:,2], out[:,3]
        
        # 假设训练时用的是标准化标签，这里需要获取均值和标准差
        # 如果你没存，可以根据 target_dataset 重新算
        y_vals = torch.tensor([d.y for d in target_dataset])
        y_mean, y_std = y_vals.mean(), y_vals.std()
        
        preds.extend((gamma * y_std + y_mean).cpu().numpy())
        trues.extend(b.y.view(-1).cpu().numpy())
        uncs.extend((beta / (nu * (alpha - 1) + 1e-8)).cpu().numpy())

    viz_logs['test_predictions'] = {'pred': np.array(preds), 'true': np.array(trues)}
    viz_logs['calibration'] = {'errors': np.abs(np.array(preds) - np.array(trues)), 'uncs': np.array(uncs)}

    # 6. 保存
    save_path = 'DETS/viz_data_extracted.pt'
    torch.save(viz_logs, save_path)
    print(f"-> Extraction complete! Data saved to {save_path}")

if __name__ == "__main__":
    extract()