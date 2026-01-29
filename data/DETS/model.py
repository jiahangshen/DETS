import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool

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
        
        # --- DETS 投影头 (Projection Head) ---
        # 仅利用 GNN 提取的结构特征 g，不拼接 global_feat
        # 因为 Tanimoto 相似度只应该衡量"结构拓扑"，而不应该受分子大小影响
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 128), 
            nn.Sigmoid() 
        )
        
        # --- [关键修改] 回归预测头 (Prediction Head) ---
        # 输入维度 = GNN隐向量(hidden_dim) + 显式物理特征(2)
        # 让模型同时利用 "Deep Features" 和 "Explicit Physical Features"
        self.fc_final = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim), # 这里拼接了 +2
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.mu_head = nn.Linear(hidden_dim, 1) 

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 1. GNN Backbone
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 2. Global Pooling -> 结构特征
        g = global_add_pool(x, batch)  # (Batch, hidden_dim)
        
        # 3. DETS Projection (只用结构特征 g)
        z = self.proj(g)
        
        # 4. [关键修改] Feature Concatenation for Regression
        # data.global_feat 维度是 (Batch, 1, 2) -> 需要 view 成 (Batch, 2)
        phys_feat = data.global_feat.view(g.shape[0], -1) 
        
        # 拼接：[GNN特征, 原子数, 分子量]
        g_concat = torch.cat([g, phys_feat], dim=1) # (Batch, hidden_dim + 2)
        
        # 5. Prediction
        g_final = self.fc_final(g_concat)
        mu = self.mu_head(g_final).view(-1)
        
        return mu, z, g