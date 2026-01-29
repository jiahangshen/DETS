import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool

class GNNRegressor(nn.Module):
    def __init__(self, num_node_features, num_edge_features,
                 hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 独立的 GNN Backbone
        self.gnn_solvent = self._build_gnn_backbone(num_node_features, num_edge_features, hidden_dim, num_layers)
        self.gnn_solute = self._build_gnn_backbone(num_node_features, num_edge_features, hidden_dim, num_layers)

        # DETS 投影头
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 128), 
            nn.Sigmoid()
        )
        
        # 预测头：(128 * 2) + 4 = 260
        input_dim_final = hidden_dim * 2 + 4 
        self.fc_final = nn.Sequential(
            nn.Linear(input_dim_final, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.edl_head = nn.Linear(hidden_dim, 4) 

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
        # --- [关键修改] 使用 follow_batch 生成的特定 batch 向量 ---
        # 溶剂
        x_solvent = data.x_solvent
        edge_index_solvent = data.edge_index_solvent
        edge_attr_solvent = data.edge_attr_solvent
        batch_solvent = data.x_solvent_batch 

        h_solvent = self._forward_gnn(self.gnn_solvent, x_solvent, edge_index_solvent, edge_attr_solvent)
        
        # [修改点 1] 显式传入 size=data.num_graphs
        g_solvent = global_add_pool(h_solvent, batch_solvent, size=data.num_graphs) 
        
        # 溶质
        x_solute = data.x_solute
        edge_index_solute = data.edge_index_solute
        edge_attr_solute = data.edge_attr_solute
        batch_solute = data.x_solute_batch

        h_solute = self._forward_gnn(self.gnn_solute, x_solute, edge_index_solute, edge_attr_solute)
        
        # [修改点 2] 显式传入 size=data.num_graphs
        g_solute = global_add_pool(h_solute, batch_solute, size=data.num_graphs) 

        # --- DETS 投影 ---
        z_solute = self.proj(g_solute)
        
        # --- 特征拼接 ---
        # data.global_feat 是 (Batch, 1, 4) -> view 成 (Batch, 4)
        phys_feat = data.global_feat.view(g_solvent.shape[0], -1) 
        
        # [GNN溶剂, GNN溶质, 物理特征]
        g_concat = torch.cat([g_solvent, g_solute, phys_feat], dim=1) 
        
        # --- 预测 ---
        g_final = self.fc_final(g_concat)
        outputs = self.edl_head(g_final)
        
        gamma = outputs[:, 0]
        nu = F.softplus(outputs[:, 1]) + 1e-6
        alpha = F.softplus(outputs[:, 2]) + 1.0 + 1e-6
        beta = F.softplus(outputs[:, 3]) + 1e-6
        
        edl_outputs = torch.stack([gamma, nu, alpha, beta], dim=1)
        
        return edl_outputs, z_solute, g_final

    def _forward_gnn(self, gnn_module, x, edge_index, edge_attr):
        for conv, bn in zip(gnn_module['convs'], gnn_module['bns']):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x