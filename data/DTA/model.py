import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Batch

class GraphEncoder(nn.Module):

    def __init__(self,in_dim,hidden_dim=128,num_layers=3,dropout=0.1):
        super().__init__()
        self.layers=nn.ModuleList()

        for i in range(num_layers):
            mlp=nn.Sequential(nn.Linear(in_dim if i==0 else hidden_dim,hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,hidden_dim))
            self.layers.append(GINConv(mlp))

        self.dropout=nn.Dropout(dropout)
        
    def forward(self,graph:Batch):
        x,h=graph.x,None
        for conv in self.layers:
            x=self.dropout(conv(x,graph.edge_index))
        return global_mean_pool(x,graph.batch)

class ProteinEncoder(nn.Module):

    def __init__(self,vocab_size=30,emb_dim=128,n_heads=4,n_layers=2,proj_dim=256,dropout=0.1,max_len=1024):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        self.pos_embedding=nn.Embedding(max_len,emb_dim)
        layer=nn.TransformerEncoderLayer(d_model=emb_dim,nhead=n_heads,dropout=dropout,dim_feedforward=emb_dim*4,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=n_layers)
        self.proj=nn.Sequential(nn.Linear(emb_dim,proj_dim),nn.ReLU(),nn.Dropout(dropout))

    @staticmethod
    def masked_mean(x,mask):
        mask=mask.unsqueeze(-1).type_as(x)
        return (x*mask).sum(1)/mask.sum(1).clamp(min=1e-6)

    def forward(self,input_ids,attention_mask):
        bsz,seq_len=input_ids.size()
        pos=torch.arange(seq_len,device=input_ids.device).unsqueeze(0).expand(bsz,seq_len)
        x=self.embedding(input_ids)+self.pos_embedding(pos)
        x=self.encoder(x,src_key_padding_mask=(attention_mask==0))
        return self.proj(self.masked_mean(x,attention_mask))

class MLP(nn.Module):

    def __init__(self,in_dim,hiddens,out_dim,dropout=0.1):
        super().__init__()
        layers=[]
        last=in_dim
        for h in hiddens:
            layers+=[nn.Linear(last,h),nn.ReLU(),nn.Dropout(dropout)]
            last=h
        layers.append(nn.Linear(last,out_dim))
        self.net=nn.Sequential(*layers)

    def forward(self,x):return self.net(x)

class MolProtRegressor(nn.Module):

    def __init__(self,graph_in_dim,g_hidden=128,g_layers=3,seq_vocab_size=30,seq_emb_dim=128,seq_heads=4,seq_layers=2,seq_proj_dim=256,mlp_hidden=(256,128),dropout=0.1):
        super().__init__()
        self.graph_encoder=GraphEncoder(graph_in_dim,g_hidden,g_layers,dropout)
        self.protein_encoder=ProteinEncoder(vocab_size=seq_vocab_size,emb_dim=seq_emb_dim,n_heads=seq_heads,n_layers=seq_layers,proj_dim=seq_proj_dim,dropout=dropout)
        fusion_in=g_hidden+seq_proj_dim
        self.head=MLP(fusion_in,mlp_hidden,1,dropout)

    def forward(self,graph:Batch,input_ids,attention_mask):
        g_emb=self.graph_encoder(graph)
        p_emb=self.protein_encoder(input_ids,attention_mask)
        return self.head(torch.cat([g_emb,p_emb],-1)).squeeze(-1),torch.cat([g_emb,p_emb],-1)
