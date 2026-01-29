import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from ogb.utils import smiles2graph
import pandas as pd

AA_VOCAB = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","X","-","?"]
AA_TO_ID = {aa:i for i,aa in enumerate(AA_VOCAB)}

class MolProtDataset(Dataset):
    def __init__(self, csv_path, max_len=1024):
        df = pd.read_csv(csv_path)
        self.smiles = df["COMPOUND_SMILES"].tolist()
        self.seqs = df["PROTEIN_SEQUENCE"].tolist()
        self.labels = df["REG_LABEL"].astype(float).tolist()
        self.max_len = max_len
        g = smiles2graph("CC")
        self.graph_in_dim = g["node_feat"].shape[1]

    def encode_seq(self, seq):
        ids = [AA_TO_ID.get(ch, AA_TO_ID["?"]) for ch in seq]
        if len(ids) > self.max_len: ids = ids[:self.max_len]
        attn_mask = [1]*len(ids)
        while len(ids)<self.max_len: ids.append(AA_TO_ID["-"]); attn_mask.append(0)
        return torch.tensor(ids), torch.tensor(attn_mask)

    def __len__(self): return len(self.smiles)

    def __getitem__(self, idx):
        gdict = smiles2graph(self.smiles[idx])
        graph = Data(x=torch.tensor(gdict["node_feat"],dtype=torch.float),
                     edge_index=torch.tensor(gdict["edge_index"],dtype=torch.long),
                     edge_attr=torch.tensor(gdict["edge_feat"],dtype=torch.float) if gdict["edge_feat"] is not None else None)
        input_ids, attn_mask = self.encode_seq(self.seqs[idx])
        label = torch.tensor(self.labels[idx],dtype=torch.float)
        return graph, input_ids, attn_mask, label

def collate_fn(batch):
    graphs, ids, masks, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(ids), torch.stack(masks), torch.stack(labels)
