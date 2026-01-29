import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
from ogb.utils import smiles2graph
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

class ATCTDataset(Dataset):
    def __init__(self, csv_file, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.df = pd.read_csv(csv_file)
        
        # 1. 检查列名
        if 'smiles' not in self.df.columns:
            raise ValueError(f"CSV {csv_file} missing 'smiles' column")
        # 支持多种可能的列名写法
        target_col = None
        for col in ['ΔfH°(298.15 K)', 'Hf298', 'target', 'value']:
            if col in self.df.columns:
                target_col = col
                break
        if target_col is None:
            raise ValueError(f"CSV {csv_file} missing target column (e.g., 'ΔfH°(298.15 K)')")
            
        self.smiles_list = []
        self.targets = []
        
        # 2. 清洗与规范化
        print(f"Processing {csv_file}...")
        valid_count = 0
        for i, row in self.df.iterrows():
            smi = row['smiles']
            val = row[target_col]
            
            # RDKit 规范化
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    g = smiles2graph(smi)
                    if g['node_feat'].shape[0] > 0:
                        canon_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
                        self.smiles_list.append(canon_smi) 
                        self.targets.append(float(val))
                        valid_count += 1
                except:
                    pass
        
        print(f"-> Loaded {valid_count} valid molecules from {len(self.df)} rows.")
        self.targets = np.array(self.targets)
        
    def len(self):
        return len(self.smiles_list)

    def get(self, idx):
        smiles = self.smiles_list[idx]
        y = self.targets[idx]
        gdict = smiles2graph(smiles)
        
        # --- [新增] 显式物理特征提取 (Explicit Physical Features) ---
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # 根据 EDA 图表观察到的最大值进行归一化
            # Heavy Atom Count (Max ~25 in Source) -> 除以 30
            n_atoms = mol.GetNumHeavyAtoms() / 30.0 
            # Molecular Weight (Max ~350 in Source) -> 除以 400
            mw = Descriptors.MolWt(mol) / 400.0
        else:
            # 理论上不会进这里，因为 init 已经过滤过了，以防万一
            n_atoms = 0.0
            mw = 0.0
        
        # 存为一个 (1, 2) 的向量
        global_feat = torch.tensor([n_atoms, mw], dtype=torch.float).unsqueeze(0) 
        # --------------------------------------------------------

        data = Data(
            x=torch.tensor(gdict['node_feat'], dtype=torch.float),
            edge_index=torch.tensor(gdict['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(gdict['edge_feat'], dtype=torch.float),
            y=torch.tensor([y], dtype=torch.float),
            mol_index=torch.tensor([idx], dtype=torch.long),
            raw_idx=torch.tensor([idx], dtype=torch.long), 
            # 将全局特征存入 Data 对象，以便 DataLoader 自动 batching
            global_feat=global_feat 
        )
        return data