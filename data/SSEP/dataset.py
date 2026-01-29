import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from ogb.utils import smiles2graph
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from tqdm import tqdm
import os

RDLogger.DisableLog('rdApp.*') 

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_solvent':
            return self.x_solvent.size(0)
        if key == 'edge_index_solute':
            return self.x_solute.size(0)
        return 0

class SolvationDataset(InMemoryDataset):
    def __init__(self, root, csv_file=None, transform=None, pre_transform=None):
        """
        root: 数据存储的根目录 (processed 文件会保存在 root/processed 下)
        csv_file: 原始 CSV 文件路径
        """
        self.csv_file = csv_file
        # 初始化 InMemoryDataset，这会自动调用 process() 如果 processed 文件不存在
        super().__init__(root, transform, pre_transform)
        # 加载处理好的数据到内存
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # 这是一个虚拟检查，只要 csv_file 存在即可
        return []

    @property
    def processed_file_names(self):
        # 处理后的文件名，如果修改了数据预处理逻辑，请手动删除 processed 目录下的这个文件
        if self.csv_file:
            # 根据 csv 文件名生成对应的 processed 文件名，防止 source 和 target 冲突
            name = os.path.basename(self.csv_file).replace('.csv', '')
            return [f'data_{name}.pt']
        else:
            return ['data.pt']

    def download(self):
        pass

    def process(self):
        if self.csv_file is None:
            raise ValueError("CSV file not provided for processing!")
            
        print(f"Processing {self.csv_file} into memory (Run Once)...")
        df = pd.read_csv(self.csv_file)
        
        # 检查列名
        if 'smiles_solvent' not in df.columns or 'smiles_solute' not in df.columns:
            raise ValueError(f"CSV missing 'smiles_solvent' or 'smiles_solute'")
        
        target_col = None
        for col in ['dGsolv_avg', 'target', 'value', 'dG', 'expt']: 
            if col in df.columns:
                target_col = col
                break
        if target_col is None:
            raise ValueError("CSV missing target column")

        data_list = []
        
        # 使用 tqdm 显示预处理进度
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Converting Graphs"):
            smi_solvent = row['smiles_solvent']
            smi_solute = row['smiles_solute']
            val = row[target_col]
            
            if pd.isna(smi_solvent) or pd.isna(smi_solute) or pd.isna(val):
                continue
                
            # 1. 图转换 (最耗时步骤)
            try:
                gdict_solvent = smiles2graph(smi_solvent)
                gdict_solute = smiles2graph(smi_solute)
            except:
                continue # 解析失败跳过

            # 2. 物理特征
            mol_solvent = Chem.MolFromSmiles(smi_solvent)
            mol_solute = Chem.MolFromSmiles(smi_solute)
            
            if mol_solvent is None or mol_solute is None:
                continue

            n_atoms_solvent = mol_solvent.GetNumHeavyAtoms() / 30.0 
            mw_solvent = Descriptors.MolWt(mol_solvent) / 400.0
            
            n_atoms_solute = mol_solute.GetNumHeavyAtoms() / 30.0
            mw_solute = Descriptors.MolWt(mol_solute) / 400.0
            
            global_feat = torch.tensor([
                n_atoms_solvent, mw_solvent, n_atoms_solute, mw_solute
            ], dtype=torch.float).unsqueeze(0) 

            data = PairData(
                x_solvent=torch.tensor(gdict_solvent['node_feat'], dtype=torch.float),
                edge_index_solvent=torch.tensor(gdict_solvent['edge_index'], dtype=torch.long),
                edge_attr_solvent=torch.tensor(gdict_solvent['edge_feat'], dtype=torch.float),
                
                x_solute=torch.tensor(gdict_solute['node_feat'], dtype=torch.float),
                edge_index_solute=torch.tensor(gdict_solute['edge_index'], dtype=torch.long),
                edge_attr_solute=torch.tensor(gdict_solute['edge_feat'], dtype=torch.float),
                
                y=torch.tensor([float(val)], dtype=torch.float),
                # 注意：这里不再存储 mol_index 和 raw_idx，以节省内存
                # 如果代码强依赖它们，可以加回去，但通常不需要
                global_feat=global_feat 
            )
            data_list.append(data)

        # 内存数据整理 (Collating into huge tensors)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # 保存到 disk，下次直接 load
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"-> Saved processed data to {self.processed_paths[0]}")
        
    # 为了兼容之前的代码逻辑，添加一个 targets 属性
    @property
    def targets(self):
        return self.data.y.numpy()