import pandas as pd
import numpy as np
import os
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
import argparse # 引入 argparse
from matminer.featurizers.conversions import Composition
from matminer.featurizers.composition import ElementProperty
class CompositionDataset(Dataset):
    def __init__(self, csv_path, formula_col, bandgap_col):
        self.df = pd.read_csv(csv_path)
        self.formula_col = formula_col
        self.bandgap_col = bandgap_col
        self.featurizer = ElementProperty.from_preset("magpie", impute_nan=True)

        element_list = ['Kr', 'B', 'P', 'La', 'Ti', 'Tc', 'Ce', 'Sr', 'Ni', 'N', 'Al', 
                        'Ru', 'Hf', 'Ne', 'Mn', 'H', 'Cu', 'Mg', 'Lu', 'Au', 'Ir', 'F', 
                        'Sn', 'Pt', 'He', 'O', 'Ar', 'Nb', 'Li', 'Rh', 'Zn', 'Ca', 'Be', 
                        'I', 'C', 'Os', 'Co', 'Na', 'Ge', 'Se', 'Y', 'Tl', 'Cr', 'Ta', 'Zr', 
                        'S', 'Ag', 'Mo', 'Ba', 'Cd', 'Dy', 'Ga', 'Xe', 'As', 'Si', 'Pb', 
                        'Rb', 'In', 'Fe', 'Bi', 'Pd', 'Th', 'Cs', 'Sc', 'K', 'Sb', 'W',
                        'Re', 'Cl', 'Hg', 'V', 'Te', 'Br']

        self.compositions = []
        self.bandgaps = []
        self.total_features = []

        element_to_idx = {el: i for i, el in enumerate(element_list)}
        num_elements = len(element_list)

        for _, row in self.df.iterrows():
            formula = row[self.formula_col]
            bandgap = float(row[self.bandgap_col])

            comp = Composition(formula)
            comp_vec = torch.zeros(num_elements, dtype=torch.float32)

            # Magpie features
            x_total_feat = self.featurizer.featurize(comp)

            self.total_features.append(x_total_feat)
            # One-hot fractional vector
            for el, amt in comp.get_el_amt_dict().items():
                if el in element_to_idx:
                    idx = element_to_idx[el]
                    comp_vec[idx] = amt

            total = comp_vec.sum()
            comp_frac = comp_vec / total if total > 0 else comp_vec

            self.compositions.append(comp_frac)
            self.bandgaps.append([bandgap])

        self.total_features = torch.tensor(self.total_features, dtype=torch.float32)
        self.compositions = torch.stack(self.compositions)
        self.bandgaps = torch.tensor(self.bandgaps, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x_comp": self.compositions[idx],
            "x_total_feats": self.total_features[idx],
            "y_bandgap": self.bandgaps[idx],
            "y_comp": self.compositions[idx],
        }