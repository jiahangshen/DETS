import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors

def analyze_correlation(csv_file, dataset_name):
    print(f"Analyzing {dataset_name} ({csv_file})...")
    df = pd.read_csv(csv_file)
    
    # 提取目标值 (生成焓)
    # 注意：根据你的 csv 实际列名修改，这里假设是 'ΔfH°(298.15 K)'
    target_col = 'ΔfH°(298.15 K)' 
    
    atom_counts = []
    mol_weights = []
    targets = []
    
    for i, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol:
                # 1. 计算重原子数 (Heavy Atom Count)
                n_atoms = mol.GetNumHeavyAtoms()
                # 2. 计算分子量 (Molecular Weight)
                mw = Descriptors.MolWt(mol)
                
                atom_counts.append(n_atoms)
                mol_weights.append(mw)
                targets.append(row[target_col])
        except:
            pass
            
    # 绘图
    plt.figure(figsize=(12, 5))
    
    # 子图 1: 原子数 vs 生成焓
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=atom_counts, y=targets, alpha=0.5)
    plt.xlabel("Heavy Atom Count")
    plt.ylabel("Formation Enthalpy (kcal/mol)")
    plt.title(f"{dataset_name}: Atoms vs Enthalpy")
    
    # 子图 2: 分子量 vs 生成焓
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=mol_weights, y=targets, alpha=0.5, color='orange')
    plt.xlabel("Molecular Weight")
    plt.ylabel("Formation Enthalpy (kcal/mol)")
    plt.title(f"{dataset_name}: MW vs Enthalpy")
    
    plt.tight_layout()
    plt.savefig(f"correlation_{dataset_name}.png")
    print(f"Saved plot to correlation_{dataset_name}.png")

# 运行分析
analyze_correlation('enthalpy/atct.csv', 'Experiment_ATcT')
analyze_correlation('enthalpy/wudily_cho.csv', 'Theory_Wudily')