import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import Draw
from dataset import ATCTDataset
from model import GNNRegressor
import os

# 配置
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "final_model.pt"
CSV_PATH = "enthalpy/atct.csv"
NUM_PROTOTYPES = 8  

def visualize():
    print(f"-> Loading Model and Data from {CSV_PATH}...")
    dataset = ATCTDataset(CSV_PATH)
    # Shuffle=False 保证顺序
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    sample = dataset[0]
    # 注意：这里的参数必须和你训练时的参数一致 (hidden_dim, layers)
    model = GNNRegressor(
        num_node_features=sample.x.shape[1],
        num_edge_features=sample.edge_attr.shape[1],
        hidden_dim=256, num_layers=3, dropout=0.0
    ).to(DEVICE)
    
    print(f"-> Loading weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 1. 提取所有分子的 z 特征
    print("-> Extracting features...")
    all_z = []
    all_y = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            # model forward 返回 (mu, z, g)
            _, z, _ = model(batch) 
            all_z.append(z.cpu().numpy())
            all_y.extend(batch.y.view(-1).cpu().numpy())

    X = np.concatenate(all_z, axis=0)
    y_vals = np.array(all_y)
    
    # 直接使用 Dataset 中的 SMILES 列表
    all_smiles = dataset.smiles_list
    
    print(f"-> Extracted {len(X)} samples.")
    
    # 2. 聚类
    print(f"-> Clustering into {NUM_PROTOTYPES} prototypes...")
    kmeans = KMeans(n_clusters=NUM_PROTOTYPES, random_state=42, n_init=10).fit(X)
    labels = kmeans.labels_
    
    # 3. t-SNE
    print("-> Running t-SNE (this might take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    # 4. 绘图
    plt.figure(figsize=(12, 8))
    # 使用 Tab10 调色板，区分度高
    sns.scatterplot(
        x=X_embedded[:,0], y=X_embedded[:,1], 
        hue=labels, palette="tab10", s=80, alpha=0.8, legend='full'
    )
    plt.title(f"DETS Latent Space Visualization\n(Clusters represent learned chemical motifs)", fontsize=15)
    
    # 5. 寻找代表分子
    print("-> Identifying representative molecules...")
    centroid_mols = []
    centroid_legends = []
    
    for k in range(NUM_PROTOTYPES):
        mask = (labels == k)
        points = X[mask]
        indices = np.where(mask)[0]
        
        if len(points) == 0: continue
        
        # 找离质心最近的点
        center = kmeans.cluster_centers_[k].reshape(1, -1)
        dists = cdist(points, center)
        nearest_local_idx = np.argmin(dists)
        global_idx = indices[nearest_local_idx]
        
        x_2d, y_2d = X_embedded[global_idx]
        smi = all_smiles[global_idx]
        val = y_vals[global_idx]
        
        # 在图上标注
        plt.text(x_2d, y_2d, f"C{k}", fontsize=12, fontweight='bold', color='black', 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))
        
        mol = Chem.MolFromSmiles(smi)
        if mol:
            centroid_mols.append(mol)
            # 图例: 簇ID | 焓值
            centroid_legends.append(f"C{k}\nH={val:.1f}")

    plt.tight_layout()
    plt.savefig("tsne_clusters.png", dpi=300)
    print("-> Saved plot to 'tsne_clusters.png'")
    
    # 6. 画分子结构
    if len(centroid_mols) > 0:
        print("-> Drawing structures...")
        img = Draw.MolsToGridImage(
            centroid_mols, 
            molsPerRow=4, 
            subImgSize=(300, 300), 
            legends=centroid_legends,
            returnPNG=False
        )
        img.save("prototype_structures.png")
        print("-> Saved structures to 'prototype_structures.png'")
    else:
        print("-> No valid molecules found to draw.")

if __name__ == "__main__":
    visualize()