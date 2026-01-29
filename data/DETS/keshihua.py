import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
import os
import sys

# ==========================================
# 绘图函数定义
# ==========================================
def plot_tsne_fixed_positions(epoch_stats, features, save_path='plots/tsne_fixed_position.png'):
    """
    固定样本位置，仅通过颜色变化展示不同Epoch下样本权重的变化。
    """
    # 检查数据长度一致性
    if len(epoch_stats) > 0:
        w_len = len(epoch_stats[0]['weights'].flatten())
        f_len = len(features)
        if w_len != f_len:
            print(f"Error: Feature length ({f_len}) and Weight length ({w_len}) do not match!")
            return

    print("1. Computing global t-SNE (Fixed Map)...")
    # 随机种子固定，保证每次运行生成的地图形状一样
    # perplexity 设为 30-50 之间通常效果较好
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
    embedding = tsne.fit_transform(features)
    
    # 将坐标存入 DataFrame
    df_base = pd.DataFrame(embedding, columns=['x', 'y'])
    
    # 2. 选择要展示的时间点 (起点, 1/3, 2/3, 终点)
    n_epochs = len(epoch_stats)
    if n_epochs < 4:
        indices = list(range(n_epochs))
    else:
        indices = [0, int(n_epochs*0.25) ,int(n_epochs*0.75), n_epochs-1]
    indices = sorted(list(set(indices))) 
    
    # 3. 绘图设置
    fig, axes = plt.subplots(1, len(indices), figsize=(6 * len(indices), 5), sharex=True, sharey=True)
    if len(indices) == 1: axes = [axes]
    
    # 颜色条的配置
    cmap = "viridis" # 推荐 magma_r, viridis, 或 spectral
    
    print("2. Plotting evolution...")
    for i, idx in enumerate(indices):
        log = epoch_stats[idx]
        epoch_num = log['epoch']
        weights = np.array(log['weights']).flatten()
        
        ax = axes[i]
        
        # --- 技巧：分两层画图，视觉效果更好 ---
        
        # 第一层：背景（所有点）
        # 用非常淡的灰色画出所有点的位置，充当“地图”背景
        ax.scatter(df_base['x'], df_base['y'], c="#6F6D6D", s=10, alpha=0.5, zorder=1)
        
        # 第二层：前景（仅高权重样本）
        # 筛选出有权重的样本（避免权重为0的样本干扰视觉）
        # 这里的阈值取决于权重的分布，如果权重归一化了，可能很小
        active_mask = weights > 0.6
        
        if active_mask.sum() > 0:
            active_x = df_base.loc[active_mask, 'x']
            active_y = df_base.loc[active_mask, 'y']
            active_w = weights[active_mask]
            
            # 排序：让权重大的点画在最上面 (zorder)
            # 这样红色的点不会被蓝色的点遮住
            sort_idx = np.argsort(active_w)
            active_x = active_x.iloc[sort_idx]
            active_y = active_y.iloc[sort_idx]
            active_w = active_w[sort_idx]
            
            # 画散点图，颜色 c=权重
            sc = ax.scatter(active_x, active_y, c=active_w, cmap=cmap, 
                            s=15, alpha=0.9, zorder=2, edgecolors='none')
            
            # 仅在最后一个图添加 colorbar
            if i == len(indices) - 1:
                cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Sample Weight', rotation=270, labelpad=15)
        
        ax.set_title(f'Epoch {epoch_num}', fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 去除边框，显得更像流形分布图
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle('Evolution of Sample Weights on Fixed t-SNE Manifold', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Plot saved to {save_path}")

# ==========================================
# 主程序逻辑
# ==========================================
if __name__ == "__main__":
    # 创建输出目录
    os.makedirs('plots', exist_ok=True)

    print("-> Loading visualization data...")
    try:
        # weights_only=False 是为了兼容旧版 PyTorch 保存的数据结构
        data = torch.load('DETS/viz_data.pt', map_location='cpu',weights_only=False)
        epoch_stats = data['epoch_stats']
        
        # 获取源域特征
        # 注意：这里假设 final_z 是一个字典 {'source': ..., 'target': ...}
        if isinstance(data['final_z'], dict):
            features_source = data['final_z']['source']
        else:
            # 兼容旧版本直接存 source 的情况
            features_source = data['final_z']
            
        print(f"-> Data loaded. Epochs: {len(epoch_stats)}, Feature Shape: {features_source.shape}")
        
    except FileNotFoundError:
        print("Error: 'DETS/viz_data.pt' not found. Please run full_train.py first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # ----------------------------------------------------
    # [关键] 数据降采样 (为了 t-SNE 速度)
    # 如果数据量超过 5000，t-SNE 会非常慢。
    # 我们随机抽取 5000 个点进行可视化，同时同步筛选 weights。
    # ----------------------------------------------------
    MAX_SAMPLES = 8000
    total_samples = len(features_source)

    if total_samples > MAX_SAMPLES:
        print(f"-> Dataset too large ({total_samples}). Downsampling to {MAX_SAMPLES} for visualization...")
        
        # 固定随机种子以复现结果
        np.random.seed(42)
        indices = np.random.choice(total_samples, MAX_SAMPLES, replace=False)
        
        # 1. 筛选特征
        features_vis = features_source[indices]
        
        # 2. 同步筛选每个 Epoch 的权重
        epoch_stats_vis = []
        for log in epoch_stats:
            new_log = log.copy() # 浅拷贝，避免修改原数据
            # 假设 weights 是 numpy array
            if isinstance(log['weights'], torch.Tensor):
                w = log['weights'].numpy()
            else:
                w = log['weights']
                
            new_log['weights'] = w[indices]
            epoch_stats_vis.append(new_log)
            
    else:
        features_vis = features_source
        epoch_stats_vis = epoch_stats

    # ----------------------------------------------------
    # 执行绘图
    # ----------------------------------------------------
    try:
        plot_tsne_fixed_positions(epoch_stats_vis, features_vis)
    except Exception as e:
        print(f"Plotting failed: {e}")
        import traceback
        traceback.print_exc()