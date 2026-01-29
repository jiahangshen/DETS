import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from rdkit import Chem
from math import pi

# 设置风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
colors = sns.color_palette("tab10")

# ==========================================
# 数据加载
# ==========================================
print("-> Loading visualization data...")
try:
    data = torch.load('DETS/viz_data.pt', map_location='cpu', weights_only=False)
    epoch_stats = data['epoch_stats']
    print("-> Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'DETS/viz_data.pt' not found. Please run full_train.py first.")
    exit()
print("-> Loading Baseline predictions...")
baseline_path = 'baseline_preds.npy'  # 确保这个文件在你当前目录下

try:
    # allow_pickle=True 是必须的，因为我们要加载字典
    base_data = np.load(baseline_path, allow_pickle=True).item()
    
    # 获取预测值和真实值
    base_preds = np.array(base_data['pred'])
    base_true = np.array(base_data['true'])
    
    # 计算 Baseline 的绝对误差 (用于 Plot 3)
    base_abs_errors = np.abs(base_preds - base_true)
    
    # 计算 Baseline 的相对误差 (用于 Plot 12)
    # 加 1e-8 防止除以 0
    base_rel_errors = np.abs((base_preds - base_true) / (base_true + 1e-8))
    
    print(f"-> Baseline data loaded. Samples: {len(base_preds)}")

except FileNotFoundError:
    print(f"Warning: '{baseline_path}' not found. Baseline plots will use simulated data (NOT for publication).")
    # 如果没找到文件，设置为空或模拟数据，防止报错（仅用于调试）
    base_preds = None
    base_abs_errors = None
    base_rel_errors = None
# 创建输出目录
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# ==========================================
# 方案 1: 热力学调制热力图 (Thermodynamic Heatmap)
# ==========================================
def plot_1_thermo_heatmap():
    print("Plotting 1: Thermodynamic Heatmap...")
    # 取最后一个 Epoch 的数据
    last_log = epoch_stats[-1]
    E = last_log['E_stat']
    L = last_log['Delta_L']
    W = last_log['weights']
    
    plt.figure(figsize=(8, 6))
    # 过滤掉权重极小的点以减少绘图负担，突出重点
    mask = W > 1e-4
    sc = plt.scatter(E[mask], L[mask], c=W[mask], cmap='inferno', s=15, alpha=0.8)
    
    plt.colorbar(sc, label='Sampling Weight ($w$)')
    plt.xlabel(r'Static Structure Similarity ($E_{stat}$)')
    plt.ylabel(r'Dynamic Property Discrepancy ($\Delta L$)')
    plt.title('Thermodynamic Weight Modulation')
    plt.ylim(0, np.percentile(L, 95)) # 限制Y轴范围，避免极端值压缩图像
    plt.tight_layout()
    plt.savefig('plots/1_thermo_heatmap.png', dpi=300)
    plt.close()

# ==========================================
# 方案 2: 被“拯救”的长尾分布 (The Rescued Long-tail)
# ==========================================
def plot_2_long_tail():
    print("Plotting 2: Rescued Long-tail...")
    last_log = epoch_stats[-1]
    
    # 1. 构造 DataFrame (最稳妥的方式)
    # flatten() 确保是一维数据
    df = pd.DataFrame({
        'Similarity': np.array(last_log['E_stat']).flatten(),
        'Weight': np.array(last_log['weights']).flatten()
    })
    
    plt.figure(figsize=(8, 5))
    
    # 1. 全量分布 (Background)
    # x='Similarity' 指定列名
    sns.kdeplot(data=df, x='Similarity', fill=True, color='gray', alpha=0.3, 
                label='All Theory Data', warn_singular=False)
    
    # 2. DETS 选中分布 (Rescued)
    if df['Weight'].sum() > 1e-6:
        # 关键修复：使用 data=df, x=..., weights=...
        sns.kdeplot(data=df, x='Similarity', weights='Weight', fill=True, color='red', alpha=0.4, 
                    label='DETS Selected', warn_singular=False)
    
    plt.xlabel(r'Structure Similarity to Experiment ($E_{stat}$)')
    plt.ylabel('Density')
    plt.title('Distribution Shift: Rescuing the Long-tail')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/2_long_tail.png', dpi=300)
    plt.close()

# ==========================================
# 方案 3: 难易样本性能分解 (Hard/Easy Breakdown)
# ==========================================
def plot_3_hard_easy():
    print("Plotting 3: Hard/Easy Breakdown...")
    
    # 1. 获取数据
    dets_preds = np.array(data['test_predictions']['pred'])
    dets_true = np.array(data['test_predictions']['true'])
    dets_errors = np.abs(dets_preds - dets_true)
    
    if base_abs_errors is None:
        print("Skipping Plot 3: No baseline data found.")
        return

    # ==========================================
    # [修复] 强制对齐数据长度
    # ==========================================
    n_dets = len(dets_errors)
    n_base = len(base_abs_errors)
    
    if n_dets != n_base:
        print(f"Warning: Sample size mismatch! DETS({n_dets}) vs Baseline({n_base}). Truncating to minimum...")
        min_len = min(n_dets, n_base)
        # 截断数组，只取前 min_len 个
        dets_errors_aligned = dets_errors[:min_len]
        base_abs_errors_aligned = base_abs_errors[:min_len]
    else:
        dets_errors_aligned = dets_errors
        base_abs_errors_aligned = base_abs_errors

    # 3. 定义难易样本 (使用对齐后的数据)
    avg_errors = (dets_errors_aligned + base_abs_errors_aligned) / 2
    threshold = np.median(avg_errors)
    
    easy_mask = avg_errors <= threshold
    hard_mask = avg_errors > threshold
    
    # 4. 计算分组 MAE
    metrics = {
        'Easy Set': [
            np.mean(base_abs_errors_aligned[easy_mask]), 
            np.mean(dets_errors_aligned[easy_mask])      
        ],
        'Hard Set (OOD)': [
            np.mean(base_abs_errors_aligned[hard_mask]), 
            np.mean(dets_errors_aligned[hard_mask])      
        ]
    }
    
    # 5. 绘图
    df = pd.DataFrame(metrics, index=['Baseline', 'DETS']).T.reset_index()
    df_melt = df.melt(id_vars='index', var_name='Method', value_name='MAE')
    
    plt.figure(figsize=(6, 6))
    sns.barplot(data=df_melt, x='index', y='MAE', hue='Method', palette=['gray', 'tab:red'])
    plt.xlabel('Sample Difficulty')
    plt.ylabel('MAE (kcal/mol)') 
    plt.title('Performance on Hard vs. Easy Samples')
    plt.tight_layout()
    plt.savefig('plots/3_hard_easy_breakdown.png', dpi=300)
    plt.close()
# ==========================================
# 方案 4: 采样关注点演变 (Focus Evolution)
# ==========================================
def plot_4_focus_evolution():
    print("Plotting 4: Focus Evolution...")
    n_epochs = len(epoch_stats)
    indices = [0, int(n_epochs*0.3), int(n_epochs*0.6), n_epochs-1]
    indices = sorted(list(set(indices))) 
    
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("viridis", len(indices))
    
    for i, idx in enumerate(indices):
        log = epoch_stats[idx]
        epoch_num = log['epoch']
        
        # 构造临时 DataFrame
        df = pd.DataFrame({
            'Similarity': np.array(log['E_stat']).flatten(),
            'Weight': np.array(log['weights']).flatten()
        })
        
        # 筛选 High Confidence 样本
        df_high = df[df['Weight'] > 0.01]
        
        if len(df_high) > 5:
            # 使用 DataFrame 绘图
            sns.kdeplot(data=df_high, x='Similarity', 
                        label=f'Epoch {epoch_num}', color=palette[i], 
                        linewidth=2, warn_singular=False)
            
    plt.xlabel(r'Structure Similarity ($E_{stat}$)')
    plt.ylabel('Density of Selected Samples')
    plt.title('Evolution of Sampling Focus')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/4_focus_evolution.png', dpi=300)
    plt.close()
# ==========================================
# 方案 5: 数据效率曲线 (Data Efficiency)
# ==========================================
def plot_5_efficiency():
    print("Plotting 5: Data Efficiency...")
    
    # [修改] 从 CSV 加载真实实验记录
    csv_path = 'efficiency_results.csv'
    if not os.path.exists(csv_path):
        print(f"Skipping Plot 5: '{csv_path}' not found. Please summarize your experiment results first.")
        return

    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(7, 5))
    
    # 使用 Seaborn 自动绘图 (支持多条线)
    sns.lineplot(data=df, x='data_ratio', y='mae', hue='method', 
                 style='method', markers=True, dashes=False, linewidth=2,
                 palette={'DETS': 'red', 'Baseline': 'gray'}) # 指定颜色
    
    plt.xlabel('Percentage of Source Data Used (%)')
    plt.ylabel('Test MAE (kcal/mol)')
    plt.title('Data Efficiency Comparison')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/5_efficiency_curve.png', dpi=300)
    plt.close()

# ==========================================
# 方案 6: 物理一致性 (Weight vs Size)
# ==========================================
def plot_6_physical_consistency():
    print("Plotting 6: Physical Consistency...")
    # 证明我们过滤掉了不合理的大分子
    
    weights = data['epoch_stats'][-1]['weights']
    sizes = data['source_sizes'] # 原子数
    
    plt.figure(figsize=(8, 5))
    plt.scatter(sizes, weights, alpha=0.5, c=weights, cmap='Blues')
    
    plt.axvline(x=15, color='r', linestyle='--', label='Physical Constraint (Size <= 15)')
    
    plt.xlabel('Heavy Atom Count')
    plt.ylabel('Assigned Weight ($w$)')
    plt.title('Physical Constraint: Filtering Size Effects')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/6_physical_consistency.png', dpi=300)
    plt.close()

# ==========================================
# 方案 7: 化学空间覆盖地图 (Chemical Space Map)
# ==========================================
def plot_7_chemical_space():
    print("Plotting 7: Chemical Space Map (t-SNE)...")
    z_source = data['final_z']['source']
    z_target = data['final_z']['target']
    weights = np.array(data['epoch_stats'][-1]['weights']).flatten()
    
    # 1. 降采样 Source (如果数据量太大)
    # 假设我们只画 2000 个点，或者画全部
    n_source = len(z_source)
    if n_source > 2000:
        idx = np.random.choice(n_source, 2000, replace=False)
        z_source_sub = z_source[idx]
        w_sub = weights[idx]
    else:
        z_source_sub = z_source
        w_sub = weights
        
    n_sub = len(z_source_sub) # 记录实际画的 Source 点数
    
    # 2. 合并做 t-SNE
    # X 的前 n_sub 行是 Source，后面是 Target
    X = np.concatenate([z_source_sub, z_target], axis=0)
    
    print(f"   Running t-SNE on {len(X)} samples...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_emb = tsne.fit_transform(X)
    
    # 3. 拆分坐标
    src_emb = X_emb[:n_sub] # 前 n_sub 个是 Source
    tgt_emb = X_emb[n_sub:] # 剩下的是 Target
    
    # 4. 绘图
    plt.figure(figsize=(10, 8))
    
    # 画 Source (用权重染色)
    # 关键：c=w_sub 必须长度等于 src_emb 的行数
    sc = plt.scatter(src_emb[:,0], src_emb[:,1], c=w_sub, cmap='plasma', s=15, alpha=0.5, label='Theory Data')    
    # 画 Target (高亮)
    plt.scatter(tgt_emb[:,0], tgt_emb[:,1], c='red', marker='*', s=80, alpha=0.9, edgecolors='black', linewidth=0.5, label='Experiment Anchors')
    
    plt.title('Chemical Space Coverage (t-SNE)')
    
    # 调整图例位置，防止遮挡
    plt.legend(loc='upper right', frameon=True, framealpha=0.8)
    plt.axis('off')
    plt.colorbar(sc, label='DETS Sampling Weight')
    plt.tight_layout()
    plt.savefig('plots/7_chemical_space.png', dpi=300)
    plt.close()

# ==========================================
# 方案 8: 证据互补性 (Evidence Correlation)
# ==========================================
def plot_8_evidence_correlation():
    print("Plotting 8: Evidence Correlation...")
    last_log = epoch_stats[-1]
    E = last_log['E_stat']
    L = last_log['Delta_L']
    
    plt.figure(figsize=(7, 7))
    sns.kdeplot(x=E, y=L, fill=True, cmap="Blues", thresh=0.05)
    plt.scatter(E, L, s=5, alpha=0.1, color='black')
    
    plt.xlabel(r'Static Evidence ($E_{stat}$)')
    plt.ylabel(r'Dynamic Evidence ($\Delta L$)')
    plt.title('Complementarity of Dual Evidences')
    plt.tight_layout()
    plt.savefig('plots/8_evidence_correlation.png', dpi=300)
    plt.close()

# ==========================================
# 方案 9: 消融实验雷达图 (Ablation Radar)
# ==========================================
def plot_9_ablation_radar():
    print("Plotting 9: Ablation Radar...")
    
    # 模拟数据 (请替换为你真实的消融实验结果)
    # 指标需要归一化到 [0, 1] 之间，越大越好
    labels = np.array(['MAE', 'RMSE', 'MAPE', 'Acc(10%)', 'Stability'])
    
    # 假设数据：DETS 全面领先
    dets_scores = [0.9, 0.85, 0.9, 0.95, 0.9]
    no_dynamic = [0.7, 0.65, 0.7, 0.75, 0.8]
    no_static = [0.6, 0.6, 0.5, 0.4, 0.6]
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1] # 闭环
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    def plot_radar(scores, color, label):
        scores += scores[:1]
        ax.plot(angles, scores, color=color, linewidth=2, label=label)
        ax.fill(angles, scores, color=color, alpha=0.25)
        
    plot_radar(dets_scores, 'red', 'DETS (Full)')
    plot_radar(no_dynamic, 'blue', 'w/o Dynamic')
    plot_radar(no_static, 'gray', 'w/o Static')
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title('Ablation Study: Component Contribution')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig('plots/9_ablation_radar.png', dpi=300)
    plt.close()

# ==========================================
# 方案 10: 系统温度监控 (Temperature Monitoring)
# ==========================================
def plot_10_temp_curve():
    print("Plotting 10: Temperature Curve...")
    epochs = [log['epoch'] for log in epoch_stats]
    temps = [log['avg_sigma'] for log in epoch_stats]
    
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, temps, color='orange', linewidth=2)
    plt.fill_between(epochs, temps, color='orange', alpha=0.1)
    
    plt.xlabel('Epoch')
    plt.ylabel(r'System Temperature ($\sigma_{eff}$)')
    plt.title('Dynamic Adaptation of Sampling Temperature')
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig('plots/10_temp_curve.png', dpi=300)
    plt.close()

# ==========================================
# 方案 11: 关键官能团频率图 (Motif Shift)
# ==========================================
def plot_11_motif_shift():
    print("Plotting 11: Motif Frequency Shift...")
    # 定义几个感兴趣的子结构 (SMARTS)
    motifs = {
        'Benzene': 'c1ccccc1',
        'Amine': '[NX3;H2]',
        'Carbonyl': 'C=O',
        'Hydroxyl': '[OX2H]',
        'Halogen': '[F,Cl,Br,I]'
    }
    
    src_smiles = data['source_smiles']
    tgt_smiles = data['target_smiles']
    weights = data['epoch_stats'][-1]['weights']
    
    # 筛选 DETS 选中的 High Weight 样本
    high_w_indices = np.where(weights > 0.1)[0]
    dets_smiles = [src_smiles[i] for i in high_w_indices]
    
    # 计算频率函数
    def calc_freq(smiles_list):
        counts = {k: 0 for k in motifs}
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                for name, smarts in motifs.items():
                    patt = Chem.MolFromSmarts(smarts)
                    if mol.HasSubstructMatch(patt):
                        counts[name] += 1
        return [c/len(smiles_list) for c in counts.values()]

    freq_src = calc_freq(src_smiles)
    freq_tgt = calc_freq(tgt_smiles)
    freq_dets = calc_freq(dets_smiles)
    
    # 绘图
    x = np.arange(len(motifs))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, freq_src, width, label='Raw Source (Biased)', color='gray', alpha=0.5)
    plt.bar(x, freq_tgt, width, label='Target (Goal)', color='gold', alpha=0.8)
    plt.bar(x + width, freq_dets, width, label='DETS Selected', color='red', alpha=0.8)
    
    plt.xticks(x, motifs.keys())
    plt.ylabel('Frequency of Occurrence')
    plt.title('Chemical Motif Distribution Shift')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/11_motif_shift.png', dpi=300)
    plt.close()

# ==========================================
# 方案 12: 容差敏感度分析 (Tolerance Sensitivity)
# ==========================================
def plot_12_tolerance_sensitivity():
    print("Plotting 12: Tolerance Sensitivity...")
    
    dets_preds = np.array(data['test_predictions']['pred'])
    dets_true = np.array(data['test_predictions']['true'])
    
    if base_rel_errors is None:
        return

    # ==========================================
    # [修复] 强制对齐
    # ==========================================
    min_len = min(len(dets_preds), len(base_rel_errors))
    
    # 截断 DETS 数据
    dets_preds = dets_preds[:min_len]
    dets_true = dets_true[:min_len]
    
    # 截断 Baseline 误差
    base_rel_err_aligned = base_rel_errors[:min_len]
    
    # 重新计算 DETS 相对误差 (基于截断后的数据)
    dets_rel_err = np.abs((dets_preds - dets_true) / (dets_true + 1e-8))

    tolerances = np.linspace(0.01, 0.25, 50)
    accs_dets = []
    accs_base = []
    
    for tol in tolerances:
        acc_d = np.mean(dets_rel_err <= tol) * 100
        accs_dets.append(acc_d)
        
        # 使用对齐后的 Baseline 误差
        acc_b = np.mean(base_rel_err_aligned <= tol) * 100
        accs_base.append(acc_b)
        
    plt.figure(figsize=(8, 5))
    plt.plot(tolerances*100, accs_dets, label='DETS (Ours)', color='red', linewidth=3)
    plt.plot(tolerances*100, accs_base, label='Baseline (MDG-IS)', color='gray', linestyle='--', linewidth=2)
    
    # [修改] 1. 绘制垂直线
    target_tol_x = 10 # 10%
    plt.axvline(x=target_tol_x, color='blue', linestyle=':', alpha=0.6)
    
    # [新增] 2. 计算 10% 处的准确率差值，并添加标注
    # 找到 x=10 对应的索引 (tolerances 是 0.01 到 0.25)
    # 10% 对应的是 0.10
    idx_10 = (np.abs(tolerances - 0.10)).argmin()
    y_dets = accs_dets[idx_10]
    y_base = accs_base[idx_10]
    delta_acc = y_dets - y_base
    
    # 添加双向箭头标注差值
    plt.annotate(
        '', xy=(target_tol_x, y_base), xytext=(target_tol_x, y_dets),
        arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5)
    )
    # 添加文本说明
    plt.text(target_tol_x + 0.5, (y_dets + y_base)/2, f'$\\Delta$ Acc ≈ +{delta_acc:.1f}%', 
             color='blue', fontsize=12, va='center', fontweight='bold')

    plt.xlabel('Tolerance Threshold (%)')
    plt.ylabel('Accuracy within Tolerance (%)')
    plt.title('Accuracy vs. Tolerance Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/12_tolerance_sensitivity.png', dpi=300)
    plt.close()
# ==========================================
# 执行
# ==========================================
if __name__ == "__main__":
    print("Starting visualization generation...")
    plot_1_thermo_heatmap()
    plot_2_long_tail()
    plot_3_hard_easy()
    plot_4_focus_evolution()
    plot_5_efficiency()
    plot_6_physical_consistency()
    plot_7_chemical_space()
    plot_8_evidence_correlation()
    plot_9_ablation_radar()
    plot_10_temp_curve()
    plot_11_motif_shift()
    plot_12_tolerance_sensitivity()
    print("\nAll 12 plots have been generated in the 'plots/' directory!")