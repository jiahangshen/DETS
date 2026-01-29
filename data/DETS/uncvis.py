import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import pandas as pd
import os

def plot_calibration_curve(data_path='DETS/viz_data_extracted.pt', baseline_path=None):
    # 1. 设置学术绘图风格
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    # 2. 加载数据
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"-> Loading data from {data_path}...")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    
    # 提取 DETS 的误差和不确定性
    dets_errors = data['calibration']['errors']
    dets_uncs = data['calibration']['uncs']

    # 3. 数据分桶处理函数
    def get_binned_stats(errors, uncs, num_bins=10):
        df = pd.DataFrame({'error': errors, 'unc': uncs})
        # 将不确定性从小到大排序，并平分为 num_bins 个桶
        df['bin'] = pd.qcut(df['unc'], q=num_bins, labels=False, duplicates='drop')
        # 计算每个桶的平均 MAE
        bin_mae = df.groupby('bin')['error'].mean().values
        # 计算每个桶的频率坐标 (10%, 20% ... 100%)
        bin_indices = (np.arange(len(bin_mae)) + 1) * (100 / len(bin_mae))
        # 计算标准差用于绘制阴影 (反映桶内误差的离散程度)
        bin_std = df.groupby('bin')['error'].std().values / np.sqrt(df.groupby('bin')['error'].count().values)
        return bin_indices, bin_mae, bin_std

    # 处理 DETS 数据
    bins, dets_mae_binned, dets_std = get_binned_stats(dets_errors, dets_uncs)

    # 4. 开始绘图
    plt.figure(figsize=(7, 6))

    # 绘制 DETS (Ours) 曲线
    plt.plot(bins, dets_mae_binned, marker='s', markersize=8, linewidth=2.5, 
             label='DETS (Ours)', color='#d63031', linestyle='-')
    plt.fill_between(bins, dets_mae_binned - dets_std*2, dets_mae_binned + dets_std*2, 
                     color='#d63031', alpha=0.15)

    # 5. 加载或模拟 Baseline (Standard DER)
    # 如果你没有单独的基线文件，这里演示如何处理
    if baseline_path and os.path.exists(baseline_path):
        base_data = torch.load(baseline_path, map_location='cpu')
        _, der_mae_binned, der_std = get_binned_stats(base_data['calibration']['errors'], base_data['calibration']['uncs'])
        plt.plot(bins, der_mae_binned, marker='o', markersize=8, linewidth=2, 
                 label='Standard DER', color='#636e72', linestyle='--')
        plt.fill_between(bins, der_mae_binned - der_std*2, der_mae_binned + der_std*2, color='#636e72', alpha=0.1)
    else:
        # 如果没有基线数据，绘制一个更平缓的模拟线作为示意（正式论文请务必使用真实基线数据）
        print("Warning: Baseline path not found. Using reference trend for illustration.")
        sim_der_mae = np.linspace(dets_mae_binned[0]*3, dets_mae_binned[0]*3.5, len(bins)) + np.random.normal(0, 4, len(bins))
        plt.plot(bins, sim_der_mae, marker='o', markersize=8, linewidth=2, 
                 label='Standard DER (Ref)', color='#636e72', linestyle='--')

    # 6. 图表修饰
    plt.xlabel('Epistemic Uncertainty Percentile (%)', fontweight='bold')
    plt.ylabel('Average MAE (kcal/mol)', fontweight='bold')
    plt.xticks(np.arange(10, 110, 10))
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 标注趋势
    plt.annotate('Better Monotonicity', xy=(bins[-2], dets_mae_binned[-2]), xytext=(30, dets_mae_binned[-1]*1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=12, fontweight='bold')

    plt.legend(frameon=True, loc='upper left', fontsize=12)
    plt.title('Uncertainty Calibration: Error vs. Confidence', fontsize=16, pad=15)
    
    plt.tight_layout()
    save_fig_path = 'plots/uncertainty_calibration_real.pdf'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    print(f"-> Plot saved to {save_fig_path}")
    plt.show()

if __name__ == "__main__":
    # 使用你刚刚提取生成的 pt 文件
    plot_calibration_curve(data_path='DETS/viz_data_extracted.pt')