import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ================= 配置 =================
INPUT_FILE = 'bandgap/processed_bandgap.csv'
OUTPUT_DIR = 'bandgap'

# 划分比例
THEORY_RATIO = 0.9  # 90% 数据作为理论数据
EXP_RATIO    = 0.1  # 10% 数据作为实验数据
SEED         = 42   # 随机种子，保证可复现
# ========================================

def main():
    print(f"-> Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Error: Input file not found!")
        return

    # 1. 筛选列 (Drop optical columns)
    # 我们只需要 Formula, GGA, HSE
    # 假设列名固定，如果可能有空格，strip一下
    df.columns = df.columns.str.strip()
    
    keep_cols = ['material formula', 'Band_gap_GGA', 'Band_gap_HSE']
    
    # 检查列是否存在
    if not all(col in df.columns for col in keep_cols):
        print(f"Error: Missing columns. Expected {keep_cols}, found {df.columns.tolist()}")
        return
        
    df_clean = df[keep_cols].copy()
    
    # 重命名方便后续处理 (统一叫 formula)
    df_clean = df_clean.rename(columns={'material formula': 'formula'})
    
    # 2. 清洗无效值
    # 去除 NaN 或 Inf
    original_len = len(df_clean)
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"-> Data cleaned: {len(df_clean)} samples (dropped {original_len - len(df_clean)} invalid rows).")

    # 3. 数据划分
    # 这里的逻辑是：
    # 理论数据 = 随机选 90% 的样本，取其 GGA 值
    # 实验数据 = 随机选 10% 的样本，取其 HSE 值
    # 注意：这两个集合应该是【互斥】的吗？
    # 通常模拟"迁移学习"时，我们假设不仅标签不同，样本也是不同的（模拟 unseen data）。
    
    # 方法 A: 互斥划分 (Disjoint Split) -- 推荐，更符合真实场景
    df_theory_raw, df_exp_raw = train_test_split(
        df_clean, 
        test_size=EXP_RATIO, 
        random_state=SEED,
        shuffle=True
    )
    
    # 4. 构建最终数据集
    
    # --- Theory Dataset ---
    # 只保留 formula 和 GGA (重命名为 target)
    df_theory = df_theory_raw[['formula', 'Band_gap_GGA']].copy()
    df_theory = df_theory.rename(columns={'Band_gap_GGA': 'target'})
    
    # --- Experiment Dataset ---
    # 只保留 formula 和 HSE (重命名为 target)
    df_exp = df_exp_raw[['formula', 'Band_gap_HSE']].copy()
    df_exp = df_exp.rename(columns={'Band_gap_HSE': 'target'})
    
    # 5. 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    theory_path = os.path.join(OUTPUT_DIR, 'new_theory.csv')
    exp_path = os.path.join(OUTPUT_DIR, 'new_exp.csv')
    
    df_theory.to_csv(theory_path, index=False)
    df_exp.to_csv(exp_path, index=False)
    
    print("\n[Summary]")
    print(f"Total Samples: {len(df_clean)}")
    print("-" * 30)
    print(f"Theory Data (GGA): {len(df_theory)} samples saved to {theory_path}")
    print(f"   (Source Domain: Low Fidelity)")
    print("-" * 30)
    print(f"Exp Data (HSE):    {len(df_exp)} samples saved to {exp_path}")
    print(f"   (Target Domain: High Fidelity)")
    print("\nDone! Now run 'preprocess_data.py'.")

if __name__ == "__main__":
    main()