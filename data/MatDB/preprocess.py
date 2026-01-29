import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

try:
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
except ImportError:
    raise ImportError("请先安装 matminer: pip install matminer")

# 配置
FILES = {
    'theory': ('bandgap/new_theory.csv', 'bandgap/theory_features_clean.csv'),
    'exp':    ('bandgap/new_exp.csv',    'bandgap/exp_features_clean.csv')
}

class FeatureEngine:
    def __init__(self):
        print("-> Initializing Matminer Featurizer...")
        self.featurizer = ElementProperty.from_preset("magpie")
        # 用于记录 Theory 中保留下来的有效特征列名
        self.valid_columns = None 

    def process_file(self, input_path, output_path, is_theory=False):
        print(f"\nProcessing {input_path}...")
        
        # 1. 读取 & 2. 统一列名
        try:
            df = pd.read_csv(input_path, sep=None, engine='python')
        except Exception as e:
            print(f"Error: {e}"); return

        if len(df.columns) >= 2:
            cols = df.columns.tolist()
            df = df.rename(columns={cols[0]: 'formula', cols[1]: 'target'})
            
        df['formula'] = df['formula'].astype(str).str.replace(' ', '')
        
        print(f"  Extracting features for {len(df)} samples...")
        
        # 3. 提取特征
        try:
            df = StrToComposition().featurize_dataframe(df, 'formula')
            df = self.featurizer.featurize_dataframe(df, col_id='composition', ignore_errors=True)
        except Exception as e:
            print(f"Featurization failed: {e}"); return

        # 4. 后处理
        if 'composition' in df.columns:
            df = df.drop(columns=['composition'])
            
        meta_cols = ['formula', 'target']
        feat_cols = [c for c in df.columns if c not in meta_cols]
        df[feat_cols] = df[feat_cols].fillna(0.0)
        
        # --- [关键逻辑] 特征筛选与对齐 ---
        if is_theory:
            print("  [Theory] Identifying valid features...")
            # 筛选：去除全0或常数列
            valid_cols = []
            for col in feat_cols:
                # 检查方差是否为0 (即是否全是同一个数)
                if df[col].std() > 1e-6: # 给一点点容错
                    valid_cols.append(col)
                else:
                    # 可选：打印被删掉的列
                    # print(f"    Dropping constant col: {col}")
                    pass
            
            # 保存这一份"白名单"
            self.valid_columns = valid_cols
            print(f"  [Theory] Kept {len(valid_cols)}/{len(feat_cols)} features.")
            
            # 对 Theory 数据应用筛选
            final_cols = meta_cols + valid_cols
            df = df[final_cols]
            
        else:
            # 如果是 Exp 数据
            if self.valid_columns is None:
                raise ValueError("Error: Must process Theory data first to determine feature set!")
            
            print(f"  [Exp] Aligning features to Theory ({len(self.valid_columns)} cols)...")
            
            # 1. 补齐缺失的列 (如果 Exp 里没有 Theory 有的列，补 0)
            for col in self.valid_columns:
                if col not in df.columns:
                    df[col] = 0.0
            
            # 2. 丢弃多余的列 (如果 Exp 里有 Theory 没有的列，删掉)
            # 3. 重新排序，确保列顺序完全一致
            final_cols = meta_cols + self.valid_columns
            # 只保留白名单里的列
            df = df[final_cols] 
            
        # ----------------------------
        
        print(f"  Saving to {output_path} (Shape: {df.shape})")
        df.to_csv(output_path, index=False, float_format='%.6f')

if __name__ == "__main__":
    engine = FeatureEngine()
    
    # 强制先处理 Theory
    print("--- Step 1: Processing Theory Data (Source) ---")
    if os.path.exists(FILES['theory'][0]):
        engine.process_file(FILES['theory'][0], FILES['theory'][1], is_theory=True)
    else:
        print("Theory file not found!")
        exit()
        
    # 再处理 Exp
    print("\n--- Step 2: Processing Exp Data (Target) ---")
    if os.path.exists(FILES['exp'][0]):
        engine.process_file(FILES['exp'][0], FILES['exp'][1], is_theory=False)
    else:
        print("Exp file not found!")