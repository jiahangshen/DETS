import pandas as pd

# 加载数据
df_target = pd.read_csv('enthalpy/atct.csv')
df_source = pd.read_csv('enthalpy/wudily_cho.csv')

# 找到公共分子 (假设都有 'smiles' 和 'ΔfH°(298.15 K)' 列)
# 注意：实际代码中你的列名可能略有不同，请检查 dataset.py
# Target可能有 'Species Name'，Source直接是smiles
# 这里简化处理，假设你 dataset.py 里的清洗逻辑是对的

# 为了快速检查，我们手动看几个常见分子
common_smiles = ['C', 'CC', 'C=C', 'O', 'CO'] # 甲烷, 乙烷, 乙烯, 水, 一氧化碳

print(f"{'SMILES':<10} | {'Target (Exp)':<15} | {'Source (Theo)':<15} | {'Diff'}")
print("-" * 50)

for smi in common_smiles:
    # 模糊匹配，实际情况可能需要 RDKit Canonicalize
    # 这里只是演示逻辑
    val_t = df_target[df_target['smiles'] == smi]['ΔfH°(298.15 K)'].values
    val_s = df_source[df_source['smiles'] == smi]['ΔfH°(298.15 K)'].values
    
    if len(val_t) > 0 and len(val_s) > 0:
        vt = val_t[0]
        vs = val_s[0]
        print(f"{smi:<10} | {vt:<15.2f} | {vs:<15.2f} | {vt - vs:.2f}")