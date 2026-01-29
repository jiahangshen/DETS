import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
import os

# 忽略警告
warnings.filterwarnings('ignore')

# ==========================================
# 辅助函数
# ==========================================

def smiles_to_morgan(smiles_list, radius=2, n_bits=256):
    """将SMILES转换为Morgan指纹"""
    fps = []
    print(f"  正在转换 {len(smiles_list)} 个 SMILES 到指纹...")
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fps.append(np.zeros(n_bits)) # 无效分子填0
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fps.append(np.array(fp))
        except:
            fps.append(np.zeros(n_bits))
    return np.array(fps)

def fit_gmm(X, n_components, random_state=0):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        reg_covar=1e-6,
        random_state=random_state,
        n_init=3
    )
    gmm.fit(X)
    return gmm

def kl_divergence(gmm_a, gmm_b, X_sample):
    log_pa = gmm_a.score_samples(X_sample)
    log_pb = gmm_b.score_samples(X_sample)
    return np.mean(log_pa - log_pb)

def detect_subspaces(X_target_red, min_samples=5, xi=0.2, min_cluster_size=0.05):
    X_scaled = StandardScaler().fit_transform(X_target_red)
    optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    optics.fit(X_scaled)
    labels = optics.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters

def weighted_log_prob_cluster_gmms(gmm_list, weights, X):
    log_probs = np.zeros((len(gmm_list), len(X)))
    for i, gmm in enumerate(gmm_list):
        log_probs[i] = gmm.score_samples(X) + np.log(weights[i] + 1e-9)
    return np.max(log_probs, axis=0)

# ==========================================
# 核心逻辑
# ==========================================

def generate_core_subset_dta(
    target_csv, 
    source_csv, 
    output_csv='davis_core_subset.csv',
    fingerprint_bits=256, # DTA 分子较复杂，建议增加位数
    fp_radius=2, 
    pca_dim=4, # 增加 PCA 维度
    n_clusters=6, # 增加聚类数
    base_batch=500,
    alpha0=1.0, 
    beta0=1.0,
    tol=1e-4, 
    max_no_improve=5,
    random_state=42,
    verbose=True
):
    print(f"=== 启动 DTA 核心子集筛选 ===")
    print(f"Target (High-Res): {target_csv}")
    print(f"Source (Low-Res):  {source_csv}")
    
    # 1. 数据加载与列名适配
    df_target = pd.read_csv(target_csv)
    df_source = pd.read_csv(source_csv)
    
    # 自动寻找 SMILES 列
    def find_smiles_col(df, name):
        candidates = ['COMPOUND_SMILES', 'smiles', 'SMILES', 'Ligand']
        for col in candidates:
            if col in df.columns:
                print(f"  [{name}] 找到 SMILES 列: {col}")
                return col
        raise ValueError(f"在 {name} 中未找到 SMILES 列，请检查 CSV 表头。")

    target_col = find_smiles_col(df_target, "Target")
    source_col = find_smiles_col(df_source, "Source")
    
    # 2. 数据划分 (仅针对 Target 做划分，Source 全量筛选)
    # 我们只希望 Target 的训练集分布被对齐
    idx_target = np.arange(len(df_target))
    # 8:1:1 划分
    tr_val_idx, test_idx = train_test_split(idx_target, test_size=0.1, random_state=random_state)
    train_idx, val_idx = train_test_split(tr_val_idx, test_size=0.1/0.9, random_state=random_state)
    
    df_target_train = df_target.iloc[train_idx].reset_index(drop=True)
    print(f"Target Split: Train={len(df_target_train)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # 3. 特征工程
    print("\n[Step 1] 生成指纹 & PCA 降维...")
    smiles_target = df_target_train[target_col].astype(str).tolist()
    smiles_source = df_source[source_col].astype(str).tolist()
    
    X_target = smiles_to_morgan(smiles_target, radius=fp_radius, n_bits=fingerprint_bits)
    X_source = smiles_to_morgan(smiles_source, radius=fp_radius, n_bits=fingerprint_bits)
    
    # 合并 PCA
    X_union = np.vstack([X_target, X_source])
    pca = PCA(n_components=min(pca_dim, X_union.shape[1]))
    X_union_red = pca.fit_transform(X_union)
    
    X_target_red = X_union_red[:len(X_target)]
    X_source_red = X_union_red[len(X_target):]
    print(f"  PCA 降维完成: {X_union.shape[1]} -> {X_union_red.shape[1]}")

    # 4. Target 空间建模
    print("\n[Step 2] 建模 Target 化学空间...")
    labels, n_subspaces = detect_subspaces(X_target_red, min_samples=3) # CASF 数据少，min_samples 调小
    subspaces = [np.where(labels == i)[0] for i in range(n_subspaces)]
    weights = np.array([len(idx) for idx in subspaces])
    if len(weights) > 0: weights = weights / weights.sum()
    else: weights = np.array([1.0]); subspaces = [np.arange(len(X_target_red))]
    
    print(f"  检测到 {len(subspaces)} 个高密度子空间")

    # 5. GMM 初始化
    print("\n[Step 3] 初始化 GMM & Bandit...")
    # 这里的 n_clusters 如果大于样本数会报错，做个保护
    n_clusters_safe = min(n_clusters, len(X_target_red))
    gmm_target = fit_gmm(X_target_red, n_components=n_clusters_safe, random_state=random_state)
    gmm_source = fit_gmm(X_source_red, n_components=n_clusters_safe, random_state=random_state)
    
    cluster_labels = gmm_source.predict(X_source_red)
    clusters = [np.where(cluster_labels == k)[0] for k in range(n_clusters_safe)]
    
    # Target 子空间 GMMs
    cluster_gmms_target = []
    valid_weights = []
    for i, idx in enumerate(subspaces):
        if len(idx) < 2: continue
        try:
            gmm = fit_gmm(X_target_red[idx], n_components=min(5, len(idx)), random_state=random_state)
            cluster_gmms_target.append(gmm)
            valid_weights.append(weights[i])
        except: continue
    
    if len(valid_weights) > 0: valid_weights = np.array(valid_weights) / np.sum(valid_weights)

    # Bandit 参数
    alpha = np.ones(n_clusters_safe) * alpha0
    beta = np.ones(n_clusters_safe) * beta0
    
    # 初始核心集
    init_size = max(100, int(0.01 * len(X_source_red)))
    core_idx = set(np.random.choice(len(X_source_red), size=init_size, replace=False))
    
    gmm_core = fit_gmm(X_source_red[list(core_idx)], n_components=n_clusters_safe, random_state=random_state)
    prev_kl = kl_divergence(gmm_target, gmm_core, X_target_red)
    
    # 6. 迭代优化
    print(f"\n[Step 4] 开始 Bandit 迭代 (Target Size: {len(X_target_red)})...")
    no_improve = 0
    
    for it in range(1, 1501):
        # Thompson Sampling
        sampled_lambda = np.random.beta(alpha, beta)
        chosen_arm = np.argmax(sampled_lambda)
        
        # 动态步长
        rel_change = np.clip(abs(prev_kl) / (prev_kl + 1e-9), 0.1, 5.0)
        add_size = int(base_batch * rel_change)
        
        # Add Samples
        if len(clusters[chosen_arm]) > 0:
            candidates = np.random.choice(
                clusters[chosen_arm],
                size=min(add_size, len(clusters[chosen_arm])),
                replace=False
            )
            core_idx.update(candidates)
            
        # Pruning & Update
        try:
            # 剪枝逻辑
            if len(cluster_gmms_target) > 0:
                log_pa = gmm_target.score_samples(X_source_red[list(core_idx)]) + \
                         0.2 * weighted_log_prob_cluster_gmms(cluster_gmms_target, valid_weights, X_source_red[list(core_idx)])
            else:
                log_pa = gmm_target.score_samples(X_source_red[list(core_idx)])
                
            log_pw = gmm_source.score_samples(X_source_red[list(core_idx)])
            score = log_pw - log_pa
            threshold = np.mean(score) + 0.5 * np.std(score) # 放宽一点剪枝条件
            
            remove_ids = np.array(list(core_idx))[score > threshold]
            for rid in remove_ids: core_idx.discard(int(rid))
            
            # Recalculate KL
            gmm_core = fit_gmm(X_source_red[list(core_idx)], n_components=n_clusters_safe, random_state=random_state)
            new_kl = kl_divergence(gmm_target, gmm_core, X_target_red)
            
            # Update Bandit
            reward = 1 if (prev_kl - new_kl) > 0 else 0
            alpha[chosen_arm] += reward
            beta[chosen_arm] += (1 - reward)
            
            if verbose:
                print(f"  Iter {it:02d} | KL: {new_kl:.4f} (Δ {prev_kl-new_kl:.4f}) | Size: {len(core_idx)}")
            
            if abs(new_kl - prev_kl) < tol:
                no_improve += 1
            else:
                no_improve = 0
                
            if no_improve >= max_no_improve:
                print("  -> 收敛提前停止。")
                break
            prev_kl = new_kl
            
        except Exception as e:
            print(f"  Iter {it} Warning: {e}")
            continue

    # 7. 保存
    print(f"\n[Step 5] 筛选完成！Core Subset: {len(core_idx)}/{len(df_source)}")
    df_core = df_source.iloc[list(core_idx)].reset_index(drop=True)
    df_core.to_csv(output_csv, index=False)
    print(f"  -> 已保存至: {output_csv}")

# ==========================================
# 入口
# ==========================================
if __name__ == "__main__":
    # 配置你的路径
    TARGET_FILE = "./drug-target/CASF2016_graph_valid.csv"
    SOURCE_FILE = "./drug-target/Davis.csv"
    OUTPUT_FILE = "./drug-target/davis_core_subset.csv"
    
    try:
        generate_core_subset_dta(
            target_csv=TARGET_FILE,
            source_csv=SOURCE_FILE,
            output_csv=OUTPUT_FILE,
            fingerprint_bits=1024, # 提高指纹分辨率
            pca_dim=16,            # 提高维度
            n_clusters=15,         # 增加聚类
            base_batch=300
        )
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")