import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm
import math

class CoresetSamplerGNN:
    """
    Adapter for GNN Coreset Selection.
    """
    def __init__(self, model, source_dataset, target_dataset, device, batch_size=1024, num_workers=0):
        self.model = model
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset 
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def _get_inference_loader(self, dataset):
        # PyG DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=True)

    def _extract_metrics(self, dataset, desc="Extracting Info"):
        self.model.eval()
        loader = self._get_inference_loader(dataset)
        
        all_z, all_y, all_preds, all_uncs = [], [], [], []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                batch = batch.to(self.device)
                
                # [适配] 调用模型 (需修改模型以返回 features)
                pred, z = self.model.forward_with_features(batch)
                
                # Baseline GNN 没有 Uncertainty，用 L1 Error 近似
                # 这里假设 y 已经在 dataset 中，且未归一化
                # 但为了采样（排序），只要 pred 和 y 是同一空间即可
                # 这里我们提取原始预测值和原始 y
                # 如果模型输出是归一化的，这里计算的 Error 会有尺度问题
                # 但只要所有样本一致，排序就不受影响
                
                # 注意：DETS 中使用了 y_std 反归一化
                # 这里为了简单，直接用模型输出 vs batch.y (假设已对齐或都未对齐)
                # 你的 baseline 代码里 train_epoch 用的 mse_loss(pred, y_norm)
                # 所以模型输出的是 pred (normalized)
                # 为了计算真实 error，我们需要反归一化，或者把 y 也归一化
                # 这里选择提取 raw y，并在外部处理，或者简单地用 normalized error
                
                # 简化：直接存，后续在 select 里处理
                
                all_z.append(z.cpu())
                all_y.append(batch.y.view(-1).cpu())
                all_preds.append(pred.view(-1).cpu())
                
        return {
            "z": torch.cat(all_z).numpy(),
            "y": torch.cat(all_y).numpy(),
            "pred": torch.cat(all_preds).numpy()
        }

    def select(self, method, ratio, y_mean=None, y_std=None):
        n_total = len(self.source_dataset)
        n_select = int(n_total * ratio)
        indices = np.arange(n_total)
        
        print(f"\n[Sampler] Running {method} | Target: {n_select}/{n_total} ({ratio*100:.1f}%)")

        if method == "Hard Random":
            return np.random.choice(indices, n_select, replace=False)

        src_data = self._extract_metrics(self.source_dataset, desc=f"Scanning Source ({method})")
        src_emb = src_data["z"]
        
        # 计算误差
        # 如果提供了 mean/std，先反归一化预测值
        if y_mean is not None and y_std is not None:
            pred_real = src_data["pred"] * y_std.item() + y_mean.item()
            src_err = np.abs(src_data["y"] - pred_real)
        else:
            # 假设模型输出和 y 已经对齐
            src_err = np.abs(src_data["y"] - src_data["pred"]) 
        
        # --- Uncertainty / Error ---
        if method in ["Entropy", "Least Confidence", "EL2N-2", "MolPeg"]:
            scores = src_err
            top_k_idx = np.argsort(scores)[::-1][:n_select]
            return top_k_idx
            
        if method == "InfoBatch":
            mean_loss = src_err.mean()
            is_hard = src_err >= mean_loss
            hard_indices = indices[is_hard]
            easy_indices = indices[~is_hard]
            if len(hard_indices) >= n_select:
                hard_scores = src_err[is_hard]
                local_top_k = np.argsort(hard_scores)[::-1][:n_select]
                return hard_indices[local_top_k]
            else:
                n_easy = n_select - len(hard_indices)
                if len(easy_indices) >= n_easy:
                    selected_easy = np.random.choice(easy_indices, n_easy, replace=False)
                    return np.concatenate([hard_indices, selected_easy])
                else:
                    return hard_indices

        # --- Dynamic ---
        if method == "Soft Random":
            probs = src_err + 1e-6; probs /= probs.sum()
            return np.random.choice(indices, n_select, replace=False, p=probs)
            
        if method == "epsilon-greedy":
            eps = 0.1; n_greedy = int(n_select*(1-eps)); n_random = n_select - n_greedy
            greedy_indices = np.argsort(src_err)[::-1][:n_greedy]
            remaining = np.setdiff1d(indices, greedy_indices)
            if len(remaining) >= n_random:
                random_indices = np.random.choice(remaining, n_random, replace=False)
                return np.concatenate([greedy_indices, random_indices])
            return greedy_indices

        if method == "UCB":
            # GNN Baseline 没有 Epistemic Uncertainty，用 Feature Norm 或 Local Density 替代？
            # 简单起见，这里 UCB 退化为 EL2N (只看 Error)
            # 或者用 Feature Distance to Train Mean 作为 Uncertainty Proxy
            scores = src_err 
            top_k_idx = np.argsort(scores)[::-1][:n_select]
            return top_k_idx

        # --- Gradient ---
        if method in ["GraNd-20", "DP"]:
            z_norm = np.linalg.norm(src_emb, axis=1)
            scores = z_norm * src_err
            top_k_idx = np.argsort(scores)[::-1][:n_select]
            return top_k_idx

        # --- Geometric ---
        if method == "K-Means":
            print("   Running MiniBatchKMeans...")
            n_clusters = min(n_select, 5000)
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, n_init=3, random_state=42).fit(src_emb)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            selected = []
            per_cluster = math.ceil(n_select / n_clusters)
            for k in tqdm(range(n_clusters), desc="Clusters", leave=False):
                cluster_idxs = indices[labels == k]
                if len(cluster_idxs) == 0: continue
                cluster_z = src_emb[cluster_idxs]
                dists = cdist(cluster_z, centers[k].reshape(1, -1)).flatten()
                local_top = np.argsort(dists)[:per_cluster]
                selected.extend(cluster_idxs[local_top])
            selected = np.array(selected)
            if len(selected) > n_select: selected = selected[:n_select]
            return selected
            
        if method == "Herding":
            print("   [Note] Using K-Means approximation for Herding.")
            return self.select("K-Means", ratio, y_mean, y_std)

        # --- Cross-Domain ---
        if method in ["Glister", "Influence"]:
            print("   Scanning Target...")
            tgt_data = self._extract_metrics(self.target_dataset, desc="Scanning Target")
            
            # Target Error (需反归一化对齐)
            tgt_pred_real = tgt_data["pred"] * y_std.item() + y_mean.item()
            tgt_err = (tgt_data["y"] - tgt_pred_real).reshape(-1, 1)
            
            tgt_grad = np.mean(tgt_data["z"] * tgt_err, axis=0).reshape(1, -1)
            
            src_err_vec = src_err.reshape(-1, 1)
            align = np.dot(src_emb, tgt_grad.T)
            scores = (src_err_vec * align).flatten()
            top_k_idx = np.argsort(scores)[::-1][:n_select]
            return top_k_idx
            
        if method == "CSIE":
            print("   Calculating CSIE Coverage...")
            tgt_data = self._extract_metrics(self.target_dataset, desc="Scanning Target")
            kmeans = MiniBatchKMeans(n_clusters=100, n_init=3).fit(tgt_data["z"])
            anchors = kmeans.cluster_centers_
            dists = []
            chunk_size = 4096
            for i in range(0, len(src_emb), chunk_size):
                chunk = src_emb[i:i+chunk_size]
                d = cdist(chunk, anchors, metric='euclidean')
                min_d = np.min(d, axis=1)
                dists.append(min_d)
            min_dists = np.concatenate(dists)
            top_k_idx = np.argsort(min_dists)[:n_select]
            return top_k_idx

        return np.random.choice(indices, n_select, replace=False)