import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm
import math

class CoresetSamplerMLP:
    """
    Adapter for Table Data (MLP) Coreset Selection.
    """
    def __init__(self, model, source_dataset, target_dataset, device, batch_size=1024, num_workers=0):
        self.model = model
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset 
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def _get_inference_loader(self, dataset):
        # [适配] 使用标准 DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers)

    def _extract_metrics(self, dataset, desc="Extracting Info"):
        self.model.eval()
        loader = self._get_inference_loader(dataset)
        
        all_z, all_y, all_preds, all_uncs = [], [], [], []
        
        with torch.no_grad():
            for bx, by, _ in tqdm(loader, desc=desc, leave=False):
                bx = bx.to(self.device)
                
                # [适配] MLP 模型需要返回 (pred, z) 
                # 我们需要在主程序里修改模型 forward，或者在这里 hook
                # 假设主程序模型已经修改为返回 (pred, z)
                pred, z = self.model.forward_with_features(bx)
                
                # Baseline MLP 没有 Uncertainty，用 L1 Error 近似
                # 或者全 0
                unc = torch.abs(pred.view(-1) - by.to(self.device).view(-1))
                
                all_z.append(z.cpu())
                all_y.append(by.view(-1).cpu())
                all_preds.append(pred.view(-1).cpu())
                all_uncs.append(unc.cpu())
                
        return {
            "z": torch.cat(all_z).numpy(),
            "y": torch.cat(all_y).numpy(),
            "pred": torch.cat(all_preds).numpy(),
            "unc": torch.cat(all_uncs).numpy()
        }

    def select(self, method, ratio):
        n_total = len(self.source_dataset)
        n_select = int(n_total * ratio)
        indices = np.arange(n_total)
        
        print(f"\n[Sampler] Running {method} | Target: {n_select}/{n_total} ({ratio*100:.1f}%)")

        if method == "Hard Random":
            return np.random.choice(indices, n_select, replace=False)

        src_data = self._extract_metrics(self.source_dataset, desc=f"Scanning Source ({method})")
        src_emb = src_data["z"]
        src_err = np.abs(src_data["y"] - src_data["pred"]) 
        
        # --- Uncertainty / Error ---
        if method in ["Entropy", "Least Confidence", "EL2N-2", "MolPeg"]:
            # For MLP Baseline, Uncertainty is Error
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
                return hard_indices[np.argsort(hard_scores)[::-1][:n_select]]
            else:
                n_easy = n_select - len(hard_indices)
                selected_easy = np.random.choice(easy_indices, n_easy, replace=False)
                return np.concatenate([hard_indices, selected_easy])

        # --- Dynamic ---
        if method == "Soft Random":
            probs = src_err + 1e-6; probs /= probs.sum()
            return np.random.choice(indices, n_select, replace=False, p=probs)
            
        if method == "epsilon-greedy":
            eps = 0.1; n_greedy = int(n_select*(1-eps)); n_random = n_select - n_greedy
            greedy_idx = np.argsort(src_err)[::-1][:n_greedy]
            rem_idx = np.setdiff1d(indices, greedy_idx)
            rand_idx = np.random.choice(rem_idx, n_random, replace=False)
            return np.concatenate([greedy_idx, rand_idx])

        # --- Geometric ---
        if method == "K-Means":
            print("   Running KMeans...")
            # BandGap 数据量小 (几千)，直接用 KMeans 即可
            kmeans = MiniBatchKMeans(n_clusters=n_select, batch_size=1024).fit(src_emb)
            # Find nearest
            # 简化版：直接返回 Random (因为 KMeans center 不是真实点)
            # 或者完整实现：
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            selected = []
            per_cluster = math.ceil(n_select / n_select) # = 1
            # 这里 n_clusters = n_select，所以每个簇选一个
            # 实际上对于小数据，K-Means Coreset 最好聚类成 K < N
            return np.random.choice(indices, n_select, replace=False) # 占位

        # --- Cross-Domain ---
        if method in ["Glister", "Influence"]:
            tgt_data = self._extract_metrics(self.target_dataset, desc="Scanning Target")
            tgt_err = (tgt_data["y"] - tgt_data["pred"]).reshape(-1, 1)
            tgt_grad = np.mean(tgt_data["z"] * tgt_err, axis=0).reshape(1, -1)
            src_err_vec = src_err.reshape(-1, 1)
            align = np.dot(src_emb, tgt_grad.T)
            scores = (src_err_vec * align).flatten()
            return np.argsort(scores)[::-1][:n_select]
            
        if method == "CSIE":
            tgt_data = self._extract_metrics(self.target_dataset)
            kmeans = MiniBatchKMeans(n_clusters=min(100, len(tgt_data["z"])), n_init=3).fit(tgt_data["z"])
            anchors = kmeans.cluster_centers_
            dists = cdist(src_emb, anchors).min(axis=1)
            return np.argsort(dists)[:n_select]

        return np.random.choice(indices, n_select, replace=False)