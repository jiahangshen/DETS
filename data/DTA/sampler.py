import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm
import math

# [关键] 引入你 DTA 代码中的 collate_fn
# 假设主程序叫 baseline_dta_train.py，为了避免循环引用，
# 建议把 collate_fn 放到一个单独的 utils.py 或 dataset.py 中。
# 这里我们假设 collate_fn 作为参数传入，或者用户在外部已经定义好。

class CoresetSamplerDTA:
    """
    Adapter for Drug-Target Affinity (Graph + Sequence) Coreset Selection.
    """
    def __init__(self, model, source_dataset, target_dataset, device, collate_fn, batch_size=1024, num_workers=4):
        self.model = model
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset 
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn # [适配] 必须传入 collate_fn
        
    def _get_inference_loader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, 
                          collate_fn=self.collate_fn, # [适配] 使用动态填充
                          num_workers=self.num_workers, pin_memory=True)

    def _extract_metrics(self, dataset, desc="Extracting Info"):
        self.model.eval()
        loader = self._get_inference_loader(dataset)
        
        all_z, all_y, all_preds, all_uncs = [], [], [], []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                # [适配] 解包 DTA 数据 (graph, input_ids, attn_mask, labels, idx)
                graphs, input_ids, attn_mask, labels, _ = batch
                graphs = graphs.to(self.device)
                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)
                labels = labels.to(self.device)
                
                # [适配] 调用模型 (需修改模型以返回 features)
                # 假设模型实现了 forward_with_features(graph, input_ids, attn_mask)
                pred, z = self.model.forward_with_features(graphs, input_ids, attn_mask)
                
                # Baseline DTA 没有 Uncertainty，用 L1 Error 近似
                unc = torch.abs(pred.view(-1) - labels.view(-1))
                
                all_z.append(z.cpu())
                all_y.append(labels.view(-1).cpu())
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
            k_ucb = 1.0
            norm_err = (src_err - src_err.mean()) / (src_err.std() + 1e-8)
            norm_unc = (src_data["unc"] - src_data["unc"].mean()) / (src_data["unc"].std() + 1e-8)
            scores = norm_err + k_ucb * norm_unc
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
            return self.select("K-Means", ratio)

        # --- Cross-Domain ---
        if method in ["Glister", "Influence"]:
            print("   Scanning Target...")
            tgt_data = self._extract_metrics(self.target_dataset, desc="Scanning Target")
            tgt_err = (tgt_data["y"] - tgt_data["pred"]).reshape(-1, 1)
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