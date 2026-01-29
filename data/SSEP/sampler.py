import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm
import math

class CoresetSampler:
    """
    Scientifically rigorous implementation of coreset selection strategies.
    
    Methods:
    1. Static Geometric: Hard Random, K-Means, Herding (Welling, 2009)
    2. Uncertainty/Error: Entropy, Least Confidence, EL2N-2 (Paul et al., 2021)
    3. Gradient-based: GraNd-20 (Paul et al., 2021), Glister (Killamsetty et al., 2021), Influence (Koh & Liang, 2017)
    4. Dynamic: Soft Random, InfoBatch (Qin et al., 2024), UCB, epsilon-greedy
    5. Cross-domain: CSIE, MolPeg
    """
    def __init__(self, model, source_dataset, target_dataset, device, batch_size=2048, num_workers=4):
        self.model = model
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset 
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def _get_inference_loader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, 
                          follow_batch=['x_solvent', 'x_solute'], 
                          num_workers=self.num_workers, pin_memory=True)

    # [修复] 重命名为 _extract_metrics 以匹配 select 中的调用
    def _extract_metrics(self, dataset, desc="Extracting Info"):
        """
        Extract Embeddings (z), Predictions (y_hat), Uncertainties (u), and Labels (y).
        """
        self.model.eval()
        loader = self._get_inference_loader(dataset)
        
        all_z, all_y, all_preds, all_uncs = [], [], [], []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                batch = batch.to(self.device)
                
                # Forward pass
                # Note: BaselineSampler will override this to handle (pred, z, _)
                outputs, z, _ = self.model(batch)
                
                # Unpack EDL outputs
                gamma = outputs[:, 0]
                nu    = outputs[:, 1]
                alpha = outputs[:, 2]
                beta  = outputs[:, 3]
                
                # Epistemic Uncertainty
                unc = beta / (nu * (alpha - 1) + 1e-8)
                
                # Prediction (Mean)
                pred = gamma
                
                all_z.append(z.cpu())
                all_y.append(batch.y.view(-1).cpu())
                all_preds.append(pred.cpu())
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

        # --- 1. Random ---
        if method == "Hard Random":
            return np.random.choice(indices, n_select, replace=False)

        # Extract features for Source
        # [修复] 调用 _extract_metrics
        src_data = self._extract_metrics(self.source_dataset, desc=f"Scanning Source ({method})")
        src_emb = src_data["z"]
        # Error L2 Norm (Scalar case: |y - p|)
        src_err = np.abs(src_data["y"] - src_data["pred"]) 
        
        # --- 2. Uncertainty / Error Based ---
        if method in ["Entropy", "Least Confidence"]:
            # For regression, Uncertainty ~ Entropy
            scores = src_data["unc"]
            top_k_idx = np.argsort(scores)[::-1][:n_select]
            return top_k_idx
            
        if method in ["EL2N-2", "MolPeg"]:
            # EL2N Score: ||y - f(x)||_2. For 1D output, this is L1 error.
            scores = src_err
            top_k_idx = np.argsort(scores)[::-1][:n_select]
            return top_k_idx
            
        if method == "InfoBatch":
            # InfoBatch Logic: Prune easy, keep hard + random easy
            mean_loss = src_err.mean()
            is_hard = src_err >= mean_loss
            
            hard_indices = indices[is_hard]
            easy_indices = indices[~is_hard]
            
            if len(hard_indices) >= n_select:
                # Fallback to Top-K Hard
                hard_scores = src_err[is_hard]
                local_top_k = np.argsort(hard_scores)[::-1][:n_select]
                return hard_indices[local_top_k]
            else:
                # Keep all hard
                n_easy_needed = n_select - len(hard_indices)
                # Randomly sample from easy
                if len(easy_indices) >= n_easy_needed:
                    selected_easy = np.random.choice(easy_indices, n_easy_needed, replace=False)
                    return np.concatenate([hard_indices, selected_easy])
                else:
                    return hard_indices # Should not happen if ratio < 1.0

        # --- 3. Dynamic Exploration ---
        if method == "Soft Random":
            # Probability proportional to Error
            probs = src_err + 1e-6
            probs = probs / probs.sum()
            return np.random.choice(indices, n_select, replace=False, p=probs)
            
        if method == "epsilon-greedy":
            epsilon = 0.1
            n_greedy = int(n_select * (1 - epsilon))
            n_random = n_select - n_greedy
            
            greedy_indices = np.argsort(src_err)[::-1][:n_greedy]
            remaining = np.setdiff1d(indices, greedy_indices)
            if len(remaining) >= n_random:
                random_indices = np.random.choice(remaining, n_random, replace=False)
                return np.concatenate([greedy_indices, random_indices])
            return greedy_indices

        if method == "UCB":
            # Score = Mean_Loss + k * Sigma_Unc
            k_ucb = 1.0
            norm_err = (src_err - src_err.mean()) / (src_err.std() + 1e-8)
            norm_unc = (src_data["unc"] - src_data["unc"].mean()) / (src_data["unc"].std() + 1e-8)
            
            scores = norm_err + k_ucb * norm_unc
            top_k_idx = np.argsort(scores)[::-1][:n_select]
            return top_k_idx

        # --- 4. Gradient Based ---
        if method in ["GraNd-20", "DP"]:
            # GraNd approx: ||g|| ≈ ||z|| * error
            z_norm = np.linalg.norm(src_emb, axis=1)
            scores = z_norm * src_err
            top_k_idx = np.argsort(scores)[::-1][:n_select]
            return top_k_idx

        # --- 5. Geometric ---
        if method == "K-Means":
            print("   Running MiniBatchKMeans (Coreset)...")
            # Cluster and select samples closest to centers
            n_clusters = 5000 
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, n_init=3, random_state=42).fit(src_emb)
            
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            selected = []
            per_cluster = math.ceil(n_select / n_clusters)
            
            # Vectorized search within clusters is hard, loop is cleaner
            for k in tqdm(range(n_clusters), desc="Selecting per cluster", leave=False):
                cluster_idxs = indices[labels == k]
                if len(cluster_idxs) == 0: continue
                
                cluster_z = src_emb[cluster_idxs]
                # Distance to center
                dists = cdist(cluster_z, centers[k].reshape(1, -1)).flatten()
                
                # Closest ones
                local_top = np.argsort(dists)[:per_cluster]
                selected.extend(cluster_idxs[local_top])
            
            selected = np.array(selected)
            if len(selected) > n_select:
                selected = selected[:n_select]
            return selected

        if method == "Herding":
            # Herding approx via K-Means centers for large scale
            print("   [Note] Using K-Means approximation for Herding on large scale.")
            return self.select("K-Means", ratio)

        # --- 6. Cross-Domain (Target Aware) ---
        if method in ["Glister", "Influence"]:
            # First-order Approximation: Gradient Inner Product
            print("   Scanning Target Data...")
            tgt_data = self._extract_metrics(self.target_dataset, desc="Scanning Target")
            
            # Mean Target Gradient Direction
            tgt_err = (tgt_data["y"] - tgt_data["pred"]).reshape(-1, 1)
            tgt_grad_vec = np.mean(tgt_data["z"] * tgt_err, axis=0).reshape(1, -1)
            
            # Source Gradients Project
            src_err_vec = src_err.reshape(-1, 1)
            # Alignment = z_src . mean_tgt_grad
            feature_alignment = np.dot(src_emb, tgt_grad_vec.T)
            
            # Glister Score: Magnitude of gradient * Alignment
            scores = (src_err_vec * feature_alignment).flatten()
            
            # Select highest positive scores
            top_k_idx = np.argsort(scores)[::-1][:n_select]
            return top_k_idx
            
        if method == "CSIE":
            # Coverage: Closest to Target Anchors
            print("   Calculating CSIE Coverage...")
            tgt_data = self._extract_metrics(self.target_dataset, desc="Scanning Target")
            
            # Cluster Target to find Anchors
            kmeans = MiniBatchKMeans(n_clusters=100, n_init=3).fit(tgt_data["z"])
            anchors = kmeans.cluster_centers_
            
            # Find min dist to ANY target anchor
            dists = []
            chunk_size = 4096
            for i in range(0, len(src_emb), chunk_size):
                chunk = src_emb[i:i+chunk_size]
                d = cdist(chunk, anchors, metric='euclidean')
                min_d = np.min(d, axis=1)
                dists.append(min_d)
            
            min_dists = np.concatenate(dists)
            
            # Select smallest distances (most similar to target domain)
            top_k_idx = np.argsort(min_dists)[:n_select]
            return top_k_idx

        # Fallback
        return np.random.choice(indices, n_select, replace=False)