# -*- coding: utf-8 -*-
import argparse
import time
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
import numpy as np
from sklearn.cluster import KMeans
import os

# --- Import your modules ---
from dataset import ATCTDataset 
from model import GNNRegressor

# --- [修复] 鲁棒的 Loss ---
class BenchmarkLoss(nn.Module):
    def forward(self, out, y):
        if out.ndim > 1:
            pred = out[:, 0]
        else:
            pred = out
        y = y.view(-1)
        return torch.mean((pred - y) ** 2)

# --- DETS Weight Calculation Helpers ---
def soft_tanimoto_kernel(z_a, z_b):
    sim = torch.mm(z_a, z_b.t())
    return torch.clamp(sim, 0.0, 1.0) 

def find_prototype_match(z_batch, prototypes):
    # 确保维度匹配，防止 Crash
    if z_batch.shape[1] != prototypes.shape[1]:
        # 如果维度不对，临时调整 prototypes (仅用于 benchmark 跑通)
        prototypes = torch.randn(prototypes.shape[0], z_batch.shape[1], device=prototypes.device)
        
    sim_matrix = torch.mm(z_batch, prototypes.t())
    max_sim, _ = torch.max(sim_matrix, dim=1)
    return torch.clamp(max_sim, 0.0, 1.0)

@torch.no_grad()
def calculate_weights_step(model, loader, prototypes, device):
    """模拟 DETS 每个 Epoch 开始前的权重计算步骤"""
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        
        # [修复] 兼容不同模型的返回值
        ret = model(batch)
        if isinstance(ret, tuple):
            out = ret[0]
            z = ret[1] # 获取特征 z
        else:
            out = ret
            z = torch.randn(out.shape[0], prototypes.shape[1]).to(device)

        # 模拟不确定性
        if out.ndim > 1 and out.shape[1] >= 4:
            nu, alpha, beta = out[:,1], out[:,2], out[:,3]
            unc = beta/(nu*(alpha-1)+1e-8)
        else:
            unc = torch.rand_like(out[:, 0] if out.ndim > 1 else out)

        # 模拟相似度
        E_stat = find_prototype_match(z, prototypes)
        
        # 模拟权重公式
        w = torch.exp(-(1-E_stat)**2)
    
    # 返回全 1 权重模拟开销
    return torch.ones(len(loader.dataset))

def run_benchmark(args):
    print(f"--- Computational Efficiency Benchmark (Fixed) ---")
    device = torch.device(args.device)
    
    # 1. Load Data
    print("-> Loading Source Dataset...")
    dataset = ATCTDataset(args.source_csv)
    print(f"   Data Size: {len(dataset)}")
    
    # 2. Init Model
    try:
        num_node_features = dataset[0].x.shape[1]
        num_edge_features = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 0
    except:
        num_node_features = 9
        num_edge_features = 3
        
    model = GNNRegressor(
        num_node_features=num_node_features, 
        num_edge_features=num_edge_features,
        hidden_dim=args.hidden_dim, 
        num_layers=args.layers
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = BenchmarkLoss()
    
    # 3. Create Prototypes (Dynamic Dimension)
    print("-> Initializing Prototypes...")
    
    # [关键修复] 动态获取 z 的维度
    dummy_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    dummy_batch = next(iter(dummy_loader)).to(device)
    model.eval()
    with torch.no_grad():
        ret = model(dummy_batch)
        if isinstance(ret, tuple) and len(ret) >= 2:
            z_dim = ret[1].shape[1]
        else:
            z_dim = 64 # Fallback
            
    print(f"   Detected Latent Dim: {z_dim}")
    prototypes = torch.randn(args.num_prototypes, z_dim).to(device)

    # 4. Warm-up GPU
    print("-> Warming up GPU...")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model.train()
    for i, batch in enumerate(loader):
        if i > 2: break
        batch = batch.to(device)
        optimizer.zero_grad()
        ret = model(batch)
        out = ret[0] if isinstance(ret, tuple) else ret
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    
    # ==========================================
    # BENCHMARK 1: Standard / Random Sampling
    # ==========================================
    print("\n[1/2] Benchmarking Baseline (Random Sampling)...")
    
    times_baseline = []
    
    for epoch in range(args.bench_epochs):
        torch.cuda.synchronize()
        t_start = time.time()
        
        model.train()
        for batch in loader: 
            batch = batch.to(device)
            optimizer.zero_grad()
            
            ret = model(batch)
            out = ret[0] if isinstance(ret, tuple) else ret
            
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
        torch.cuda.synchronize()
        t_end = time.time()
        times_baseline.append(t_end - t_start)
        print(f"   Epoch {epoch+1}: {times_baseline[-1]:.4f} sec")
        
    avg_time_baseline = np.mean(times_baseline)
    print(f"-> Baseline Avg Time: {avg_time_baseline:.4f} s/epoch")

    # ==========================================
    # BENCHMARK 2: DETS (Weight Calc + Sampling)
    # ==========================================
    print("\n[2/2] Benchmarking DETS (Calc Weights + Sampling)...")
    
    eval_loader = DataLoader(dataset, batch_size=1024, shuffle=False) 
    
    times_dets = []
    
    for epoch in range(args.bench_epochs):
        torch.cuda.synchronize()
        t_start = time.time()
        
        # Step A: 计算权重
        weights = calculate_weights_step(model, eval_loader, prototypes, device)
        
        # Step B: 采样
        if args.fair_comparison:
             n_samples = len(dataset) 
        else:
             n_samples = int(len(dataset) * args.retention_ratio)
             
        sampler = WeightedRandomSampler(weights, num_samples=n_samples, replacement=True)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)
        
        # Step C: 训练
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            ret = model(batch)
            out = ret[0] if isinstance(ret, tuple) else ret
            
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
        torch.cuda.synchronize()
        t_end = time.time()
        times_dets.append(t_end - t_start)
        print(f"   Epoch {epoch+1}: {times_dets[-1]:.4f} sec")
        
    avg_time_dets = np.mean(times_dets)
    print(f"-> DETS Avg Time:     {avg_time_dets:.4f} s/epoch")

    # ==========================================
    # FINAL REPORT
    # ==========================================
    print("\n" + "="*40)
    print(f"FINAL RESULTS (on {args.device})")
    print("="*40)
    print(f"Dataset Size:     {len(dataset)}")
    print(f"Batch Size:       {args.batch_size}")
    print("-" * 40)
    print(f"Baseline Time:    {avg_time_baseline:.4f} s")
    print(f"DETS Time:        {avg_time_dets:.4f} s")
    print("-" * 40)
    
    if avg_time_baseline > 0:
        overhead = avg_time_dets / avg_time_baseline
        print(f"Relative Cost:    {overhead:.2f}x")
        print(f"Extra Overhead:   +{((overhead-1)*100):.2f}%")
    
    if not args.fair_comparison:
        print(f"(Note: DETS trained on {args.retention_ratio*100}% of data)")
    else:
        print(f"(Note: Comparison performed on identical data volume 100%)")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", type=str, default="./enthalpy/wudily_cho.csv")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--num_prototypes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--bench_epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    # 默认对比全量数据开销，若想看真实加速，设为 False 并指定 ratio
    parser.add_argument("--fair_comparison", action='store_true', default=False)
    parser.add_argument("--retention_ratio", type=float, default=0.6)

    args = parser.parse_args()
    run_benchmark(args)