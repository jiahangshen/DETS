import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from dataset import MolProtDataset, collate_fn
from model import MolProtRegressor
from rdkit import RDLogger

# 屏蔽 RDKit 警告
RDLogger.DisableLog('rdApp.*')

# --------------------------- utils ---------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def concordance_index(y_true, y_pred):
    n = 0; n_concordant = 0; n_tied = 0
    for i in range(len(y_true)):
        for j in range(i+1, len(y_true)):
            if y_true[i] == y_true[j]: continue
            n += 1
            diff_true = y_true[i] - y_true[j]
            diff_pred = y_pred[i] - y_pred[j]
            if diff_true * diff_pred > 0: n_concordant += 1
            elif diff_pred == 0: n_tied += 1
    return (n_concordant + 0.5 * n_tied) / n if n > 0 else 0

def calc_rm2(y_true, y_pred):
    y_true, y_pred = np.array(y_true).reshape(-1,1), np.array(y_pred).reshape(-1,1)
    lr = LinearRegression().fit(y_pred, y_true)
    r2 = lr.score(y_pred, y_true)
    lr0 = LinearRegression(fit_intercept=False).fit(y_pred, y_true)
    r2_0 = lr0.score(y_pred, y_true)
    return r2 * (1 - np.sqrt(abs(r2 - r2_0)))

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for graphs, input_ids, attn_mask, labels in loader:
        graphs, input_ids, attn_mask, labels = graphs.to(device), input_ids.to(device), attn_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # [修改] 解包元组，只取预测值 out
        out, _ = model(graphs, input_ids, attn_mask)
        
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device,abs_tol=0.1):
    model.eval()
    losses = []
    preds = []
    trues = []
    for graphs, input_ids, attn_mask, labels in loader:
        graphs, input_ids, attn_mask, labels = graphs.to(device), input_ids.to(device), attn_mask.to(device), labels.to(device)
        
        # [修改] 解包元组
        out, _ = model(graphs, input_ids, attn_mask)
        
        loss = criterion(out, labels)
        losses.append(loss.item())
        preds.extend(out.cpu().numpy())
        trues.extend(labels.cpu().numpy())
        
    preds, trues = np.array(preds), np.array(trues)
    
    # 1. 基础指标
    mae = mean_absolute_error(trues, preds)
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    
    # 2. DTA 指标
    ci = concordance_index(trues, preds)
    rm2 = calc_rm2(trues, preds)
    
    # 3. [修改] 相对误差指标 (MAPE & Acc) - 引入混合容忍度逻辑
    abs_diff = np.abs(trues - preds)
    
    # Acc@10 logic: 成功标准 = 误差 < (10% 真实值) OR 误差 < abs_tol
    # 这能保证在数值较小时，只要满足绝对误差要求（例如 0.1），也算预测正确
    tol_threshold = np.maximum(0.10 * np.abs(trues), abs_tol)
    acc_10 = np.mean(abs_diff <= tol_threshold) * 100
    
    # MAPE logic: 分母设置保底值 1.0
    # 防止除以接近 0 的数导致 MAPE 爆炸，同时符合 pKd 的物理尺度意义
    mape = np.mean(abs_diff / np.maximum(np.abs(trues), 1.0)) * 100
    
    # 返回增加 mape 和 acc_10
    return np.mean(losses), mae, rmse, ci, rm2, mape, acc_10

def main():
    parser = argparse.ArgumentParser(description="CASF Transfer Learning")
    parser.add_argument("--csv", type=str, default="./drug-target/CASF2016_graph_valid.csv")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32) # 建议调大一点，既然用了动态填充
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp_hidden", type=str, default="256,128")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="DTA_checkpoints_baseline")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--pretrained", type=str, default="./DTA_checkpoints_baseline/baseline_best.pt")
    
    # [新增] 显式指定模型参数，防止 size mismatch
    # 必须与预训练时保持一致！
    parser.add_argument("--g_hidden", type=int, default=128)
    parser.add_argument("--g_layers", type=int, default=3)
    parser.add_argument("--seq_emb_dim", type=int, default=128)
    parser.add_argument("--seq_layers", type=int, default=2)
    parser.add_argument("--seq_proj_dim", type=int, default=256)
    
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    mlp_hidden = tuple(int(x) for x in args.mlp_hidden.split(","))

    # ------------------- dataset -------------------
    print(f"Loading dataset: {args.csv}")
    # ... (前面的代码)
    
    dataset = MolProtDataset(args.csv, max_len=args.max_seq_len)
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val
    
    # [核心修改] 创建一个独立的生成器，并指定种子
    # 这里的种子 (seed=42) 专门用于控制数据集划分，与模型初始化的种子解耦
    split_generator = torch.Generator().manual_seed(42)
    
    # 将 generator 传入 random_split
    train_set, val_set, test_set = random_split(
        dataset, 
        [n_train, n_val, n_test], 
        generator=split_generator  # <--- 关键点
    )

    # DataLoader 的 shuffle=True 依然依赖全局种子，或者可以在这里指定 worker_init_fn
    # 但通常只要在 main 开头 set_seed 了，这里就没问题
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # ------------------- model -------------------
    print(f"Initializing model on {args.device}...")
    model = MolProtRegressor(
        graph_in_dim=dataset.graph_in_dim,
        g_hidden=args.g_hidden, 
        g_layers=args.g_layers,
        seq_vocab_size=30, 
        seq_emb_dim=args.seq_emb_dim,
        seq_heads=4, 
        seq_layers=args.seq_layers, 
        seq_proj_dim=args.seq_proj_dim,
        mlp_hidden=mlp_hidden, 
        dropout=args.dropout
    ).to(args.device)

    # ------------------- load pretrained -------------------
    if os.path.exists(args.pretrained):
        print(f"Loading pretrained weights from {args.pretrained}")
        state = torch.load(args.pretrained, map_location=args.device)
        model.load_state_dict(state, strict=False) # strict=False 以防万一有些层不匹配
    else:
        print("Warning: No pretrained weights found! Training from scratch.")

    # ------------------- Freeze logic -------------------
    print("Freezing backbone layers...")
    for param in model.graph_encoder.parameters():
        param.requires_grad = False
    for param in model.protein_encoder.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_path = os.path.join(args.save_dir, "transfer_best.pt")
    patience_counter = 0

    print("Starting fine-tuning...")
    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        # [修改] 接收 7 个返回值
        val_loss, mae, rmse, ci, rm2, mape, acc = eval_epoch(model, val_loader, criterion, args.device)
        
        print(f"Epoch {epoch}: Loss {tr_loss:.4f} | Val MAE {mae:.4f} RMSE {rmse:.4f} CI {ci:.4f} | Acc {acc:.1f}%")

        if mae < best_val_mae:
            best_val_mae = mae
            torch.save(model.state_dict(), best_path)
            # print(f"  Best model saved! (MAE: {mae:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    # ------------------- final test -------------------
    print("\nLoading best model for final evaluation...")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=args.device))
        
        # [修改] 接收 7 个返回值
        test_loss, test_mae, test_rmse, test_ci, test_rm2, test_mape, test_acc = eval_epoch(model, test_loader, criterion, args.device)
        
        print("-" * 60)
        print(f"[TRANSFER RESULT] Test Set:")
        print(f"   MAE:  {test_mae:.4f}")
        print(f"   RMSE: {test_rmse:.4f}")
        print(f"   CI:   {test_ci:.4f}")
        print(f"   RM2:  {test_rm2:.4f}")
        print(f"   MAPE: {test_mape:.2f} %")
        print(f"   Acc:  {test_acc:.2f} %")
        print("-" * 60)
    else:
        print("Training failed or no model saved.")

if __name__=="__main__":
    main()