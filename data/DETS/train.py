import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from dataset import ATCTDataset
from model import GNNRegressor
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def train_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        # model.forward() 现在返回 (mu, z, g)
        pred, z_proj, mol_feat = model(batch) 
        y = batch.y.view(-1).to(device)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def evaluate(model, loader, device):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # model.forward() 现在返回 (mu, z, g)
            pred, z_proj, mol_feat = model(batch) 
            y = batch.y.view(-1).cpu().numpy()
            preds.extend(pred.cpu().numpy())
            ys.extend(y)
    mae = mean_absolute_error(ys, preds)
    mse = mean_squared_error(ys, preds)
    rmse = np.sqrt(mse)
    return mae, rmse


def main(args):
    dataset = ATCTDataset(args.csv)
    
    # --- 修复代码：显式创建索引列表以避免 PyG 内部冲突 ---
    # all_indices 必须是一个 list，以避免 torch_geometric 内部的 self.indices() 属性冲突
    train_val_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=(0.1/0.9), random_state=42) # 10% of remaining 90%
    # --- 修复代码结束 ---

    train_dataset = dataset.index_select(train_idx)
    val_dataset = dataset.index_select(val_idx)
    test_dataset = dataset.index_select(test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    sample = dataset[0]
    model = GNNRegressor(
        num_node_features=sample.x.shape[1],
        num_edge_features=sample.edge_attr.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, args.device)
        val_mae, val_rmse = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val MAE {val_mae:.4f} | Val RMSE {val_rmse:.4f}")

        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), "new_df_core.pt")
            patience_counter = 0  # reset
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load("new_df_core.pt"))
    test_mae, test_rmse = evaluate(model, test_loader, args.device)
    print(f"Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="enthalpy/wudily_cho.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-7)
    parser.add_argument("--patience", type=int, default=50)  
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)