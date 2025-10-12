from NPZGraphDataset import NPZGraphDataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

# 1) 读取 manifest
manifest_path = Path("./graph_npz/manifest.csv")
root_dir = Path("./")  # 若 file_path 是相对路径，作为公共前缀

manifest = pd.read_csv(manifest_path)

# 2) 分层标签：把连续的 Tg 分成若干箱做 stratify（若标签太少会自动退化为非分层）
if "label" in manifest:
    # 这里用 10 个分箱；你也可以改成 5 或 20
    try:
        bins = pd.qcut(manifest["label"], q=10, duplicates="drop")
        stratify_labels = bins.astype(str)
    except Exception:
        stratify_labels = None
else:
    stratify_labels = None

# 3) 先切出测试集（10%），再从剩余里切验证集（10%）
train_val_df, test_df = train_test_split(
    manifest,
    test_size=0.10,
    random_state=42,
    shuffle=True,
    stratify=stratify_labels if stratify_labels is not None else None,
)
# 对 train_val 再按 10% 切出 val => 0.9 * 0.1 = 9%（接近 80/10/10）
if stratify_labels is not None:
    train_val_bins = pd.qcut(train_val_df["label"], q=10, duplicates="drop")
    stratify_trainval = train_val_bins.astype(str)
else:
    stratify_trainval = None

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.10/0.90,   # 约 0.111..., 使得整体 ~80/10/10
    random_state=42,
    shuffle=True,
    stratify=stratify_trainval if stratify_trainval is not None else None,
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# 4) 构建三个 Dataset（验证/测试共用训练集的标准化参数）
train_dataset = NPZGraphDataset(
    manifest=train_df,
    root=root_dir,
    separate_pos=True,      # (x,y,z) 放在 data.pos
    feature_cols=(0,1,2,3),
    coord_cols=(4,5,6),
    standardize_y=True,     # 仅用训练集统计均值方差
)

val_dataset = NPZGraphDataset(
    manifest=val_df,
    root=root_dir,
    separate_pos=True,
    feature_cols=(0,1,2,3),
    coord_cols=(4,5,6),
    standardize_y=True,     # 先开着，随后覆盖为 train 的均值方差
)
# 覆盖验证集 y 的标准化为“训练集统计”，防止数据泄漏
val_dataset._y_mean = train_dataset.y_mean
val_dataset._y_std  = train_dataset.y_std

test_dataset = NPZGraphDataset(
    manifest=test_df,
    root=root_dir,
    separate_pos=True,
    feature_cols=(0,1,2,3),
    coord_cols=(4,5,6),
    standardize_y=True,
)
# 覆盖测试集 y 的标准化为“训练集统计”，防止数据泄漏
test_dataset._y_mean = train_dataset.y_mean
test_dataset._y_std  = train_dataset.y_std
print(f"y_mean(train)={train_dataset.y_mean:.4f}, y_std(train)={train_dataset.y_std:.4f}")

# 5) DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, pin_memory=True)

# 6) 简单检查一个 batch
batch = next(iter(train_loader))
print(batch)
print("x:", None if batch.x is None else batch.x.shape)
print("pos:", None if batch.pos is None else batch.pos.shape)
print("edge_index:", batch.edge_index.shape)
print("edge_attr:", batch.edge_attr.shape)
print("y:", batch.y.shape)

# ====== 评估时如需把预测还原为物理单位（K）：=====
# pred_real = pred_norm * train_dataset.y_std + train_dataset.y_mean
# y_real    = y_norm    * train_dataset.y_std + train_dataset.y_mean

import os, logging, datetime, json
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from BestModel.ConvModel import GatedGCNModel

# ========= 日志配置 =========
run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = Path("./logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"train_{run_time}.log"

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
# 控制台
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# 文件
fh = logging.FileHandler(log_path)
fh.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch.setFormatter(fmt); fh.setFormatter(fmt)
logger.addHandler(ch); logger.addHandler(fh)

# ========= 设备选择 =========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# ========= 模型 =========
model = GatedGCNModel(
    layers_in_conv=3,
    channels=64,
    use_nodetype_coeffs=False,
    num_node_types=0,
    num_edge_types=4,
    use_jumping_knowledge=False,
    use_bias_for_update=True,
    use_dropout=True,
    num_convs=3,
    num_fc_layers=3,
    neighbors_aggr='add',
    dropout_p=0.1,
    num_targets=1,
    geom_K=16,          # <--- 新增：RBF 基数
    geom_rmax=4.0,      # <--- 新增：RBF 最大半径
    concat_original_edge=True,  # 与原 4 维边特征拼接
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # 可选

# ========= 反标准化所需的均值/方差（来自训练集）=========
y_mean = torch.tensor(train_dataset.y_mean, dtype=torch.float32, device=device)
y_std  = torch.tensor(train_dataset.y_std,  dtype=torch.float32, device=device)

def denorm(t):  # t: [B,1] 标准化空间 -> 物理单位 K
    return t * y_std + y_mean

# ========= 训练循环 =========
epochs = 50
best_val_rmse = float("inf")
ckpt_dir = Path("./checkpoints"); ckpt_dir.mkdir(exist_ok=True, parents=True)
logger.info(json.dumps({
    "epochs": epochs, "batch_size": train_loader.batch_size,
    "optimizer": "AdamW", "lr": optimizer.param_groups[0]["lr"],
    "weight_decay": optimizer.param_groups[0]["weight_decay"],
    "scheduler": "CosineAnnealingLR(T_max=50)"
}, ensure_ascii=False))

for epoch in range(1, epochs+1):
    # ---- Train ----
    model.train()
    total_loss = 0.0; n_tr = 0
    for batch in train_loader:
        batch = batch.to(device)
        pred = model(batch)                       # [B,1] 标准化空间
        y = batch.y.view(-1,1).float().to(device) # [B,1] 标准化空间
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        n_tr += batch.num_graphs

    if scheduler is not None:
        scheduler.step()

    train_loss = total_loss / max(n_tr, 1)

    # ---- Eval (以 K 为单位计算指标) ----
    model.eval()
    mae = 0.0; rmse = 0.0; n_val = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred_norm = model(batch)                       # [B,1] 标准化
            y_norm = batch.y.view(-1,1).float().to(device)
            # 反标准化到 K
            pred_k = denorm(pred_norm)
            y_k = denorm(y_norm)
            diff = pred_k - y_k
            mae  += diff.abs().sum().item()
            rmse += (diff**2).sum().item()
            n_val += batch.num_graphs
    mae  = mae / max(n_val, 1)           # K
    rmse = (rmse / max(n_val, 1)) ** 0.5 # K

    lr_now = optimizer.param_groups[0]["lr"]
    logger.info(f"Epoch {epoch:03d} | TrainLoss(norm) {train_loss:.4f} | "
                f"Val MAE(K) {mae:.3f} | Val RMSE(K) {rmse:.3f} | LR {lr_now:.6f}")

    # 保存最优
    if rmse < best_val_rmse:
        best_val_rmse = rmse
        ckpt_path = ckpt_dir / f"best_rmse_{best_val_rmse:.3f}K_ep{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "y_mean": float(y_mean.item()),
            "y_std": float(y_std.item()),
            "config": {
                "layers_in_conv": 3, "channels": 64, "num_convs": 3,
                "dropout_p": 0.1, "neighbors_aggr": "add"
            }
        }, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

# ====== 测试集评估（K）======
model.eval()
mae_t = 0.0; rmse_t = 0.0; n_te = 0
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred_k = denorm(model(batch))
        y_k = denorm(batch.y.view(-1,1).float().to(device))
        diff = pred_k - y_k
        mae_t  += diff.abs().sum().item()
        rmse_t += (diff**2).sum().item()
        n_te   += batch.num_graphs
mae_t  /= max(n_te, 1)
rmse_t = (rmse_t / max(n_te, 1)) ** 0.5
logger.info(f"[TEST] MAE(K) {mae_t:.3f} | RMSE(K) {rmse_t:.3f}")