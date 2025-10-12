#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python TrainModel.py - Command-line training script converted from the user's notebook.
Usage example:
python TrainModel.py \
  --data_path ./graph_npz/manifest.csv \
  --root_dir ./ \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --optimizer adamw \
  --loss mse \
  --scheduler cosine \
  --seed 42 \
  --device auto \
  --log_dir ./logs \
  --checkpoint_dir ./checkpoints
"""

import argparse
import json
import datetime
import logging
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from NPZGraphDataset import NPZGraphDataset
from BestModel.ConvModel import GatedGCNModel


# ================== Utilities ==================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denorm(t, y_mean, y_std):
    return t * y_std + y_mean


# ================== Main ==================

def main():
    parser = argparse.ArgumentParser(description="Train GatedGCN model for polymer property prediction.")

    # Dataset arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to manifest CSV.")
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory for .npz files.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam"])
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "smoothl1"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    parser.add_argument("--clip_grad_norm", type=float, default=5.0)

    # Device
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # Logging
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    args = parser.parse_args()
    set_seed(args.seed)

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Logging setup
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / f"train_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    fh = logging.FileHandler(log_path)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(fmt); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)

    logger.info(json.dumps(vars(args), indent=2))

    # ========== Load and split manifest ==========
    manifest = pd.read_csv(args.data_path)
    if "label" in manifest:
        try:
            bins = pd.qcut(manifest["label"], q=10, duplicates="drop")
            stratify_labels = bins.astype(str)
        except Exception:
            stratify_labels = None
    else:
        stratify_labels = None

    train_val_df, test_df = train_test_split(
        manifest, test_size=args.test_split, random_state=args.seed, stratify=stratify_labels
    )

    if stratify_labels is not None:
        train_val_bins = pd.qcut(train_val_df["label"], q=10, duplicates="drop")
        stratify_trainval = train_val_bins.astype(str)
    else:
        stratify_trainval = None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=args.val_split / (1 - args.test_split),
        random_state=args.seed,
        stratify=stratify_trainval if stratify_trainval is not None else None,
    )

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ========== Build datasets ==========
    root_dir = Path(args.root_dir)
    train_dataset = NPZGraphDataset(
        manifest=train_df, root=root_dir,
        separate_pos=True, feature_cols=(0,1,2,3), coord_cols=(4,5,6), standardize_y=True
    )
    val_dataset = NPZGraphDataset(
        manifest=val_df, root=root_dir,
        separate_pos=True, feature_cols=(0,1,2,3), coord_cols=(4,5,6), standardize_y=True
    )
    val_dataset._y_mean, val_dataset._y_std = train_dataset.y_mean, train_dataset.y_std
    test_dataset = NPZGraphDataset(
        manifest=test_df, root=root_dir,
        separate_pos=True, feature_cols=(0,1,2,3), coord_cols=(4,5,6), standardize_y=True
    )
    test_dataset._y_mean, test_dataset._y_std = train_dataset.y_mean, train_dataset.y_std

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    logger.info(f"y_mean(train)={train_dataset.y_mean:.4f}, y_std(train)={train_dataset.y_std:.4f}")

    # ========== Model ==========
    model = GatedGCNModel(
        layers_in_conv=3, channels=64, use_nodetype_coeffs=False, num_node_types=0,
        num_edge_types=4, use_jumping_knowledge=False, use_bias_for_update=True,
        use_dropout=True, num_convs=3, num_fc_layers=3, neighbors_aggr='add',
        dropout_p=0.1, num_targets=1, geom_K=16, geom_rmax=4.0, concat_original_edge=True,
    ).to(device)

    if args.loss == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.SmoothL1Loss()

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    y_mean = torch.tensor(train_dataset.y_mean, dtype=torch.float32, device=device)
    y_std  = torch.tensor(train_dataset.y_std, dtype=torch.float32, device=device)

    best_val_rmse = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_tr = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            y = batch.y.view(-1, 1).float().to(device)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            n_tr += batch.num_graphs

        if scheduler:
            scheduler.step()
        train_loss = total_loss / max(n_tr, 1)

        # Validation
        model.eval()
        mae, rmse, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_norm = model(batch)
                y_norm = batch.y.view(-1, 1).float().to(device)
                pred_k = denorm(pred_norm, y_mean, y_std)
                y_k = denorm(y_norm, y_mean, y_std)
                diff = pred_k - y_k
                mae += diff.abs().sum().item()
                rmse += (diff ** 2).sum().item()
                n_val += batch.num_graphs
        mae = mae / max(n_val, 1)
        rmse = (rmse / max(n_val, 1)) ** 0.5

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch:03d} | TrainLoss(norm) {train_loss:.4f} | Val MAE(K) {mae:.3f} | Val RMSE(K) {rmse:.3f} | LR {lr_now:.6f}")

        if rmse < best_val_rmse:
            best_val_rmse = rmse
            ckpt_path = Path(args.checkpoint_dir) / f"best_rmse_{best_val_rmse:.3f}K_ep{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "y_mean": float(y_mean.item()),
                "y_std": float(y_std.item()),
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Final test
    model.eval()
    mae_t, rmse_t, n_te = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_k = denorm(model(batch), y_mean, y_std)
            y_k = denorm(batch.y.view(-1,1).float().to(device), y_mean, y_std)
            diff = pred_k - y_k
            mae_t += diff.abs().sum().item()
            rmse_t += (diff ** 2).sum().item()
            n_te += batch.num_graphs
    mae_t /= max(n_te, 1)
    rmse_t = (rmse_t / max(n_te, 1)) ** 0.5
    logger.info(f"[TEST] MAE(K) {mae_t:.3f} | RMSE(K) {rmse_t:.3f}")


if __name__ == "__main__":
    main()
