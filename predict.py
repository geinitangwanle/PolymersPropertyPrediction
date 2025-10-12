#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script converted from demo3.ipynb (best-effort).
It expects the same project structure used in the notebook:
- BestModel/ConvModel.py providing `GatedGCNModel`
- NPZGraphDataset.py providing `NPZGraphDataset`
- A checkpoint file with model weights
- A CSV manifest that lists .npz graph files and labels (optional)

Usage example:
python predict.py \
  --ckpt_path ./weights/BestModel.pt \
  --manifest ./graph_npz/manifest.csv \
  --root_dir ./ \
  --batch_size 64 \
  --device auto \
  --out_png output.png \
  --out_csv preds.csv \
  --test_size 0.2 \
  --seed 42 \
  --layers_in_conv 3 \
  --channels 64 \
  --num_node_types 9

Note:
- If your project uses different argument names or model signature, adjust below.
- This script focuses on **inference/evaluation** (推理) rather than training.
"""

import argparse
import os
from pathlib import Path
import json
import math
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project-specific imports inferred from the notebook
try:
    from BestModel.ConvModel import GatedGCNModel
except Exception as e:
    print("ERROR: Failed to import BestModel.ConvModel:GatedGCNModel. "
          "Please ensure your project structure is available on PYTHONPATH.")
    raise

try:
    from NPZGraphDataset import NPZGraphDataset
except Exception as e:
    print("ERROR: Failed to import NPZGraphDataset. "
          "Please ensure NPZGraphDataset.py is in the working directory or PYTHONPATH.")
    raise

import matplotlib
matplotlib.use("Agg")  # for headless savefig
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(args) -> torch.nn.Module:
    """
    Model signature inferred from the notebook snippet.
    Update as needed if your GatedGCNModel has different args.
    """
    model = GatedGCNModel(
        layers_in_conv=args.layers_in_conv,
        channels=args.channels,
        num_node_types=args.num_node_types,
    )
    return model


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        # NPZGraphDataset should return graph tensors; adapt if your collate differs
        # We assume batch is a dict-like with 'x', 'edge_index', etc. and 'y' for labels.
        # If your dataset returns (inputs, target), adjust here accordingly.
        if isinstance(batch, dict):
            y = batch.get("y", None)
            inputs = batch
        else:
            # fallback: (inputs, y)
            inputs, y = batch

        if y is None:
            raise RuntimeError("Batch does not contain labels 'y'. Please adjust dataset/loader.")

        # Move to device (handle nested tensors)
        def to_device(obj):
            if torch.is_tensor(obj):
                return obj.to(device)
            if isinstance(obj, (list, tuple)):
                return type(obj)(to_device(o) for o in obj)
            if isinstance(obj, dict):
                return {k: to_device(v) for k, v in obj.items()}
            return obj

        inputs = to_device(inputs)

        # Forward
        out = model(inputs)  # adapt if your forward signature differs
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out.squeeze(-1).detach().cpu().numpy()

        y_np = y.detach().cpu().numpy().reshape(-1)
        y_true.append(y_np)
        y_pred.append(out.reshape(-1))

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Graph NPZ inference (converted from notebook).")

    # Data & IO
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--manifest", type=str, required=True, help="Path to CSV manifest listing .npz files and labels.")
    parser.add_argument("--root_dir", type=str, default=".", help="Root dir prefix for relative npz paths in manifest.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=0, help="Num workers for DataLoader.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"], help="Device selection.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction for test split (0-1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out_png", type=str, default="output.png", help="Output path for scatter plot.")
    parser.add_argument("--out_csv", type=str, default="preds.csv", help="Output CSV for predictions.")

    # Model hyperparameters (from notebook defaults)
    parser.add_argument("--layers_in_conv", type=int, default=3)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num_node_types", type=int, default=9)

    args = parser.parse_args()

    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Read manifest
    manifest_path = Path(args.manifest)
    root_dir = Path(args.root_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)

    # Optional stratification by label bins
    if "label" in manifest.columns:
        try:
            bins = pd.qcut(manifest["label"], q=10, duplicates="drop")
            stratify_labels = bins.astype(str)
        except Exception:
            stratify_labels = None
    else:
        stratify_labels = None

    # Train/test split indices (we only evaluate test here)
    idx_all = np.arange(len(manifest))
    train_idx, test_idx = train_test_split(
        idx_all, test_size=args.test_size, random_state=args.seed,
        stratify=stratify_labels if stratify_labels is not None else None
    )

    # Build dataset & loaders
    dataset = NPZGraphDataset(
        manifest=str(manifest_path),
        root=str(root_dir),
    )

    test_ds = Subset(dataset, test_idx)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    # Build model & load checkpoint
    model = build_model(args)
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    # support typical checkpoint formats
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Inference
    y_true, y_pred = run_inference(model, test_loader, device)

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"Test MAE : {mae:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test R2  : {r2:.6f}")

    # Save CSV
    out_csv = Path(args.out_csv)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(out_csv, index=False)
    print(f"Saved predictions to: {out_csv.resolve()}")

    # Scatter plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    lo = float(np.min([y_true.min(), y_pred.min()]))
    hi = float(np.max([y_true.max(), y_pred.max()]))
    plt.plot([lo, hi], [lo, hi], "r--", lw=2, label="Ideal")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Prediction vs True")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = Path(args.out_png)
    plt.savefig(out_png, dpi=200)
    print(f"Saved plot to: {out_png.resolve()}")


if __name__ == "__main__":
    main()
