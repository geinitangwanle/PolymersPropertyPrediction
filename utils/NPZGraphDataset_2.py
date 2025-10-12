import os
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

def load_graph_npz(
    file_path: Union[str, Path],
    *,
    mode: str = "compat",                 # "compat" 或 "rich"
    merge_nonbond: bool = False,          # 是否把非键边合并进主图
    separate_pos: bool = True,            # 是否把坐标单独放入 data.pos
    feature_cols: Optional[Sequence[int]] = (0, 1, 2, 3),  # compat 下 x 的列
    coord_cols: Optional[Sequence[int]] = (4, 5, 6),       # pos 的列
    dtype: torch.dtype = torch.float32,
) -> Data:
    """
    期望 .npz 至少包含：
      - node_feats: (N, F) float32
      - edge_index: (2, E) int64
      - edge_attr : (E, A) float32
      - label     : (1,) float32
    可选：
      - node_feats_extra: (N, Fe)
      - edge_attr_extra : (E, Ae)
      - edge_index_nb   : (2, Enb)
      - edge_attr_nb    : (Enb, Anb)
    """
    file_path = Path(file_path)
    with np.load(file_path, allow_pickle=False) as npz:
        node_feats = npz["node_feats"]
        edge_index = npz["edge_index"]
        edge_attr  = npz["edge_attr"]
        label      = npz["label"]

        node_feats_extra = npz["node_feats_extra"] if "node_feats_extra" in npz else None
        edge_attr_extra  = npz["edge_attr_extra"]  if "edge_attr_extra"  in npz else None
        edge_index_nb    = npz["edge_index_nb"]    if "edge_index_nb"    in npz else None
        edge_attr_nb     = npz["edge_attr_nb"]     if "edge_attr_nb"     in npz else None

    # --- pos / x ---
    if separate_pos and coord_cols is not None:
        pos_t = torch.as_tensor(node_feats[:, coord_cols], dtype=dtype)
    else:
        pos_t = None

    if mode == "compat":
        if feature_cols is None or len(feature_cols) == 0:
            x_t = None
        else:
            x_t = torch.as_tensor(node_feats[:, feature_cols], dtype=dtype)
    elif mode == "rich":
        base_x = (torch.as_tensor(node_feats[:, feature_cols], dtype=dtype)
                  if (feature_cols is not None and len(feature_cols) > 0) else None)
        if node_feats_extra is not None:
            x_extra = torch.as_tensor(node_feats_extra, dtype=dtype)
            x_t = x_extra if base_x is None else torch.cat([base_x, x_extra], dim=1)
        else:
            x_t = base_x
    else:
        raise ValueError(f"mode must be 'compat' or 'rich', got {mode}")

    # --- edges ---
    edge_index_t = torch.as_tensor(edge_index, dtype=torch.long)
    edge_attr_t  = torch.as_tensor(edge_attr,  dtype=dtype)

    if mode == "rich" and edge_attr_extra is not None:
        e_extra_t = torch.as_tensor(edge_attr_extra, dtype=dtype)
        edge_attr_t = torch.cat([edge_attr_t, e_extra_t], dim=1)

    # --- 合并非键边（可选）---
        # --- 合并非键边（可选，带对齐）---
    if merge_nonbond and (edge_index_nb is not None) and (edge_attr_nb is not None):
        nb_idx_t  = torch.as_tensor(edge_index_nb, dtype=torch.long)
        nb_attr_t = torch.as_tensor(edge_attr_nb,  dtype=dtype)

        cov_flag = torch.zeros((edge_attr_t.size(0), 1), dtype=dtype)
        nb_flag  = torch.ones((nb_attr_t.size(0), 1), dtype=dtype)

        cov_feat = torch.cat([cov_flag, edge_attr_t], dim=1)  # [E, 1+A]
        nb_feat  = torch.cat([nb_flag,  nb_attr_t],  dim=1)   # [Enb, 1+Anb]

        # 右侧 0 填充到相同列数
        cov_D = cov_feat.size(1)
        nb_D  = nb_feat.size(1)
        D = max(cov_D, nb_D)

        if cov_D < D:
            pad = torch.zeros((cov_feat.size(0), D - cov_D), dtype=dtype, device=cov_feat.device)
            cov_feat = torch.cat([cov_feat, pad], dim=1)
        if nb_D < D:
            pad = torch.zeros((nb_feat.size(0), D - nb_D), dtype=dtype, device=nb_feat.device)
            nb_feat = torch.cat([nb_feat, pad], dim=1)

        edge_attr_t  = torch.cat([cov_feat, nb_feat], dim=0)   # [E+Enb, D]
        edge_index_t = torch.cat([edge_index_t, nb_idx_t], dim=1)


    y_t = torch.as_tensor(label, dtype=dtype).view(-1)

    return Data(x=x_t, edge_index=edge_index_t, edge_attr=edge_attr_t, y=y_t, pos=pos_t)


class NPZGraphDataset(Dataset):
    """
    基于 manifest 的 .npz 图数据集（兼容 compat/rich 模式与非键边合并）。
    """
    def __init__(
        self,
        manifest: Union[pd.DataFrame, str, Path],
        root: Optional[Union[str, Path]] = None,
        *,
        mode: str = "compat",                 # "compat" / "rich"
        merge_nonbond: bool = False,          # 是否把非键边合并进主图
        separate_pos: bool = True,
        feature_cols: Optional[Sequence[int]] = (0, 1, 2, 3),
        coord_cols: Optional[Sequence[int]] = (4, 5, 6),
        dtype: torch.dtype = torch.float32,
        standardize_y: bool = False,
    ):
        if isinstance(manifest, (str, Path)):
            manifest = pd.read_csv(manifest)
        self.manifest = manifest.reset_index(drop=True).copy()
        self.root = Path(root) if root is not None else None

        self.mode = mode
        self.merge_nonbond = merge_nonbond
        self.separate_pos = separate_pos
        self.feature_cols = feature_cols
        self.coord_cols = coord_cols
        self.dtype = dtype
        self.standardize_y = standardize_y

        # 预先计算 y 的均值方差（若启用）
        self._y_mean = None
        self._y_std  = None
        if self.standardize_y:
            if "label" in self.manifest and self.manifest["label"].notna().all():
                y = self.manifest["label"].astype(float).to_numpy()
            else:
                ys = []
                for fp in self.manifest["file_path"]:
                    full_path = self._resolve_path(fp)
                    with np.load(full_path, allow_pickle=False) as npz:
                        ys.append(float(npz["label"][0]))
                y = np.array(ys, dtype=np.float32)
            self._y_mean = float(np.mean(y))
            self._y_std  = float(np.std(y) + 1e-8)

    def _resolve_path(self, fp: str) -> Path:
        p = Path(fp)
        if not p.is_absolute() and self.root is not None:
            p = self.root / p
        return p

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Data:
        row = self.manifest.iloc[idx]
        file_path = self._resolve_path(row["file_path"])
        data = load_graph_npz(
            file_path,
            mode=self.mode,
            merge_nonbond=self.merge_nonbond,
            separate_pos=self.separate_pos,
            feature_cols=self.feature_cols,
            coord_cols=self.coord_cols,
            dtype=self.dtype,
        )
        if self.standardize_y and data.y is not None:
            data.y = (data.y - self._y_mean) / self._y_std

        data.mol_id = row.get("mol_id", idx)
        data.file_path = str(file_path)
        return data

    @property
    def y_mean(self) -> Optional[float]:
        return self._y_mean

    @property
    def y_std(self) -> Optional[float]:
        return self._y_std

    def peek_dims(self) -> dict:
        """读第一个可用样本，返回 {x_dim, edge_attr_dim, num_nodes, num_edges, has_pos}。"""
        for i in range(len(self.manifest)):
            try:
                d = self[i]
                return {
                    "x_dim": None if d.x is None else int(d.x.size(1)),
                    "edge_attr_dim": None if d.edge_attr is None else int(d.edge_attr.size(1)),
                    "num_nodes": int(d.num_nodes),
                    "num_edges": int(d.edge_index.size(1)),
                    "has_pos": d.pos is not None,
                }
            except Exception:
                continue
        raise RuntimeError("peek_dims: 无法读取任何样本，请检查 manifest / 路径与 .npz 内容。")
