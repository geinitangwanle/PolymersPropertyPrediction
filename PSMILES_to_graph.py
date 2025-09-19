from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any
import math
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from tqdm.auto import tqdm


def _resolved_positions(
    mol_raw: Chem.Mol,
    conf: Chem.Conformer,
    star_to_carbon: Dict[int, int],
) -> List[Tuple[float, float, float]]:
    """Return 3D coordinates aligned with mol_raw atoms; stars reuse the replacement carbon."""
    coords: List[Tuple[float, float, float]] = []
    for atom in mol_raw.GetAtoms():
        idx = atom.GetIdx()
        mapped_idx = star_to_carbon.get(idx, idx)
        pos = conf.GetAtomPosition(mapped_idx)
        coords.append((float(pos.x), float(pos.y), float(pos.z)))
    return coords


def graph_from_psmiles(psmiles: str) -> Dict[str, pd.DataFrame]:
    """
    Convert a PSMILES string (with star attachment points) into graph-friendly features.

    Returns
    -------
    dict with
        • node_feats: per-atom properties + 3D coordinates
        • edge_index: bidirectional adjacency (COO)
        • edge_attr: bond descriptors (type, conjugation, ring flag, length)
        • coords: raw coordinates aligned with the original atoms
    """
    mol_raw = Chem.MolFromSmiles(psmiles, sanitize=False)
    if mol_raw is None:
        raise ValueError(f"Could not parse PSMILES: {psmiles!r}")
    mol_raw.UpdatePropertyCache(strict=False)

    rw = Chem.RWMol(mol_raw)
    star_to_carbon: Dict[int, int] = {}

    star_indices = sorted(
        (atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() == 0),
        reverse=True,
    )
    for star_idx in star_indices:
        star_atom = rw.GetAtomWithIdx(star_idx)
        neighbors = star_atom.GetNeighbors()
        if len(neighbors) != 1:
            raise ValueError(f"Star atom {star_idx} has {len(neighbors)} neighbors; expected exactly one.")

        neighbor_idx = neighbors[0].GetIdx()
        bond = rw.GetBondBetweenAtoms(star_idx, neighbor_idx)

        new_c_idx = rw.AddAtom(Chem.Atom("C"))
        rw.AddBond(new_c_idx, neighbor_idx, bond.GetBondType())

        star_to_carbon[star_idx] = new_c_idx
        rw.RemoveAtom(star_idx)

    mol_geom = rw.GetMol()
    Chem.SanitizeMol(mol_geom)
    mol_geom = Chem.AddHs(mol_geom)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    embed_status = AllChem.EmbedMolecule(mol_geom, params)
    if embed_status != 0:
        params.useRandomCoords = True
        for _ in range(4):
            embed_status = AllChem.EmbedMolecule(mol_geom, params)
            if embed_status == 0:
                break
    if embed_status != 0:
        raise RuntimeError("Failed to embed 3D coordinates for the molecule.")

    AllChem.UFFOptimizeMolecule(mol_geom, maxIters=200)

    conf = mol_geom.GetConformer()
    coords = _resolved_positions(mol_raw, conf, star_to_carbon)

    node_feats = []
    for atom, (x, y, z) in zip(mol_raw.GetAtoms(), coords):
        node_feats.append([
            float(atom.GetAtomicNum()),
            float(atom.GetTotalDegree()),
            float(atom.GetFormalCharge()),
            float(int(atom.GetIsAromatic())),
            x,
            y,
            z,
        ])

    edge_index = [[], []]
    edge_attr = []
    for bond in mol_raw.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        pos_i = coords[i]
        pos_j = coords[j]
        length = math.dist(pos_i, pos_j)

        feat = [
            float(bond.GetBondTypeAsDouble()),
            float(int(bond.GetIsConjugated())),
            float(int(bond.IsInRing())),
            length,
        ]
        edge_index[0].extend([i, j])
        edge_index[1].extend([j, i])
        edge_attr.extend([feat, feat])

    return {
        "node_feats": pd.DataFrame(node_feats, columns=["atomic_num", "degree", "formal_charge", "is_aromatic", "x", "y", "z"]),
        "edge_index": pd.DataFrame(edge_index, index=["row", "col"]),
        "edge_attr": pd.DataFrame(edge_attr, columns=["bond_type", "is_conjugated", "is_in_ring", "bond_length"])
    }
def convert_csv_to_graphs(
    csv_path: str | Path,
    save_dir: str | Path | None = None,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    读取 CSV 中的 PSMILES，批量生成图数据，并附带全局标签。

    Parameters
    ----------
    csv_path : str or Path
        CSV 文件路径，需包含列 `PSMILES` 与 `labels.Exp_Tg(K)`。
    save_dir : str or Path or None, default=None
        若提供，将把每个图保存为 `.npz` 文件，便于后续快速加载。

    Returns
    -------
    graphs : list of dict
        每项是 graph_from_psmiles 的返回结果，额外包含键 `label` 与 `mol_id`。
    manifest : pandas.DataFrame
        索引信息，含 `mol_id`、`label`、节点/边数量以及（如保存）对应文件路径。
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    graphs: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    iterator = tqdm(
        df.itertuples(index=True),
        total=len(df),
        desc="Converting PSMILES to graphs",
    )


    for idx, row in enumerate(iterator):
        psmiles = getattr(row, "PSMILES")
        label = getattr(row, "Tg")

        if not isinstance(psmiles, str) or not psmiles.strip():
            continue

        graph = graph_from_psmiles(psmiles)
        mol_id = getattr(row, "polymer_id", row.Index)  # 如果 CSV 有 polymer_id 一列则使用

        graph["mol_id"] = mol_id
        graph["label"] = float(label)
        graphs.append(graph)

        record: Dict[str, Any] = {
            "mol_id": mol_id,
            "label": float(label),
            "num_nodes": len(graph["node_feats"]),
            "num_edges": len(graph["edge_attr"]) // 2,
            "csv_row": row.Index,
        }

        if save_dir is not None:
            batch_idx = idx // 1000
            batch_dir = save_dir / f"batch_{batch_idx}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            file_path = batch_dir / f"{mol_id}.npz"
            np.savez_compressed(
                file_path,
                node_feats=graph["node_feats"].to_numpy(dtype=np.float32),
                edge_index=graph["edge_index"].to_numpy(dtype=np.int64),
                edge_attr=graph["edge_attr"].to_numpy(dtype=np.float32),
                label=np.array([label], dtype=np.float32),
            )
            record["file_path"] = str(file_path)

        manifest_rows.append(record)

    manifest = pd.DataFrame(manifest_rows)
    return graphs, manifest

