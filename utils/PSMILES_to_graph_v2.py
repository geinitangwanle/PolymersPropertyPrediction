from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import os, math, warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, rdMolDescriptors as rdMD
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm.auto import tqdm
from collections import deque


def _resolved_positions(
    mol_raw: Chem.Mol,
    conf: Chem.Conformer,
    star_to_carbon: Dict[int, int],
) -> List[Tuple[float, float, float]]:
    coords: List[Tuple[float, float, float]] = []
    for atom in mol_raw.GetAtoms():
        idx = atom.GetIdx()
        mapped_idx = star_to_carbon.get(idx, idx)
        pos = conf.GetAtomPosition(mapped_idx)
        coords.append((float(pos.x), float(pos.y), float(pos.z)))
    return coords


def _rbf_expand(dist: float, K: int = 16, r_max: float = 4.0, eps: float = 1e-6) -> List[float]:
    centers = np.linspace(0.0, r_max, K)
    gamma = 1.0 / (centers[1] - centers[0] + eps) ** 2
    return np.exp(-gamma * (dist - centers) ** 2).astype(np.float32).tolist()


def _build_hbd_hba_sets(mol: Chem.Mol) -> Tuple[set, set]:
    """
    稳健版：优先在原 mol（sanitize=False，但已 FastFindRings）上跑；
    若失败，复制一份做“最小化 sanitize”（环+芳香），仍失败则兜底为空集。
    """
    fdef = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef)

    def _from_m(m: Chem.Mol) -> Tuple[set, set]:
        feats = factory.GetFeaturesForMol(m)
        hbd, hba = set(), set()
        for f in feats:
            fam = f.GetFamily()
            ids = f.GetAtomIds()
            if fam == 'Donor':
                hbd.update(ids)
            elif fam == 'Acceptor':
                hba.update(ids)
        return hbd, hba

    # 尝试直接跑
    try:
        return _from_m(mol)
    except Exception:
        pass

    # 复制 + 最小化 sanitize
    mol2 = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(
            mol2,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                        Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        )
    except Exception:
        try:
            Chem.FastFindRings(mol2)
        except Exception:
            return set(), set()

    try:
        return _from_m(mol2)
    except Exception:
        return set(), set()


def graph_from_psmiles(
    psmiles: str,
    *,
    add_atom_feats: bool = True,
    add_edge_rbf: bool = True,
    rbf_K: int = 16,
    rbf_rmax: float = 4.0,
    add_nonbond_edges: bool = False,
    nonbond_cutoff: float = 4.0,
    add_anchor_context: bool = True,
    compute_gasteiger: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Convert a PSMILES string into graph-friendly features (backward-compatible).
    """
    mol_raw = Chem.MolFromSmiles(psmiles, sanitize=False)
    if mol_raw is None:
        raise ValueError(f"Could not parse PSMILES: {psmiles!r}")
    mol_raw.UpdatePropertyCache(strict=False)

    # ✅ 关键：初始化环信息，避免 RingInfo not initialized
    try:
        Chem.FastFindRings(mol_raw)
    except Exception:
        pass

    # 记录原始星位
    star_idxs = [a.GetIdx() for a in mol_raw.GetAtoms() if a.GetAtomicNum() == 0]

    # 用 C 替星位以生成几何
    rw = Chem.RWMol(mol_raw)
    star_to_carbon: Dict[int, int] = {}
    star_indices = sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0), reverse=True)
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

    # embed + optimize（带重试）
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

    # ---------- node features ----------
    node_basic = []
    node_extra_rows = []

    # HBD/HBA（稳健）
    try:
        hbd_set, hba_set = _build_hbd_hba_sets(mol_raw)
    except Exception:
        hbd_set, hba_set = set(), set()

    # Gasteiger charge（可选）
    if compute_gasteiger:
        try:
            Chem.rdPartialCharges.ComputeGasteigerCharges(mol_geom)
        except Exception:
            compute_gasteiger = False

    # ring sizes（稳健）
    try:
        sssr = Chem.GetSymmSSSR(mol_raw)
        ring_sets = [set(r) for r in sssr]
    except Exception:
        try:
            Chem.FastFindRings(mol_raw)
            sssr = Chem.GetSymmSSSR(mol_raw)
            ring_sets = [set(r) for r in sssr]
        except Exception:
            ring_sets = []

    def smallest_ring_size_idx(i: int) -> int:
        sizes = [len(r) for r in ring_sets if i in r] if ring_sets else []
        return int(min(sizes)) if sizes else 0

    # anchor 邻居集合（一次性计算）
    anchor_neighbors = set()
    if add_anchor_context and star_idxs:
        for s in star_idxs:
            for nb in mol_raw.GetAtomWithIdx(s).GetNeighbors():
                anchor_neighbors.add(nb.GetIdx())
        # 图邻接用于 BFS
        adj = {a.GetIdx(): [n.GetIdx() for n in a.GetNeighbors()] for a in mol_raw.GetAtoms()}
    else:
        adj = None

    for atom, (x, y, z) in zip(mol_raw.GetAtoms(), coords):
        i = atom.GetIdx()
        # 基础：保持兼容
        node_basic.append([
            float(atom.GetAtomicNum()),
            float(atom.GetTotalDegree()),
            float(atom.GetFormalCharge()),
            float(int(atom.GetIsAromatic())),
            x, y, z
        ])

        if add_atom_feats:
            hyb = int(atom.GetHybridization())
            srs = smallest_ring_size_idx(i)
            try:
                q = float(mol_geom.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) if compute_gasteiger else 0.0
            except Exception:
                q = 0.0
            extra = [
                float(atom.GetImplicitValence()),
                float(atom.GetTotalValence()),
                float(hyb == rdchem.HybridizationType.SP),
                float(hyb == rdchem.HybridizationType.SP2),
                float(hyb == rdchem.HybridizationType.SP3),
                float(atom.IsInRing()),
                float(srs == 5),
                float(srs == 6),
                float(srs),
                float(i in hbd_set),
                float(i in hba_set),
                float(atom.GetMass()),
                float(q),
            ]
            if add_anchor_context:
                is_anchor_nb = float(i in anchor_neighbors)
                # BFS 到最近 anchor 邻居
                if adj is not None and anchor_neighbors:
                    qd = deque([(i, 0)]); seen = {i}
                    dist = 999
                    while qd:
                        u, d = qd.popleft()
                        if u in anchor_neighbors:
                            dist = d; break
                        for v in adj[u]:
                            if v not in seen:
                                seen.add(v); qd.append((v, d+1))
                else:
                    dist = 0 if is_anchor_nb else 999
                extra += [is_anchor_nb, float(dist)]
            node_extra_rows.append(extra)

    node_feats = pd.DataFrame(node_basic,
                              columns=["atomic_num", "degree", "formal_charge", "is_aromatic", "x", "y", "z"])
    out: Dict[str, pd.DataFrame] = {"node_feats": node_feats}

    if add_atom_feats:
        extra_cols = [
            "implicit_valence", "total_valence",
            "hyb_sp", "hyb_sp2", "hyb_sp3",
            "is_in_ring", "in_5ring", "in_6ring", "smallest_ring",
            "is_hbd", "is_hba", "atomic_mass", "gasteiger_q"
        ]
        if add_anchor_context:
            extra_cols += ["is_anchor_neighbor", "dist_to_anchor"]
        out["node_feats_extra"] = pd.DataFrame(node_extra_rows, columns=extra_cols)

    # ---------- covalent edges ----------
    edge_row, edge_col = [], []
    edge_attr_basic, edge_attr_extra = [], []

    for bond in mol_raw.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        length = math.dist(coords[i], coords[j])

        basic_feat = [
            float(bond.GetBondTypeAsDouble()),
            float(int(bond.GetIsConjugated())),
            float(int(bond.IsInRing())),
            float(length),
        ]

        if add_edge_rbf:
            bt = bond.GetBondType()
            one_hot = [
                float(bt == rdchem.BondType.SINGLE),
                float(bt == rdchem.BondType.DOUBLE),
                float(bt == rdchem.BondType.TRIPLE),
                float(bt == rdchem.BondType.AROMATIC),
            ]
            st = bond.GetStereo()
            stereo = [
                float(st in (rdchem.BondStereo.STEREOE, rdchem.BondStereo.STEREOTRANS)),
                float(st in (rdchem.BondStereo.STEREOZ, rdchem.BondStereo.STEREOCIS)),
            ]
            rbf = _rbf_expand(length, K=rbf_K, r_max=rbf_rmax)
            extra_feat = one_hot + [
                float(bond.GetIsConjugated()),
                float(bond.IsInRing()),
            ] + stereo + rbf
        else:
            extra_feat = []

        edge_row.extend([i, j]); edge_col.extend([j, i])
        edge_attr_basic.extend([basic_feat, basic_feat])
        if add_edge_rbf:
            edge_attr_extra.extend([extra_feat, extra_feat])

    edge_index = pd.DataFrame([edge_row, edge_col], index=["row", "col"])
    edge_attr = pd.DataFrame(edge_attr_basic, columns=["bond_type", "is_conjugated", "is_in_ring", "bond_length"])
    out["edge_index"] = edge_index
    out["edge_attr"] = edge_attr

    if add_edge_rbf:
        extra_cols = [
            "bt_single", "bt_double", "bt_triple", "bt_aromatic",
            "ex_is_conj", "ex_is_ring",
            "stereo_EorTrans", "stereo_ZorCis",
        ] + [f"rbf_{k}" for k in range(rbf_K)]
        out["edge_attr_extra"] = pd.DataFrame(edge_attr_extra, columns=extra_cols)

    # ---------- non-bond edges ----------
    if add_nonbond_edges:
        N = len(coords)
        nb_row, nb_col, nb_attr = [], [], []
        for i in range(N):
            for j in range(i+1, N):
                if not mol_raw.GetBondBetweenAtoms(i, j):
                    d = math.dist(coords[i], coords[j])
                    if d <= nonbond_cutoff:
                        rbf = _rbf_expand(d, K=rbf_K, r_max=rbf_rmax)
                        feat = [1.0] + rbf
                        nb_row.extend([i, j]); nb_col.extend([j, i])
                        nb_attr.extend([feat, feat])
        if nb_row:
            out["edge_index_nb"] = pd.DataFrame([nb_row, nb_col], index=["row", "col"])
            out["edge_attr_nb"] = pd.DataFrame(nb_attr, columns=["is_nonbond"] + [f"rbf_{k}" for k in range(rbf_K)])

    return out


def convert_csv_to_graphs(
    csv_path: str | Path,
    save_dir: str | Path | None = None,
    *,
    save_extras: bool = True,
    graph_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    读取 CSV 中的 PSMILES，批量生成图数据，并附带标签。
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    graphs: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    graph_kwargs = graph_kwargs or {}

    iterator = tqdm(df.itertuples(index=True), total=len(df), desc="Converting PSMILES to graphs")

    for idx, row in enumerate(iterator):
        psmiles = getattr(row, "PSMILES")
        label = getattr(row, "Tg")  # 若列名不同这里改一下

        if not isinstance(psmiles, str) or not psmiles.strip():
            continue

        try:
            g = graph_from_psmiles(psmiles, **graph_kwargs)
        except Exception as e:
            failures.append({
                "csv_row": row.Index,
                "mol_id": getattr(row, "polymer_id", row.Index),
                "psmiles": psmiles,
                "error": str(e),
            })
            continue

        mol_id = getattr(row, "polymer_id", row.Index)
        g["mol_id"] = mol_id
        g["label"] = float(label)
        graphs.append(g)

        rec: Dict[str, Any] = {
            "mol_id": mol_id,
            "label": float(label),
            "num_nodes": len(g["node_feats"]),
            "num_edges": len(g["edge_attr"]) // 2,
            "csv_row": row.Index,
        }

        if save_dir is not None:
            batch_idx = idx // 1000
            batch_dir = save_dir / f"batch_{batch_idx}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            file_path = batch_dir / f"{mol_id}.npz"

            npz_dict = dict(
                node_feats=g["node_feats"].to_numpy(dtype=np.float32),
                edge_index=g["edge_index"].to_numpy(dtype=np.int64),
                edge_attr=g["edge_attr"].to_numpy(dtype=np.float32),
                label=np.array([label], dtype=np.float32),
            )
            if save_extras:
                if "node_feats_extra" in g:
                    npz_dict["node_feats_extra"] = g["node_feats_extra"].to_numpy(dtype=np.float32)
                if "edge_attr_extra" in g:
                    npz_dict["edge_attr_extra"] = g["edge_attr_extra"].to_numpy(dtype=np.float32)
                if "edge_index_nb" in g and "edge_attr_nb" in g:
                    npz_dict["edge_index_nb"] = g["edge_index_nb"].to_numpy(dtype=np.int64)
                    npz_dict["edge_attr_nb"]  = g["edge_attr_nb"].to_numpy(dtype=np.float32)

            np.savez_compressed(file_path, **npz_dict)
            rec["file_path"] = str(file_path)

        manifest_rows.append(rec)

    manifest = pd.DataFrame(manifest_rows)

    # 把失败样本记录下来，不影响后续流程
    if failures:
        fail_df = pd.DataFrame(failures)
        out_csv = (save_dir / "convert_failures.csv") if save_dir is not None else Path("convert_failures.csv")
        fail_df.to_csv(out_csv, index=False)
        warnings.warn(f"转换失败 {len(fail_df)} 条，详情见：{out_csv}")

    return graphs, manifest
