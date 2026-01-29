from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
except Exception as e:
    raise RuntimeError(
        "缺少 RDKit。建议使用 conda 安装 rdkit，或 pip 安装 rdkit-pypi。"
    ) from e


# ============================
# 1 分子图特征构建
# ============================

@dataclass
class MolGraph:
    smiles: str
    x: torch.Tensor           # [N, node_dim]
    edge_index: torch.Tensor  # [2, E]
    edge_attr: torch.Tensor   # [E, edge_dim]


def _one_hot(value: int, choices: List[int]) -> List[int]:
    return [1 if value == c else 0 for c in choices]


def atom_features(atom: Chem.Atom) -> List[float]:
    atomic_num = atom.GetAtomicNum()
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    total_h = atom.GetTotalNumHs()
    is_aromatic = 1 if atom.GetIsAromatic() else 0

    common_elems = [1, 6, 7, 8, 9, 16, 17, 35]  # H C N O F S Cl Br
    elem_oh = _one_hot(atomic_num, common_elems)
    elem_other = [0 if atomic_num in common_elems else 1]

    hyb = atom.GetHybridization()
    hyb_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4,
    }
    hyb_id = hyb_map.get(hyb, 5)
    hyb_oh = _one_hot(hyb_id, [0, 1, 2, 3, 4, 5])

    # 总维度 19
    return (
        elem_oh
        + elem_other
        + [float(degree), float(formal_charge), float(total_h), float(is_aromatic)]
        + hyb_oh
    )


def bond_features(bond: Chem.Bond) -> List[float]:
    btype = bond.GetBondType()
    btype_map = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    btype_id = btype_map.get(btype, 4)
    btype_oh = _one_hot(btype_id, [0, 1, 2, 3, 4])

    is_conj = 1 if bond.GetIsConjugated() else 0
    is_arom = 1 if bond.GetIsAromatic() else 0
    is_ring = 1 if bond.IsInRing() else 0

    stereo = bond.GetStereo()
    stereo_map = {
        Chem.rdchem.BondStereo.STEREONONE: 0,
        Chem.rdchem.BondStereo.STEREOANY: 1,
        Chem.rdchem.BondStereo.STEREOZ: 2,
        Chem.rdchem.BondStereo.STEREOE: 3,
        Chem.rdchem.BondStereo.STEREOCIS: 4,
        Chem.rdchem.BondStereo.STEREOTRANS: 5,
    }
    stereo_id = stereo_map.get(stereo, 6)
    stereo_oh = _one_hot(stereo_id, [0, 1, 2, 3, 4, 5, 6])

    # 总维度 15
    return btype_oh + [float(is_conj), float(is_arom), float(is_ring)] + stereo_oh


def smiles_to_graph(smiles: str) -> MolGraph:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)

    edge_pairs: List[Tuple[int, int]] = []
    edge_attrs: List[List[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        # 双向有向边
        edge_pairs.append((i, j)); edge_attrs.append(bf)
        edge_pairs.append((j, i)); edge_attrs.append(bf)

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    return MolGraph(smiles=smiles, x=x, edge_index=edge_index, edge_attr=edge_attr)


def batch_graphs(graphs: List[MolGraph]) -> Dict[str, torch.Tensor]:
    xs, edge_indices, edge_attrs, batch = [], [], [], []
    node_offset = 0
    for gid, g in enumerate(graphs):
        n = g.x.shape[0]
        xs.append(g.x)
        batch.append(torch.full((n,), gid, dtype=torch.long))
        edge_indices.append(g.edge_index + node_offset)
        edge_attrs.append(g.edge_attr)
        node_offset += n

    return {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "batch": torch.cat(batch, dim=0),
    }


def global_mean_pool(h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    out = torch.zeros((num_graphs, h.size(1)), device=h.device, dtype=h.dtype)
    out.index_add_(0, batch, h)

    counts = torch.zeros((num_graphs,), device=h.device, dtype=h.dtype)
    ones = torch.ones((batch.size(0),), device=h.device, dtype=h.dtype)
    counts.index_add_(0, batch, ones)

    return out / counts.clamp_min(1.0).unsqueeze(1)


# ============================
# 2 分子指纹特征构建
# ============================

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def _bitvect_to_numpy(bitvect) -> np.ndarray:
    arr = np.zeros((bitvect.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def smiles_to_fingerprint(smiles: str, use_morgan: bool = True, use_maccs: bool = True) -> torch.Tensor:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    parts = []
    if use_morgan:
        cnt = _MORGAN_GEN.GetCountFingerprintAsNumPy(mol)
        parts.append((cnt > 0).astype(np.float32))  # 2048 维二值
    if use_maccs:
        maccs = MACCSkeys.GenMACCSKeys(mol)
        parts.append(_bitvect_to_numpy(maccs).astype(np.float32))  # 167 维二值

    if not parts:
        raise ValueError("At least one fingerprint type must be enabled")

    return torch.from_numpy(np.concatenate(parts, axis=0))


def batch_fingerprints(smiles_list: List[str]) -> torch.Tensor:
    return torch.stack([smiles_to_fingerprint(s) for s in smiles_list], dim=0)


def tanimoto_matrix(bitvecs: List) -> np.ndarray:
    n = len(bitvecs)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            mat[i, j] = DataStructs.TanimotoSimilarity(bitvecs[i], bitvecs[j])
    return mat


# ============================
# 3 分子图编码器
# ============================

class GraphConvEdge(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        m = self.msg_mlp(torch.cat([h[src], edge_attr], dim=1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        dh = self.upd_mlp(torch.cat([h, agg], dim=1))
        return self.norm(h + self.drop(dh))


class MolGraphEncoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([GraphConvEdge(hidden_dim, edge_dim, dropout=0.1) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.node_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return self.readout(global_mean_pool(h, batch))


# ============================
# 4 指纹编码器
# ============================

class FingerprintEncoder(nn.Module):
    def __init__(self, fp_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        return self.net(fp)


# ============================
# 5 二模态门控融合
# ============================

class TwoModalGatedFusion(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, z_graph: torch.Tensor, z_fp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate_net(torch.cat([z_graph, z_fp], dim=1))  # [B, dim] 取值 0 到 1
        z = gate * z_graph + (1.0 - gate) * z_fp
        return self.out(z), gate


# ============================
# 6 单分子编码器
# ============================

class MoleculeEncoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, fp_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.gnn = MolGraphEncoder(node_dim, edge_dim, hidden_dim=hidden_dim, num_layers=3)
        self.fp = FingerprintEncoder(fp_dim, hidden_dim=hidden_dim)
        self.fuse = TwoModalGatedFusion(dim=hidden_dim)

    def forward(self, graph_batch: Dict[str, torch.Tensor], fp_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_graph = self.gnn(**graph_batch)      # [B, D]
        z_fp = self.fp(fp_batch)               # [B, D]
        z_fused, gate = self.fuse(z_graph, z_fp)
        return {"z_graph": z_graph, "z_fp": z_fp, "z_fused": z_fused, "gate": gate}


# ============================
# 7 多分子组合融合
# ============================

class MixtureEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, per_mol_fused: torch.Tensor, weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # per_mol_fused: [B, M, D]
        logits = self.attn(per_mol_fused).squeeze(-1)  # [B, M]
        if weights is not None:
            w = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            logits = logits + torch.log(w.clamp_min(1e-12))
        alpha = torch.softmax(logits, dim=1)  # [B, M]
        z_mix = torch.sum(alpha.unsqueeze(-1) * per_mol_fused, dim=1)  # [B, D]
        return z_mix, alpha


# ============================
# 8 运行示例与输出
# ============================

def cosine_sim_df(z: torch.Tensor, names: List[str]) -> pd.DataFrame:
    zn = F.normalize(z, dim=1)
    sim = (zn @ zn.t()).detach().cpu().numpy()
    return pd.DataFrame(sim, index=names, columns=names)


def main(out_dir: str) -> None:
    torch.manual_seed(0)

    molecules = [
        {"cn": "乙酸", "smiles": "CC(=O)O"},
        {"cn": "壬醛", "smiles": "CCCCCCCCC=O"},
        {"cn": "癸酸", "smiles": "CCCCCCCCCC(=O)O"},
    ]
    names = [m["cn"] for m in molecules]
    smiles_list = [m["smiles"] for m in molecules]

    graphs = [smiles_to_graph(s) for s in smiles_list]
    graph_batch = batch_graphs(graphs)
    fp_batch = batch_fingerprints(smiles_list)

    node_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    fp_dim = fp_batch.shape[1]

    model = MoleculeEncoder(node_dim, edge_dim, fp_dim, hidden_dim=128)
    model.eval()

    with torch.no_grad():
        out = model(graph_batch, fp_batch)

    z_graph = out["z_graph"]
    z_fp = out["z_fp"]
    z_fused = out["z_fused"]
    gate = out["gate"]

    print("Fingerprints shape:", tuple(fp_batch.shape))
    print("Graph batch x shape:", tuple(graph_batch["x"].shape))
    print("Graph batch edge_index shape:", tuple(graph_batch["edge_index"].shape))
    print("z_graph shape:", tuple(z_graph.shape))
    print("z_fp shape:", tuple(z_fp.shape))
    print("z_fused shape:", tuple(z_fused.shape))

    rd_mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    morgan_bv = [_MORGAN_GEN.GetFingerprint(m) for m in rd_mols]
    maccs_bv = [MACCSkeys.GenMACCSKeys(m) for m in rd_mols]

    tan_morgan = pd.DataFrame(tanimoto_matrix(morgan_bv), index=names, columns=names)
    tan_maccs = pd.DataFrame(tanimoto_matrix(maccs_bv), index=names, columns=names)

    cos_graph = cosine_sim_df(z_graph, names)
    cos_fp = cosine_sim_df(z_fp, names)
    cos_fused = cosine_sim_df(z_fused, names)

    df_stats = pd.DataFrame({
        "分子": names,
        "SMILES": smiles_list,
        "原子数": [m.GetNumAtoms() for m in rd_mols],
        "化学键数": [m.GetNumBonds() for m in rd_mols],
        "有向边数": [m.GetNumBonds() * 2 for m in rd_mols],
        "指纹维度": [fp_dim] * 3,
        "节点特征维度": [node_dim] * 3,
        "边特征维度": [edge_dim] * 3,
    })

    df_gate = pd.DataFrame({
        "分子": names,
        "gate 平均值 越大越偏向图分支": gate.mean(dim=1).detach().cpu().numpy()
    })

    mix_enc = MixtureEncoder(hidden_dim=128)
    mix_enc.eval()
    with torch.no_grad():
        per_mol = z_fused.unsqueeze(0)  # [1, 3, 128]
        z_mix, alpha = mix_enc(per_mol, weights=None)

    print("Mixture embedding shape:", tuple(z_mix.shape))
    print("Mixture attention alpha:", alpha.detach().cpu().numpy())

    alpha_df = pd.DataFrame(alpha.detach().cpu().numpy(), columns=names)
    mix_df = pd.DataFrame(z_mix.detach().cpu().numpy())

    if out_dir:
        import os
        os.makedirs(out_dir, exist_ok=True)

        df_stats.to_csv(os.path.join(out_dir, "stats.csv"), index=False)
        tan_morgan.to_csv(os.path.join(out_dir, "tanimoto_morgan.csv"))
        tan_maccs.to_csv(os.path.join(out_dir, "tanimoto_maccs.csv"))
        cos_graph.to_csv(os.path.join(out_dir, "cos_graph.csv"))
        cos_fp.to_csv(os.path.join(out_dir, "cos_fp.csv"))
        cos_fused.to_csv(os.path.join(out_dir, "cos_fused.csv"))
        df_gate.to_csv(os.path.join(out_dir, "gate.csv"), index=False)
        alpha_df.to_csv(os.path.join(out_dir, "mixture_alpha.csv"), index=False)
        mix_df.to_csv(os.path.join(out_dir, "mixture_embedding.csv"), index=False)

        np.save(os.path.join(out_dir, "z_graph.npy"), z_graph.detach().cpu().numpy())
        np.save(os.path.join(out_dir, "z_fp.npy"), z_fp.detach().cpu().numpy())
        np.save(os.path.join(out_dir, "z_fused.npy"), z_fused.detach().cpu().numpy())
        np.save(os.path.join(out_dir, "z_mix.npy"), z_mix.detach().cpu().numpy())

        print("Saved files to", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="out", help="输出目录")
    args = parser.parse_args()
    main(args.out_dir)
