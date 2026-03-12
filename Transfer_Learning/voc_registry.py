import torch
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from config import STRESS_MAP

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
except Exception as e:
    raise RuntimeError("缺少 RDKit。请使用 conda 或 pip 安装。") from e

# 全局唯一 VOC 知识库
VOC_KNOWLEDGE_BASE = [
    {"id": 0, "en": "Acetic acid", "smiles": "CC(=O)O", "stress": ["Salt"]},
    {"id": 1, "en": "Nonanal", "smiles": "CCCCCCCCC=O", "stress": ["Salt", "Heat"]},
    {"id": 2, "en": "Decanoic acid", "smiles": "CCCCCCCCCC(=O)O", "stress": ["Salt"]},
    {"id": 3, "en": "Heptanal", "smiles": "CCCCCCC=O", "stress": ["Drought", "Heat"]},
    {"id": 4, "en": "Hexanal", "smiles": "CCCCCC=O", "stress": ["Drought"]},
    {"id": 5, "en": "Decanal", "smiles": "CCCCCCCCCC=O", "stress": ["Drought"]},
    {"id": 6, "en": "2-Octanone", "smiles": "CCCCCCC(=O)C", "stress": ["Heat"]}
]

def get_stress_to_voc_mask() -> torch.Tensor:
    """构建 [3, 7] 的掩码，用于指导 Attention 分布。"""
    mask = torch.zeros((3, len(VOC_KNOWLEDGE_BASE)))
    for voc in VOC_KNOWLEDGE_BASE:
        for s in voc["stress"]:
            mask[STRESS_MAP[s], voc["id"]] = 1.0
    return mask

@dataclass
class MolGraph:
    smiles: str; x: torch.Tensor; edge_index: torch.Tensor; edge_attr: torch.Tensor

def _one_hot(value: int, choices: List[int]) -> List[int]:
    return [1 if value == c else 0 for c in choices]

def atom_features(atom: Chem.Atom) -> List[float]:
    elems = [1, 6, 7, 8, 9, 16, 17, 35]
    hyb_map = {Chem.rdchem.HybridizationType.SP:0, Chem.rdchem.HybridizationType.SP2:1, Chem.rdchem.HybridizationType.SP3:2, Chem.rdchem.HybridizationType.SP3D:3, Chem.rdchem.HybridizationType.SP3D2:4}
    return _one_hot(atom.GetAtomicNum(), elems) + [0 if atom.GetAtomicNum() in elems else 1] + \
           [float(atom.GetDegree()), float(atom.GetFormalCharge()), float(atom.GetTotalNumHs()), float(atom.GetIsAromatic())] + \
           _one_hot(hyb_map.get(atom.GetHybridization(), 5), [0,1,2,3,4,5])

def bond_features(bond: Chem.Bond) -> List[float]:
    btype_map = {Chem.rdchem.BondType.SINGLE:0, Chem.rdchem.BondType.DOUBLE:1, Chem.rdchem.BondType.TRIPLE:2, Chem.rdchem.BondType.AROMATIC:3}
    stereo_map = {Chem.rdchem.BondStereo.STEREONONE:0, Chem.rdchem.BondStereo.STEREOANY:1, Chem.rdchem.BondStereo.STEREOZ:2, Chem.rdchem.BondStereo.STEREOE:3, Chem.rdchem.BondStereo.STEREOCIS:4, Chem.rdchem.BondStereo.STEREOTRANS:5}
    return _one_hot(btype_map.get(bond.GetBondType(), 4), [0,1,2,3,4]) + \
           [float(bond.GetIsConjugated()), float(bond.GetIsAromatic()), float(bond.IsInRing())] + \
           _one_hot(stereo_map.get(bond.GetStereo(), 6), [0,1,2,3,4,5,6])

def smiles_to_graph(smiles: str) -> MolGraph:
    mol = Chem.MolFromSmiles(smiles)
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    edge_pairs, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_pairs.extend([(i, j), (j, i)]); edge_attrs.extend([bf, bf])
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous() if edge_pairs else torch.empty((2,0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32) if edge_attrs else torch.empty((0,15), dtype=torch.float32)
    return MolGraph(smiles=smiles, x=x, edge_index=edge_index, edge_attr=edge_attr)

def batch_graphs(graphs: List[MolGraph]) -> Dict[str, torch.Tensor]:
    xs, edge_indices, edge_attrs, batch = [], [], [], []
    offset = 0
    for gid, g in enumerate(graphs):
        n = g.x.shape[0]
        xs.append(g.x); batch.append(torch.full((n,), gid, dtype=torch.long))
        edge_indices.append(g.edge_index + offset); edge_attrs.append(g.edge_attr)
        offset += n
    return {"x": torch.cat(xs, dim=0), "edge_index": torch.cat(edge_indices, dim=1), "edge_attr": torch.cat(edge_attrs, dim=0), "batch": torch.cat(batch, dim=0)}

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_fingerprint(smiles: str) -> torch.Tensor:
    mol = Chem.MolFromSmiles(smiles)
    cnt = (_MORGAN_GEN.GetCountFingerprintAsNumPy(mol) > 0).astype(np.float32)
    maccs = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs)
    return torch.from_numpy(np.concatenate([cnt, maccs.astype(np.float32)], axis=0))

def build_knowledge_bank_tensors():
    """预处理得到全局唯一的分子图和指纹 Batch Tensors"""
    smiles_list = [v["smiles"] for v in VOC_KNOWLEDGE_BASE]
    g_batch = batch_graphs([smiles_to_graph(s) for s in smiles_list])
    fp_batch = torch.stack([smiles_to_fingerprint(s) for s in smiles_list], dim=0)
    return g_batch, fp_batch