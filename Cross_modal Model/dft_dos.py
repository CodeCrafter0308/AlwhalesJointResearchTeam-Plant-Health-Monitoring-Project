
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from voc_registry import VOC_KNOWLEDGE_BASE


def _normalize_name(s: str) -> str:
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("_", "").replace("-", "")
    return s


def _build_filename_keywords() -> Dict[int, List[str]]:
    """
    VOC id -> filename/folder keywords list.
    Must stay consistent with voc_registry.VOC_KNOWLEDGE_BASE ordering.
    """
    zh = {
        0: ["乙酸", "醋酸"],
        1: ["壬醛"],
        2: ["癸酸"],
        3: ["庚醛"],
        4: ["正己醛", "己醛"],
        5: ["癸醛"],
        6: ["辛酮", "2辛酮", "2-辛酮", "2octanone", "octanone"],
    }
    kw: Dict[int, List[str]] = {}
    for item in VOC_KNOWLEDGE_BASE:
        vid = int(item["id"])
        en = item.get("en", "")
        kw_list = []
        if en:
            kw_list.extend([en, en.lower(), en.replace(" ", ""), en.lower().replace(" ", "")])
        kw_list.extend(zh.get(vid, []))
        # normalize
        kw[vid] = list({ _normalize_name(k) for k in kw_list if str(k).strip() })
    return kw


def infer_voc_id_from_name(name: str, keywords: Dict[int, List[str]]) -> Optional[int]:
    nn = _normalize_name(name)
    for vid, kw_list in keywords.items():
        for k in kw_list:
            if k and k in nn:
                return vid
    return None


def _read_two_col_txt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    # Handles whitespace-separated columns, first line is header.
    data = np.loadtxt(str(path), skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    e = data[:, 0].astype(float)
    y = data[:, 1].astype(float)
    return e, y


def _read_pdos_tot(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    # First line header with many columns; last column is 'tot' DOS.
    data = np.loadtxt(str(path), skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    e = data[:, 0].astype(float)
    y = data[:, -1].astype(float)
    return e, y


def _integral_in_window(e: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    mask = (e >= lo) & (e <= hi)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapz(y[mask], e[mask]))


def _interp_at_zero(e: np.ndarray, y: np.ndarray) -> float:
    # Robust interpolation even if 0 is out of bounds.
    if len(e) < 2:
        return float(y[0]) if len(y) else 0.0
    # Ensure ascending for np.interp
    if e[0] > e[-1]:
        e = e[::-1]
        y = y[::-1]
    return float(np.interp(0.0, e, y, left=y[0], right=y[-1]))


def extract_dos_features(e: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return 4 features:
      0: integral_-2_0
      1: integral_0_2
      2: integral_-0.5_0.5
      3: dos_at_0 (interpolated)
    """
    feats = [
        _integral_in_window(e, y, -2.0, 0.0),
        _integral_in_window(e, y, 0.0, 2.0),
        _integral_in_window(e, y, -0.5, 0.5),
        _interp_at_zero(e, y),
    ]
    return np.asarray(feats, dtype=np.float32)



def build_dos_feature_bank(dos_dir: str, device=None, dtype=None) -> torch.Tensor:
    """
    Scan dos_dir which contains 7 subfolders (one per VOC).
    Each subfolder contains PDOS files:
      - C_pdos.txt, H_pdos.txt, O_pdos.txt, Sn_pdos.txt
    TDOS.txt is expected for each VOC folder. A fallback path remains in code to approximate TDOS from element PDOS totals if needed.

    Return: [N_voc=7, D_dos=20] in fixed order:
      [TDOS(4), C(4), H(4), O(4), Sn(4)]
    """
    dos_dir = str(dos_dir)
    if not os.path.isdir(dos_dir):
        raise FileNotFoundError(f"DOS directory not found: {dos_dir}")

    keywords = _build_filename_keywords()
    D_PER = 4
    elems = [("C", "C_pdos.txt"), ("H", "H_pdos.txt"), ("O", "O_pdos.txt"), ("Sn", "Sn_pdos.txt")]
    D = D_PER * (1 + len(elems))  # TDOS + 4 elems
    bank = np.zeros((len(VOC_KNOWLEDGE_BASE), D), dtype=np.float32)
    found = set()

    for entry in Path(dos_dir).iterdir():
        if not entry.is_dir():
            continue
        vid = infer_voc_id_from_name(entry.name, keywords)
        if vid is None:
            continue

        # Read element PDOS (required)
        elem_data = {}
        for el, fname in elems:
            fpath = entry / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Missing PDOS file for VOC id {vid}: {fpath}")
            e_el, y_el = _read_pdos_tot(fpath)
            elem_data[el] = (e_el, y_el)

        # TDOS: read TDOS.txt (fallback: approximate by summing element PDOS totals)
        tdos_path = entry / "TDOS.txt"
        if tdos_path.exists():
            e_t, y_t = _read_two_col_txt(tdos_path)
        else:
            # Use Sn energy grid as reference (usually densest / most stable)
            e_ref, _ = elem_data["Sn"]
            # Ensure ascending
            if e_ref[0] > e_ref[-1]:
                e_ref = e_ref[::-1]
            y_sum = np.zeros_like(e_ref, dtype=np.float64)
            for el, (e_el, y_el) in elem_data.items():
                e2, y2 = e_el, y_el
                if e2[0] > e2[-1]:
                    e2 = e2[::-1]
                    y2 = y2[::-1]
                y_interp = np.interp(e_ref, e2, y2, left=y2[0], right=y2[-1])
                y_sum += y_interp
            e_t = e_ref.astype(float)
            y_t = y_sum.astype(float)

        feats_all = [extract_dos_features(e_t, y_t)]
        for el, _ in elems:
            e_el, y_el = elem_data[el]
            feats_all.append(extract_dos_features(e_el, y_el))

        bank[vid, :] = np.concatenate(feats_all, axis=0)
        found.add(vid)

    missing = sorted(set(range(len(VOC_KNOWLEDGE_BASE))) - found)
    if missing:
        raise RuntimeError(f"Missing DOS folders/files for VOC ids: {missing}. Check folder names under: {dos_dir}")

    t = torch.from_numpy(bank)
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t

