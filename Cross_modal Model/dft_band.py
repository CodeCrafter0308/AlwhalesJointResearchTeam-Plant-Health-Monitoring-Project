
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from voc_registry import VOC_KNOWLEDGE_BASE


def _normalize_name(s: str) -> str:
    s = str(s)
    s = s.strip().lower()
    s = s.replace(" ", "").replace("_", "").replace("-", "")
    return s


def _build_filename_keywords() -> Dict[int, List[str]]:
    """
    VOC id -> filename keywords list.
    Includes Chinese abbreviations and English names.
    """
    # Chinese keywords based on user file names.
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
        en = _normalize_name(item["en"])
        kw_list = [en]
        # add simplified english tokens
        if "aceticacid" in en:
            kw_list += ["acetic"]
        if "decanoicacid" in en:
            kw_list += ["decanoic"]
        if "2octanone" in en:
            kw_list += ["2octanone", "octanone"]
        kw_list += zh.get(vid, [])
        kw[vid] = list(dict.fromkeys(kw_list))  # unique preserve order
    return kw


def _discover_band_files(band_dir: Union[str, Path]) -> Dict[int, Path]:
    band_dir = Path(band_dir)
    if not band_dir.exists():
        raise FileNotFoundError(f"DFT band dir not found: {band_dir}")
    files = [p for p in band_dir.rglob("*") if p.is_file() and p.suffix.lower() in [".csv", ".xlsx", ".xls"]]
    if len(files) == 0:
        raise FileNotFoundError(f"No band CSV/XLSX files found under: {band_dir}")

    kw_map = _build_filename_keywords()
    found: Dict[int, Path] = {}

    for p in files:
        fname = p.name
        fkey = _normalize_name(fname)
        for vid, kws in kw_map.items():
            if vid in found:
                continue
            for k in kws:
                if k == "":
                    continue
                if _normalize_name(k) in fkey:
                    found[vid] = p
                    break

    # Ensure all ids are present
    missing = [vid for vid in range(len(VOC_KNOWLEDGE_BASE)) if vid not in found]
    if missing:
        # try looser match: check stem tokens
        for p in files:
            stem = _normalize_name(p.stem)
            for vid in missing[:]:
                for k in kw_map[vid]:
                    if _normalize_name(k) in stem:
                        found[vid] = p
                        missing.remove(vid)
                        break

    if missing:
        raise RuntimeError(
            f"Band files missing for VOC ids: {missing}. "
            f"Expected 7 files with names containing VOC keywords."
        )
    return found


def _read_band_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported band file type: {path}")


def extract_band_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract fixed-length features from a band structure table.

    Assumptions:
      - First column is k-path coordinate (monotonic).
      - Remaining columns are band energies (eV) aligned to Ef=0 (recommended).
    Output:
      - feature vector of length 15.
    """
    if df.shape[1] < 3:
        raise ValueError("Band table must have at least 1 k column + >=2 band columns.")

    k = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    E = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Drop rows with NaN in k
    valid = np.isfinite(k)
    k = k[valid]
    E = E[valid, :]

    # Basic sanitization: replace NaN energies with inf so they won't be selected
    E = np.where(np.isfinite(E), E, np.nan)

    n_k, n_b = E.shape
    if n_k < 5:
        raise ValueError("Too few k-points to extract curvature features.")

    # Per-k VBM/CBM relative to 0 eV
    vbm_k = np.full((n_k,), np.nan, dtype=float)
    cbm_k = np.full((n_k,), np.nan, dtype=float)
    for i in range(n_k):
        row = E[i, :]
        if not np.isfinite(row).any():
            continue
        le0 = row[np.isfinite(row) & (row <= 0.0)]
        ge0 = row[np.isfinite(row) & (row >= 0.0)]
        if le0.size > 0:
            vbm_k[i] = le0.max()
        if ge0.size > 0:
            cbm_k[i] = ge0.min()

    # Fallback if alignment is imperfect (no <=0 or >=0)
    if np.all(np.isnan(vbm_k)) or np.all(np.isnan(cbm_k)):
        # use median split
        flat = E[np.isfinite(E)]
        if flat.size == 0:
            raise ValueError("Band energies are all NaN.")
        mid = np.median(flat)
        for i in range(n_k):
            row = E[i, :]
            row = row[np.isfinite(row)]
            if row.size == 0:
                continue
            vbm_k[i] = row[row <= mid].max() if np.any(row <= mid) else row.min()
            cbm_k[i] = row[row >= mid].min() if np.any(row >= mid) else row.max()

    gap_k = cbm_k - vbm_k
    # Global
    vbm = np.nanmax(vbm_k)
    cbm = np.nanmin(cbm_k)
    eg = cbm - vbm

    # Identify VBM band and CBM band at their k indices
    k_idx_vbm = int(np.nanargmax(vbm_k))
    k_idx_cbm = int(np.nanargmin(cbm_k))

    # Find band index at those k points
    row_v = E[k_idx_vbm, :]
    row_c = E[k_idx_cbm, :]
    # valence band: closest to vbm from below
    val_mask = np.isfinite(row_v) & (row_v <= 0.0)
    if not np.any(val_mask):
        val_mask = np.isfinite(row_v)
    val_band_idx = int(np.nanargmax(np.where(val_mask, row_v, -np.inf)))

    cond_mask = np.isfinite(row_c) & (row_c >= 0.0)
    if not np.any(cond_mask):
        cond_mask = np.isfinite(row_c)
    cond_band_idx = int(np.nanargmin(np.where(cond_mask, row_c, np.inf)))

    val_band = E[:, val_band_idx]
    cond_band = E[:, cond_band_idx]

    val_disp = np.nanmax(val_band) - np.nanmin(val_band)
    cond_disp = np.nanmax(cond_band) - np.nanmin(cond_band)

    def _safe_derivatives(band: np.ndarray, idx: int) -> Tuple[float, float]:
        # first derivative and second derivative w.r.t. k
        idx = int(idx)
        if idx <= 0:
            idx = 1
        if idx >= n_k - 1:
            idx = n_k - 2
        dk1 = k[idx] - k[idx - 1]
        dk2 = k[idx + 1] - k[idx]
        dk = (dk1 + dk2) / 2.0 if (dk1 > 0 and dk2 > 0) else (k[idx + 1] - k[idx - 1]) / 2.0
        if dk == 0:
            dk = 1.0

        e_prev = band[idx - 1]
        e_mid = band[idx]
        e_next = band[idx + 1]
        if not (np.isfinite(e_prev) and np.isfinite(e_mid) and np.isfinite(e_next)):
            return float("nan"), float("nan")

        slope = (e_next - e_prev) / (2.0 * dk)
        curv = (e_next - 2.0 * e_mid + e_prev) / (dk * dk)
        return float(slope), float(curv)

    val_slope, val_curv = _safe_derivatives(val_band, k_idx_vbm)
    cond_slope, cond_curv = _safe_derivatives(cond_band, k_idx_cbm)

    # Symmetry-ish points along path: first, ~1/3, ~2/3, last
    idx1 = max(int(round(n_k / 3)) - 1, 0)
    idx2 = max(int(round(2 * n_k / 3)) - 1, 0)
    sym_idx = [0, idx1, idx2, n_k - 1]
    gap_sym = [float(gap_k[i]) if np.isfinite(gap_k[i]) else float("nan") for i in sym_idx]

    gap_mean = float(np.nanmean(gap_k))
    gap_std = float(np.nanstd(gap_k))

    feat = np.array(
        [vbm, cbm, eg, val_disp, cond_disp, val_slope, cond_slope, val_curv, cond_curv,
         gap_sym[0], gap_sym[1], gap_sym[2], gap_sym[3], gap_mean, gap_std],
        dtype=np.float32
    )
    # Replace any remaining nan with 0 to keep training stable
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feat


def build_band_feature_bank(band_dir: Union[str, Path], device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Build band-structure feature bank: [N_voc=7, D_band=15].

    Ensures VOC id alignment by matching filenames against VOC_KNOWLEDGE_BASE keywords.
    """
    file_map = _discover_band_files(band_dir)
    feats = []
    for vid in range(len(VOC_KNOWLEDGE_BASE)):
        path = file_map[vid]
        df = _read_band_table(path)
        f = extract_band_features(df)
        feats.append(f)
    bank = torch.tensor(np.stack(feats, axis=0), dtype=dtype)
    if device is not None:
        bank = bank.to(device)
    return bank
