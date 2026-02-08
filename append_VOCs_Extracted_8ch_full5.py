#!/usr/bin/env python3
# append_VOCs_Extracted_8ch_full5.py
#
# Data layout, 1-based columns
# 1: time
# 2..9: Mix responses for sensor 1..8
# 10..17: Ethanol+Water responses for sensor 1..8
# 18..25: Pure water responses for sensor 1..8
#
# Outputs
# - segment_XX.csv
# - segment_XX_8ch_with_VOCs_Extracted.csv with VOCs_Extracted_S1..S8 appended
# - all_5_segments_8ch_concatenated_with_VOCs_Extracted.csv
# - summary.csv and results_8ch.xlsx
#
# Example
# python append_VOCs_Extracted_8ch_full5.py --input "data.csv" --output_dir "out" --n_segments 5

import argparse
import os
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, medfilt, find_peaks


def robust_mad_scale(a: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float)
    med = np.nanmedian(a)
    mad = np.nanmedian(np.abs(a - med))
    return 1.4826 * mad + eps


def whittaker_smooth(
    y: np.ndarray,
    lam: float = 2e4,
    order: int = 2,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = y.size
    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)

    D = sparse.eye(n, format="csc")
    for _ in range(order):
        D = D[1:] - D[:-1]
    P = (D.T @ D).tocsc()

    W = sparse.diags(np.clip(w, 1e-12, None), 0, format="csc")
    z = spsolve(W + lam * P, w * y)
    return np.asarray(z)


def _odd_clip(k: int, n: int, kmin: int = 3) -> int:
    k = int(k)
    if k % 2 == 0:
        k += 1
    k = max(kmin, k)
    if n <= 3:
        return 3
    kmax = n - 1 if (n - 1) % 2 == 1 else n - 2
    k = min(k, kmax)
    if k < kmin:
        k = kmin
    if k % 2 == 0:
        k += 1
    return k


def smooth_two_stage(
    y: np.ndarray,
    med_kernel: int = 5,
    sg_window: int = 21,
    sg_poly: int = 3,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 7:
        return y.copy()

    k = _odd_clip(med_kernel, n, kmin=3)
    y1 = medfilt(y, kernel_size=k)

    wlen = _odd_clip(sg_window, n, kmin=max(7, sg_poly + 3))
    if wlen <= sg_poly + 1:
        wlen = _odd_clip(sg_poly + 3, n, kmin=7)

    return savgol_filter(y1, window_length=wlen, polyorder=sg_poly, mode="interp")


def smooth_light(
    y: np.ndarray,
    sg_window: int = 11,
    sg_poly: int = 2,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 7:
        return y.copy()

    wlen = _odd_clip(sg_window, n, kmin=max(7, sg_poly + 3))
    if wlen <= sg_poly + 1:
        wlen = _odd_clip(sg_poly + 3, n, kmin=7)

    return savgol_filter(y, window_length=wlen, polyorder=sg_poly, mode="interp")


def find_cycle_boundaries_by_reset(
    water_curve: np.ndarray,
    thr_mult: float = 6.0,
    min_dist: int = 120,
    min_seg_len: int = 80,
) -> List[int]:
    water_det = smooth_light(water_curve, sg_window=11, sg_poly=2)
    d = np.diff(water_det)
    thr = thr_mult * robust_mad_scale(d)

    peaks, _ = find_peaks(-d, height=thr, distance=int(min_dist))
    b = [0] + [int(p + 1) for p in peaks] + [water_curve.size]
    b = sorted(set(b))

    clean = [b[0]]
    for x in b[1:]:
        if x - clean[-1] >= int(min_seg_len):
            clean.append(x)
    if clean[-1] != water_curve.size:
        clean.append(water_curve.size)
    return clean


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    raise ValueError("Unsupported file type, please use xlsx or csv")


def extract_vocs_from_arrays(
    mix: np.ndarray,
    bg: np.ndarray,
    water: np.ndarray,
    thr_mult: float = 6.0,
    min_dist: int = 120,
    min_seg_len: int = 80,
    lam_water: float = 6e4,
    lam_delta: float = 3e3,
    lambda_min: float = 0.2,
    lambda_max: float = 2.0,
    med_kernel: int = 5,
    sg_window_pre: int = 21,
    sg_poly_pre: int = 3,
    sg_window_post: int = 11,
    sg_poly_post: int = 2,
) -> Tuple[np.ndarray, List[Dict]]:
    mix = np.asarray(mix, dtype=float)
    bg = np.asarray(bg, dtype=float)
    water = np.asarray(water, dtype=float)

    boundaries = find_cycle_boundaries_by_reset(
        water,
        thr_mult=thr_mult,
        min_dist=min_dist,
        min_seg_len=min_seg_len,
    )
    if len(boundaries) < 2:
        boundaries = [0, len(mix)]

    vocs = np.full(mix.size, np.nan, dtype=float)
    meta: List[Dict] = []

    for ci in range(len(boundaries) - 1):
        s = boundaries[ci]
        e = boundaries[ci + 1]
        if e - s < 30:
            meta.append({"cycle": ci + 1, "start_idx": s, "end_idx": e, "lambda": np.nan, "skipped": True})
            continue

        mix_s = smooth_two_stage(mix[s:e], med_kernel, sg_window_pre, sg_poly_pre)
        bg_s = smooth_two_stage(bg[s:e], med_kernel, sg_window_pre, sg_poly_pre)
        water_s = smooth_two_stage(water[s:e], med_kernel, sg_window_pre, sg_poly_pre)

        Zw = whittaker_smooth(water_s, lam=lam_water, order=2)

        delta_pos = np.maximum(0.0, mix_s - bg_s)
        delta_pos_sm = whittaker_smooth(delta_pos, lam=lam_delta, order=2)

        bg_strength = np.abs(bg_s - Zw)
        med_bg = float(np.nanmedian(bg_strength))
        med_d = float(np.nanmedian(delta_pos_sm)) + 1e-12

        lam = 0.3 * (med_bg / med_d)
        lam = float(np.clip(lam, lambda_min, lambda_max))

        pure = mix_s + lam * delta_pos_sm
        pure = smooth_light(pure, sg_window_post, sg_poly_post)

        vocs[s:e] = pure
        meta.append({"cycle": ci + 1, "start_idx": s, "end_idx": e, "lambda": lam, "skipped": False})

    return vocs, meta


def _segment_cost_precompute(X: np.ndarray):
    n, p = X.shape
    cs = np.vstack([np.zeros((1, p)), np.cumsum(X, axis=0)])
    cs2 = np.hstack([0.0, np.cumsum(np.sum(X * X, axis=1))])
    return cs, cs2


def _seg_cost(cs: np.ndarray, cs2: np.ndarray, i: int, j: int) -> float:
    m = j - i
    if m <= 0:
        return np.inf
    s = cs[j] - cs[i]
    mean = s / m
    s2 = cs2[j] - cs2[i]
    return float(s2 - m * np.sum(mean * mean))


def segment_sweeps_into_n_blocks(sweep_features: np.ndarray, n_blocks: int, min_block_len: int) -> List[int]:
    X = np.asarray(sweep_features, dtype=float)
    n, _ = X.shape
    K = int(n_blocks)
    if K <= 1:
        return [0, n]
    if n < K * min_block_len:
        min_block_len = max(1, n // K)

    cs, cs2 = _segment_cost_precompute(X)
    dp = np.full((K + 1, n + 1), np.inf)
    prev = np.full((K + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for k in range(1, K + 1):
        for j in range(1, n + 1):
            best = np.inf
            best_i = -1
            for i in range(0, j):
                if j - i < min_block_len:
                    continue
                if k == 1 and i != 0:
                    continue
                if dp[k - 1, i] == np.inf:
                    continue
                c = dp[k - 1, i] + _seg_cost(cs, cs2, i, j)
                if c < best:
                    best = c
                    best_i = i
            dp[k, j] = best
            prev[k, j] = best_i

    cuts = [n]
    j = n
    for k in range(K, 0, -1):
        i = prev[k, j]
        if i < 0:
            break
        cuts.append(i)
        j = i
    cuts = sorted(cuts)
    if cuts[0] != 0:
        cuts = [0] + cuts
    if cuts[-1] != n:
        cuts.append(n)
    cuts = sorted(set(cuts))
    return cuts


def split_full_into_segments(
    df: pd.DataFrame,
    time_col: str,
    mix_cols: List[str],
    bg_cols: List[str],
    water_cols: List[str],
    n_segments: int,
    min_sweeps_per_segment: int,
    thr_mult: float,
    min_dist: int,
    min_seg_len: int,
) -> Tuple[List[pd.DataFrame], List[Dict]]:
    water_mat = np.vstack([pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in water_cols])
    water_mean = np.nanmean(water_mat, axis=0)
    time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)

    sweep_bounds = find_cycle_boundaries_by_reset(
        water_mean,
        thr_mult=thr_mult,
        min_dist=min_dist,
        min_seg_len=min_seg_len,
    )
    if len(sweep_bounds) < 3:
        return [df.copy()], [{"segment": 1, "start_idx": 0, "end_idx": len(df), "rows": len(df)}]

    mix_mat = np.vstack([pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in mix_cols])
    bg_mat = np.vstack([pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in bg_cols])
    mix_mean = np.nanmean(mix_mat, axis=0)
    bg_mean = np.nanmean(bg_mat, axis=0)

    feats = []
    for i in range(len(sweep_bounds) - 1):
        s, e = sweep_bounds[i], sweep_bounds[i + 1]
        feats.append([
            np.nanmean(mix_mean[s:e]), np.nanstd(mix_mean[s:e]),
            np.nanmean(bg_mean[s:e]), np.nanstd(bg_mean[s:e]),
            np.nanmean(water_mean[s:e]), np.nanstd(water_mean[s:e]),
        ])
    feats = np.asarray(feats, dtype=float)

    cuts = segment_sweeps_into_n_blocks(
        feats,
        n_blocks=n_segments,
        min_block_len=min_sweeps_per_segment,
    )

    segments = []
    meta = []
    for seg_i, (a, bcut) in enumerate(zip(cuts[:-1], cuts[1:]), start=1):
        start_idx = sweep_bounds[a]
        end_idx = sweep_bounds[bcut]
        segments.append(df.iloc[start_idx:end_idx].copy())
        meta.append({
            "segment": seg_i,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_time": float(time[start_idx]) if np.isfinite(time[start_idx]) else np.nan,
            "end_time": float(time[end_idx - 1]) if end_idx > start_idx else np.nan,
            "rows": int(end_idx - start_idx),
            "sweeps": int(bcut - a),
        })
    return segments, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--time_idx", type=int, default=0)
    ap.add_argument("--mix_start_idx", type=int, default=1)
    ap.add_argument("--bg_start_idx", type=int, default=9)
    ap.add_argument("--water_start_idx", type=int, default=17)
    ap.add_argument("--n_sensors", type=int, default=8)

    ap.add_argument("--n_segments", type=int, default=5)
    ap.add_argument("--min_sweeps_per_segment", type=int, default=8)

    ap.add_argument("--thr_mult", type=float, default=6.0)
    ap.add_argument("--min_dist", type=int, default=120)
    ap.add_argument("--min_seg_len", type=int, default=80)

    ap.add_argument("--lam_water", type=float, default=6e4)
    ap.add_argument("--lam_delta", type=float, default=3e3)
    ap.add_argument("--lambda_min", type=float, default=0.2)
    ap.add_argument("--lambda_max", type=float, default=2.0)

    ap.add_argument("--med_kernel", type=int, default=5)
    ap.add_argument("--sg_window_pre", type=int, default=21)
    ap.add_argument("--sg_poly_pre", type=int, default=3)
    ap.add_argument("--sg_window_post", type=int, default=11)
    ap.add_argument("--sg_poly_post", type=int, default=2)

    ap.add_argument("--concat_output", default="all_5_segments_8ch_concatenated_with_VOCs_Extracted.csv")

    args = ap.parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    df = read_table(args.input)
    cols = list(df.columns)
    need_cols = args.water_start_idx + args.n_sensors
    if df.shape[1] < need_cols:
        raise ValueError(f"Need at least {need_cols} columns, got {df.shape[1]}")

    time_col = cols[args.time_idx]
    mix_cols = cols[args.mix_start_idx: args.mix_start_idx + args.n_sensors]
    bg_cols = cols[args.bg_start_idx: args.bg_start_idx + args.n_sensors]
    water_cols = cols[args.water_start_idx: args.water_start_idx + args.n_sensors]

    segments, seg_meta = split_full_into_segments(
        df,
        time_col=time_col,
        mix_cols=mix_cols,
        bg_cols=bg_cols,
        water_cols=water_cols,
        n_segments=args.n_segments,
        min_sweeps_per_segment=args.min_sweeps_per_segment,
        thr_mult=args.thr_mult,
        min_dist=args.min_dist,
        min_seg_len=args.min_seg_len,
    )

    summary_rows = []
    concat_parts = []

    for meta, seg_df in zip(seg_meta, segments):
        seg_idx = meta["segment"]
        seg_df.to_csv(os.path.join(args.output_dir, f"segment_{seg_idx:02d}.csv"), index=False)

        seg_out = seg_df.copy()
        for si in range(args.n_sensors):
            mix = pd.to_numeric(seg_df[mix_cols[si]], errors="coerce").to_numpy(float)
            bg = pd.to_numeric(seg_df[bg_cols[si]], errors="coerce").to_numpy(float)
            water = pd.to_numeric(seg_df[water_cols[si]], errors="coerce").to_numpy(float)

            vocs, cycle_meta = extract_vocs_from_arrays(
                mix, bg, water,
                thr_mult=args.thr_mult,
                min_dist=args.min_dist,
                min_seg_len=args.min_seg_len,
                lam_water=args.lam_water,
                lam_delta=args.lam_delta,
                lambda_min=args.lambda_min,
                lambda_max=args.lambda_max,
                med_kernel=args.med_kernel,
                sg_window_pre=args.sg_window_pre,
                sg_poly_pre=args.sg_poly_pre,
                sg_window_post=args.sg_window_post,
                sg_poly_post=args.sg_poly_post,
            )
            seg_out[f"VOCs_Extracted_S{si+1}"] = vocs

            summary_rows.append({
                "segment": seg_idx,
                "sensor": si + 1,
                "rows": int(len(seg_df)),
                "cycles_detected": int(len(cycle_meta)),
                "nan_count": int(np.isnan(vocs).sum()),
                "min": float(np.nanmin(vocs)),
                "max": float(np.nanmax(vocs)),
                "mean": float(np.nanmean(vocs)),
                "std": float(np.nanstd(vocs)),
            })

        seg_out.to_csv(os.path.join(args.output_dir, f"segment_{seg_idx:02d}_8ch_with_VOCs_Extracted.csv"), index=False)

        seg_out2 = seg_out.copy()
        seg_out2.insert(0, "SegmentIndex", seg_idx)
        seg_out2.insert(1, "SourceSegment", f"segment_{seg_idx:02d}")
        concat_parts.append(seg_out2)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)

    concat_df = pd.concat(concat_parts, axis=0, ignore_index=True)
    concat_path = args.concat_output
    if not os.path.isabs(concat_path):
        concat_path = os.path.join(args.output_dir, concat_path)
    concat_df.to_csv(concat_path, index=False)

    xlsx_path = os.path.join(args.output_dir, "results_8ch.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        concat_df.to_excel(writer, sheet_name="concat", index=False)
        for seg_idx in range(1, len(segments) + 1):
            seg_file = os.path.join(args.output_dir, f"segment_{seg_idx:02d}_8ch_with_VOCs_Extracted.csv")
            if os.path.exists(seg_file):
                pd.read_csv(seg_file).to_excel(writer, sheet_name=f"seg_{seg_idx:02d}", index=False)


if __name__ == "__main__":
    main()
