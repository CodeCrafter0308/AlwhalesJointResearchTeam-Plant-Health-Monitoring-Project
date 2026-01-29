#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch align 8-channel dielectric excitation scan cycles in CSV files.

Assumed input structure
- A time column in column 1
- Eight raw sensor channels in columns 2 to 9, or columns named like:
  Chan.1 - Raw (mV) ... Chan.8 - Raw (mV)

What this script does
1. Estimates the scan cycle length in samples using autocorrelation on a robust derivative feature
2. Detects cycle reset points using large negative jumps in each channel
3. Uses reset points to compute per-channel integer sample shifts that align scan cycles in time
4. Applies a global anchor so the maximum absolute movement is minimized
5. Optionally performs a small residual refinement by cross-correlation
6. Trims the output to the common valid region where all 8 channels and time are present

Output
- One CSV per input file, containing only:
  time + 8 aligned raw channels
- The output is trimmed, so there are no NaN edges

Example
python align_8ch_batch_v3.py --input "D:\data\all_csv" --output_dir "D:\data\out" --dt 0.5
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def robust_zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    scale = mad / 0.6745
    if (not np.isfinite(scale)) or scale == 0:
        scale = np.nanstd(v) + 1e-12
    return (v - med) / (scale + 1e-12)


def preprocess_for_alignment(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        x = np.where(np.isfinite(x), x, np.nanmedian(x))
    dx = np.diff(x, prepend=x[0])
    return robust_zscore(dx)


def fft_xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Full cross-correlation using FFT.
    Returns length 2*n - 1 with lags from -(n-1) to +(n-1).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    cc = np.fft.irfft(A * np.conj(B), nfft)
    cc = np.concatenate([cc[-(n - 1):], cc[:n]])
    return cc


def best_lag_to_ref(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    """
    Find lag that maximizes sum a[n] * b[n + lag] within [-max_lag, max_lag].
    Positive lag means a is earlier and should be delayed by lag samples.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    cc = fft_xcorr(a, b)
    lags = np.arange(-(n - 1), n)
    mask = (lags >= -max_lag) & (lags <= max_lag)
    ccw = cc[mask]
    lw = lags[mask]
    idx = int(np.nanargmax(ccw))
    return int(lw[idx]), float(ccw[idx])


def estimate_period_from_autocorr(z: np.ndarray, min_period: int, max_period: int) -> int:
    n = len(z)
    max_period = int(min(max_period, n - 2))
    min_period = int(max(2, min_period))
    if max_period <= min_period:
        return int(max(10, min_period))

    ac = fft_xcorr(z, z)
    lags = np.arange(-(n - 1), n)

    pos = lags > 0
    ac_pos = ac[pos]
    lags_pos = lags[pos]

    win = (lags_pos >= min_period) & (lags_pos <= max_period)
    if not np.any(win):
        return int(min_period)

    ac_win = ac_pos[win]
    lags_win = lags_pos[win]
    idx = int(np.nanargmax(ac_win))
    return int(lags_win[idx])


def find_reset_peaks(x: np.ndarray, period: int, k_mad: float, min_sep_ratio: float) -> np.ndarray:
    """
    Detect scan reset points via large negative jumps in dx.
    Threshold is median(dx) - k_mad * MAD(dx).
    Contiguous candidates are collapsed to the most negative point.
    A minimum separation is enforced to yield one reset per cycle.
    """
    x = np.asarray(x, dtype=float)
    dx = np.diff(x, prepend=x[0])

    med = float(np.nanmedian(dx))
    mad = _mad(dx) + 1e-12
    thr = med - float(k_mad) * mad

    cand = np.where(dx < thr)[0]
    if cand.size == 0:
        return np.array([], dtype=int)

    groups = []
    start = int(cand[0])
    prev = int(cand[0])
    for idx in cand[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            groups.append((start, prev))
            start = idx
            prev = idx
    groups.append((start, prev))

    peaks = []
    for a, b in groups:
        seg = dx[a:b + 1]
        off = int(np.nanargmin(seg))
        peaks.append(a + off)

    peaks = np.array(sorted(set(peaks)), dtype=int)

    min_sep = max(1, int(period * float(min_sep_ratio)))
    order = peaks[np.argsort(dx[peaks])]  # most negative first
    selected = []
    for idx in order:
        if all(abs(idx - s) >= min_sep for s in selected):
            selected.append(int(idx))
    return np.array(sorted(selected), dtype=int)


def match_boundary_shift_constrained(B_ref: np.ndarray, B_k: np.ndarray, period: int, d_window: int) -> tuple[int, int, float]:
    """
    Align boundary index sequences with a small cycle offset search.
    Returns shift_samples, cycle_offset, residual_mad.

    cycle_offset d means B_k index i+d is matched to B_ref index i.
    shift is median(B_ref - B_k_matched) in samples.
    """
    B_ref = np.asarray(B_ref, dtype=int)
    B_k = np.asarray(B_k, dtype=int)
    if B_ref.size < 10 or B_k.size < 10:
        raise RuntimeError("Not enough reset points detected for matching.")

    delta0 = int(B_ref[0] - B_k[0])
    guess_d = int(round(delta0 / float(period)))

    best = None
    for d in range(guess_d - int(d_window), guess_d + int(d_window) + 1):
        if d >= 0:
            n = min(len(B_ref), len(B_k) - d)
            if n < 10:
                continue
            i_ref = np.arange(0, n)
            i_k = i_ref + d
        else:
            n = min(len(B_k), len(B_ref) + d)
            if n < 10:
                continue
            i_k = np.arange(0, n)
            i_ref = i_k - d

        diffs = B_ref[i_ref] - B_k[i_k]
        med = float(np.median(diffs))
        madv = float(np.median(np.abs(diffs - med)))

        cand = (madv, abs(med), abs(d), d, med)
        if best is None or cand < best:
            best = cand

    if best is None:
        raise RuntimeError("Boundary matching failed.")

    madv, _, _, d, med = best
    return int(round(med)), int(d), float(madv)


def choose_anchor(shifts_to_ref: np.ndarray, mode: str) -> int:
    shifts_to_ref = np.asarray(shifts_to_ref, dtype=int)
    if mode == "median":
        return int(np.median(shifts_to_ref))
    if mode == "midrange":
        lo = int(np.min(shifts_to_ref))
        hi = int(np.max(shifts_to_ref))
        return int(np.floor((lo + hi) / 2))
    raise ValueError("anchor mode must be median or midrange")


def shift_with_nan(x: np.ndarray, s: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = np.full(n, np.nan, dtype=float)
    if s == 0:
        y[:] = x
    elif s > 0:
        y[s:] = x[:n - s]
    else:
        y[:n + s] = x[-s:]
    return y


def trim_common_valid(time: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    time shape N
    Y shape 8, N
    """
    finite = np.isfinite(time)
    finite &= np.all(np.isfinite(Y), axis=0)
    idx = np.where(finite)[0]
    if idx.size == 0:
        raise RuntimeError("No overlap region after shifting.")
    start = int(idx[0])
    end = int(idx[-1]) + 1
    return time[start:end], Y[:, start:end]


def detect_channel_columns(df: pd.DataFrame) -> tuple[str, list[str]]:
    """
    Returns time_col, ch_cols for raw channels in order 1..8.
    If raw channel columns are not found by name, uses columns 2..9.
    """
    cols = list(df.columns)
    time_col = cols[0]

    pat = re.compile(r"chan\.(\d+)\s*-\s*raw", re.IGNORECASE)
    candidates = []
    for c in cols:
        m = pat.search(str(c))
        if m:
            candidates.append((int(m.group(1)), c))
    if len(candidates) >= 8:
        candidates.sort(key=lambda t: t[0])
        ch_cols = [c for _, c in candidates[:8]]
        return time_col, ch_cols

    if df.shape[1] >= 9:
        ch_cols = cols[1:9]
        return time_col, ch_cols

    raise ValueError("Cannot identify 8 channel columns.")


def align_one_file(
    input_path: Path,
    output_path: Path,
    ref_index: int,
    dt_seconds: float | None,
    min_period: int,
    max_period: int,
    k_mad: float,
    min_sep_ratio: float,
    d_window: int,
    anchor_mode: str,
    refine_lag: int
) -> dict:
    df = pd.read_csv(input_path)
    time_col, ch_cols = detect_channel_columns(df)

    time = df[time_col].to_numpy(dtype=float)
    X = np.vstack([df[c].to_numpy(dtype=float) for c in ch_cols])

    Z = np.vstack([preprocess_for_alignment(X[k]) for k in range(8)])
    period = estimate_period_from_autocorr(Z[ref_index], min_period=min_period, max_period=max_period)

    B_ref = find_reset_peaks(X[ref_index], period=period, k_mad=k_mad, min_sep_ratio=min_sep_ratio)
    if B_ref.size < 10:
        raise RuntimeError("Reset detection failed on reference channel. Try smaller k_mad or different ref.")

    shifts_to_ref = np.zeros(8, dtype=int)
    cycle_offsets = np.zeros(8, dtype=int)
    residuals = np.zeros(8, dtype=float)

    for k in range(8):
        B_k = find_reset_peaks(X[k], period=period, k_mad=k_mad, min_sep_ratio=min_sep_ratio)
        if B_k.size < 10:
            raise RuntimeError(f"Reset detection failed on channel {k+1}. Try smaller k_mad.")
        s, d, madv = match_boundary_shift_constrained(B_ref, B_k, period=period, d_window=d_window)
        shifts_to_ref[k] = int(s)
        cycle_offsets[k] = int(d)
        residuals[k] = float(madv)

    anchor = choose_anchor(shifts_to_ref, mode=anchor_mode)
    applied = shifts_to_ref - int(anchor)

    # Optional residual refinement using small-lag cross-correlation after coarse alignment
    if refine_lag > 0:
        Y0 = np.vstack([shift_with_nan(X[k], int(applied[k])) for k in range(8)])
        t0 = shift_with_nan(time, int(applied[ref_index]))
        t_trim, Y_trim = trim_common_valid(t0, Y0)
        Z_trim = np.vstack([preprocess_for_alignment(Y_trim[k]) for k in range(8)])
        extra = np.zeros(8, dtype=int)
        for k in range(8):
            lag, _ = best_lag_to_ref(Z_trim[k], Z_trim[ref_index], max_lag=int(refine_lag))
            extra[k] = int(lag)
        applied = applied + extra

    # Apply final shifts and trim
    Y = np.vstack([shift_with_nan(X[k], int(applied[k])) for k in range(8)])
    t_aligned = shift_with_nan(time, int(applied[ref_index]))
    t_out, Y_out = trim_common_valid(t_aligned, Y)

    out = pd.DataFrame({time_col: t_out})
    for k in range(8):
        out[ch_cols[k]] = Y_out[k]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    if dt_seconds is not None:
        shift_seconds = (applied.astype(float) * float(dt_seconds)).tolist()
    else:
        shift_seconds = None

    return {
        "input": str(input_path),
        "output": str(output_path),
        "period_samples": int(period),
        "reset_count_ref": int(len(B_ref)),
        "shift_to_ref_samples": shifts_to_ref.tolist(),
        "anchor_samples": int(anchor),
        "applied_shift_samples": applied.tolist(),
        "applied_shift_seconds": shift_seconds,
        "cycle_offsets": cycle_offsets.tolist(),
        "boundary_residual_mad": residuals.tolist(),
        "trimmed_length": int(len(t_out)),
    }


def collect_csvs(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.glob("*.csv") if x.is_file()])
    raise FileNotFoundError(f"Input not found: {p}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Align 8-channel scan cycles and export trimmed aligned CSV.")
    ap.add_argument("--input", required=True, help="Input CSV file path or a folder containing CSV files")
    ap.add_argument("--output_dir", required=True, help="Output folder for aligned CSV files")
    ap.add_argument("--suffix", default="_aligned_trimmed", help="Suffix for output file names")
    ap.add_argument("--ref", type=int, default=1, help="Reference channel index 1 to 8")
    ap.add_argument("--dt", type=float, default=None, help="Sample interval in seconds, used only for printing shifts in seconds")

    ap.add_argument("--min_period", type=int, default=30, help="Min period in samples for scan cycle estimation")
    ap.add_argument("--max_period", type=int, default=2000, help="Max period in samples for scan cycle estimation")

    ap.add_argument("--k_mad", type=float, default=50.0, help="Reset detection threshold strength, larger means stricter")
    ap.add_argument("--min_sep_ratio", type=float, default=0.70, help="Minimum separation ratio times period for reset peaks")
    ap.add_argument("--d_window", type=int, default=3, help="Cycle offset search window around an initial guess")

    ap.add_argument("--anchor", choices=["midrange", "median"], default="midrange", help="How to choose the global anchor")
    ap.add_argument("--refine_lag", type=int, default=20, help="Residual refinement max lag in samples, set 0 to disable")

    args = ap.parse_args()

    ref_index = int(args.ref) - 1
    if ref_index < 0 or ref_index > 7:
        raise ValueError("ref must be between 1 and 8")

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    files = collect_csvs(in_path)
    if not files:
        print("No CSV files found.")
        return 0

    print(f"Found {len(files)} file(s).")

    for p in files:
        out_name = p.stem + args.suffix + ".csv"
        out_path = out_dir / out_name
        info = align_one_file(
            input_path=p,
            output_path=out_path,
            ref_index=ref_index,
            dt_seconds=args.dt,
            min_period=int(args.min_period),
            max_period=int(args.max_period),
            k_mad=float(args.k_mad),
            min_sep_ratio=float(args.min_sep_ratio),
            d_window=int(args.d_window),
            anchor_mode=str(args.anchor),
            refine_lag=int(args.refine_lag),
        )

        print("")
        print(f"Input:  {info['input']}")
        print(f"Output: {info['output']}")
        print(f"Period samples: {info['period_samples']}")
        print(f"Reset points ref: {info['reset_count_ref']}")
        print(f"Applied shifts samples: {info['applied_shift_samples']}")
        if info["applied_shift_seconds"] is not None:
            print(f"Applied shifts seconds: {info['applied_shift_seconds']}")
        print(f"Trimmed length: {info['trimmed_length']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
