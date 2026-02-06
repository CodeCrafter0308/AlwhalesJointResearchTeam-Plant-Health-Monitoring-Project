#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd

def estimate_period_samples(x, dt, min_s=30.0, max_s=200.0):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    n = len(x)
    x0 = np.nan_to_num(x)
    f = np.fft.rfft(x0, n=2*n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] != 0:
        ac = ac / ac[0]
    min_lag = int(round(min_s / dt))
    max_lag = int(round(max_s / dt))
    max_lag = min(max_lag, n - 1)
    seg = ac[min_lag:max_lag + 1]
    lag = min_lag + int(np.argmax(seg))
    return lag

def cycle_average(x, Np):
    x = np.asarray(x, dtype=float)
    n_cycles = len(x) // Np
    x_use = x[:n_cycles * Np]
    X = x_use.reshape(n_cycles, Np)
    return np.nanmean(X, axis=0)

def circular_lag_to_align(a, b):
    # find lag k such that roll(a, -k) best matches b
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    fa = np.fft.fft(a0)
    fb = np.fft.fft(b0)
    cc = np.fft.ifft(fa * np.conj(fb)).real
    k = int(np.argmax(cc))
    N = len(a)
    if k > N // 2:
        k = k - N
    return k

def shift_with_nan(x, shift):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if shift == 0:
        return x.copy()
    if shift > 0:
        out = np.empty(n, dtype=float)
        out[:shift] = np.nan
        out[shift:] = x[:n - shift]
        return out
    s = -shift
    out = np.empty(n, dtype=float)
    out[n - s:] = np.nan
    out[:n - s] = x[s:]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input CSV file path")
    ap.add_argument("--output", required=True, help="output CSV file path")
    ap.add_argument("--time_col", default="Time (s)", help="time column name")
    ap.add_argument("--ref", default="Chan.1 - Raw (mV)", help="reference channel column name")
    ap.add_argument("--min_period_s", type=float, default=30.0, help="min period search window in seconds")
    ap.add_argument("--max_period_s", type=float, default=200.0, help="max period search window in seconds")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # select 8 channels, default assumes Chan.1..Chan.8 raw columns exist
    raw_cols = [f"Chan.{i} - Raw (mV)" for i in range(1, 9)]
    missing = [c for c in [args.time_col, args.ref] + raw_cols if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    t = df[args.time_col].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t)))

    ref = df[args.ref].to_numpy(dtype=float)
    Np = estimate_period_samples(ref, dt, min_s=args.min_period_s, max_s=args.max_period_s)

    ref_wave = cycle_average(ref, Np)

    shifts = {}
    for c in raw_cols:
        w = cycle_average(df[c].to_numpy(dtype=float), Np)
        lag = circular_lag_to_align(w, ref_wave)
        shifts[c] = int(-lag)

    # shift all channels
    shifted = {}
    for c in raw_cols:
        shifted[c] = shift_with_nan(df[c].to_numpy(dtype=float), shifts[c])

    # crop to common overlap to remove NaN edges
    n = len(df)
    start = 0
    end = n
    for c, s in shifts.items():
        if s > 0:
            start = max(start, s)
        elif s < 0:
            end = min(end, n + s)

    out = pd.DataFrame({args.time_col: t[start:end]})
    for c in raw_cols:
        out[c] = shifted[c][start:end]

    out.to_csv(args.output, index=False)

    # print summary
    print("dt_s:", dt)
    print("period_samples:", Np)
    print("period_s:", Np * dt)
    print("reference:", args.ref)
    print("crop_start_idx:", start, "crop_end_idx:", end, "rows_out:", len(out))
    for c in raw_cols:
        print(c, "shift_samples", shifts[c], "shift_s", shifts[c] * dt)

if __name__ == "__main__":
    main()
