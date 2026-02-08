#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import re

def estimate_period_samples(x, dt, min_s, max_s):
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
    if n_cycles < 2:
        raise ValueError("Too few cycles. Check period or data length.")
    x = x[:n_cycles * Np]
    X = x.reshape(n_cycles, Np)
    return np.nanmean(X, axis=0)

def circular_lag_to_align(a, b):
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
    out = np.empty(n, dtype=float)
    out[:] = np.nan
    if shift == 0:
        return x.copy()
    if shift > 0:
        out[shift:] = x[:n - shift]
    else:
        s = -shift
        out[:n - s] = x[s:]
    return out

def detect_sensor_cols(df, time_col, sensor_regex):
    cols = [c for c in df.columns if c != time_col]
    cols = [c for c in cols if not df[c].isna().all()]
    if sensor_regex:
        rgx = re.compile(sensor_regex)
        cols = [c for c in cols if rgx.search(c)]
    if not cols:
        raise ValueError("No sensor columns detected. Set --sensor_regex to match your sensor columns.")
    return cols

def align(df, time_col, ref_col, sensor_cols, min_period_s, max_period_s):
    df = df.loc[~df[time_col].isna()].copy()
    t = df[time_col].to_numpy(dtype=float)

    dt = float(np.nanmedian(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid dt. Check time column.")

    ref = df[ref_col].to_numpy(dtype=float)
    Np = estimate_period_samples(ref, dt, min_period_s, max_period_s)

    ref_wave = cycle_average(ref, Np)

    shift_map = {}
    for c in sensor_cols:
        w = cycle_average(df[c].to_numpy(dtype=float), Np)
        lag = circular_lag_to_align(w, ref_wave)
        shift_map[c] = int(-lag)

    n = len(df)
    start = max([s for s in shift_map.values() if s > 0] + [0])
    min_neg = min([s for s in shift_map.values() if s < 0] + [0])
    end = n + min_neg

    shifted = {c: shift_with_nan(df[c].to_numpy(dtype=float), shift_map[c]) for c in sensor_cols}

    out = pd.DataFrame({time_col: t[start:end]})
    for c in sensor_cols:
        out[c] = shifted[c][start:end]

    # one refinement pass to remove residual 1-sample misalignment
    ref2 = out[ref_col].to_numpy(dtype=float)
    ref_wave2 = cycle_average(ref2, Np)
    for c in sensor_cols:
        w2 = cycle_average(out[c].to_numpy(dtype=float), Np)
        lag2 = circular_lag_to_align(w2, ref_wave2)
        if lag2 != 0:
            shift_map[c] = int(shift_map[c] - lag2)

    start = max([s for s in shift_map.values() if s > 0] + [0])
    min_neg = min([s for s in shift_map.values() if s < 0] + [0])
    end = n + min_neg

    out = pd.DataFrame({time_col: t[start:end]})
    for c in sensor_cols:
        out[c] = shift_with_nan(df[c].to_numpy(dtype=float), shift_map[c])[start:end]

    return out, shift_map, dt, Np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input CSV path")
    ap.add_argument("--output", required=True, help="output CSV path")
    ap.add_argument("--time_col", default="Time (s)", help="time column name")
    ap.add_argument("--ref", default="", help="reference sensor column name. Default is first detected sensor column")
    ap.add_argument("--sensor_regex", default="", help="regex to select sensor columns, for example '^Chan\.' or 'Raw'")
    ap.add_argument("--min_period_s", type=float, default=30.0, help="min period search window in seconds")
    ap.add_argument("--max_period_s", type=float, default=200.0, help="max period search window in seconds")
    ap.add_argument("--shift_csv", default="", help="optional path to save shift summary as CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    if args.time_col not in df.columns:
        # fallback to first column
        args.time_col = df.columns[0]

    sensor_cols = detect_sensor_cols(df, args.time_col, args.sensor_regex)
    ref_col = args.ref if args.ref else sensor_cols[0]
    if ref_col not in df.columns:
        raise ValueError("Reference column not found: " + ref_col)

    out, shift_map, dt, Np = align(df, args.time_col, ref_col, sensor_cols, args.min_period_s, args.max_period_s)

    out.to_csv(args.output, index=False)

    if args.shift_csv:
        shift_df = pd.DataFrame([
            {"sensor": c, "shift_samples": int(shift_map[c]), "shift_seconds": float(shift_map[c] * dt)}
            for c in sensor_cols
        ]).sort_values("sensor")
        shift_df.to_csv(args.shift_csv, index=False)

    print("dt_s:", dt)
    print("period_samples:", int(Np))
    print("period_s:", float(Np * dt))
    print("reference_sensor:", ref_col)
    print("rows_out:", int(len(out)))
    for c in sensor_cols:
        print(c, "shift_samples", int(shift_map[c]), "shift_s", float(shift_map[c] * dt))

if __name__ == "__main__":
    main()
