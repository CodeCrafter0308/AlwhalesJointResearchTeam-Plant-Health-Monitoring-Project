# infer_drought_detector_min2000_topk.py
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import D_MODEL, TARGET_LEN, PERIOD_TO_INDEX
from model_sensor import SensorEncoder


CSV_PATH = r"C:\Users\niwangze\Desktop\Plant_health_monitoring_project\Model\Transfer_Learning\Drought_Test_High.csv"
CKPT_PATH = r"C:\Users\niwangze\Desktop\Plant_health_monitoring_project\Model\Transfer_Learning\Transfer_Output\transfer_best_ckpt.pt"
OUT_DIR = r"C:\Users\niwangze\Desktop\Plant_health_monitoring_project\Model\Transfer_Learning\Transfer_Output"

WINDOW_SEC = 1000.0
STEP_SEC = 50.0

MIN_SEG_SEC = 2000.0

SMOOTH_ALPHA = 0.2

Z_THRESH = 1.0
ABS_BIN_THR = 0.45
BG_Q = 40

BATCH = 128


class SensorPeriodNet(nn.Module):
    def __init__(self, d_sensor=48, in_channels=8, num_period_classes=5):
        super().__init__()
        self.sensor_net = SensorEncoder(d=d_sensor, in_channels=in_channels)
        self.period_base_proj = nn.Sequential(
            nn.Linear(d_sensor, d_sensor),
            nn.LayerNorm(d_sensor),
            nn.GELU(),
        )
        self.period_cls_head = nn.Sequential(
            nn.Linear(d_sensor, d_sensor),
            nn.LayerNorm(d_sensor),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_sensor, num_period_classes),
        )
        self.period_reg_head = nn.Sequential(
            nn.Linear(d_sensor, d_sensor),
            nn.GELU(),
            nn.Linear(d_sensor, 1),
        )

    def forward(self, x):
        h = self.sensor_net(x)
        h = self.period_base_proj(h)
        logits = self.period_cls_head(h)
        reg = self.period_reg_head(h).squeeze(-1)
        return logits, reg


def build_interp_weights(seg_len, target_len):
    pos = np.linspace(0.0, seg_len - 1, target_len, dtype=np.float32)
    idx0 = np.floor(pos).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, seg_len - 1)
    w = (pos - idx0).astype(np.float32)[None, None, :]
    w0 = 1.0 - w
    return idx0, idx1, w0, w


def robust_z_from_background(values, bg_q=40):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return None, None, None

    thr = np.percentile(v, bg_q)
    bg = v[v <= thr]
    if len(bg) < 10:
        bg = v

    med = float(np.median(bg))
    mad = float(np.median(np.abs(bg - med)) + 1e-9)
    return med, mad, thr


def pick_top_k_nonoverlap(dfw, roll, z, top_k, L, abs_thr, z_thr):
    cand = []
    for i in range(len(dfw)):
        if not np.isfinite(roll[i]):
            continue
        if roll[i] < abs_thr:
            continue
        if z[i] < z_thr:
            continue
        s_idx = i - L + 1
        if s_idx < 0:
            continue
        cand.append((i, s_idx, float(roll[i])))

    cand.sort(key=lambda x: x[2], reverse=True)

    t0 = dfw["t_start_s"].values.astype(float)
    t1 = dfw["t_end_s"].values.astype(float)

    blocked = np.zeros(len(dfw), dtype=bool)
    chosen = []

    for i, s_idx, rscore in cand:
        e_idx = i
        if blocked[s_idx : e_idx + 1].any():
            continue

        chosen.append(
            {
                "t_start_s": float(t0[s_idx]),
                "t_end_s": float(t1[e_idx]),
                "duration_s": float(t1[e_idx] - t0[s_idx]),
                "n_windows": int(e_idx - s_idx + 1),
                "roll_mean_score": float(rscore),
                "z_score": float(z[i]),
                "start_window_id": int(dfw["window_id"].iloc[s_idx]),
                "end_window_id": int(dfw["window_id"].iloc[e_idx]),
            }
        )

        blocked[s_idx : e_idx + 1] = True
        if len(chosen) >= top_k:
            break

    out = pd.DataFrame(chosen)
    if len(out):
        out = out.sort_values("roll_mean_score", ascending=False).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out, len(cand)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        top_k = int(input("请输入你希望输出的段数 TOP_K：").strip())
        if top_k <= 0:
            top_k = 5
    except Exception:
        top_k = 5

    df = pd.read_csv(CSV_PATH)
    t = df.iloc[:, 0].to_numpy(np.float32)
    y = df.iloc[:, 1].to_numpy(np.float32)
    dt = float(np.median(np.diff(t)))

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    sd = ckpt["model_state_dict"]
    mean = ckpt["mean"]
    std = ckpt["std"]
    target_len = int(ckpt.get("target_len", TARGET_LEN))
    segments = int(ckpt.get("segments", 8))
    baseline_pts = int(ckpt.get("baseline_pts", 10))
    period_to_index = ckpt.get("PERIOD_TO_INDEX", PERIOD_TO_INDEX)
    index_to_period = {v: k for k, v in period_to_index.items()}

    stress_idx = int(period_to_index[12])
    normal_idx = int(period_to_index[0])

    win_pts = int(round(WINDOW_SEC / dt))
    step_pts = int(round(STEP_SEC / dt))
    seg_len = win_pts // segments
    win_pts_used = seg_len * segments

    idx0, idx1, w0, w = build_interp_weights(seg_len, target_len)

    mean_b = mean[0].astype(np.float32)
    std_b = std[0].astype(np.float32)
    if mean_b.ndim == 2 and mean_b.shape[-1] == 1:
        mean_b = mean_b[:, 0]
    if std_b.ndim == 2 and std_b.shape[-1] == 1:
        std_b = std_b[:, 0]
    mean_b = mean_b.reshape(1, segments, 1)
    std_b = std_b.reshape(1, segments, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SensorPeriodNet(d_sensor=D_MODEL, in_channels=segments, num_period_classes=len(period_to_index)).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    starts = np.arange(0, max(1, len(t) - win_pts_used + 1), step_pts, dtype=np.int64)

    rows = []
    with torch.no_grad():
        for b0 in range(0, len(starts), BATCH):
            s_idx = starts[b0 : b0 + BATCH]

            win = np.stack([y[s : s + win_pts_used] for s in s_idx], axis=0).astype(np.float32)
            win = win.reshape(-1, segments, seg_len)

            b = min(baseline_pts, max(1, seg_len // 10))
            base = win[..., :b].mean(axis=-1, keepdims=True).astype(np.float32)
            win_norm = (win - base) / (np.abs(base) + 1e-6)

            v0 = win_norm[..., idx0]
            v1 = win_norm[..., idx1]
            x_np = v0 * w0 + v1 * w
            x_np = (x_np - mean_b) / (std_b + 1e-6)

            x = torch.from_numpy(x_np).to(device)
            logits, _ = model(x)
            prob = F.softmax(logits, dim=-1).cpu().numpy()

            p12 = prob[:, stress_idx].astype(np.float32)
            p0 = prob[:, normal_idx].astype(np.float32)
            p_bin = p12 / (p12 + p0 + 1e-9)

            pred_idx = prob.argmax(axis=1)
            pred_period = np.array([index_to_period.get(int(i), -1) for i in pred_idx], dtype=np.int32)

            t_start = t[s_idx].astype(np.float64)
            t_end = t[s_idx + (win_pts_used - 1)].astype(np.float64)
            t_center = 0.5 * (t_start + t_end)

            for i in range(len(s_idx)):
                rows.append(
                    {
                        "window_id": int(b0 + i + 1),
                        "t_start_s": float(t_start[i]),
                        "t_end_s": float(t_end[i]),
                        "t_center_s": float(t_center[i]),
                        "pred_period": int(pred_period[i]),
                        "p_stress_bin": float(p_bin[i]),
                    }
                )

    dfw = pd.DataFrame(rows)
    dfw["p_stress_bin_smooth"] = dfw["p_stress_bin"].ewm(alpha=SMOOTH_ALPHA, adjust=False).mean()

    window_sec_real = float(dfw["t_end_s"].iloc[0] - dfw["t_start_s"].iloc[0])
    step_sec_real = float(np.median(np.diff(dfw["t_start_s"].values)))

    L = int(math.ceil(max(0.0, MIN_SEG_SEC - window_sec_real) / step_sec_real) + 1)

    roll = pd.Series(dfw["p_stress_bin_smooth"].values).rolling(L, min_periods=L).mean().values

    med, mad, bg_thr = robust_z_from_background(roll, bg_q=BG_Q)
    if med is None:
        print("窗口数量不足，无法计算阈值")
        return

    z = (roll - med) / (1.4826 * mad)

    dfseg, n_cand = pick_top_k_nonoverlap(
        dfw=dfw,
        roll=roll,
        z=z,
        top_k=top_k,
        L=L,
        abs_thr=ABS_BIN_THR,
        z_thr=Z_THRESH,
    )

    out_win = os.path.join(OUT_DIR, "stress_windows_long.csv")
    out_seg = os.path.join(OUT_DIR, f"stress_segments_min{int(MIN_SEG_SEC)}_top{top_k}_long_Medium.csv")

    dfw.assign(roll_mean=roll, z_score=z).to_csv(out_win, index=False, encoding="utf-8-sig")
    dfseg.to_csv(out_seg, index=False, encoding="utf-8-sig")

    print("roll_mean 最大值", float(np.nanmax(roll)))
    print("z_score 最大值", float(np.nanmax(z)))
    print("候选数量", int(n_cand))
    print("输出段数", int(len(dfseg)))
    print("保存窗口结果", out_win)
    print("保存段结果", out_seg)


if __name__ == "__main__":
    main()