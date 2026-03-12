# infer_long_segments_min2000.py
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


CSV_PATH = r"C:\Users\niwangze\Desktop\Plant_health_monitoring_project\Model\Transfer_Learning\Drought_Test.csv"
CKPT_PATH = r"C:\Users\niwangze\Desktop\Plant_health_monitoring_project\Model\Transfer_Learning\Transfer_Output\transfer_best_ckpt.pt"
OUT_DIR = r"C:\Users\niwangze\Desktop\Plant_health_monitoring_project\Model\Transfer_Learning\Transfer_Output"

WINDOW_SEC = 1000.0
STEP_SEC = 25.0

MIN_SEG_SEC = 2000.0
TOP_K = 5

SMOOTH_ALPHA = 0.2


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


def pick_top_k_segments_min_duration(dfw, score_col, min_seg_sec, top_k, window_sec, step_sec):
    L = int(math.ceil(max(0.0, min_seg_sec - window_sec) / step_sec) + 1)
    score = dfw[score_col].values.astype(float)

    roll = pd.Series(score).rolling(L, min_periods=L).mean().values
    valid = np.isfinite(roll)

    t_start = dfw["t_start_s"].values.astype(float)
    t_end = dfw["t_end_s"].values.astype(float)

    chosen = []
    blocked = np.zeros(len(dfw), dtype=bool)

    cand_idx = np.argsort(np.where(valid, roll, -np.inf))[::-1]
    for i in cand_idx:
        if not valid[i]:
            break
        s_idx = i - L + 1
        e_idx = i
        if s_idx < 0:
            continue
        if blocked[s_idx : e_idx + 1].any():
            continue

        seg_start = float(t_start[s_idx])
        seg_end = float(t_end[e_idx])
        seg_scores = score[s_idx : e_idx + 1]

        chosen.append(
            {
                "t_start_s": seg_start,
                "t_end_s": seg_end,
                "duration_s": seg_end - seg_start,
                "n_windows": int(e_idx - s_idx + 1),
                "mean_score": float(np.mean(seg_scores)),
                "max_score": float(np.max(seg_scores)),
                "roll_mean_score": float(roll[i]),
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
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

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

    batch = 128

    rows = []
    with torch.no_grad():
        for b0 in range(0, len(starts), batch):
            s_idx = starts[b0 : b0 + batch]

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

            p_stress = prob[:, stress_idx].astype(np.float32)
            p_normal = prob[:, normal_idx].astype(np.float32)
            p_bin = p_stress / (p_stress + p_normal + 1e-9)

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
                        "p_stress": float(p_stress[i]),
                        "p_normal": float(p_normal[i]),
                        "p_stress_bin": float(p_bin[i]),
                    }
                )

    dfw = pd.DataFrame(rows)
    dfw["p_stress_bin_smooth"] = dfw["p_stress_bin"].ewm(alpha=SMOOTH_ALPHA, adjust=False).mean()

    window_sec_real = float(dfw["t_end_s"].iloc[0] - dfw["t_start_s"].iloc[0])
    step_sec_real = float(np.median(np.diff(dfw["t_start_s"].values)))

    dfseg = pick_top_k_segments_min_duration(
        dfw=dfw,
        score_col="p_stress_bin_smooth",
        min_seg_sec=MIN_SEG_SEC,
        top_k=TOP_K,
        window_sec=window_sec_real,
        step_sec=step_sec_real,
    )

    out_win = os.path.join(OUT_DIR, "stress_windows_1000s_step25s.csv")
    out_seg = os.path.join(OUT_DIR, f"stress_segments_min{int(MIN_SEG_SEC)}s_top{TOP_K}.csv")

    dfw.to_csv(out_win, index=False, encoding="utf-8-sig")
    dfseg.to_csv(out_seg, index=False, encoding="utf-8-sig")

    print("saved", out_win)
    print("saved", out_seg)


if __name__ == "__main__":
    main()