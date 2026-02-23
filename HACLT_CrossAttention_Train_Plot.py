
import os
import re
import io
import csv
import json
import zipfile
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


PERIOD_TO_INDEX = {0: 0, 3: 1, 6: 2, 9: 3, 12: 4}
INDEX_TO_PERIOD = {v: k for k, v in PERIOD_TO_INDEX.items()}


# =========================
# 直接在程序里配置参数
# 修改下面这些值即可
# =========================
ZIP_PATH = r"C:\Users\niwangze\Desktop\Plant_health_monitoring_project\Model\Gas mixture_Dataset.zip"
OUT_DIR = r"C:\Users\niwangze\Desktop\Plant_health_monitoring_project\Model"
START_ROW = 1570
TARGET_LEN = 96
EPOCHS = 300
BATCH_SIZE = 16
LR = 2e-3
SEED = 42



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def parse_name(path_in_zip):
    base = path_in_zip.split("/")[-1]
    m = re.match(r"([A-Za-z]+)_(\d+)h_(\d+)\.csv$", base)
    if not m:
        return None
    kind = m.group(1).lower()
    hour = int(m.group(2))
    fold = int(m.group(3))
    if "salt" in kind:
        stress = 0
    elif "drought" in kind:
        stress = 1
    elif "heat" in kind:
        stress = 2
    else:
        return None
    return base, hour, fold, stress


def load_records_from_zip(zip_path, start_row_1based=1570, target_len=96):
    records = []
    start_idx = start_row_1based - 1

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted([n for n in zf.namelist() if n.lower().endswith(".csv")])

        for name in names:
            meta = parse_name(name)
            if meta is None:
                continue

            base, hour, fold, stress = meta
            with zf.open(name, "r") as fb:
                reader = csv.reader(io.TextIOWrapper(fb, encoding="utf-8-sig", newline=""))
                header = next(reader)

                sensor_cols = []
                for i, c in enumerate(header):
                    if c.startswith("VOCs_Extracted_S"):
                        mm = re.findall(r"(\d+)$", c)
                        if mm:
                            sensor_cols.append((i, int(mm[0])))

                sensor_cols = [i for i, _ in sorted(sensor_cols, key=lambda t: t[1])]
                if len(sensor_cols) != 8:
                    continue

                rows = []
                for ridx, row in enumerate(reader):
                    if ridx < start_idx:
                        continue
                    vals = []
                    for j in sensor_cols:
                        try:
                            vals.append(float(row[j]))
                        except Exception:
                            vals.append(np.nan)
                    rows.append(vals)

            if len(rows) < 20:
                continue

            X = np.asarray(rows, dtype=np.float32).T  # [8, L]

            # NaN 线性插值
            for ch in range(8):
                y = X[ch]
                idx = np.arange(len(y))
                m = np.isfinite(y)
                if m.sum() == 0:
                    X[ch] = 0
                elif m.sum() < len(y):
                    y[~m] = np.interp(idx[~m], idx[m], y[m])
                    X[ch] = y

            # 从第1570行后保留的数据做 5 等分
            L = X.shape[1]
            seg_len = L // 5
            if seg_len < 10:
                continue

            X = X[:, : seg_len * 5]
            old = np.linspace(0, 1, seg_len, dtype=np.float32)
            new = np.linspace(0, 1, target_len, dtype=np.float32)

            for seg_id, seg in enumerate(np.split(X, 5, axis=1)):
                arr = np.vstack([np.interp(new, old, seg[ch]) for ch in range(8)]).astype(np.float32)
                records.append(
                    {
                        "x": arr,
                        "stress": stress,
                        "period_idx": PERIOD_TO_INDEX[hour],
                        "period_hour": hour,
                        "fold_id": fold,
                        "file_name": base,
                        "seg_id": seg_id,
                    }
                )

    return records


class GasResponseDataset(Dataset):
    def __init__(self, records, mean=None, std=None):
        self.records = records
        X = np.stack([r["x"] for r in records], axis=0)

        if mean is None:
            mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
            std = (X.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)

        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        x = ((r["x"] - self.mean[0]) / self.std[0]).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.tensor(r["stress"], dtype=torch.long),
            torch.tensor(r["period_idx"], dtype=torch.long),
        )


def ordinal_targets(y, K=5):
    th = torch.arange(K - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > th).float()


def ordinal_loss(logits, y):
    return F.binary_cross_entropy_with_logits(logits, ordinal_targets(y, logits.shape[1] + 1))


def ordinal_pred(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        h = max(8, d // 2)
        self.score = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

    def forward(self, x):
        a = self.score(x).squeeze(-1)
        w = torch.softmax(a, dim=1)
        h = (w.unsqueeze(-1) * x).sum(dim=1)
        return h, w


class SelfAttnBlock(nn.Module):
    def __init__(self, d=32, heads=4, drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=drop)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(2 * d, d),
            nn.Dropout(drop),
        )

    def forward(self, x):
        y, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + y
        x = x + self.ff(self.ln2(x))
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, d=32, heads=4, drop=0.1):
        super().__init__()

        self.ln_sq = nn.LayerNorm(d)
        self.ln_tkv = nn.LayerNorm(d)
        self.a_s = nn.MultiheadAttention(d, heads, batch_first=True, dropout=drop)

        self.ln_tq = nn.LayerNorm(d)
        self.ln_skv = nn.LayerNorm(d)
        self.a_t = nn.MultiheadAttention(d, heads, batch_first=True, dropout=drop)

        self.ln_sff = nn.LayerNorm(d)
        self.ff_s = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(2 * d, d),
            nn.Dropout(drop),
        )

        self.ln_tff = nn.LayerNorm(d)
        self.ff_t = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(2 * d, d),
            nn.Dropout(drop),
        )

    def forward(self, s, t, return_weights=False):
        s_delta, w_s = self.a_s(
            self.ln_sq(s),
            self.ln_tkv(t),
            self.ln_tkv(t),
            need_weights=return_weights,
            average_attn_weights=False if return_weights else True,
        )
        s = s + s_delta
        s = s + self.ff_s(self.ln_sff(s))

        t_delta, w_t = self.a_t(
            self.ln_tq(t),
            self.ln_skv(s),
            self.ln_skv(s),
            need_weights=return_weights,
            average_attn_weights=False if return_weights else True,
        )
        t = t + t_delta
        t = t + self.ff_t(self.ln_tff(t))

        return s, t, w_s if return_weights else None, w_t if return_weights else None


class MultiScaleStem(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        c1 = d // 3
        c2 = d // 3
        c3 = d - c1 - c2
        self.b1 = nn.Sequential(nn.Conv1d(1, c1, 3, padding=1), nn.GELU())
        self.b2 = nn.Sequential(nn.Conv1d(1, c2, 7, padding=3), nn.GELU())
        self.b3 = nn.Sequential(nn.Conv1d(1, c3, 11, padding=5), nn.GELU())
        self.merge = nn.Sequential(nn.Conv1d(d, d, 1), nn.GELU(), nn.BatchNorm1d(d))

    def forward(self, x):
        return self.merge(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))


class HACLTNetCrossAttention(nn.Module):
    def __init__(self, d=32, patch=12, time_depth=2, cross_depth=2, num_sensors=8):
        super().__init__()

        # Branch A: CNN + BiLSTM + Attn
        self.stem_a = MultiScaleStem(d)
        self.bilstm = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True)
        self.tpool_a = AttnPool(d)
        self.sensor_pos = nn.Parameter(torch.randn(1, num_sensors, d) * 0.02)
        self.sensor_self = SelfAttnBlock(d=d)

        # Branch B: CNN + Patch Transformer
        self.stem_b = MultiScaleStem(d)
        self.patch_proj = nn.Conv1d(d, d, kernel_size=patch, stride=patch)
        self.time_pos = nn.Parameter(torch.randn(1, 64, d) * 0.02)
        self.time_blocks = nn.ModuleList([SelfAttnBlock(d=d) for _ in range(time_depth)])

        # Cross-Attention fusion blocks
        self.cross_blocks = nn.ModuleList([CrossAttnBlock(d=d) for _ in range(cross_depth)])
        self.sensor_refine = SelfAttnBlock(d=d)
        self.time_refine = SelfAttnBlock(d=d)

        self.sensor_pool = AttnPool(d)
        self.time_pool = AttnPool(d)

        self.gate = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, d), nn.Sigmoid())
        self.fuse = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(0.1))

        self.head_stress = nn.Linear(d, 3)
        self.head_period = nn.Linear(d, 4)
        self.head_joint = nn.Linear(d, 15)

    def forward(self, x):
        B, C, T = x.shape

        # Branch A
        za = self.stem_a(x.reshape(B * C, 1, T)).transpose(1, 2)
        za, _ = self.bilstm(za)
        h_each, w_time_each = self.tpool_a(za)
        s = h_each.view(B, C, -1) + self.sensor_pos[:, :C, :]
        s = self.sensor_self(s)
        w_time_lstm = w_time_each.view(B, C, T).mean(dim=1)

        # Branch B
        zb = self.stem_b(x.reshape(B * C, 1, T))
        p = self.patch_proj(zb)
        N = p.shape[-1]
        t = p.transpose(1, 2).reshape(B, C, N, -1).mean(dim=1)
        t = t + self.time_pos[:, :N, :]
        for blk in self.time_blocks:
            t = blk(t)

        # Cross fusion
        ws = None
        wt = None
        for i, blk in enumerate(self.cross_blocks):
            s, t, ws, wt = blk(s, t, return_weights=(i == len(self.cross_blocks) - 1))

        s = self.sensor_refine(s)
        t = self.time_refine(t)

        hs, attn_sensor = self.sensor_pool(s)
        ht, attn_time_tf = self.time_pool(t)

        prod = hs * ht
        diff = torch.abs(hs - ht)
        feat = torch.cat([hs, ht, prod, diff], dim=1)

        g = self.gate(feat)
        h = self.fuse(torch.cat([g * hs, (1.0 - g) * ht, prod, diff], dim=1))

        return {
            "logits_stress": self.head_stress(h),
            "logits_period": self.head_period(h),
            "logits_joint": self.head_joint(h),
            "attn_sensor": attn_sensor,
            "attn_time_lstm": w_time_lstm,
            "attn_time_tf": attn_time_tf,
            "gate_mean": g.mean(dim=1),
            "cross_w_sensor_from_time": ws,
            "cross_w_time_from_sensor": wt,
        }

    def loss(self, out, y_stress, y_period, alpha_joint=0.2):
        ls = F.cross_entropy(out["logits_stress"], y_stress)
        lp = ordinal_loss(out["logits_period"], y_period)
        lj = F.cross_entropy(out["logits_joint"], y_stress * 5 + y_period)
        return ls + lp + alpha_joint * lj


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys = []
    ps = []
    yp = []
    pp = []
    loss_sum = 0.0
    n = 0

    for x, y_s, y_p in loader:
        out = model(x)
        loss = model.loss(out, y_s, y_p)
        pred_s = out["logits_stress"].argmax(dim=1)
        pred_p = ordinal_pred(out["logits_period"])

        b = x.size(0)
        loss_sum += float(loss) * b
        n += b

        ys.append(y_s)
        ps.append(pred_s)
        yp.append(y_p)
        pp.append(pred_p)

    ys = torch.cat(ys)
    ps = torch.cat(ps)
    yp = torch.cat(yp)
    pp = torch.cat(pp)

    true_h = torch.tensor([INDEX_TO_PERIOD[int(i)] for i in yp], dtype=torch.float32)
    pred_h = torch.tensor([INDEX_TO_PERIOD[int(i)] for i in pp], dtype=torch.float32)

    return {
        "loss": loss_sum / max(n, 1),
        "stress_acc": float((ys == ps).float().mean()),
        "period_acc": float((yp == pp).float().mean()),
        "joint_acc": float(((ys == ps) & (yp == pp)).float().mean()),
        "period_mae_hour": float(torch.mean(torch.abs(true_h - pred_h))),
    }


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    rows = []
    for x, y_s, y_p in loader:
        out = model(x)
        pred_s = out["logits_stress"].argmax(dim=1).cpu().numpy()
        pred_p = ordinal_pred(out["logits_period"]).cpu().numpy()
        for a, b, c, d in zip(y_s.cpu().numpy(), y_p.cpu().numpy(), pred_s, pred_p):
            rows.append([int(a), int(b), int(c), int(d)])
    return rows


def main():
    set_seed(SEED)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records_from_zip(ZIP_PATH, start_row_1based=START_ROW, target_len=TARGET_LEN)
    train_recs = [r for r in records if r["fold_id"] == 1]
    val_recs = [r for r in records if r["fold_id"] == 2]

    ds_tr = GasResponseDataset(train_recs)
    ds_va = GasResponseDataset(val_recs, mean=ds_tr.mean, std=ds_tr.std)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    model = HACLTNetCrossAttention(d=32, patch=12, time_depth=2, cross_depth=2)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    history = []
    best_score = -1e9
    best_state = None
    best_metrics = None
    bad = 0
    patience = 8

    for ep in range(1, EPOCHS + 1):
        model.train()

        tr_loss_sum = 0.0
        tr_n = 0
        tr_ys = []
        tr_ps = []
        tr_yp = []
        tr_pp = []

        for x, y_s, y_p in dl_tr:
            out = model(x)
            loss = model.loss(out, y_s, y_p)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            with torch.no_grad():
                pred_s = out["logits_stress"].argmax(dim=1)
                pred_p = ordinal_pred(out["logits_period"])
                b = x.size(0)
                tr_loss_sum += float(loss) * b
                tr_n += b
                tr_ys.append(y_s)
                tr_ps.append(pred_s)
                tr_yp.append(y_p)
                tr_pp.append(pred_p)

        tr_ys = torch.cat(tr_ys)
        tr_ps = torch.cat(tr_ps)
        tr_yp = torch.cat(tr_yp)
        tr_pp = torch.cat(tr_pp)

        tr_true_h = torch.tensor([INDEX_TO_PERIOD[int(i)] for i in tr_yp], dtype=torch.float32)
        tr_pred_h = torch.tensor([INDEX_TO_PERIOD[int(i)] for i in tr_pp], dtype=torch.float32)

        tr = {
            "loss": tr_loss_sum / max(tr_n, 1),
            "stress_acc": float((tr_ys == tr_ps).float().mean()),
            "period_acc": float((tr_yp == tr_pp).float().mean()),
            "joint_acc": float(((tr_ys == tr_ps) & (tr_yp == tr_pp)).float().mean()),
            "period_mae_hour": float(torch.mean(torch.abs(tr_true_h - tr_pred_h))),
        }

        va = evaluate(model, dl_va)

        row = {"epoch": ep}
        for k, v in tr.items():
            row["train_" + k] = v
        for k, v in va.items():
            row["val_" + k] = v
        history.append(row)

        score = (
            0.5 * va["stress_acc"]
            + 0.3 * va["period_acc"]
            + 0.2 * va["joint_acc"]
            - 0.01 * va["period_mae_hour"]
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = va
            bad = 0
        else:
            bad += 1

        print(
            f"Epoch {ep:02d} | "
            f"TrainLoss {tr['loss']:.4f} | ValLoss {va['loss']:.4f} | "
            f"Val Stress {va['stress_acc']:.4f} | Val Period {va['period_acc']:.4f} | "
            f"Val Joint {va['joint_acc']:.4f} | Val MAE {va['period_mae_hour']:.2f}h"
        )

        if bad >= patience:
            print(f"Early stopping at epoch {ep}")
            break

    model.load_state_dict(best_state)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_history.csv", index=False)

    pred_rows = collect_predictions(model, dl_va)
    pd.DataFrame(pred_rows, columns=["y_stress", "y_period_idx", "pred_stress", "pred_period_idx"]).to_csv(
        out_dir / "val_predictions.csv", index=False
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "norm_mean": ds_tr.mean,
            "norm_std": ds_tr.std,
            "config": {
                "d": 32,
                "patch": 12,
                "time_depth": 2,
                "cross_depth": 2,
                "target_len": TARGET_LEN,
                "start_row": START_ROW,
            },
        },
        out_dir / "haclt_crossattn_model.pt",
    )

    metrics = {
        "n_total_samples": len(records),
        "n_train_samples": len(train_recs),
        "n_val_samples": len(val_recs),
        "data_rule": "start at row 1570 inclusive and split remaining rows into 5 equal segments",
        "target_len": TARGET_LEN,
        "best_val_metrics": best_metrics,
    }
    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Plot 1 loss
    plt.figure(figsize=(8, 4))
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss")
    plt.plot(hist_df["epoch"], hist_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("HACLT-Net Cross-Attention Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=180)
    plt.close()

    # Plot 2 accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_stress_acc"], label="Train Stress Acc")
    plt.plot(hist_df["epoch"], hist_df["val_stress_acc"], label="Val Stress Acc")
    plt.plot(hist_df["epoch"], hist_df["train_period_acc"], label="Train Period Acc")
    plt.plot(hist_df["epoch"], hist_df["val_period_acc"], label="Val Period Acc")
    plt.plot(hist_df["epoch"], hist_df["val_joint_acc"], label="Val Joint Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("HACLT-Net Cross-Attention Accuracy Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curves.png", dpi=180)
    plt.close()

    print("\nBest validation metrics")
    print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
