# transfer_train_real_hardcoded.py
# -*- coding: utf-8 -*-

import os
import random
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from config import D_MODEL, TARGET_LEN, PERIOD_TO_INDEX
from model_sensor import SensorEncoder


# =========================
# 1) 路径与超参数写死在这里
# =========================

DATA_DIR = r"C:\Users\niwangze\Desktop\Plant_health_monitoring_project\Model\Transfer_Learning\Drought_Real_Data"

# 这里填你要用来迁移学习的预训练 best_model.pt 的路径
# 如果你有 5 折的 best_model.pt，建议先选一个效果最好的折
PRETRAINED_PATH = r"C:\Users\niwangze\Desktop\Plant_health_monitoring_project\Model\CrossModal_Output\fold_0\best_model.pt"

OUT_DIR = r"C:\Users\niwangze\Desktop\Plant_health_monitoring_project\Model\Transfer_Learning\Transfer_Output"

SEED = 42
EPOCHS = 300
BATCH_SIZE = 8
LR = 1e-4
VAL_RATIO = 0.2

# 输入构造方式
TARGET_LEN_LOCAL = TARGET_LEN
SEGMENTS = 8

# CSV 列设置
# 第 1 列是时间，第 2 列是信号，因此 data_col 取 1
DATA_COL = 1

# period 标签规则
# 前 8 个 CSV 赋为 12，后 8 个 CSV 赋为 0
FIRST_HALF_HOUR = 12
SECOND_HALF_HOUR = 0

# 预处理
BASELINE_PTS = 10
SLICE_START: Optional[int] = None
SLICE_END: Optional[int] = None

# 迁移学习策略
# True: 让 sensor_net 的尾部也参与微调
# False: 只训练 period 相关的全连接层
TRAIN_SENSOR_TAIL = True

# 分类与回归损失加权
LAMBDA_REG = 0.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def interp_nans(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    m = np.isfinite(y)
    if m.sum() == 0:
        return np.zeros_like(y, dtype=np.float32)
    if m.sum() == y.size:
        return y
    x = np.arange(y.size, dtype=np.float32)
    y2 = y.copy()
    y2[~m] = np.interp(x[~m], x[m], y[m]).astype(np.float32)
    return y2


def build_real_records(
    data_dir: str,
    target_len: int,
    segments: int,
    data_col: int,
    first_half_hour: int,
    second_half_hour: int,
    baseline_pts: int,
    slice_start: Optional[int],
    slice_end: Optional[int],
) -> List[Dict[str, Any]]:
    files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(".csv")])
    if len(files) == 0:
        raise FileNotFoundError(f"未找到CSV文件: {data_dir}")

    if len(files) != 16:
        print(f"警告: 期望16个CSV，但实际为 {len(files)}，仍按前后对半方式赋标签")

    half = len(files) // 2
    recs: List[Dict[str, Any]] = []

    for i, fn in enumerate(files):
        fpath = os.path.join(data_dir, fn)
        df = pd.read_csv(fpath)

        if df.shape[1] <= data_col:
            raise ValueError(f"{fn}: data_col={data_col} 超出列范围, 当前列为 {list(df.columns)}")

        y = df.iloc[:, data_col].to_numpy(np.float32)
        y = interp_nans(y)

        L = y.size
        seg_len = L // segments
        if seg_len < 10:
            raise ValueError(f"{fn}: 数据点太少, 无法切成 {segments} 段, 点数为 {L}")

        y = y[:seg_len * segments]

        x = np.zeros((segments, target_len), dtype=np.float32)

        for s in range(segments):
            seg = y[s * seg_len:(s + 1) * seg_len]

            if slice_start is not None or slice_end is not None:
                ss = 0 if slice_start is None else int(slice_start)
                ee = seg.size if slice_end is None else int(slice_end)
                ee = max(ss + 2, min(ee, seg.size))
                seg = seg[ss:ee]

            b = min(int(baseline_pts), max(1, seg.size // 10))
            r0 = float(np.mean(seg[:b]))
            seg = (seg - r0) / (abs(r0) + 1e-6)

            x_old = np.linspace(0.0, 1.0, seg.size, dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
            x[s] = np.interp(x_new, x_old, seg).astype(np.float32)

        hour = first_half_hour if i < half else second_half_hour
        if hour not in PERIOD_TO_INDEX:
            raise ValueError(f"hour={hour} 不在 PERIOD_TO_INDEX 中, 当前映射为 {PERIOD_TO_INDEX}")

        recs.append(
            {
                "x": x,
                "period_idx": int(PERIOD_TO_INDEX[hour]),
                "file": fn,
            }
        )

    return recs


class RealGasDataset(Dataset):
    def __init__(self, records, mean=None, std=None):
        self.records = records
        X = np.stack([r["x"] for r in records], axis=0)
        self.mean = X.mean(axis=(0, 2), keepdims=True) if mean is None else mean
        self.std = X.std(axis=(0, 2), keepdims=True) + 1e-6 if std is None else std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        x = (self.records[idx]["x"] - self.mean[0]) / self.std[0]
        y = self.records[idx]["period_idx"]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)


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


def load_pretrained_partial(model: nn.Module, pretrained_path: str) -> None:
    sd = torch.load(pretrained_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    model_sd = model.state_dict()
    kept = {}
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            kept[k] = v

    missing, unexpected = model.load_state_dict(kept, strict=False)
    print(f"已加载预训练张量数: {len(kept)}")
    print(f"未加载键数量: {len(missing)}")
    print(f"多余键数量: {len(unexpected)}")


def freeze_for_transfer(model: nn.Module, train_sensor_tail: bool) -> None:
    for p in model.parameters():
        p.requires_grad = False

    for p in model.period_base_proj.parameters():
        p.requires_grad = True
    for p in model.period_cls_head.parameters():
        p.requires_grad = True
    for p in model.period_reg_head.parameters():
        p.requires_grad = True

    if train_sensor_tail:
        for p in model.sensor_net.fuse_shared.parameters():
            p.requires_grad = True


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = build_real_records(
        data_dir=DATA_DIR,
        target_len=TARGET_LEN_LOCAL,
        segments=SEGMENTS,
        data_col=DATA_COL,
        first_half_hour=FIRST_HALF_HOUR,
        second_half_hour=SECOND_HALF_HOUR,
        baseline_pts=BASELINE_PTS,
        slice_start=SLICE_START,
        slice_end=SLICE_END,
    )

    idxs = list(range(len(recs)))
    random.shuffle(idxs)
    n_val = max(1, int(round(len(idxs) * VAL_RATIO)))
    val_ids = set(idxs[:n_val])

    tr_recs = [recs[i] for i in range(len(recs)) if i not in val_ids]
    va_recs = [recs[i] for i in range(len(recs)) if i in val_ids]
    print(f"数据量 total={len(recs)} train={len(tr_recs)} val={len(va_recs)}")

    ds_tr = RealGasDataset(tr_recs)
    ds_va = RealGasDataset(va_recs, mean=ds_tr.mean, std=ds_tr.std)

    y_tr = torch.tensor([r["period_idx"] for r in tr_recs], dtype=torch.long)
    class_counts = torch.bincount(y_tr, minlength=5)
    class_w = 1.0 / (class_counts.float() + 1e-6)
    sample_w = class_w[y_tr]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = SensorPeriodNet(d_sensor=D_MODEL, in_channels=SEGMENTS, num_period_classes=5).to(device)
    load_pretrained_partial(model, PRETRAINED_PATH)
    freeze_for_transfer(model, train_sensor_tail=TRAIN_SENSOR_TAIL)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数量: {sum(p.numel() for p in trainable)}")

    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS))

    best_val_acc = -1.0
    best_state_path = out_dir / "transfer_best_model.pt"
    best_ckpt_path = out_dir / "transfer_best_ckpt.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss_sum = 0.0
        tr_correct = 0
        tr_total = 0

        for x, y in dl_tr:
            x = x.to(device)
            y = y.to(device)

            logits, reg = model(x)

            loss_cls = F.cross_entropy(logits, y, label_smoothing=0.05)
            loss_reg = F.smooth_l1_loss(reg, y.float())
            loss = loss_cls + LAMBDA_REG * loss_reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            bs = y.size(0)
            tr_loss_sum += loss.item() * bs
            tr_correct += (logits.argmax(dim=-1) == y).sum().item()
            tr_total += bs

        scheduler.step()

        model.eval()
        va_loss_sum = 0.0
        va_correct = 0
        va_total = 0

        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(device)
                y = y.to(device)

                logits, reg = model(x)
                loss_cls = F.cross_entropy(logits, y)
                loss_reg = F.smooth_l1_loss(reg, y.float())
                loss = loss_cls + LAMBDA_REG * loss_reg

                bs = y.size(0)
                va_loss_sum += loss.item() * bs
                va_correct += (logits.argmax(dim=-1) == y).sum().item()
                va_total += bs

        tr_loss = tr_loss_sum / max(1, tr_total)
        tr_acc = tr_correct / max(1, tr_total)
        va_loss = va_loss_sum / max(1, va_total)
        va_acc = va_correct / max(1, va_total)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_state_path)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean": ds_tr.mean,
                    "std": ds_tr.std,
                    "target_len": TARGET_LEN_LOCAL,
                    "segments": SEGMENTS,
                    "data_col": DATA_COL,
                    "baseline_pts": BASELINE_PTS,
                    "slice_start": SLICE_START,
                    "slice_end": SLICE_END,
                    "PERIOD_TO_INDEX": PERIOD_TO_INDEX,
                },
                best_ckpt_path,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            lr_now = opt.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d} | lr={lr_now:.2e} | "
                f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                f"val_loss={va_loss:.4f} acc={va_acc:.3f} | "
                f"best_val_acc={best_val_acc:.3f}"
            )

    print(f"训练结束, best_val_acc={best_val_acc:.3f}")
    print(f"已保存: {best_state_path}")
    print(f"已保存: {best_ckpt_path}")


if __name__ == "__main__":
    main()