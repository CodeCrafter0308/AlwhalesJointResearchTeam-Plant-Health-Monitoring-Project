import io
import re
import csv
import zipfile
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from config import PERIOD_TO_INDEX


def parse_name(path_in_zip):
    base = path_in_zip.split("/")[-1]
    m = re.match(r"([A-Za-z]+)_(\d+)h_(\d+)\.csv$", base)
    if not m: return None
    kind, hour, file_fold = m.group(1).lower(), int(m.group(2)), int(m.group(3))
    if "drought" in kind:
        stress = 0
    elif "salt" in kind:
        stress = 1
    elif "heat" in kind:
        stress = 2
    else:
        return None
    return base, hour, file_fold, stress


def load_records_from_zip(zip_path, start_row, target_len):
    records = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in [n for n in zf.namelist() if n.lower().endswith(".csv")]:
            meta = parse_name(name)
            if not meta: continue
            base, hour, file_fold, stress = meta

            with zf.open(name, "r") as fb:
                reader = csv.reader(io.TextIOWrapper(fb, encoding="utf-8-sig"))
                header = next(reader)
                cols = [i for i, c in enumerate(header) if c.startswith("Chan")]
                if len(cols) != 8: continue

                rows = []
                for ridx, row in enumerate(reader):
                    if ridx < start_row - 1: continue
                    rows.append([float(row[j]) if row[j] else np.nan for j in cols])

            if len(rows) < 20: continue
            X = np.asarray(rows, dtype=np.float32).T

            for ch in range(8):
                m = np.isfinite(X[ch])
                if m.sum() > 0 and m.sum() < X.shape[1]:
                    X[ch, ~m] = np.interp(np.arange(X.shape[1])[~m], np.arange(X.shape[1])[m], X[ch, m])
                elif m.sum() == 0:
                    X[ch] = 0

            # 将总数据按时间维度（axis=1）均分为 5 个气敏响应过程
            seg_len = X.shape[1] // 5
            for seg_id, seg in enumerate(np.split(X[:, :seg_len * 5], 5, axis=1)):

                # --- 主要修改部分 ---
                # 提取出每个响应过程的第 400~800 行
                # (注意：Python 切片左闭右开，400:800 会提取索引 400 到 799 的数据，即精确的 400 行)
                sub_seg = seg[:, 400:1000]
                sub_seg_len = sub_seg.shape[1]

                # 如果单个响应过程的总长度不足，切片后数据不够进行插值计算，则跳过
                if sub_seg_len < 2:
                    continue

                # 将提取出的 400 行数据插值对齐到设定的 target_len
                # (如果你外部调用时传入的 target_len=400，这一步其实就是保持原样)
                arr = np.vstack(
                    [np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, sub_seg_len), sub_seg[ch]) for ch in
                     range(8)]
                )
                # ------------------

                records.append({"x": arr.astype(np.float32), "stress": stress, "period_idx": PERIOD_TO_INDEX[hour],
                                "cv_fold": random.randint(0, 4)})
    return records


class GasResponseDataset(Dataset):
    def __init__(self, records, mean=None, std=None):
        self.records = records
        X = np.stack([r["x"] for r in records], axis=0)
        self.mean = X.mean(axis=(0, 2), keepdims=True) if mean is None else mean
        self.std = X.std(axis=(0, 2), keepdims=True) + 1e-6 if std is None else std

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        x = (self.records[idx]["x"] - self.mean[0]) / self.std[0]
        return torch.from_numpy(x).float(), torch.tensor(self.records[idx]["stress"], dtype=torch.long), torch.tensor(
            self.records[idx]["period_idx"], dtype=torch.long)