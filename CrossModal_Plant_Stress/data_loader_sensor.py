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
    if "salt" in kind:
        stress = 0
    elif "drought" in kind:
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
                cols = [i for i, c in enumerate(header) if c.startswith("VOCs_Extracted_S")]
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

            seg_len = X.shape[1] // 5
            for seg_id, seg in enumerate(np.split(X[:, :seg_len * 5], 5, axis=1)):
                arr = np.vstack(
                    [np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, seg_len), seg[ch]) for ch in range(8)])
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