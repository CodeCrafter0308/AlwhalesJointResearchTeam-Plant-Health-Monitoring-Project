import os, zipfile, re
import numpy as np
import pandas as pd

# zip_path = "/mnt/data/584b25e3-fc56-470d-b6a9-421a4779562a.zip"
# assert os.path.exists(zip_path), f"Missing zip: {zip_path}"

extract_dir = r"C:\Users\niwangze\Desktop\Plant_health_monitoring_project\DFT Calculation\差分电荷-能带-态密度\3-能带态密度\N1025"
os.makedirs(extract_dir, exist_ok=True)

# with zipfile.ZipFile(zip_path, 'r') as z:
#     z.extractall(extract_dir)

# find band_structure.txt inside extracted content
band_path = None
for root, dirs, files in os.walk(extract_dir):
    if "band_structure.txt" in files:
        band_path = os.path.join(root, "band_structure.txt")
        break

assert band_path is not None, "band_structure.txt not found in extracted zip"

band_pat = re.compile(r"^\s*band_index_(\d+)\s*$")
num_pat = re.compile(r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$")

bands = {}
cur = None
ks, es = [], []

with open(band_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        s = line.strip()
        if not s:
            continue
        m = band_pat.match(s)
        if m:
            if cur is not None and ks:
                bands[int(cur)] = (np.array(ks, float), np.array(es, float))
            cur = m.group(1)
            ks, es = [], []
            continue
        m = num_pat.match(s)
        if m and cur is not None:
            ks.append(float(m.group(1)))
            es.append(float(m.group(2)))

if cur is not None and ks:
    bands[int(cur)] = (np.array(ks, float), np.array(es, float))

bands = dict(sorted(bands.items(), key=lambda x: x[0]))
k_ref = next(iter(bands.values()))[0]
n_k = len(k_ref)

data = {"k-path": k_ref}
for bid, (k, e) in bands.items():
    if len(k) != n_k or not np.allclose(k, k_ref, rtol=0, atol=1e-10):
        e = np.interp(k_ref, k, e)
    data[f"band_{bid:03d}"] = e

df = pd.DataFrame(data)

csv_path = "band_structure_origin.csv"
df.to_csv(csv_path, index=False)
