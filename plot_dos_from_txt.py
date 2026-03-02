import os
import pandas as pd

extract_dir = r"C:\Users\niwangze\Desktop\Plant_health_monitoring_project\DFT Calculation\差分电荷-能带-态密度\3-能带态密度\N1025"
dos_files = {
    "TDOS": os.path.join(extract_dir, "TDOS.txt"),
    "C_pdos": os.path.join(extract_dir, "C_pdos.txt"),
    "H_pdos": os.path.join(extract_dir, "H_pdos.txt"),
    "O_pdos": os.path.join(extract_dir, "O_pdos.txt"),
    "Sn_pdos": os.path.join(extract_dir, "Sn_pdos.txt"),
}

# Read and write per-file CSV + combined XLSX with sheets
out_xlsx = "DOS_tables.xlsx"
out_dir_csv = "DOS_csvs"
os.makedirs(out_dir_csv, exist_ok=True)

tables = {}
for name, path in dos_files.items():
    assert os.path.exists(path), f"Missing {name}: {path}"
    df = pd.read_csv(path, sep=r"\s+")
    tables[name] = df
    df.to_csv(os.path.join(out_dir_csv, f"{name}.csv"), index=False)

with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    for name, df in tables.items():
        # Excel sheet name length limit 31 chars; ours are fine
        df.to_excel(writer, sheet_name=name, index=False)