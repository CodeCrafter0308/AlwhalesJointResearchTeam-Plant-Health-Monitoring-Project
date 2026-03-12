import os

# ================= 数据路径 =================
ZIP_PATH = "C:/Users/niwangze/Desktop/Plant_health_monitoring_project/Model/Gas mixture_Dataset.zip"
OUT_DIR = "C:/Users/niwangze/Desktop/Plant_health_monitoring_project/Model/CrossModal_Output"

# ================= 训练超参数 =================
START_ROW = 0
TARGET_LEN = 256
EPOCHS = 250
BATCH_SIZE = 64
LR = 5e-4
SEED = 42

# ================= 模型维度配置 =================
D_MODEL = 48           # 传感器通道隐藏维度
MOL_DIM = 128          # 分子特征隐藏维度

# ================= 标签映射 =================
PERIOD_TO_INDEX = {0: 0, 3: 1, 6: 2, 9: 3, 12: 4}
INDEX_TO_PERIOD = {v: k for k, v in PERIOD_TO_INDEX.items()}
STRESS_MAP = {"Drought": 0, "Salt": 1, "Heat": 2}

# ================= DFT 第三分支配置 =================
# 目前只接入吸附能特征。后续接入 band/DOS 时，可把 DFT_DIM 改为新特征维度。
DFT_DIM = 6

# Band-structure directory containing 7 files (csv/xlsx), one per VOC, filenames must include VOC keywords.
DFT_BAND_DIR = "./dft_inputs/band_structure"
# DFT_DIM is kept for compatibility, but main_train.py will infer actual DFT dim from the built feature bank.

# DFT DOS/PDOS directory (contains 7 VOC subfolders)
DFT_DOS_DIR = "./dft_inputs/DOS"