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