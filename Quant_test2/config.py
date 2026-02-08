# config.py
import os
import sys

# --- 1. 路径配置 ---
# 获取当前脚本所在目录 (Quant_test2)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据库路径 (Database_test 在上一级目录的兄弟目录)
# 假设结构是:
# test/
#   ├── Quant_test2/  (当前代码)
#   └── Database_test/ (数据库)
DB_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "Database_test")
DB_PATH = os.path.join(DB_ROOT, "stocks.db")
UPDATE_SCRIPT = os.path.join(DB_ROOT, "run_stock_db.py")

# 检查数据库是否存在
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"CRITICAL: Database not found at {DB_PATH}. \n请检查路径或先运行 Database_test 下的 run_stock_db.py")

# --- 2. 模型参数 ---
SEQ_LENGTH = 15      # 过去 15 天
FEATURE_DIM = 9      # 特征数量 (Open, Close, Vol, MA20...) 根据 AShareDataset 实际生成的列数调整
HIDDEN_DIM = 128
LAYERS = 2
TARGET_DAYS = 3      # 预测未来 3 天

# --- 3. 训练配置 ---
TRAIN_START = '2020-01-01'
TRAIN_END = '2024-12-31'
TEST_START = '2025-01-01'