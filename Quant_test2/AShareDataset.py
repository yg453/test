try:
    from config import DB_PATH, SEQ_LENGTH, FEATURE_DIM
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import DB_PATH, SEQ_LENGTH, FEATURE_DIM

import sqlite3
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class AShareDataset(Dataset):
    def __init__(self, db_path, table_name='daily_kline', seq_length=15, target_days=3, train_mode=True):
        """
        Args:
            db_path: SQLite数据库路径
            seq_length: window_size 滑动窗口长度 
            target_days: 预测未来几天的最高收益
            train_mode: 是否为训练模式 
        """
        self.seq_length = seq_length
        self.target_days = target_days
        self.train_mode = train_mode
        
        print(f"Loading data from {db_path}...")
        conn = sqlite3.connect(db_path)
        query = f"SELECT date, code, open, high, low, close, volume FROM {table_name} ORDER BY code, date"
        self.df = pd.read_sql(query, conn)
        conn.close()
        
        # 1. 基础数据清洗 (剔除停牌日与无穷值)
        self.df['volume'] = self.df['volume'].replace(0, np.nan)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 采用按股票前向填充，防止串码泄露
        self.df = self.df.groupby('code').ffill()
        self.df.dropna(subset=['volume'], inplace=True)
        
        # 2. 特征与标签工程 
        self._generate_features_and_labels()
        
        # 3. 终局清理
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        # 4. 序列构建 (Sliding Window)
        self.samples = []
        self._build_samples()
        
        # 5. 特征提取与标准化
        self.feature_values = self.df[self.feature_cols].values
        
        if self.train_mode:
            self.scaler = StandardScaler()
            self.feature_values = self.scaler.fit_transform(self.feature_values)
        else:
            self.scaler = StandardScaler() 
            self.feature_values = self.scaler.fit_transform(self.feature_values)

        self.data_tensor = torch.FloatTensor(self.feature_values)
        self.label_tensor = torch.FloatTensor(self.df['label'].values)
        
        # [核心自检机制]
        self._check_logic()
        
        print(f"Dataset ready. Total samples: {len(self.samples)}")
        print(f"Input Shape: (Batch, {self.seq_length}, {self.data_tensor.shape[1]})")

    def _check_logic(self):
        """运行时自检：拦截 NaN/Inf 与形状异常"""
        assert not np.isnan(self.feature_values).any(), "[逻辑崩溃] 特征矩阵中存在未被清洗的 NaN"
        assert not np.isinf(self.feature_values).any(), "[逻辑崩溃] 特征矩阵中存在 Inf 导致梯度爆炸隐患"
        assert len(self.samples) > 0, "[逻辑崩溃] 样本集合为空，请检查 window_size 与数据日期交集"
        assert self.data_tensor.shape[1] == len(self.feature_cols), "特征维度匹配失败"

    def _generate_features_and_labels(self):
        """
        合并生成特征与标签，修复原版 apply MultiIndex 异常
        """
        df = self.df
        grouped = df.groupby('code')
        
        # --- 标签生成 (Y) ---
        # 严格使用 shift(-N) 探测未来数据，不参与后续 X 构建
        future_high = grouped['high'].transform(lambda x: x.rolling(window=self.target_days, min_periods=1).max().shift(-self.target_days))
        df['target_return'] = future_high / df['close'] - 1.0
        # threshold=0.02 作为二元分类阈值
        df['label'] = np.where(df['target_return'] > 0.02, 1.0, 0.0)

        # --- 特征工程 (X) - 强制使用 transform ---
        df['log_ret'] = grouped['close'].transform(lambda x: np.log(x / x.shift(1)))
        
        df['ma20'] = grouped['close'].transform(lambda x: x.rolling(window=20).mean())
        df['bias_20'] = (df['close'] - df['ma20']) / (df['ma20'] + 1e-8)
        
        df['vol_ma5'] = grouped['volume'].transform(lambda x: x.rolling(window=5).mean())
        df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-8)
        
        ema12 = grouped['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        ema26 = grouped['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        dif = ema12 - ema26
        dea = dif.groupby(df['code']).transform(lambda x: x.ewm(span=9, adjust=False).mean())
        df['macd_norm'] = (dif - dea) / (df['close'] + 1e-8)
        
        def calc_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
            
        df['rsi'] = grouped['close'].transform(lambda x: calc_rsi(x) / 100.0)
        df['volatility'] = grouped['log_ret'].transform(lambda x: x.rolling(window=20).std())
        df['slope_5'] = grouped['close'].transform(lambda x: (x - x.shift(5)) / (x.shift(5) + 1e-8))

        # --- 真实 ATR 构建 (仅供底层风控仓位使用) ---
        df['prev_close'] = grouped['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['prev_close']).abs()
        tr3 = (df['low'] - df['prev_close']).abs()
        df['tr'] = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        df['atr_real'] = grouped['tr'].transform(lambda x: x.rolling(window=14, min_periods=1).mean())

        # 核心修复点：显式声明进入神经网络训练的列，不再暴力 drop high/close
        self.feature_cols = ['log_ret', 'bias_20', 'vol_ratio', 'macd_norm', 'rsi', 'volatility', 'slope_5']

    def _build_samples(self):
        code_groups = self.df.groupby('code').groups
        
        for code, indices in code_groups.items():
            idx_array = np.array(sorted(indices))
            if len(idx_array) < self.seq_length:
                continue
            
            for i in range(len(idx_array) - self.seq_length):
                start_row = idx_array[i]
                current_idx = idx_array[i + self.seq_length - 1] 
                self.samples.append((start_row, self.seq_length, current_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_row, length, label_row = self.samples[idx]
        x = self.data_tensor[start_row : start_row + length]
        y = self.label_tensor[label_row]
        return x, y.unsqueeze(0)

if __name__ == "__main__":
    # ==========================================
    # 模块独立测试与防御性验证入口 (Smoke Test)
    # 仅在直接运行 python AShareDataset.py 时触发
    # ==========================================
    import os
    from torch.utils.data import DataLoader
    
    try:
        from config import DB_PATH
    except ImportError:
        print("[警告] 无法导入 config.DB_PATH，尝试使用当前目录查找数据库...")
        DB_PATH = "stock_data.db" # 降级回退路径
        
    print(">>> 开始执行 AShareDataset 独立模块测试...")
    
    if not os.path.exists(DB_PATH):
        print(f"[致命错误] 找不到数据库文件: {DB_PATH}。请检查测试环境！")
    else:
        try:
            # 1. 初始化数据集 (默认测试模式)
            test_dataset = AShareDataset(
                db_path=DB_PATH, 
                table_name='daily_kline', 
                seq_length=15, 
                target_days=3, 
                train_mode=True
            )
            
            # 2. 挂载 DataLoader
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
            
            # 3. 抽取一个 Batch 验证 Shape 与内存流转
            batch_x, batch_y = next(iter(test_loader))
            
            print("\n[测试通过] 数据管道流转正常！")
            print(f" -> Batch X Shape : {batch_x.shape} (预期: [BatchSize, SeqLength, FeatureDim])")
            print(f" -> Batch Y Shape : {batch_y.shape} (预期: [BatchSize, 1])")
            print(f" -> Feature Cols  : {test_dataset.feature_cols}")
            
            # 4. 严苛断言：确保拿到的数据不存在 NaN
            assert not torch.isnan(batch_x).any(), "验证失败：DataLoader 抽取的 X 中存在 NaN！"
            assert not torch.isnan(batch_y).any(), "验证失败：DataLoader 抽取的 Y 中存在 NaN！"
            
        except Exception as e:
            print(f"\n[测试崩溃] AShareDataset 内部逻辑发生严重错误:\n{e}")