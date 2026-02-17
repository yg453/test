# -*- coding: utf-8 -*-
"""
debug_stock.py
A股量化交易系统 - 支持随机探索打榜与实盘微调预测
"""
import os
import sys
import random
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta

try:
    from config import DB_PATH, SEQ_LENGTH, FEATURE_DIM, TARGET_DAYS
    from AShareDataset import AShareDataset
    from HybridLSTM import HybridLSTM
    from TradeManager import TradeManager
    from ModelManager import ModelManager
except ImportError as e:
    print(f"CRITICAL: 模块导入失败: {e}")
    sys.exit(1)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    print(f"[System] Global Random Seed locked to {seed}.")

class DebugDataset(AShareDataset):
    def __init__(self, db_path, code, start_date, end_date, scaler=None, fit_scaler=False):
        self.seq_length = SEQ_LENGTH
        self.target_days = TARGET_DAYS
        print(f"Loading data for {code} [{start_date} to {end_date}]...")
        conn = sqlite3.connect(db_path)
        query = f"SELECT date, code, open, high, low, close, volume FROM daily_kline WHERE code='{code}' AND date >= '{start_date}' AND date <= '{end_date}' ORDER BY date"
        self.df = pd.read_sql(query, conn)
        conn.close()
        
        if self.df.empty: raise ValueError(f"No data found for {code} in {start_date}-{end_date}")

        self.df['volume'] = self.df['volume'].replace(0, np.nan)
        self.df.ffill(inplace=True)
        self._generate_features_and_labels()
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        raw_features = self.df[self.feature_cols].values
        if fit_scaler:
            self.scaler = StandardScaler()
            self.scaled_features = self.scaler.fit_transform(raw_features)
        else:
            self.scaler = scaler
            self.scaled_features = self.scaler.transform(raw_features)
            
        self.x_tensor, self.y_tensor = self._build_tensor_dataset()

    def _build_tensor_dataset(self):
        xs, ys = [], []
        for i in range(len(self.df) - self.seq_length):
            x_window = self.scaled_features[i : i+self.seq_length]
            y_val = self.df.iloc[i + self.seq_length - 1]['label']
            xs.append(x_window)
            ys.append(y_val)
        if not xs: return torch.FloatTensor([]), torch.FloatTensor([])
        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys)).unsqueeze(1)

    def get_loader(self, batch_size=32, shuffle=False):
        dataset = TensorDataset(self.x_tensor, self.y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_model(train_loader, input_dim, existing_model=None, epochs=30, lr=0.001):
    model = HybridLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2)
    if existing_model is not None:
        model.load_state_dict(existing_model)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % max(1, epochs//5) == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss/max(1, len(train_loader)):.4f}")
    return model

def backtest_strategy(model, test_dataset):
    print("\n>>> [Phase 2] Backtesting with TradeManager...")
    model.eval()
    manager = TradeManager(initial_capital=10000000.0, min_prob_threshold=0.52, max_risk_per_trade=0.03, risk_reward_ratio=2.0) 
    test_loader = test_dataset.get_loader(batch_size=1, shuffle=False)
    df_ref = test_dataset.df
    valid_length = len(test_dataset.x_tensor)
    
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= valid_length - 1: break 
            prob = model(x).item()
            idx_T = i + test_dataset.seq_length - 1
            if idx_T >= len(df_ref): break

            signal = manager.on_bar_close(
                df_ref.iloc[idx_T]['date'], df_ref.iloc[idx_T]['close'], df_ref.iloc[idx_T]['atr_real'], prob, i
            )
            
            if idx_T + 1 < len(df_ref) and signal['action'] != 'HOLD':
                manager.execute_order(
                    signal['action'], signal['shares'], df_ref.iloc[idx_T + 1]['open'], df_ref.iloc[idx_T + 1]['date'], i+1
                )
    return manager.total_assets

def predict_future_with_finetuning(model, scaler, code):
    """【机构级实战引擎】取近半年数据进行模型微调，然后预测明天"""
    print("\n>>> [Predict Phase] 启动微调与明日预测引擎...")
    
    # 动态获取最近半年时间窗口
    end_date_str = '2035-12-31' # 取到数据库最新
    start_date_str = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    # 对于回测环境，强制使用固定时间保证测试（可根据需要修改）
    start_date_str = '2024-06-01'
    
    try:
        recent_ds = DebugDataset(DB_PATH, code, start_date=start_date_str, end_date=end_date_str, fit_scaler=False, scaler=scaler)
    except ValueError as e:
        print(f"数据不足无法微调: {e}"); return

    # 1. 微调阶段 (Fine-Tuning)
    print("-> 正在使用近半年数据微调模型 (冻结记忆，极低学习率)...")
    recent_loader = recent_ds.get_loader(batch_size=16, shuffle=True)
    # [核心] lr=1e-5，仅训练 5 个 epoch，融合最新市场特征而不破坏长期记忆
    model = train_model(recent_loader, input_dim=recent_ds.data_tensor.shape[2] if hasattr(recent_ds, 'data_tensor') else recent_ds.x_tensor.shape[2], 
                        existing_model=model.state_dict(), epochs=5, lr=1e-5)
    
    # 2. 预测阶段
    model.eval()
    raw_features = recent_ds.df[recent_ds.feature_cols].values
    last_window_raw = raw_features[-recent_ds.seq_length:]
    last_window_scaled = scaler.transform(last_window_raw)
    x_tensor = torch.FloatTensor(last_window_scaled).unsqueeze(0) 
    
    with torch.no_grad():
        prob = model(x_tensor).item()
        
    last_date = recent_ds.df.iloc[-1]['date']
    last_close = recent_ds.df.iloc[-1]['close']
    
    print("\n" + "="*45)
    print("📈 明日走势与实盘策略预测 (含微调增强) 📉")
    print("="*45)
    print(f"分析标的: {code}")
    print(f"数据截止: {last_date} (收盘价: {last_close:.2f})")
    print(f"网络做多置信度: {prob:.2%}")
    print("-" * 45)
    
    if prob > 0.52:
        print("💡 [操作建议]: 强烈看多 / 建议买入持仓")
    elif prob < 0.48:
        print("🛡️ [操作建议]: 悲观预警 / 建议卖出平仓")
    else:
        print("⏳ [操作建议]: 空仓观望")
    print("="*45)

def main(mode='explore', code='sh.600000', rank=1):
    print(f"\n🚀 System Starting... Mode: [{mode.upper()}] | Code: [{code}]")
    mm = ModelManager()

    if mode == 'predict':
        try:
            checkpoint, seed = mm.load_model(code, rank)
            set_seed(seed) # 使用原种子保证环境一致
            model = HybridLSTM(input_dim=len(DebugDataset(DB_PATH, code, '2020-01-01', '2020-01-30').feature_cols), hidden_dim=64, num_layers=2)
            model.load_state_dict(checkpoint['model_state'])
            predict_future_with_finetuning(model, checkpoint['scaler'], code)
        except ValueError as e:
            print(f"错误: {e}")
        return

    # 获取系统运行种子
    if mode == 'explore':
        current_seed = random.randint(1, 999999)
        existing_weights = None
    elif mode == 'retrain':
        try:
            checkpoint, current_seed = mm.load_model(code, rank)
            existing_weights = checkpoint['model_state']
        except ValueError as e:
            print(f"错误: {e}"); return
    
    set_seed(current_seed)
    
    # 加载训练集并训练
    train_ds = DebugDataset(DB_PATH, code, start_date='2020-01-01', end_date='2024-12-31', fit_scaler=True)
    train_loader = train_ds.get_loader(batch_size=32, shuffle=True)
    input_dim = train_ds.data_tensor.shape[2] if hasattr(train_ds, 'data_tensor') else train_ds.x_tensor.shape[2]
    
    print("\n>>> [Phase 1] 深度神经网络训练中...")
    model = train_model(train_loader, input_dim, existing_model=existing_weights, epochs=30, lr=0.001)
    
    # 回测并打榜
    test_ds = DebugDataset(DB_PATH, code, start_date='2025-01-01', end_date='2026-12-31', fit_scaler=False, scaler=train_ds.scaler)
    final_pnl = backtest_strategy(model, test_ds)
    print(f"\n=== Diagnosis Report ===\nFinal Assets: {final_pnl:.2f}")
    
    # 尝试入库
    mm.save_if_top(code, model, train_ds.scaler, final_pnl, current_seed)

if __name__ == "__main__":
    # ==========================================
    # 🎮 架构师控制台 (控制程序行为的最高开关)
    # ==========================================
    RUN_MODE = 'explore'   # 'explore', 'retrain', 或 'predict'
    TARGET_CODE = 'sz.000158'
    TARGET_RANK = 1        
    
    main(mode=RUN_MODE, code=TARGET_CODE, rank=TARGET_RANK)