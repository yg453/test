# -*- coding: utf-8 -*-
"""
debug_stock.py
A股量化交易系统 - 单标的诊断与全流程调试脚本
"""

import os
import sys
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from config import DB_PATH, SEQ_LENGTH, FEATURE_DIM, TARGET_DAYS
    from AShareDataset import AShareDataset
    from HybridLSTM import HybridLSTM
    from TradeManager import TradeManager
except ImportError as e:
    print(f"CRITICAL: 模块导入失败。\n错误详情: {e}")
    sys.exit(1)

TARGET_CODE = 'sh.600000' 
BATCH_SIZE = 32

class DebugDataset(AShareDataset):
    def __init__(self, db_path, code, start_date, end_date, scaler=None, fit_scaler=False):
        self.seq_length = SEQ_LENGTH
        self.target_days = TARGET_DAYS
        
        print(f"Loading data for {code} [{start_date} to {end_date}]...")
        conn = sqlite3.connect(db_path)
        query = f"""
            SELECT date, code, open, high, low, close, volume 
            FROM daily_kline 
            WHERE code='{code}' AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date
        """
        self.df = pd.read_sql(query, conn)
        conn.close()
        
        if self.df.empty:
            raise ValueError(f"No data found for {code} in {db_path}")

        # 前处理
        self.df['volume'] = self.df['volume'].replace(0, np.nan)
        # [旧代码] self.df.fillna(method='ffill', inplace=True)
        # [新代码] 严格遵循现代 Pandas 接口规范
        self.df.ffill(inplace=True)
        
        # 构建并清理
        self._generate_features_and_labels()
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        # 严格获取特征子集
        raw_features = self.df[self.feature_cols].values
        
        if fit_scaler:
            self.scaler = StandardScaler()
            self.scaled_features = self.scaler.fit_transform(raw_features)
        else:
            if scaler is None:
                raise ValueError("Test set requires a fitted scaler from training set!")
            self.scaler = scaler
            self.scaled_features = self.scaler.transform(raw_features)
            
        self.x_tensor, self.y_tensor = self._build_tensor_dataset()
        self._check_logic_ext()
        
    def _build_tensor_dataset(self):
        xs, ys = [], []
        for i in range(len(self.df) - self.seq_length):
            x_window = self.scaled_features[i : i+self.seq_length]
            y_val = self.df.iloc[i + self.seq_length - 1]['label']
            xs.append(x_window)
            ys.append(y_val)
        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys)).unsqueeze(1)

    def _check_logic_ext(self):
        """扩展自检函数"""
        assert not torch.isnan(self.x_tensor).any(), "[逻辑崩溃] X 张量中含有 NaN 泄露"
        assert not torch.isnan(self.y_tensor).any(), "[逻辑崩溃] Y 张量计算错误"
        assert self.x_tensor.shape[1] == self.seq_length, f"Window size expected {self.seq_length}"

    def get_loader(self, batch_size=32, shuffle=False):
        dataset = TensorDataset(self.x_tensor, self.y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_model(train_loader):
    print("\n>>> [Phase 1] Training HybridLSTM...")
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[2] 
    
    model = HybridLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    epochs = 30
    
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
        
    return model

def backtest_strategy(model, test_dataset, scaler):
    print("\n>>> [Phase 2] Backtesting with TradeManager...")
    model.eval()
    
    # [核心修改] 将 min_prob_threshold 降到 0.52，让凯利公式决定仓位
    manager = TradeManager(
        initial_capital=100000.0, 
        min_prob_threshold=0.52, 
        max_risk_per_trade=0.03, 
        risk_reward_ratio=2.0
    )
    test_loader = test_dataset.get_loader(batch_size=1, shuffle=False)
    
    df_ref = test_dataset.df
    valid_length = len(test_dataset.x_tensor)
    buy_markers, sell_markers = [], []
    
    # --- 新增：概率统计探针 ---
    predicted_probs = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= valid_length - 1: break 
            
            prob = model(x).item()
            predicted_probs.append(prob) # 记录预测概率
            
            idx_T = i + test_dataset.seq_length - 1
            if idx_T >= len(df_ref): break

            current_date = df_ref.iloc[idx_T]['date']
            current_close = df_ref.iloc[idx_T]['close']
            current_atr = df_ref.iloc[idx_T]['atr_real'] 
            
            signal = manager.on_bar_close(
                date=current_date, current_close=current_close,
                current_atr=current_atr, model_prob=prob, time_index=i
            )
            
            if idx_T + 1 < len(df_ref) and signal['action'] != 'HOLD':
                next_open = df_ref.iloc[idx_T + 1]['open']
                next_date = df_ref.iloc[idx_T + 1]['date']
                
                manager.execute_order(
                    action=signal['action'], shares=signal['shares'],
                    execute_price=next_open, date=next_date, time_index=i+1
                )
                if signal['action'] == 'BUY': buy_markers.append((next_date, next_open))
                elif signal['action'] == 'SELL': sell_markers.append((next_date, next_open))
    
    # --- 新增：打印预测概率分布情况 ---
    if predicted_probs:
        print("\n[模型预测探针] 测试集概率分布:")
        print(f"  -> 最大预测概率: {max(predicted_probs):.4f}")
        print(f"  -> 最小预测概率: {min(predicted_probs):.4f}")
        print(f"  -> 平均预测概率: {sum(predicted_probs)/len(predicted_probs):.4f}")
        print(f"  -> 大于开仓阈值(0.60)的次数: {sum(1 for p in predicted_probs if p > 0.60)}")
    
    return manager, df_ref, buy_markers, sell_markers

def plot_results(manager, df_ref, buy_markers, sell_markers):
    print("\n>>> [Phase 3] Visualizing Results...")
    asset_df = pd.DataFrame(manager.history) # 由于简化，具体绘图保留原逻辑
    print("Done. Visualization skipped in code block.")

def main():
    print(f"=== Starting Diagnostic for {TARGET_CODE} ===")
    
    train_ds = DebugDataset(
        DB_PATH, TARGET_CODE, 
        start_date='2020-01-01', end_date='2024-12-31', 
        fit_scaler=True
    )
    train_loader = train_ds.get_loader(BATCH_SIZE, shuffle=True)
    model = train_model(train_loader)
    
    try:
        test_ds = DebugDataset(
            DB_PATH, TARGET_CODE, 
            start_date='2025-01-01', end_date='2026-12-31', 
            fit_scaler=False, scaler=train_ds.scaler
        )
    except ValueError as e:
        print(f"Skipping backtest: {e}") 
        return

    manager, df_ref, buys, sells = backtest_strategy(model, test_ds, train_ds.scaler)
    
    print("\n=== Diagnosis Report ===")
    print(f"Final Assets: {manager.total_assets:.2f}")

if __name__ == "__main__":
    main()