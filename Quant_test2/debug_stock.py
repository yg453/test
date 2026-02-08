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

# --- 1. 导入配置与模块 ---
try:
    from config import DB_PATH, SEQ_LENGTH, FEATURE_DIM, TARGET_DAYS
    from AShareDataset import AShareDataset
    from HybridLSTM import HybridLSTM
    from TradeManager import TradeManager
except ImportError as e:
    print(f"CRITICAL: 模块导入失败。请确保 config.py, AShareDataset.py, HybridLSTM.py, TradeManager.py 在同一目录。\n错误详情: {e}")
    sys.exit(1)

# 硬编码调试目标
TARGET_CODE = 'sh.600000' # 浦发银行 (示例)
BATCH_SIZE = 32

# --- 2. 扩展数据集类 (解决单股票读取与Scaler泄露问题) ---
class DebugDataset(AShareDataset):
    """
    继承 AShareDataset，但覆盖初始化逻辑以支持：
    1. 仅读取特定股票 (Speed Optimization)
    2. 分离 Scaler 的 Fit 和 Transform (Anti-Leakage)
    """
    def __init__(self, db_path, code, start_date, end_date, scaler=None, fit_scaler=False):
        # 不调用 super().__init__，因为原版会强制读取全量数据并立即标准化
        self.seq_length = SEQ_LENGTH
        self.target_days = TARGET_DAYS
        
        # A. 定向读取数据
        print(f"Loading data for {code} [{start_date} to {end_date}]...")
        conn = sqlite3.connect(db_path)
        # 兼容表名，根据你的 AShareDataset.py 默认为 daily_kline
        query = f"""
            SELECT date, code, open, high, low, close, volume 
            FROM daily_kline 
            WHERE code='{code}' AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date
        """
        try:
            self.df = pd.read_sql(query, conn)
        except Exception as e:
            print(f"Database Read Error: {e}")
            self.df = pd.DataFrame() # Fallback for empty
        conn.close()
        
        if self.df.empty:
            raise ValueError(f"No data found for {code} in {db_path}")

        # B. 复用父类的特征工程
        self._generate_features() 
        self._generate_labels()
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True) # 关键：重置索引以对其
        
        # C. 严格的标准化处理 (防止 Look-ahead Bias)
        feature_cols = [c for c in self.df.columns if c not in ['date', 'code', 'label', 'target_return']]
        # 确保列序一致
        self.feature_cols = sorted(feature_cols) 
        
        raw_features = self.df[self.feature_cols].values
        
        if fit_scaler:
            # 训练集：Fit + Transform
            self.scaler = StandardScaler()
            self.scaled_features = self.scaler.fit_transform(raw_features)
        else:
            # 测试集：仅 Transform (必须传入训练好的 Scaler)
            if scaler is None:
                raise ValueError("Test set requires a fitted scaler from training set!")
            self.scaler = scaler
            self.scaled_features = self.scaler.transform(raw_features)
            
        # D. 转换为 Tensor
        self.x_tensor, self.y_tensor = self._build_tensor_dataset()
        
    def _build_tensor_dataset(self):
        """构建滑动窗口 Tensor"""
        xs, ys = [], []
        # 遍历数据构建序列
        # 注意：feature_data 的长度为 N
        # i 从 0 到 N - SEQ_LENGTH
        # X: [i : i+SEQ_LENGTH]
        # Y: [i+SEQ_LENGTH-1] 的 label (对应未来)
        
        for i in range(len(self.df) - self.seq_length):
            x_window = self.scaled_features[i : i+self.seq_length]
            # Label 对齐：AShareDataset 的 Label 是预先计算好的 (shift 过的)
            # 所以取窗口最后一天的 label 即可
            y_val = self.df.iloc[i + self.seq_length - 1]['label']
            
            xs.append(x_window)
            ys.append(y_val)
            
        return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys)).unsqueeze(1)

    def get_loader(self, batch_size=32, shuffle=False):
        dataset = TensorDataset(self.x_tensor, self.y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# --- 3. 训练函数 ---
def train_model(train_loader):
    print("\n>>> [Phase 1] Training HybridLSTM...")
    # 动态获取特征维度
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[2] 
    
    model = HybridLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    epochs = 10
    loss_history = []
    
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
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
        
    return model

# --- 4. 回测函数 ---
def backtest_strategy(model, test_dataset, scaler):
    print("\n>>> [Phase 2] Backtesting with TradeManager...")
    model.eval()
    
    # 实例化交易管理器
    manager = TradeManager(initial_capital=100000.0, min_prob_threshold=0.60)
    
    # 获取测试集 Tensor (不 shuffle，保证时间顺序)
    test_loader = test_dataset.get_loader(batch_size=1, shuffle=False)
    
    # 提取测试集的原始 DataFrame 用于获取价格
    # 注意：DataSet 构建时丢弃了前 seq_length 的数据作为 pre-buffer
    # 所以测试集的有效数据是从 seq_length 开始的
    # 我们需要对齐索引：Test_Loader[i] 对应 Dataset.df.iloc[i + SEQ_LENGTH] 吗？
    # 不，DebugDataset 构建时 xs[0] 是 df[0:15]，对应的当前时间点是 df[14]。
    # 所以 prediction[i] 对应的时间点是 df.iloc[i + SEQ_LENGTH - 1]
    
    df_ref = test_dataset.df
    valid_length = len(test_dataset.x_tensor)
    
    logs = []
    buy_markers = []
    sell_markers = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= valid_length - 1: break # 防止越界
            
            # 1. 模型预测 (T日收盘后)
            prob = model(x).item()
            
            # 2. 获取上下文数据
            # 对应的行索引是 i + SEQ_LENGTH - 1
            # 这里的逻辑：Sequence 是 [T-14 ... T]，预测基于 T 的信息
            # 2. 获取上下文数据
            # 修正逻辑：i 是 DataLoader 的批次索引
            # X[i] 对应的数据窗口是 df[i : i+seq_len]
            # 我们需要的是窗口【最后一天】的数据作为 T日，来获取 T+1日的价格
            seq_len = test_dataset.seq_length
            idx_T = i + seq_len - 1
            
            # 防御性检查：确保索引不越界
            if idx_T >= len(df_ref):
                break

            current_date = df_ref.iloc[idx_T]['date']
            current_close = df_ref.iloc[idx_T]['close']
            # 修正：DebugDataset循环是从 0 到 len-seq，xs[0] 对应 df[0:seq_len]
            # 窗口最后一个点索引 = i + seq_length - 1
            idx_T = i + test_dataset.seq_length - 1
            
            current_date = df_ref.iloc[idx_T]['date']
            current_close = df_ref.iloc[idx_T]['close']
            
            # 计算 ATR (简单模拟，因为 DebugDataset 特征可能没保留 ATR 原值)
            # 如果特征里有 volatility，可以用它反推，这里简化计算
            high = df_ref.iloc[idx_T]['close'] * 1.02 # Mock high
            low = df_ref.iloc[idx_T]['close'] * 0.98  # Mock low
            current_atr = (high - low) # 简化版 ATR
            
            # 3. 生成信号 (T日决策)
            signal = manager.on_bar_close(
                date=current_date,
                current_close=current_close,
                current_atr=current_atr,
                model_prob=prob,
                time_index=i
            )
            
            # 4. 执行交易 (T+1日开盘)
            # 获取 T+1 数据
            if idx_T + 1 < len(df_ref):
                next_open = df_ref.iloc[idx_T + 1]['open']
                next_date = df_ref.iloc[idx_T + 1]['date']
                
                if signal['action'] != 'HOLD':
                    manager.execute_order(
                        action=signal['action'],
                        shares=signal['shares'],
                        price=next_open,
                        date=next_date,
                        time_index=i+1
                    )
                    
                    # 记录 Marker 用于绘图
                    if signal['action'] == 'BUY':
                        buy_markers.append((next_date, next_open))
                    elif signal['action'] == 'SELL':
                        sell_markers.append((next_date, next_open))
    
    return manager, df_ref, buy_markers, sell_markers

# --- 5. 可视化 ---
def plot_results(manager, df_ref, buy_markers, sell_markers):
    print("\n>>> [Phase 3] Visualizing Results...")
    
    # 准备数据
    asset_df = pd.DataFrame(manager.asset_curve)
    if asset_df.empty:
        print("No assets data recorded.")
        return
        
    asset_df['date'] = pd.to_datetime(asset_df['date'])
    asset_df.set_index('date', inplace=True)
    
    # 截取回测期间的基准价格 (Buy & Hold)
    # 找到回测开始的第一个日期
    start_date = asset_df.index[0]
    df_ref['date'] = pd.to_datetime(df_ref['date'])
    mask = df_ref['date'] >= start_date
    benchmark_df = df_ref[mask].copy()
    benchmark_df.set_index('date', inplace=True)
    
    # 归一化用于对比
    initial_value = manager.history[0]['total_assets'] if manager.history else 100000
    asset_df['Strategy_Norm'] = asset_df['total_assets'] / 100000
    benchmark_df['Benchmark_Norm'] = benchmark_df['close'] / benchmark_df['close'].iloc[0]
    
    # 绘图
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 子图1: K线与买卖点
    ax1.plot(benchmark_df.index, benchmark_df['close'], label='Stock Price', color='gray', alpha=0.5)
    
    if buy_markers:
        bx, by = zip(*buy_markers)
        ax1.scatter(pd.to_datetime(bx), by, marker='^', color='red', s=100, label='Buy Signal', zorder=5)
    
    if sell_markers:
        sx, sy = zip(*sell_markers)
        ax1.scatter(pd.to_datetime(sx), sy, marker='v', color='green', s=100, label='Sell Signal', zorder=5)
        
    ax1.set_title(f'Trade Signals: {TARGET_CODE}')
    ax1.legend()
    ax1.grid(True)
    
    # 子图2: 资金曲线对比
    ax2.plot(asset_df.index, asset_df['Strategy_Norm'], label='AI Strategy', color='blue', linewidth=2)
    ax2.plot(benchmark_df.index, benchmark_df['Benchmark_Norm'], label='Buy & Hold', color='orange', linestyle='--')
    ax2.set_title('Equity Curve (Normalized)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = "debug_result.png"
    plt.savefig(plot_path)
    print(f"Chart saved to {plot_path}")
    plt.show()

# --- 6. 主程序 ---
def main():
    print(f"=== Starting Diagnostic for {TARGET_CODE} ===")
    
    # 1. 准备训练集 (2020-2024)
    train_ds = DebugDataset(
        DB_PATH, TARGET_CODE, 
        start_date='2020-01-01', end_date='2024-12-31', 
        fit_scaler=True
    )
    train_loader = train_ds.get_loader(BATCH_SIZE, shuffle=True)
    
    # 2. 训练模型
    model = train_model(train_loader)
    
    # 3. 准备测试集 (2025-2026)
    # 关键：传入训练集的 scaler
    try:
        test_ds = DebugDataset(
            DB_PATH, TARGET_CODE, 
            start_date='2025-01-01', end_date='2026-12-31', 
            fit_scaler=False, scaler=train_ds.scaler
        )
    except ValueError as e:
        print(f"Skipping backtest: {e}") # 可能是没有2025的数据
        return

    if len(test_ds.df) < SEQ_LENGTH + 5:
        print("Not enough test data for backtesting.")
        return

    # 4. 运行回测
    manager, df_ref, buys, sells = backtest_strategy(model, test_ds, train_ds.scaler)
    
    # 5. 输出结果
    print("\n=== Diagnosis Report ===")
    print(f"Final Assets: {manager.total_assets:.2f}")
    if manager.history:
        win_count = sum(1 for h in manager.history if h['pnl'] > 0)
        print(f"Trades: {len(manager.history)}")
        print(f"Win Rate: {win_count/len(manager.history):.2%}")
    else:
        print("No completed trades.")
        
    # 6. 绘图
    plot_results(manager, df_ref, buys, sells)

if __name__ == "__main__":
    main()