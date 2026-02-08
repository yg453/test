try:
    from config import DB_PATH, SEQ_LENGTH, FEATURE_DIM # 导入配置
except ImportError:
    # Fallback if running directly
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import DB_PATH, SEQ_LENGTH, FEATURE_DIM

import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from typing import Optional

# ==========================================
# 1. 核心交易管理器 (TradeManager)
#    复用之前的逻辑，确保资金管理一致性
# ==========================================
@dataclass
class PositionState:
    symbol: str
    shares: int = 0
    entry_price: float = 0.0
    entry_date: Optional[str] = None
    highest_price: float = 0.0
    is_holding: bool = False
    frozen_until: int = -1

class TradeManager:
    def __init__(self, initial_capital=100000.0, min_prob_threshold=0.60, risk_factor=0.02):
        self.cash = initial_capital
        self.total_assets = initial_capital
        self.position = PositionState(symbol="N/A")
        self.history = []  # 记录交易历史
        self.asset_curve = [] # 记录每日资产
        
        self.min_prob_threshold = min_prob_threshold
        self.risk_factor = risk_factor
        self.trailing_stop_pct = 0.05

    def _calculate_dynamic_position(self, prob, atr, price):
        if atr <= 0 or price <= 0: return 0
        risk_budget = self.total_assets * self.risk_factor
        base_shares = risk_budget / atr
        aggressiveness = (prob - 0.5) * 2.0
        target_shares = int(base_shares * (1 + aggressiveness))
        max_cash_shares = int(self.cash / price)
        return min(target_shares, max_cash_shares) // 100 * 100

    def on_bar_close(self, date, current_close, current_atr, model_prob, time_index):
        # 记录资产曲线
        holding_value = self.position.shares * current_close
        self.total_assets = self.cash + holding_value
        self.asset_curve.append({'date': date, 'total_assets': self.total_assets})

        signal = {"action": "HOLD", "shares": 0}
        
        if self.position.is_holding:
            self.position.highest_price = max(self.position.highest_price, current_close)
            if time_index >= self.position.frozen_until:
                drawdown = (self.position.highest_price - current_close) / self.position.highest_price
                if drawdown >= self.trailing_stop_pct:
                    signal = {"action": "SELL", "shares": self.position.shares, "reason": "Trailing Stop"}
                elif model_prob < 0.4: # 概率走弱止盈
                    signal = {"action": "SELL", "shares": self.position.shares, "reason": "Weak Signal"}
        else:
            if model_prob > self.min_prob_threshold:
                target_shares = self._calculate_dynamic_position(model_prob, current_atr, current_close)
                if target_shares > 0:
                    signal = {"action": "BUY", "shares": target_shares, "reason": "Signal Entry"}
        
        return signal

    def execute_order(self, action, shares, price, date, time_index):
        if action == "BUY" and self.cash >= shares * price:
            cost = shares * price * 1.0003
            self.cash -= cost
            self.position.shares = shares
            self.position.entry_price = price
            self.position.highest_price = price
            self.position.is_holding = True
            self.position.frozen_until = time_index + 1
            print(f"[{date}] 买入 {shares} 股 @ {price:.2f}")
        
        elif action == "SELL" and self.position.is_holding:
            revenue = shares * price * (1 - 0.0013)
            self.cash += revenue
            pnl = (price - self.position.entry_price) / self.position.entry_price
            self.history.append({'date': date, 'pnl': pnl, 'type': 'SELL'})
            self.position.shares = 0
            self.position.is_holding = False
            print(f"[{date}] 卖出 {shares} 股 @ {price:.2f}, 盈亏: {pnl:.2%}")

# ==========================================
# 2. 模型定义 (Simple LSTM)
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# ==========================================
# 3. 数据处理与特征工程
# ==========================================
def load_and_process_data(db_path, symbol='000001'):
    # 1. 连接数据库读取 (模拟)
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT date, open, high, low, close, volume FROM stock_data WHERE symbol='{symbol}' ORDER BY date"
        df = pd.read_sql(query, conn)
        conn.close()
    except:
        print("Warning: Database not found. Generating Mock Data for testing.")
        dates = pd.date_range(start='2020-01-01', end='2025-12-31')
        df = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(10, 20, len(dates)),
            'close': np.random.uniform(10, 20, len(dates)),
            'high': np.random.uniform(10, 20, len(dates)),
            'low': np.random.uniform(10, 20, len(dates)),
            'volume': np.random.uniform(1000, 5000, len(dates))
        })
        df['close'] = df['close'].cumsum() # Make it look like a trend
        df['high'] = df['close'] + 0.5
        df['low'] = df['close'] - 0.5

    df['date'] = pd.to_datetime(df['date'])
    
    # 2. 计算 ATR (用于仓位管理，非特征)
    df['tr'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    
    # 3. 构建标签 (未来3天上涨为1)
    df['target'] = (df['close'].shift(-3) > df['close']).astype(int)
    
    # 4. 清洗数据
    df.dropna(inplace=True)
    return df

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length] # Target is at the end of sequence
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ==========================================
# 4. 主流程 (Main)
# ==========================================
def main():
    # 参数配置
    SEQ_LENGTH = 30
    HIDDEN_DIM = 64
    EPOCHS = 10  # 演示用，实际建议 50+
    
    # 1. 数据准备
    print(">>> [Phase 1] Loading Data...")
    df = load_and_process_data(DB_PATH)
    
    # 划分训练集 (2020-2024) 和 测试集 (2025-)
    train_mask = (df['date'] >= '2020-01-01') & (df['date'] <= '2024-12-31')
    test_mask = (df['date'] >= '2025-01-01')
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")

    # 特征归一化 (关键：只在训练集上 Fit)
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'atr']
    scaler = MinMaxScaler()
    
    # Fit & Transform Train
    train_feats = scaler.fit_transform(df_train[feature_cols].values)
    train_targets = df_train['target'].values
    
    # Transform Test (No Fit!)
    test_feats = scaler.transform(df_test[feature_cols].values)
    # test_targets = df_test['target'].values # 测试集不需要Target来训练，只需要来验证
    
    # 创建序列
    X_train, y_train = create_sequences(train_feats, SEQ_LENGTH)
    # 测试集序列构建略有不同，需要保留对应的原始数据索引以进行回测
    X_test, _ = create_sequences(test_feats, SEQ_LENGTH)
    
    # 转为 Tensor
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)

    # 2. 模型训练
    print(">>> [Phase 2] Training LSTM Model...")
    model = LSTMModel(input_dim=len(feature_cols), hidden_dim=HIDDEN_DIM, num_layers=2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

    # 3. 滚动回测 (Walk-Forward Testing)
    print(">>> [Phase 3] Walk-Forward Testing (2025-Now)...")
    model.eval()
    engine = TradeManager(initial_capital=100000)
    
    # 对齐索引：X_test[i] 对应的是 df_test 中第 (SEQ_LENGTH + i) 行的数据
    # 我们需要在 T 日收盘后预测，并在 T+1 日执行
    
    with torch.no_grad():
        test_preds = model(X_test_t).numpy()

    # 遍历测试集 (注意索引对齐)
    # X_test 的长度比 df_test 少 SEQ_LENGTH
    for i in range(len(test_preds) - 1):
        # 当前是 T 日
        current_idx_in_df = SEQ_LENGTH + i
        current_date = df_test.iloc[current_idx_in_df]['date']
        current_close = df_test.iloc[current_idx_in_df]['close']
        current_atr = df_test.iloc[current_idx_in_df]['atr']
        
        # 模型预测的 T+N 日上涨概率
        prob = float(test_preds[i])
        
        # --- 决策 (T日收盘) ---
        signal = engine.on_bar_close(
            date=current_date,
            current_close=current_close,
            current_atr=current_atr,
            model_prob=prob,
            time_index=i
        )
        
        # --- 执行 (T+1日开盘) ---
        if signal['action'] != 'HOLD':
            next_idx = current_idx_in_df + 1
            if next_idx < len(df_test):
                next_open = df_test.iloc[next_idx]['open'] # 实盘是次日开盘价
                next_date = df_test.iloc[next_idx]['date']
                
                engine.execute_order(
                    action=signal['action'],
                    shares=signal['shares'],
                    price=next_open,
                    date=next_date,
                    time_index=i+1
                )

    # 4. 报告与绘图
    print(">>> [Phase 4] Generating Report...")
    
    # 提取资金曲线
    if not engine.asset_curve:
        print("No data to plot.")
        return

    curve_df = pd.DataFrame(engine.asset_curve)
    curve_df['date'] = pd.to_datetime(curve_df['date'])
    curve_df.set_index('date', inplace=True)
    
    # 计算胜率
    wins = [x for x in engine.history if x['pnl'] > 0]
    win_rate = len(wins) / len(engine.history) if engine.history else 0
    total_return = (engine.total_assets - 100000) / 100000
    
    print(f"Final Assets: {engine.total_assets:.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Win Rate: {win_rate:.2%} ({len(wins)}/{len(engine.history)})")
    
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(curve_df.index, curve_df['total_assets'], label='Strategy Equity')
    plt.title(f'Backtest Result (2025-Now)\nWin Rate: {win_rate:.2%} | Return: {total_return:.2%}')
    plt.xlabel('Date')
    plt.ylabel('Asset Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('backtest_result.png')
    print("Result saved to backtest_result.png")

if __name__ == '__main__':
    main()