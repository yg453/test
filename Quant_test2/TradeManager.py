import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class PositionState:
    symbol: str
    shares: int = 0
    entry_price: float = 0.0
    entry_date: Optional[str] = None
    highest_price: float = 0.0  # 用于移动止损
    is_holding: bool = False
    frozen_until: int = -1  # T+1 锁定期，存储可以卖出的索引

class TradeManager:
    """
    基于 Abupy 思想的抗过拟合交易管理器
    集成：PyTorch概率信号解析 + 动态仓位管理 + 移动止损
    """
    
    def __init__(self, initial_capital: float = 100000.0, 
                 min_prob_threshold: float = 0.55,
                 risk_factor: float = 0.02,
                 max_position_pct: float = 0.8):
        """
        :param initial_capital: 初始资金
        :param min_prob_threshold: 开仓概率阈值（低于此值不开仓）
        :param risk_factor: 风险暴露因子（类似 Abupy 的 risk_gains），用于控制单笔交易亏损上限
        :param max_position_pct: 最大单只股票持仓比例（风控硬约束）
        """
        self.cash = initial_capital
        self.total_assets = initial_capital
        self.position = PositionState(symbol="N/A")
        self.history = []
        
        # 策略参数
        self.min_prob_threshold = min_prob_threshold
        self.risk_factor = risk_factor  # 假设单笔交易愿意承担总资金 2% 的波动风险
        self.max_position_pct = max_position_pct
        self.trailing_stop_pct = 0.05   # 5% 回撤止损

        self._check_logic_integrity()

    def _check_logic_integrity(self):
        """启动时自检"""
        assert 0 < self.min_prob_threshold < 1, "Probability threshold must be between 0 and 1"
        assert 0 < self.max_position_pct <= 1, "Max position must be <= 100%"
        print("[System] TradeManager initialized. Logic integrity check passed.")

    def _calculate_dynamic_position(self, prob: float, atr: float, price: float) -> int:
        """
        核心仓位管理逻辑 (Kelly-like + Volatility Scaling)
        逻辑：
        1. 基础仓位：根据 ATR 计算，确保 1倍 ATR 的波动只亏损总资金的 risk_factor (如2%)。
           Base Shares = (Total Capital * Risk Factor) / ATR
        2. 概率加权：
           Weight = (Prob - 0.5) * 2  -> Prob 0.5时为0, Prob 1.0时为1
           User Logic: Prob 0.8 -> Weight 0.6. 
        3. 最终仓位 = Base Shares * (1 + Weight) * 调优系数
        """
        if atr <= 0 or price <= 0:
            return 0
        
        # 1. 风险平价基础量：让账户承受的波动风险恒定
        risk_budget = self.total_assets * self.risk_factor
        base_shares = risk_budget / atr  # ATR越大，Shares越少
        
        # 2. 概率激进程度修正 (Abupy 凯利思想变种)
        # 将 0.5~1.0 的概率映射到 0.0~2.0 的激进倍数
        # Prob=0.55 -> Multiplier=0.1
        # Prob=0.80 -> Multiplier=0.6
        # Prob=0.90 -> Multiplier=0.8
        aggressiveness = (prob - 0.5) * 2.0 
        
        target_shares = int(base_shares * (1 + aggressiveness))
        
        # 3. 硬约束：最大持仓比例限制
        max_allowed_cash = self.total_assets * self.max_position_pct
        max_shares_by_cash = int(max_allowed_cash / price)
        
        # 4. 硬约束：现金限制
        max_shares_by_current_cash = int(self.cash / price)
        
        final_shares = min(target_shares, max_shares_by_cash, max_shares_by_current_cash)
        
        # A股一手100股逻辑
        final_shares = (final_shares // 100) * 100
        
        return final_shares

    def on_bar_close(self, date: str, current_close: float, current_atr: float, 
                     model_prob: float, time_index: int) -> dict:
        """
        T日收盘后运行：更新状态，计算移动止损，生成次日交易计划
        :param current_atr: 绝对值 ATR (e.g., 2.5 元)
        :param model_prob: PyTorch 模型输出的上涨概率 (0~1)
        :param time_index: 当前时间索引 (用于 T+1 判断)
        """
        signal = {"action": "HOLD", "shares": 0, "reason": ""}
        
        # 1. 数据防御
        if np.isnan(current_close) or np.isnan(current_atr) or current_close <= 0:
            return signal

        # 2. 更新持仓状态 (如果持有)
        if self.position.is_holding:
            # 更新最高价用于移动止损
            self.position.highest_price = max(self.position.highest_price, current_close)
            
            # --- 卖出逻辑检查 (Stop Loss / Profit Check) ---
            # 只有满足 T+1 才能卖出
            if time_index >= self.position.frozen_until:
                # 逻辑：最高价回撤 > 5%
                drawdown = (self.position.highest_price - current_close) / self.position.highest_price
                
                if drawdown >= self.trailing_stop_pct:
                    signal = {
                        "action": "SELL", 
                        "shares": self.position.shares,
                        "reason": f"Trailing Stop: Drawdown {drawdown:.2%} > {self.trailing_stop_pct:.2%}"
                    }
                # 也可以加入基于概率的止盈：如果概率突然暴跌到 0.4 以下，提前离场
                elif model_prob < 0.4:
                    signal = {
                        "action": "SELL",
                        "shares": self.position.shares,
                        "reason": f"Model Signal Weak: {model_prob:.2f}"
                    }
        
        # 3. 买入逻辑检查 (仅当空仓时，或者支持加仓逻辑)
        # 这里简化为：空仓才买入
        elif not self.position.is_holding:
            if model_prob > self.min_prob_threshold:
                target_shares = self._calculate_dynamic_position(model_prob, current_atr, current_close)
                if target_shares > 0:
                    signal = {
                        "action": "BUY",
                        "shares": target_shares,
                        "reason": f"Entry: Prob {model_prob:.2f} > {self.min_prob_threshold}, ATR {current_atr:.2f}"
                    }
        
        return signal

    def execute_order(self, action: str, shares: int, execute_price: float, 
                      date: str, time_index: int):
        """
        T+1日开盘执行订单
        注意：实际回测中，这里通常传入 T+1 的 Open 价
        """
        if action == "BUY":
            cost = shares * execute_price * 1.0003 # 加上万三佣金
            if self.cash >= cost:
                self.cash -= cost
                self.position.symbol = "000001" # 示例
                self.position.shares = shares
                self.position.entry_price = execute_price
                self.position.highest_price = execute_price
                self.position.entry_date = date
                self.position.is_holding = True
                self.position.frozen_until = time_index + 1 # T+1 锁定，明天及以后才能卖
                print(f"[{date}] EXEC BUY: {shares} shares @ {execute_price:.2f}")
            else:
                print(f"[{date}] REJECT BUY: Insufficient Cash")

        elif action == "SELL":
            if self.position.is_holding and shares <= self.position.shares:
                revenue = shares * execute_price * (1 - 0.0013) # 减去千一印花税+万三佣金
                self.cash += revenue
                pnl = (execute_price - self.position.entry_price) / self.position.entry_price
                
                print(f"[{date}] EXEC SELL: {shares} shares @ {execute_price:.2f}, PnL: {pnl:.2%}")
                
                # 重置持仓
                self.position.shares = 0
                self.position.is_holding = False
                self.position.highest_price = 0.0
        
        # 更新总资产市值
        holding_value = self.position.shares * execute_price
        self.total_assets = self.cash + holding_value

# ==========================================
# 模拟回测调用示例 (防止未来函数演示)
# ==========================================
def mock_backtest_loop():
    # 模拟数据：日期, 收盘价, 开盘价(次日), ATR, 预测概率
    # 注意：next_open 是 T+1 的数据，仅在 execute 阶段由于
    data = pd.DataFrame({
        'close': [100, 102, 105, 104, 98],
        'next_open': [101, 103, 104, 100, 97], # T+1 开盘价
        'atr': [2.0, 2.1, 1.5, 3.0, 3.5],
        'prob': [0.85, 0.70, 0.60, 0.30, 0.40] # 模型T日收盘后预测
    })
    
    engine = TradeManager(initial_capital=100000)
    
    for t in range(len(data)):
        # 1. 获取 T 日数据
        current_data = data.iloc[t]
        date_str = f"Day_{t}"
        
        # 2. 决策阶段 (T日收盘)
        # 输入：T日收盘价, T日ATR, T日模型预测
        # 严禁传入 next_open
        signal = engine.on_bar_close(
            date=date_str,
            current_close=current_data['close'],
            current_atr=current_data['atr'],
            model_prob=current_data['prob'],
            time_index=t
        )
        
        # 3. 执行阶段 (T+1日开盘)
        # 如果有信号，尝试在次日开盘执行
        if signal['action'] != "HOLD":
            # 必须检查是否越界
            if t < len(data) - 1:
                next_day_open = current_data['next_open'] 
                # 这里模拟 T+1 撮合
                engine.execute_order(
                    action=signal['action'],
                    shares=signal['shares'],
                    execute_price=next_day_open,
                    date=f"Day_{t+1}_Open",
                    time_index=t+1 # 执行时间是 T+1
                )
            else:
                print("End of data, cannot execute next day.")

mock_backtest_loop()