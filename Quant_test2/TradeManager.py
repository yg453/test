# -*- coding: utf-8 -*-
"""
TradeManager.py
A股交易管理器 - Kelly + ATR 动态仓位版
"""
import math

class TradeManager:
    def __init__(self, initial_capital=10000000.0, min_prob_threshold=0.52, max_risk_per_trade=0.03, risk_reward_ratio=2.0):
        """
        :param initial_capital: 初始资金
        :param min_prob_threshold: 最小开仓概率（降至0.52，依托Kelly控制风险）
        :param max_risk_per_trade: 单笔交易最大允许回撤本金比例 (如 3%)
        :param risk_reward_ratio: 预期盈亏比 (赔率 b)，用于凯利公式
        """
        self.initial_capital = initial_capital
        self.total_assets = initial_capital
        self.cash = initial_capital
        self.position = 0          # 持股数量
        self.buy_price = 0.0       # 持仓成本
        self.frozen_until = 0      # T+1 冻结时间戳
        
        self.min_prob_threshold = min_prob_threshold
        self.max_risk_per_trade = max_risk_per_trade
        self.risk_reward_ratio = risk_reward_ratio 
        self.history = []

        # [新增] 专门用于记录触发交易时的模型置信度
        self.last_signal_prob = 0.0
        
        self._check_logic()

    def _check_logic(self):
        """运行时自检"""
        assert self.initial_capital > 0, "初始资金必须大于0"
        assert 0 < self.max_risk_per_trade < 1, "风险敞口必须在 (0, 1) 之间"
        print("[System] TradeManager (Kelly Edition) initialized. Logic integrity check passed.")

    def on_bar_close(self, date, current_close, current_atr, model_prob, time_index):
        """
        每日收盘时的核心决策逻辑
        """
        signal = {'action': 'HOLD', 'shares': 0}

        # --- 卖出逻辑 (平仓) ---
        if self.position > 0:
            # 规则1: T+1 保护，今天买的不能今天卖
            if time_index < self.frozen_until:
                return signal
            
            # 规则2: 模型胜率显著恶化（低于0.48）或者触发 ATR 止损/止盈
            # 简化版：这里演示当概率反转时直接平仓
            if model_prob < 0.48:
                signal['action'] = 'SELL'
                signal['shares'] = self.position
                self.last_signal_prob = model_prob  # 记录平仓时的萎靡置信度
            return signal

        # --- 买入逻辑 (开仓) ---
        if self.position == 0 and model_prob >= self.min_prob_threshold:
            # 1. 计算凯利建议仓位系数 (Kelly Fraction)
            p = model_prob
            b = self.risk_reward_ratio
            kelly_f = p - ((1.0 - p) / b)
            
            if kelly_f > 0:
                # 2. 结合 ATR 计算基础购买力
                # 可承受的最大亏损金额
                risk_capital = self.total_assets * self.max_risk_per_trade
                # 止损距离设为 2 倍真实 ATR
                stop_distance = 2 * current_atr
                if stop_distance <= 0: stop_distance = 0.01
                
                # ATR 原生建议股数
                base_shares = risk_capital / stop_distance
                
                # 3. 核心融合：ATR 风险平价 * 凯利降维打击
                # 最高不允许超过凯利公式建议的比例，并设置 1.0 上限以防极端
                target_shares = base_shares * min(kelly_f, 1.0)
                
                # 4. A 股规则：向下取整到 100 的倍数 (1手)
                final_shares = int(target_shares // 100) * 100
                
                # 5. 资金购买力校验
                cost = final_shares * current_close
                if final_shares >= 100 and cost <= self.cash:
                    signal['action'] = 'BUY'
                    signal['shares'] = final_shares
                    self.last_signal_prob = model_prob  # 记录开仓时的爆发置信度

        return signal

    def execute_order(self, action, shares, execute_price, date, time_index):
        """执行订单撮合"""
        if action == 'BUY':
            cost = shares * execute_price
            self.cash -= cost
            self.position += shares
            self.buy_price = execute_price
            self.frozen_until = time_index + 1  # [关键] 锁定至 T+1
            self.history.append({'date': date, 'action': 'BUY', 'price': execute_price, 'shares': shares, 'reason': 'Kelly Entry'})
            print(f"[{date}] EXEC BUY: {shares} shares @ {execute_price:.2f} (模型买入置信度: {self.last_signal_prob:.2%})")
            
        elif action == 'SELL':
            revenue = shares * execute_price
            pnl_pct = (execute_price / self.buy_price - 1.0) * 100 if self.buy_price > 0 else 0
            self.cash += revenue
            self.position -= shares
            self.history.append({'date': date, 'action': 'SELL', 'price': execute_price, 'shares': shares, 'pnl': pnl_pct})
            print(f"[{date}] EXEC SELL: {shares} shares @ {execute_price:.2f}, PnL: {pnl_pct:.2f}% (模型平仓置信度: {self.last_signal_prob:.2%})")

        # 更新总资产市值
        self.total_assets = self.cash + (self.position * execute_price)