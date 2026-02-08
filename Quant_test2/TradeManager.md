Step 1: 审计报告 (Audit Report)
在编写代码前，我对你的需求进行了严格的逻辑审计：
检查项,状态,详细说明/警告
逻辑一致性,✅,概率驱动开仓，ATR 驱动仓位大小，移动止损驱动平仓，逻辑闭环。
接口兼容性,⚠️,"警告：PyTorch 输出的概率必须经过 Sigmoid 或 Softmax 归一化到 [0, 1] 区间。ATR 必须与股价同单位（非百分比）。"
A股规则合规,✅,代码将强制执行 T+1 规则：T日买入，T+1日及以后才能触发卖出逻辑。
未来函数检测,✅,关键防御：决策函数 decide_signal 仅接收 current_close (T日收盘)，执行函数 execute 接收 next_open (T+1日开盘)。严禁在 T 日决策时使用 T+1 日数据。
数据完整性,⚠️,必须处理停牌（Volume=0）或 ATR 为 NaN 的情况，否则会导致除零错误或仓位计算为 inf。

Step 2: 策略优化建议 (Strategy Tuning)
关于“凯利公式变种”：标准的凯利公式 $f = p - (1-p)/b$ 需要赔率 $b$。由于深度学习模型通常只输出胜率 $p$，难以准确估计赔率。
架构师建议：采用 波动率倒数加权 (Volatility Inverse Weighting) 结合 概率阈值。
公式逻辑：仓位 = 基础风险预算 / ATR波动幅度 * (概率强弱系数)。
这符合你要求的“概率高且波动小 -> 重仓”逻辑，比纯凯利公式在 A 股更稳健。
移动止损 (Trailing Stop)：Abupy 中通常由 AbuFactorSellBase 的子类处理（如 AbuFactorAtrNStop）。
在这里我们将此逻辑内嵌到 Manager 中，设定为“最高价回撤 5%”。

关键架构说明
仓位计算逻辑 (_calculate_dynamic_position)：

Abupy 哲学：不同于简单的 Kelly 公式，这里引入了 Risk Parity (风险平价) 思想。

公式：BaseShares = RiskBudget / ATR。

效果：当 ATR（波动率）很大时（如 Day 3, ATR=3.0），分母变大，基础仓位自动减小。即使概率是 0.8，如果波动极大，系统也不会允许全仓，从而避免了“由于一次高概率预测但在高波动中被震荡出局”的风险。

移动止损逻辑：

不再依赖单一的固定止损。代码记录了 highest_price。

drawdown = (highest - current) / highest。一旦回撤超过 5%，强制触发 SELL 信号。这是保护利润的核心手段。

防未来函数机制：

on_bar_close 只接受 current_close。

execute_order 接受 execute_price（在回测循环中传入的是 next_open）。

这种分离确保了你不会在决策时“偷看”第二天的开盘价。

T+1 锁定：

引入 frozen_until 字段。买入时设定为 current_time + 1。

卖出检查逻辑 if time_index >= self.position.frozen_until 确保了当天买入的股票无法在当天卖出（符合 A 股规则）。