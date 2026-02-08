这个脚本将集成 数据工厂 (Data Processing)、模型工厂 (Model Factory) 和 交易内核 (Execution Kernel)。

设计重点 (Architectural Highlights)
数据隔离 (Data Isolation)：Scaler（归一化器）只在训练集上拟合 (Fit)，然后应用到测试集。这是防止“前视偏差 (Look-ahead Bias)”的关键步骤。

滚动窗口 (Rolling Window)：在回测阶段，模拟真实的“每日收盘”场景：取过去 N 天数据 -> 预测 -> 决策 -> 次日执行。

模块化：即使在一个文件中，也将 Model、Data、Strategy 严格分拆。

架构师注释 (Architect's Notes)
索引对齐的艺术： 在回测循环中，最容易出错的是索引。请注意 current_idx_in_df = SEQ_LENGTH + i。

X_test 的第 i 个数据，实际上是由 df_test 中从 i 到 i+SEQ_LENGTH 的切片组成的。

所以预测做出的决策，对应的时间点是 df_test 的第 SEQ_LENGTH + i 行。

Mock Data (模拟数据)： 代码中包含了一个 try-except 块。如果你没有真实的 stocks.db，脚本会自动生成模拟数据运行，确保你可以立即看到代码逻辑是如何跑通的。

T+1 严格执行： 注意代码中的 next_open = df_test.iloc[next_idx]['open']。所有的交易执行（买/卖）都是用次日开盘价撮合的，而不是当天的收盘价。这是 A 股回测最基本的底线。