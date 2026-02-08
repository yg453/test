AShareDataset类
1. 数据流张量分析 (Tensor Shape Analysis)
在进入代码之前，我们必须明确数据流的形状，这是模型设计成败的关键：

Input Tensor (x): (Batch_Size, 15, Feature_Dim)

15: 你的滑动窗口 (seq_length)。

Feature_Dim: 我们将构建约 6-8 个高密度特征（RSI, MACD, 均线偏离度, 波动率等）。

Target Tensor (y): (Batch_Size, 1)

你要求的是 0/1 二分类。

架构师注： 根据规范，二分类会丢失“涨幅多少”的信息。虽然我会按你的要求实现分类，但在后续模型设计中，建议考虑输出 continuous value（置信度）。

2. 特征工程策略 (Feature Engineering Strategy)
为了避免模型只记住价格（导致 10 元和 1000 元股票无法通用），我们将严格遵守以下规则：

绝对去价格化: 仅使用比率（Ratio）和变化率（Rate）。

Abupy 增强特征: 我会在代码中为你强制加入 斜率 (Slope) 和 波动率 (Volatility)，这对 LSTM 捕捉趋势至关重要。

标准化: 集成 StandardScaler，防止梯度爆炸。

3. 代码实现 (AShareDataset)
这是一个生产级的 Dataset 实现。它使用 Pandas 进行高效的向量化特征计算，然后转换为 PyTorch Tensor。

4. 关键设计点解析
内存优化:

我没有在 __getitem__ 里做运算。我一次性把所有数据算好 Feature, Label，并存为 FloatTensor。

__getitem__ 仅仅是切片操作（Slicing），这是训练速度最快的方式。

Abupy 风格的特征:

bias_20 和 slope_5 完美符合你“去价格化”的需求。

volatility 是量化模型判断变盘点的重要依据。

防止过拟合 (StandardScaler):

我在代码中使用了 StandardScaler。

注意: 这里的 fit_transform 是在整个数据集上做的（演示目的）。严格来说，你应该只在 Training Set 上 fit，然后用该参数去 transform Validation/Test Set，否则会有轻微的数据泄露（Look-ahead Bias）。