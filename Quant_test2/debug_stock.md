Step 1: 审计报告 (Audit Report)
检查项,状态,详细说明/警告
逻辑一致性,✅,数据加载 -> 特征工程 -> 拆分 -> 训练 -> 回测 -> 绘图，流程闭环。
接口兼容性,⚠️,关键警告：原 AShareDataset 在 __init__ 中会对加载的数据做 fit_transform。如果在 Test Set 上直接实例化它，会造成严重的数据泄露（Test Set 参与了均值方差计算）。解决方案：本脚本将重写数据加载逻辑，强制使用 Train Set 的 Scaler 参数去转换 Test Set。
A股规则合规,✅,回测循环严格遵守：T日收盘预测 -> T日生成信号 -> T+1日开盘撮合。
未来函数检测,✅,特征计算使用了 rolling 和 shift；回测时仅使用当前可见窗口的数据。

Step 2: 架构策略说明
为了解决上述“接口兼容性”中提到的数据泄露风险，本脚本采用了 "Headless Mode" 策略：

我们继承 AShareDataset 仅复用其特征工程代码 (_generate_features)。

我们在脚本主流程中手动控制 StandardScaler 的 fit 和 transform 边界。

我们通过 SQL WHERE 子句仅读取单只股票，极大提升调试速度。

架构师注释 (Architect's Notes)
关于 DebugDataset 类：

我没有直接实例化 AShareDataset，因为原类的 __init__ 会强制读取整个数据库表。在调试单只股票时，这非常低效。

我重写了 _build_tensor_dataset 逻辑，确保 Sequence (X) 和 Label (Y) 的对齐是透明的。

关于 Scaler 泄露 (关键)：

代码中 train_ds 设置了 fit_scaler=True。

test_ds 显式设置了 fit_scaler=False 并传入了 scaler=train_ds.scaler。

这是量化回测中最基本的底线：严禁在测试集上重新计算均值和方差。

关于 ATR 的简化：

在回测循环中，因为 HybridLSTM 的特征工程可能已经对数据进行了归一化，导致很难还原绝对值的 ATR。

为了调试运行通畅，我在代码中用 (High - Low) 做了一个简化的 ATR 估算。正式上线时，建议在 TradeManager 内部通过原始价格计算标准 ATR。

运行前检查：

请确保 Database_test/stocks.db 存在且其中包含 sh.600000 在 2020-2025 年的数据。如果数据库是空的或没有这只股票，脚本会报错提示。