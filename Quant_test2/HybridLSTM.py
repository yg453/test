import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    时间步注意力机制 (Temporal Attention Mechanism)
    目的: 自动学习过去 N 天中，哪一天对当前的预测最重要。
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        # 这里的 Linear 层用于学习每个时间步的"重要性分数"
        self.query = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Input x: [Batch_Size, Seq_Length, Hidden_Dim]
        
        # 1. 计算原始注意力分数 (Attention Scores)
        # scores: [Batch_Size, Seq_Length, 1]
        scores = self.query(x) 
        
        # 2. 归一化为概率分布 (Attention Weights)
        # weights: [Batch_Size, Seq_Length, 1]
        # Dim=1 代表在时间轴上做 Softmax，保证所有天的权重之和为 1
        weights = F.softmax(scores, dim=1) 
        
        # 3. 加权求和 (Context Vector)
        # 我们用计算出的权重对 LSTM 的输出进行加权
        # context: [Batch_Size, Hidden_Dim]
        # Operation: Sum( (Batch, Seq, Hidden) * (Batch, Seq, 1) ) -> (Batch, Hidden)
        context = torch.sum(x * weights, dim=1)
        
        return context, weights

class HybridLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        """
        Args:
            input_dim: 输入特征维度 (Feature_Dim)
            hidden_dim: LSTM 隐藏层维度 (Default: 128)
            num_layers: LSTM 层数 (Default: 2)
            dropout: Dropout 概率 (Default: 0.3)
        """
        super(HybridLSTM, self).__init__()
        
        # 1. 主干网络: 双层 LSTM
        # batch_first=True 使得输入 Tensor 格式为 (Batch, Seq, Feature)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # 2. 注意力层: 捕捉关键转折点
        self.attention = TemporalAttention(hidden_dim)
        
        # 3. 规范化与正则化
        # 在进入全连接层之前再次 Drop，防止对特定特征的过拟合
        # 符合 "防止过拟合机制" 规范
        self.dropout = nn.Dropout(dropout)
        
        # 4. 输出头 (Policy Head)
        # 用于二分类 (买/不买)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Standard Forward Pass with Shape Annotations
        """
        # --------------------------------------------
        # 1. Input Processing
        # x: [Batch_Size, 15, Feature_Dim]
        # --------------------------------------------
        
        # 2. LSTM Extraction
        # lstm_out: [Batch_Size, 15, 128] -> 包含了过去 15 天每一天的 Hidden State
        # _ (hidden_state): [2, Batch_Size, 128] -> 我们不需要最后的隐状态，因为我们有 Attention
        lstm_out, _ = self.lstm(x)
        
        # 3. Attention Aggregation
        # context: [Batch_Size, 128] -> 聚合后的全局特征向量
        # weights: [Batch_Size, 15, 1] -> 每一天的重要性 (可用于可视化解释模型逻辑)
        context, weights = self.attention(lstm_out)
        
        # 4. Regularization
        # context: [Batch_Size, 128]
        out = self.dropout(context)
        
        # 5. Classification Head
        # logits: [Batch_Size, 1]
        logits = self.fc(out)
        
        # 6. Activation
        # probs: [Batch_Size, 1] -> 范围 0.0 ~ 1.0
        probs = self.sigmoid(logits)
        
        return probs

# -----------------------------------------------------------
# 快速验证代码 (Sanity Check)
# -----------------------------------------------------------
if __name__ == "__main__":
    # 假设有 32 个样本，看过去 15 天，每天 22 个特征
    batch_size = 32
    seq_len = 15
    feature_dim = 22
    
    # 模拟输入数据
    dummy_input = torch.randn(batch_size, seq_len, feature_dim)
    
    # 初始化模型
    model = HybridLSTM(input_dim=feature_dim)
    
    # 前向传播
    output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")   # torch.Size([32, 15, 22])
    print(f"Output Shape: {output.shape}")       # torch.Size([32, 1])
    print("\nModel Architecture constructed successfully.")