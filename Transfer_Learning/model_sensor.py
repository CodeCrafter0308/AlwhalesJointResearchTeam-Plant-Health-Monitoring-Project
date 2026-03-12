import torch
import torch.nn as nn


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, max(8, d // 2)),
            nn.GELU(),
            nn.Linear(max(8, d // 2), 1)
        )

    def forward(self, x):
        # x shape: (B, T, d)
        w = torch.softmax(self.score(x), dim=1)
        return (w * x).sum(dim=1)  # 合并时间维度 -> (B, d)


class MultiScaleStem(nn.Module):
    def __init__(self, in_channels=8, d=48):
        super().__init__()
        c1, c2 = d // 3, d // 3
        c3 = d - c1 - c2

        # 核心改动：直接接收 in_channels (8个通道)，联合提取多尺度跨通道特征
        self.b1 = nn.Sequential(nn.Conv1d(in_channels, c1, 3, padding=1), nn.BatchNorm1d(c1), nn.GELU())
        self.b2 = nn.Sequential(nn.Conv1d(in_channels, c2, 7, padding=3), nn.BatchNorm1d(c2), nn.GELU())
        self.b3 = nn.Sequential(nn.Conv1d(in_channels, c3, 11, padding=5), nn.BatchNorm1d(c3), nn.GELU())
        self.merge = nn.Sequential(nn.Conv1d(d, d, 1), nn.BatchNorm1d(d), nn.GELU())

    def forward(self, x):
        return self.merge(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))


class TemporalBlock(nn.Module):
    """新增：用于捕捉长程时间依赖的 Transformer 编码块"""

    def __init__(self, d=48, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.2)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(2 * d, d), nn.Dropout(0.3)
        )
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        return x + self.ff(self.ln2(x))


class SensorEncoder(nn.Module):
    def __init__(self, d=48, in_channels=8):
        super().__init__()
        # 修复 SE Block：使用 Sigmoid 将权重约束在 (0, 1)
        self.se = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.GELU(),
            nn.Linear(in_channels // 2, in_channels)
        )

        self.stem = MultiScaleStem(in_channels=in_channels, d=d)

        # 增强时序提取：加深双向 LSTM 并串联注意力块
        self.bilstm = nn.LSTM(d, d // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.temporal_attn = TemporalBlock(d=d, heads=4)

        self.tpool = AttnPool(d)

        self.fuse_shared = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        # x shape: (Batch, Channels=8, Time=256)
        B, C, T = x.shape

        # 1. 通道注意力加权
        w_se = torch.sigmoid(self.se(x.mean(dim=-1)))  # (B, 8)
        x_se = x * w_se.unsqueeze(-1)

        # 2. 跨通道联合空间-时间特征提取
        # 这一步将 8 个通道的数据糅合，提取局部的动力学特征
        z = self.stem(x_se)  # (B, d, T)
        z = z.transpose(1, 2)  # 转换为 (B, T, d) 以适应 RNN/Attention

        # 3. 捕捉长程时间段的动态演变 (针对 Period)
        z, _ = self.bilstm(z)
        z = self.temporal_attn(z)

        # 4. 时间维度池化
        h_shared = self.tpool(z)  # (B, d)

        # 5. 特征融合与正则化
        h_shared = self.fuse_shared(h_shared)

        return h_shared