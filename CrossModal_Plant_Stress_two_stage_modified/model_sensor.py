import torch
import torch.nn as nn


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, max(8, d // 2)),
            nn.GELU(),
            nn.Linear(max(8, d // 2), 1),
        )

    def forward(self, x):
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return (w.unsqueeze(-1) * x).sum(dim=1), w


class MultiScaleStem(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        c1, c2 = d // 3, d // 3
        c3 = d - c1 - c2
        self.b1 = nn.Sequential(nn.Conv1d(1, c1, 3, padding=1), nn.GELU())
        self.b2 = nn.Sequential(nn.Conv1d(1, c2, 7, padding=3), nn.GELU())
        self.b3 = nn.Sequential(nn.Conv1d(1, c3, 11, padding=5), nn.GELU())
        self.merge = nn.Sequential(nn.Conv1d(d, d, 1), nn.GELU(), nn.BatchNorm1d(d))

    def forward(self, x):
        return self.merge(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))


class SelfAttnBlock(nn.Module):
    def __init__(self, d=32, heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.1)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d, d),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        y, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        return x + self.ff(self.ln2(x + y))


class SensorEncoder(nn.Module):
    """
    输出只包含传感器共享表征 h_shared（用于 stress -> period 的两阶段推理）以及 sensor_tokens（用于可选的 token 级对齐）。
    说明：
    - h_shared: (B, d)
    - sensor_tokens: (B, C, d) 其中 C=8 个传感器通道
    """
    def __init__(self, d=48):
        super().__init__()
        self.se = nn.Sequential(nn.Linear(8, 4), nn.GELU(), nn.Linear(4, 8))
        self.stem = MultiScaleStem(d)
        self.bilstm = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True)
        self.tpool = AttnPool(d)
        self.sensor_self = SelfAttnBlock(d, 4)

        # 强力 dropout 抑制过拟合与曲线震荡
        self.fuse_shared = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(p=0.5),
        )

    def forward(self, x):
        # x: (B, C=8, T)
        B, C, T = x.shape
        w_se = torch.softmax(self.se(x.mean(dim=-1)), dim=-1)
        x_se = x * w_se.unsqueeze(-1)

        # 每个通道做 1D 时序/频序建模
        z = self.stem(x_se.reshape(B * C, 1, T)).transpose(1, 2)  # (B*C, T, d)
        z, _ = self.bilstm(z)
        h_each, _ = self.tpool(z)  # (B*C, d)

        # 汇聚到 8 个通道 token，再做一次通道间 self-attn
        sensor_tokens = h_each.view(B, C, -1)
        sensor_tokens = self.sensor_self(sensor_tokens)  # (B, C, d)

        h_shared = self.fuse_shared(sensor_tokens.mean(dim=1))  # (B, d)
        return h_shared, sensor_tokens
