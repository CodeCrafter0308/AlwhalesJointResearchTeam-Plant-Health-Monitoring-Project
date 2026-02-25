import torch
import torch.nn as nn


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, max(8, d // 2)), nn.GELU(),
                                   nn.Linear(max(8, d // 2), 1))

    def forward(self, x):
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return (w.unsqueeze(-1) * x).sum(dim=1), w


class MultiScaleStem(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        c1, c2 = d // 3, d // 3;
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
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(0.1), nn.Linear(2 * d, d), nn.Dropout(0.1))

    def forward(self, x):
        y, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        return x + self.ff(self.ln2(x + y))


class SensorEncoder(nn.Module):
    def __init__(self, d=48):
        super().__init__()
        # 回归到极速单分支架构
        self.se = nn.Sequential(nn.Linear(8, 4), nn.GELU(), nn.Linear(4, 8))
        self.stem = MultiScaleStem(d)
        self.bilstm = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True)
        self.tpool = AttnPool(d)
        self.sensor_self = SelfAttnBlock(d, 4)

        self.fuse_shared = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))

        # Period 预测变为：分类头 (5类) + 回归头 (连续值)
        self.head_period_cls = nn.Linear(d, 5)
        self.head_period_reg = nn.Linear(d, 1)

    def forward(self, x):
        B, C, T = x.shape
        w_se = torch.softmax(self.se(x.mean(dim=-1)), dim=-1)
        x_se = x * w_se.unsqueeze(-1)

        z = self.stem(x_se.reshape(B * C, 1, T)).transpose(1, 2)
        z, _ = self.bilstm(z)
        h_each, _ = self.tpool(z)

        s = h_each.view(B, C, -1)
        s = self.sensor_self(s)
        h_shared = self.fuse_shared(s.mean(dim=1))

        # 输出：共享特征, 分类Logits, 回归连续值
        return h_shared, self.head_period_cls(h_shared), self.head_period_reg(h_shared).squeeze(-1)