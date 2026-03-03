import torch
import torch.nn as nn


class DFTEncoder(nn.Module):
    """
    将 DFT 定长特征向量映射到与传感器隐藏维度一致的 token 表示。
    输入:  dft_feat_bank [N_voc, D_dft]
    输出:  dft_tokens    [N_voc, d_sensor]
    """
    def __init__(self, d_dft: int, d_sensor: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_dft, d_sensor),
            nn.LayerNorm(d_sensor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_sensor, d_sensor),
            nn.LayerNorm(d_sensor),
            nn.GELU()
        )

    def forward(self, dft_feat_bank: torch.Tensor) -> torch.Tensor:
        return self.net(dft_feat_bank)
