import torch
import torch.nn as nn
import torch.nn.functional as F
from model_sensor import SensorEncoder
from model_molecule import MoleculeEncoder


class CrossModalNetwork(nn.Module):
    def __init__(self, d_sensor=48, d_mol=128):
        super().__init__()
        self.sensor_net = SensorEncoder(d=d_sensor)
        self.mol_net = MoleculeEncoder(hidden_d=d_mol)

        self.mol_proj = nn.Linear(d_mol, d_sensor)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_sensor, num_heads=4, batch_first=True)

        self.fusion_head = nn.Sequential(
            nn.Linear(d_sensor * 2, d_sensor),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_sensor, 3)
        )

    def forward(self, x_sensor, g_batch, fp_batch):
        # 接收轻量化单分支输出的回归项 period_reg
        h_shared, logits_period_cls, period_reg = self.sensor_net(x_sensor)

        z_mol = self.mol_net(g_batch, fp_batch)
        z_mol = self.mol_proj(z_mol)

        h_query = h_shared.unsqueeze(1)
        z_kv = z_mol.unsqueeze(0).repeat(h_query.size(0), 1, 1)

        c_stress, attn_w = self.cross_attn(query=h_query, key=z_kv, value=z_kv)

        h_final = torch.cat([h_shared, c_stress.squeeze(1)], dim=-1)
        logits_stress = self.fusion_head(h_final)

        return {
            "logits_stress": logits_stress,
            "logits_period_cls": logits_period_cls,
            "period_reg": period_reg,  # 传递回归值
            "attn_voc": attn_w.squeeze(1)
        }

    def loss(self, out, y_s, y_p, voc_mask_true):
        # 标签平滑防止震荡
        l_stress = F.cross_entropy(out["logits_stress"], y_s, label_smoothing=0.1)
        l_period_cls = F.cross_entropy(out["logits_period_cls"], y_p, label_smoothing=0.1)

        # 【致胜关键】：时间连续性回归损失 (Smooth L1 Loss)
        # 把类别 0,1,2,3,4 视为浮点数，强迫网络输出与它靠近。
        l_period_reg = F.smooth_l1_loss(out["period_reg"], y_p.float())

        # 将分类的细致度和回归的时间连续性结合
        l_period = 1.0 * l_period_cls + 1.5 * l_period_reg

        target_mask = voc_mask_true[y_s]
        l_guidance = F.binary_cross_entropy(out["attn_voc"], target_mask)

        return 1.0 * l_stress + l_period + 2.0 * l_guidance