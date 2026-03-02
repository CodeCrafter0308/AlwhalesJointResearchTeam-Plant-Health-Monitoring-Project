import torch
import torch.nn as nn
import torch.nn.functional as F

from model_sensor import SensorEncoder
from model_molecule import MoleculeEncoder


class CrossModalNetwork(nn.Module):
    """
    两阶段推理逻辑：
    1) 先做 stress 识别：sensor 表征 h_shared 与 molecule 表征 z_mol 做 cross-attn 得到 c_stress，再输出 logits_stress
    2) 再做 period 识别：使用 stress 的输出作为条件（训练用真值 one-hot，推理用 softmax 概率）去调制 period head
    """
    def __init__(self, d_sensor=48, d_mol=128, n_stress=3, n_period=5):
        super().__init__()
        self.n_stress = n_stress
        self.n_period = n_period

        self.sensor_net = SensorEncoder(d=d_sensor)
        self.mol_net = MoleculeEncoder(hidden_d=d_mol)

        self.mol_proj = nn.Linear(d_mol, d_sensor)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_sensor, num_heads=4, batch_first=True)

        # ===== Stage-1: stress head =====
        self.stress_head = nn.Sequential(
            nn.Linear(d_sensor * 2, d_sensor),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_sensor, n_stress),
        )

        # ===== Stage-2: period head (conditioned on stress) =====
        # 把 stress 的 one-hot/prob 映射到 d 维条件向量
        self.stress_cond_proj = nn.Sequential(
            nn.Linear(n_stress, d_sensor),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # period 融合：h_shared + c_stress + stress_cond_emb
        self.period_fuse = nn.Sequential(
            nn.Linear(d_sensor * 3, d_sensor),
            nn.GELU(),
            nn.LayerNorm(d_sensor),
            nn.Dropout(0.2),
        )
        self.head_period_cls = nn.Linear(d_sensor, n_period)
        self.head_period_reg = nn.Linear(d_sensor, 1)

    @staticmethod
    def _one_hot(y: torch.Tensor, n: int) -> torch.Tensor:
        return F.one_hot(y.long(), num_classes=n).float()

    def forward(self, x_sensor, g_batch, fp_batch, y_stress=None):
        """
        Args:
            x_sensor: (B, 8, T)
            g_batch, fp_batch: molecule knowledge bank tensors
            y_stress: (B,) 训练阶段可传入 stress 真值，用于 teacher forcing 让 period 真正实现“先 stress 后 period”
        """
        # ---- sensor encoding ----
        h_shared, _sensor_tokens = self.sensor_net(x_sensor)  # (B, d)

        # ---- molecule encoding ----
        z_mol = self.mol_net(g_batch, fp_batch)              # (N_voc, d_mol)
        z_mol = self.mol_proj(z_mol)                         # (N_voc, d)

        # ---- cross-attn: query is sensor, key/value are molecule bank ----
        h_query = h_shared.unsqueeze(1)                      # (B, 1, d)
        z_kv = z_mol.unsqueeze(0).repeat(h_query.size(0), 1, 1)  # (B, N_voc, d)
        c_stress, attn_w = self.cross_attn(query=h_query, key=z_kv, value=z_kv)  # (B, 1, d)

        # ===== Stage-1: stress =====
        h_stress = torch.cat([h_shared, c_stress.squeeze(1)], dim=-1)
        logits_stress = self.stress_head(h_stress)

        # ===== Stage-2: period conditioned on stress =====
        if y_stress is not None:
            stress_cond = self._one_hot(y_stress, self.n_stress)   # teacher forcing
        else:
            stress_cond = torch.softmax(logits_stress, dim=-1)     # inference

        stress_cond_emb = self.stress_cond_proj(stress_cond)       # (B, d)
        h_period_in = torch.cat([h_shared, c_stress.squeeze(1), stress_cond_emb], dim=-1)
        h_period = self.period_fuse(h_period_in)

        logits_period_cls = self.head_period_cls(h_period)
        period_reg = self.head_period_reg(h_period).squeeze(-1)

        return {
            "logits_stress": logits_stress,
            "logits_period_cls": logits_period_cls,
            "period_reg": period_reg,
            "attn_voc": attn_w.squeeze(1),
            "stress_cond_used": stress_cond,  # 便于调试：看 period 用到的条件是 one-hot 还是真实预测概率
        }

    def loss(self, out, y_s, y_p, voc_mask_true,
             w_stress=1.0, w_period_cls=1.0, w_period_reg=1.5, w_guidance=2.0,
             label_smoothing=0.1):
        """
        统一 loss 入口，方便 main_train 直接调用，减少遗漏项导致训练不稳定。
        """
        l_stress = F.cross_entropy(out["logits_stress"], y_s, label_smoothing=label_smoothing)
        l_period_cls = F.cross_entropy(out["logits_period_cls"], y_p, label_smoothing=label_smoothing)
        l_period_reg = F.smooth_l1_loss(out["period_reg"], y_p.float())

        target_mask = voc_mask_true[y_s]  # (B, N_voc)
        l_guidance = F.binary_cross_entropy(out["attn_voc"], target_mask)

        return w_stress * l_stress + w_period_cls * l_period_cls + w_period_reg * l_period_reg + w_guidance * l_guidance
