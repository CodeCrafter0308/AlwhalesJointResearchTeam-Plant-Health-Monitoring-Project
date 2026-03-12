import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_sensor import SensorEncoder
from model_molecule import MoleculeEncoder
from model_dft import DFTEncoder


class CrossModalNetwork(nn.Module):
    def __init__(self, d_sensor=48, d_mol=128, d_dft=6):
        super().__init__()
        # 为了兼容之前的修改，这里确保 in_channels=8
        self.sensor_net = SensorEncoder(d=d_sensor, in_channels=8)
        self.mol_net = MoleculeEncoder(hidden_d=d_mol)
        self.dft_net = DFTEncoder(d_dft=d_dft, d_sensor=d_sensor)

        self.mol_proj = nn.Linear(d_mol, d_sensor)
        # 将 [mol_token, dft_token] 融合为统一的 VOC token
        self.voc_token_fuse = nn.Sequential(
            nn.Linear(d_sensor * 2, d_sensor),
            nn.LayerNorm(d_sensor),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_sensor, num_heads=4, batch_first=True)

        feature_dim = d_sensor * 2

        # 【阶段一】：Stress 模式识别头 (输入跨模态融合特征)
        self.fusion_head = nn.Sequential(
            nn.Linear(feature_dim, d_sensor),
            nn.LayerNorm(d_sensor),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_sensor, 3)  # 3种胁迫: Drought, Salt, Heat
        )

        # 【核心新增】：为 Period 分支开辟独立的特征投影层
        # 让 Period 分支直接使用未经化学注意力污染的、包含丰富时序演变的纯传感器特征
        self.period_base_proj = nn.Sequential(
            nn.Linear(d_sensor, d_sensor),
            nn.LayerNorm(d_sensor),
            nn.GELU()
        )

        # 【优化】：FiLM 调制发生器
        # 接收 Stress 概率，生成作用于 d_sensor (而不是 feature_dim) 的参数
        self.film_gen = nn.Sequential(
            nn.Linear(3, d_sensor),
            nn.GELU(),
            nn.Linear(d_sensor, d_sensor * 2)
        )

        # 【阶段二】：Period 级联识别头 (输入维度降为 d_sensor)
        self.period_cls_head = nn.Sequential(
            nn.Linear(d_sensor, d_sensor),
            nn.LayerNorm(d_sensor),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_sensor, 5)  # 5个时间段: 0, 3, 6, 9, 12h
        )
        self.period_reg_head = nn.Sequential(
            nn.Linear(d_sensor, d_sensor),
            nn.GELU(),
            nn.Linear(d_sensor, 1)
        )

    def forward(self, x_sensor, g_batch, fp_batch, dft_feat_bank=None):
        # 1. 提取基础传感器特征
        h_shared = self.sensor_net(x_sensor)  # Shape: (B, d_sensor)

        # 2. 跨模态化学知识融合 (专供 Stress 分支使用)
        z_mol = self.mol_net(g_batch, fp_batch)
        z_mol = self.mol_proj(z_mol)  # [N_voc, d_sensor]

        # 2.1 DFT 特征编码 (第三分支)。dft_feat_bank: [N_voc, D_dft]
        if dft_feat_bank is None:
            d_dft = self.dft_net.net[0].in_features
            dft_feat_bank = torch.zeros((z_mol.size(0), d_dft), device=z_mol.device, dtype=z_mol.dtype)
        z_dft = self.dft_net(dft_feat_bank.to(device=z_mol.device, dtype=z_mol.dtype))  # [N_voc, d_sensor]

        # 2.2 将分子指纹 token 与 DFT token 融合为 VOC token
        z_voc = self.voc_token_fuse(torch.cat([z_mol, z_dft], dim=-1))  # [N_voc, d_sensor]

        h_query = h_shared.unsqueeze(1)
        z_kv = z_voc.unsqueeze(0).repeat(h_query.size(0), 1, 1)

        c_stress, attn_w = self.cross_attn(query=h_query, key=z_kv, value=z_kv)

        # Stress 专用特征拼接
        h_stress_fused = torch.cat([h_shared, c_stress.squeeze(1)], dim=-1)

        # 3. 阶段一推断：预测 Stress
        logits_stress = self.fusion_head(h_stress_fused)

        # 4. 阶段二推断：特征解耦后的 Period 预测
        # 提取纯粹的传感器时序基底特征
        h_period_base = self.period_base_proj(h_shared)

        # 计算 Stress 概率并截断梯度，生成 FiLM 参数
        stress_prob = torch.softmax(logits_stress, dim=-1).detach()
        film_params = self.film_gen(stress_prob)
        gamma, beta = film_params.chunk(2, dim=-1)

        # 实施 FiLM 调制：将先验的 Stress 状态注入到 Period 特征中
        h_period_modulated = h_period_base * (1.0 + gamma) + beta

        logits_period_cls = self.period_cls_head(h_period_modulated)
        period_reg = self.period_reg_head(h_period_modulated).squeeze(-1)

        return {
            "logits_stress": logits_stress,
            "logits_period_cls": logits_period_cls,
            "period_reg": period_reg,
            "attn_voc": attn_w.squeeze(1)
        }

    def loss(self, out, y_s, y_p, voc_mask_true):
        device = y_p.device

        # 1. Stress 交叉熵损失 (类别是离散且正交的，保持不变)
        l_stress = F.cross_entropy(out["logits_stress"], y_s, label_smoothing=0.1)

        # 2. 【核心修改】：Period 标签分布学习 (Label Distribution Learning)
        # 为 y_p 生成以真实标签为中心的高斯分布。sigma 控制容错宽度。
        num_classes = 5
        classes = torch.arange(num_classes, device=device).float()
        y_p_float = y_p.float().unsqueeze(1)

        # Sigma 设为 0.7，意味着如果真实是 6h (idx 2)，那 3h(idx 1) 和 9h(idx 3) 也会得到约 0.14 的目标概率，
        # 这教导模型：偏离一点点是可以理解的，从而拉平损失地貌。
        sigma = 0.7
        gaussian_target = torch.exp(-((classes - y_p_float) ** 2) / (2 * sigma ** 2))
        gaussian_target = gaussian_target / gaussian_target.sum(dim=1, keepdim=True)

        # 计算预测的 Log-Softmax 与高斯目标的 KL 散度/交叉熵
        log_preds_period = F.log_softmax(out["logits_period_cls"], dim=-1)
        l_period_cls = torch.mean(torch.sum(-gaussian_target * log_preds_period, dim=-1))

        # 3. Period 回归损失
        l_period_reg = F.smooth_l1_loss(out["period_reg"], y_p.float())

        # 4. Attention Guidance
        target_mask = voc_mask_true[y_s]
        l_guidance = F.binary_cross_entropy(out["attn_voc"], target_mask)

        # 最终组合权重
        return 1.0 * l_stress + 2.0 * l_period_cls + 0.5 * l_period_reg + 2.0 * l_guidance