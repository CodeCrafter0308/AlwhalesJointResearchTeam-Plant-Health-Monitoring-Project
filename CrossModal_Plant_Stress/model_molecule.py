import torch
import torch.nn as nn


def global_mean_pool(h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    out = torch.zeros((num_graphs, h.size(1)), device=h.device, dtype=h.dtype)
    out.index_add_(0, batch, h)
    counts = torch.zeros((num_graphs,), device=h.device, dtype=h.dtype)
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
    return out / counts.clamp_min(1.0).unsqueeze(1)


class GraphConvEdge(nn.Module):
    def __init__(self, d, e_d):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(d + e_d, d), nn.ReLU(), nn.Linear(d, d))
        self.upd = nn.Sequential(nn.Linear(d + d, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, ei, ea):
        m = self.msg(torch.cat([h[ei[0]], ea], dim=1))
        agg = torch.zeros_like(h).index_add_(0, ei[1], m)
        return self.norm(h + self.upd(torch.cat([h, agg], dim=1)))


class MoleculeEncoder(nn.Module):
    def __init__(self, node_d=19, edge_d=15, fp_d=2215, hidden_d=128):
        super().__init__()
        self.node_proj = nn.Linear(node_d, hidden_d)
        self.g_layers = nn.ModuleList([GraphConvEdge(hidden_d, edge_d) for _ in range(3)])
        self.g_readout = nn.Sequential(nn.Linear(hidden_d, hidden_d), nn.ReLU(), nn.Linear(hidden_d, hidden_d))
        self.fp_net = nn.Sequential(nn.Linear(fp_d, 512), nn.ReLU(), nn.Linear(512, hidden_d), nn.ReLU(),
                                    nn.Linear(hidden_d, hidden_d))
        self.fuse_gate = nn.Sequential(nn.Linear(hidden_d * 2, hidden_d), nn.ReLU(), nn.Linear(hidden_d, hidden_d),
                                       nn.Sigmoid())
        self.fuse_out = nn.Sequential(nn.Linear(hidden_d, hidden_d), nn.ReLU())

    def forward(self, g_batch, fp_batch):
        h = self.node_proj(g_batch["x"])
        for layer in self.g_layers:
            h = layer(h, g_batch["edge_index"], g_batch["edge_attr"])
        z_g = self.g_readout(global_mean_pool(h, g_batch["batch"]))
        z_fp = self.fp_net(fp_batch)
        gate = self.fuse_gate(torch.cat([z_g, z_fp], dim=1))
        return self.fuse_out(gate * z_g + (1.0 - gate) * z_fp)