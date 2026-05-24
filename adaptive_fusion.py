import torch
import torch.nn as nn


class AdaptiveGatedFusion(nn.Module):
    def __init__(self, dim: int = 128):
        super(AdaptiveGatedFusion, self).__init__()
        gate_in_dim = dim * 2 + 2
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate_net[2].bias, -0.7)

    def forward(self, h_m, h_d, pre_MHM, pre_HHN):
        gate_input = torch.cat(
            [h_m, h_d, pre_MHM.unsqueeze(1), pre_HHN.unsqueeze(1)],
            dim=1
        )
        gate = self.gate_net(gate_input).squeeze(1)
        pre_asso = gate * pre_MHM + (1.0 - gate) * pre_HHN
        uncertainty = torch.abs(pre_MHM - pre_HHN)
        confidence = 1.0 - uncertainty
        pre_final = confidence * pre_asso + (1.0 - confidence) * 0.5
        return pre_final, uncertainty

    def get_gate_weights(self, h_m, h_d, pre_MHM, pre_HHN):
        gate_input = torch.cat(
            [h_m, h_d, pre_MHM.unsqueeze(1), pre_HHN.unsqueeze(1)],
            dim=1
        )
        return self.gate_net(gate_input).squeeze(1)