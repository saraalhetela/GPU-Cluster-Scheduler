"""model.py -- Dueling MLP Q-network for the GPU cluster scheduler.

GPUClusterEnv's state is a flat 9-dim snapshot (hour sin/cos, GPU price,
job progress, deadline remaining, GPU-hours remaining, cluster utilization,
urgency ratio, priority rank), so a plain MLP feature extractor is used,
with separate value and advantage heads combined the standard dueling way.
"""
import torch
import torch.nn as nn


class DuelingMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_fc = nn.Linear(hidden_size, 1)
        self.adv_fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        f   = self.feature(x)
        val = self.value_fc(f)
        adv = self.adv_fc(f)
        return val + adv - adv.mean(dim=1, keepdim=True)
