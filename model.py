"""
model.py -- GPU cluster scheduler version.

Replaces the stock-trading DuelingConv1D. That network convolved over a
(5, 50) OHLCV window -- meaningless here, since GPUClusterEnv's state is a
flat 7-dim snapshot (hour_sin, hour_cos, gpu_price, job_progress,
deadline_remaining, gpu_hours_remaining, cluster_utilization), not a
time-series window. Same situation as the EV project's DuelingConv1D ->
DuelingMLP swap -- dropped the conv layers entirely, kept the dueling
value/advantage head split.
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