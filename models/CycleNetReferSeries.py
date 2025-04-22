import torch
import torch.nn as nn
from math import sin, pi

from args import Args


class RecurrentCycle(torch.nn.Module):

    def __init__(self, seq_len, dn_layers):
        super().__init__()
        bps = [(24, 1), (168, 1), (24, 2), (24, 3), (24, 4), (168, 2)]
        dn_samples = [DnSample(in_len=seq_len // 2**i) for i in range(dn_layers)]
        self.dn_samples = nn.Sequential(*dn_samples)
        self.N = seq_len // 2 ** (dn_layers + 1)
        with torch.no_grad():
            self.rs = torch.tensor(
                [
                    [sin(t * 2 * pi * bp[1] / bp[0]) for t in range(seq_len)]
                    for bp in bps[: self.N]
                ]
            ).T

    def forward(self, x):
        a = self.dn_samples(x)
        ax = a[:, : self.N, :]
        ay = a[:, self.N :, :]
        self.rs = self.rs.to(a.device)
        cycle_x = self.rs @ ax
        cycle_y = self.rs @ ay
        return cycle_x, cycle_y


class DnSample(nn.Module):

    def __init__(self, in_len):
        super().__init__()
        self.in_len = in_len
        self.avg = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        assert in_len % 2 == 0, "The input length must be even."
        self.proj = nn.Linear(in_len, in_len // 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.avg(x) + self.proj(x)
        x = x.permute(0, 2, 1)
        return x


class Model(nn.Module):
    def __init__(self, configs: Args):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin

        self.cycleQueue = RecurrentCycle(configs.seq_len, configs.dn_layers)

        assert self.model_type in ["linear", "mlp"]
        if self.model_type == "linear":
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == "mlp":
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len),
            )

    def forward(self, x, cycle_index):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # remove the cycle of the input data
        cycle_x, cycle_y = self.cycleQueue(x)

        x = x - cycle_x

        # forecasting with channel independence (parameters-sharing)
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # add back the cycle of the output data
        y = y + cycle_y

        # instance denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y
