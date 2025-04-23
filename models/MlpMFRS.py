from math import sqrt

import torch
import torch.nn as nn

from args import Args


class Model(nn.Module):
    def __init__(self, configs: Args):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.use_norm = configs.use_revin

        self.ref_in = len(configs.period_list)
        self.pool_layers = configs.dn_layers

        # self.moving_avg = moving_avg(kernel_size=31, stride=1)
        # self.coefficient = nn.Parameter(torch.rand(len(configs.period_list) * 2, configs.enc_in), requires_grad=True)
        self.dc = nn.Parameter(
            torch.ones(self.seq_len + self.pred_len, configs.enc_in), requires_grad=True
        )
        # self.pos = nn.Parameter(torch.ones(self.seq_len, configs.enc_in), requires_grad=True)

        self.embedding = nn.Linear(self.seq_len, self.seq_len + self.pred_len)

        self.coefficient_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (2 ** i),
                        configs.seq_len // (2 ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (2 ** (i + 1)),
                        configs.seq_len // (2 ** (i + 1)),
                    ),
                )
                for i in range(self.pool_layers)
            ]
        )

        # self.coefficient_layers = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Linear(
        #                 configs.seq_len // (2 ** i),
        #                 configs.seq_len // (2 ** (i - 1)),
        #             ),
        #             nn.GELU(),
        #             nn.Linear(
        #                 configs.seq_len // (2 ** (i - 1)),
        #                 configs.seq_len // (2 ** (i - 1)),
        #             ),

        #         )
        #         for i in range(self.pool_layers, 0, -1)
        #     ]
        # )

        self.coefficient_projection = nn.Linear(
            self.seq_len // (2 ** self.pool_layers), self.ref_in * 2
        )
        # self.dc_projection = nn.Linear(self.seq_len, self.seq_len + self.pred_len)

        self.model = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.pred_len),
        )

    def forward(self, x, x_refer, x_dec, y_refer):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)

        ###############################  instance norm  ############################
        if self.use_norm:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            # seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            # x = (x - seq_mean) / torch.sqrt(seq_var)
            x = x - seq_mean

            # means_xr = x_refer.mean(1, keepdim=True).detach()
            # x_refer = x_refer - means_xr
            # stdev_xr = torch.sqrt(torch.var(x_refer, dim=1, keepdim=True, unbiased=False) + 1e-5)
            # x_refer /= stdev_xr

        ###############################    featuere    #############################
        x_list = self.pooling(x.permute(0, 2, 1))
        feature = x_list[0]

        for i in range(self.pool_layers):
            feature = x_list[i + 1] + self.coefficient_layers[i](feature)

        ###############################   coefficient   #############################
        # B, _, C = x.shape
        scale = 1.0 / sqrt(self.ref_in)

        coeff = self.coefficient_projection(feature)  # (B, C, N * 2)
        # coeff = coeff.reshape(B, C, self.seq_len + self.pred_len, self.ref_in)   # (B, C, S + T, N)
        coeff = torch.softmax(scale * coeff, dim=2).permute(0, 2, 1)  # (B, N * 2, C)
        # coeff = coeff.permute(0, 2, 1)   # (B, N, C)

        x_coeff = coeff[:, : self.ref_in, :]  # (B, N, C)
        y_coeff = coeff[:, self.ref_in:, :]  # (B, N, C)

        ###############################       dc       #############################
        # res = self.moving_avg(x.permute(0, 2, 1))
        # dc = self.dc_projection(res).permute(0, 2, 1)   # (B, S + T, C)

        # x_dc = dc[:, :self.seq_len, :]   # (B, S, C)
        # y_dc = dc[:, self.seq_len:, :]   # (B, T, C)

        x_dc = self.dc[: self.seq_len, :]  # (B, S, C)
        y_dc = self.dc[self.seq_len:, :]  # (B, T, C)

        ###############################     refer      ############################
        refer = self.embedding(x_refer.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # (B, S + T, N)

        x_refer = refer[:, : self.seq_len, :]  # (B, S, N)
        y_refer = refer[:, self.seq_len:, :]  # (B, T, N)

        ###############################     cycle      ############################
        # x_cycle = torch.bmm(x_refer, x_coeff) + x_dc   # (B, S, C)
        # y_cycle = torch.bmm(y_refer, y_coeff) + y_dc   # (B, T, C)
        x_cycle = torch.bmm(x_refer, x_coeff)  # (B, S, C)
        y_cycle = torch.bmm(y_refer, y_coeff)  # (B, T, C)

        ###############################     model      ############################
        # remove the cycle of the input data
        x = x - x_cycle

        # # forecasting with channel independence (parameters-sharing)
        x = x.permute(0, 2, 1)
        y = self.model(x)
        y = y.permute(0, 2, 1)

        # add back the cycle of the output data
        y = y + y_cycle

        ############################  instance denorm  ###########################
        if self.use_norm:
            # y = y * torch.sqrt(seq_var) + seq_mean
            y = y + seq_mean

        return y, y

    def pooling(self, x):
        # x: (B, C, S)

        down_pool = torch.nn.AvgPool1d(kernel_size=2, stride=2)

        x_ori = x
        x_list = [x]

        for i in range(self.pool_layers):
            x_sampling = down_pool(x_ori)

            x_list.append(x_sampling)
            x_ori = x_sampling

        return x_list


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
