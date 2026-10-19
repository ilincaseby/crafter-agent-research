import torch.nn as nn
import torch
from cnn_encoder import CNNEncoder


class RNNDeepQNet(nn.Module):
    def __init__(
        self,
        h,
        w,
        c,
        c1,
        c2,
        hidden_size,
        kernel_size,
        hs1,
        hs2,
        num_actions,
        seq_length,
        device,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cnn_encoder = CNNEncoder(
            h, w, c, c1, c2, hidden_size, kernel_size, stride, padding, dilation, groups
        )
        self.hh = nn.Sequential(nn.Linear(hidden_size, hidden_size))
        self.dqn = nn.Sequential(
            nn.Linear(hidden_size, hs1),
            nn.ReLU(),
            nn.Linear(hs1, hs2),
            nn.ReLU(),
            nn.Linear(hs2, num_actions),
            nn.Softmax(),
        )
        self.tanh = nn.Tanh()
        self.device = device
        self.seq_length = seq_length

    def forward(self, x):
        dimensions = len(x.shape)

        if dimensions == 5:
            hidden_info = torch.zeros(x.shape[0], self.hidden_size).to(self.device)
            for i in range(self.seq_length):
                hidden_info == self.tanh(
                    self.hh(hidden_info) + self.cnn_encoder(x[:, i, :, :, :])
                )
        else:
            hidden_info = torch.zeros(self.hidden_size).to(self.device)
            for i in range(self.seq_length):
                hidden_info = self.tanh(self.hh(hidden_info) + self.cnn_encoder(x[i]))

        return self.dqn(hidden_info)
