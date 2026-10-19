import torch.nn as nn
import torch


class CNNEncoder(nn.Module):
    def __init__(
        self,
        h,
        w,
        c,
        c1,
        c2,
        hidden_size,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        super().__init__()
        self.params = {
            "h": h,
            "w": w,
            "c": c,
            "c1": c1,
            "c2": c2,
            "hidden_size": hidden_size,
        }
        self.conv1 = nn.Conv2d(
            c,
            c1,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.conv2 = nn.Conv2d(
            c1,
            c2,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.conv_net = nn.Sequential(
            self.conv1,
            nn.LeakyReLU(),
            self.conv2,
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.hidden_linear = nn.Linear(c2, hidden_size)

    def forward(self, x):
        conved_x = self.conv_net(x)
        conved_x = torch.squeeze(conved_x)
        y = self.hidden_linear(conved_x)
        return y
