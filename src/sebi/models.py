import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(
        self,
        h,
        w,
        c,
        c1=32,
        c2=64,
        c3=128,
        c4=256,
        hidden_size=1024,
        kernel_size=4,
        stride=2,
        padding=1,
        dilation=1,
        groups=1,
    ):
        super().__init__()
        h4o, w4o = h, w
        for _ in range(4):
            h4o = (h4o + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            w4o = (w4o + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        if min(h4o, w4o) < 1:
            raise ValueError("Image dimensions are too small for the four convolutions")
        self.net = nn.Sequential(
            nn.Conv2d(
                c,
                c1,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ),
            nn.ELU(),
            nn.Conv2d(
                c1,
                c2,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ),
            nn.ELU(),
            nn.Conv2d(
                c2,
                c3,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ),
            nn.ELU(),
            nn.Conv2d(
                c3,
                c4,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(c4 * h4o * w4o, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class LatentStateEncoder(nn.Module):
    def __init__(
        self,
        h,
        w,
        c,
        num_layers=2,
        hidden_size_lstm=1024,
        c1=32,
        c2=64,
        c3=128,
        c4=256,
        hidden_size_cnn=1024,
        kernel_size=4,
        stride=2,
        padding=1,
        dilation=1,
        groups=1,
    ):
        super().__init__()
        self.input_size_lstm = hidden_size_cnn
        self.visual_encoder = CNNEncoder(
            h,
            w,
            c,
            c1,
            c2,
            c3,
            c4,
            hidden_size_cnn,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        )
        self.lstm = nn.LSTM(
            hidden_size_cnn, hidden_size_lstm, num_layers=num_layers, batch_first=True
        )

    def forward(self, x):
        if len(x.shape) != 5:
            raise ValueError(
                "Expected observations shaped [batch, sequence, channels, height, width]"
            )
        batch, seq_length, channels, h, w = x.shape
        x = x.reshape(batch * seq_length, channels, h, w)
        encoded_states = self.visual_encoder(x)
        encoded_states = encoded_states.reshape(batch, seq_length, self.input_size_lstm)
        return self.lstm(encoded_states)


class ActorCritic(nn.Module):
    def __init__(
        self, latent_space_encoder_model, num_layers, hidden_size, num_actions
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lse_model = latent_space_encoder_model
        self.actor_net = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_actions),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        if len(x.shape) != 5:
            raise ValueError(
                "Expected observations shaped [batch, sequence, channels, height, width]"
            )
        batch_size, seq_length, channel_size, h, w = x.shape
        # lstm_output shape ===> batch_size, seq_length, hidden_size
        lstm_output, h_c_n = self.lse_model(x)
        # h_n shape ===> num_layers, batch_size, hidden_size
        # c_n shape ===> num_layers, batch_size, hidden_size
        h_n, c_n = h_c_n
        last_output = lstm_output[:, -1, :]
        return (
            self.actor_net(last_output),
            self.critic_net(last_output),
            last_output,
            lstm_output,
            h_n,
            c_n,
        )
