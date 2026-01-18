import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class CNN_V3(nn.Module):
    """
    Multilayer CNN with 1D convolutions that can expose either logits or pooled representations.
    """
    def __init__(
        self,
        in_channels: Optional[int] = None,
        L_in: Optional[int] = None,
        output_size: int = 1,
        depth: int = 2,
        filter_size: int = 3,
        n_filters: int = 64,
        n_neurons: int = 64,
        dropout: float = 0.2,
        activation: str = 'relu',
    ):
        super().__init__()
        self.depth = depth
        self.filter_size = filter_size
        self.n_filters = n_filters
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.output_size = output_size
        self._activation_name = activation.lower()
        if self._activation_name not in {'relu', 'elu'}:
            raise ValueError(f"Unsupported activation: {activation}")

        padding = int(np.floor(filter_size / 2))
        conv_cls = nn.Conv1d if in_channels is not None else nn.LazyConv1d

        self.conv1 = conv_cls(n_filters, filter_size, padding=padding)
        self.pool1 = nn.MaxPool1d(2, 2)

        if depth >= 2:
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool2 = nn.MaxPool1d(2, 2)
        else:
            self.conv2 = None
            self.pool2 = None

        if depth >= 3:
            self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool3 = nn.MaxPool1d(2, 2)
        else:
            self.conv3 = None
            self.pool3 = None

        self._split_mode = False
        self.split_proj: Optional[nn.Module] = None

        if L_in is not None:
            proj_in = self._compute_flatten_dim(L_in)
            self.pre_head = nn.Linear(proj_in, n_neurons)
        else:
            self.pre_head = nn.LazyLinear(n_neurons)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(n_neurons, output_size)

    def _compute_flatten_dim(self, L_in: int) -> int:
        length = L_in
        for _ in range(self.depth):
            length = int(np.floor(length / 2))
            if length < 1:
                length = 1
        return length * self.n_filters

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: tensor (batch_size, L_in, in_channels)
        if x.dim() == 2:
            # treat as (batch, length) with a singleton channel dimension
            x = x.unsqueeze(-1)
        elif x.dim() != 3:
            raise ValueError(f"CNN_V3 expects input with 2 or 3 dims, got shape {tuple(x.shape)}")
        x = x.transpose(1, 2).contiguous()  # swap time and feature axes

        x = self.pool1(self._apply_activation(self.conv1(x)))
        if self.depth >= 2 and self.conv2 is not None and self.pool2 is not None:
            x = self.pool2(self._apply_activation(self.conv2(x)))
        if self.depth >= 3 and self.conv3 is not None and self.pool3 is not None:
            x = self.pool3(self._apply_activation(self.conv3(x)))

        x = x.reshape(x.size(0), -1)
        x = self.pre_head(x)
        x = self.dropout_layer(x)
        x = self._apply_activation(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._encode(x)
        if self._split_mode:
            if self.split_proj is None:
                raise RuntimeError("CNN_V3 split mode enabled but split_proj not initialized.")
            return self.split_proj(features)
        return self.fc(features)

    def _apply_activation(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._activation_name == 'relu':
            return F.relu(tensor)
        return F.elu(tensor)

    def enable_split(self, feature_dim: int) -> bool:
        self._split_mode = True
        if feature_dim == self.n_neurons:
            self.split_proj = nn.Identity()
        else:
            self.split_proj = nn.Linear(self.n_neurons, feature_dim)
        return True

class RNN_V2(nn.Module):
    """
    Multi-layer LSTM network
    """
    def __init__(
        self, 
        input_size: Optional[int] = None,
        input_length: Optional[int] = None,
        output_size: int = 1,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        n_neurons=64,
        activation='relu',
    ):
        super().__init__()
        activation = activation.lower()
        if activation not in {'relu', 'elu'}:
            raise ValueError(f"Unsupported activation: {activation}")
        self._activation_name = activation

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self._input_size = input_size
        self._dropout = float(dropout)

        if input_size is not None:
            lstm_dropout = self._dropout if self.num_layers > 1 else 0.0
            self.lstm = nn.LSTM(int(input_size), int(hidden_size), int(num_layers),
                                batch_first=True, dropout=lstm_dropout)
        else:
            self.lstm = None

        self.projection = nn.Linear(hidden_size, n_neurons)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(n_neurons, output_size)
        self._split_mode = False
        self.split_proj: Optional[nn.Module] = None
        self._encoder_dim = n_neurons
    
    def forward(self, x):
        # x: tensor (batch_size, T, input_size)
        # h_all: (batch_size, T, hidden_size)
        seq_lens = None
        if isinstance(x, (list, tuple)):
            x, seq_lens = x
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() != 3:
            raise ValueError(f"RNN_V2 expects input with 2 or 3 dims, got shape {tuple(x.shape)}")

        if self.lstm is None:
            input_size = int(x.size(-1))
            lstm_dropout = self._dropout if self.num_layers > 1 else 0.0
            self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers,
                                batch_first=True, dropout=lstm_dropout)
            self.lstm.to(x.device)

        h_0, c_0 = self.init_hidden(x)
        h_all, (h_T, c_T) = self.lstm(x, (h_0, c_0))
        if seq_lens is not None:
            # gather the last valid hidden state per sequence
            seq_lens = torch.as_tensor(seq_lens, device=h_all.device, dtype=torch.long)
            indices = (seq_lens - 1).clamp(min=0)
            output = h_all[torch.arange(h_all.size(0), device=h_all.device), indices]
        else:
            output = h_T[-1]
        features = self.projection(output)
        features = self.dropout_layer(features)
        features = self._apply_activation(features)
        if self._split_mode:
            if self.split_proj is None:
                raise RuntimeError("RNN_V2 split mode enabled but split_proj not initialized.")
            return self.split_proj(features)
        return self.fc(features)
    
    def init_hidden(self, x):
        batch_size = x.size(0)
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device))

    def _apply_activation(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._activation_name == 'relu':
            return F.relu(tensor)
        return F.elu(tensor)

    def enable_split(self, feature_dim: int) -> bool:
        self._split_mode = True
        if feature_dim == self._encoder_dim:
            self.split_proj = nn.Identity()
        else:
            self.split_proj = nn.Linear(self._encoder_dim, feature_dim)
        return True


class BasicBlock1D(nn.Module):
    """
    A single ResNet basic block for 1D data
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        use_bn=True,
        use_do=False,
        dropout=0.5,
    ):
        super().__init__()
        self.use_bn = use_bn
        self.use_do = use_do
        self.stride = stride
        self.out_channels = out_channels

        # first convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout) if use_do else nn.Identity()

        # second convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.drop2 = nn.Dropout(dropout) if use_do else nn.Identity()

        # shortcut when dimensions change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride),
                nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu(out)
        out = self.drop2(out)
        return out
    


class ResNet1D(nn.Module):
    """
    Simplified ResNet1D adapted from Shenda Hong (2019).
    """
    def __init__(
        self,
        in_channels: Optional[int] = None,
        base_filters=64,
        n_blocks=4,
        downsample_gap=2,
        increasefilter_gap=4,
        n_classes=1,
        use_bn=True,
        use_do=False,
        dropout=0.5,
    ):
        super().__init__()
        self.use_do = use_do

        # initial conv
        if in_channels is None:
            conv1 = nn.LazyConv1d(base_filters, kernel_size=3, padding=1)
        else:
            conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=3, padding=1)

        self.stem = nn.Sequential(
            conv1,
            nn.BatchNorm1d(base_filters) if use_bn else nn.Identity(),
            nn.ReLU()
        )
        # residual blocks
        blocks = []
        in_ch = base_filters
        out_ch = base_filters
        for i in range(n_blocks):
            stride = 2 if (i % downsample_gap == downsample_gap - 1) else 1
            # increase filters at specified gap
            if i > 0 and i % increasefilter_gap == 0:
                out_ch *= 2
            blocks.append(
                BasicBlock1D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    stride=stride,
                    use_bn=use_bn,
                    use_do=use_do,
                    dropout=dropout
                )
            )
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # final head
        self.final_bn = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.final_relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_ch, n_classes)
        self._split_mode = False
        self.split_proj: Optional[nn.Module] = None
        self._encoder_dim = out_ch

    def forward(self, x):
        # x: (batch, L, channels) -> (batch, channels, L)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() != 3:
            raise ValueError(f"ResNet1D expects input with 2 or 3 dims, got shape {tuple(x.shape)}")
        # x = x.transpose(1, 2)
        x = x.transpose(1, 2).contiguous()
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.global_pool(x).squeeze(-1)
        if self._split_mode:
            if self.split_proj is None:
                raise RuntimeError("ResNet1D split mode enabled but split_proj not initialized.")
            return self.split_proj(x)
        return self.fc(x)

    def enable_split(self, feature_dim: int) -> bool:
        self._split_mode = True
        if feature_dim == self._encoder_dim:
            self.split_proj = nn.Identity()
        else:
            self.split_proj = nn.Linear(self._encoder_dim, feature_dim)
        return True

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        # SAME padding: preserve sequence length
        pad = dilation * (kernel_size - 1) // 2
        if in_channels is None:
            self.conv = nn.LazyConv1d(
                out_channels,
                kernel_size,
                padding=pad,
                dilation=dilation
            )
        else:
            self.conv = nn.Conv1d(
                in_channels, out_channels,
                kernel_size,
                padding=pad,
                dilation=dilation
            )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        if in_channels is None:
            self.downsample = nn.LazyConv1d(out_channels, 1)
        elif in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        res = self.downsample(x)
        return out + res


class TCN(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes=1,
        n_filters=64,
        kernel_size=3,
        num_levels=3,
        dropout=0.2
    ):
        super().__init__()
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(
                TCNBlock(
                    in_channels if i == 0 else n_filters,
                    n_filters,
                    kernel_size,
                    dilation,
                    dropout
                )
            )
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_filters, n_classes)
        self._split_mode = False
        self.split_proj: Optional[nn.Module] = None
        self._encoder_dim = n_filters

    def forward(self, x):
        # x: (batch, seq_len, channels) → (batch, channels, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() != 3:
            raise ValueError(f"TCN expects input with 2 or 3 dims, got shape {tuple(x.shape)}")
        x = x.transpose(1, 2).contiguous()
        x = self.network(x)
        x = self.global_pool(x).squeeze(-1)
        if self._split_mode:
            if self.split_proj is None:
                raise RuntimeError("TCN split mode enabled but split_proj not initialized.")
            return self.split_proj(x)
        return self.fc(x)

    def enable_split(self, feature_dim: int) -> bool:
        self._split_mode = True
        if feature_dim == self._encoder_dim:
            self.split_proj = nn.Identity()
        else:
            self.split_proj = nn.Linear(self._encoder_dim, feature_dim)
        return True
