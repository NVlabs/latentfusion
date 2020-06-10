import torch
from torch import nn

from latentfusion.modules import EqualizedConv3d


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, bias=True,
                 conv_module=EqualizedConv3d):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        in_channels: int
            Number of channels of input tensor.
        hidden_channels: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = conv_module(in_channels=self.in_channels + self.hidden_channels,
                                out_channels=4 * self.hidden_channels,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
