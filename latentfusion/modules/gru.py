import torch
from torch import nn

from latentfusion.modules import EqualizedConv3d


class ConvGRUCell(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, bias=True,
                 conv_module=EqualizedConv3d):
        super().__init__()

        self.input_dim = in_channels
        self.hidden_dim = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.update_gate = conv_module(in_channels=self.input_dim + self.hidden_dim,
                                       out_channels=self.hidden_dim,
                                       kernel_size=self.kernel_size,
                                       padding=self.padding,
                                       bias=self.bias)
        self.reset_gate = conv_module(in_channels=self.input_dim + self.hidden_dim,
                                      out_channels=self.hidden_dim,
                                      kernel_size=self.kernel_size,
                                      padding=self.padding,
                                      bias=self.bias)

        self.out_gate = conv_module(in_channels=self.input_dim + self.hidden_dim,
                                    out_channels=self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

    def forward(self, x, h_cur):
        x_in = torch.cat([x, h_cur], dim=1)
        update = torch.sigmoid(self.update_gate(x_in))
        reset = torch.sigmoid(self.reset_gate(x_in))
        x_out = self.out_gate(torch.cat([x, h_cur * reset], dim=1))
        h_new = h_cur * (1 - update) + x_out * update

        return h_new

    def init_hidden(self, b, h, w):
        return torch.zeros(b, self.hidden_dim, h, w).cuda()
