import numpy as np
import torch


def deconv2d_outsize(
    height,
    width,
    kernel_size,
    stride,
    padding,
    dilation=None,
    output_padding=None,
):
    if output_padding is None:
        output_padding = [0, 0]
    if dilation is None:
        dilation = [1, 1]
    h_out = (
        (height - 1) * stride[0]
        - 2 * padding[0]
        + dilation[0] * (kernel_size[0] - 1)
        + output_padding[0]
        + 1
    )
    w_out = (
        (width - 1) * stride[1]
        - 2 * padding[1]
        + dilation[1] * (kernel_size[1] - 1)
        + output_padding[1]
        + 1
    )
    return h_out, w_out


def conv2d_outsize(height, width, kernel_size, stride, padding):
    h_out = ((height + 2 * padding[0] - (kernel_size[0] - 1) - 1) // stride[0]) + 1
    w_out = ((width + 2 * padding[1] - (kernel_size[1] - 1) - 1) // stride[1]) + 1
    return h_out, w_out


class MLP(torch.nn.Module):
    """Two-layer MLP body for low-dimensional (vector) observations.

    Accepts either a 1-D shape tuple ``(in_dim,)`` or an int ``in_dim``
    so it slots into both the ``input_size`` and ``input_shape``
    interfaces used across the agent-model wrappers.
    """

    def __init__(
        self,
        input_shape,
        hidden_size: int = 64,
        activation: type[torch.nn.Module] = torch.nn.Tanh,
    ) -> None:
        super().__init__()
        in_dim = int(input_shape[0]) if hasattr(input_shape, "__len__") else int(input_shape)
        self.dense_size = hidden_size
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_size),
            activation(),
            torch.nn.Linear(hidden_size, hidden_size),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeconvUniverse(torch.nn.Module):
    def __init__(
        self,
        output_size,
        deconv1_size=64,
        deconv2_size=64,
        deconv3_size=64,
        deconv4_size=64,
        padding=None,
        conv_activation=torch.nn.ELU,
        weight_initialiser=torch.nn.init.xavier_uniform_,
        trainable=True,
    ):
        # output_size [channels, height, width] size of output after convolutions
        if padding is None:
            padding = (0, 0)
        super().__init__()
        self.output_size = output_size
        self.dense_size = np.prod(output_size)

        self.h1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                output_size[0],
                deconv1_size,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=padding,
                output_padding=1,
            ),
            conv_activation(),
        )
        self.h2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                deconv1_size,
                deconv2_size,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=padding,
                output_padding=0,
            ),
            conv_activation(),
        )
        self.h3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                deconv2_size,
                deconv3_size,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=padding,
                output_padding=0,
            ),
            conv_activation(),
        )
        self.h4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                deconv3_size,
                deconv4_size,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=padding,
                output_padding=1,
            ),
            conv_activation(),
        )
        c, h, w = self._conv_outsize()

        print('final outsize', (c, h, w))
        self.initialiser = weight_initialiser
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            self.initialiser(module.weight)

    def _conv_outsize(self):
        _, h, w = self.output_size
        h, w = deconv2d_outsize(
            h,
            w,
            self.h1[0].kernel_size,
            self.h1[0].stride,
            self.h1[0].padding,
            self.h1[0].dilation,
            self.h1[0].output_padding,
        )
        h, w = deconv2d_outsize(
            h,
            w,
            self.h2[0].kernel_size,
            self.h2[0].stride,
            self.h2[0].padding,
            self.h2[0].dilation,
            self.h2[0].output_padding,
        )
        h, w = deconv2d_outsize(
            h,
            w,
            self.h3[0].kernel_size,
            self.h3[0].stride,
            self.h3[0].padding,
            self.h3[0].dilation,
            self.h3[0].output_padding,
        )
        h, w = deconv2d_outsize(
            h,
            w,
            self.h4[0].kernel_size,
            self.h4[0].stride,
            self.h4[0].padding,
            self.h4[0].dilation,
            self.h4[0].output_padding,
        )
        return self.h4[0].out_channels, h, w

    def forward(self, x):
        x = x.view(-1, *self.output_size)
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        return x


class UniverseCNN(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        conv1_size=64,
        conv2_size=64,
        conv3_size=64,
        conv4_size=64,
        padding=None,
        dense_size=256,
        conv_activation=torch.nn.ELU,
        dense_activation=torch.nn.ReLU,
        weight_initialiser=torch.nn.init.xavier_uniform_,
        scale=True,
        trainable=True,
    ):
        # input_shape [channels, height, width]
        if padding is None:
            padding = (0, 0)
        super().__init__()
        self.scale = scale
        self.input_shape = input_shape

        self.h1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                input_shape[0], conv1_size, kernel_size=(3, 3), stride=(2, 2), padding=padding
            ),
            conv_activation(),
        )
        self.h2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                conv1_size, conv2_size, kernel_size=(3, 3), stride=(2, 2), padding=padding
            ),
            conv_activation(),
        )
        self.h3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                conv2_size, conv3_size, kernel_size=(3, 3), stride=(2, 2), padding=padding
            ),
            conv_activation(),
        )
        self.h4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                conv3_size, conv4_size, kernel_size=(3, 3), stride=(2, 2), padding=padding
            ),
            conv_activation(),
        )
        self.flatten = torch.nn.Flatten()
        c, h, w = self._conv_outsize()
        self.dense = torch.nn.Sequential(torch.nn.Linear(h * w * c, dense_size), dense_activation())
        # self.dense_size = h*w*c
        self.dense_size = dense_size
        print('final outsize', (c, h, w))
        self.initialiser = weight_initialiser
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            self.initialiser(module.weight)

    def _conv_outsize(self):
        _, h, w = self.input_shape
        h, w = conv2d_outsize(h, w, self.h1[0].kernel_size, self.h1[0].stride, self.h1[0].padding)
        h, w = conv2d_outsize(h, w, self.h2[0].kernel_size, self.h2[0].stride, self.h2[0].padding)
        h, w = conv2d_outsize(h, w, self.h3[0].kernel_size, self.h3[0].stride, self.h3[0].padding)
        h, w = conv2d_outsize(h, w, self.h4[0].kernel_size, self.h4[0].stride, self.h4[0].padding)
        return self.h4[0].out_channels, h, w

    def forward(self, x):
        x = x / 255 if self.scale else x
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class NatureCNN(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        conv1_size=32,
        conv2_size=64,
        conv3_size=64,
        dense_size=512,
        padding=None,
        conv_activation=torch.nn.ReLU,
        dense_activation=torch.nn.ReLU,
        weight_initialiser=torch.nn.init.xavier_uniform_,
        scale=True,
        trainable=True,
    ):
        # input_shape [channels, height, width]
        if padding is None:
            padding = (0, 0)
        super().__init__()
        self.scale = scale
        self.dense_size = dense_size
        self.input_shape = input_shape
        self.h1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                input_shape[0], conv1_size, kernel_size=(8, 8), stride=(4, 4), padding=padding
            ),
            conv_activation(),
        )
        self.h2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                conv1_size, conv2_size, kernel_size=(4, 4), stride=(2, 2), padding=padding
            ),
            conv_activation(),
        )
        self.h3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                conv2_size, conv3_size, kernel_size=(3, 3), stride=(1, 1), padding=padding
            ),
            conv_activation(),
        )
        self.flatten = torch.nn.Flatten()
        c, h, w = self._conv_outsize()
        self.dense = torch.nn.Sequential(torch.nn.Linear(h * w * c, dense_size), dense_activation())
        self.initialiser = weight_initialiser
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            self.initialiser(module.weight)

    def _conv_outsize(self):
        _, h, w = self.input_shape
        h, w = conv2d_outsize(h, w, self.h1[0].kernel_size, self.h1[0].stride, self.h1[0].padding)
        h, w = conv2d_outsize(h, w, self.h2[0].kernel_size, self.h2[0].stride, self.h2[0].padding)
        h, w = conv2d_outsize(h, w, self.h3[0].kernel_size, self.h3[0].stride, self.h3[0].padding)
        return self.h3[0].out_channels, h, w

    def forward(self, x):
        x = x / 255 if self.scale else x
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class MaskedRNN(torch.nn.Module):
    '''dynamic masked *hidden state* RNN for sequences that reset part way through an observation
    e.g. A2C
    args :
        cell - a recurrent cell module (e.g. ``torch.nn.LSTMCell`` or ``torch.nn.GRUCell``)
        X - tensor of rank [time, batch, hidden] if time major == True (Default); or [batch, time, hidden] if time major == False
        hidden_init - tensor of initial cell hidden state
        mask - boolean tensor of length time, used for hidden state masking e.g. [True, False, False] will mask the first hidden state
        time_major - bool flag to determine order of indices of input tensor
    '''

    def __init__(self, cell, time_major=True):
        super().__init__()
        self.cell = cell
        self.time_major = time_major

    def forward(self, x, hidden=None, mask=None):
        '''Args:
        x: tensor of rank [time, batch, hidden] if time_major == True (default); or [batch, time, hidden] if time_major == False.
        mask: tensor of rank [time], for hidden state masking e.g. [True, False, False] will mask first hidden state.
        '''

        if not self.time_major:
            x = x.transpose(1, 0, 2)

        if mask is None:
            mask = torch.zeros(x.shape[0], x.shape[1]).to(x.device)

        outputs = []
        for t in range(x.shape[0]):
            output, hidden = self.cell(x[t], hidden, mask[t])
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        outputs = outputs if self.time_major else outputs.transpose(1, 0, 2)
        return outputs, hidden


def lstmgate(cell_size, input_size, trainable=True):
    input_weight = torch.nn.Parameter(
        torch.nn.init.xavier_uniform_(
            torch.zeros(size=[input_size, cell_size], requires_grad=trainable)
        )
    )
    hidden_weight = torch.nn.Parameter(
        torch.nn.init.xavier_uniform_(
            torch.zeros(size=[cell_size, cell_size], requires_grad=trainable)
        )
    )
    bias = torch.nn.Parameter(torch.zeros(size=[cell_size], requires_grad=trainable))
    return input_weight, hidden_weight, bias


def gemmlstmgate(cell_size, input_size, trainable=True):
    input_weight = torch.nn.Parameter(
        torch.nn.init.xavier_uniform_(
            torch.zeros(size=[cell_size * 4, input_size], requires_grad=trainable)
        )
    )
    hidden_weight = torch.nn.Parameter(
        torch.nn.init.xavier_uniform_(
            torch.zeros(size=[cell_size * 4, cell_size], requires_grad=trainable)
        )
    )
    bias_input = torch.nn.Parameter(torch.zeros(size=[cell_size * 4], requires_grad=trainable))
    bias_hidden = torch.nn.Parameter(torch.zeros(size=[cell_size * 4], requires_grad=trainable))
    return input_weight, hidden_weight, bias_input, bias_hidden


class MaskedLSTMCell(torch.nn.Module):
    def __init__(self, cell_size, input_size=None, trainable=True):
        super().__init__()
        self._cell_size = cell_size
        input_size = (
            input_size if input_size is not None else cell_size
        )  # input_size == cell_size by default
        self._input_size = input_size
        self.Wi, self.Wh, self.bi, self.bh = gemmlstmgate(
            cell_size, input_size, trainable
        )  # batch gemm

    def init_hidden(self, batch_size, dtype, device):
        cell = torch.zeros(1, batch_size, self._cell_size, dtype=dtype, device=device)
        hidden = torch.zeros(1, batch_size, self._cell_size, dtype=dtype, device=device)
        return (cell, hidden)

    def forward(self, x, state=None, done=None):
        if state is None:
            prev_cell, prev_hidden = self.init_hidden(x.shape[0], input.dtype, input.device)
        else:
            prev_cell, prev_hidden = state
        if done is not None:
            prev_cell *= (1 - done).view(-1, 1)
            prev_hidden *= (1 - done).view(-1, 1)

        gates = (
            torch.matmul(x, self.Wi.t()) + self.bi + torch.matmul(prev_hidden[0], self.Wh.t())
        ) + self.bh
        i, f, c, o = gates.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c = torch.tanh(c)
        o = torch.sigmoid(o)

        cell = prev_cell * f + i * c
        hidden = o * torch.tanh(cell)
        return hidden, (cell, hidden)


class MaskedLSTMBlock(torch.nn.Module):
    def __init__(self, input_size, hidden_size, time_major=True):
        super().__init__()
        self.time_major = time_major
        batch_first = not time_major
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )

    def forward(self, x, hidden, done):
        if not self.time_major:
            x = x.transpose(1, 0, 2)

        if done is not None:
            mask = 1 - done
        else:
            mask = torch.ones(x.shape[0], x.shape[1]).to(x.device)

        mask_zeros = ((mask[1:] == 0).any(dim=-1).nonzero() + 1).view(-1).cpu().numpy().tolist()
        mask_zeros = [0] + mask_zeros + [mask.shape[0] + 1]
        outputs = []
        for i in range(len(mask_zeros) - 1):
            start = mask_zeros[i]
            end = mask_zeros[i + 1]
            # print('start, end', (start, end))
            hidden = (mask[start].view(-1, 1) * hidden[0], mask[start].view(-1, 1) * hidden[1])
            out, hidden = self.lstm(x[start:end], hidden)
            outputs.append(out)

        outputs = torch.cat(outputs, dim=0)
        outputs = outputs if self.time_major else outputs.transpose(1, 0, 2)
        return outputs, hidden
