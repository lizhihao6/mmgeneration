import math

import torch
import torch.nn as nn
import torch.nn.init as init


def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def uniform_binning_correction(x, n_bits=8):
    """Replaces x^i with q^i(x) = U(x, x + 1.0 / 256.0).
    Args:
        x: 4-D Tensor of shape (NCHW)
        n_bits: optional.
    Returns:
        x: x ~ U(x, x + 1.0 / 256)
        objective: Equivalent to -q(x)*log(q(x)).
    """
    b, c, h, w = x.size()
    n_bins = 2 ** n_bits
    chw = c * h * w
    x += torch.zeros_like(x).uniform_(0, 1.0 / n_bins)

    objective = -math.log(n_bins) * chw * torch.ones(b, device=x.device)
    return x, objective


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        # return tensor[:, : C // 2, ...], tensor[:, C // 2 :, ...]
        return tensor[:, :1, ...], tensor[:, 1:, ...]
    elif type == "cross":
        # return tensor[:, 0::2, ...], tensor[:, 1::2, ...]
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def gaussian_p(mean, logs, x):
    """
    lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
            k = 1 (Independent)
            Var = logs ** 2
    """
    c = math.log(2 * math.pi)
    return -0.5 * (logs * 2.0 + ((x - mean) ** 2) / torch.exp(logs * 2.0) + c)


def gaussian_likelihood(mean, logs, x):
    p = gaussian_p(mean, logs, x)
    return torch.sum(p, dim=[1, 2, 3])


def gaussian_sample(mean, logs, temperature=1):
    # Sample from Gaussian with temperature
    z = torch.normal(mean, torch.exp(logs) * temperature)

    return z


def squeeze2d(input, factor):
    if factor == 1:
        return input

    B, C, H, W = input.size()

    assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"

    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)

    return x


def unsqueeze2d(input, factor):
    if factor == 1:
        return input

    factor2 = factor ** 2

    B, C, H, W = input.size()

    assert C % (factor2) == 0, "C module factor squared is not 0"

    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)

    return x


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.
    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.0):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = nn.Parameter(torch.zeros(1).bool(), requires_grad=False)

    def initialize_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = -torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.inited[0] = True

    def _center(self, input, reverse=False):
        if reverse:
            return input - self.bias
        else:
            return input + self.bias

    def _scale(self, input, logdet=None, reverse=False):

        if reverse:
            input = input * torch.exp(-self.logs)
        else:
            input = input * torch.exp(self.logs)

        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply by number of pixels
            """
            b, c, h, w = input.shape

            dlogdet = torch.sum(self.logs) * h * w

            if reverse:
                dlogdet *= -1

            logdet = logdet + dlogdet

        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        self._check_input_dim(input)

        if not self.inited:
            self.initialize_parameters(input)

        if reverse:
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        else:
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)

        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.0):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()
            )
        )


class LinearZeros(nn.Module):
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.logscale_factor = logscale_factor

        self.logs = nn.Parameter(torch.zeros(out_channels))

    def forward(self, input):
        output = self.linear(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Conv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            do_actnorm=True,
            weight_std=0.05,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=(not do_actnorm),
        )

        # init weight with std
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)

        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)

        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = self.conv(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            logscale_factor=3,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, input):
        output = self.conv(input)
        return output * torch.exp(self.logs * self.logscale_factor)


# class InvertibleConv1x1(nn.Module):
#     def __init__(self, num_channels, LU_decomposed):
#         super().__init__()
#         w_shape = [num_channels, num_channels]
#         w_init = torch.linalg.qr(torch.randn(*w_shape), 'reduced')[0]

#         if not LU_decomposed:
#             self.weight = nn.Parameter(torch.Tensor(w_init))
#         else:
#             p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
#             s = torch.diag(upper)
#             sign_s = torch.sign(s)
#             log_s = torch.log(torch.abs(s))
#             upper = torch.triu(upper, 1)
#             l_mask = torch.tril(torch.ones(w_shape), -1)
#             eye = torch.eye(*w_shape)

#             self.register_buffer("p", p)
#             self.register_buffer("sign_s", sign_s)
#             self.lower = nn.Parameter(lower)
#             self.log_s = nn.Parameter(log_s)
#             self.upper = nn.Parameter(upper)
#             self.l_mask = l_mask
#             self.eye = eye

#         self.w_shape = w_shape
#         self.LU_decomposed = LU_decomposed

#     def get_weight(self, input, reverse):
#         b, c, h, w = input.shape

#         if not self.LU_decomposed:
#             dlogdet = torch.slogdet(self.weight)[1] * h * w
#             if reverse:
#                 weight = torch.inverse(self.weight)
#             else:
#                 weight = self.weight
#         else:
#             self.l_mask = self.l_mask.to(input.device)
#             self.eye = self.eye.to(input.device)

#             lower = self.lower * self.l_mask + self.eye

#             u = self.upper * self.l_mask.transpose(0, 1).contiguous()
#             u += torch.diag(self.sign_s * torch.exp(self.log_s))

#             dlogdet = torch.sum(self.log_s) * h * w

#             if reverse:
#                 u_inv = torch.inverse(u)
#                 l_inv = torch.inverse(lower)
#                 p_inv = torch.inverse(self.p)

#                 weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
#             else:
#                 weight = torch.matmul(self.p, torch.matmul(lower, u))

#         return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

#     def forward(self, input, logdet=None, reverse=False):
#         """
#         log-det = log|abs(|W|)| * pixels
#         """
#         weight, dlogdet = self.get_weight(input, reverse)

#         if not reverse:
#             z = F.conv2d(input, weight)
#             if logdet is not None:
#                 logdet = logdet + dlogdet
#             return z, logdet
#         else:
#             z = F.conv2d(input, weight)
#             if logdet is not None:
#                 logdet = logdet - dlogdet
#             return z, logdet

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        # self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        # self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        # self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2], 0.1)
            # initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2], 0.1)
            # initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv3, 0)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        return x3
        # x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        # x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        # return x5


# class _ResBlock(nn.Module):
#     def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
#         super(_ResBlock, self).__init__()
#         self.seq = nn.Sequential(
#             Conv2d(channel_in, channel_out),
#             nn.ReLU(inplace=False),
#             # nn.LeakyReLU(negative_slope=0.2, inplace=True)
#             # nn.PReLU(),
#             Conv2d(channel_out, channel_out),
#             nn.ReLU(inplace=False)
#             # nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         )

#         if channel_in != channel_out:
#             self.residual = nn.Conv2d(channel_in, channel_out, 1, 1, 0)
#             initialize_weights(self.residual, 0)

#     def forward(self, x):
#         residual = self.residual(x) if hasattr(self, 'residual') else x
#         x = self.seq(x) + residual
#         return x


class _ResBlock(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.seq = nn.Sequential(
            Conv2d(hidden_channels, hidden_channels),
            nn.ReLU(inplace=False),
            # nn.PReLU(),
            Conv2d(hidden_channels, hidden_channels),
            nn.ReLU(inplace=False)
            # nn.PReLU()
        )

    def forward(self, input):
        x = self.seq(input)
        return x + input


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, init=None, hidden_channels=32):
        super().__init__()
        self.stem = nn.Sequential(
            Conv2d(in_channels, hidden_channels, kernel_size=(1, 1)),
            nn.ReLU(inplace=False)
        )
        self.res1 = _ResBlock(hidden_channels)
        # self.res2 = _ResBlock(hidden_channels)
        # self.res3 = _ResBlock(hidden_channels)
        self.root = Conv2dZeros(hidden_channels, out_channels, kernel_size=(1, 1))

    def forward(self, z):
        z = self.stem(z)
        z = self.res1(z)
        # z = self.res2(z)
        # z = self.res3(z)
        z = self.root(z)
        return z


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
            # return UNetBlock(channel_in, channel_out)
        elif net_structure == 'RBNet':
            if init == 'xavier':
                return ResBlock(channel_in, channel_out, init)
            else:
                return ResBlock(channel_in, channel_out)
        else:
            return None

    return constructor


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        # in_channels = 3        
        # self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        # self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        if not rev:
            # invert1x1conv 
            # x, logdet = self.flow_permutation(x, logdet=0, rev=False) 

            # split to 1 channel and 2 channel. 
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

            y1 = x1 + self.F(x2)  # 1 channel
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
            out = torch.cat((y1, y2), 1)
        else:
            # split. 
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

            x = torch.cat((y1, y2), 1)

            # inv permutation 
            # out, logdet = self.flow_permutation(x, logdet=0, rev=True)
            out = x

        return out


class TinyEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channnels=64, input_size=128):
        super().__init__()
        self.input_size = input_size
        self.net = torch.nn.Sequential(
            nn.Conv2d(in_channels * 2, 32, 5, 1, 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(32, channnels, 3, 1, 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(channnels, channnels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.linear = nn.Sequential(
            nn.Linear(channnels, 256),
            nn.Linear(256, out_channels)
        )

    def forward(self, x):
        assert x.size(2) == self.input_size
        x = torch.cat([x, x ** 2], dim=1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def differentiable_histogram(x, bins=255, min=0.0, max=1.0):
    if len(x.shape) == 4:
        n_samples, n_chns, _, _ = x.shape
    elif len(x.shape) == 2:
        n_samples, n_chns = 1, 1
    else:
        raise AssertionError('The dimension of input tensor should be 2 or 4.')

    hist_torch = torch.zeros(n_samples, n_chns, bins).to(x.device)
    delta = (max - min) / bins

    BIN_Table = torch.arange(start=0, end=bins + 1, step=1) * delta

    for dim in range(1, bins - 1, 1):
        h_r = BIN_Table[dim].item()  # h_r
        h_r_sub_1 = BIN_Table[dim - 1].item()  # h_(r-1)
        h_r_plus_1 = BIN_Table[dim + 1].item()  # h_(r+1)

        mask_sub = ((h_r > x) & (x >= h_r_sub_1)).float()
        mask_plus = ((h_r_plus_1 > x) & (x >= h_r)).float()

        hist_torch[:, :, dim] += torch.sum(((x - h_r_sub_1) * mask_sub).view(n_samples, n_chns, -1), dim=-1)
        hist_torch[:, :, dim] += torch.sum(((h_r_plus_1 - x) * mask_plus).view(n_samples, n_chns, -1), dim=-1)

    return hist_torch / delta
