# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import load_checkpoint

import torch
from mmgen.models.builder import MODULES
from mmgen.utils import get_root_logger
import torch.nn.init as init
from .modules import subnet, InvBlock, TinyEncoder, differentiable_histogram
from functools import partial
import torch.distributed as dist
from mmgen.models.common import set_requires_grad


@MODULES.register_module()
class inverseISP(nn.Module):
    def __init__(self, pretrain=True, subnet_constructor=subnet('RBNet'), block_num=1, wb=None, ccm=None, global_size=128):
        super().__init__()
        self.pretrain = pretrain
        operations = []
        channel_num = 3
        channel_split_num = 1

        for j in range(block_num): 
            b = InvBlock(subnet_constructor, channel_num, channel_split_num) # one block is one flow step. 
            operations.append(b)
        
        self.operations = nn.ModuleList(operations)
        self.initialize()

        wb = [
            [2.0931, 1.6701],
            [2.1932, 1.7702],
            [2.2933, 1.8703],
            [2.3934, 1.9704],
            [2.4935, 1.9705]

        ] if wb is None else wb
        self.wb = nn.Parameter(torch.FloatTensor(wb), requires_grad=True)

        ccm = [
            [[ 1.67557, -0.52636, -0.04920],
             [-0.16799,  1.32824, -0.36024],
             [ 0.03188, -0.22302,  1.59114]],
            [[ 1.57558, -0.52637, -0.04921],
             [-0.16798,  1.52823, -0.36023],
             [0.031885, -0.42303,  1.39115]]
        ] if ccm is None else ccm
        self.ccm = nn.Parameter(torch.FloatTensor(ccm), requires_grad=True)

        self.x_size = global_size
        self.resize_fn = partial(nn.functional.interpolate, size=global_size)

        # inverse ISP
        self.ccm_estimator = TinyEncoder(out_channels=2)
        self.bright_estimator = TinyEncoder(out_channels=2)
        self.wb_estimator = TinyEncoder(out_channels=2)

        # ISP
        self.wb_evaluator = TinyEncoder(out_channels=1)
        self.bright_evaluator = TinyEncoder(out_channels=1)
        self.ccm_evaluator = TinyEncoder(out_channels=1)

        # torch.autograd.set_detect_anomaly(True)
        if not self.pretrain:
            self.wb.required_grad = False
            self.ccm.required_grad = False
            set_requires_grad([self.ccm_estimator, self.bright_estimator, self.wb_estimator, self.wb_evaluator, self.bright_evaluator, self.ccm_evaluator], False)

        self.counter = 0

    def initialize(self):
        pass

    def init_weights(self, pretrained=None, strict=True):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.xavier_normal_(m.weight)
                    m.weight.data *= 1.  # for residual block
                    if m.bias is not None:
                        m.bias.data.zero_() 
                elif isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
                    m.weight.data *= 1.
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias.data, 0.0)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')


    def safe_inverse_gain(self, x, gain):
        gray = x.mean(1, keepdim=True)
        inflection = 0.9
        mask = torch.maximum(gray-inflection, torch.zeros_like(gray)) / (1-inflection)
        mask = mask ** 2
        mask = torch.clamp(mask, 0, 1)
        safe_gain = torch.maximum(mask+(1-mask)*gain, gain)
        x = x * safe_gain
        return x

    def mosaic(self, x):
        b, _, h, w = x.shape
        raw = torch.zeros([b, 4, h//2, w//2]).to(x.device)
        raw[:, 0] = x[:, 0, 0::2, 0::2]
        raw[:, 1] = x[:, 1, 0::2, 1::2]
        raw[:, 2] = x[:, 1, 1::2, 1::2]
        raw[:, 3] = x[:, 2, 1::2, 1::2]
        return raw

    def demosaic(self, x):
        b, _, h, w = x.shape
        rgb = torch.zeros([b, 3, h*2, w*2]).to(x.device)
        x = torch.nn.functional.pad(x, 1, 'reflect')
        r, gb, gr, b = torch.split(x, 1, dim=1) 
        
        rgb[..., 0, 0::2, 0::2] = r[..., 1:-1, 1:-1]
        rgb[..., 1, 0::2, 0::2] = (gr[..., 1:-1, 1:-1] + gr[..., :-2, 1:-1] + gb[..., 1:-1, 1:-1] + gb[..., 1:-1, :-2]) / 4
        rgb[..., 2, 0::2, 0::2] = (b[..., 1:-1, 1:-1] + b[..., :-2, :-2] + b[..., 1:-1, :-2] + b[..., :-2, 1:-1]) / 4

        rgb[..., 0, 1::2, 0::2] = (r[..., 1:-1, 1:-1] + r[..., 2:, 1:-1]) / 2
        rgb[..., 1, 1::2, 0::2] = gr[..., 1:-1, 1:-1]
        rgb[..., 2, 1::2, 0::2] = (b[..., 1:-1, 1:-1] + b[..., 1:-1, :-2]) / 2

        rgb[..., 0, 0::2, 1::2] = (r[..., 1:-1, 1:-1] + r[..., 1:-1, 2:]) / 2
        rgb[..., 1, 0::2, 1::2] = gb[..., 1:-1, 1:-1]
        rgb[..., 2, 0::2, 1::2] = (b[..., 1:-1, 1:-1] + b[..., :-2, 1:-1]) / 2

        rgb[..., 0, 1::2, 1::2] = (r[..., 1:-1, 1:-1] + r[..., 2:, 2:] + r[..., 1:-1, 2:] + r[..., 2:, 1:-1]) / 4
        rgb[..., 1, 1::2, 1::2] = (gr[..., 1:-1, 1:-1] + gr[..., 1:-1, 2:] + gb[..., 1:-1, 1:-1] + gb[..., 2:, 1:-1]) / 4
        rgb[..., 2, 1::2, 1::2] = b[..., 1:-1, 1:-1]
            
        return rgb


    def forward(self, x, rev=False):
        x = x.clone()

        if self.pretrain and dist.get_rank() == 0 and not rev:
            if self.counter == 600:
                self.counter = 0
                print('wb:', self.wb.detach().cpu())
                print('ccm:', self.ccm.detach().cpu())
            self.counter += 1
            
        # inverse ISP
        if not rev: 
            # inverse local
            if not self.pretrain:
                for op in self.operations:
                    x = op.forward(x, rev)
            
            ## inverse gamma
            x = torch.maximum(x, 1e-8*torch.ones_like(x))
            x = torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
            
            # inverse global
            b = x.shape[0]
            
            ## inverse ccm
            x_resize = self.resize_fn(x)
            ccm = self.ccm / self.ccm.reshape([-1, 3]).sum(1).reshape([-1, 3, 1])
            inv_ccm = torch.linalg.pinv(ccm.transpose(-2, -1))
            ccm_preview = torch.einsum('bchw,ncj->bnjhw', x_resize, inv_ccm)
            ccm_preview = ccm_preview.reshape([-1, 3, self.x_size, self.x_size])
            ccm_distribution = self.ccm_estimator(ccm_preview)
            ccm_prob = torch.zeros([ccm_distribution.shape[0]]).to(x.device)
            for i in range(ccm_distribution.shape[0]):
                mean, std = ccm_distribution[i, 0], torch.abs(ccm_distribution[i, 1])
                m = torch.distributions.normal.Normal(mean, std)
                ccm_prob[i] = m.sample()
            ccm_prob = ccm_prob.reshape(b, self.ccm.shape[0]) # [B, N]
            ccm_prob = torch.softmax(ccm_prob, dim=-1)[..., None, None] # [B, N, 1, 1] 
            inv_ccm = inv_ccm[None] * ccm_prob # [B, N, 3, 3]
            inv_ccm = inv_ccm.sum(1) # [B, 3, 3]
            x = torch.einsum('bchw,bcj->bjhw', [x, inv_ccm]) # [B, 3, 128, 128]

            ## inverse brightness adjustment
            x_resize = self.resize_fn(x)
            bright_distribution = self.bright_estimator(x_resize)
            bright_distribution = torch.tanh(bright_distribution)
            bright_adjust = torch.zeros([b]).to(x.device)
            for i in range(b):
                mean = bright_distribution[i, 0] * 0.2 + 0.8
                std = bright_distribution[i, 1] * 0.05 + 0.1
                m = torch.distributions.normal.Normal(mean, std)
                bright_adjust[i] = m.sample()
            bright_adjust = bright_adjust[:, None, None, None]
            x = self.safe_inverse_gain(x, bright_adjust) # [B, 3, H, W]

            ## inverse awb
            x_resize = self.resize_fn(x)
            gain = torch.ones([b, self.wb.shape[0], 3]).to(x.device)
            gain[..., (0, 2)] = 1/self.wb[None] # [B, N, 3]
            wb_preview = x_resize[:, None].repeat([1, self.wb.shape[0], 1, 1, 1]) # [B, N, 3, 128, 128]
            wb_preview = wb_preview.reshape([-1, 3, self.x_size, self.x_size]) # [B*N, 3, 128, 128]
            gain = gain.reshape([-1, 3, 1, 1]) # [B*N, 3, 1, 1]
            wb_preview = self.safe_inverse_gain(wb_preview, gain)
            wb_distribution = self.wb_estimator(wb_preview)
            wb_prob = torch.zeros([wb_distribution.shape[0]]).to(x.device)
            for i in range(wb_distribution.shape[0]):
                mean, std = wb_distribution[i, 0], torch.abs(wb_distribution[i, 1])
                m = torch.distributions.normal.Normal(mean, std)
                wb_prob[i] = m.sample()
            wb_prob = wb_prob.reshape(b, self.wb.shape[0]) # [B, N]
            wb_prob = torch.softmax(wb_prob, dim=-1)[..., None, None, None] # [B, N, 1, 1, 1]
            gain = gain.reshape([b, -1, 3, 1, 1]) * wb_prob # [B, N, 3, 1, 1]
            gain = gain.sum(1) # [B, 3, 1, 1]
            x = self.safe_inverse_gain(x, gain) # [B, 3, H, W]

            # mosaic
            # x = self.mosaic(x)

        # ISP
        else:
            ## global
            b = x.shape[0]

            # demosaic
            # x = self.demosaic(x)

            # white balance
            x_resize = self.resize_fn(x)
            x_resize = x_resize[:, None].repeat([1, self.wb.shape[0], 1, 1, 1]) # [B, N, 3, 128, 128]
            x_resize[:, :, (0, 2)] = x_resize[:, :, (0, 2)] * self.wb[None, :, :, None, None] # [B, N, 3, 128, 128]
            x_resize = x_resize.reshape([-1, 3, self.x_size, self.x_size]) # [B*N, 3, 128, 128]
            wb_prob = self.wb_evaluator(x_resize).reshape([b, self.wb.shape[0]]) # [B, N]
            wb_prob = torch.softmax(wb_prob, dim=-1)[..., None, None, None] # [B, N, 1, 1, 1]
            wb = self.wb.reshape([1, -1, 2, 1, 1]) # [B, N, 2, 1, 1]
            wb = (wb*wb_prob).sum(1) # [B, 2, 1, 1]
            x[:, (0, 2)] = x[:, (0, 2)] * wb

            # brightness adjustment
            x_resize = self.resize_fn(x)
            bright_adjust = self.bright_evaluator(x_resize)
            bright_adjust = torch.tanh(bright_adjust)*0.2 + 0.8
            x = x / bright_adjust[:, :, None, None]

            # ccm
            x_resize = self.resize_fn(x)
            ccm = self.ccm / self.ccm.sum(2, keepdims=True)
            ccm = ccm.transpose(1, 2)
            ccm_preview = torch.einsum('bchw,ncj->bnjhw', x_resize, ccm) # [B, N, 3, 128, 128]
            ccm_preview = ccm_preview.reshape([-1, 3, self.x_size, self.x_size]) # [B*N, 3, 128, 128]
            ccm_prob = self.ccm_evaluator(ccm_preview).reshape([b, self.ccm.shape[0]]) # [B, N]
            ccm_prob = torch.softmax(ccm_prob, dim=-1)[..., None, None] # [B, N, 1, 1]
            ccm = ccm[None] * ccm_prob # [B, N, 3, 3]
            ccm = ccm.sum(1) # [B, 3, 3]
            x = torch.einsum('bchw,bcj->bjhw', x, ccm) # [B, 3, H, W]
            
            # gamma correction
            x = torch.maximum(x, 1e-8*torch.ones_like(x))
            x = torch.where(x <= 0.0031308, 12.92 * x, 1.055 * torch.pow(x, 1 / 2.4) - 0.055)

            # local
            if not self.pretrain:
                for op in reversed(self.operations):
                    x = op.forward(x, rev)
        
        return x


class FCDiscriminator(nn.Module):

    def __init__(self, in_channels=3, ndf=64):
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        x = x.reshape([x.shape[0], -1]).mean(1)
        return x


@MODULES.register_module()
class HistAwareDiscriminator(nn.Module):

    def __init__(self, in_channels=3, ndf=64, bins = 255, global_size=128):
        super(HistAwareDiscriminator, self).__init__()
        self.bins = bins
        self.classifer = nn.Sequential(
            nn.Linear(bins*in_channels, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 256),
            nn.Linear(256, 1)
        )
        self.local_fcd = FCDiscriminator(in_channels=in_channels, ndf=ndf)
        self.global_fcd = FCDiscriminator(in_channels=in_channels, ndf=ndf)
        self.x_size = global_size

    def init_weights(self, pretrained=None, strict=True):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.xavier_normal_(m.weight)
                    m.weight.data *= 1.  # for residual block
                    if m.bias is not None:
                        m.bias.data.zero_() 
                elif isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
                    m.weight.data *= 1.
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias.data, 0.0)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')

    def forward(self, x):
        hist = differentiable_histogram(x, self.bins) # [B, C, 256]
        hist /= x.shape[2]*x.shape[3]
        hist_judge = self.classifer(hist.reshape([hist.shape[0], -1]))
        local_judge = self.local_fcd(x.clone())
        x_global = torch.nn.functional.interpolate(x, size=self.x_size)
        global_judge = self.global_fcd(x_global)
        final_judge = 0.3*hist_judge + 0.3*local_judge + 0.4*global_judge
        return final_judge
