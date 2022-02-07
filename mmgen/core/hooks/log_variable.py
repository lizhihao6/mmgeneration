# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import torch
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only

import cv2
import numpy as np


def save_image(tensor, path, nrow, padding):
    # replace torchvision.utils import save_image because of segmentation error
    img = tensor.detach().cpu().numpy()
    b, c, h, w = img.shape
    assert c == 3
    total_h = h * b + (b-1) * padding
    _img = np.zeros([total_h, w, 3])
    start_h = 0
    for i in range(b):
        _img[start_h:start_h+h] = img[i].transpose([1, 2, 0])
        start_h += (h + padding)
    _img = np.clip(_img, 0, 1) * 255
    _img = np.round(_img).astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(path, _img)


@HOOKS.register_module('MMGenLogVariableHook')
class LogVariableHook(Hook):
    """Visualization hook.

    In this hook, log variable in res_name_list with freq interval.

    Args:
        res_name_list (str): The list contains the name of results in outputs
            dict. The results in outputs dict must be a torch.Tensor with shape
            (n, c, h, w).
        interval (int): The interval of calling this hook. If set to -1,
            the visualization hook will not be called. Default: -1.
    """

    def __init__(self,
                 res_name_list,
                 interval=-1):
        assert mmcv.is_list_of(res_name_list, str)
        self.res_name_list = res_name_list
        self.interval = interval

    @master_only
    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        results = runner.outputs['results']
        for k in self.res_name_list:
            if k in results.keys():
                print(f'{k}: {results[k]}')
