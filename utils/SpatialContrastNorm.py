#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 13:58:41 2022

@author: liang
"""
import torch.nn.functional as nnF
import torch.nn as nn


# %%
def spatial_contrast_norm(x, kernel_size=None, eps=1e-5):
    # x.shape (B,C,H,W)
    B, C, H, W = x.shape
    if kernel_size is None:
        kernel_size = (H // 8, W // 8)
        if H <= 8 and W <= 8:
            x = nnF.group_norm(x, 1)
            return x
    if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
        pad_h = kernel_size[0] // 2
        pad_w = kernel_size[1] // 2
        h = 2 * pad_h + 1
        w = 2 * pad_w + 1
    else:
        pad_h = pad_w = kernel_size // 2
        h = w = 2 * pad_h + 1
    padding = (pad_w, pad_w, pad_h, pad_h)
    x_mean = x.mean(dim=1, keepdim=True)
    if h > 1 or w > 1:
        x_mean = nnF.pad(x_mean, padding, 'reflect')
        x_mean = nnF.avg_pool2d(x_mean, kernel_size=(h, w), padding=0, stride=1)
    x_var = (x - x_mean) ** 2
    x_var = x_var.mean(dim=1, keepdim=True)
    if h > 1 or w > 1:
        x_var = nnF.pad(x_var, padding, 'reflect')
        x_var = nnF.avg_pool2d(x_var, kernel_size=(h, w), padding=0, stride=1)
    x_std = (x_var + eps).sqrt()
    y = (x - x_mean) / x_std
    return y


class SpatialContrastNorm(nn.Module):
    def __init__(self, kernel_size=None, eps=1e-5):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        y = spatial_contrast_norm(x, self.kernel_size, self.eps)
        return y
