#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as nnF
from utils.ImageSampler import SampleImage2D


# %%
def CreateGrid2D(H, W):
    x_range = torch.arange(0, W, dtype=torch.int64)
    y_range = torch.arange(0, H, dtype=torch.int64)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='xy')
    # for x in x_range:
    #     x=int(x)
    #     for y in y_range:
    #         y=int(y)
    #         if (grid_x[y,x] != x) or (grid_y[y,x]!=y):
    #             print('wrong')
    grid = torch.cat([grid_x.reshape(1, H, W, 1), grid_y.reshape(1, H, W, 1)], dim=3)
    return grid  # (1, H, W,2)


# %%
def CreateGrid2D_rand(B, H, W, sigma, cutoff=0.5, dtype=torch.float32, device='cpu'):
    # B batch_szie
    # H height
    # W width
    cutoff = abs(cutoff)
    grid = CreateGrid2D(H, W)
    grid = grid.expand(B, H, W, 2)
    grid = grid.to(dtype).to(device)
    noise = torch.randn_like(grid) * sigma
    # noise=(2*torch.rand_like(grid)-1)*sigma
    noise.clamp_(min=-cutoff, max=cutoff)
    noise = nnF.avg_pool2d(noise.permute(0, 3, 1, 2), kernel_size=3, stride=1, padding=1)
    noise = noise.permute(0, 2, 3, 1)
    noise.clamp_(min=-cutoff, max=cutoff)
    grid_rand = grid + noise
    return grid_rand  # (B,H,W,2)


# %%
def InterpolateGrid2D(grid_in, H_out, W_out):
    # grid_in.shape (B,H_in,W_in,2)
    H_in = grid_in.shape[1]
    W_in = grid_in.shape[2]
    grid = CreateGrid2D(H_out, W_out)
    grid = grid.to(grid_in.dtype).to(grid_in.device)
    grid[..., 0] *= (W_in - 1) / (W_out - 1)
    grid[..., 1] *= (H_in - 1) / (H_out - 1)
    grid = grid.view(-1, H_out * W_out, 2)
    grid_out = SampleImage2D(grid_in.permute(0, 3, 1, 2), grid, mode='bilinear', padding_mode='zeros',
                             align_corners=True,
                             output_shape='BNC')
    grid_out[..., 0] *= (W_out - 1) / (W_in - 1)
    grid_out[..., 1] *= (H_out - 1) / (H_in - 1)
    return grid_out  # (B,H_out,W_out,2)


# %%
def ElasticDeform2D(image, label, deformed_grid):
    # image (B,C,H,W)
    # label (B,1,H,W)
    # deformed_grid (B,H,W,2)
    B, C, H, W = image.shape
    deformed_grid = deformed_grid.view(B, H * W, 2)
    image_new = SampleImage2D(image, deformed_grid, mode='bilinear', padding_mode='zeros', align_corners=True,
                              output_shape='BCN')
    image_new = image_new.reshape(*image.shape)
    label_new = SampleImage2D(label, deformed_grid, mode='nearest', padding_mode='zeros', align_corners=True,
                              output_shape='BCN')
    label_new = label_new.reshape(*label.shape)
    return image_new, label_new


# %%
def ElasticDeform2D_rand(image, label, deformed_grid_size, sigma, cutoff=0.5):
    # image (B,C,H,W)
    # label (B,1,H,W)
    # deformed_grid_size is (H1, W1)
    # sigma: 0 to 0.5

    # check if the input is one single image with label
    image_shape = image.shape
    label_shape = label.shape
    if len(image.shape) == 2:  # (H,W)
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        label = label.reshape(1, 1, label.shape[-2], label.shape[-1])
    elif len(image.shape) == 3:  # (C,H,W)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        label = label.reshape(1, 1, label.shape[-2], label.shape[-1])
    if torch.is_tensor(image) == False:
        image = torch.tensor(image, dtype=torch.float32)
    if torch.is_tensor(label) == False:
        label = torch.tensor(label, dtype=torch.float32)
    B, C, H, W = image.shape
    H1, W1 = deformed_grid_size

    deformed_grid = CreateGrid2D_rand(B, H1, W1, sigma, cutoff, dtype=image.dtype, device=image.device)
    deformed_grid = InterpolateGrid2D(deformed_grid, H, W)

    image_new, label_new = ElasticDeform2D(image, label, deformed_grid)
    image_new = image_new.reshape(*image_shape)
    label_new = label_new.reshape(*label_shape)
    return image_new, label_new


# %%
if __name__ == "__main__":
    # %%
    from aug_dataloader import aug_disk_512
    import matplotlib.pyplot as plt

    train_dataset = aug_disk_512('/data2/jiasong/100data_aligned/100_aligned_train_11masks.txt', 12, 0, 0, 0, 0)

    img, label, all_mask = train_dataset[0]
    # %%
    img, label = ElasticDeform2D_rand(img, label, (9, 9), 0.25)
    img = img.numpy().reshape(512, 512)
    label = label.numpy().reshape(512, 512)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(label, cmap='gray')
    # %%
