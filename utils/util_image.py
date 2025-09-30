import os
from PIL import Image
from einops import reduce, rearrange
from matplotlib import pyplot as plt
import numpy as np
# from .util_tiff import *
import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce


def to_cpu(tensor):
    if type(tensor) is torch.Tensor:
        tensor = tensor.detach().cpu()
    elif type(tensor) is dict:
        for k in tensor:
            tensor[k] = to_cpu(tensor[k])
    else:
        for i in range(len(tensor)):
            tensor[i] = to_cpu(tensor[i])
    return tensor


def read_png(path):
    image = Image.open(path)
    image = np.array(image).transpose(2, 0, 1)
    # image_tensor = torch.from_numpy(image).to(torch.float32)
    # image_tensor = F.pad(image_tensor, (2, 2, 2, 2))
    return image


def scan_tensor(input_tensor, window_size, stride):
    batch_size, channels, height, width = input_tensor.size()
    # 计算窗口的数量
    h_window_num = (height - window_size[1]) // stride + 1
    w_window_num = (width - window_size[2]) // stride + 1
    # 使用unfold函数将张量展开为窗口
    unfolded_tensor = input_tensor.unfold(2, window_size[1], stride).unfold(
        3, window_size[2], stride
    )
    # 展开后的张量形状为（batch_size, channels, h_window_num, w_window_num, window_size[1], window_size[2]）
    # 重新排列维度，得到形状为（batch_size, channels, h_window_num * w_window_num, window_size[1], window_size[2]）的张量
    unfolded_tensor = unfolded_tensor.contiguous().view(
        batch_size,
        channels,
        h_window_num * w_window_num,
        window_size[1],
        window_size[2],
    )
    return unfolded_tensor

# 图像宽度、条带距离
def apply_select_mask(raw_img, masks, masks_size, p: float = 1, t=2):
    c, h, w = raw_img.shape
    mh, mw = masks_size
    mask_img = raw_img.copy()
    # obs = 1, missing = 0
    # 至少一个通道不为0，则为有效观测
    obs_mask = (raw_img != 0).any(axis=0).reshape(1, h, w) * 1.
    # print('obs:', obs_mask.shape, obs_mask[0, 0, :5])
    origin_mask = obs_mask.copy()
    art_mask = np.zeros_like(obs_mask)
    # print('art:', art_mask.shape)
    # Cloud Mask
    for _ in range(t):
        shift_h = np.random.randint(2 * h)
        shift_w = np.random.randint(2 * w)
        # shift_h, shift_w = 0, 0
        mask_img = np.roll(mask_img, (shift_h, shift_w))
        obs_mask = np.roll(obs_mask, (shift_h, shift_w))
        art_mask = np.roll(art_mask, (shift_h, shift_w))
        for i in range(0, h - mh + 1, mh):
            for j in range(0, w - mw + 1, mw):
                if np.random.random() > p:
                    continue
                window = mask_img[:, i:i+mh, j:j+mw]  # 获取滑动窗口
                obs_window = obs_mask[0, i:i+mh, j:j+mw]
                art_window = art_mask[0, i:i+mh, j:j+mw]
                pid = np.random.randint(len(masks))
                mask_pattern = masks[pid]
                indices = mask_pattern == 0
                # print(indices[0].shape)
                window[:, indices] = 0  # 使用掩码将对应位置的像素值设为0
                # 图案内 and 已观测 = 人工遮掩
                addition_art = indices * (obs_window==1)
                art_window[addition_art] = 1
                obs_window[indices] = 0
        mask_img = np.roll(mask_img, (-shift_h, -shift_w))
        obs_mask = np.roll(obs_mask, (-shift_h, -shift_w))
        art_mask = np.roll(art_mask, (-shift_h, -shift_w))

    # SLC Mask
    dis = 120
    ww = int(dis * 0.35)
    bia = np.random.randint(w)
    unit = np.ones((1, dis))
    unit[0, :ww] = 0
    unit = unit.repeat(w//dis + 1, axis=0)
    unit = unit.reshape(-1)[:w]
    # loop = np.arange(4000)
    unit = unit
    unit = np.roll(unit, bia)
    units = []
    BIA_OF_LAYER = 7
    for _ in range(h):
        units.append(unit)
        unit = np.roll(unit, BIA_OF_LAYER)
    units = np.array(units)
    # 更新原图与人工遮掩
    mask_img[:, units == 0] = 0
    # obs_mask==1 and unit==0 -> 1
    addition_art = ((obs_mask[0]==1) * (units==0)).reshape(1, h, w)
    art_mask[addition_art == 1] = 1
    # 更新观察值
    obs_mask[:, units == 0] = 0
    art_mask = origin_mask - obs_mask

    return mask_img, obs_mask * 1., art_mask * 1.


def calc_mean_face(X, obs_mask):
    # 对各个时间的有效性求和
    valid_cnt = reduce(obs_mask >= 0.5, 'b t 1 h w -> b 1 h w', 'sum')
    # 广播至img的形状
    valid_cnt = repeat(valid_cnt, 'b 1 h w -> b c h w', c=X.shape[2])
    # 对各个时间的像素值求和
    mean_face = reduce(X, 'b t c h w -> b c h w', 'sum')
    # 除以有效值个数
    mean_face = mean_face / (valid_cnt + 1e-5)
    mean_face[valid_cnt == 0] = 0
    mean_face = repeat(mean_face, 'b c h w -> b t c h w', t=X.shape[1])
    return mean_face


def tiff2rgb(tiff):
    rgb = tiff[:3][::-1].astype(float).copy()
    art_maxs = [2200, 2000, 1600]
    # art_maxs = [1800, 1700, 1300]
    for i in range(3):
        # mean = rgb[i][rgb[i]>0].mean()
        # std = rgb[i][rgb[i]>0].std()
        art_max = art_maxs[i]
        rgb[i][rgb[i] > art_max] = art_max
        rgb[i][rgb[i] < 0] = 0
        rgb[i] *= 255 / art_max
    return rgb


def plot_images(images, layout=None, tiff=False, **kwargs):
    num_images = len(images)
    if layout is None:
        fig, axes = plt.subplots(1, num_images, **kwargs)
    else:
        layout[0] = num_images // layout[1] if layout[0] < 0 else layout[0]
        layout[1] = num_images // layout[0] if layout[1] < 0 else layout[1]
        fig, axes = plt.subplots(*layout, **kwargs)
    for index, image in enumerate(images):
        image_np = image.detach().squeeze().numpy()
        if tiff:
            image_np = tiff2rgb(image_np)
        image_np = np.transpose(image_np, (1, 2, 0)) / 255.
        image_np = np.clip(image_np, 0.0, 1.0)
        sub_plot = axes[index // layout[-1]][index %
                                             layout[-1]] if layout else axes[index]
        sub_plot.imshow(image_np)
        sub_plot.axis('off')
        sub_plot.set_title(f'Img{index + 0}')
    plt.tight_layout()
    plt.show()

# 白色为观测值（所有时序），黑色为未观测值
def plot_observe(X, target_channel=-2, **kwargs):
    # observe = 255, else = 0
    merge = X[:, :, target_channel]
    merge = reduce(merge, 'b t h w -> b h w', 'max')
    merge = repeat(merge, 'b h w -> b 3 h w')
    plot_images(merge, **kwargs)


def plot_each_observe(X, target_channel=-2, **kwargs):
    # observe = 255, else = 0
    # merge = X[:, target_channel]
    merge = X
    merge = repeat(merge, 't 1 h w -> t 3 h w')
    plot_images(merge, **kwargs)


def broad_gray(mask):
    h, w = mask.shape[-2:]
    return np.broadcast_to(mask, (3, h, w)) * 255
