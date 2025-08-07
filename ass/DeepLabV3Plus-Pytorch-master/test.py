import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 仅临时应急用
# 可选：减少线程数，避免过度并行（不解决重复加载，但有时更稳一些）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from os.path import splitext, isfile, join

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import time
from torch import nn
import utils

import network


import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils import data

# from utils.data_loading import BasicDataset, compute_ndvi
# from unet import UNet
# from utils.utils import plot_img_and_mask

def compute_ndvi(rgb: np.ndarray, nrg: np.ndarray, eps=1e-6):
    # nrg[0] = NIR, nrg[1] = R
    nir = nrg[0].astype(np.float32)
    red = nrg[1].astype(np.float32)
    ndvi = (nir - red) / (nir + red + eps)
    # 归一化到 [0,1]
    ndvi = (ndvi + 1) / 2
    return ndvi[np.newaxis, ...]

def spectral_scale(img: np.ndarray, scale_range=(0.9, 1.1)):
    """
    img: numpy array, shape (C, H, W), float in [0,1]
    scale_range: 放缩因子的取值范围
    """
    C, H, W = img.shape
    for c in range(C):
        factor = np.random.uniform(*scale_range)
        img[c] = np.clip(img[c] * factor, 0.0, 1.0)
    return img

class BasicDataset(data.Dataset):
    def __init__(self, images_dir: str, dir_nrg: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.augment = False
        self.images_dir = Path(images_dir)
        self.nrg_dir = Path(dir_nrg)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.ids = [splitext(file)[0] for file in os.listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        self.mask_values = [1]

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            return (img > 0).astype(np.int8)
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        nrg_file = list(self.nrg_dir.glob(name + '.*'))
        mask = Image.open(mask_file[0])
        img  = Image.open(img_file[0])
        nrg  = Image.open(nrg_file[0])
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        nrg = self.preprocess(self.mask_values, nrg, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        ndvi = compute_ndvi(img, nrg)
        r_chan = img[0:1, ...]  # (1, H, W)
        g_chan = img[1:2, ...]  # (1, H, W)
        b_chan = img[2:3, ...]  # (1, H, W)
        nir_chan = nrg[0:1, ...]  # (1, H, W)

        use = [r_chan, g_chan, b_chan, nir_chan, ndvi]
        img = np.vstack(use)

        if self.augment:
            img = spectral_scale(img, scale_range=(0.9, 1.1))
            noise = np.random.normal(0, 0.02, size=img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
# denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5], std=[0.229, 0.224, 0.225, 0.1, 0.1])


def predict_one(model, img, nrg, mask_, device, cl):
    model.eval()

    img = BasicDataset.preprocess(None, img, 0.5, is_mask=False)
    nrg = BasicDataset.preprocess(None, nrg, 0.5, is_mask=False)
    # nrg = BasicDataset.preprocess(None, mask_, 0.5, is_mask=True)
    # Compute NDVI and extract green channel
    ndvi = compute_ndvi(img, nrg)
    r_chan = img[0:1, ...]  # Green channel
    g_chan = img[1:2, ...]  # Green channel
    b_chan = img[2:3, ...]  # Green channel
    nir_chan = nrg[0:1, ...]
    match cl:
        case 1:
            use = [g_chan, ndvi]
        case 2:
            use = [r_chan, g_chan, b_chan, nir_chan, ndvi]
        case 3:
            use = [r_chan, g_chan, b_chan, nir_chan, ndvi]
        case 4:
            use = [r_chan, g_chan, b_chan, nir_chan]
        case 5:
            use = [r_chan, g_chan, nir_chan, ndvi]
        case _:
            return "Unknown"
    # Stack channels as in training dataset
    img_stack = np.vstack(use)  # shape (2, H, W)
    mean = np.array([0.485, 0.456, 0.406, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225, 0.1, 0.1], dtype=np.float32)
    img_stack = (img_stack - mean[:, None, None]) / std[:, None, None]
    img_tensor = torch.from_numpy(img_stack).unsqueeze(0)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

    with torch.no_grad():
        output = model(img_tensor).cpu()
        output = F.interpolate(output, (mask_.size[1], mask_.size[0]), mode='bilinear')
        out = torch.sigmoid(output) > 0.5
    mask_ = (np.array(mask_) > 0).astype(np.int8)
    out = out[0].long().squeeze().numpy()

    intersection = np.logical_and(mask_, out).sum()
    union        = np.logical_or(mask_, out).sum()

    return intersection / union if union != 0 else 0.0

def test(dir, n, cl):
    net = network.modeling.__dict__['deeplabv3plus_resnet50'](num_classes=1, output_stride=8)
    utils.set_bn_momentum(net.backbone, momentum=0.01)

    old_conv = net.backbone.conv1
    new_conv = torch.nn.Conv2d(
        in_channels=5,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )
    net.backbone.conv1 = new_conv

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    checkpoint = torch.load(r'D:\workspace\unsw\9517\ass\DeepLabV3Plus-Pytorch-master\runs\Jul31_16-50-52_M16_lr0.01_bs16\checkpoints\best_deeplabv3plus_resnet50_voc_os16.pth', map_location=device, weights_only=False)
    # checkpoint = torch.load(r"D:\workspace\unsw\9517\ass\DeepLabV3Plus-Pytorch-master\runs\Jul31_16-17-24_M16_lr0.01_bs16\checkpoints\best_deeplabv3plus_resnet50_voc_os16.pth", map_location=device, weights_only=False)
    # mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(checkpoint["model_state"])
    net = nn.DataParallel(net)
    files = os.listdir(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\RGB_images\val")
    iou = 0
    start_time = time.time()
    for i in files:
        rgb_img = Image.open(os.path.join(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\RGB_images\val",i))
        nrg_img = Image.open(os.path.join(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\NRG_images\val",i))
        mask    = Image.open(os.path.join(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\masks\val",i))
        iou += predict_one(net, rgb_img, nrg_img, mask, device, cl)
    end_time = time.time()
    return [iou/len(files), end_time - start_time]


def overlay_and_get_vis(rgb_array, nrg_array, mask_array,
                        color_rgb=(255, 0, 0), color_nrg=(0, 0, 255),
                        alpha=0.9, return_img=False, show=True):
    """
    将二值掩膜分别叠加到原始 RGB 与 NRG 图上，得到可视化结果。
    支持 channel‐first (3,H,W) 或 channel‐last (H,W,3) 输入。

    Args:
      rgb_array: HxWx3 或 3xHxW 原始 RGB 图 (0–255 uint8 或 0–1 float)
      nrg_array: HxWx3 或 3xHxW 原始 NRG 图
      mask_array: HxW 或 HxWx1 二值掩膜 (0/1 或 True/False)
      color_rgb: 在 RGB 图上叠加的颜色
      color_nrg: 在 NRG 图上叠加的颜色
      alpha: 透明度 (0.0–1.0)
      return_img: 若 True，则返回 (rgb_vis, nrg_vis)
      show:       若 True，则调用 plt.show()

    Returns:
      (rgb_vis, nrg_vis) 或 None
    """
    # 把 channel-first 转到 channel-last
    def to_chlast(img):
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[0] == 3:
            return np.transpose(img, (1, 2, 0))
        return img
    rgb = to_chlast(rgb_array).astype(np.float32)
    nrg = to_chlast(nrg_array).astype(np.float32)

    # 标准化掩膜到 (H,W)
    mask = mask_array
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_bool = mask > 0  # (H,W)
    mask3 = mask_bool[..., None]  # (H,W,1)

    # 叠加函数：np.where 保证广播安全
    def _overlay(img, mask3, color):
        clr = np.array(color, dtype=np.float32)[None, None, :]
        out = np.where(mask3,
                       img * (1 - alpha) + clr * alpha,
                       img)
        return np.clip(out, 0, 255).astype(np.uint8)

    rgb_vis = _overlay(rgb, mask3, color_rgb)
    nrg_vis = _overlay(nrg, mask3, color_nrg)

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(rgb_vis)
        axes[0].set_title("RGB + Mask")
        axes[0].axis("off")
        axes[1].imshow(nrg_vis)
        axes[1].set_title("NRG + Mask")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    if return_img:
        return rgb_vis, nrg_vis


def plot_four_quadrants(rgb_gt, nrg_gt, rgb_pred, nrg_pred,
                        titles=("RGB + GT", "NRG + GT", "RGB + Pred", "NRG + Pred")):
    """
    把 4 张叠加图按 2×2 四宫格展示，方便对比真值 vs. 预测。

    Args:
      rgb_gt, nrg_gt   – 真实掩膜叠加的 RGB/NRG 可视化 (H,W,3)
      rgb_pred, nrg_pred – 预测掩膜叠加的 RGB/NRG 可视化
      titles – 四个子图的标题元组
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    imgs = [[rgb_gt, nrg_gt],
            [rgb_pred, nrg_pred]]
    for i in range(2):
        for j in range(2):
            axes[i, j].imshow(imgs[i][j])
            axes[i, j].set_title(titles[i*2 + j])
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()

def pltone(name):
    # 读原始
    base = r"D:\workspace\unsw\9517\ass\USA_segmentation\resize"
    rgb_path = os.path.join(base, "RGB_images/val", name)
    nrg_path = os.path.join(base, "NRG_images/val", name)
    mask_path = os.path.join(base, "masks/val", name)
    rgb_arr = np.array(Image.open(rgb_path).convert("RGB"))
    nrg_arr = np.array(Image.open(nrg_path).convert("RGB"))
    mask_gt = np.array(Image.open(mask_path).convert("L"))

    net = network.modeling.__dict__['deeplabv3plus_resnet50'](num_classes=1, output_stride=8)
    utils.set_bn_momentum(net.backbone, momentum=0.01)

    old_conv = net.backbone.conv1
    new_conv = torch.nn.Conv2d(
        in_channels=5,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )
    net.backbone.conv1 = new_conv

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    checkpoint = torch.load(
        r'D:\workspace\unsw\9517\ass\DeepLabV3Plus-Pytorch-master\runs\Jul31_16-50-52_M16_lr0.01_bs16\checkpoints\best_deeplabv3plus_resnet50_voc_os16.pth',
        map_location=device, weights_only=False)
    # checkpoint = torch.load(r"D:\workspace\unsw\9517\ass\DeepLabV3Plus-Pytorch-master\runs\Jul31_16-17-24_M16_lr0.01_bs16\checkpoints\best_deeplabv3plus_resnet50_voc_os16.pth", map_location=device, weights_only=False)
    # mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(checkpoint["model_state"])
    net = nn.DataParallel(net)

    net.eval()
    rgb_ = Image.open(os.path.join(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\RGB_images\val", name))
    nrg_ = Image.open(os.path.join(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\NRG_images\val", name))
    mask_ = Image.open(os.path.join(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\masks\val", name))

    img = BasicDataset.preprocess(None, rgb_, 0.5, is_mask=False)
    nrg = BasicDataset.preprocess(None, nrg_, 0.5, is_mask=False)
    # nrg = BasicDataset.preprocess(None, mask_, 0.5, is_mask=True)
    # Compute NDVI and extract green channel
    ndvi = compute_ndvi(img, nrg)
    r_chan = img[0:1, ...]  # Green channel
    g_chan = img[1:2, ...]  # Green channel
    b_chan = img[2:3, ...]  # Green channel
    nir_chan = nrg[0:1, ...]

    img_stack = np.vstack([r_chan, g_chan, b_chan, nir_chan, ndvi])  # shape (2, H, W)
    mean = np.array([0.485, 0.456, 0.406, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225, 0.1, 0.1], dtype=np.float32)
    img_stack = (img_stack - mean[:, None, None]) / std[:, None, None]
    img_tensor = torch.from_numpy(img_stack).unsqueeze(0)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

    with torch.no_grad():
        output = net(img_tensor).cpu()
        output = F.interpolate(output, (mask_.size[1], mask_.size[0]), mode='bilinear')
        out = torch.sigmoid(output) > 0.5
    mask_ = (np.array(mask_) > 0).astype(np.int8)
    out = out[0].long().squeeze().numpy()

    # Generate the truth value superposition graph
    rgb_gt_vis, nrg_gt_vis = overlay_and_get_vis(
        rgb_arr, nrg_arr, mask_gt,
        color_rgb=(255, 0, 0), color_nrg=(0, 0, 255),
        alpha=0.9, return_img=True, show=False)

    # Generate the prediction overlay map.
    rgb_pred_vis, nrg_pred_vis = overlay_and_get_vis(
        rgb_arr, nrg_arr, out,
        color_rgb=(255, 0, 0), color_nrg=(0, 0, 255),
        alpha=0.9, return_img=True, show=False)

    # Four-square comparison
    plot_four_quadrants(
        rgb_gt_vis, nrg_gt_vis,
        rgb_pred_vis, nrg_pred_vis
    )

if __name__ == '__main__':
    t = test("","",3)
    print(f"Iou: {t[0]}, Time: {t[1]}s")
    # for i in random.sample(
    #         [f for f in os.listdir(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\RGB_images\val")], 10):
    #     pltone(i)

    # pltone("mo097_2018_n_07_11_0.png")