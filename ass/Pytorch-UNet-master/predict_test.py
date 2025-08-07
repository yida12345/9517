import random

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 仅临时应急用
# 可选：减少线程数，避免过度并行（不解决重复加载，但有时更稳一些）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import pandas as pd
import time

import argparse
import logging
import os
import torch

import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from utils.data_loading import BasicDataset, compute_ndvi
from unet import UNet
from utils.utils import plot_img_and_mask
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"
plt.style.use('default')


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
    net = UNet(n_channels=n, n_classes=1, bilinear=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    state_dict = torch.load(dir, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    files = os.listdir(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\RGB_images\val")
    iou = 0
    start_time = time.time()
    for i in files:
        rgb_img = Image.open(os.path.join(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\RGB_images\val",i))
        nrg_img = Image.open(os.path.join(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\NRG_images\val",i))
        mask    = Image.open(os.path.join(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\masks\val",i))
        iou += predict_one(net, rgb_img, nrg_img, mask, device, cl)
    end_time = time.time()
    return [round(iou/len(files),4), round(end_time - start_time,2)]


# def overlay_and_get_vis(rgb_array, nrg_array, mask_array,
#                         color_rgb=(255, 0, 0), color_nrg=(0, 255, 255), alpha=0.7):
#     """
#     rgb_array: HxWx3 uint8
#     nrg_array: HxWx3 uint8
#     mask_array: HxW 或 HxWx1  二值（0/1）或布尔
#     """
#
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     # 1) 规范掩膜到 HxW bool
#     mask = mask_array
#     if mask.ndim == 3:
#         mask = mask[..., 0]
#     mask_bool = mask > 0  # shape (H, W), dtype=bool
#
#     # 2) 扩到三通道用于广播
#     mask3 = mask_bool[:, :, None]  # shape (H, W, 1)
#
#     # 3) overlay 函数：用 where 做混合
#     def overlay(image, mask3, color):
#         """
#         image: HxWx3 float32
#         mask3: HxWx1 bool
#         color: tuple of 3 floats
#         """
#         img = image.astype(np.float32)
#         clr = np.array(color, dtype=np.float32)[None, None, :]  # shape (1,1,3)
#         # 对掩膜为 True 的位置混合，否则保留原值
#         out = np.where(mask3,
#                        img * (1 - alpha) + clr * alpha,
#                        img)
#         return np.clip(out, 0, 255).astype(np.uint8)
#
#     rgb_vis = overlay(rgb_array, mask3, color_rgb)
#     nrg_vis = overlay(nrg_array, mask3, color_nrg)
#
#     # 显示
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     axes[0].imshow(rgb_vis)
#     axes[0].set_title("RGB + Mask Overlay")
#     axes[0].axis("off")
#     axes[1].imshow(nrg_vis)
#     axes[1].set_title("NRG + Mask Overlay")
#     axes[1].axis("off")
#     plt.tight_layout()
#     plt.show()
#
# def pltone():
#     net = UNet(n_channels=5, n_classes=1, bilinear=False)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     net.to(device=device)
#     state_dict = torch.load(r"E:\Downloads\BaiduNetdiskDownload\unet\[r, g, b, n, ndvi]-wubeilv\[r, g, b, n, ndvi]-wubeilv_checkpoint_epoch100.pth", map_location=device)
#     mask_values = state_dict.pop('mask_values', [0, 1])
#     net.load_state_dict(state_dict)
#     # files = os.listdir(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\RGB_images\val")
#     n = name
#     rgb = np.array(Image.open(os.path.join(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\RGB_images\val",n)).convert("RGB"))
#     nrg = np.array(Image.open(os.path.join(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\NRG_images\val",n)).convert("RGB"))
#     mask = np.array(Image.open(os.path.join(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\masks\val",n)).convert("L"))
#     overlay_and_get_vis(rgb, nrg, mask)
#     net.eval()
#
#     img_ = BasicDataset.preprocess(None, Image.open(os.path.join(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\RGB_images\val",n)), 0.5, is_mask=False)
#     nrg_ = BasicDataset.preprocess(None, Image.open(os.path.join(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\NRG_images\val",n)), 0.5, is_mask=False)
#     mask_ = Image.open(os.path.join(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\masks\val",n))
#     # nrg = BasicDataset.preprocess(None, mask_, 0.5, is_mask=True)
#     # Compute NDVI and extract green channel
#     ndvi = compute_ndvi(img_, nrg_)
#     r_chan = img_[0:1, ...]  # Green channel
#     g_chan = img_[1:2, ...]  # Green channel
#     b_chan = img_[2:3, ...]  # Green channel
#     nir_chan = nrg_[0:1, ...]
#     use = [r_chan, g_chan, b_chan, nir_chan, ndvi]
#     img_stack = np.vstack(use)  # shape (2, H, W)
#     img_tensor = torch.from_numpy(img_stack).unsqueeze(0)
#     img_tensor = img_tensor.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
#
#     with torch.no_grad():
#         output = net(img_tensor).cpu()
#         output = F.interpolate(output, (mask_.size[1], mask_.size[0]), mode='bilinear')
#         out = torch.sigmoid(output) > 0.5
#     # mask_ = (np.array(mask) > 0).astype(np.int8)
#     out = out[0].long().squeeze().numpy()
#     overlay_and_get_vis(rgb, nrg, out)


def overlay_and_get_vis(rgb_array, nrg_array, mask_array,
                        color_rgb=(255, 0, 0), color_nrg=(0, 0, 255),
                        alpha=0.7, return_img=False, show=True):
    """
    支持 channel-first 或 channel-last 输入，返回并/或显示叠加结果。
    Args:
      rgb_array: HxWx3 或 3xHxW 原始 RGB 图
      nrg_array: HxWx3 或 3xHxW NRG 假彩色图
      mask_array: HxW 或 HxWx1 二值掩膜
      return_img: 是否返回叠加后的 (rgb_vis, nrg_vis)
      show: 是否 plt.show()
    """
    # 1) 确保是 channel-last: (H,W,3)
    def to_chlast(img):
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[0] == 3:
            return np.transpose(img, (1, 2, 0))
        return img

    rgb = to_chlast(rgb_array)
    nrg = to_chlast(nrg_array)
    mask = mask_array
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask3 = (mask > 0)[..., None]  # (H,W,1)

    # 2) 定义 overlay
    def overlay(image, mask3, color):
        img = image.astype(np.float32)
        clr = np.array(color, dtype=np.float32)[None, None, :]
        out = np.where(mask3,
                       img * (1 - alpha) + clr * alpha,
                       img)
        return np.clip(out, 0, 255).astype(np.uint8)

    rgb_vis = overlay(rgb, mask3, color_rgb)
    nrg_vis = overlay(nrg, mask3, color_nrg)

    # 3) 可视化
    if show:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(rgb_vis)
        axes[0].set_title("RGB + Mask Overlay")
        axes[0].axis("off")
        axes[1].imshow(nrg_vis)
        axes[1].set_title("NRG + Mask Overlay")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    if return_img:
        return rgb_vis, nrg_vis


def plot_four_quadrants(rgb_gt, nrg_gt, rgb_pred, nrg_pred,
                        titles=("RGB+Ture","NRG+Ture","RGB+Pred","NRG+Pred")):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    imgs = [[rgb_gt, nrg_gt], [rgb_pred, nrg_pred]]
    for i in range(2):
        for j in range(2):
            ax = axes[i][j]
            ax.imshow(imgs[i][j])
            ax.set_title(titles[i*2 + j])
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def pltone(name):
    # 加载模型
    net = UNet(n_channels=5, n_classes=1, bilinear=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    state_dict = torch.load(
        r"E:\Downloads\BaiduNetdiskDownload\unet\[r, g, b, n, ndvi]-No magnification\[r, g, b, n, ndvi]-wubeilv_checkpoint_epoch100.pth",
        map_location=device)
    _ = state_dict.pop('mask_values', [0,1])
    net.load_state_dict(state_dict)
    net.eval()

    # 路径设置
    base = r"D:\workspace\unsw\9517\ass\USA_segmentation\resize"
    rgb_path = os.path.join(base, "RGB_images/val", name)
    nrg_path = os.path.join(base, "NRG_images/val", name)
    mask_path = os.path.join(base, "masks/val", name)

    # 读取原始图与掩膜
    rgb_array = np.array(Image.open(rgb_path).convert("RGB"))
    nrg_array = np.array(Image.open(nrg_path).convert("RGB"))
    mask_array = np.array(Image.open(mask_path).convert("L"))

    # 真实掩膜叠加可视化
    rgb_vis_gt, nrg_vis_gt = overlay_and_get_vis(
        rgb_array, nrg_array, mask_array,
        return_img=True, show=False)

    # Preprocessing & Model Prediction
    img_pt = BasicDataset.preprocess(None, Image.open(rgb_path), 0.5, is_mask=False)
    nrg_pt = BasicDataset.preprocess(None, Image.open(nrg_path), 0.5, is_mask=False)
    ndvi = compute_ndvi(img_pt, nrg_pt)
    chans = [img_pt[0:1], img_pt[1:2], img_pt[2:3], nrg_pt[0:1], ndvi]
    inp = torch.from_numpy(np.vstack(chans)).unsqueeze(0)
    inp = inp.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
    with torch.no_grad():
        out = net(inp).cpu()
        out = F.interpolate(out, (mask_array.shape[0], mask_array.shape[1]), mode='bilinear')
        pred_mask = (torch.sigmoid(out) > 0.5)[0, 0].numpy().astype(np.uint8)

    # Visualization of predicted mask overlay
    rgb_vis_pred, nrg_vis_pred = overlay_and_get_vis(
        rgb_array, nrg_array, pred_mask,
        return_img=True, show=False)

    # Four-grid display
    plot_four_quadrants(
        rgb_vis_gt, nrg_vis_gt,
        rgb_vis_pred, nrg_vis_pred,
        titles=("RGB + Ture Mask", "NRG + Ture Mask",
                "RGB + Pred Mask", "NRG + Pred Mask")
    )


if __name__ == '__main__':
    df = pd.DataFrame(columns=['model', 'IoU', 'Time'])
    df.loc[len(df)] = ["g,ndvi-No magnification"] + test(r"E:\Downloads\BaiduNetdiskDownload\unet\[g,ndvi]-No magnification\[g,ndvi]-wubeilv_checkpoint_epoch100.pth",2,1)
    df.loc[len(df)] = ["r, g, b, n, ndvi-2.5 times"] + test(r"E:\Downloads\BaiduNetdiskDownload\unet\[r, g, b, n, ndvi]-2.5 times\[r, g, b, n, ndvi]-2.5bei_checkpoint_epoch100.pth",5,2)
    df.loc[len(df)] = ["r, g, b, n, ndvi-No magnification"] + test(r"E:\Downloads\BaiduNetdiskDownload\unet\[r, g, b, n, ndvi]-No magnification\[r, g, b, n, ndvi]-wubeilv_checkpoint_epoch100.pth",5,3)
    df.loc[len(df)] = ["r, g, b, n-No magnification"] + test(r"E:\Downloads\BaiduNetdiskDownload\unet\[r, g, b, n]-No magnification\[r, g, b, n]-wubeilv_checkpoint_epoch100.pth",4,4)
    df.loc[len(df)] = ["r, g, n, ndvi-No magnification"] + test(r"E:\Downloads\BaiduNetdiskDownload\unet\[r, g, n, ndvi]-No magnification\[r, g, n, ndvi]-wubeilv_checkpoint_epoch100.pth",4,5)
    print(df)
    for i in random.sample(
            [f for f in os.listdir(r"D:\workspace\unsw\9517\ass\USA_segmentation\resize\RGB_images\val")], 10):
        pltone(i)