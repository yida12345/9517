import os

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
plt.rcParams['font.size'] = 15

def plot_four_quadrants(rgb_gt, nrg_gt, rgb_pred, nrg_pred,
                        titles=("RGB_old","RGB_new","mask_old","mask_new")):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    imgs = [[rgb_gt, nrg_gt], [rgb_pred, nrg_pred]]
    for i in range(2):
        for j in range(2):
            ax = axes[i][j]
            im = imgs[i][j]
            if im.ndim == 2:  # 二维掩膜
                ax.imshow(im, cmap='gray')  # ← 指定灰度
            else:
                ax.imshow(im)
            # ax.imshow(imgs[i][j])
            ax.set_title(titles[i*2 + j], fontsize=20)
            # ax.axis("off")
            ax.axis("on")
            # 设置坐标轴范围（像素坐标）
            h, w = im.shape[:2]
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)  # 因为 origin='upper'，y 轴 0 在上方
            # 选择合适的刻度间隔，比如每隔 w/4 或 h/4 刻度
            xticks = np.linspace(0, w, 5)
            yticks = np.linspace(0, h, 5)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    base = r"D:\workspace\unsw\9517\ass\USA_segmentation"
    name = "ar037_2019_n_06_04_0.png"
    rgb   = np.array(Image.open(os.path.join(base, "RGB_images", "RGB_"+name)).convert("RGB"))
    rgb_  = np.array(Image.open(os.path.join(base, r"resize\RGB_images\train", name)).convert("RGB"))
    mask  = np.array(Image.open(os.path.join(base, "masks", "mask_"+name)).convert("L"))
    mask_ = np.array(Image.open(os.path.join(base, r"resize\masks\train", name)).convert("L"))
    # mask_ = (mask_ * 255).astype(np.int8)
    mask_non_black = mask_ != 0  # 任何不是 0 的都算“非黑”
    mask_[mask_non_black] = 255
    plot_four_quadrants(rgb, rgb_, mask, mask_)