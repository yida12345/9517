import os
from PIL import Image
import numpy as np

def calc_white_black_ratio_from_array(arr):
    total = arr.size
    white = np.count_nonzero(arr == 255)
    black = np.count_nonzero(arr == 0)
    return white / total, black / total

def batch_calc(input_mask_dir, summary=False):
    """
    遍历文件夹下所有掩码，计算白/黑比例：
    - summary=False: 输出每张掩码的比例
    - summary=True: 输出所有掩码像素合并后的总体比例
    """
    files = [f for f in os.listdir(input_mask_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    if summary:
        total_white = total_black = total_pixels = 0
        for fname in files:
            arr = np.array(Image.open(os.path.join(input_mask_dir, fname)).convert('L'))
            total_pixels += arr.size
            total_white  += np.count_nonzero(arr <= 255)
            total_black  += np.count_nonzero(arr == 0)
        print("【汇总】")
        print(f"白像素总比例: {total_white/total_pixels:.4f}")
        print(f"黑像素总比例: {total_black/total_pixels:.4f}")
    else:
        for fname in files:
            arr = np.array(Image.open(os.path.join(input_mask_dir, fname)).convert('L'))
            w_ratio, b_ratio = calc_white_black_ratio_from_array(arr)
            print(f"{fname} -> 白: {w_ratio:.4f}, 黑: {b_ratio:.4f}")

if __name__ == "__main__":
    mask_dir = "./resize/masks"
    # 单独查看每张
    # batch_calc(mask_dir, summary=False)
    # 或者查看汇总
    batch_calc(mask_dir, summary=True)
