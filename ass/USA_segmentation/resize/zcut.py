import os
import shutil
import random

# 设置随机种子，保证可复现
random.seed(42)

# 定义原始文件夹路径
dirs = {
    'masks': './masks',
    'nrg': './NRG_images',
    'rgb': './RGB_images'
}

# 定义输出子文件夹
datasets = ['train', 'val']
split_ratio = 0.8  # 训练集比例

# 获取文件列表（以 masks 文件夹为准）
all_files = sorted([f for f in os.listdir(dirs['masks']) if os.path.isfile(os.path.join(dirs['masks'], f))])
random.shuffle(all_files)

# 计算分界点
split_index = int(len(all_files) * split_ratio)
train_files = set(all_files[:split_index])
val_files = set(all_files[split_index:])

# 为每个类型和数据集创建目标文件夹
for key, path in dirs.items():
    for ds in datasets:
        out_dir = os.path.join(path, ds)
        os.makedirs(out_dir, exist_ok=True)

# 移动文件并删除原始位置
for filename in all_files:
    subset = 'train' if filename in train_files else 'val'
    for key, src_dir in dirs.items():
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(src_dir, subset, filename)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"警告: 未找到文件 {src_path}")

print("数据集划分完成！")
