import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 读取图像（请替换为你的图像路径）
img = mpimg.imread(r'D:\workspace\unsw\9517\ass\USA_segmentation\RGB_images\RGB_ar037_2019_n_06_04_0.png')
nrg = mpimg.imread(r'D:\workspace\unsw\9517\ass\USA_segmentation\NRG_images\NRG_ar037_2019_n_06_04_0.png')

# 拆分通道
R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]
N_ = nrg[:, :, 0]
R_ = nrg[:, :, 1]
G_ = nrg[:, :, 2]

# 创建一个包含三个子图的画布
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 显示各通道
axes[0].imshow(R, cmap='gray')
axes[0].set_title('Red 通道')
axes[0].axis('off')

axes[1].imshow(G, cmap='gray')
axes[1].set_title('Green 通道')
axes[1].axis('off')

axes[2].imshow(B, cmap='gray')
axes[2].set_title('Blue 通道')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# 创建一个包含三个子图的画布
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 显示各通道
axes[0].imshow(N_, cmap='gray')
axes[0].set_title('N 通道')
axes[0].axis('off')

axes[1].imshow(R_, cmap='gray')
axes[1].set_title('R 通道')
axes[1].axis('off')

axes[2].imshow(G_, cmap='gray')
axes[2].set_title('G 通道')
axes[2].axis('off')

plt.tight_layout()
plt.show()
