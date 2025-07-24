# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# images = [cv2.imread(f'COMP9517_25T2_Lab1_Images/Castle{i:02}.jpg', cv2.IMREAD_GRAYSCALE) for i in range(1,10)]
#
# stack = np.stack(images, axis=0)
# avg_img = np.mean(stack, axis=0)
# plt.imshow(avg_img)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the 10 noisy images (assumed grayscale) into a list
images = []
for i in range(1, 11):
    img = cv2.imread(f"COMP9517_25T2_Lab1_Images/Castle{i:02}.jpg", cv2.IMREAD_GRAYSCALE)
    images.append(img.astype(np.float32))

# 2. Stack and compute per-pixel mean
stack = np.stack(images, axis=0)             # shape: (10, H, W)
avg_img = np.mean(stack, axis=0)           # shape: (H, W)

# 3. Display the averaged image
plt.figure(figsize=(6, 6))
plt.imshow(avg_img.astype(np.uint8), cmap='gray')
plt.title('Averaged Image')
plt.axis('off')
plt.show()