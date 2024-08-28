import cv2
import numpy as np

# 读取图像
image = cv2.imread('../../data/0827/1/Image_20240827143919937.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊去噪声
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
# 使用Canny边缘检测
edges = cv2.Canny(gray, 50, 150)

# 使用findContours找到边缘连通区域
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建标记 (所有元素初始化为0)
markers = np.zeros_like(gray, dtype=int)

# 填充轮廓来创建标记
for i, contour in enumerate(contours):
    cv2.drawContours(markers, contours, i, i+1, -1)  # 给每个轮廓一个不同的标记

# 增加一以避免0作为标记
markers = markers + 1

# 应用分水岭算法
cv2.watershed(image, markers)

# 生成分割后的图像
segmented_image = np.zeros_like(gray, dtype=np.uint8)
segmented_image[markers == -1] = 255  # 边界标记为白色

cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()