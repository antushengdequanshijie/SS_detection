import os
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def calculate_brightness(image_path):
    """ 计算图片的平均亮度 """
    img = Image.open(image_path).convert('L')  # 转换为灰度图
    numpy_img = np.array(img)
    return np.mean(numpy_img)

def plot_brightness_distribution(data):
    """ 绘制亮度分布图 """
    plt.figure(figsize=(10, 8))
    for label, brightnesses in data.items():
        plt.hist(brightnesses, bins=30, alpha=0.5, label=label)
    plt.title('Brightness Distribution by Folder')
    plt.xlabel('Brightness')
    plt.ylabel('Frequency')
    plt.legend(title='Folder Label')
    plt.grid(True)
    plt.show()

def main(directory):
    """ 处理目录中的所有图片 """
    brightness_data = {}
    # 遍历目录下的所有文件夹
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            brightness_data[folder] = []
            # 遍历文件夹内的所有图片文件
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    brightness = calculate_brightness(file_path)
                    brightness_data[folder].append(brightness)
    # 绘制亮度分布图
    plot_brightness_distribution(brightness_data)

if __name__ == '__main__':
    main("../data/0822_test")