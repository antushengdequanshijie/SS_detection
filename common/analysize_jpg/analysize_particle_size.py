import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
from PIL import Image
import matplotlib.pyplot as plt


def load_images_from_folder(folder):
    """ 从文件夹中加载所有图片 """
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".bmp", ".jpg")):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images


def process_images(images, reference_image):
    """ 处理图片：每张图片减去基准图片 """
    reference = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)
    processed_images = []
    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_img, reference)
        processed_images.append(diff)
    return processed_images


def save_images(images, folder):
    """ 保存处理后的图片 """
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(folder, f'processed_{i}.png'), img)


def analyze_images(images):
    """ 分析图片：计算亮度、颗粒数量和粒径分布 """
    brightnesses = []
    particle_sizes = []
    for img in images:
        brightness = np.mean(img)
        brightnesses.append(brightness)

        # 颗粒检测
        ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sizes = [cv2.contourArea(cnt) for cnt in contours]
        particle_sizes.extend(sizes)

    # 绘制颗粒大小分布图
    plt.hist(particle_sizes, bins=20, color='blue', edgecolor='black')
    plt.title('Particle Size Distribution')
    plt.xlabel('Size')
    plt.ylabel('Frequency')
    plt.show()

    return brightnesses, particle_sizes


def main(base_path, reference_image):
    """ 主函数 """
    labels = os.listdir(base_path)
    for label in labels:
        folder_path = os.path.join(base_path, label)
        images = load_images_from_folder(folder_path)
        processed_images = process_images(images, reference_image)
        save_images(processed_images, folder_path)
        brightnesses, particle_sizes = analyze_images(processed_images)
        print(f'Label: {label}, Average Brightness: {np.mean(brightnesses)}, Particle Count: {len(particle_sizes)}')


if __name__ == '__main__':
    main('../../data/0826', '../../data/0826/000/0.bmp')