import os
import shutil
import random

# 原始数据集目录
original_dataset_dir = '../data/jpg_500'

# 划分后的数据集目录
base_dir = '../data/splitted_data_500'
os.makedirs(base_dir, exist_ok=True)

# 训练集和测试集目录
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# 数据集中的类别列表
classes = os.listdir(original_dataset_dir)

# 每个类别创建训练集和测试集文件夹
for class_name in classes:
    # 类别训练集和测试集目录
    train_class_dir = os.path.join(train_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(test_class_dir, exist_ok=True)

    # 获取类别下的所有图片文件名列表
    class_images = os.listdir(os.path.join(original_dataset_dir, class_name))

    # 随机打乱图片列表
    random.shuffle(class_images)

    # 将80%的图片复制到训练集文件夹，20%的图片复制到测试集文件夹
    split_index = int(0.8 * len(class_images))
    train_images = class_images[:split_index]
    test_images = class_images[split_index:]

    # 将图片复制到对应的训练集和测试集文件夹中
    for img_name in train_images:
        src = os.path.join(original_dataset_dir, class_name, img_name)
        dst = os.path.join(train_class_dir, img_name)
        shutil.copyfile(src, dst)

    for img_name in test_images:
        src = os.path.join(original_dataset_dir, class_name, img_name)
        dst = os.path.join(test_class_dir, img_name)
        shutil.copyfile(src, dst)

print("数据集划分完成。训练集保存在:", train_dir)
print("测试集保存在:", test_dir)