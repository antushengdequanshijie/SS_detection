import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split #可以利用其来对数据集进行分割
from data_process import get_video_label, get_label_info
import psutil
from functools import wraps
from torchvision.io import read_video
import torchvision.transforms as T
def monitor_memory_usage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 ** 2  # 获取执行前的内存使用量，单位为MB
        result = func(*args, **kwargs)
        mem_after = process.memory_info().rss / 1024 ** 2  # 获取执行后的内存使用量，单位为MB
        print(f"{func.__name__} - Memory Before: {mem_before:.2f} MB, Memory After: {mem_after:.2f} MB, Memory Used: {mem_after - mem_before:.2f} MB")
        return result
    return wrapper

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_frames=18):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames # 期望的帧数
        self.samples = [] # 存储视频路径和标签

        # 遍历根目录下的所有文件夹
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir): # 确保是目录
                for video_file in os.listdir(label_dir):
                    if video_file.endswith('.avi'):
                        video_path = os.path.join(label_dir, video_file)
                        self.samples.append((video_path, torch.tensor([float(label)])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        # 尝试读取视频
        try:
            frames, _, _ = read_video(video_path)
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return None # 处理失败的视频返回 None

        # 检查帧数是否足够
        if frames.shape[0] < self.num_frames:
            # 如果帧数不足，用最后一帧填充
            pad_size = self.num_frames - frames.shape[0]
            last_frame = frames[-1].unsqueeze(0).repeat(pad_size, 1, 1, 1)
            frames = torch.cat([frames, last_frame], dim=0)
        elif frames.shape[0] > self.num_frames:
            # 如果帧数过多，随机选择 num_frames 帧
            indices = torch.linspace(0, frames.shape[0] - 1, self.num_frames).long()
            frames = frames[indices]

        # 确保形状正确
        frames = frames.permute(0, 3, 1, 2) # 从 (T, H, W, C) 转为 (T, C, H, W)
        if self.transform:
            frames = self.transform(frames)

        return frames, label
def custom_collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch)==0:
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)
def get_data(train_path):
    # 示例视频文件和标签
    transform = T.Compose([
        T.Resize((112, 112)),  # 调整图像大小
        T.ConvertImageDtype(torch.float32),  # 转换数据类型为float32
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 创建数据集对象
    video_dataset = VideoDataset(train_path, transform=transform)
    #
    # 创建数据加载器
    dataloader = DataLoader(video_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    return dataloader


import cv2
import os
def avi2jpg(input_folder, output_folder):

    # 检查输出文件夹是否存在，如果不存在就创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    count = 0
    for filename in os.listdir(input_folder):
        if filename.endswith(".avi"):
            output_label_name = os.path.join(output_folder, str(count))
            if not os.path.exists(output_label_name):
                os.makedirs(output_label_name)
            # 构建完整的文件路径
            video_path = os.path.join(input_folder, filename)
            # 读取视频文件
            cap = cv2.VideoCapture(video_path)

            # 初始化一个计数器用于图片命名
            frame_count = 0

            # 逐帧读取视频
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 构建输出图片的路径
                output_image_path = os.path.join(output_label_name, f"{os.path.splitext(filename)[0]}_frame_{frame_count}.jpg")
                # 保存图片
                cv2.imwrite(output_image_path, frame)
                frame_count += 1

            # 释放视频文件
            cap.release()
            count = count + 2
    print("转换完成！")

def split_video(video_path, output_folder, segment_length=1):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取视频文件
    count = 0
    for video_name in os.listdir(video_path):
        if video_name.endswith(".avi"):
            output_label_name = os.path.join(output_folder, str(count))
            if not os.path.exists(output_label_name):
                os.makedirs(output_label_name)

            cap = cv2.VideoCapture(os.path.join(video_path, video_name))
            fps = int(cap.get(cv2.CAP_PROP_FPS))  # 获取视频的帧率
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

            # 计算每个视频段的帧数
            frames_per_segment = fps * segment_length

            # 初始化视频分割
            current_segment = 0
            frame_count = 0

            # 为每个短视频创建一个视频写入对象
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 定义编码格式
            out = cv2.VideoWriter(os.path.join(output_label_name, f'segment_{current_segment}.avi'), fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 写入帧到当前视频文件
                out.write(frame)
                frame_count += 1

                # 如果当前视频段的帧数达到了设定值，开始新的视频段
                if frame_count == frames_per_segment:
                    out.release()  # 关闭当前视频文件
                    current_segment += 1
                    frame_count = 0
                    out = cv2.VideoWriter(os.path.join(output_label_name, f'segment_{current_segment}.avi'), fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

            # 释放资源
            cap.release()
            out.release()
            count = count + 2
import pandas as pd
if __name__ == '__main__':
    video_path = "../../../train_data/MV-CS050-10GC (DA3215281)"
    output_jpg = "../../../train_data/train_jpg"
    output_video = "../../../train_data/train_video"
    data = get_data(output_video)
    # avi2jpg(video_path, output_jpg)
    # split_video(video_path, output_video)
    # 重新加载数据，这次指定表头在第二行
    # 重新加载数据，指定第三行为表头
