import cv2
import os
import numpy as np

def frame2video(frames_folder, output_video_path):
    # 获取所有帧的文件名并排序
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])

    # 读取第一帧来获取图像尺寸
    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, layers = first_frame.shape

    # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

    for file in frame_files:
        frame = cv2.imread(os.path.join(frames_folder, file))
        out.write(frame)  # 写入帧到视频文件中

    # 释放资源
    out.release()
    print('视频已成功保存至:', output_video_path)

def thres_image(diff_folder, thresh_folder):
    # 如果结果文件夹不存在，则创建
    os.makedirs(thresh_folder, exist_ok= True)

    # 获取所有差分图像的文件名并排序
    diff_files = sorted([f for f in os.listdir(diff_folder) if f.endswith('.jpg')])

    for file in diff_files:
        # 读取差分图像
        diff_image = cv2.imread(os.path.join(diff_folder, file), cv2.IMREAD_GRAYSCALE)

        # 应用阈值处理来分割前景和背景
        _, thresh_image = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

        # 可选：进行形态学操作改善结果，例如膨胀和侵蚀
        kernel = np.ones((5, 5), np.uint8)
        thresh_image = cv2.dilate(thresh_image, kernel, iterations=1)
        thresh_image = cv2.erode(thresh_image, kernel, iterations=1)

        # 保存处理后的图像
        thresh_path = os.path.join(thresh_folder, f"thresh_{file}")
        cv2.imwrite(thresh_path, thresh_image)

    print('前景背景分割图像已保存。')

def video2frames(video_path, save_folder):
    os.makedirs(save_folder, exist_ok= True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 保存帧到文件
        frame_path = os.path.join(save_folder, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_number += 1

    # 释放视频文件
    cap.release()
    print('所有帧已保存。')
def frame_diff_process(frames_folder, diff_folder):
    # 如果差分图像文件夹不存在，则创建
    os.makedirs(diff_folder, exist_ok= True)

    # 获取所有帧的文件名并排序
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])

    # 读取第一帧作为初始帧
    previous_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]), cv2.IMREAD_GRAYSCALE)

    for i in range(1, len(frame_files)):
        # 读取下一帧
        current_frame = cv2.imread(os.path.join(frames_folder, frame_files[i]), cv2.IMREAD_GRAYSCALE)

        # 计算当前帧和前一帧之间的差异
        frame_diff = cv2.absdiff(current_frame, previous_frame)

        # 保存差异图像
        diff_path = os.path.join(diff_folder, f"diff_{i:04d}.jpg")
        cv2.imwrite(diff_path, frame_diff)

        # 更新前一帧
        previous_frame = current_frame

    print('帧差分图像已保存。')
def stabilize_video(input_path, output_path):
    #有效果但是慢，然后就是看不到颗粒了
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取视频帧的宽度和高度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 准备输出视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 读取第一帧
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # 变换矩阵的初始值
    transforms = np.zeros((2, 3), np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换到灰度
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算光流来得到运动向量
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # 计算累计的运动
        dx, dy = flow[..., 0].mean(), flow[..., 1].mean()

        # 更新变换矩阵
        transforms[0, 2] += dx
        transforms[1, 2] += dy

        # 应用仿射变换
        stabilized_frame = cv2.warpAffine(frame, transforms, (frame_width, frame_height))

        # 更新上一帧
        prev_gray = curr_gray

        # 写入稳定化的帧
        out.write(stabilized_frame)

    # 释放所有资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def video_stabilization(video_path):
    # 打开视频文件，效果不好
    cap = cv2.VideoCapture(video_path)

    # 获取第一帧图像
    ret, prev_frame = cap.read()

    # 创建视频输出对象
    out = cv2.VideoWriter('outpu1.avi',
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          cap.get(cv2.CAP_PROP_FPS),
                          (prev_frame.shape[1], prev_frame.shape[0]))

    while cap.isOpened():
        # 读取当前帧
        ret, cur_frame = cap.read()
        if not ret:
            break

        # 检测特征点并计算光流
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        feature_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, feature_points, None)

        # 计算帧间运动矢量
        motion_vectors = next_pts - feature_points

        # 估计相机的整体运动
        motion_x = np.mean(motion_vectors[:, 0, 1])
        motion_y = np.mean(motion_vectors[:, 0, 0])

        # 对当前帧进行校正
        M = np.float32([[1, 0, -motion_x], [0, 1, -motion_y]])
        stabilized_frame = cv2.warpAffine(cur_frame, M, (cur_frame.shape[1], cur_frame.shape[0]))

        # 将校正后的帧写入输出视频
        out.write(stabilized_frame)

        # 更新上一帧为当前帧
        prev_frame = stabilized_frame

    # 释放资源
    cap.release()
    out.release()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def count_SS_num(images_path):
    count = 0
    for image in os.listdir(images_path):
        if image.lower().endswith(("jpg", "png")):
            image_path = os.path.join(images_path, image)
            img = cv2.imread(image_path,0)
            ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sizes = [cv2.contourArea(cnt) for cnt in contours]
            print(len(sizes))
            count = count + 1
            if count > 10:
                break
    # particle_sizes.extend(sizes)
# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    input_videopath = '../data/0826'
    frame_folder = "../data/process_0826/0826_frames_folder"
    diff_folder = "../data/process_0826/0826_diff_folder"
    thresh_folder = "../data/process_0826/0826_thresh_folder"
    # output_video = "../data/0826_"
    count_SS_num("D:/work/algorithm/SS_detection/data/process_0826/0826_thresh_folder/003")
    for root, dirs, files in os.walk(input_videopath):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_videopath)
                frame_out_folder = os.path.join(frame_folder, relative_path)
                diff_out_folder = os.path.join(diff_folder, relative_path)
                thresh_out_folder = os.path.join(thresh_folder, relative_path)
                ensure_dir(frame_out_folder)
                ensure_dir(diff_out_folder)
                ensure_dir(thresh_out_folder)
                # video2frames(video_path, frame_out_folder)
                # frame_diff_process(frame_out_folder, diff_out_folder)
                # thres_image(diff_out_folder, thresh_out_folder)
    # frame2video(diff_folder, output_video)
    # 调用视频去抖动函数
    # video_stabilization('../1.avi')
    # 使用函数
    # stabilize_video('../1.avi', 'stabilized_video.avi')

