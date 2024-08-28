import os
import cv2
import numpy as np
from analysize_video import frame_diff_process

def select_roi(frame):
    roi = cv2.selectROI("Select ROI", frame)
    cv2.destroyWindow("Select ROI")
    return roi

def process_video(video_path, roi, output_path):
    x, y, w, h = roi
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        roi_frame = frame[y:y+h, x:x+w]
        out.write(roi_frame)

    cap.release()
    out.release()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_video2jpg(input_directory, output_directory):

    first_video = None
    for root, dirs, files in os.walk(input_directory):
        for file in files:

            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_directory)
                if relative_path in ["004", "010", "013", "020"]:
                    continue
                else:
                    output_root = os.path.join(output_directory, relative_path)
                    ensure_dir(output_root)
                    # if not first_video:
                    #     first_video = video_path
                    #     cap = cv2.VideoCapture(first_video)
                    #     success, frame = cap.read()
                    #     cap.release()
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_num = 0
                    new_name = file[:-4]
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # 保存每一帧为JPEG图片
                        new_jpg_path = os.path.join(output_root, f'{new_name}_frame_{frame_num}.jpg')
                        cv2.imwrite(new_jpg_path, frame)
                        frame_num += 1

                    cap.release()
                # if not ret:
                #     break
                #
                # output_path = os.path.join(output_root, f"ROI_{file}")
                # process_video(video_path, roi, output_path)


def process_directory(input_directory, output_directory):
    first_video = None
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_directory)
                output_root = os.path.join(output_directory, relative_path)
                ensure_dir(output_root)
                if not first_video:
                    first_video = video_path
                    cap = cv2.VideoCapture(first_video)
                    success, frame = cap.read()
                    cap.release()
                    roi = [844, 371, 382, 402]
                    # if success:
                    #     roi = select_roi(frame)
                    # else:
                    #     print(f"Failed to read first video in {root}. Skipping ROI selection.")
                    #     continue
                output_path = os.path.join(output_root, f"ROI_{file}")
                process_video(video_path, roi, output_path)

def process_jpg_directory(input_directory, output_directory):
    first_video = None
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_directory)
                output_root = os.path.join(output_directory, relative_path)
                ensure_dir(output_root)
                if not first_video:
                    first_video = video_path
                    cap = cv2.VideoCapture(first_video)
                    success, frame = cap.read()
                    cap.release()
                    roi = [844, 371, 382, 402]
                    # if success:
                    #     roi = select_roi(frame)
                    # else:
                    #     print(f"Failed to read first video in {root}. Skipping ROI selection.")
                    #     continue
                output_path = os.path.join(output_root, f"ROI_{file}")
                process_video(video_path, roi, output_path)

def process_diff_jpg_direcrtory(base_jpg_directory,output_jpg_diretory ):
    for label in os.listdir(base_jpg_directory):
        if label in ["0", "10", "12", "14", ]:
            continue
        else:
            output_root = os.path.join(output_jpg_diretory, label)
            image_folder = os.path.join(base_jpg_directory, label)
            frame_diff_process(image_folder, output_root)
# 使用方法
base_directory = '../data/test'  # 更改为你的一级目录路径
output_jpg_diretory = "../data/test_jpg"
process_video2jpg(base_directory, output_jpg_diretory)
# process_directory(base_directory, output_diretory)