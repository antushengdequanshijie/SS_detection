import cv2
import numpy as np
import os
# 创建一个全局变量来存储手动选取的目标物体区域
selected_regions = []
current_region = None
resized_image = None
output_dir = 'tran_data/3'  # 指定保存目录
os.makedirs(output_dir, exist_ok=True)  # 创建目录（如果不存在）
read_path = f"D:/data/test_data/3"
img_names = os.listdir(read_path)
num = 0
# 缩放比例
scale_percent = 0.40  # 缩放百分比

def comapre_cood(cood):
    temp_cood = cood[0]
    if cood[0]> cood[1]:
            cood[0] = cood[1]
            cood[1] = temp_cood
    return cood
# 鼠标事件回调函数
def mouse_callback(event, x, y, flags, param):
    global selected_regions, current_region

    if event == cv2.EVENT_LBUTTONDOWN:
        # 左键按下，开始选择区域
        current_region = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and current_region:
        # 鼠标移动，绘制选择框
        temp_image = resized_image.copy()
        cv2.rectangle(temp_image, current_region[0], (x, y), (0, 255, 0), 2)
        cv2.imshow('Select Regions', temp_image)
    elif event == cv2.EVENT_LBUTTONUP:
        # 左键释放，确定选择区域
        current_region.append((x, y))
        selected_regions.append(tuple(current_region))
        current_region = None

# 读取原始图像文件列表
# original_image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg']  # 原始图像文件列表
# 遍历每个原始图像文件
for img_name in img_names:
    img_path = os.path.join(read_path, img_name)

    # 读取原始图像
    original_image = cv2.imread(img_path)
    
    # 缩放图像
    width = int(original_image.shape[1] * scale_percent )
    height = int(original_image.shape[0] * scale_percent )
    dim = (width, height)
    resized_image = cv2.resize(original_image, dim, interpolation=cv2.INTER_AREA)
    
    # 创建一个缩放图像副本用于手动选择目标物体区域
    temp_image = resized_image.copy()
    
    # 显示缩放后的图像并设置鼠标事件回调函数
    cv2.imshow('Select Regions', temp_image)
    cv2.setMouseCallback('Select Regions', mouse_callback)
    
    # 等待用户选择目标物体区域
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存目标物体区域的截图，并保持与原始图像相同大小
    for i, region in enumerate(selected_regions):
        x1, y1 = region[0]
        x2, y2 = region[1]
        region_y = [int(y1 / scale_percent), int(y2 / scale_percent)]
        region_x = [int(x1 / scale_percent), int(x2 / scale_percent)]
        region_y = comapre_cood(region_y)
        region_x = comapre_cood(region_x)
        roi = original_image[region_y[0]:region_y[1], region_x[0]:region_x[1]]
        save_path = os.path.join(output_dir, f'{num}_region_{i+1}.jpg')
        try:
            # print(roi.shape)
            if roi.shape[0]<1:
                print(x1,y1,x2,y2, int(y1 / scale_percent), int(y2 / scale_percent)) 
                continue
            else:
                cv2.imwrite(save_path, roi)
        except cv2.error as e:
            print("Opencv error:", e)
    num = num + 1
    
    # 清空选取的区域列表，为下一个图像做准备
    selected_regions.clear()