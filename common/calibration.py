# import cv2
# import numpy as np
# import glob
# # 初始化棋盘格角点在世界坐标系中的位置
# corner_x, corner_y = 9, 6  # 假设棋盘格是9x6
# objp = np.zeros((corner_x*corner_y, 3), np.float32)
# objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)
#
# # 存储所有图像的世界坐标和图像坐标
# objpoints = []
# imgpoints = []
# # 读取图像
# images = glob.glob('calibration_images/*.jpg')
#
# for img in images:
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)
#     if ret:
#         objpoints.append(objp)
#         imgpoints.append(corners)
#         # 绘制并显示角点
#         img = cv2.drawChessboardCorners(img, (7, 6), corners, ret)
#         cv2.imshow('img', img)
#         cv2.waitKey(500)
#
# ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# # 传感器尺寸
# sensor_width_mm = 36  # 假设
# sensor_height_mm = 24  # 假设
#
# # 图像分辨率
# resolution_x = 4000  # 假设
# resolution_y = 3000  # 假设
#
# # 计算每个像素的物理尺寸
# pixel_width_mm = sensor_width_mm / resolution_x
# pixel_height_mm = sensor_height_mm / resolution_y
#
# print("每个像素的物理宽度:", pixel_width_mm, "mm")
# print("每个像素的物理高度:", pixel_height_mm, "mm")

# # 假设的相机参数
# focal_length_mm = 50  # 焦距（毫米）
# sensor_width_mm = 7.0656  # 传感器宽度（毫米）
# sensor_width_pixels = 2048  # 传感器宽度（像素）
# object_distance = 1000  # 物体到相机的距离（毫米）
#
# # 计算焦距（像素单位）
# focal_length_pixels = focal_length_mm * (sensor_width_pixels / sensor_width_mm)
#
# # 计算一个像素代表的实际宽度和高度（毫米）
# pixel_width_mm = sensor_width_mm / sensor_width_pixels
# real_world_mm_per_pixel = pixel_width_mm * (object_distance / focal_length_pixels)
#
# print(f"每个像素代表的实际宽度（毫米）: {real_world_mm_per_pixel}")

def calculate_concentration(initial_concentration, original_concentration, iterations):
    volume_original = 0.03  # in L
    volume_total = 0.5      # in L
    concentration = initial_concentration

    for i in range(iterations):
        removed_mass = concentration * volume_original
        added_mass = original_concentration * volume_original
        remaining_mass = concentration * (volume_total - volume_original)
        new_mass = remaining_mass + added_mass
        concentration = new_mass / volume_total
        print(f"After {i+1} iteration(s), concentration: {concentration:.4f} g/L")

# Initial concentration after mixing
init_SS = 1.69 * 0.01  #第一次取的重量
init_volume = 0.52  #混合后的体积
initial_concentration = (init_SS + 0) / init_volume
original_concentration = 1.69  # g/L

calculate_concentration(initial_concentration, original_concentration, 10)