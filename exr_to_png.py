import os
import cv2
import numpy as np
import pyexr

def load_depth_image(exr_path):
    # 使用 pyexr 读取 EXR 文件
    depth = pyexr.read(exr_path)
    # 假设深度信息在 R 通道（索引为 0）
    depth = depth[:, :, 0]  # 提取 R 通道
    return depth

def convert_depth_to_metric(depth):
    # 尺度转换参数
    far_ = 4.0  # 远平面距离
    near_ = 0.01  # 近平面距离

    x = 1.0 - far_ / near_
    y = far_ / near_
    z = x / far_
    w = y / far_

    # 进行尺度转换
    metric_depth = 1.0 / (z * (1 - depth) + w)
    return metric_depth

def save_depth_as_png(depth, png_path):
    # 将深度值归一化到0-255范围
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = np.uint8(depth_normalized)

    # 保存为PNG文件
    cv2.imwrite(png_path, depth_normalized)

def process_exr_to_png(exr_folder, png_folder):
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    # 遍历文件夹中的所有 .exr 文件
    for exr_file in os.listdir(exr_folder):
        if exr_file.endswith('.exr'):
            exr_path = os.path.join(exr_folder, exr_file)
            png_path = os.path.join(png_folder, exr_file.replace('.exr', '.png'))

            print(f"Processing {exr_path}...")
            try:
                # 加载深度图像
                depth = load_depth_image(exr_path)
                print("Depth image loaded successfully.")

                # 转换为公制距离
                metric_depth = convert_depth_to_metric(depth)
                print("Depth converted to metric units.")

                # 保存为PNG文件
                save_depth_as_png(metric_depth, png_path)
                print(f"Saved {png_path}")
            except Exception as e:
                print(f"Error processing {exr_path}: {e}")

# 设置路径
exr_folder = '/root/MonoGS/datasets/endomapper/endomapper_seq1/depth'
png_folder = '/root/MonoGS/datasets/endomapper/endomapper_seq1/new_depth'

# 处理所有EXR文件
process_exr_to_png(exr_folder, png_folder)