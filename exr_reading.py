import Imath
#import OpenEXR as exr
import pyexr
import numpy as np
import os
from PIL import Image  # 新增导入语句

'''def readEXR_onlydepth(filename):
    try:
        exrfile = exr.InputFile(filename)
        header = exrfile.header()
        dw = header['dataWindow']
        isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

        channelData = dict()

        for c in header['channels']:
            C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
            C = np.frombuffer(C, dtype=np.float32)
            C = np.reshape(C, isize)

            channelData[c] = C

        Y = None if 'R' not in header['channels'] else channelData['R']

        far_=4.
        near_=0.01
        x=1.0-far_/near_
        y=far_/near_
        z=x/far_
        w=y/far_
        #for i in range(dw.max.y - dw.min.y + 1):
        #    for j in range(dw.max.x - dw.min.x + 1):
        #        Y[i][j]= 1./(z*(1-Y[i][j])+w)

        # 使用向量化操作替代嵌套循环
        Y = 1. / (z * (1 - Y) + w)

        exrfile.close()  # 显式关闭文件
        return Y

    except Exception as e:
        print(f"Error reading EXR file {filename}: {e}")
        return None

#folder_path = '/root/MonoGS/datasets/endomapper/endomapper_seq1/depth'
folder_path = '/root/MonoGS/datasets/tum/rgbd_dataset_freiburg3_long_office_household/depth'
file_count = 0  # 初始化文件计数器

for filename in os.listdir(folder_path):
    if filename.endswith('.exr'):
        depth_path = os.path.join(folder_path, filename)
        depth_data = readEXR_onlydepth(depth_path)
        if depth_data is not None:
            # depth = depth_data.astype(np.float16)
            depth = depth_data.astype(np.float32)
            print(f"Successfully read {filename} and obtained depth data.")
        else:
            print(f"Failed to read {filename}.")
    else:
        depth_path = os.path.join(folder_path, filename)
        depth = np.array(Image.open(depth_path)) /5000

        file_count += 1  # 增加文件计数器
        if file_count >= 20:  # 当读取 20 个文件后停止
            break'''

def readEXR_onlydepth(filename):
    try:
        # 使用 pyexr 读取 EXR 文件
        exr_data = pyexr.read(filename)
        
        # 检查返回的数据类型
        if isinstance(exr_data, np.ndarray):
            # 如果返回的是多维数组，假设深度数据在第一个通道
            if exr_data.ndim == 3 and exr_data.shape[2] >= 3:   # 确保是多通道数据
                Y = exr_data[:, :, 2]  # 假设深度数据在第3个通道
            else:
                print(f"Error: Unexpected array shape in {filename}")
                return None
        else:
            print(f"Error: Unexpected data format in {filename}")
            return None

        # 深度转换参数（根据实际数据调整）
        far_ = 4.0
        near_ = 0.01
        x = 1.0 - far_ / near_
        y = far_ / near_
        z = x / far_
        w = y / far_
        Y = 1.0 / (z * (1 - Y) + w)

        return Y

    except Exception as e:
        print(f"Error reading EXR file {filename}: {e}")
        return None

# 处理两种数据集的统一逻辑
folder_path = '/root/MonoGS/datasets/endomapper/endomapper_seq1/depth'
file_count = 0

for filename in os.listdir(folder_path):
    if filename.endswith('.exr'):
        depth_path = os.path.join(folder_path, filename)
        depth_data = readEXR_onlydepth(depth_path)
        if depth_data is not None:
            depth = depth_data.astype(np.float32)
            print(f"Successfully processed EXR: {filename}")
        else:
            print(f"Failed to process EXR: {filename}")
    elif filename.endswith('.png'):
        depth_path = os.path.join(folder_path, filename)
        depth = np.array(Image.open(depth_path)) / 5000.0  # TUM数据集转换
        print(f"Successfully processed PNG: {filename}")
    
    file_count += 1
    if file_count >= 20:
        break
    