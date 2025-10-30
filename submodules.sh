#!/bin/bash

set -e

# Step 1: Clone cotracker and r2d2 (与OneSLAM一致的commit版本)
mkdir -p submodules

# cotracker (OneSLAM指定commit)
if [ ! -d submodules/cotracker ]; then
    git clone https://github.com/facebookresearch/co-tracker.git submodules/cotracker
    cd submodules/cotracker
    git checkout 8d364031971f6b3efec945dd15c468a183e58212
    cd ../..
else
    echo "submodules/cotracker already exists, skipping clone."
fi

# r2d2 (OneSLAM用的是主分支)
if [ ! -d submodules/r2d2 ]; then
    git clone https://github.com/naver/r2d2.git submodules/r2d2
else
    echo "submodules/r2d2 already exists, skipping clone."
fi

echo "[INFO] cotracker and r2d2 downloaded."

# Step 2: 安装 cotracker 和 r2d2 的依赖
echo "[INFO] Installing pip dependencies for cotracker and r2d2 ..."

# 一些项目没有 requirements.txt，我们列出核心依赖
pip install einops yacs munch kornia imageio torchmetrics wget opencv-python scipy matplotlib tqdm pycocotools

# 可选：安装r2d2依赖
pip install -r submodules/r2d2/requirements.txt || true

# Step 3: 检查模型权重是否自动下载（如无则手动下载）
echo "[INFO] Checking for cotracker weights ..."
if [ ! -f submodules/cotracker/checkpoints/cotracker_stride_4_wind_8.pth ]; then
    echo "[INFO] Downloading cotracker weights ..."
    wget -P submodules/cotracker/checkpoints https://huggingface.co/andreasveit/cotracker/resolve/main/cotracker_stride_4_wind_8.pth
fi

echo "[SUCCESS] Submodules and dependencies are ready."