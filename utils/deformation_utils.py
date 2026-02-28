"""体内可形变SLAM的形变约束工具模块。

本模块实现了面向软组织形变建模的物理约束，设计原则：
- ARAP / AMAP / 体积保持 / 法向一致性损失完全可微分，可直接加入高斯点优化的损失函数
- 自适应刚度、形变势能、自适应损失权重运行在 torch.no_grad() 上下文中，无副作用
- 所有函数只依赖 torch，无额外第三方依赖

参考文献：
  [1] Sumner et al., "Embedded deformation for shape manipulation", SIGGRAPH 2007
  [2] Sorkine & Alexa, "As-rigid-as-possible surface modeling", SGP 2007
  [3] Holzapfel, "Nonlinear solid mechanics", Wiley 2000
  [4] GuoAnn et al., "Learning-based Isometric NRSfM", ECCV 2024
"""

import math

import torch


# ---------------------------------------------------------------------------
# 1. ARAP（As-Rigid-As-Possible）形变正则化损失
# ---------------------------------------------------------------------------

def compute_arap_loss(current_positions, reference_positions, knn_index):
    """ARAP 边长应变正则化损失（可微分）。

    对每条锚点-近邻弹簧边 (i, j)，计算当前边长与参考边长之间的**工程应变**，
    以均方应变作为损失。这是 ARAP 能量的高效近似：
    保持局部边长不变等价于局部刚体约束，在允许全局非刚性形变的同时维持
    空间连续性与局部刚性。

    能量形式：
        E_ARAP = mean_{i,j} [ ( ||p_j' - p_i'|| - ||p_j - p_i|| ) / ||p_j - p_i|| ]^2

    关键性质：
    - 完全可微分（通过 current_positions）
    - 无量纲（归一化为应变，不受绝对尺度影响）
    - 对称：只对每条边计算一次，避免重复惩罚

    Args:
        current_positions:   [M, 3] Tensor, 当前锚点位置（在优化器计算图中）
        reference_positions: [M, 3] Tensor, 参考（初始）锚点位置（已 detach）
        knn_index:           [M, K] long Tensor, 每个锚点的 K 个最近邻索引

    Returns:
        scalar Tensor: ARAP 均方应变损失
    """
    M, K = knn_index.shape

    ref_j = reference_positions[knn_index]                                   # [M, K, 3]
    ref_len = torch.norm(ref_j - reference_positions.unsqueeze(1), dim=2).clamp(min=1e-6)  # [M, K]

    cur_j = current_positions[knn_index]                                     # [M, K, 3]
    cur_len = torch.norm(cur_j - current_positions.unsqueeze(1), dim=2)      # [M, K]

    strain = (cur_len - ref_len) / ref_len                      # [M, K]
    return (strain ** 2).mean()


# ---------------------------------------------------------------------------
# 2. 体积保持损失（软组织近似不可压缩性）
# ---------------------------------------------------------------------------

def compute_volume_preservation_loss(current_positions, reference_positions, knn_index):
    """体积保持正则化损失（可微分）。

    软组织（如消化道壁、器官表面）在生物力学上近似不可压缩（泊松比 ≈ 0.5）。
    这里使用邻域**回转半径平方**（Radius of Gyration squared, RG²）的变化量
    作为局部体积变化的代理指标：

        RG²_i = mean_j || p_j - centroid(N_i) ||^2

    若邻域体积保持不变，则 RG² 应保持不变。

    能量形式：
        E_vol = mean_i [ ( RG²_i' - RG²_i ) / RG²_i ]^2

    关键性质：
    - 可微分（通过 current_positions）
    - 无量纲，不受绝对尺度影响
    - 比直接计算四面体体积的计算量小 O(K) 倍

    Args:
        current_positions:   [M, 3] Tensor, 当前锚点位置（可微）
        reference_positions: [M, 3] Tensor, 参考锚点位置（已 detach）
        knn_index:           [M, K] long Tensor

    Returns:
        scalar Tensor: 体积保持损失
    """
    # 参考构型回转半径平方
    ref_nb = reference_positions[knn_index]                         # [M, K, 3]
    ref_centroid = ref_nb.mean(dim=1, keepdim=True)                 # [M, 1, 3]
    ref_rg2 = ((ref_nb - ref_centroid) ** 2).sum(dim=2).mean(dim=1).clamp(min=1e-8)  # [M]

    # 当前构型回转半径平方
    cur_nb = current_positions[knn_index]                           # [M, K, 3]
    cur_centroid = cur_nb.mean(dim=1, keepdim=True)                 # [M, 1, 3]
    cur_rg2 = ((cur_nb - cur_centroid) ** 2).sum(dim=2).mean(dim=1).clamp(min=1e-8)  # [M]

    volume_strain = (cur_rg2 - ref_rg2) / ref_rg2                  # [M]
    return (volume_strain ** 2).mean()


# ---------------------------------------------------------------------------
# 3. 形变势能（仅用于监控 / 评估）
# ---------------------------------------------------------------------------

def compute_deformation_energy(current_positions, reference_positions, knn_index, spring_k):
    """计算弹簧系统的总形变势能（非可微，仅用于监控）。

    E = 0.5 * sum_{i,j} k_ij * ( ||p_j' - p_i'|| - ||p_j - p_i|| )^2

    Args:
        current_positions:   [M, 3] 当前锚点位置
        reference_positions: [M, 3] 参考锚点位置
        knn_index:           [M, K] KNN 索引
        spring_k:            [M, K] 每条弹簧的刚度系数

    Returns:
        float: 总形变势能（标量）
    """
    with torch.no_grad():
        ref_j = reference_positions[knn_index]                              # [M, K, 3]
        ref_len = torch.norm(ref_j - reference_positions.unsqueeze(1), dim=2)  # [M, K]
        cur_j = current_positions[knn_index]                               # [M, K, 3]
        cur_len = torch.norm(cur_j - current_positions.unsqueeze(1), dim=2)   # [M, K]
        delta_len = cur_len - ref_len                                       # [M, K]
        return (0.5 * spring_k * delta_len ** 2).sum().item()


# ---------------------------------------------------------------------------
# 4. AMAP（As-Maximal-As-Possible）形变约束损失
# ---------------------------------------------------------------------------

def compute_amap_loss(current_positions, knn_index, max_edge_lengths):
    """AMAP（As-Maximal-As-Possible）形变约束损失（可微分）。

    基于 GuoAnn 等人在 Learning-based Isometric NRSfM（ECCV 2024）中提出的
    AMAP 启发式约束：

    对于等距形变，拉伸上界是测地距离（不可延伸性约束），因此松弛后的最优解
    趋于逼近可行域边界。AMAP 损失将当前帧每条弹簧边的长度向**历史观测最大
    边长**驱动，显式抑制"表面收缩"退化解——这是内镜 SLAM 场景中弹簧点云
    收敛过快时的常见退化模式。

    能量形式（与 ARAP 互补，两者协同约束形变）：
        L_AMAP = mean_{i,j} ( ||z_j' - z_i'|| - max_k ||z_j^k - z_i^k|| )^2

    其中 max_k 是训练过程中观测到的历史最大边长（已 detach，不参与梯度）。

    与 ARAP 的区别：
    - ARAP：惩罚任何边长变化（保刚性）
    - AMAP：驱动边长趋向历史最大值（防收缩）

    Args:
        current_positions: [M, 3] Tensor, 当前帧锚点位置（在优化器计算图中）
        knn_index:         [M, K] long Tensor, 每个锚点的K个最近邻索引
        max_edge_lengths:  [M, K] Tensor, 历史最大边长（已 detach，不参与梯度）

    Returns:
        scalar Tensor: AMAP 均方损失
    """
    M, K = knn_index.shape
    cur_j = current_positions[knn_index]                              # [M, K, 3]
    cur_len = torch.norm(cur_j - current_positions.unsqueeze(1), dim=2)  # [M, K]
    # max_edge_lengths 已 detach，梯度只流向 current_positions
    return ((cur_len - max_edge_lengths) ** 2).mean()


# ---------------------------------------------------------------------------
# 5. 自适应损失权重 ω_l（NRSfM 论文公式(10)）
# ---------------------------------------------------------------------------

def compute_adaptive_loss_weight(n_points, c1=100, c2=300):
    """基于稀疏点数量计算自适应形变损失权重 ω_l（纯标量函数，非可微）。

    来源：GuoAnn et al., "Learning-based Isometric NRSfM", ECCV 2024, Eq.(10)。

    物理意义：形变约束的可靠性依赖于点云密度：
    - 点数极少（< c1）：覆盖不足，约束噪声大 → 降低权重
    - 点数适中（c1 ~ c2）：约束可靠性正常 → 权重 = 1
    - 点数充足（>= c2）：覆盖充分，约束更可信 → 权重 > 1（最大约 3）

    注：本函数输入为整数 n_points，返回 Python float，作为损失函数的
    无梯度乘法系数使用，不需要对 n_points 可微。

    公式：
        ω_l = (N_p / c1)^2           if N_p < c1
            = 1                        if c1 ≤ N_p < c2
            = 3 - 2*exp(-(N_p-c2)/c1) if N_p ≥ c2

    Args:
        n_points: int，当前锚点（或高斯点）数量
        c1: int, 点数下阈值（默认 100）
        c2: int, 点数上阈值（默认 300）

    Returns:
        float: 无量纲自适应权重
    """
    if n_points < c1:
        return (n_points / c1) ** 2
    elif n_points < c2:
        return 1.0
    else:
        return 3.0 - 2.0 * math.exp(-(n_points - c2) / c1)


# ---------------------------------------------------------------------------
# 6. 表面法向一致性损失（局部 PCA 法向场平滑）
# ---------------------------------------------------------------------------

def compute_normal_consistency_loss(current_positions, knn_index):
    """表面法向一致性正则化损失（可微分）。

    原理：
    将形变表面建模为黎曼流形。局部法向量通过对 K 近邻点云做 PCA 估计，
    对应 NRSfM 论文中度量张量不变性的离散近似：
    - 度量张量不变 ⟺ 局部曲面形状（含法向）在等距形变下不变
    - 对稀疏点云，局部 PCA 法向是可微分、无参数的曲面法向估计器
    相邻锚点之间的法向夹角惩罚即为**离散第一基本形式一致性约束**。

    能量形式：
        L_normal = mean_{i,j∈N(i)} ( 1 - (n_i · n_j)^2 )

    其中 (·)^2 对法向符号歧义具有鲁棒性（n 与 -n 等价）。

    关键性质：
    - 完全可微分（通过 torch.linalg.eigh 对 current_positions 求梯度）
    - 无需任何预计算或外部法向标注
    - 与 ARAP/AMAP 互补：ARAP 约束边长，本损失约束法向，共同覆盖
      第零阶（位置）和第一阶（曲率/法向）几何信息

    Args:
        current_positions: [M, 3] Tensor, 当前帧锚点位置（在优化器计算图中）
        knn_index:         [M, K] long Tensor, K 近邻索引

    Returns:
        scalar Tensor: 法向一致性均方损失（0 = 完全一致）
    """
    M, K = knn_index.shape

    # 近邻坐标，以锚点为原点做中心化
    neighbors = current_positions[knn_index]                 # [M, K, 3]
    centered = neighbors - current_positions.unsqueeze(1)    # [M, K, 3]

    # 局部协方差矩阵（[M, 3, 3]），最小特征向量 = 法向量
    cov = torch.bmm(centered.transpose(1, 2), centered) / K  # [M, 3, 3]

    # torch.linalg.eigh 对实对称矩阵稳定、可微
    # 特征值升序排列，第 0 列对应最小特征值（法向方向）
    _, eigvecs = torch.linalg.eigh(cov)  # [M, 3], [M, 3, 3]
    normals = torch.nn.functional.normalize(eigvecs[:, :, 0], dim=1)  # [M, 3]

    # 邻域法向与中心法向的余弦平方（符号无关）
    neighbor_normals = normals[knn_index]                              # [M, K, 3]
    cos2 = (normals.unsqueeze(1) * neighbor_normals).sum(dim=2) ** 2  # [M, K]

    return (1.0 - cos2).mean()


# ---------------------------------------------------------------------------
# 7. 自适应弹簧刚度（应变软化模型）
# ---------------------------------------------------------------------------

def compute_adaptive_stiffness(
    current_positions, reference_positions, knn_index,
    k_base, strain_threshold=0.1, softening_factor=2.0
):
    """基于局部工程应变计算自适应弹簧刚度系数（非可微）。

    生物力学依据：软组织具有**应变软化**特性（类似 Yeoh 超弹性模型）：
    - 小形变（< strain_threshold）：接近线弹性，刚度保持 k_base
    - 大形变（> strain_threshold）：刚度按指数衰减（允许大形变而不引发数值爆炸）

    k_adaptive = k_base * exp( -softening_factor * strain / strain_threshold )

    Args:
        current_positions:   [M, 3] 当前锚点位置（detached）
        reference_positions: [M, 3] 参考锚点位置
        knn_index:           [M, K] KNN 索引
        k_base:              [M, K] 基础刚度系数
        strain_threshold:    float, 开始显著软化的应变阈值（默认 10%）
        softening_factor:    float, 软化速率（越大软化越剧烈）

    Returns:
        [M, K] Tensor（detached）: 自适应刚度系数，与 k_base 同 shape
    """
    with torch.no_grad():
        ref_j = reference_positions[knn_index]
        ref_len = torch.norm(
            ref_j - reference_positions.unsqueeze(1), dim=2
        ).clamp(min=1e-6)                                                   # [M, K]

        cur_j = current_positions[knn_index]
        cur_len = torch.norm(
            cur_j - current_positions.unsqueeze(1), dim=2
        )                                                                   # [M, K]

        strain = torch.abs(cur_len - ref_len) / ref_len                    # [M, K]
        softening = torch.exp(
            -softening_factor * strain / strain_threshold
        )                                                                   # [M, K]
        return (k_base * softening).detach()
