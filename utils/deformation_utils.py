"""体内可形变SLAM的形变约束工具模块。

本模块实现了四种面向软组织形变建模的物理约束，设计原则：
- ARAP / 体积保持损失完全可微分，可直接加入高斯点优化的损失函数
- 自适应刚度与形变势能计算运行在 torch.no_grad() 上下文中，无副作用
- 所有函数只依赖 torch，无额外第三方依赖

参考文献：
  [1] Sumner et al., "Embedded deformation for shape manipulation", SIGGRAPH 2007
  [2] Sorkine & Alexa, "As-rigid-as-possible surface modeling", SGP 2007
  [3] Holzapfel, "Nonlinear solid mechanics", Wiley 2000
"""

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

    ref_i = reference_positions.unsqueeze(1).expand(M, K, 3)   # [M, K, 3]
    ref_j = reference_positions[knn_index]                      # [M, K, 3]
    ref_len = torch.norm(ref_j - ref_i, dim=2).clamp(min=1e-6)  # [M, K]

    cur_i = current_positions.unsqueeze(1).expand(M, K, 3)      # [M, K, 3]
    cur_j = current_positions[knn_index]                        # [M, K, 3]
    cur_len = torch.norm(cur_j - cur_i, dim=2)                  # [M, K]

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
# 4. 自适应弹簧刚度（应变软化模型）
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
