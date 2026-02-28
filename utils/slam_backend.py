import random
import time
import os

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
from utils.gaussian_interpolation import interpolate_gaussian
from utils.deformation_utils import (
    compute_arap_loss,
    compute_volume_preservation_loss,
    compute_deformation_energy,
    compute_adaptive_stiffness,
    compute_amap_loss,
    compute_adaptive_loss_weight,
    compute_normal_consistency_loss,
)


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.use_gt_pose = os.environ.get("USE_GT_POSE") == "True"
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        
        # 弹簧模型相关属性
        self.spring_models = {}  # 存储每个关键帧的弹簧模型
        self.anchor_points = {}  # 存储每个关键帧的锚点
        self.template_anchors = None  # 存储模板帧的锚点
        self.anchor_velocities = {}  # 存储每个关键帧锚点的速度
        self.spring_model_enabled = config["Training"]["spring_model"]["enabled"]
        # 新增：锚点相关属性字典，防止AttributeError
        self.anchor_indices = {}
        self.anchor_features_dc = {}
        self.anchor_features_rest = {}
        self.anchor_opacity = {}
        
        # 弹簧模型参数
        self.spring_k = config["Training"]["spring_model"].get("spring_k", 0.1)  # 弹性系数
        self.damping = config["Training"]["spring_model"].get("damping", 0.01)  # 阻尼系数
        self.dt = config["Training"]["spring_model"].get("dt", 0.01)  # 时间步长
        self.n_iterations = config["Training"]["spring_model"].get("n_iterations", 10)  # 迭代次数

        # 形变正则化参数（ARAP + 体积保持 + AMAP + 法向一致性）
        self.arap_weight = config["Training"]["spring_model"].get("arap_weight", 0.1)
        self.vol_weight = config["Training"]["spring_model"].get("vol_weight", 0.05)
        self.amap_weight = config["Training"]["spring_model"].get("amap_weight", 0.0)
        self.normal_weight = config["Training"]["spring_model"].get("normal_weight", 0.0)
        # 每个 spring model 的历史最大边长缓存（用于 AMAP 约束）
        self.max_edge_lengths_cache = {}  # {frame_idx: [M, K] Tensor}

        # 自适应弹簧刚度参数（应变软化模型）
        self.adaptive_stiffness_enabled = config["Training"]["spring_model"].get(
            "adaptive_stiffness", True
        )
        self.strain_threshold = config["Training"]["spring_model"].get(
            "strain_threshold", 0.1
        )

        self.warned_frames = set()
        self.mapping_error_logged = set()
        self.invalid_gaussian_logged = False

        self._last_gaussian_shape = None

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )
        #Log(f"[DEBUG] after add_next_kf: gaussians.get_xyz shape: {self.gaussians.get_xyz.shape if self.gaussians.get_xyz is not None else 'None'}")

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        #Log(f"[DEBUG] after reset: gaussians.get_xyz shape: {self.gaussians.get_xyz.shape if self.gaussians.get_xyz is not None else 'None'}")
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        """初始化地图"""
        # 确保高斯点正确初始化
        if self.gaussians is None or self.gaussians.get_xyz is None:
            Log("Error: Gaussians not properly initialized")
            return None

        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        # 在地图初始化完成后创建锚点
        if self.spring_model_enabled and self.gaussians.get_xyz is not None:
            gaussian_points = self.gaussians.get_xyz
            if gaussian_points.numel() > 0:
                # 注释掉后端的generate_anchor_points函数（只保留前端采样）
                # anchor_points, anchor_indices, anchor_features_dc, anchor_features_rest, anchor_opacity = self.generate_anchor_points(gaussian_points)
                # self.anchor_points[cur_frame_idx] = anchor_points
                # self.template_anchors = anchor_points  # 将第一帧设为模板
                Log(f"Created initial anchor points for frame {cur_frame_idx}")

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg


    def optimize_spring_model(self, current_window):
        """使用弹簧模型约束锚点形变的空间一致性，减小形变误差。

        算法步骤：
        1. 从当前高斯点云中读取每个锚点的最新位置（KNN查找最近高斯点）。
        2. 基于当前应变计算自适应弹簧刚度（应变软化：大形变时刚度降低，
           模拟软组织的非线性力学特性）。
        3. 运行 SpringModel.step() —— 基于胡克定律的相邻锚点弹簧物理，
           使形变在局部邻域内保持平滑一致（空间相关性约束）。
        4. 将锚点形变量通过IDW插值传播到全体高斯点的几何位置（仅位置，
           不影响外观属性，外观由光度优化器负责）。
        """
        try:
            if not self.spring_model_enabled:
                return

            gaussian_xyz = self.gaussians.get_xyz
            if gaussian_xyz is None or gaussian_xyz.numel() == 0:
                return

            gaussian_xyz_detached = gaussian_xyz.detach()

            from pytorch3d.ops import knn_points

            total_energy = 0.0
            n_models = 0

            for frame_idx in current_window:
                spring_model = self.spring_models.get(frame_idx)
                if spring_model is None:
                    continue

                # 弹簧模型创建时的锚点初始位置（参考构型）
                original_anchors = spring_model.anchor_points  # [M, 3]
                M = original_anchors.shape[0]
                if M < 2:
                    continue

                # 从当前高斯点云中找到每个锚点最近的高斯点，以获取锚点的实时位置。
                # 这样使弹簧模型能够感知光度优化后高斯点的移动。
                _, idx, _ = knn_points(
                    original_anchors.unsqueeze(0),
                    gaussian_xyz_detached.unsqueeze(0),
                    K=1,
                )
                idx = idx.squeeze(0).squeeze(-1).long()  # [M]
                current_anchors = gaussian_xyz_detached[idx].clone()  # [M, 3]

                # 自适应刚度：基于当前应变，对大形变区域降低弹簧刚度（应变软化）。
                # 每次迭代重新计算，不修改 spring_model 的基础参数。
                original_k = spring_model.k.clone()
                if self.adaptive_stiffness_enabled:
                    spring_model.k = compute_adaptive_stiffness(
                        current_anchors,
                        original_anchors,
                        spring_model.knn_index,
                        original_k,
                        strain_threshold=self.strain_threshold,
                    )

                # 运行弹簧物理模拟（胡克定律 + 阻尼）：
                # 相邻锚点之间的弹簧力使形变保持空间一致性。
                # 每次调用使用零速度初始值，避免跨帧速度状态污染。
                velocity = torch.zeros_like(current_anchors)
                for _ in range(self.n_iterations):
                    current_anchors, velocity = spring_model.step(
                        current_anchors, velocity, self.dt
                    )

                # 恢复基础刚度（自适应刚度仅在本次迭代有效）
                spring_model.k = original_k

                # 计算锚点相对于初始位置的形变量
                anchor_deltas = current_anchors - original_anchors

                # 记录形变势能（用于监控，不参与梯度计算）
                total_energy += compute_deformation_energy(
                    current_anchors,
                    original_anchors,
                    spring_model.knn_index,
                    original_k,
                )
                n_models += 1

                # 将弹簧平滑后的形变量通过IDW插值传播到全体高斯点（仅更新位置）
                if anchor_deltas.abs().max().item() > 1e-8:
                    self._apply_anchor_deformation_to_gaussians(
                        original_anchors, gaussian_xyz_detached, anchor_deltas
                    )

            if n_models > 0 and self.iteration_count % 50 == 0:
                avg_energy = total_energy / n_models
                Log(f"[Deformation] mean spring energy: {avg_energy:.4f} over {n_models} frames")

        except Exception as e:
            Log(f"Spring model error: {e}")

    def _compute_deformation_reg_loss(self, current_window):
        """计算可微分的形变正则化损失（ARAP + 体积保持 + AMAP + 法向一致性）。

        设计原则：
        - ARAP：约束边长应变（第零阶位置一致性）
        - 体积保持：约束邻域回转半径（不可压缩性）
        - AMAP：驱动边长趋向历史最大值，防止"表面收缩"退化（NRSfM paper eq.(3)）
        - 法向一致性：通过局部 PCA 法向场估计，惩罚相邻法向偏差（NRSfM paper
          度量张量不变性的离散近似）
        - 自适应权重 ω_l：根据锚点数量缩放整体形变损失（NRSfM paper eq.(10)）

        Args:
            current_window: list[int], 当前关键帧窗口的帧索引列表

        Returns:
            scalar Tensor: 总形变正则化损失
        """
        if not self.spring_model_enabled:
            return torch.tensor(0.0, device=self.device)
        if (self.arap_weight <= 0 and self.vol_weight <= 0
                and self.amap_weight <= 0 and self.normal_weight <= 0):
            return torch.tensor(0.0, device=self.device)

        from pytorch3d.ops import knn_points

        gaussian_xyz = self.gaussians._xyz  # [N, 3], 在优化器计算图中
        total_loss = torch.tensor(0.0, device=self.device)
        count = 0

        for frame_idx in current_window:
            spring_model = self.spring_models.get(frame_idx)
            if spring_model is None:
                continue

            ref_anchors = spring_model.anchor_points  # [M, 3], detached
            knn_index = spring_model.knn_index        # [M, K], long
            M = ref_anchors.shape[0]
            if M < 2:
                continue

            # 非可微的KNN查找：找到每个参考锚点在当前高斯点云中的最近邻索引
            # 索引本身不在计算图中，但通过索引查找得到的 current_anchors 仍可微
            with torch.no_grad():
                _, idx_map, _ = knn_points(
                    ref_anchors.unsqueeze(0),
                    gaussian_xyz.detach().unsqueeze(0),
                    K=1,
                )
                idx_map = idx_map.squeeze(0).squeeze(-1).long()  # [M]

            # 当前锚点位置（可微分：梯度通过 gaussian_xyz[idx_map] 流向 _xyz）
            current_anchors = gaussian_xyz[idx_map]  # [M, 3]

            # NRSfM eq.(10): 自适应权重 ω_l（依据锚点数量缩放形变损失）
            omega_l = compute_adaptive_loss_weight(M)

            if self.arap_weight > 0:
                arap = compute_arap_loss(current_anchors, ref_anchors, knn_index)
                total_loss = total_loss + omega_l * self.arap_weight * arap

            if self.vol_weight > 0:
                vol = compute_volume_preservation_loss(
                    current_anchors, ref_anchors, knn_index
                )
                total_loss = total_loss + omega_l * self.vol_weight * vol

            # AMAP：更新历史最大边长缓存，并驱动当前边长趋向历史最大值
            # 防止软组织 SLAM 产生"表面收缩"退化解（见 NRSfM AMAP 约束）
            if self.amap_weight > 0:
                with torch.no_grad():
                    anc_det = current_anchors.detach()
                    det_j = anc_det[knn_index]                               # [M, K, 3]
                    cur_len_det = torch.norm(det_j - anc_det.unsqueeze(1), dim=2)  # [M, K]
                    if frame_idx in self.max_edge_lengths_cache:
                        self.max_edge_lengths_cache[frame_idx] = torch.max(
                            self.max_edge_lengths_cache[frame_idx], cur_len_det
                        )
                    else:
                        self.max_edge_lengths_cache[frame_idx] = cur_len_det.clone()
                amap = compute_amap_loss(
                    current_anchors,
                    knn_index,
                    self.max_edge_lengths_cache[frame_idx],
                )
                total_loss = total_loss + omega_l * self.amap_weight * amap

            # 法向一致性：局部 PCA 法向场平滑
            # 对应 NRSfM 度量张量不变性的离散近似（第一阶微分几何约束）
            # 要求 M >= 4：PCA 需要至少 3 个近邻点（K=8 时锚点数最低限制）
            # 以保证 3×3 协方差矩阵的特征分解在数值上是非奇异的
            if self.normal_weight > 0 and M >= 4:
                normal = compute_normal_consistency_loss(current_anchors, knn_index)
                total_loss = total_loss + omega_l * self.normal_weight * normal

            count += 1

        if count > 0:
            total_loss = total_loss / count

        return total_loss

    def _apply_anchor_deformation_to_gaussians(
        self, anchor_points, gaussian_points, anchor_deltas
    ):
        """将锚点形变量通过逆距离加权（IDW）插值传播到全体高斯点的几何位置。

        只更新位置（_xyz），不修改外观属性（颜色、不透明度、尺度、旋转），
        因为弹簧模型仅约束几何形变，外观属性由光度优化器负责更新。
        """
        from pytorch3d.ops import knn_points

        K_BINDING = min(16, anchor_points.shape[0])
        dist, idx, _ = knn_points(
            gaussian_points.unsqueeze(0),
            anchor_points.unsqueeze(0),
            K=K_BINDING,
        )
        dist = dist.squeeze(0)   # [N, K]
        idx = idx.squeeze(0).long()  # [N, K]

        dist = torch.clamp(dist, min=1e-6)
        eps = 1e-14
        weights = 1.0 / (dist.sqrt() + eps)       # [N, K]
        weights = weights / (weights.sum(dim=1, keepdim=True) + eps)  # 归一化

        # IDW：每个高斯点的位移 = 最近 K 个锚点形变量的加权平均
        anchor_deltas_knn = anchor_deltas[idx]                          # [N, K, 3]
        delta_gaussians = (anchor_deltas_knn * weights.unsqueeze(-1)).sum(dim=1)  # [N, 3]
        new_xyz = gaussian_points + delta_gaussians

        if torch.isnan(new_xyz).any() or torch.isinf(new_xyz).any():
            Log("Warning: Invalid positions from spring deformation, skipping update.")
            return

        with torch.no_grad():
            self.gaussians._xyz.data.copy_(new_xyz)

    def interpolate_gaussians(self, anchor_points, gaussian_points, anchor_deltas, 
                              anchor_features_dc=None, anchor_features_rest=None, anchor_opacity=None, mode="idw"):
        """
        使用IDW插值更新高斯点所有属性
        """
        from pytorch3d.ops import knn_points
        K_BINDING = min(16, anchor_points.shape[0])
        dist, idx, _ = knn_points(gaussian_points.unsqueeze(0), anchor_points.unsqueeze(0), K=K_BINDING)
        dist = dist.squeeze(0)  # [N, K]
        idx = idx.squeeze(0).long()    # [N, K]，确保为long类型

        # 数值稳定性：距离不能为0
        dist = torch.clamp(dist, min=1e-6)
        eps = 1e-14
        weights = 1.0 / (dist.sqrt() + eps)  # [N, K]
        weights = weights / (weights.sum(dim=1, keepdim=True) + eps)  # 归一化

        N = gaussian_points.shape[0]
        new_attrs = {}

        # 1. 位置
        anchor_xyz = anchor_points  # [M, 3]
        anchor_deltas_knn = anchor_deltas[idx]  # [N, K, 3]
        delta_gaussians = (anchor_deltas_knn * weights.unsqueeze(-1)).sum(dim=1)  # [N, 3]
        new_xyz = gaussian_points + delta_gaussians
        new_attrs['xyz'] = new_xyz

        # 2. 颜色（SH特征）
        if mode == "nearest":
            nearest_idx = idx[:, 0].long()  # [N]
            new_features_dc = anchor_features_dc[nearest_idx]
            new_features_rest = anchor_features_rest[nearest_idx]
            new_opacity = anchor_opacity[nearest_idx]
        else:
            new_features_dc = (anchor_features_dc[idx] * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            new_features_rest = (anchor_features_rest[idx] * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            new_opacity = (anchor_opacity[idx] * weights.unsqueeze(-1)).sum(dim=1)
        new_attrs['features_dc'] = new_features_dc
        new_attrs['features_rest'] = new_features_rest

        # 3. 不透明度（已在上方IDW分支中计算完毕，无需重复加权）
        new_attrs['opacity'] = new_opacity

        # 4. 尺度
        anchor_scaling = self.gaussians._scaling[idx]
        new_scaling = (anchor_scaling * weights.unsqueeze(-1)).sum(dim=1)
        new_attrs['scaling'] = new_scaling

        # 5. 旋转
        anchor_rotation = self.gaussians._rotation[idx]
        new_rotation = (anchor_rotation * weights.unsqueeze(-1)).sum(dim=1)
        new_attrs['rotation'] = new_rotation

        for k, v in new_attrs.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                Log(f"Warning: Invalid {k} detected in interpolated gaussian points, skipping update.")
                return getattr(self.gaussians, f"_{k}").data

        self.gaussians._xyz.data.copy_(new_attrs['xyz'])
        self.gaussians._features_dc.data.copy_(new_attrs['features_dc'])
        self.gaussians._features_rest.data.copy_(new_attrs['features_rest'])
        self.gaussians._opacity.data.copy_(new_attrs['opacity'])
        self.gaussians._scaling.data.copy_(new_attrs['scaling'])
        self.gaussians._rotation.data.copy_(new_attrs['rotation'])

        return new_attrs['xyz']

    def map(self, current_window, prune=False, iters=1):
        # --- DEBUG: map开始时高斯点状态 ---
        cur_shape = self.gaussians.get_xyz.shape if self.gaussians is not None and hasattr(self.gaussians, "get_xyz") and self.gaussians.get_xyz is not None else None
        if cur_shape != self._last_gaussian_shape:
            #Log(f"[DEBUG] map start: gaussians.get_xyz shape: {cur_shape}")
            self._last_gaussian_shape = cur_shape
        
        if len(current_window) == 0:
            return

        # 检查高斯点状态
        if self.gaussians is None or self.gaussians.get_xyz is None or self.gaussians.get_xyz.numel() == 0:
            if not self.invalid_gaussian_logged:
                Log("Error: Invalid gaussians state")
                self.invalid_gaussian_logged = True
            return False
        else:
            self.invalid_gaussian_logged = False  # 恢复

        # 在进入建图模块前优化弹簧模型
        self.optimize_spring_model(current_window)

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        # 记录已经警告过的帧
        warned_frames = set()

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            try:
                for cam_idx in range(len(current_window)):
                    viewpoint = viewpoint_stack[cam_idx]
                    frame_idx = current_window[cam_idx]
                    keyframes_opt.append(viewpoint)
                    
                    # 检查viewpoint状态
                    if not hasattr(viewpoint, 'R') or not hasattr(viewpoint, 'T'):
                        if frame_idx not in warned_frames:
                            Log(f"Error: Invalid viewpoint state for frame {frame_idx}")
                            warned_frames.add(frame_idx)
                        continue
                        
                    render_pkg = render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background
                    )
                    
                    # 检查render_pkg是否为None
                    if render_pkg is None:
                        if frame_idx not in warned_frames:
                            Log(f"Warning: render_pkg is None for frame {frame_idx}")
                            warned_frames.add(frame_idx)
                        continue
                        
                    (
                        image,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        depth,
                        opacity,
                        n_touched,
                    ) = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                        render_pkg["n_touched"],
                    )

                    loss_mapping += get_loss_mapping(
                        self.config, image, depth, viewpoint, opacity
                    )
                    viewspace_point_tensor_acm.append(viewspace_point_tensor)
                    visibility_filter_acm.append(visibility_filter)
                    radii_acm.append(radii)
                    n_touched_acm.append(n_touched)

                for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                    viewpoint = random_viewpoint_stack[cam_idx]
                    render_pkg = render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background
                    )
                    
                    # 检查render_pkg是否为None
                    if render_pkg is None:
                        continue
                        
                    (
                        image,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        depth,
                        opacity,
                        n_touched,
                    ) = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                        render_pkg["n_touched"],
                    )
                    loss_mapping += get_loss_mapping(
                        self.config, image, depth, viewpoint, opacity
                    )
                    viewspace_point_tensor_acm.append(viewspace_point_tensor)
                    visibility_filter_acm.append(visibility_filter)
                    radii_acm.append(radii)

                    if len(viewspace_point_tensor_acm) == 0:
                        if self.iteration_count % 100 == 0:  # 降低警告频率
                            Log("Warning: No valid frames rendered in this iteration")
                        return False

                scaling = self.gaussians.get_scaling
                isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
                loss_mapping += 10 * isotropic_loss.mean()

                # ARAP 形变正则化 + 体积保持损失
                # 为高斯点位置优化提供物理约束梯度，防止形变产生物理上不合理的结果
                if self.spring_model_enabled:
                    deform_reg = self._compute_deformation_reg_loss(current_window)
                    loss_mapping += deform_reg

                loss_mapping.backward()
                gaussian_split = False
                
            except Exception as e:
                err_str = str(e)
                if err_str not in self.mapping_error_logged:
                    Log(f"Error in mapping: {err_str}")
                    self.mapping_error_logged.add(err_str)
                return False
            
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        if not hasattr(self, '_n_obs_initialized') or not self._n_obs_initialized:
                            self.gaussians.n_obs.fill_(0)
                            self._n_obs_initialized = True
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                        if prune_mode == "slam":
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            # DEBUG: prune_points后高斯点状态
                            #if self.gaussians.get_xyz is not None:
                                #Log(f"[DEBUG] after prune_points: gaussians.get_xyz shape: {self.gaussians.get_xyz.shape}")
                            #else:
                                #Log("[DEBUG] after prune_points: gaussians.get_xyz is None")
                            if self.gaussians.get_xyz is None or self.gaussians.get_xyz.numel() == 0:
                                Log("Warning: All gaussians have been pruned! Restoring at least one point.")
                                return False
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    if self.gaussians.get_xyz is not None and self.gaussians.get_xyz.shape[0] <= 100:
                        Log("Warning: Too few gaussians, skip densify_and_prune to avoid deleting all points.")
                    else:
                        self.gaussians.densify_and_prune(
                            self.opt_params.densify_grad_threshold,
                            self.gaussian_th,
                            self.gaussian_extent,
                            self.size_threshold,
                        )
                        # DEBUG: densify_and_prune后高斯点状态
                        if self.gaussians.get_xyz is not None:
                            #Log(f"[DEBUG] after densify_and_prune: gaussians.get_xyz shape: {self.gaussians.get_xyz.shape}")
                            if self.gaussians.get_xyz.shape[0] == 0:
                                Log("Warning: All gaussians have been pruned by densify_and_prune! Skipping this prune.")
                                return False
                        else:
                            #Log("[DEBUG] after densify_and_prune: gaussians.get_xyz is None")
                            return False
                    gaussian_split = True

                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    # 合并所有frame的可见性mask
                    if isinstance(visibility_filter_acm, list):
                        global_visibility = torch.zeros_like(self.gaussians.get_xyz[:, 0], dtype=torch.bool)
                        for vf in visibility_filter_acm:
                            global_visibility |= vf
                        self.gaussians.reset_opacity_nonvisible(global_visibility)
                    else:
                        self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    # DEBUG: reset_opacity_nonvisible后高斯点状态
                    #if self.gaussians.get_xyz is not None:
                        #Log(f"[DEBUG] after reset_opacity_nonvisible: gaussians.get_xyz shape: {self.gaussians.get_xyz.shape}")
                    #else:
                        #Log("[DEBUG] after reset_opacity_nonvisible: gaussians.get_xyz is None")
                    if self.gaussians.get_xyz is None or self.gaussians.get_xyz.numel() == 0:
                        Log("Warning: All gaussians have been removed after reset_opacity_nonvisible")
                        return False
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)

                if not self.use_gt_pose:
                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)

                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        # --- DEBUG: map结束时高斯点状态 ---
        cur_shape = self.gaussians.get_xyz.shape if self.gaussians is not None and hasattr(self.gaussians, "get_xyz") and self.gaussians.get_xyz is not None else None
        if cur_shape != self._last_gaussian_shape:
            #Log(f"[DEBUG] map end: gaussians.get_xyz shape: {cur_shape}")
            self._last_gaussian_shape = cur_shape
        return gaussian_split

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    spring_model = data[5]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)
                    
                    # 存储弹簧模型和锚点
                    self.spring_models[cur_frame_idx] = spring_model
                    self.anchor_points[cur_frame_idx] = spring_model.anchor_points
                    self.anchor_indices[cur_frame_idx] = spring_model.anchor_indices
                    self.anchor_features_dc[cur_frame_idx] = spring_model.anchor_features_dc
                    self.anchor_features_rest[cur_frame_idx] = spring_model.anchor_features_rest
                    self.anchor_opacity[cur_frame_idx] = spring_model.anchor_opacity

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    if not self.use_gt_pose:
                        self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
