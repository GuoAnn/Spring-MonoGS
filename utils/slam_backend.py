import random
import time
import os

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
from pytorch3d.ops import knn_points
from utils.slam_frontend import AnchorManager

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

        # Spring-Mass related parameters
        self.spring_mass_initialized = False
        self.spring_cfg = config.get("Spring", {})  # 添加默认值防止KeyError
        self.spring_mass_cfg = config["Training"].get("Spring_Mass", {})  # 确保路径正确
        self.device = "cuda:0"  # 强制指定后端设备
        self.global_k = nn.Parameter(torch.log10(torch.tensor(self.spring_mass_cfg["DATA"]["GLOBAL_K"], dtype=torch.float32,device=self.device)), requires_grad=True)
        self.global_m = nn.Parameter(torch.log10(torch.tensor(self.spring_mass_cfg["DATA"]["GLOBAL_M"], dtype=torch.float32,device=self.device)), requires_grad=True)
        self.init_velocity = nn.Parameter(torch.tensor(self.spring_mass_cfg["INIT_VELOCITY"], dtype=torch.float32), requires_grad=True)
        self.damp = nn.Parameter(torch.log10(torch.tensor(self.spring_mass_cfg["DATA"]["GLOBAL_DAMP"], dtype=torch.float32,device=self.device)), requires_grad=True)
        self.rebound_k = nn.Parameter(torch.tensor([-1.0], dtype=torch.float32), requires_grad=True)
        self.fric_k = nn.Parameter(torch.tensor([-1.0], dtype=torch.float32), requires_grad=True)
        self.k_bc = nn.Parameter(torch.log10(torch.tensor(self.spring_mass_cfg["DATA"]["GLOBAL_K_BC"], dtype=torch.float32,device=self.device)), requires_grad=True)
        self.soft_vector = nn.Parameter(torch.tensor([0.0], dtype=torch.float32), requires_grad=True)
        self.init_v = None  # 延迟初始化
        self.anchor_mgr = AnchorManager(config)  # 需要确保AnchorManager已定义
        self.spring_cfg = config.get("Spring", {"k_neighbors": 8})  # 安全获取配置
        self.spring_mass_cfg = config["Training"].get("Spring_Mass", {})
        self.device = "cuda:0"  # 强制指定设备
        # 新增弹簧损失权重
        self.spring_loss_weight = self.spring_cfg.get("loss_weight", 0.1)

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]
        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = self.cameras_extent * self.config["Training"]["gaussian_extent"]
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Dataset"].get("single_thread", False)

    def knn(self, x: torch.Tensor, ref: torch.Tensor, k, rm_self=False, sqrt_dist=True):
        if rm_self:
            dist, knn_dix, x_neighbor = knn_points(x.unsqueeze(0), ref.unsqueeze(0), K=k + 1, return_nn=True)
            dist = dist.squeeze(0)[:, 1:]
            knn_dix = knn_dix.squeeze(0)[:, 1:]
            x_neighbor = x_neighbor.squeeze(0)[:, 1:]
        else:
            dist, knn_dix, x_neighbor = knn_points(x.unsqueeze(0), ref.unsqueeze(0), K=k, return_nn=True)
            dist = dist.squeeze(0)
            knn_dix = knn_dix.squeeze(0)
            x_neighbor = x_neighbor.squeeze(0)
        return (torch.sqrt(dist), knn_dix, x_neighbor) if sqrt_dist else (dist, knn_dix, x_neighbor)

    def initialize_spring_mass(self, xyz: torch.Tensor):
        if self.spring_mass_initialized:
            return

        # 强制指定设备并确保数据在GPU
        self.device = "cuda:0"
        xyz = xyz.to(self.device)
        self.n_points = xyz.shape[0]
        self.init_xyz = xyz.detach().clone()
        self.init_v = torch.zeros_like(self.init_xyz, dtype=torch.float32,device=self.device)

        # KNN拓扑构建
        self.origin_len, self.knn_index, _ = self.knn(
            self.init_xyz.float(), self.init_xyz.float(), 
            self.spring_cfg.get("k_neighbors", 8),  # 从Spring块读取
            rm_self=True
        )

        Log(f"Spring_Mass initialized with {colored(self.n_points, 'yellow', attrs=['bold'])} points")  # 替换logger为Log
        self.spring_mass_initialized = True

    def optimize_spring_mass(self, xyz, v):
        xyz = xyz.to(self.device)
        v = v.to(self.device)
        K = 10**self.global_k
        m = 10**self.global_m.to(self.device)
        damp = 10**self.damp.to(self.device)

        rebound_k = torch.sigmoid(self.rebound_k)
        fric_k = torch.clamp(torch.sigmoid(self.fric_k) * 1.2 - 0.1, min=0, max=1)
        K = K / (self.origin_len + 1e-14)
        dt = self.spring_mass_cfg["DATA"]["DT"] / self.spring_mass_cfg["N_STEP"]

        for _ in range(self.spring_mass_cfg["N_STEP"]):
            force = self.compute_force(xyz, v, K, damp)
            force_sum = force + m.unsqueeze(1) * torch.tensor([0.0, 1.0, 0.0], device=self.device) * 9.8
            v = v + force_sum * dt / m.unsqueeze(1)
            xyz = xyz + v * dt
        return xyz

    def compute_force(self, xyz, v, K, damp):
        knn_xyz = xyz[self.knn_index.long()]
        delta_pos = knn_xyz - xyz.unsqueeze(1).float()
        curr_len = torch.norm(delta_pos, dim=2)
        norm_delta_pos = delta_pos / (curr_len.unsqueeze(2) + 1e-14)
        delta_len = (curr_len - self.origin_len)
        delta_len[(delta_len > -1e-6) & (delta_len < 1e-6)] = 0.0
        force = (delta_len * K).unsqueeze(2) * norm_delta_pos

        if self.spring_mass_cfg["DAMPING"]:
            knn_v = v[self.knn_index]
            delta_v = knn_v - v.unsqueeze(1)
            damp_force = (damp * torch.sum(delta_v * norm_delta_pos, dim=-1)).unsqueeze(-1) * norm_delta_pos
            force += damp_force

        return force.sum(dim=1)

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map)
        if init:
            self.initialize_spring_mass(self.gaussians.get_positions())

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        while not self.backend_queue.empty():
            self.backend_queue.get()


        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
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

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        ''' 核心目标：通过优化高斯分布的位置、形状和外观，使渲染结果更接近真实图像。
            关键步骤：
            1.数据准备:收集当前窗口中的视点(viewpoints)和随机视点。
            2.渲染与损失计算：对每个视点进行渲染，计算损失函数。
            3.高斯优化：调整高斯分布的参数（位置、半径、不透明度等）。
            4.剪枝与分裂：移除不必要的高斯点或增加新的高斯点。
            5.位姿优化：调整相机位姿以提高一致性。'''
        if self.gaussians.get_positions().device != self.device:
            self.gaussians.to(self.device)
        if len(current_window) == 0:
            return
        if hasattr(self, 'apply_deformation'):
            self.apply_deformation(self.gaussians)  # 在优化前应用形变
        spring_loss = self._compute_spring_loss()
        loss_mapping += spring_loss * self.spring_loss_weight

        #数据准备
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)
        
        if self.spring_mass_initialized:
            xyz = self.gaussians.get_positions()
            v = self.init_v.clone()
            new_xyz = self.optimize_spring_mass(xyz, v)
            self.gaussians.update_positions(new_xyz)

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []
            # 渲染与损失计算
            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
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

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward() #反向传播
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune: #剪枝与分裂
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
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
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)

                # 根据 use_gt_pose 决定是否跳过位姿优化
                if not self.use_gt_pose:
                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)

                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        return gaussian_split

    def _compute_spring_loss(self):
        """计算锚点弹簧系统的弹性势能"""
        if self.anchor_mgr.template_kf is None:
            return torch.tensor(0.0, device=self.device)
        
        # 获取当前锚点和模板锚点
        template_id = self.anchor_mgr.template_kf
        current_id = self.current_window[-1]
        template_anchors, _ = self.anchor_mgr.anchor_graph[template_id]
        current_anchors, connections = self.anchor_mgr.anchor_graph[current_id]
        
        # 计算弹簧伸长量
        neighbor_pos = current_anchors[connections]  # [N, K, 3]
        delta = neighbor_pos - current_anchors.unsqueeze(1)
        curr_lengths = torch.norm(delta, dim=2)
        rest_lengths = torch.norm(
            template_anchors[connections] - template_anchors.unsqueeze(1), 
            dim=2
        )
        
        # 胡克定律计算能量
        length_diffs = curr_lengths - rest_lengths
        return torch.mean(0.5 * (10**self.global_k) * length_diffs**2)
    
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

    def apply_deformation(self, gaussians):
        """从FrontEnd迁移的形变插值逻辑"""
        if not hasattr(self, 'anchor_mgr') or self.anchor_mgr.template_kf is None:
            return

        template_anchors = self.anchor_mgr.anchor_graph[self.anchor_mgr.template_kf][0].to(self.device)
        current_anchors = self.anchor_mgr.anchor_graph[self.current_window[-1]][0].to(self.device)

        # KNN对齐锚点
        _, idx, _ = knn_points(current_anchors.unsqueeze(0).float(), template_anchors.unsqueeze(0).float(), K=1)
        matched_template = template_anchors[idx.squeeze()]
        delta = current_anchors - matched_template

        # IDW插值零值处理
        points = gaussians.positions.to(self.device).float()
        dist = torch.cdist(points, template_anchors)
        weights = 1.0 / (dist + 1e-6)
        weights_sum = weights.sum(dim=1, keepdim=True)
        valid_mask = weights_sum > 1e-8
        weights = torch.where(
            valid_mask,
            weights/weights_sum, 
            torch.zeros_like(weights)
        )
        displacement = torch.einsum('nk,nkd->nd', weights, delta)
        gaussians.update_positions(points + displacement)

    def run(self):
        while True:
            if not self.gaussians.get_positions().device == self.device:
                self.gaussians.to(self.device)
            if self.backend_queue.empty():
                if self.pause or len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                # 新增：处理锚点数据
                if data[0] == "anchor_data":  # 消息类型为 "anchor_data"
                    kf_id = data[1]["kf_id"]
                    anchors = torch.tensor(data[1]["anchors"], device=self.device, dtype=torch.float32)
                    connections = torch.tensor(data[1]["connections"], device=self.device, dtype=torch.long)
                    self.anchor_mgr.anchor_graph[kf_id] = (anchors, connections)
                    Log(f"后端接收KF{kf_id}的锚点，数量：{len(anchors)}")
                if self.single_thread:
                    time.sleep(0.01)
                    continue
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

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                           
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
                            iter_per_kf = 50 if self.live_mode else 300 #150 1000
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num #*3
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
                    # 根据 use_gt_pose 决定是否跳过位姿优化
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

    def _synchronize_anchors(self):
        """确保所有锚点数据在正确设备"""
        for kf_id in self.anchor_mgr.anchor_graph:
            anchors, conn = self.anchor_mgr.anchor_graph[kf_id]
            self.anchor_mgr.anchor_graph[kf_id] = (
                anchors.to(self.device),
                conn.to(self.device)
            )