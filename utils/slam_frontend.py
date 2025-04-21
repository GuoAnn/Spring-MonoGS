# 主要新增功能：
# 1. 关键帧锚点采样与存储
# 2. 动态模板帧管理
# 3. IDW形变插值接口
# 4. 物理约束可视化支持
import time
import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import numpy as np
from termcolor import colored

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from pytorch3d.ops import knn_points

# 新增锚点管理模块
class AnchorManager:
    """管理锚点生成与拓扑关系的类"""
    def __init__(self, config):
        self.config = config["Spring"]
        self.anchor_graph = {}  # {kf_id: (positions, connections)}
        self.template_kf = None
        self.last_anchors_hash = None  # 用于缓存KNN结果
        self.cached_connections = None
    
    def voxel_sampling(self, depth_map):
        """体积采样生成锚点坐标
        Args:
            depth_map (Tensor): [H,W]深度图
        Returns:
            anchors (Tensor): [N,3] 3D锚点坐标
        """
        H, W = depth_map.shape
        voxel_size = self.config["voxel_size"]
        device = depth_map.device  # 确保设备一致性

        # 生成采样网格
        x = torch.arange(0, W, voxel_size, device=device)
        y = torch.arange(0, H, voxel_size, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        
        # 获取深度值（最近邻采样）
        depth_values = depth_map[
            (grid_y.long().clamp(0, H-1)), 
            (grid_x.long().clamp(0, W-1))
        ]
        
        # 转换为3D坐标 (假设内参已知)
        fx = self.config["fx"]
        fy = self.config["fy"]
        cx = self.config["cx"]
        cy = self.config["cy"]
        
        X = (grid_x - cx) * depth_values / fx
        Y = (grid_y - cy) * depth_values / fy
        Z = depth_values
        
        return torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
    
    def build_spring_topology(self, anchors):
        """构建弹簧连接拓扑
        Args:
            anchors (Tensor): [N,3] 锚点坐标
        Returns:
            connections (LongTensor): [N,K] 每个锚点的邻居索引
        """
        """新增KNN结果缓存机制"""
        curr_hash = hash(anchors.cpu().numpy().tobytes())
        if self.last_anchors_hash == curr_hash:
            return self.cached_connections
        # KNN搜索（排除自身）
        _, knn_idx, _ = knn_points(
            anchors.unsqueeze(0).to(torch.float32),  # 增加类型转换
            anchors.unsqueeze(0).to(torch.float32),
            K=self.config["k_neighbors"]+1
        )
        connections = knn_idx.squeeze(0)[:, 1:]  # [N, K]
        
        # 缓存结果
        self.last_anchors_hash = curr_hash
        self.cached_connections = connections
        return connections

class FrontEnd(mp.Process): 
    def __init__(self, config): 
        super().__init__()
        self.gt_pose = os.environ.get("USE_GT_POSE") == "True" 
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None 
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

        # Spring-Mass相关的参数定义
        self.spring_mass_initialized = False
        self.spring_mass_cfg = config["Training"]["Spring_Mass"]
        self.global_k = nn.Parameter(torch.log10(torch.tensor(self.spring_mass_cfg["DATA"]["GLOBAL_K"], dtype=torch.float32,device="cuda:0")), requires_grad=True)
        self.global_m = nn.Parameter(torch.log10(torch.tensor(self.spring_mass_cfg["DATA"]["GLOBAL_M"], dtype=torch.float32,device="cuda:0")), requires_grad=True)
        self.init_velocity = nn.Parameter(torch.tensor(self.spring_mass_cfg["INIT_VELOCITY"], dtype=torch.float32), requires_grad=True)
        self.damp = nn.Parameter(torch.log10(torch.tensor(self.spring_mass_cfg["DATA"]["GLOBAL_DAMP"], dtype=torch.float32,device="cuda:0")), requires_grad=True)
        self.rebound_k = nn.Parameter(torch.tensor([-1.0], dtype=torch.float32), requires_grad=True)
        self.fric_k = nn.Parameter(torch.tensor([-1.0], dtype=torch.float32), requires_grad=True)
        self.k_bc = nn.Parameter(torch.log10(torch.tensor(self.spring_mass_cfg["DATA"]["GLOBAL_K_BC"], dtype=torch.float32,device="cuda:0")), requires_grad=True)
        self.soft_vector = nn.Parameter(torch.tensor([0.0], dtype=torch.float32), requires_grad=True)
        # 新增物理约束相关成员 ============================
        self.anchor_mgr = AnchorManager(config)  # 锚点管理器
        self.current_template = None  # 当前模板锚点
        self.deformation_params = {}  # {kf_id: (stiffness, damping)}
        
        # 物理约束可视化参数
        self.visualize_springs = config["Spring"].get("visualize", False)

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def knn(self, x: torch.Tensor, ref: torch.Tensor, k, rm_self=False, sqrt_dist=True):
        if rm_self:
            dist, knn_dix, x_neighbor = knn_points(x.unsqueeze(0), ref.unsqueeze(0), K=k + 1, return_nn=True)
            dist = dist.squeeze(0)[:, 1:]  # [N, k]
            knn_dix = knn_dix.squeeze(0)[:, 1:]  # [N, k]
            x_neighbor = x_neighbor.squeeze(0)[:, 1:]  # [N, k, 3]
        else:
            dist, knn_dix, x_neighbor = knn_points(x.unsqueeze(0), ref.unsqueeze(0), K=k, return_nn=True)
            dist = dist.squeeze(0)  # [N, k]
            knn_dix = knn_dix.squeeze(0)  # [N, k]
            x_neighbor = x_neighbor.squeeze(0)  # [N, k, 3]

        if sqrt_dist:
            return torch.sqrt(dist), knn_dix, x_neighbor
        else:
            return dist, knn_dix, x_neighbor

    def initialize_spring_mass(self, xyz: torch.Tensor):
        if self.spring_mass_initialized:
            return

        self.device = "cuda:0"  # 强制指定设备
        xyz = xyz.to(self.device)
        self.n_points = xyz.shape[0]
        self.init_xyz = xyz.detach().clone()
        self.init_v = torch.zeros_like(self.init_xyz, dtype=torch.float32)

        self.origin_len, self.knn_index, _ = self.knn(self.init_xyz, self.init_xyz, self.spring_mass_cfg["K_NEIGHBORS"], rm_self=True)

        Log(f"Spring_Mass got {colored(self.n_points, 'yellow', attrs=['bold'])} points")  # 替换logger为Log
        self.spring_mass_initialized = True

    # ================ 新增可视化支持 ================
    def push_to_gui(self):
        """将物理约束数据推送到GUI（在原有可视化逻辑中添加）"""
        # 原有可视化代码...
        
        # 添加锚点可视化
        if self.visualize_springs and self.current_template is not None:
            anchors, connections = self.current_template
            self.q_main2vis.put(gui_utils.SpringPacket(
                anchors=anchors.cpu().numpy(),
                connections=connections.cpu().numpy()
            ))

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            if init:
                initial_depth = initial_depth.to(self.device)  # 确保设备一致
                self.initialize_spring_mass(initial_depth) # 直接使用张量
            else:
                # 从深度图生成锚点
                anchors = self.anchor_mgr.voxel_sampling(depth.squeeze())  # 使用depth参数
                anchors = anchors.to(self.device)
                # 构建弹簧连接拓扑
                connections = self.anchor_mgr.build_spring_topology(anchors)
                connections = connections.to(self.device) 
                
                # 存储到锚点图
                self.anchor_mgr.anchor_graph[cur_frame_idx] = (anchors, connections)
                
                # 新增：显式发送锚点数据到后端队列
                anchor_data = {
                    "kf_id": cur_frame_idx,
                    "anchors": anchors.cpu().numpy(),   # 转换为CPU数据避免共享内存问题
                    "connections": connections.cpu().numpy()
                }
                self.backend_queue.put(("anchor_data", anchor_data))  # 使用专用消息类型
                
                # 首个关键帧设为模板
                if len(self.anchor_mgr.anchor_graph) == 1:
                    self.anchor_mgr.template_kf = cur_frame_idx
                    Log(f"初始模板帧设为KF {cur_frame_idx}")
                return initial_depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def optimize_spring_mass(self, xyz, v):
        K = 10**self.global_k
        m = 10**self.global_m
        damp = 10**self.damp

        rebound_k = torch.sigmoid(self.rebound_k)
        fric_k = torch.clamp(torch.sigmoid(self.fric_k) * 1.2 - 0.1, min=0, max=1)

        K = K / (self.origin_len + 1e-14)

        dt = self.spring_mass_cfg["DATA"]["DT"] / self.spring_mass_cfg["N_STEP"]

        for _ in range(self.spring_mass_cfg["N_STEP"]):
            force = self.compute_force(xyz, v, K, damp)
            force_sum = force + m.unsqueeze(1) * torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).to(self.device) * 9.8
            v = v + force_sum * dt / m.unsqueeze(1)
            xyz = xyz + v * dt

        return xyz

    def compute_force(self, xyz, v, K, damp):
        knn_xyz = xyz[self.knn_index]  # [N, k, 3]
        delta_pos = knn_xyz - xyz.unsqueeze(1)  # [N, k, 3]
        curr_len = torch.norm(delta_pos, dim=2)  # [N, k]
        norm_delta_pos = delta_pos / (curr_len.unsqueeze(2) + 1e-14)  # [N, k, 3]

        delta_len = (curr_len - self.origin_len)
        delta_len[(delta_len > -1e-6) & (delta_len < 1e-6)] = 0.0
        force = (delta_len * K).unsqueeze(2) * norm_delta_pos  # [N, k, 3]

        if self.spring_mass_cfg["DAMPING"]:
            knn_v = v[self.knn_index]  # [N, k, 3]
            delta_v = knn_v - v.unsqueeze(1)  # [N, k, 3]
            damp_force = (damp * torch.sum(delta_v * norm_delta_pos, dim=-1)).unsqueeze(-1) * norm_delta_pos
            force = force + damp_force

        return force.sum(dim=1)

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    # Tracking with ground truth pose
    def tracking_with_gt_pose(self, cur_frame_idx, viewpoint):
        # 使用 ground truth pose 进行渲染
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        # 渲染当前帧
        render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
        image, depth, opacity = render_pkg["render"], render_pkg["depth"], render_pkg["opacity"]
        self.q_main2vis.put(gui_utils.GaussianPacket(current_frame=viewpoint, gtcolor=viewpoint.original_image, gtdepth=viewpoint.depth if not self.monocular else np.zeros((viewpoint.image_height, viewpoint.image_width))))
        # 更新中值深度
        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg

    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        viewpoint.update_RT(prev.R, prev.T)

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
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

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg

    def is_keyframe(self, cur_frame_idx, last_keyframe_idx, cur_frame_visibility_filter, occ_aware_visibility):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]
        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth
        union = torch.logical_or(cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]).count_nonzero()
        intersection = torch.logical_and(cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # 移除和现在的帧有重叠的帧
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(cur_frame_visibility_filter, occ_aware_visibility[kf_idx]).count_nonzero()
            denom = min(cur_frame_visibility_filter.count_nonzero(), occ_aware_visibility[kf_idx].count_nonzero())
            point_ratio_2 = intersection / denom
            cut_off = self.config["Training"].get("kf_cutoff", 0.4)
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)
        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))
        if len(window) > self.config["Training"]["window_size"]:
            # 计算每个帧的距离，找到需要移除的关键帧
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))
            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)
            # 动态更新模板帧 ----------------------------
            new_template = window[0]
            if new_template != self.anchor_mgr.template_kf:
                Log(f"检测到窗口变化，更新模板帧至KF {new_template}")
                self._update_template_frame(new_template)
        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def request_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility
        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=self.dataset.fx, fy=self.dataset.fy, cx=self.dataset.cx, cy=self.dataset.cy, W=self.dataset.width, H=self.dataset.height).transpose(0, 1).to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                self.q_main2vis.put(gui_utils.GaussianPacket(...))
                self.push_to_gui()  # 添加可视化调用
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(self.cameras, self.kf_indices, self.save_dir, 0, final=True, monocular=self.monocular)
                        save_gaussians(self.gaussians, self.save_dir, "final", final=True)
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(self.dataset, cur_frame_idx, projection_matrix)
                viewpoint.compute_grad_mask(self.config)
                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (len(self.current_window) == self.window_size)

                if self.gt_pose:
                    render_pkg = self.tracking_with_gt_pose(cur_frame_idx, viewpoint)
                else:
                    render_pkg = self.tracking(cur_frame_idx, viewpoint)

                current_window_dict = {self.current_window[0]: self.current_window[1:]}
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
                self.q_main2vis.put(gui_utils.GaussianPacket(gaussians=clone_obj(self.gaussians), current_frame=viewpoint, keyframes=keyframes, kf_window=current_window_dict))

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(cur_frame_idx, last_keyframe_idx, curr_visibility, self.occ_aware_visibility)
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(curr_visibility, self.occ_aware_visibility[last_keyframe_idx]).count_nonzero()
                    intersection = torch.logical_and(curr_visibility, self.occ_aware_visibility[last_keyframe_idx]).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = check_time and point_ratio < self.config["Training"]["kf_overlap"]

                if self.single_thread:
                    create_kf = check_time and create_kf

                if create_kf:
                    self.current_window, removed = self.add_to_window(cur_frame_idx, curr_visibility, self.occ_aware_visibility, self.current_window)
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log("Keyframes lacks sufficient overlap to initialize the map, resetting.")
                        continue
                    depth_map = self.add_new_keyframe(cur_frame_idx, depth=render_pkg["depth"], opacity=render_pkg["opacity"], init=False)
                    self.request_keyframe(cur_frame_idx, viewpoint, self.current_window, depth_map)
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if self.save_results and self.save_trj and create_kf and len(self.kf_indices) % self.save_trj_kf_intv == 0:
                    Log(f"Evaluating ATE at frame: {cur_frame_idx}")
                    eval_ate(self.cameras, self.kf_indices, self.save_dir, cur_frame_idx, monocular=self.monocular)
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    #当添加关键帧时，节流为3fps
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)
                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1
                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False
                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
        # 在渲染后添加
        self.push_to_gui()

    def _update_template_frame(self, new_kf_id):
        """更新当前模板帧"""
        if new_kf_id not in self.anchor_mgr.anchor_graph:  # 存在性检查
            Log(f"错误：KF {new_kf_id} 不存在于锚点图中,跳过更新")
            return
        # 1. 存储旧模板参数
        old_kf = self.anchor_mgr.template_kf
        if old_kf is not None:
            self.deformation_params[old_kf] = self._get_template_params(old_kf)
        
        # 2. 更新模板指针
        self.anchor_mgr.template_kf = new_kf_id
        self.current_template = self.anchor_mgr.anchor_graph[new_kf_id]
        
        # 初始化形变参数（设备一致性处理）
        init_stiffness = self.config["Training"]["Spring_Mass"]["init_stiffness"]  # 键名修正
        init_damping = self.config["Training"]["Spring_Mass"]["init_damping"]
        num_anchors = len(self.current_template[0])
        
        self.deformation_params[new_kf_id] = (
            torch.ones(num_anchors, device=self.device) * init_stiffness,
            torch.ones(num_anchors, device=self.device) * init_damping
        )
    
    # ================ 新增形变插值接口 ================
    def apply_deformation(self, gaussians):
        """应用形变到高斯点（在后端优化后调用）"""
        if self.anchor_mgr.template_kf is None or self.current_template is None:  # 健壮性检查
            return
        
        # 获取当前锚点形变量
        template_anchors = self.current_template[0].to(self.device)
        current_anchors = self.anchor_mgr.anchor_graph[self.current_window[-1]][0].to(self.device)
        
        # 使用KNN匹配对齐锚点
        _, idx, _ = knn_points(current_anchors.unsqueeze(0).float(),template_anchors.unsqueeze(0).float(), K=1)
        matched_template = template_anchors[idx.squeeze()]
        delta = current_anchors - matched_template
        
        # IDW插值
        deformed_positions = self._idw_interpolation(
            gaussians.positions,
            template_anchors,
            delta
        )
        
        # 更新高斯位置
        gaussians.update_positions(deformed_positions)
        
    def _idw_interpolation(self, points, anchors, delta):
        """关键修改：数值稳定性增强"""
        # 设备同步
        points = points.to(anchors.device).float()
        anchors = anchors.float()
        delta = delta.float()
        
        # 距离计算
        dist = torch.cdist(points, anchors)
        weights = 1.0 / (dist + 1e-6)  # 防止除零
        
        # 归一化处理
        weights_sum = weights.sum(dim=1, keepdim=True)
        valid_mask = weights_sum > 1e-8  # 新增有效性检查
        weights = torch.where(
            valid_mask,
            weights / weights_sum,
            torch.zeros_like(weights))
        
        # 位移计算（维度修正）
        displacement = torch.einsum('nk,nkd->nd', weights, delta)  # 修正维度
        return points + displacement
    
    # ============== 其他原有方法保持不变 ==============
    # ...（如run()、tracking()等方法未修改部分保持原样）...
    
    # ============== 新增可视化支持 ==============
    def push_to_gui(self):
        """修改：添加锚点数据包"""
        # 原有可视化代码...
        if self.visualize_springs and self.current_template is not None:
            anchors, connections = self.current_template
            # 设备同步（CUDA -> CPU）
            self.q_main2vis.put(gui_utils.SpringPacket(
                anchors=anchors.cpu().numpy(),
                connections=connections.cpu().numpy()
            ))

