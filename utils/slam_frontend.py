import time
import os

import numpy as np
import torch
import torch.multiprocessing as mp
from pytorch3d.ops import knn_points
from scipy.interpolate import griddata

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth

class SpringModel:
    def __init__(self, anchor_points, k_neighbors=8):
        self.anchor_points = anchor_points
        self.k_neighbors = k_neighbors
        self.device = anchor_points.device
        self.n_points = anchor_points.shape[0]
        
        # 计算初始弹簧长度和最近邻索引
        self.origin_len, self.knn_index, _ = self.knn(anchor_points, anchor_points, k_neighbors, rm_self=True)
        
        # 初始化弹簧参数
        self.k = torch.ones_like(self.origin_len) * 0.1  # 弹性系数
        self.damping = torch.ones_like(self.origin_len) * 0.01  # 阻尼系数
        
    def knn(self, x, ref, k, rm_self=False):
        # 确保输入张量有效且不为空
        if x is None or ref is None:
            raise ValueError("Input tensors cannot be None")
            
        # 打印输入张量的形状以便调试
        Log(f"Input tensor shapes - x: {x.shape}, ref: {ref.shape}")
        
        # 确保输入张量至少是2维的
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if ref.dim() == 1:
            ref = ref.unsqueeze(0)
            
        # 添加批次维度（如果需要）
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if ref.dim() == 2:
            ref = ref.unsqueeze(0)
            
        # 确保点云维度正确 (B, N, 3)
        if x.shape[-1] != 3:
            if x.shape[1] == 3:  # 如果是 (B, 3, N) 格式
                x = x.permute(0, 2, 1)
        if ref.shape[-1] != 3:
            if ref.shape[1] == 3:  # 如果是 (B, 3, N) 格式
                ref = ref.permute(0, 2, 1)
                
        if rm_self:
            dist, knn_idx, _ = knn_points(x, ref, K=k+1, return_nn=True)
            dist = dist.squeeze(0)[:, 1:]
            knn_idx = knn_idx.squeeze(0)[:, 1:]
        else:
            dist, knn_idx, _ = knn_points(x, ref, K=k, return_nn=True)
            dist = dist.squeeze(0)
            knn_idx = knn_idx.squeeze(0)
        return torch.sqrt(dist), knn_idx, None

    def compute_force(self, xyz, v):
        # 获取每个锚点的近邻点位置
        knn_xyz = xyz[self.knn_index]
        delta_pos = knn_xyz - xyz.unsqueeze(1)
        curr_len = torch.norm(delta_pos, dim=2)
        norm_delta_pos = delta_pos / (curr_len.unsqueeze(2) + 1e-6)
        
        # 计算弹簧力
        delta_len = curr_len - self.origin_len
        force = (delta_len * self.k).unsqueeze(2) * norm_delta_pos
        
        # 计算阻尼力
        knn_v = v[self.knn_index]
        delta_v = knn_v - v.unsqueeze(1)
        damp_force = (self.damping * torch.sum(delta_v * norm_delta_pos, dim=-1)).unsqueeze(-1) * norm_delta_pos
        
        return (force + damp_force).sum(dim=1)

    def step(self, xyz, v, dt):
        force = self.compute_force(xyz, v)
        v = v + force * dt
        xyz = xyz + v * dt
        return xyz, v

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
        
        # 添加弹簧模型相关属性
        self.spring_models = {}  # 存储每个关键帧的弹簧模型
        self.anchor_points = {}  # 存储每个关键帧的锚点
        self.k_neighbors = 8  # 弹簧连接的近邻点数量
        self.n_anchors = 1000  # 每个关键帧的锚点数量
        
        # 确保 CUDA 在子进程中正确初始化
        torch.cuda.set_device(self.device)
        torch.cuda.init()

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

        # 添加弹簧模型参数
        self.spring_model_enabled = self.config["Training"]["spring_model"]["enabled"]
        if self.spring_model_enabled:
            self.k_neighbors = self.config["Training"]["spring_model"]["k_neighbors"]
            self.n_anchors = self.config["Training"]["spring_model"]["n_anchors"]

    def generate_anchor_points(self, gaussian_points):
        """从高斯点中生成锚点"""
        # 使用体积采样选择锚点
        n_points = gaussian_points.shape[0]
        if n_points <= self.n_anchors:
            return gaussian_points
        
        # 计算点云边界框
        min_coords = torch.min(gaussian_points, dim=0)[0]
        max_coords = torch.max(gaussian_points, dim=0)[0]
        
        # 在边界框内随机采样点
        anchor_points = torch.rand(self.n_anchors, 3, device=self.device)
        anchor_points = anchor_points * (max_coords - min_coords) + min_coords
        
        # 找到每个锚点的最近高斯点
        _, knn_idx, _ = knn_points(anchor_points.unsqueeze(0), gaussian_points.unsqueeze(0), K=1)
        anchor_points = gaussian_points[knn_idx.squeeze(0).squeeze(1)]
        
        return anchor_points

    def interpolate_gaussians(self, anchor_points, gaussian_points, anchor_deltas):
        """使用IDW插值更新高斯点位置"""
        # 计算每个高斯点到锚点的距离
        dist = torch.cdist(gaussian_points, anchor_points)
        weights = 1.0 / (dist + 1e-6)
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # 计算高斯点的位移
        gaussian_deltas = torch.matmul(weights, anchor_deltas)
        
        return gaussian_points + gaussian_deltas

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        
        # 检查 self.gaussians 是否已初始化
        if hasattr(self, 'gaussians') and self.gaussians is not None:
            gaussian_points = self.gaussians.get_xyz
            
            # 确保 gaussian_points 是有效的张量
            if gaussian_points is None or gaussian_points.numel() == 0:
                Log("Warning: No valid gaussian points available")
                self.anchor_points[cur_frame_idx] = None
                self.spring_models[cur_frame_idx] = None
            else:
                # 确保 gaussian_points 具有正确的形状
                if gaussian_points.dim() == 1:
                    gaussian_points = gaussian_points.view(-1, 3)
                elif gaussian_points.dim() == 3:
                    gaussian_points = gaussian_points.squeeze(0)
                    
                Log(f"Gaussian points shape: {gaussian_points.shape}")
                anchor_points = self.generate_anchor_points(gaussian_points)
                self.anchor_points[cur_frame_idx] = anchor_points
                self.spring_models[cur_frame_idx] = SpringModel(anchor_points, self.k_neighbors)
        else:
            # 如果是初始化阶段，创建空的锚点和弹簧模型
            self.anchor_points[cur_frame_idx] = None
            self.spring_models[cur_frame_idx] = None
        
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
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

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
        render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )
        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )

        self.q_main2vis.put(
            gui_utils.GaussianPacket(
                current_frame=viewpoint,
                gtcolor=viewpoint.original_image,
                gtdepth=viewpoint.depth
                if not self.monocular
                else np.zeros((viewpoint.image_height, viewpoint.image_width)),
            )
        )

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

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
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

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
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

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        # 将弹簧模型信息添加到消息中
        spring_model = self.spring_models[cur_frame_idx]
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap, spring_model]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
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
        # 在子进程中重新初始化 CUDA
        torch.cuda.set_device(self.device)
        torch.cuda.init()

        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
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
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
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

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                if self.gt_pose:
                    render_pkg = self.tracking_with_gt_pose(cur_frame_idx, viewpoint)
                else:
                    render_pkg = self.tracking(cur_frame_idx, viewpoint)

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
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
