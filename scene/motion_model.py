#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
import json
from utils.rigid_utils import rp_to_se3
from utils.time_utils import get_embedder
import os
from utils.system_utils import searchForMaxIteration

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]

class MotionNetwork(nn.Module):
    def __init__(self, D=4, W=256, input_ch=3, output_ch=59, multires=10, is_blender=True, is_6dof=False, obj_num = 1):
        super(MotionNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.multires = 1
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]
        self.obj_num = obj_num

        
        self.is_blender = is_blender
        self.is_6dof = is_6dof



    def init(self):
        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        embed_fn, xyz_input_ch = get_embedder(self.multires, self.obj_num)

        obj_lab = torch.nn.functional.one_hot(torch.arange(0, self.obj_num).cuda().long()).float()

        self.obj_feat = embed_fn(obj_lab)


        self.input_ch = time_input_ch + xyz_input_ch
        if self.is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30

            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out)).cuda()

            self.linear = nn.ModuleList(
                [nn.Linear(self.time_out, self.W)] + [
                    nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.time_out, self.W)
                    for i in range(self.D - 1)]
            ).cuda()

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, self.W)] + [
                    nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.input_ch + xyz_input_ch, self.W)
                    for i in range(self.D - 1)]
            ).cuda()

        

        if self.is_6dof:
            self.branch_w = nn.Linear(self.W, 3).cuda()
            self.branch_v = nn.Linear(self.W, 3).cuda()
        else:
            self.gaussian_warp = nn.Linear(self.W, 3).cuda()
        self.gaussian_rotation = nn.Linear(self.W, 4).cuda()

        self.init_linear()

    def init_linear(self):
        nn.init.constant_(self.gaussian_rotation.weight, 0.0)
        init_bias = torch.tensor([1.0, 0.0, 0.0, 0.0], device='cuda', requires_grad=True)
        self.gaussian_rotation.bias = torch.nn.parameter.Parameter(init_bias)
        # nn.init.constant_(self.gaussian_rotation.bias, 0.0)
        nn.init.constant_(self.gaussian_warp.weight, 0.0)
        nn.init.constant_(self.gaussian_warp.bias, 0.0)


    def forward(self, t):
        t_emb = self.embed_time_fn(t)

        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset

        h = torch.cat([t_emb], dim=-1)

        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([t_emb, h], -1)

        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            # d_xyz = exp_se3(screw_axis, theta)
        else:
            d_xyz = self.gaussian_warp(h)
        
        rotation = self.gaussian_rotation(h)



        return (d_xyz, rotation)

class MotionModel:
    def __init__(self):

        self._rots = torch.empty(0)
        self._locs = torch.empty(0)
        self.motion_model = None
        self.obj_lst = ["obj"]

        


    def get_rot(self, idx):
        if isinstance(idx, str):
            idx = int(idx.split("_")[-1])
        assert idx < self._rots.shape[0]
        idx = torch.tensor(idx).to(device = self._rots.device)
        rot = self._rots.index_select(dim=0, index=idx).squeeze(0)
        return rot
        
    def get_loc(self, idx):
        if isinstance(idx, str):
            idx = int(idx.split("_")[-1])
        assert idx < self._locs.shape[0]
        idx = torch.tensor(idx).to(device = self._locs.device)
        loc = self._locs.index_select(dim=0, index=idx).squeeze(0)
        return loc
        
    def get_motion(self, idx):
        if isinstance(idx, str):
            idx = int(idx.split("_")[-1])
        assert idx < self._rots.shape[0]
        if isinstance(idx, int):
            rot = self.get_rot(idx)
            loc = self.get_loc(idx)
        motion = (loc, rot)
        # print(loc.shape, rot.shape)

        return motion
    
    def pred_motion(self, t):
        return self.motion_model(t)
        
    def init_pose(self, json_path, use_gt = True):
        with open(json_path, 'r') as fp:
            camera_json = json.load(fp)
        self.obj_lst = camera_json["obj_lst"]

        self.motion_model = MotionNetwork(obj_num = len(self.obj_lst)).cuda()
        self.motion_model.init()

        camera_json = camera_json["frames"]
        rots, locs = [], []
        for frame in camera_json:
            if use_gt:
                pose = frame["obj_mat"]
                pose = torch.tensor(np.array(pose, dtype=np.float32), device='cuda')
                
            else:
                pose = torch.eye(4, device='cuda').unsqueeze(0).repeat(len(self.obj_lst), 1, 1)
            
            pose[:, :3, 1:3] *= -1
            rot = pose[:, :3, :3]
            loc = pose[:, :3, 3]

            # if not use_gt:
            rot = rot.unsqueeze(0)
            loc = loc.unsqueeze(0)

            rots.append(rot)
            locs.append(loc)

        rots = torch.cat(rots, dim=0)
        locs = torch.cat(locs, dim=0)

        rots = matrix_to_quaternion(rots)

        self._rots = nn.Parameter(rots.requires_grad_(True))

        self._locs = nn.Parameter(locs.requires_grad_(True))
    
    def init_obj(self, json_path):
        try:
            with open(json_path, 'r') as fp:
                camera_json = json.load(fp)
            if "obj_lst" in camera_json:
                self.obj_lst = camera_json["obj_lst"]
        except:
            self.obj_lst = ["obj"]
        self.motion_model = MotionNetwork(obj_num = len(self.obj_lst)).cuda()
        self.motion_model.init()
        
        

    def training_setup(self, training_args):
        self.spatial_lr_scale = 5

        l = [
            {'params': list(self.motion_model.parameters()),
             'lr': training_args.motion_lr_init * self.spatial_lr_scale,
             "name": "motion"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.motion_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.motion_lr_max_steps)
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "motion":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            
    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "motion/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.motion_model.state_dict(), os.path.join(out_weights_path, 'motion.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "motion"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "motion/iteration_{}/motion.pth".format(loaded_iter))
        self.motion_model.load_state_dict(torch.load(weights_path))



    

    