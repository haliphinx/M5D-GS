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
import math

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous, rp_to_se3
from utils.graphics_utils import getWorld2View2
from scene.motion_model import quaternion_to_matrix, quaternion_apply, quaternion_invert, quaternion_raw_multiply


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)

use_feat_gs = False

if use_feat_gs:
    pass
    # from feat_diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    # def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, is_6dof=False,
    #        scaling_modifier=1.0, override_color=None, offset_mat = None, is_warm_up = True, obj_idx = None):
    #     """
    #     Render the scene. 
        
    #     Background tensor (bg_color) must be on GPU!
    #     """
        
    #     # offset_mat = None

    #     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    #     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    #     try:
    #         screenspace_points.retain_grad()
    #     except:
    #         pass

    #     # Set up rasterization configuration
    #     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    #     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    #     semantic_feat = pc.get_semantic
    #     # print(semantic_feat)
    #     # dd

    #     if offset_mat is not None:
    #         rot = offset_mat[1]
    #         loc = offset_mat[0]
    #         gs_label = pc.get_label
    #         # print(offset_mat)
    #         rot = rot.index_select(dim=0, index=gs_label)
    #         loc = loc.index_select(dim=0, index=gs_label)
    #         # print(loc.shape, rot.shape)
    #     else:
    #         rot,loc = None, None
        
    #     # if not is_warm_up:
    #     #     # rot = rot.detach()
    #     #     # loc = loc.detach()
    #     #     offset_mat = None


    #     if is_6dof:
    #         if torch.is_tensor(d_xyz) is False:
    #             means3D = pc.get_xyz
    #         else:
    #             means3D = from_homogenous(
    #                 torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    #     else:
    #         means3D = pc.get_xyz

    #         if offset_mat is not None:
    #             # gs_label = pc.get_label
    #             # matrix vesion
    #             # means3D = to_homogenous(means3D)
    #             # if not is_warm_up:
    #             #     means3D = torch.matmul(means3D, offset_mat.detach().to(means3D))
    #             # else:
    #             #     means3D = torch.matmul(means3D, offset_mat.detach().to(means3D))
    #             # means3D = from_homogenous(means3D)

    #             # quaternion vesion
    #             # rot = offset_mat[0].index_select(dim=0, index=gs_label).detach()
    #             # loc = offset_mat[1].index_select(dim=0, index=gs_label).detach()
    #             # print(rot.shape, loc.shape, )
    #             # dd
    #             # if not is_warm_up:
    #             #     rot = rot.detach()
    #             #     loc = loc.detach()
    #             # q_rot = matrix_to_quaternion(rot)
    #             assert rot is not None and loc is not None
    #             # if not is_warm_up:
    #             # print(rot.shape, means3D.shape)
    #             # dd
    #             means3D = quaternion_apply(rot, means3D)
    #             means3D += loc


                
    #         means3D = means3D + d_xyz

        
    #     means2D = screenspace_points
    #     opacity = pc.get_opacity

    #     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    #     # scaling / rotation by the rasterizer.
    #     scales = None
    #     rotations = None
    #     cov3D_precomp = None
    #     if pipe.compute_cov3D_python:
    #         cov3D_precomp = pc.get_covariance(scaling_modifier)
    #     else:
    #         scales = pc.get_scaling + d_scaling
    #         rotations = pc.get_rotation
    #         if offset_mat is not None:
    #             assert rot is not None
    #             # rot = offset_mat[0].index_select(dim=0, index=gs_label).detach()
    #             # quat_inv = quaternion_invert(rot)
    #             rotations = quaternion_raw_multiply(rot, rotations)
    #         rotations = rotations + d_rotation

    #     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    #     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    #     shs = None
    #     colors_precomp = None
    #     if colors_precomp is None:
    #         if pipe.convert_SHs_python:
    #             shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    #             dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    #             dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    #             sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #             colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #         else:
    #             shs = pc.get_features
    #     else:
    #         colors_precomp = override_color

    #     # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    #     # viewmat = viewpoint_camera.world_view_transform
    #     # projectmat = viewpoint_camera.full_proj_transform
        

    #     # if not is_warm_up and offset_mat is not None:
    #     #     offset_mat = rp_to_se3(quaternion_to_matrix(rot1), loc1.unsqueeze(-1))
    #     #     offset_mat = offset_mat.to(viewmat)
    #     #     viewmat_obj = viewpoint_camera.world_view_transform
    #     #     viewmat = torch.matmul(offset_mat,viewmat_obj)
    #     #     projectmat = viewpoint_camera.get_extrinsic(viewmat)


    #     raster_settings = GaussianRasterizationSettings(
    #         image_height=int(viewpoint_camera.image_height),
    #         image_width=int(viewpoint_camera.image_width),
    #         tanfovx=tanfovx,
    #         tanfovy=tanfovy,
    #         bg=bg_color,
    #         scale_modifier=scaling_modifier,
    #         viewmatrix=viewpoint_camera.world_view_transform,
    #         projmatrix=viewpoint_camera.full_proj_transform,
    #         sh_degree=pc.active_sh_degree,
    #         campos=viewpoint_camera.camera_center,
    #         prefiltered=False,
    #         debug=pipe.debug,
    #     )
    #     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    #     if obj_idx is not None:
    #         rendered_mask = pc.get_label == obj_idx
    #         means3D=means3D[rendered_mask]
    #         means2D=means2D[rendered_mask]
    #         shs=shs[rendered_mask]
    #         colors_precomp=None if colors_precomp is None else colors_precomp[rendered_mask]
    #         opacity=opacity[rendered_mask]
    #         scales=scales[rendered_mask]
    #         rotations=rotations[rendered_mask]
    #         cov3D_precomp=None if cov3D_precomp is None else cov3D_precomp[rendered_mask]

    #     # print(means3D.dtype, means2D.dtype, shs.dtype, semantic_feat.dtype)
    #     # dd
    #     # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    #     rendered_image, feature_map, radii = rasterizer(
    #         means3D = means3D,
    #         means2D = means2D,
    #         shs = shs,
    #         colors_precomp = colors_precomp,
    #         semantic_feature = semantic_feat, 
    #         opacities = opacity,
    #         scales = scales,
    #         rotations = rotations,
    #         cov3D_precomp = cov3D_precomp)
        


    #     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    #     # They will be excluded from value updates used in the splitting criteria.
    #     return {"render": rendered_image,
    #             "viewspace_points": screenspace_points,
    #             "visibility_filter" : radii > 0,
    #             "radii": radii,
    #             'feature_map': feature_map}

else:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, is_6dof=False,
            scaling_modifier=1.0, override_color=None, offset_mat = None, is_warm_up = True, obj_idx = None):
        """
        Render the scene. 
        
        Background tensor (bg_color) must be on GPU!
        """
        
        # offset_mat = None

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        if offset_mat is not None:
            rot = offset_mat[1]
            
            loc = offset_mat[0]

            # for the scene w/o init point cloud
            if not is_warm_up:
                loc *= torch.norm(rot, p=2)
                rot = torch.nn.functional.normalize(rot)

                
            gs_label = pc.get_label
            rot = rot.index_select(dim=0, index=gs_label)
            loc = loc.index_select(dim=0, index=gs_label)

        else:
            rot,loc = None, None


        if is_6dof:
            if torch.is_tensor(d_xyz) is False:
                means3D = pc.get_xyz
            else:
                means3D = from_homogenous(
                    torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
        else:
            means3D = pc.get_xyz

            if offset_mat is not None:
                assert rot is not None and loc is not None
                means3D = quaternion_apply(rot, means3D)
                means3D += loc


                
            means3D = means3D + d_xyz

        
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling + d_scaling
            rotations = pc.get_rotation
            if offset_mat is not None:
                assert rot is not None
                rotations = quaternion_raw_multiply(rot, rotations)
            rotations = rotations + d_rotation
        # rotations *= 0.4
        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        projectmat = viewpoint_camera.full_proj_transform



        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=projectmat,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
        )

    
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        if obj_idx is not None:
            rendered_mask = pc.get_label == obj_idx
            means3D=means3D[rendered_mask]
            means2D=means2D[rendered_mask]
            shs=shs[rendered_mask]
            colors_precomp=None if colors_precomp is None else colors_precomp[rendered_mask]
            opacity=opacity[rendered_mask]
            scales=scales[rendered_mask]
            rotations=rotations[rendered_mask]
            cov3D_precomp=None if cov3D_precomp is None else cov3D_precomp[rendered_mask]

        rendered_image, radii, depth = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "depth": depth}
