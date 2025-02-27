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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

import numpy as np

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

def validate_rasterize_args(args):
    # Unpack the arguments
    (
        bg, 
        means3D,
        colors_precomp,
        opacities,
        scales,
        rotations,
        scale_modifier,
        cov3Ds_precomp,
        viewmatrix,
        projmatrix,
        tanfovx,
        tanfovy,
        image_height,
        image_width,
        sh,
        sh_degree,
        campos,
        prefiltered,
        debug
    ) = args

    def check_tensor(tensor, name):
        if not isinstance(tensor, torch.Tensor):
            print(f"Error: {name} is not a tensor.")
            return
        if not tensor.is_cuda:
            print(f"Error: {name} is not on the GPU.")
        if not tensor.is_contiguous():
            print(f"Error: {name} is not contiguous.")
        if torch.isnan(tensor).any():
            print(f"Error: {name} contains NaNs.")
        if torch.isinf(tensor).any():
            print(f"Error: {name} contains Infs.")
        print(f"{name}: Shape = {tuple(tensor.shape)}, Dtype = {tensor.dtype}, Device = {tensor.device}")

    def check_array(arr, name):
        if not isinstance(arr, np.ndarray):
            print(f"Error: {name} is not a numpy array.")
            return
        if np.isnan(arr).any():
            print(f"Error: {name} contains NaNs.")
        if np.isinf(arr).any():
            print(f"Error: {name} contains Infs.")
        print(f"{name}: Shape = {arr.shape}, Dtype = {arr.dtype}")

    def check_scalar(scalar, name):
        if not isinstance(scalar, (int, float)):
            print(f"Error: {name} is not a scalar.")
            return
        if np.isnan(scalar):
            print(f"Error: {name} is NaN.")
        if np.isinf(scalar):
            print(f"Error: {name} is Inf.")
        print(f"{name}: Value = {scalar}")

    # Check tensors
    check_tensor(means3D, "means3D")
    check_tensor(colors_precomp, "colors_precomp")
    check_tensor(opacities, "opacities")
    check_tensor(scales, "scales")
    check_tensor(rotations, "rotations")
    check_tensor(cov3Ds_precomp, "cov3Ds_precomp")
    check_tensor(sh, "sh")

    # Check arrays/matrices
    check_array(viewmatrix, "viewmatrix")
    check_array(projmatrix, "projmatrix")
    check_array(campos, "campos")

    # Check scalars
    check_scalar(scale_modifier, "scale_modifier")
    check_scalar(tanfovx, "tanfovx")
    check_scalar(tanfovy, "tanfovy")
    check_scalar(image_height, "image_height")
    check_scalar(image_width, "image_width")
    check_scalar(sh_degree, "sh_degree")

    # Check boolean flags
    if not isinstance(prefiltered, bool):
        print("Error: prefiltered is not a boolean.")
    else:
        print(f"prefiltered: {prefiltered}")

    if not isinstance(debug, bool):
        print("Error: debug is not a boolean.")
    else:
        print(f"debug: {debug}")

    # Check background
    if not isinstance(bg, (torch.Tensor, np.ndarray)):
        print("Error: bg is neither a tensor nor a numpy array.")
    else:
        print(f"bg: Type = {type(bg)}, Dtype = {bg.dtype}, Shape = {bg.shape if isinstance(bg, np.ndarray) else tuple(bg.shape)}")



class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        print("MEANS 3D SIZE: ", means3D.size(0), " --------------------------------------------------------")
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        print("IN RASTERIZE GAUSSIANS###################")

        print("means3D NaN:", torch.isnan(means3D).any(), "Inf:", torch.isinf(means3D).any())
        print("means2D NaN:", torch.isnan(means2D).any(), "Inf:", torch.isinf(means2D).any())


        if colors_precomp is not None:
            print("colors_precomp NaN:", torch.isnan(colors_precomp).any(), "Inf:", torch.isinf(colors_precomp).any())

        print("opacities NaN:", torch.isnan(opacities).any(), "Inf:", torch.isinf(opacities).any())
        print("scales NaN:", torch.isnan(scales).any(), "Inf:", torch.isinf(scales).any())
        print("rotations NaN:", torch.isnan(rotations).any(), "Inf:", torch.isinf(rotations).any())

        print("means3D is contiguous:", means3D.is_contiguous())
        print("means2D is contiguous:", means2D.is_contiguous())
        print("scales is contiguous:", scales.is_contiguous())
        print("rotations is contiguous:", rotations.is_contiguous())
        print("sh is contiguous:", sh.is_contiguous())
        print("opacities is contiguous:", opacities.is_contiguous())

        # Run the validation
        validate_rasterize_args(args)

        # import pdb; pdb.set_trace()
        # Invoke C++/CUDA rasterizer


        for idx, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                print(f"Arg {idx}: NaN: {torch.isnan(arg).any()}, Inf: {torch.isinf(arg).any()}")
                # print(f"Arg {idx} Min: {arg.min()}, Max: {arg.max()}")
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer, pixel_gaussian_counter = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer, pixel_gaussian_counter = _C.rasterize_gaussians(*args)

        print("Predicted image contains NaN: ", torch.isnan(color).any())
        print("Predicted image contains Inf: ", torch.isinf(color).any())
        # for idx, arg in enumerate(args):
        #     if isinstance(arg, torch.Tensor):
        #         print(f"Arg {idx}: NaN: {torch.isnan(arg).any()}, Inf: {torch.isinf(arg).any()}")
        #         # print(f"Arg {idx} Min: {arg.min()}, Max: {arg.max()}")

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)

        return color, depth, radii, pixel_gaussian_counter

    @staticmethod
    def backward(ctx, grad_out_color, *args_):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads
    


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
            print("shs none")
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
            print("colors precomp none ")

        if scales is None:
            scales = torch.Tensor([])
            print("scales none")
        if rotations is None:
            rotations = torch.Tensor([])
            print("rotations none")
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
            print("cov3d none")

        print("IN THE FORWARD PASS###################")

        print("means3D NaN:", torch.isnan(means3D).any(), "Inf:", torch.isinf(means3D).any())
        print("means2D NaN:", torch.isnan(means2D).any(), "Inf:", torch.isinf(means2D).any())

        if shs is not None:
            print("shs NaN:", torch.isnan(shs).any(), "Inf:", torch.isinf(shs).any())

        if colors_precomp is not None:
            print("colors_precomp NaN:", torch.isnan(colors_precomp).any(), "Inf:", torch.isinf(colors_precomp).any())

        print("opacities NaN:", torch.isnan(opacities).any(), "Inf:", torch.isinf(opacities).any())
        print("scales NaN:", torch.isnan(scales).any(), "Inf:", torch.isinf(scales).any())
        print("rotations NaN:", torch.isnan(rotations).any(), "Inf:", torch.isinf(rotations).any())

        if cov3D_precomp is not None:
            print("cov3D_precomp NaN:", torch.isnan(cov3D_precomp).any(), "Inf:", torch.isinf(cov3D_precomp).any())

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

