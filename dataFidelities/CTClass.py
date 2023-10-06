################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other
# DOLCE project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# DOLCE: A Model-Based Probabilistic Diffusion Framework for Limited-Angle CT Reconstruction
################################################################################

import os
import cv2
import sys
import math
sys.stdout.flush()
sys.path.insert(0, "../common")
import numpy as np
import torch
from LEAP_torch import Projector
import subprocess as sp

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim

evaluateSnr = lambda x, xhat: 20*np.log10(np.linalg.norm(x.flatten('F'))/np.linalg.norm(x.flatten('F')-xhat.flatten('F')))

def create_circular_mask(h, w, center=None, radius=None):
    """
    Generate a circular mask to black out none ROI
    return: mask, numpy appary, (H,W)
    """

    if center is None:
        center = (int(h/2)-0.5, int(w/2)-0.5)
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def tensor_normalize(ipt):
    """
    Generate a circular mask to black out none ROI
    return: ipt_normalize, tensor, (N,C,H,W)
    """

    ipt_normalize = ipt.clone()
    bs, C, H, W = ipt_normalize.shape
    ipt_normalize = ipt_normalize.view(ipt_normalize.size(0), -1)
    ipt_normalize -= ipt_normalize.min(1, keepdim=True)[0]
    ipt_normalize /= ipt_normalize.max(1, keepdim=True)[0]
    ipt_normalize = ipt_normalize.view(bs, C, H, W)
    return ipt_normalize

# Datafidelity class
class CTClass:
    """
    Utilities for CTClass using LEAP ToolBox.
    :param projector: projector geometry
    :param size: image size, default, 512
    :param device: if specified, the device to create the geometry on.
    :param gpu_index: if specified, the gpu_index to create the geometry on.
    :param f_fov: if specified, the region of interest to sample within a image. 
    """

    def __init__(self, projector, 
                 size=(512,512), 
                 device=None, 
                 gpu_index=None, 
                 f_fov=None):
        self.M, self.N = size
        self.projector, self.img_fov = projector, f_fov.to(device)
        self.device, self.gpu_index=device, gpu_index

    def train(self):
        self.projector.train()

    def xinit(self, sino):
        x = self.ftran(sino)
        return x
    
    def grad(self, x, sino):
        """
        performs gradient of ||Ax-b||_2^2
        return: gradient wrt image x, (N, C, H, W)
        """
        grad = self.ftran(self.fmult(x) - sino)
        return grad
        
    def cgrad(self, ipt, sino_sample, device, lam=1e5):
        """
        performs CG algorithm
        AtA: a class object that operates forward model
        return: image x, (N, C, H, W)
        """
        z_k = ipt["sample_data"]
        Aty = self.ftran(sino_sample)
        x = torch.zeros_like(z_k)
        rhs = Aty + lam * z_k - self.AtA(x, lam)
        i, r, p = 0, rhs, rhs
        rTr = torch.sum(r.conj()*r)
        while i < 100 and rTr > 1e-10: 
            Ap = self.AtA(p,lam)
            alpha = rTr / torch.sum(p.conj()*Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = torch.sum(r.conj()*r)
            beta = rTrNew / rTr
            p = r + beta * p
            i += 1
            rTr = rTrNew
        return x
    
    def apgm(self, ipt, sino_sample, rho, device):
        """
        performs APGM algorithm
        return: image x, (N, C, H, W)
        """
        accelerate=True
        x = ipt["sample_data"].detach().cpu().clone().to(device)
        z = ipt["sample_data"].detach().cpu().clone().to(device)
        s = x
        t = torch.tensor([1.]).float().to(device)
        for _ in range(30):
            grad = self.grad(s, sino_sample)
            xnext = s - 5e-6*grad - rho*(s - z)
            # acceleration
            if accelerate:
                tnext = 0.5*(1+torch.sqrt(1+4*t*t))
            else:
                tnext = 1
            s = xnext + ((t-1)/tnext)*(xnext-x)
            # update
            t = tnext
            x = xnext
        return x    

    def prox_solver(self, ipt, sino_sample, rho, solver_mode, device):
        """
        This code solves the following optimization problem:
        argmin_x ||Ax-b||_2^2 + ||x-u||^2_2       
        return: image x, (N, C, H, W)
        """
        if solver_mode=='apgm':
            x = self.apgm(ipt, sino_sample, rho, device)
        elif solver_mode=='cgrad':
            x = self.cgrad(ipt, sino_sample, device, 2.5e5)
        return x
    
    def eval(self, x, sino):
        """
        Estimate of ||Ax-b||_2^2   
        """
        data_fit=0.5*torch.pow(torch.abs(self.fmult(x) - sino),2).mean()
        return data_fit

    def AtA(self, x, lam=0.01):
        """
        Implementation of AtA in Conjugate gradient method 
        return: image x, (N, C, H, W)    
        """        
        return self.ftran(self.fmult(x)) + lam*x

    def fmult(self, x, project_mode="forward"):
        """
        Implementation of Ax
        return: sinogram b, (N, 1, thetas, detactors)
        """           
        sino = self.projector(x, project_mode)
        return sino    

    def ftran(self, sino,  project_mode="backward", clip=False):
        """
        Implementation of Atb
        return: image x, (N, C, H, W)
        """         
        x = self.projector(sino, project_mode)
        if clip:
            x[x<=0] = 0
        return x

    @staticmethod
    def eval_psnr(img1, img2, data_range=1.):
        img1.astype(np.float32)
        img2.astype(np.float32)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        return 20 * math.log10(data_range / math.sqrt(mse))

    @staticmethod
    def eval_ssim(img1, img2, data_range=1.):
        return ssim(img1, img2, data_range=data_range)

    @staticmethod
    def np_normalize(ipt):
        """
        Generate a circular mask to black out none ROI
        return: ipt_normalize, numpy arr, (N,C,H,W)
        """
        return (ipt - ipt.min())/(ipt.max() - ipt.min())

    @staticmethod
    def imsave(img, img_path):
        """
        Save image in the gaven img_path
        :param img_path: path to save, str.
        """
        img = np.squeeze(img)
        if img.ndim == 3:
            img = img[:, :, [2, 1, 0]]
        cv2.imwrite(img_path, img)    
        
    @staticmethod
    def save_video(imgs_path, save_path, fps=10):
        cmd = [
            "ffmpeg",
            "-framerate",
            str(fps),
            "-i",
            str(imgs_path) + "/%d.png",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            str(save_path),
        ]
        sp.run(cmd)

def set_CTClass(device, gpu_index, use_cuda=True, 
                batch_size=1, use_static=False, use_fov=True, param_fn=None, rank=0, logger=None):
    """
    Initialize CT geometry using LEAP.
    More Ref: https://github.com/LLNL/LEAP
    return: a Class that implements parallel beam CT.
    """    

    ################################################################################
    # 1. Read or simulate F and G using LEAP
    ################################################################################
    # read arguments
    if param_fn is None:
        param_fn = "./config/param_parallel512_la60.cfg"
        if not os.path.isfile:
            raise(FileNotFoundError)

    # initialize projector and load parameters
    proj = Projector(forward_project=None, 
                     use_static=use_static, 
                     use_gpu=use_cuda, 
                     gpu_device=device, 
                     batch_size=batch_size)
    if logger is not None:
       logger.log(f"Param setting: {param_fn}")
    proj.load_param(param_fn)
    proj.set_projector(1)
    if rank==0:
        proj.print_param()
    dimz, dimy, dimx = proj.get_volume_dim()
    views, rows, cols = proj.get_projection_dim()
    M, N = dimz, dimx
    # set mask for field of view
    if use_fov:
        f_fov = create_circular_mask(N, N).reshape(1, M, N, N).astype(np.float32)
        f_fov = torch.from_numpy(f_fov)
        if rank==0 and logger is not None:
            logger.log("Field of view masking is used")
    else:
        f_fov = None
    return CTClass(proj, (M,N), device, gpu_index, f_fov)
