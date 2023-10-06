################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other
# DOLCE project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# DOLCE: A Model-Based Probabilistic Diffusion Framework for Limited-Angle CT Reconstruction
################################################################################

"""
Generate a large batch of samples from a DOLCE model.
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import os

import torch
import argparse
import numpy as np
from functools import partial
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import dataset2run
from guided_diffusion.script_util import (
    condtion_model_and_diffusion_defaults,
    condtion_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from dataFidelities.CTClass import set_CTClass

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device, gpu_index, device_name = dist_util.dev()
    logger.configure(data_set=args.data_dir, num_angles=args.num_angs)
    #############################
    logger.log("creating testing dataset...")
    data = dataset2run(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        angle_range=args.num_angs,
        deterministic=args.deterministic
    )
    data_len = data.len_data()
    args.num_samples = args.num_samples if args.num_samples<=data_len else data_len

    logger.log("creating denoising model...")
    model, diffusion = condtion_create_model_and_diffusion(
        **args_to_dict(args, condtion_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    logger.log("creating data model...") 
    dObj = set_CTClass(device, gpu_index, use_cuda=True, 
                       batch_size=args.batch_size, use_static=False,
                       param_fn=args.param_fn, rank=dist.get_rank(), logger=logger)
    
    run_sampler = partial(diffusion.sample_loop, model_kwargs_data=dObj)
    
    logger.log("Successfully loaded conditional model. ")
    logger.log("creating samples...")
    
    all_images_data, all_images_gt, all_images_rls = [], [], []
    
    for gt, model_kwargs in data.load_data():

        gt = gt.to(device)
        model_kwargs = {k: v.to(device) for k, v in model_kwargs.items()}
        sample_data, fbp, rls = run_sampler(
            model,
            gt, #only used for generating sine online
            eta=args.eta,
            shape=gt.shape,
            samper=args.sampler,
            prox_solver=args.prox_solver,
            model_kwargs=model_kwargs,
            clip_denoised=args.clip_denoised,
        )
        sample_data=torch.mean(sample_data,0,keepdim=True)
        sample_data = sample_data.permute(0, 2, 3, 1)
        gt = gt[0,...].unsqueeze(0).contiguous()
        rls = rls[0,...].unsqueeze(0).contiguous()
        sample_data = sample_data.contiguous()
        all_gt = [torch.zeros_like(gt) for _ in range(dist.get_world_size())]
        all_rls = [torch.zeros_like(rls) for _ in range(dist.get_world_size())]
        all_samples_data = [torch.zeros_like(sample_data) for _ in range(dist.get_world_size())]
        dist.all_gather(all_gt, gt)  # gather not supported with NCCL
        dist.all_gather(all_rls, rls)  # gather not supported with NCCL
        dist.all_gather(all_samples_data, sample_data)  # gather not supported with NCCL

        for sample_data in all_samples_data:
            all_images_data.append(sample_data.cpu().numpy())
        for gt in all_gt:
            all_images_gt.append(gt.cpu().numpy())
        for rls in all_rls:
            all_images_rls.append(rls.cpu().numpy())  
        logger.log(f"created {len(all_images_data) * args.batch_size} samples")
        if len(all_images_data) * args.batch_size >= args.num_samples:
            break
    
    arr_data = np.concatenate(all_images_data, axis=0)
    arr_gt = np.concatenate(all_images_gt, axis=0)
    arr_rls = np.concatenate(all_images_rls, axis=0)
    
    arr_data = arr_data.transpose([1,2,3,0])[...,:args.num_samples]
    arr_gt = arr_gt.transpose([2,3,1,0])[...,:args.num_samples]
    arr_rls = arr_rls.transpose([2,3,1,0])[...,:args.num_samples]
    logger.log(f"used {args.num_samples} samples")
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr_data.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr_gt, arr_data, arr_rls)

        logger.log("calculating ssim/psnr")
        psnr_all_data, ssim_all_data = [],  []
        for ii in range(arr_gt.shape[-1]):

            arr_gdt_each = arr_gt[...,ii].squeeze()
            arr_rls_each = arr_rls[...,ii].squeeze()
            arr_each = arr_data[...,ii].squeeze()
            psne_each = dObj.eval_psnr(arr_gdt_each, arr_each, data_range=arr_gdt_each.max() - arr_gdt_each.min())
            ssim_each = dObj.eval_ssim(arr_gdt_each, arr_each, data_range=arr_gdt_each.max() - arr_gdt_each.min())
            psnr_all_data.append(psne_each)
            ssim_all_data.append(ssim_each)
            if args.save_fig:
                save_fig_pth = os.path.join(logger.get_dir(), f"samples_{ii}_ang={args.num_angs}.png")
                arr_all = np.concatenate(
                    [dObj.np_normalize(arr_gdt_each),
                     dObj.np_normalize(arr_rls_each),
                     dObj.np_normalize(arr_each)],
                    axis=1,)
                dObj.imsave(255*arr_all, save_fig_pth)
        psnr_all_data, ssim_all_data = np.array(psnr_all_data), np.array(ssim_all_data)
        # if args.save_fig:
        #     dObj.save_video(logger.get_dir(), logger.get_dir())

        logger.log(f"PSNR_MEAN_DATA: {psnr_all_data.mean()}, PSNR_MAX_DATA: {psnr_all_data.max()}, PSNR_MIN_DATA: {psnr_all_data.min()}")
        logger.log(f"SSIM_MEAN_DATA: {ssim_all_data.mean()}, SSIM_MAX_DATA: {ssim_all_data.max()}, SSIM_MIN_DATA: {ssim_all_data.min()}")
    dist.barrier()
    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        sampler='ddpm',       #"ddpm" or "ddim"
        prox_solver='apgm',  #"apgm" or "cgrad"
        use_condtion='rls',   #"rls" or "fbp"
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,         #bs = 1 works the best
        data_dir="",
        model_path="",
        param_fn="",
        sub_id="",
        num_angs=60,          #60, 90, 120
        eta=1.,               #used in ddim
        seed=12345, 
        weighted_condition=False,
        save_fig=True,
        root_path=None,
        deterministic=False,
    )
    defaults.update(condtion_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
