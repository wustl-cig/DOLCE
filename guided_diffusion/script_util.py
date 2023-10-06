################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other
# DOLCE project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# DOLCE: A Model-Based Probabilistic Diffusion Framework for Limited-Angle CT Reconstruction
################################################################################

import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import ConditionalModel

def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        eta=1.,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        predict_xstart=False,
    )

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=512,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        use_checkpoint=False,
        resblock_updown=False,
        use_fp16=True,
        use_new_attention_order=False,
        num_add_res=2,
        use_basemodel=False,
        weighted_condition=False,
        use_condtion='rls',
    )
    res.update(diffusion_defaults())
    return res

def condtion_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["image_size"] = 512
    arg_names = inspect.getfullargspec(condtion_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res

def condtion_create_model_and_diffusion(
    image_size,
    num_channels,
    num_res_blocks,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_checkpoint,
    resblock_updown,
    use_fp16,
    num_add_res,
    weighted_condition,
    use_condtion,    
):
    model = condtion_create_model(
        image_size,
        num_channels,
        num_res_blocks,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        num_add_res=num_add_res,
        weighted_condition=weighted_condition,
        use_condtion=use_condtion,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion

def condtion_create_model(
    image_size,
    num_channels,
    num_res_blocks,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    dropout,
    resblock_updown,
    use_fp16,
    num_add_res,
    weighted_condition,
    use_condtion,
):

    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
        
    return ConditionalModel(
        image_size=image_size,
        in_channels=1,
        model_channels=num_channels,
        out_channels=2,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        num_add_res=num_add_res,            
        weighted_condition=weighted_condition,
        use_condtion=use_condtion,
    )

def create_gaussian_diffusion(
    *,
    steps=1000,
    noise_schedule="linear",
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)

    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
    )

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
