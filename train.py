"""
The training script for Ctrl-Adapter

If you wish to train on a different backbone,
you can use Ctrl+F and search for ### to find the section of code we highlighted 
"""

import os 
import random
import socket
import argparse
import math
import traceback
import gc
import time
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from datetime import timedelta
from omegaconf import OmegaConf

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils.dataclasses import InitProcessGroupKwargs
from accelerate.logging import get_logger
logger = get_logger(os.path.realpath(__file__) if '__file__' in locals() else os.getcwd())

from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
if is_wandb_available():
    import wandb

# import adapter, router, and helper
from model.ctrl_adapter import ControlNetAdapter
from model.ctrl_router import ControlNetRouter
from model.ctrl_helper import ControlNetHelper

### import util functions. You can create a utils_xxx.py file following the same file structure for a new backbone
from utils.utils import bool_flag, count_params
from utils.utils_svd import add_time_ids_svd, rand_log_normal, _convert_to_karras, sample_svd_sigmas_timesteps
from utils.data_loader import VideoLoader, ImageLoader

# validation every n training steps
from inference import inference_main

# import our modified controlnet
from controlnet.controlnet import ControlNetModel



def parse_args():
    
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # general 
    parser.add_argument("--yaml_file", 
        type=str,  default="sdxl_train_depth.yaml", 
        help="paths to training configs."
        )
    parser.add_argument(
        "--nccl_timeout", 
        type=int, default=36000, 
        help="nccl_timeout"
        )
    parser.add_argument(
        "--project_name", 
        type=str, default="Ctrl-Adapter Training", 
        help="the name of the run",
        )
    parser.add_argument(
        "--report_to", 
        type=str, default="wandb", 
        help=('The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.')
        )
    
    # optimizer
    parser.add_argument(
        "--learning_rate", 
        type=float, default=5e-5, 
        help="Initial learning rate (after the potential warmup period) to use.",
        )
    parser.add_argument(
        "--scale_lr", 
        action="store_true", default=False, 
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
        )
    parser.add_argument(
        "--lr_scheduler", 
        type=str, default="constant",
        help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'),
        )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, default=500, 
        help="Number of steps for the warmup in the lr scheduler."
        )
    parser.add_argument(
        "--use_8bit_adam", 
        action="store_true", 
        help="Whether or not to use 8-bit Adam from bitsandbytes."
        )
    parser.add_argument(
        "--adam_beta1", 
        type=float, default=0.9, 
        help="The beta1 parameter for the Adam optimizer."
        )
    parser.add_argument(
        "--adam_beta2", 
        type=float, default=0.999, 
        help="The beta2 parameter for the Adam optimizer."
        )
    parser.add_argument(
        "--adam_weight_decay", 
        type=float, default=1e-2, 
        help="Weight decay to use."
        )
    parser.add_argument(
        "--adam_epsilon", 
        type=float, default=1e-08, 
        help="Epsilon value for the Adam optimizer"
        )
    parser.add_argument(
        "--max_grad_norm", 
        default=1.0, type=float, 
        help="Max gradient norm."
        )
    parser.add_argument(
        "--disable_optimizer_restore", 
        action="store_true"
        )
    
    # training 
    parser.add_argument(
        "--seed", 
        type=int, default=111, 
        help="A seed for reproducible training."
        )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, default=1
        )
    parser.add_argument(
        "--max_train_steps", 
        type=int, default=9999999, 
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
        )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, default=1, 
        help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
    parser.add_argument(
        "--gradient_checkpointing", 
        action="store_true", 
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
        )
    parser.add_argument(
        "--noise_offset", 
        type=float, default=0.05, 
        help="The scale of noise offset."
        )
    parser.add_argument(
        "--mixed_precision", 
        type=str, default='bf16', choices=["no", "fp16", "bf16"], 
        help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU."),
        )
    parser.add_argument(
        "--snr_gamma", 
        type=float, default=None, 
        help="Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556."
        )
    parser.add_argument(
        "--max_vae_encode", 
        type=int, default=None
        )
    parser.add_argument(
        "--vae_b16", 
        action="store_true"
        )
    parser.add_argument(
        "--latent_nan_checking", 
        action="store_true", 
        help="Check if latents contain nans - important if vae is f16",
        )
    parser.add_argument(
        "--xformers", 
        action="store_true"
        )

    # depth estimation 
    parser.add_argument(
        "--use_midas_depth_estimator", 
        type=bool_flag, default=True, 
        help="Whether or not to use depth estimator from midas library. If True, need to download checkpoint dpt_swin2_large_384 from \
        website: https://github.com/isl-org/MiDaS/tree/master?tab=readme-ov-file \
        If False, we'll use the depth estimator from transformers library (which is much slower than dpt_swin2_large_384)"
    )
    # checkpoint saving 
    parser.add_argument(
        "--save_n_steps", 
        type=int, default=5000, 
        help="Save the model every n args.global_steps",
        )
    parser.add_argument(
        "--save_starting_step", 
        type=int, default=5000, 
        help="The step from which it starts saving intermediary checkpoints",
        )
    parser.add_argument(
        "--adapter_resume_path", 
        type=str, default=None, 
        help="Path to resume training from a saved adapter checkpoint",
        )
    parser.add_argument(
        "--adapter_resume_step", 
        type=str, default=None, 
        help="The training step number to resume from",
        )
    
    # validation 
    parser.add_argument(
        "--run_validation_at_start", 
        action="store_true"
        )
    parser.add_argument(
        "--validate_every_steps", 
        type=int, default=5000, 
        help="Run inference every",
        )
    parser.add_argument(
        "--video_length", 
        type=int, default=8,
        help="this is used during inference, to determine the fps of generated gifs"
        )
    parser.add_argument(
        "--video_duration", 
        type=int, default=1000,
        help="this is used during inference, to determine the fps of generated gifs"
        )
    parser.add_argument(
        "--controlnet_conditioning_scale", 
        type=float, default=1.0,
        help="This hyper-parameter is derived from ControlNet. We recommend setting it as 1.0 by default."
        )
    parser.add_argument(
        "--control_guidance_start", 
        type=float, default=0.0,
        help="This hyper-parameter is derived from ControlNet. We recommend setting it as 0.0 by default."
        )
    parser.add_argument(
        "--control_guidance_end", 
        type=float, default=1.0,
        help="This hyper-parameter is derived from ControlNet. \
            We recommend setting it between 0.4-0.6 for single condition control, \
            and 1.0 for multi-condition control (see paper appendix for ablation details). \
            If you notice the generated image/video does not follow the spatial control well, you can increase this value; \
            and if you notice the generated image/video quality is not good because the spatial control is too strong, you can decrease this value."
            )
        
    # multi-condition control 
    parser.add_argument(
        "--max_num_multi_source_train", 
        type=int, default=4,
        help="max number of control conditions to use at the same time during multi-condition control training. \
            you can set this as higher value if there's more GPU memory available")
    parser.add_argument(
        '--use_sparsemax', 
        default=False, type=bool_flag, 
        help="If False, we'll use Softmax to aggregate weights for multi-condition control training"
        )
    
    # sparse control       
    parser.add_argument(
        '--sparse_frames', 
        nargs='+', default = None,
        help="This is only used during sparse control validation. \
            For example, setting it as 0 8 15 means we use frames 1, 9, and 16 as key frames for sparse control"
            )
    
    # inference parameters  
    parser.add_argument(
        "--huggingface_checkpoint_folder", 
        type=str, default=None, 
        help="Choose the checkpoint folder based on the task. (e.g. i2vgenxl_depth, sdxl_canny) \
            All checkpoint folders are listed in this huggingface repo: \https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main \
            If you want to load from a local checkpoint, set --huggingface_checkpoint_folder as None and use --local_checkpoint_path instead. "
            )
    parser.add_argument(
        '--extract_control_conditions', 
        default=False, type=bool_flag,
        help="If your input is raw image/frames, you can set this as True. Then this script will extract the control conditions automatically. \
            If you already have control condition images/frames prepared, you can set this as False. Then we'll use these conditions directly. "
            )
    parser.add_argument(
        '--eval_input_type', 
        default='frames', type=str, choices=["images", "frames"],
        help="for i2vgenxl and svd, use 'frames', for sdxl use 'images'"
        )

    parser.add_argument(
        "--evaluation_input_folder", 
        type=str, default='assets/evaluation/images',
        help="The input folder path for evaluation"
        )
    parser.add_argument(
        "--evaluation_output_folder", 
        type=str, default='outputs',
        help="The output folder path to save generated images/videos"
        )
    parser.add_argument(
        "--evaluation_prompt_file", 
        type=str, default='captions.json',
        help="The json file which contains evaluation prompts"
        )
    parser.add_argument(
        "--max_eval", 
        type=int,  default=None, 
        help="max number of samples to evaluate in each validation step"
        )
    parser.add_argument(
        "--num_inference_steps", 
        type=int, default=50, 
        help="We recommend setting the number of inference steps as the same default value of corresponding image/video generation backbone"
        )

    # others and experimental 
    parser.add_argument("--lora", type=str)
    parser.add_argument("--out_channels", type=int, default=None)
    parser.add_argument("--num_repeats", type=int, default=1)

    args = parser.parse_args()
    
    return args



def main(args):
    
    DATA_PATH = args.DATA_PATH # this is the path where all training checkpoints, and output images/videos will be stored
    model_name = args.model_name # sdxl, i2vgenxl, svd
    control_types = args.control_types # can be one of depth, canny, normal, segmentation, softedge, lineart, openpose, scribble
    num_experts = len(args.control_types)
    num_cnet_output_blocks = 12 # there are 12 down/output blocks in SDv1.5 ControlNet
    
    
    # set output dir
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.output_dir = os.path.join(DATA_PATH, "outputs/")
    yaml_trunc = args.yaml_file.replace("configs/", "").replace(".yaml", "")
    args.output_dir += f'{socket.gethostname()}_{yaml_trunc}_{args.timestr}'

    
    args.global_step = 0
    args.apply_sparse_frame_mask = True if 'apply_sparse_frame_mask' in args else None   
    args.multi_source_random_select_control_types = args.multi_source_random_select_control_types if 'multi_source_random_select_control_types' in args else False 
    args.equal_distance_mapping_inference_steps = config['equal_distance_mapping_inference_steps'] if "equal_distance_mapping_inference_steps" in config else None 
    args.router_type if 'router_type' in args else None


    args.next_save_iter = args.save_starting_step
    if args.save_starting_step < 1:
        args.next_save_iter = None


    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")


    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(args.nccl_timeout))]
    )

    accelerator.init_trackers(
        'Ctrl-Adapter',
        config=vars(args),
        init_kwargs={
            "wandb": {
                "tags": [model_name],
                "name": yaml_trunc,
                "save_code": True
            }
        },
    )


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        #global args.global_step
        for model in models:
            # save adapter no matter train or frozen
            if isinstance(model, type(accelerator.unwrap_model(adapter))):
                model.save_pretrained(os.path.join(output_dir, f'adapter_{args.global_step+1}'))
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            if num_experts > 1: # TODO: check if we need to deal seperately with multiple adapters 
                if isinstance(model, type(accelerator.unwrap_model(router))):
                    model.save_pretrained(os.path.join(output_dir, f'router_{args.global_step+1}'))
                    weights.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)


    device = torch.device('cuda')
    set_seed(args.seed)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    
    # Handle the repository creation
    if accelerator.is_local_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
            
            
    ### load the modules needed for the specified backbone ###
    # if the backbone model is contained in diffusers library, you can check the init function in its pipeline file
    if model_name == 'i2vgenxl':
        args.pretrained_model_name_or_path = "ali-vilab/i2vgen-xl"

        from i2vgen_xl.models.unets.unet_i2vgen_xl import I2VGenXLUNet
        from diffusers import DDIMScheduler, AutoencoderKL
        from diffusers.image_processor import VaeImageProcessor
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer
        
        # if you want to download the huggingface checkpoints to a specific folder, you can set cache_dir in .from_pretrained
        # see: https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/modeling_utils.py#L2714
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        unet = I2VGenXLUNet.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder='feature_extractor')
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder='image_encoder')
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_resize=False)
        noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    elif model_name == 'svd':
        args.pretrained_model_name_or_path = "stabilityai/stable-video-diffusion-img2vid"
        
        from svd.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
        from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
        from diffusers import EulerDiscreteScheduler, AutoencoderKL 
        from diffusers.image_processor import VaeImageProcessor
        from diffusers.models import AutoencoderKLTemporalDecoder 
        
        feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder")
        vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        unet = UNetSpatioTemporalConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        
        # since SVD is a I2V model, it doesn't have text encoder 
        # but SDv1.5 ControlNet needs text prompt as input, so we load the SDv1.5 tokenizer here
        tokenizer = AutoTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer", revision=None, use_fast=False,)
        
        # create a list of sigmas to sample from 
        # we use a relatively large number here to avoid sampling the same timestep multiple times during training
        sigmas_svd = _convert_to_karras(num_intervals=200000) 
        sigmas_svd = np.flip(sigmas_svd).copy()
        sigmas_svd = torch.from_numpy(sigmas_svd).to(dtype=torch.float32)
        
    elif model_name == 'sdxl':
        args.pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
        
        from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
        from transformers import CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer
        
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")


    ### set up the helper class, which is mainly used for extracting different control conditions from the input data ###
    # you can add more control conditions by modifying the helper class following the same structure
    helper = ControlNetHelper(use_size_512=args.use_size_512)
    if 'depth' in args.control_types or 'depth' in args.mixed_control_types_training:
        if args.use_midas_depth_estimator: # (recommended) 
            # this can extract depth map pretty fast
            # need to download checkpoint dpt_swin2_large_384 from website: https://github.com/isl-org/MiDaS/tree/master?tab=readme-ov-file
            try:
                helper.add_depth_estimator(estimator_ckpt_path = os.path.join(DATA_PATH, "ckpts/DepthMidas/dpt_swin2_large_384.pt"))    
            except:
                print(f"Please download checkpoint dpt_swin2_large_384 from here: https://github.com/isl-org/MiDaS/tree/master?tab=readme-ov-file \
                and then put the checkpoint under the path {DATA_PATH}/ckpts/DepthMidas/ \
                Otherwise, you need to set the argument --use_midas_depth_estimator as False, then the default depth estimator from transformers \
                library will be used.")
        else:
            # we'll use the depth estimator from transformers library (which is much slower than dpt_swin2_large_384)
            helper.add_depth_estimator()
    if 'canny' in args.control_types or 'canny' in args.mixed_control_types_training:
        pass # canny can be done with cv2 library directly 
    if 'normal' in args.control_types or 'normal' in args.mixed_control_types_training:
        helper.add_normal_estimator()
    if 'segmentation' in args.control_types or 'segmentation' in args.mixed_control_types_training:
        helper.add_segmentation_estimator()     
    if 'openpose' in args.control_types or 'openpose' in args.mixed_control_types_training:
        helper.add_openpose_estimator()     
    if 'softedge' in args.control_types or 'softedge' in args.mixed_control_types_training:
        helper.add_softedge_estimator()     
    if 'lineart' in args.control_types or 'lineart' in args.mixed_control_types_training:
        helper.add_lineart_estimator()     
    if 'scribble' in args.control_types or 'scribble' in args.mixed_control_types_training:
        helper.add_scribble_estimator()              



    ### load the ControlNets used during training ###
    # you can add more controlnets following the same structure
    model_identifiers = {
        'depth': "lllyasviel/control_v11f1p_sd15_depth",
        'canny': "lllyasviel/control_v11p_sd15_canny",
        'normal': "lllyasviel/control_v11p_sd15_normalbae",
        'segmentation': "lllyasviel/control_v11p_sd15_seg",
        'softedge': "lllyasviel/control_v11p_sd15_softedge",
        'lineart': "lllyasviel/control_v11p_sd15_lineart",
        'openpose': "lllyasviel/control_v11p_sd15_openpose",
        'scribble': "lllyasviel/control_v11p_sd15_scribble"
    }
    
    controlnets = {}
    for control_type, model_id in model_identifiers.items():
        if control_type in args.control_types or control_type in args.mixed_control_types_training:
            controlnets[control_type] = ControlNetModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True
            )

    all_control_types = set(args.control_types + args.mixed_control_types_training)
    for k in all_control_types:
        if k in controlnets:
            controlnets[k].requires_grad_(False)



    # (optional) add router if multi-condition control is used
    if num_experts > 1:
        assert model_name == 'i2vgenxl', "please note that we only support i2vgenxl for multi-condition control in our current codebase"
        if args.router_type == 'timestep_weights':
            router = ControlNetRouter(
                num_experts = num_experts, 
                backbone_model_name = model_name, 
                router_type = args.router_type, 
                embedding_dim = 1280, 
                use_sparsemax=args.use_sparsemax
                )
        elif args.router_type == 'simple_weights' or args.router_type == 'equal_weights':
            router = ControlNetRouter(
                num_experts = num_experts, 
                backbone_model_name = model_name, 
                router_type = args.router_type, 
                embedding_dim = None, 
                use_sparsemax=args.use_sparsemax
                )
        elif args.router_type == 'embedding_weights':
            router = ControlNetRouter(
                num_experts = num_experts, 
                backbone_model_name = model_name, 
                router_type = args.router_type, 
                embedding_dim = 1024, 
                use_sparsemax=args.use_sparsemax
                )
        elif args.router_type == 'timestep_embedding_weights':
            router = ControlNetRouter(
                num_experts = num_experts, 
                backbone_model_name = model_name, 
                router_type = args.router_type, 
                embedding_dim = 1024, 
                use_sparsemax=args.use_sparsemax
                )
        router.requires_grad_(True)
    
    

    # initialize adapter
    optimizer_resume_path = None
    if args.adapter_resume_path:
        optimizer_fp = os.path.join(args.adapter_resume_path, "optimizer.bin")
        if os.path.exists(optimizer_fp):
            optimizer_resume_path = optimizer_fp
        adapter = ControlNetAdapter.from_pretrained(
            args.adapter_resume_path, 
            subfolder=f"adapter_{args.adapter_resume_step}", 
            low_cpu_mem_usage=False, 
            device_map=None
            )
        adapter = adapter.to(torch.float32).cuda()
    else:
        adapter = ControlNetAdapter(
            num_blocks=args.num_blocks, 
            num_frames=args.n_sample_frames,
            num_adapters_per_location = args.num_adapters_per_location,
            backbone_model_name = args.model_name,
            cross_attention_dim = args.cross_attention_dim,
            add_spatial_resnet = args.add_spatial_resnet, 
            add_temporal_resnet = args.add_temporal_resnet,
            add_spatial_transformer = args.add_spatial_transformer, 
            add_temporal_transformer = args.add_temporal_transformer,
            add_adapter_location_A = True if 'A' in args.adapter_locations else False, 
            add_adapter_location_B = True if 'B' in args.adapter_locations else False, 
            add_adapter_location_C = True if 'C' in args.adapter_locations else False, 
            add_adapter_location_D = True if 'D' in args.adapter_locations else False, 
            add_adapter_location_M = True if 'M' in args.adapter_locations else False, 
            num_repeats = args.num_repeats, 
            out_channels = args.out_channels
            )
        adapter.requires_grad_(True)


    if args.xformers:
        unet.enable_xformers_memory_efficient_attention()
        vae.enable_xformers_memory_efficient_attention()

    
    ### specify trainable modules  
    # if you are adapting to a new backbone, need to add code following same structure
    if model_name in ['sdxl']:
        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
    elif model_name == 'i2vgenxl':
        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        image_encoder.requires_grad_(False)  
        if num_experts>1:
            if args.router_type == 'equal_weights':
                router.requires_grad_(False) 
            else:
                router.requires_grad_(True)  
    elif model_name == 'svd':
        unet.requires_grad_(False)
        vae.requires_grad_(False)
        image_encoder.requires_grad_(False)        
        
    
    # set up parameters to optimize
    parameters_list = []    
    if num_experts > 1 and args.router_type != 'equal_weights':
        for name, para in router.named_parameters():
            parameters_list.append(para)

    for name, para in adapter.named_parameters():
        parameters_list.append(para)
            
        
    # gradient checkpointing
    if args.gradient_checkpointing:
        adapter.enable_gradient_checkpointing()
        adapter.requires_grad_(True)
        if num_experts > 1:
            router.enable_gradient_checkpointing()
            router.requires_grad_(True)


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    # Use 8-bit Adam for lower memory usage
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    learning_rate = args.learning_rate
    params_to_optimize = [{'params': parameters_list, "lr": learning_rate},]
    count_params(parameters_list)

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if optimizer_resume_path and not args.disable_optimizer_restore:
        logger.info("Restoring the optimizer.")
        try:
            old_optimizer_state_dict = torch.load(optimizer_resume_path)
            # Extract only the state
            old_state = old_optimizer_state_dict['state']
            # Set the state of the new optimizer
            optimizer.load_state_dict({'state': old_state, 'param_groups': optimizer.param_groups})

            del old_optimizer_state_dict
            del old_state

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            logger.info("Restored the optimizer ok")
        except:
            logger.error("Failed to restore the optimizer...", exc_info=True)
            traceback.print_exc()
            raise



    # snr can be used for discrete time noise schedulers for (maybe) faster convergence
    def compute_snr(timesteps): 
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr




    ### collate_fn for data loader 
    # need to add some code following the same structure if adapting to a new backbone
    # if the backbone model is contained in diffusers library, you can check write this part by referencing the pipeline file of the backbone model
    def collate_fn(examples: list) -> dict:

        # examples = [dataset[0], dataset[1]] 
        
        if model_name in ['sdxl']:
            
            input_ids_0 = [example['input_ids_0'] for example in examples]
            input_ids_0 = tokenizer.pad({"input_ids": input_ids_0}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids

            prompt_embeds_0 = text_encoder(input_ids_0.to(device), output_hidden_states=True,)
            prompt_embeds_0 = prompt_embeds_0.hidden_states[-2]
    
            input_ids_1 = [example['input_ids_1'] for example in examples]
            input_ids_1 = tokenizer_2.pad({"input_ids": input_ids_1}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
    
            prompt_embeds_1 = text_encoder_2(input_ids_1.to(device), output_hidden_states=True)
            pooled_prompt_embeds = prompt_embeds_1[0]

            prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]
            prompt_embeds = torch.concat([prompt_embeds_0, prompt_embeds_1], dim=-1)
            
     
            *_, h, w = examples[0]['frames'].shape
            
            return {
                "frames": torch.stack([x['frames'] for x in examples]).to(memory_format=torch.contiguous_format).float(),
                "prompt_embeds": prompt_embeds.to(memory_format=torch.contiguous_format).float(),
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "additional_time_ids": torch.stack([x['additional_time_ids'] for x in examples]),
                "images_pil": [example['images_pil'] for example in examples], 
                "input_ids": input_ids_0,
                }


        elif model_name == 'i2vgenxl':
            
            input_ids = [example['input_ids'] for example in examples]
            input_ids = tokenizer.pad({"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids            
            prompt_embeds = text_encoder(input_ids.to(device))[0]

            ##### create image embeddings #####
            image_for_embeddings = torch.cat([example['image_for_embeddings'] for example in examples])
            image_for_embeddings = image_for_embeddings.to(device=device, dtype=image_encoder.dtype)
            image_embeddings = image_encoder(image_for_embeddings).image_embeds
            image_embeddings = image_embeddings.unsqueeze(1) # # torch.Size([bs, 1, 1024])

            ##### create image latents #####
            image_for_latents = torch.cat([example['image_for_latents'] for example in examples]).to(device=device)
            image_latents = vae.encode(image_for_latents).latent_dist.sample()
            image_latents = image_latents * vae.config.scaling_factor # torch.Size([2, 4, 64, 64])
            image_latents = image_latents.unsqueeze(2) # Add frames dimension to image latents # torch.Size([2, 4, 1, 64, 64])
            
            # Append a position mask for each subsequent frame after the intial image latent frame
            frame_position_mask = []
            for frame_idx in range(args.n_sample_frames - 1):
                scale = (frame_idx + 1) / (args.n_sample_frames - 1)
                frame_position_mask.append(torch.ones_like(image_latents[:, :, :1]) * scale)
            if frame_position_mask:
                frame_position_mask = torch.cat(frame_position_mask, dim=2)
                image_latents = torch.cat([image_latents, frame_position_mask], dim=2) # torch.Size([2, 4, 16, 64, 64])

            if args.input_data_type == 'videos':
                return {
                    "frames": torch.stack([x['frames'] for x in examples]).to(memory_format=torch.contiguous_format).float(),
                    "prompt_embeds": prompt_embeds.to(memory_format=torch.contiguous_format).float(),
                    "images_pil": [example['images_pil'] for example in examples], 
                    "input_ids": input_ids,
                    "image_embeddings": image_embeddings,
                    "image_latents": image_latents
                }


        elif model_name == 'svd':
            
            input_ids = [example['input_ids'] for example in examples]
            input_ids = tokenizer.pad({"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
            
            ##### create image embeddings ##### 
            image_for_embeddings = torch.cat([example['image_for_embeddings'] for example in examples])
            image_for_embeddings = image_for_embeddings.to(device=device, dtype=image_encoder.dtype)
            image_embeddings = image_encoder(image_for_embeddings).image_embeds
            image_embeddings = image_embeddings.unsqueeze(1) # b 1 c

            ##### create image latents ##### 
            image_for_latents = torch.cat([example['image_for_latents'] for example in examples]).to(device=device)
            image_latents = vae.encode(image_for_latents).latent_dist.mode() # b 4 h w
            image_latents = image_latents.unsqueeze(2) # b 4 1 h w

            if args.input_data_type == 'videos':
                return {
                    "frames": torch.stack([x['frames'] for x in examples]).to(memory_format=torch.contiguous_format).float(),
                    "images_pil": [example['images_pil'] for example in examples], 
                    "input_ids": input_ids,
                    "image_embeddings": image_embeddings,
                    "image_latents": image_latents
                }



    # general configs for data loader 
    if args.input_data_type == 'videos':
        dataloader_configs = {
                            'width': args.width, 
                            'height': args.height, 
                            'metadata_path': config['train_prompt_path'],
                            'path': config['train_data_path'],
                            'n_sample_frames': args.n_sample_frames,
                            'output_fps': args.output_fps,
                            'model_name': model_name,
                            }
    elif args.input_data_type == 'images':
        dataloader_configs = {
                            'width': args.width, 
                            'height': args.height, 
                            'path': config['train_data_path'],
                            'metadata_path': config['train_prompt_path'],
                            'model_name': model_name,
                            }            
    

    ### backbone-specific configs needed for data loader initialization 
    # if you are adapting to a new backbone, need to add code following same structure
    if model_name == 'i2vgenxl':     
        dataloader_configs['tokenizer'] = tokenizer
        dataloader_configs['feature_extractor'] = feature_extractor
        dataloader_configs['image_processor'] = image_processor  
    elif model_name == 'svd':     
        dataloader_configs['tokenizer'] = tokenizer
        dataloader_configs['feature_extractor'] = feature_extractor
        dataloader_configs['image_processor'] = image_processor      
    elif model_name == 'sdxl':
        dataloader_configs['unet_config'] = unet.config
        dataloader_configs['unet_add_embedding'] = unet.add_embedding
        dataloader_configs['og_size'] = (1024, 1024)
        dataloader_configs['crop_coords'] = (0, 0)
        dataloader_configs['required_size'] = (1024, 1024)
        dataloader_configs['tokenizer'] = tokenizer
        dataloader_configs['tokenizer_2'] = tokenizer_2

    
    # create data loader 
    if args.input_data_type == 'videos':
        dataset = VideoLoader(**dataloader_configs)
    elif args.input_data_type == 'images':
        dataset = ImageLoader(**dataloader_configs)
        
    dataloader = DataLoader(dataset, args.train_batch_size, shuffle=True, collate_fn=collate_fn)



    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    

    optimizer, lr_scheduler, dataloader, unet, adapter = accelerator.prepare(optimizer, lr_scheduler, dataloader, unet, adapter)
   
    if num_experts > 1:
        router = accelerator.prepare(router)    
        


    def run_validation(step=0, node_index=0):

        torch.cuda.empty_cache()
        gc.collect()

        args.local_checkpoint_path = args.output_dir

        with torch.no_grad():
            inference_main(inference_args=args)
            
        return



    # Move controlnets and helper to gpu
    for k in controlnets:
        controlnets[k] = controlnets[k].to(accelerator.device, dtype=weight_dtype)
    helper.to(accelerator.device, dtype=weight_dtype)
    
    ### move backbone-specific modules to gpu
    # if you are adapting to a new backbone, need to add code following same structure
    if model_name == 'i2vgenxl':      
        vae.to(accelerator.device, dtype=torch.bfloat16 if args.vae_b16 else torch.float32)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        image_encoder.to(accelerator.device, dtype=weight_dtype)
    elif model_name == 'svd':      
        vae.to(accelerator.device, dtype=torch.bfloat16 if args.vae_b16 else torch.float32)
        image_encoder.to(accelerator.device, dtype=weight_dtype)        
    elif model_name == 'sdxl':
        vae.to(accelerator.device, dtype=torch.bfloat16 if args.vae_b16 else torch.float32)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder_2.to(accelerator.device, dtype=weight_dtype)    


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterward we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.project_name, init_kwargs={"wandb": {"config": vars(args), "save_code": True}})

    def bar(prg):
        br = '|' + '█' * prg + ' ' * (25 - prg) + '|'
        return br

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    latents_scaler = vae.config.scaling_factor

    def save_checkpoint():
        save_dir = Path(args.output_dir)
        save_dir = str(save_dir)
        save_dir = save_dir.replace(" ", "_")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        accelerator.save_state(save_dir)


    # this is the core function for loss computation
    def compute_loss_from_batch(batch: dict):

        frames = batch["frames"]
        bsz, number_of_frames, c, w, h = frames.shape

        # 1. prepare latents, noise, and timesteps
        with torch.no_grad():
            ### 1.1 prepare latents encoded from raw video frames
            if args.max_vae_encode:
                latents = []
                x = rearrange(frames, "bs nf c h w -> (bs nf) c h w")
                for latent_index in range(0, x.shape[0], args.max_vae_encode):
                    sample = x[latent_index: latent_index + args.max_vae_encode]
                    latent = vae.encode(sample.to(dtype=vae.dtype)).latent_dist.sample().float()
                    if len(latent.shape) == 3:
                        latent = latent.unsqueeze(0)
                    latents.append(latent)
                latents = torch.cat(latents, dim=0)
            else:
                # convert the latents from 5d -> 4d, so we can run it though the vae encoder
                x = rearrange(frames, "bs nf c h w -> (bs nf) c h w")
                del frames
                latents = vae.encode(x.to(dtype=vae.dtype)).latent_dist.sample().float() # bf 4 h/8 w/8
                
            if args.latent_nan_checking and torch.any(torch.isnan(latents)):
                accelerator.print("NaN found in latents, replacing with zeros")
                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)

            latents = rearrange(latents, "(b f) c h w -> b c f h w", b=bsz).to(dtype=weight_dtype)
            latents = latents * latents_scaler

            ### 1.2 initialize random noise 
            noise = torch.randn_like(latents, device=latents.device).to(dtype=weight_dtype)
            if args.noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1, 1), device=latents.device) # torch.Size([b, 4, f, h/8, w/8])

            ### 1.3 initialize timesteps (and sigmas for SVD)
            if model_name in ['svd']: 
                ### please note that the timestep mapping here is slightly different from the algorithm described in our 1st arxiv version.
                # here we sample sigmas from the range [0.002, 700] instead of the unbounded lognormal distribution 
                # to better align with the timestep sampled from noise scheduler during inference.
                # we observe better spatial control with this variant, and will update our arxiv later accordingly
                u, sigmas = sample_svd_sigmas_timesteps(1, sigmas_svd, args.num_inference_steps)
                u = torch.Tensor(u.repeat(bsz)).cuda()
                sigmas = torch.Tensor(sigmas.repeat(bsz)).cuda()

                ### we use the following method in our 1st arxiv version, which will create some mismatch between training and inference
                # u = torch.rand((bsz ), device=latents.device)
                # sigmas = rand_log_normal(u=u, loc=0.7, scale=1.6)
            
                sigmas = rearrange(sigmas,  "(b f) -> b f", b=args.train_batch_size).to(latents.device)
                sigmas_reshaped = sigmas.clone()
                while len(sigmas_reshaped.shape) < len(latents.shape):
                    sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)
                    
                timesteps = (0.25 * sigmas.log()).squeeze(-1)
                
            else:
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()  


        # 2. prepare inputs
        with torch.no_grad():
            ### 2.1 prepare model-specific inputs
            if model_name in ['sdxl']:
                prompt_embeds = batch['prompt_embeds'].to(dtype=weight_dtype)
                add_text_embeds = batch['pooled_prompt_embeds']
                #add_text_embeds = batch['prompt_embeds']
                additional_time_ids = batch['additional_time_ids'] 
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": additional_time_ids}

            elif model_name == 'i2vgenxl':      
                prompt_embeds = batch['prompt_embeds'].to(dtype=weight_dtype)
                image_embeddings = batch['image_embeddings'].to(dtype=weight_dtype)
                image_latents = batch['image_latents'].to(dtype=weight_dtype)
                fps_tensor = torch.tensor([args.output_fps]).to(device)
                fps_tensor = fps_tensor.repeat(args.train_batch_size, 1).ravel().to(dtype=weight_dtype)

            elif model_name == 'svd':      
                image_embeddings = batch['image_embeddings'].to(dtype=weight_dtype) # torch.Size([1, 1, 1024])
                image_latents = batch['image_latents'].to(dtype=weight_dtype)
                fps_tensor = torch.tensor([args.output_fps]).to(device).to(dtype=weight_dtype)
                fps_tensor = fps_tensor.repeat(args.train_batch_size, 1).ravel().to(dtype=weight_dtype)


        with torch.no_grad():
            # 2.1 prepare inputs to unet, condition frames, and text embeddings
            control_condition_estimation = True 
            if len(args.mixed_control_types_training) != 0:
                current_batch_control_types = [random.choice(args.mixed_control_types_training)]
            elif args.multi_source_random_select_control_types:
                n_sparse_control_types = random.randint(1, args.max_num_multi_source_train) # max select 4 control sources by default
                current_batch_control_types = random.sample(args.control_types, n_sparse_control_types)
                sparse_mask = [0 for i in range(num_experts)]
                for i in range(num_experts):
                    if args.control_types[i] in current_batch_control_types:
                        sparse_mask[i] = 1 # 1 means activate, 0 means mask_out 
            else:
                current_batch_control_types = args.control_types
                sparse_mask = None 
                
            batch = helper.prepare_batch(batch, current_batch_control_types = current_batch_control_types, 
                                         control_condition_estimation=control_condition_estimation) # 10640
            
            encoder_hidden_states = helper.text_encoder(batch["input_ids"])[0] # b, 77, 768
        
            controlnet_images = {}
            for ctrl_type in current_batch_control_types:
                controlnet_images[ctrl_type] = rearrange(batch[ctrl_type]["conditioning_pixel_values"], "b f c h w -> (b f) c h w").to(dtype=weight_dtype)


        # 3. define noisy latents
        with torch.no_grad():
            if model_name in ['svd']:
                
                # train_noise_aug: adds some noise to the latents constructed from 1st frame
                # the value of this parameter could be quite important during training
                # Setting this to a high value could lead to better motion tracking in the generated video, but also more blurriness.
                # On the other hand, setting it to a low value could result in poor motion tracking, but the video will be less blurry.
                # We use 0.02 as default across all control conditions by default, but different control conditions probably have different optimal values for this. 
                train_noise_aug = 0.02 
                
                small_noise_latents = latents + noise * train_noise_aug
                conditional_latents = small_noise_latents[:, :, 0, :, :]
                conditional_latents = conditional_latents / vae.config.scaling_factor

                noisy_latents_4channels  = latents + noise * sigmas_reshaped 
                noisy_latents = noisy_latents_4channels  / ((sigmas_reshaped**2 + 1) ** 0.5)
                
                args.conditioning_dropout_prob = 0.1
                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = torch.zeros_like(image_embeddings)
                    image_embeddings = torch.where(prompt_mask, null_conditioning, image_embeddings)
                    # Sample masks for the original images.
                    image_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - ((random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype))
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    conditional_latents = image_mask * conditional_latents

                # Concatenate the `conditional_latents` with the `noisy_latents`.
                conditional_latents = conditional_latents.unsqueeze(2).repeat(1, 1, noisy_latents.shape[2], 1, 1)
                noisy_latents = torch.cat([noisy_latents, conditional_latents], dim=1) 
                target = rearrange(latents, "b c f h w -> b f c h w") 

                # NOTE: Stable Diffusion Video was conditioned on fps - 1, which is why it is reduced here.
                # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
                added_time_ids = add_time_ids_svd(args.output_fps - 1, 127, train_noise_aug, image_embeddings.dtype, bsz,)
                added_time_ids = added_time_ids.to(device)

            else:
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")



        noisy_latents = rearrange(noisy_latents, "bs c nf h w -> (bs nf) c h w")
        _, _, noisy_latents_h, noisy_latents_w = noisy_latents.shape

        assert args.use_size_512 == True, "extention of different noisy latent sizes for SDv1.5 ControlNet input is left for future work"
        
        # resize noisy latents to 64 * 64, which is the default input feature size for SDv1.5 ControlNet
        # extention of different noisy latent sizes is left for future work
        if (noisy_latents_h, noisy_latents_w) != (64, 64) and args.use_size_512:
            reshaped_noisy_latents = F.adaptive_avg_pool2d(noisy_latents, (64, 64)) # b, 4, 64, 64
        else:
            reshaped_noisy_latents = noisy_latents


        # set timestep for controlnet and adapter
        if model_name in ['svd']:
            controlnet_timesteps = (u*1000).round().to(device)
        else:
            controlnet_timesteps = timesteps
        adapter_timesteps = controlnet_timesteps           


        down_block_res_samples_dict = {}
        mid_block_res_sample_dict = {}

        # reshaped_noisy_latents_trans is given to ControlNets later
        if model_name in ['svd']:
            # SVD has 8 channels, we only give the first 4 channels to ControlNet
            reshaped_noisy_latents_trans = reshaped_noisy_latents[:, :4] 
        else:
            reshaped_noisy_latents_trans = reshaped_noisy_latents

        # get mid block and down blocks res samples from controlnets
        with torch.no_grad():
            for ctrl_type in current_batch_control_types:
                down_block_res_samples_dict[ctrl_type], mid_block_res_sample_dict[ctrl_type] = controlnets[ctrl_type](
                    reshaped_noisy_latents_trans.to(weight_dtype), 
                    controlnet_timesteps.to(weight_dtype), 
                    torch.repeat_interleave(encoder_hidden_states, args.n_sample_frames, dim = 0),
                    controlnet_cond=controlnet_images[ctrl_type],
                    return_dict=False,
                    skip_conv_in = args.skip_conv_in,
                    skip_time_emb = args.skip_time_emb,
                )
                down_block_res_samples_dict[ctrl_type] = [down.to(weight_dtype) for down in down_block_res_samples_dict[ctrl_type]]
                mid_block_res_sample_dict[ctrl_type] = mid_block_res_sample_dict[ctrl_type].to(weight_dtype)

    
        noisy_latents = rearrange(noisy_latents, "(bs nf) c h w -> bs c nf h w", bs = args.train_batch_size) # b, 4, 1, h/8, w/8
    

        # please note that we only support i2vgenxl for multi-condition control in our current codebase (too much work to implement it to every backbone lol)
        # if you are interested in applying multi-condition control to other image/video diffusion models, 
        # you can follow the modification we made to i2vgenxl unet and i2vgenxl pipeline 
        if num_experts > 1:
            assert model_name == 'i2vgenxl', "please note that we only support i2vgenxl for multi-condition control in our current codebase"
            #with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=weight_dtype):
                if args.router_type == 'simple_weights' or args.router_type == 'equal_weights':
                    down_block_weights, mid_block_weights = router(router_input=None, sparse_mask=sparse_mask)
                elif args.router_type == 'embedding_weights':
                    down_block_weights, mid_block_weights = router(router_input=image_embeddings, sparse_mask=sparse_mask)
                elif args.router_type == 'timestep_weights':
                    down_block_weights, mid_block_weights = router(router_input=timesteps, sparse_mask=sparse_mask)
                elif args.router_type == 'timestep_embedding_weights':
                    down_block_weights, mid_block_weights = router(router_input=[timesteps, image_embeddings], sparse_mask=sparse_mask)
        else:
            down_block_weights = torch.ones((num_cnet_output_blocks, 1)).to(weight_dtype).to(device)
            mid_block_weights = torch.ones((1)).to(weight_dtype).to(device)


        # aggregate mid/down blocks res samples
        down_block_res_samples, mid_block_res_sample = [0 for k in range(num_cnet_output_blocks)], 0

        for k in range(num_cnet_output_blocks):
            for idx, ctrl_type in enumerate(current_batch_control_types):
                down_block_res_samples[k] = down_block_res_samples[k] + down_block_res_samples_dict[ctrl_type][k] * down_block_weights[k][idx]
                if num_experts == 1:
                    down_block_res_samples[k] = down_block_res_samples[k].detach()

        if 'M' in args.adapter_locations:
            for idx, ctrl_type in enumerate(current_batch_control_types):
                mid_block_res_sample = mid_block_res_sample + mid_block_res_sample_dict[ctrl_type] * mid_block_weights[idx]  
                if num_experts == 1:
                    mid_block_res_sample = mid_block_res_sample.detach()    
        else:
            mid_block_res_sample = None
            
        
        # this part is for sparse control training 
        if args.apply_sparse_frame_mask is not None:
            # we randomly select 1 to 4 key frames for sparse control training. 
            # if you are interested in applying some specific frames for sparse control (i.e., frame1, frame8, frame15), 
            # it might be better to directly train on these specific frames compared with current random samplying strategy
            n_sparse_frames = random.randint(1, 4) # you can set 4 as random.randint(1, args.n_sample_frames), if you want to support any number of sparse frames
            sparsity_masking = sorted(random.sample(range(0, args.n_sample_frames), n_sparse_frames))

            down_block_res_samples = [down_block_res_samples[i][sparsity_masking, :] for i in range(len(down_block_res_samples))]
            mid_block_res_sample = mid_block_res_sample[sparsity_masking, :] if mid_block_res_sample is not None else None
            
        else:
            sparsity_masking = None 


        ### the encoder_hidden_states will be given to adapters
        if model_name in ['svd', 'i2vgenxl']:
            encoder_hidden_states = image_embeddings 
        elif model_name in ['sdxl']:
            encoder_hidden_states = prompt_embeds

        assert args.cross_attention_dim == encoder_hidden_states.shape[-1], \
            "Need to set cross_attention_dim in configuration file as the same value of the feature dimension of encoder_hidden_states"
            
            
        # call adapter 
        # note that if we use sparse control, only the sparse frames are given to adapters
        with torch.autocast(device_type="cuda", dtype=weight_dtype):   
            adapter_input_num_frames = n_sparse_frames if args.apply_sparse_frame_mask else args.n_sample_frames
            adapted_down_block_res_samples, adapted_mid_block_res_sample = adapter(down_block_res_samples, mid_block_res_sample, sparsity_masking=sparsity_masking, 
                                                                               num_frames = adapter_input_num_frames, timestep = adapter_timesteps, encoder_hidden_states = encoder_hidden_states)
        adapted_down_block_res_samples = [down.to(weight_dtype) for down in adapted_down_block_res_samples]
        adapted_mid_block_res_sample = adapted_mid_block_res_sample.to(weight_dtype) if adapted_mid_block_res_sample is not None else None 

        
        # transform the adapted sprase frame features back to dense features
        if args.apply_sparse_frame_mask is not None:
            full_adapted_down_block_res_samples = []
            for i in range(len(adapted_down_block_res_samples)):
                _, c, h, w = adapted_down_block_res_samples[i].shape 
                full_adapted_down_block_res_samples.append(torch.zeros((args.n_sample_frames, c, h, w)).to(device))
                for j, pos in enumerate(sparsity_masking):
                    full_adapted_down_block_res_samples[i][pos] = adapted_down_block_res_samples[i][j]
            if adapted_mid_block_res_sample is not None:
                _, c, h, w = adapted_mid_block_res_sample.shape 
                full_adapted_mid_block_res_sample = torch.zeros((args.n_sample_frames, c, h, w)).to(device)
                for j, pos in enumerate(sparsity_masking):
                    full_adapted_mid_block_res_sample[pos] = adapted_mid_block_res_sample[j]
            else:
                full_adapted_mid_block_res_sample = None
        else:
            full_adapted_down_block_res_samples = adapted_down_block_res_samples
            full_adapted_mid_block_res_sample = adapted_mid_block_res_sample


        full_adapted_down_block_res_samples = [rearrange(down_block, "(bs nf) c h w -> bs c nf h w", bs=args.train_batch_size).to(weight_dtype) for down_block in full_adapted_down_block_res_samples]
        if full_adapted_mid_block_res_sample is not None:
            full_adapted_mid_block_res_sample = rearrange(full_adapted_mid_block_res_sample, "(bs nf) c h w -> bs c nf h w", bs=args.train_batch_size).to(weight_dtype)



        # finally we can give the features after adapter to unet
        if model_name == 'i2vgenxl':     
            assert (noisy_latents_h, noisy_latents_w) == (64, 64), "please note that we only support generating videos of resolution 512 * 512 "
            # note that in our current codebase, we only support generating videos of resolution 512 * 512 
            # if you want to generate videos with different frames, you can use F.interpolate to resize full_adapted_down_block_res_samples and 
            # full_adapted_mid_block_res_sample into your desired size, before giving to i2vgenxl unet

            model_pred = unet(
                noisy_latents,
                timesteps.to(weight_dtype),
                encoder_hidden_states=prompt_embeds,
                fps=fps_tensor,
                image_latents=image_latents,
                image_embeddings=image_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals = full_adapted_down_block_res_samples,
                mid_block_additional_residual = full_adapted_mid_block_res_sample,
                return_dict=False,
            )[0]

        elif model_name == 'svd':    
            assert (noisy_latents_h, noisy_latents_w) == (64, 64), "please note that we only support generating videos of resolution 512 * 512 "
            # note that in our current codebase, we only support generating videos of resolution 512 * 512 
            # if you want to generate videos with different frames, you can use F.interpolate to resize full_adapted_down_block_res_samples and 
            # full_adapted_mid_block_res_sample into your desired size, before giving to svd unet
            
            if full_adapted_down_block_res_samples[0].dim() == 5:
                full_adapted_down_block_res_samples = [rearrange(adapt_down_res_sample, "b c f h w -> (b f) c h w") for adapt_down_res_sample in full_adapted_down_block_res_samples]
                full_adapted_mid_block_res_sample = rearrange(full_adapted_mid_block_res_sample, "b c f h w -> (b f) c h w")

            model_pred = unet(
                rearrange(noisy_latents, "b c f h w -> b f c h w"), 
                timesteps, 
                image_embeddings,
                added_time_ids=added_time_ids,
                down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in full_adapted_down_block_res_samples],
                mid_block_additional_residual=full_adapted_mid_block_res_sample.to(dtype=weight_dtype),
            ).sample 

        elif model_name == 'sdxl':    
            if full_adapted_down_block_res_samples[0].dim() == 5:
                full_adapted_down_block_res_samples = [rearrange(adapt_down_res_sample, "b c f h w -> (b f) c h w") for adapt_down_res_sample in full_adapted_down_block_res_samples]
                full_adapted_mid_block_res_sample = None 
       
            model_pred = unet(
                rearrange(noisy_latents, "b c f h w -> (b f) c h w"), 
                timesteps, 
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in full_adapted_down_block_res_samples],
                mid_block_additional_residual=0, # we use 0 here to avoid it being None (when it's none, is_controlnet becomes False)
            ).sample.unsqueeze(2) # b 4 h/8 w/8


        # compute loss
        if args.snr_gamma:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(timesteps)
            mse_loss_weights = (
                    torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            log_down_blocks = down_block_weights.cpu().detach() if num_experts > 1 else None 
            log_mid_block = mid_block_weights.cpu().detach() if (num_experts > 1 and mid_block_weights is not None) else None
            return loss.mean(), log_down_blocks, log_mid_block
        else:
            if model_name not in ['svd']:
                log_down_blocks = down_block_weights.cpu().detach() if num_experts > 1 else None 
                log_mid_block = mid_block_weights.cpu().detach() if (num_experts > 1 and mid_block_weights is not None) else None
                return F.mse_loss(model_pred.float(), target.float(), reduction='mean'), log_down_blocks, log_mid_block
            elif model_name in ['svd']:
                sigmas = sigmas_reshaped
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * rearrange(noisy_latents_4channels, "b c f h w -> b f c h w")
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)
                loss = torch.mean((weighing.float() * (denoised_latents.float() - target.float()) ** 2).reshape(target.shape[0], -1), dim=1)
                loss = loss.mean()
                return loss, down_block_weights.cpu().detach(), mid_block_weights.cpu().detach()



    def process_batch_func(batch: dict):

        if args.global_step == 0:
            # print(f"Running initial validation at step")
            if accelerator.is_main_process and args.run_validation_at_start:
                run_validation(step=args.global_step, node_index=accelerator.process_index // 8)
            accelerator.wait_for_everyone()

        loss, down_block_weights, mid_block_weights = compute_loss_from_batch(batch)
        accelerator.backward(loss)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(parameters_list, args.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        return loss, down_block_weights, mid_block_weights, lr_scheduler


    def process_batch(batch: dict):

        now = time.time()

        if num_experts > 1:
            with accelerator.accumulate(adapter, router):
                        loss, down_block_weights, mid_block_weights, lr_scheduler = process_batch_func(batch)    
        else:    
            with accelerator.accumulate(adapter):
                loss, down_block_weights, mid_block_weights, lr_scheduler = process_batch_func(batch) 
        
        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            args.global_step += 1

        fll = round((args.global_step * 100) / args.max_train_steps)
        fll = round(fll / 4)
        pr = bar(fll)

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "loss_time": (time.time() - now)}
        if num_experts > 1:
            for idx_e in range(num_experts):
                ctrl_type = control_types[idx_e]
                for idx_r in range(num_cnet_output_blocks):
                    logs[f'down_block_{idx_r}_{ctrl_type}'] = float(down_block_weights[idx_r][idx_e])
                if mid_block_weights is not None:
                    logs[f'mid_block_{ctrl_type}'] = float(mid_block_weights[idx_e])
                    
        if accelerator.is_main_process and args.next_save_iter is not None and args.global_step < args.max_train_steps and args.global_step + 1 == args.next_save_iter:
            save_checkpoint()
            torch.cuda.empty_cache()
            gc.collect()
            args.next_save_iter += args.save_n_steps

        if args.validate_every_steps is not None and args.global_step > 0 and args.global_step % args.validate_every_steps == 0:
            if accelerator.is_main_process:
                run_validation(step=args.global_step, node_index=accelerator.process_index // 8)
                    
            accelerator.wait_for_everyone()

        progress_bar.set_postfix(**logs)
        progress_bar.set_description_str("Progress:" + pr)
        accelerator.log(logs, step=args.global_step)



    for epoch in range(args.num_train_epochs):

        adapter.train()
        if num_experts > 1:
            router.train()

        for step, batch in enumerate(dataloader):
            process_batch(batch)

            if args.global_step >= args.max_train_steps:
                break

        if args.global_step >= args.max_train_steps:
            logger.info("Max train steps reached. Breaking while loop")
            break

        accelerator.wait_for_everyone()
    accelerator.end_training()





if __name__ == "__main__":
    
    mp.set_start_method('spawn')

    args = parse_args()

    config = OmegaConf.load(args.yaml_file) 
    config_dict = OmegaConf.to_container(config, resolve=True)  # Convert OmegaConf to a simple dictionary
    for key, value in config_dict.items(): # Update the argparse Namespace with configurations from the file
        setattr(args, key, value)
            
    main(args)
