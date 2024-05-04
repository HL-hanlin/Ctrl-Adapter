import os 
import time
import json
import argparse
from PIL import Image

import torch
import torchvision.transforms as transforms
from controlnet.controlnet import ControlNetModel

from model.ctrl_adapter import ControlNetAdapter
from model.ctrl_router import ControlNetRouter
from model.ctrl_helper import ControlNetHelper

from utils.utils import center_crop_and_resize, bool_flag, save_as_gif, save_concatenated_gif

to_pil = transforms.ToPILImage()



def parse_inference_args():
    inference_parser = argparse.ArgumentParser(description="Ctrl-Adapter inference", add_help=False)
    
    inference_parser.add_argument(
        "--model_name", type=str, default="i2vgenxl",
        choices=["i2vgenxl", "svd", "sdxl"]
        )
    inference_parser.add_argument(
        "--control_types",  nargs='+', default='depth', 
        choices=["depth", "canny", 'normal', 'segmentation', 'openpose', 'softedge', 'lineart', 'scribble', 'inpainting']
        )
    inference_parser.add_argument(
        "--huggingface_checkpoint_folder", 
        type=str, default=None, 
        help="Choose the checkpoint folder based on the task. (e.g. i2vgenxl_depth, sdxl_canny) \
            All checkpoint folders are listed in this huggingface repo: \https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main \
            If you want to load from a local checkpoint, set --huggingface_checkpoint_folder as None and use --local_checkpoint_path instead. "
            )
    inference_parser.add_argument(
        "--local_checkpoint_path", 
        type=str, default=None, 
        help="Path to load from a local checkpoint \
            If you want to load from a huggingface checkpoint, set --local_checkpoint_path as None and use --huggingface_checkpoint_folder instead. "
            )
    inference_parser.add_argument(
        '--extract_control_conditions', 
        default=False, type=bool_flag,
        help="If your input is raw image/frames, you can set this as True. Then this script will extract the control conditions automatically. \
            If you already have control condition images/frames prepared, you can set this as False. Then we'll use these conditions directly. "
            )
    inference_parser.add_argument(
        '--eval_input_type', 
        default='frames', type=str, choices=["images", "frames"],
        help="for i2vgenxl and svd, use 'frames', for sdxl use 'images'"
        )
    inference_parser.add_argument(
        "--max_eval", 
        type=int,  default=None, 
        help="max number of samples to evaluate in each validation step. If this is None, this script will evaluate all samples under evaluation_input_folder. "
        )
    inference_parser.add_argument(
        "--evaluation_input_folder", 
        type=str, default='assets/evaluation/images',
        help="The input folder path for evaluation"
        )
    inference_parser.add_argument(
        "--evaluation_output_folder", 
        type=str, default='outputs',
        help="The output folder path to save generated images/videos"
        )
    inference_parser.add_argument(
        "--evaluation_prompt_file", 
        type=str, default='captions.json',
        help="The json file which contains evaluation prompts"
        )
    inference_parser.add_argument(
        "--global_step", 
        type=int, default=None, 
        help="This specifies which adapter to load from the local_checkpoint_path. \
            For example, setting global_step as 10000 will load adapter_10000 under the local_checkpoint_path. \
            If you load checkpoint from huggingface, you can set this as None. "
            )
    inference_parser.add_argument(
        "--n_sample_frames", 
        type=int, default=16, 
        help="This is the number of output frames of the video generation model. \
            For image generation, this parameter is not used. \
            For video generation, we recommend setting this parameter with the same default value of the corresponding video diffusion backbone."
            )
    inference_parser.add_argument(
        "--mixed_precision", 
        type=str, default='bf16', choices=["no", "fp16", "bf16"], 
        help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU."),
        )
    inference_parser.add_argument(
        "--width", 
        type=int, default=512, 
        help="Our current implementation supports generating videos of size 512 * 512 with I2VGen-XL and SVD, \
            and images of size 1024 * 1024 with SDXL"
            )
    inference_parser.add_argument(
        "--height", 
        type=int, default=512,
        help="Our current implementation supports generating videos of size 512 * 512 with I2VGen-XL and SVD, \
            and images of size 1024 * 1024 with SDXL"
            )
    inference_parser.add_argument(
        "--video_length", 
        type=int, default=8, 
        help="This controls the speed of output gif"
        )
    inference_parser.add_argument(
        "--video_duration", 
        type=int, default=1000, 
        help="This controls the speed of output gif"
        )
    inference_parser.add_argument(
        "--controlnet_conditioning_scale", 
        type=float, default=1.0,
        help="This hyper-parameter is derived from ControlNet. We recommend setting it as 1.0 by default."
        )
    inference_parser.add_argument(
        "--control_guidance_start", 
        type=float, default=0.0,
        help="This hyper-parameter is derived from ControlNet. We recommend setting it as 0.0 by default."
        )
    inference_parser.add_argument(
        "--control_guidance_end", 
        type=float, default=1.0,
        help="This hyper-parameter is derived from ControlNet. \
            We recommend setting it between 0.4-0.6 for single condition control and 1.0 for multi-condition control (see paper appendix for ablation details). \
            If you notice the generated image/video does not follow the spatial control well, you can increase this value; \
            and if you notice the generated image/video quality is not good because the spatial control is too strong, you can decrease this value."
            )
    inference_parser.add_argument(
        '--sparse_frames',
        nargs='+', default = None, 
        help="For example, --sparse_frames 0 5 10 15 means to give frame 1, 6, 11, 16 as key frames for sparse control. \
            Please note that our model might not be able to handle complex motions with very sparse frames"
            )
    inference_parser.add_argument(
        '--use_size_512', 
        default=True, type=bool_flag,
        help="Our framework currently only support image/video generation with this parameter as True."
        )
    inference_parser.add_argument(
        '--skip_conv_in', 
        default=False, type=bool_flag,
        help="This corresponds to the latents skipping strategy as mentioned in our paper. \
            For SVD and sparse control, we recommend setting this as True."
            )
    inference_parser.add_argument(
        '--skip_time_emb', 
        default=False, type=bool_flag,
        help="This is stil experimental. The default value is False."
        )
    inference_parser.add_argument(
        '--adapter_locations',  
        nargs='+', default=['A', 'B', 'C', 'D', 'M'], choices=['A', 'B', 'C', 'D', 'M'],
        help="For I2VGen-XL and SVD, we add adapters to mid block and all output blocks (i.e., --adapter_locations A B C D M \
            For SDXL, we add adapters to output blocks A B and C (i.e., --adapter_locations A B C"
        )
    inference_parser.add_argument(
        "--num_inference_steps", 
        type=int, default=50, 
        help="We recommend setting the number of inference steps as the same default value of corresponding image/video generation backbone"
        )
    inference_parser.add_argument("--xformers", action="store_true")
    inference_parser.add_argument("--lora", type=str)
    inference_parser.add_argument("--seed", type=int, default=42)
    
    return inference_parser







def inference_main(inference_args):
    
    # read text prompts
    with open(os.path.join(inference_args.evaluation_input_folder, inference_args.evaluation_prompt_file), 'r') as file:
        captions = json.load(file)
    
    
    # set input dir
    if inference_args.extract_control_conditions or inference_args.eval_input_type == 'frames':
        # if extract_control_conditions is True, this script will extract control conditions automatically from raw images/frames
        raw_input_dir = os.path.join(inference_args.evaluation_input_folder, "raw_input")
    if not inference_args.extract_control_conditions:
        # otherwise, we'll load the extracted control conditions directly
        condition_input_dir = []
        for ctrl_type in inference_args.control_types:
            dir_path = os.path.join(inference_args.evaluation_input_folder, ctrl_type)
            condition_input_dir.append(dir_path)


    # set output folder
    inference_args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(inference_args.evaluation_output_folder, inference_args.timestr)
    os.makedirs(output_dir, exist_ok=True)


    # inference precision
    device = torch.device("cuda")
    data_type = torch.float32
    if inference_args.mixed_precision == 'f16':
        data_type = torch.half
    elif inference_args.mixed_precision == 'f32':
        data_type = torch.float32
    elif inference_args.mixed_precision == 'bf16':
        data_type = torch.bfloat16


    # load adapter 
    if inference_args.huggingface_checkpoint_folder is not None: # loading from huggingface checkpoint
        adapter = ControlNetAdapter.from_pretrained(
            "hanlincs/Ctrl-Adapter", 
            subfolder=inference_args.huggingface_checkpoint_folder, 
            low_cpu_mem_usage=False, 
            device_map=None
            )
    else:
        adapter = ControlNetAdapter.from_pretrained(
            inference_args.local_checkpoint_path, 
            subfolder = f"adapter_{inference_args.global_step}", 
            low_cpu_mem_usage=False, 
            device_map=None
            )

    adapter = adapter.to(data_type)
    adapter.eval()


    # load router if multi-condition control is used 
    num_experts = len(inference_args.control_types)
    if num_experts > 1:
        if inference_args.huggingface_checkpoint_folder is not None: # loading from huggingface checkpoint
            router = ControlNetRouter.from_pretrained(
                "hanlincs/Ctrl-Adapter", 
                subfolder=inference_args.huggingface_checkpoint_folder.replace("_adapter", "_router"), 
                low_cpu_mem_usage=False, 
                device_map=None
                ).cuda()
        else:
            router = ControlNetRouter.from_pretrained(
                inference_args.local_checkpoint_path, 
                subfolder=f"router_{inference_args.global_step}", 
                low_cpu_mem_usage=False, 
                device_map=None
                ).cuda()
        router = router.to(data_type)
        router.eval()


    # create dir for generated output images/frames 
    output_images_dir = os.path.join(output_dir, f"output_{inference_args.eval_input_type}")
    os.makedirs(output_images_dir, exist_ok=True)

    # create dir for condition images/frames 
    output_condition_images_dir = []
    for i, ctrl_type in enumerate(inference_args.control_types):
        output_condition_images_dir.append(os.path.join(output_dir, f"conditon_{ctrl_type}_{inference_args.eval_input_type}"))
        os.makedirs(output_condition_images_dir[-1], exist_ok=True)

    if inference_args.extract_control_conditions or inference_args.eval_input_type == 'frames':
        # copy of input images/frames 
        input_images_dir = os.path.join(output_dir, f"input_{inference_args.eval_input_type}")
        os.makedirs(input_images_dir, exist_ok=True)

    # create dir for concatenated output
    concat_output_dir = os.path.join(output_dir, "concat_output")
    os.makedirs(concat_output_dir, exist_ok=True)
    
    if inference_args.eval_input_type == 'frames':
        # output gifs 
        output_gifs_dir = os.path.join(output_dir, "output_gifs")
        os.makedirs(output_gifs_dir, exist_ok=True)

        # condition gifs
        output_condition_gifs_dir = []
        for ctrl_type in inference_args.control_types:
            output_condition_gifs_dir.append(os.path.join(output_dir, f"condition_{ctrl_type}_gifs"))
            os.makedirs(output_condition_gifs_dir[-1], exist_ok=True)

        # path for gifs from input images/frames
        input_gifs_dir = os.path.join(output_dir, "input_gifs")
        os.makedirs(input_gifs_dir, exist_ok=True)
        


    # initialize helper class
    helper = ControlNetHelper(use_size_512 = inference_args.use_size_512)
    if inference_args.extract_control_conditions:
        if 'depth' in inference_args.control_types:
            helper.add_depth_estimator()
        if 'canny' in inference_args.control_types:
            pass # canny can be done with cv2 library directly 
        if 'normal' in inference_args.control_types:
            helper.add_normal_estimator()
        if 'segmentation' in inference_args.control_types:
            helper.add_segmentation_estimator()    
        if 'softedge' in inference_args.control_types:
            helper.add_softedge_estimator()
        if 'lineart' in inference_args.control_types:
            helper.add_lineart_estimator()
        if 'openpose' in inference_args.control_types:
            helper.add_openpose_estimator()
        if 'scribble' in inference_args.control_types:
            helper.add_scribble_estimator()


    ### set up controlnet models
    pipe_line_args = {
        "torch_dtype": data_type, 
        "use_safetensors": True, 
        'helper': helper, 
        'adapter': adapter
        }
    if num_experts > 1:
        pipe_line_args['router'] = router
    pipe_line_args['controlnet'] = {}
    model_paths = {
        'depth': "lllyasviel/control_v11f1p_sd15_depth",
        'canny': "lllyasviel/control_v11p_sd15_canny",
        'normal': "lllyasviel/control_v11p_sd15_normalbae",
        'segmentation': "lllyasviel/control_v11p_sd15_seg",
        'softedge': "lllyasviel/control_v11p_sd15_softedge",
        'lineart': "lllyasviel/control_v11p_sd15_lineart",
        'openpose': "lllyasviel/control_v11p_sd15_openpose",
        'scribble': "lllyasviel/control_v11p_sd15_scribble"
    }
    
    for control_type, model_path in model_paths.items():
        if (len(inference_args.control_types) == 1 and control_type in inference_args.control_types) or (len(inference_args.control_types) > 1): # single-condition control
            pipe_line_args['controlnet'][control_type] = ControlNetModel.from_pretrained(model_path, torch_dtype=data_type, use_safetensors=True)
       
    if len(inference_args.control_types) == 1:
        pipe_line_args['controlnet'] = pipe_line_args['controlnet'][inference_args.control_types[0]]
        inference_expert_masks = [1]
    else:
        multi_control_list = ["depth", "canny", 'normal', 'softedge', 'segmentation', 'lineart', 'openpose']
        pipe_line_args['controlnet'] = [pipe_line_args['controlnet'][k] for k in multi_control_list]
        inference_expert_masks = [ctrl_type in inference_args.control_types for ctrl_type in multi_control_list]
       
        
    # load pipelines
    if inference_args.model_name == 'i2vgenxl':
        pretrained_model_name_or_path = "ali-vilab/i2vgen-xl"
        from i2vgen_xl.pipelines.i2vgen_xl_controlnet_adapter_pipeline import I2VGenXLControlNetAdapterPipeline
        from i2vgen_xl.models.unets.unet_i2vgen_xl import I2VGenXLUNet
        pipe = I2VGenXLControlNetAdapterPipeline.from_pretrained(pretrained_model_name_or_path, **pipe_line_args).to(device)
        # need to reload unet from our modified code under dir i2vgenxl. otherwise the default diffuser code will be used
        pipe.unet = I2VGenXLUNet.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to(device, dtype=data_type)
        
    elif inference_args.model_name == 'svd':
        pretrained_model_name_or_path = "stabilityai/stable-video-diffusion-img2vid"
        from svd.pipelines.svd_controlnet_adapter_pipeline import SVDControlNetAdapterPipeline
        from svd.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
        pipe = SVDControlNetAdapterPipeline.from_pretrained(pretrained_model_name_or_path, **pipe_line_args).to(device)
        # need to reload unet from our modified code under dir svd. otherwise the default diffuser code will be used
        pipe.unet = UNetSpatioTemporalConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to(device, dtype=data_type)

    elif inference_args.model_name == 'sdxl':
        pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
        from sdxl.pipelines.sdxl_controlnet_adapter_pipeline import SDXLControlNetAdapterPipeline
        #from diffusers import StableDiffusionXLImg2ImgPipeline
        pipe = SDXLControlNetAdapterPipeline.from_pretrained(pretrained_model_name_or_path, **pipe_line_args).to(device)
        # refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-xl-refiner-1.0", 
        #     torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device, dtype=data_type)

    if inference_args.lora:
        pipe.load_lora_weights(inference_args.lora)
    
    if inference_args.xformers:
        pipe.enable_xformers_memory_efficient_attention()

    generator = torch.Generator().manual_seed(inference_args.seed) if inference_args.seed else None


    # start generation 
    samples = list(captions.keys())
    if inference_args.max_eval is not None:
        samples = samples[:inference_args.max_eval]

    for idx, sample in enumerate(samples): 
        
        print(f"generating sample {idx+1}/{len(samples)}")
        
        # prompt
        prompt = captions[sample]
        
        # load input images or 1st frame
        if (inference_args.eval_input_type == 'images' and inference_args.extract_control_conditions) \
            or inference_args.eval_input_type == 'frames':
                
            raw_input_path = os.path.join(raw_input_dir, sample)
            if os.path.isdir(raw_input_path):
                raw_frames = sorted(os.listdir(raw_input_path))
                raw_frames = sorted([img for img in raw_frames if ("png" in img or "jpg" in img)])[:inference_args.n_sample_frames]
                images_pil = [Image.open(os.path.join(raw_input_path, frame)) for frame in raw_frames]
            else:
                images_pil = [Image.open(raw_input_path)]
            images_pil = [center_crop_and_resize(img, output_size=(inference_args.width, inference_args.height)) for img in images_pil] 
            images_pil = images_pil[:inference_args.n_sample_frames]
        

        # load or extract condition images
        if inference_args.extract_control_conditions:
            all_conditioning_images_pil = []
            for control_condition in inference_args.control_types:
                extracted_condition_image = helper.prepare_conditioning_images(
                    images_pil, 
                    current_batch_control_types=[control_condition], 
                    num_Frames=len(images_pil))[control_condition]['conditioning_images_pil'][0]
                extracted_condition_image = [cond_image.resize((512, 512)) for cond_image in extracted_condition_image]
                all_conditioning_images_pil.append(extracted_condition_image)
        else:
            if type(condition_input_dir) != list:
                condition_input_dir = [condition_input_dir]
            
            all_conditioning_images_pil = []
            for cond_dir in condition_input_dir:
                condition_images_path = os.path.join(cond_dir, sample)
                if os.path.isdir(condition_images_path):
                    condition_frames = sorted(os.listdir(condition_images_path))[:inference_args.n_sample_frames]
                    conditioning_images_pil = [Image.open(os.path.join(condition_images_path, frame)) for frame in condition_frames]
                else:
                    conditioning_images_pil = [Image.open(condition_images_path)]
                    
                if inference_args.use_size_512:
                    # before giving to SDv1.5 ControlNet, center crop and resize the condition images to 512 * 512 
                    conditioning_images_pil = [center_crop_and_resize(img, output_size=(inference_args.width, inference_args.height)) for img in conditioning_images_pil] 
                    conditioning_images_pil = [img.resize((512, 512)) for img in conditioning_images_pil] 
                all_conditioning_images_pil.append(conditioning_images_pil)
    
        
        # set inference arguments
        kwargs = {
            'controlnet_conditioning_scale': inference_args.controlnet_conditioning_scale,
            'control_guidance_start': inference_args.control_guidance_start,
            'control_guidance_end': inference_args.control_guidance_end,

            'sparse_frames': inference_args.sparse_frames,
            'skip_conv_in': inference_args.skip_conv_in,
            'skip_time_emb': inference_args.skip_time_emb,
            'use_size_512': inference_args.use_size_512,
            'inference_expert_masks': inference_expert_masks
        }
        control_images = all_conditioning_images_pil[0] if num_experts ==1 else all_conditioning_images_pil
            
        
        # run pipelines
        if inference_args.model_name == 'i2vgenxl':
            num_frames = inference_args.n_sample_frames if 'n_sample_frames' in inference_args else 16 # default 
            target_fps = inference_args.output_fps if 'output_fps' in inference_args else 16 # default 
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                i2vgenxl_outputs = pipe(
                    prompt=prompt,
                    negative_prompt="Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms",
                    height = inference_args.height, 
                    width = inference_args.width,
                    image= images_pil[0],
                    control_images = control_images,
                    num_inference_steps=inference_args.num_inference_steps,
                    guidance_scale=9.0, 
                    generator=generator,
                    target_fps = target_fps,
                    num_frames = num_frames,
                    output_type="pil",
                    **kwargs
                ) 
                output_images = i2vgenxl_outputs.frames[0]

        elif inference_args.model_name == 'svd':
            num_frames = inference_args.n_sample_frames if 'n_sample_frames' in inference_args else 14 # default 
            target_fps = inference_args.output_fps if 'output_fps' in inference_args else 14 # default 
            control_images = control_images[:num_frames]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_images = pipe(
                    image=images_pil[0], 
                    control_images = control_images, 
                    prompt=prompt, # please note that for SVD, we also need prompt, which will be given as input to SDv1.5 ControlNet
                    decode_chunk_size=8, 
                    generator=generator, 
                    motion_bucket_id=127, 
                    height=inference_args.height, 
                    width=inference_args.width,
                    noise_aug_strength=0.02,
                    num_inference_steps=inference_args.num_inference_steps,
                    fps = target_fps,
                    num_frames = num_frames,
                    **kwargs
                ).frames[0]               
        
        elif inference_args.model_name == 'sdxl':
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sdxl_outputs, _, _ = pipe(prompt,
                            negative_prompt="Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms",
                            control_images = control_images,
                            width=inference_args.width,
                            height=inference_args.height,
                            num_inference_steps=inference_args.num_inference_steps,
                            generator=generator,
                            output_type="pil", 
                            **kwargs)
            output_images = sdxl_outputs.images[0]
            #output_images = [refiner(prompt=prompt, image=output_images).images[0]][0] # using refiner is optional


        # save generated images, condition images
        if inference_args.eval_input_type == 'images':
            # 1. save input raw image (if extract condition from raw image)
            if inference_args.extract_control_conditions:
                images_pil[0].save(os.path.join(input_images_dir, sample))
            # 2. save condition image
            all_conditioning_images_pil[0][0].save(os.path.join(output_condition_images_dir[0], sample))
            # 3. save generated image
            output_images.save(os.path.join(output_images_dir, sample))
            # 4. save concatenated image
            (h, w) = output_images.size
            new_image = Image.new('RGB', (w * 2, h))
            new_image.paste(all_conditioning_images_pil[0][0].resize((h, w)), (0, 0))
            new_image.paste(output_images, (w, 0))
            new_image.save(os.path.join(concat_output_dir, sample))


        elif inference_args.eval_input_type == 'frames':
            # 1. save input frames 
            frame_input_dir = os.path.join(input_images_dir, sample)
            os.makedirs(frame_input_dir, exist_ok=True)
            num_input_frames = num_frames if inference_args.extract_control_conditions else 1
            _ = [images_pil[k].save(os.path.join(frame_input_dir, f"{k:05d}.png")) for k in range(len(images_pil[:num_input_frames]))]
            # 2. save input gif (if condition not extracted from raw frames, this gif will just be constructed from 1st frame, which is static)
            save_as_gif(images_pil[:num_input_frames], os.path.join(input_gifs_dir, f"{sample}.gif"), 
                        duration=inference_args.video_duration // inference_args.video_length)
            # 3. save condition frames             
            for i, ctrl_type in enumerate(inference_args.control_types):
                condition_frame_output_dir = os.path.join(output_condition_images_dir[i], sample)
                os.makedirs(condition_frame_output_dir, exist_ok=True)
                _ = [all_conditioning_images_pil[i][k].save(os.path.join(condition_frame_output_dir, f"{k:05d}.png")) for k in range(len(output_images))]
            # 4. save condition gif 
            for i, ctrl_type in enumerate(inference_args.control_types):
                save_as_gif(all_conditioning_images_pil[i], os.path.join(output_condition_gifs_dir[i], f"{sample}.gif"), 
                            duration=inference_args.video_duration // inference_args.video_length)
            # 5. save output frames 
            frame_output_dir = os.path.join(output_images_dir, sample)
            os.makedirs(frame_output_dir, exist_ok=True)
            _ = [output_images[k].save(os.path.join(frame_output_dir, f"{k:05d}.png")) for k in range(len(output_images))]
            # 6. save output gif 
            save_as_gif(output_images, os.path.join(output_gifs_dir, f"{sample}.gif"), 
                        duration=inference_args.video_duration // inference_args.video_length)
            # 7. save concat gifs
            save_concatenated_gif(
                images_pil[0], 
                os.path.join(concat_output_dir, f"{sample}.gif"), 
                #[all_conditioning_images_pil[0][:num_frames], output_images], 
                [cond_pil[:num_frames] for cond_pil in all_conditioning_images_pil] + [output_images],
                inference_args.video_duration // inference_args.video_length
                )
        
    return 



if __name__ == "__main__":


    inference_parser = argparse.ArgumentParser('Ctrl-Adapter inference', parents=[parse_inference_args()])
    inference_args = inference_parser.parse_args()
    
    output = inference_main(inference_args)

