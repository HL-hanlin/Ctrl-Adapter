import os
import datetime
import argparse
import torch 
from PIL import Image 

from controlnet.controlnet import ControlNetModel

from utils.utils import bool_flag, count_params, batch_to_device, save_as_gif
from utils.ctrl_adapter import ControlNetAdapter, ControlNetHelper
    
from i2vgen_xl.models.unets.unet_i2vgen_xl import I2VGenXLUNet
from i2vgen_xl.pipelines.i2vgen_xl_controlnet_adapter_pipeline import I2VGenXLControlNetAdapterPipeline


def parse_args(input_args=None):
    
    parser = argparse.ArgumentParser(description="Inference arguments for Ctrl-Adapter.", add_help=False)
    
    ### inputs ###
    parser.add_argument("--input_path", type=str, default="assets/example_depth_cat")
    parser.add_argument("--prompt", type=str, default="A cat wearing glasses reading a newspaper on a red sofa")
    
    ### model path ###
    parser.add_argument("--model_name", type=str, default="i2vgenxl")
    parser.add_argument("--checkpoint_folder", type=str, default="i2vgenxl_depth")
    
    ### inference parameters ###
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--lora", type=str)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--video_length", type=int, default=8)
    parser.add_argument("--video_duration", type=int, default=1000)
    #args.add_argument("--low_vram_mode", action="store_true")
    parser.add_argument('--scheduler', type=str, default='EulerDiscreteScheduler', help='Name of the scheduler to use')
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--control_guidance_start", type=float, default=0.0)
    parser.add_argument("--control_guidance_end", type=float, default=1.0)
    parser.add_argument("--classifier_free_guidance_scale", type=float, default=9.0)
    parser.add_argument("--negative_prompt", type=str, default="Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms")
    parser.add_argument("--precision", type=str, default='bf16', choices=['f16', 'f32', 'bf16'])
    parser.add_argument("--n_sample_frames", type=int, default=16)
    parser.add_argument("--output_fps", type=int, default=16)
    parser.add_argument("--control_types",  nargs='+', default='depth', choices=["depth", "canny", 'normal', 'segmentation', 'openpose', 'softedge', 'lineart', 'scribble'])
    parser.add_argument('--sparse_frames', nargs='+', default = None)
    parser.add_argument("--inference_expert_masks",  nargs='+', default=None)
    parser.add_argument('--use_size_512', default=True, type=bool_flag,)
    parser.add_argument('--skip_conv_in', default=False, type=bool_flag,)
    parser.add_argument('--skip_time_emb', default=False, type=bool_flag,)
    parser.add_argument('--adapter_locations',  nargs='+', default=['A', 'B', 'C', 'D', 'M'], choices=['A', 'B', 'C', 'D', 'M'])
    
    args = parser.parse_args()
    
    return args



def run_inference(args):
    
    args = parse_args()
        
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(args.seed) if args.seed else None

    data_type = torch.float32
    if args.precision == 'f16':
        data_type = torch.half
    elif args.precision == 'f32':
        data_type = torch.float32
    elif args.precision == 'bf16':
        data_type = torch.bfloat16
        
    
    
    #### load adapter and controlnet helper
    adapter = ControlNetAdapter.from_pretrained("hanlincs/Ctrl-Adapter", subfolder=args.checkpoint_folder, 
                low_cpu_mem_usage=False, device_map=None)
    
    helper = ControlNetHelper(use_size_512 = args.use_size_512)
    
    """
    ### the following code contains functions that can extract control frames directly from input raw video ###
    ### we'll add support to this option later ###
    
    estimator_functions = {
        'depth': helper.add_depth_estimator,
        'normal': helper.add_normal_estimator,
        'segmentation': helper.add_segmentation_estimator,
        'softedge': helper.add_softedge_estimator,
        'lineart': helper.add_lineart_estimator,
        'openpose': helper.add_openpose_estimator,
        'scribble': helper.add_scribble_estimator
    }
    
    for control_type, function in estimator_functions.items():
        if control_type in args.control_types:
            function()
    """


    
    ### set up controlnet models
    pipe_line_args = {"torch_dtype": data_type, 
                      "use_safetensors": True, 
                      'helper': helper, 
                      'adapter': adapter}
    
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
        if control_type in args.control_types:
            pipe_line_args['controlnet'][control_type] = ControlNetModel.from_pretrained(model_path, torch_dtype=data_type, use_safetensors=True)

    pipe_line_args['controlnet'] = [pipe_line_args['controlnet'][k] for k in pipe_line_args['controlnet'].keys()]
    pipe_line_args['controlnet'] = pipe_line_args['controlnet'][0]
    inference_expert_masks = [1 for i in range(len(args.control_types))]
    
    
    
    ### load I2VGenXL ControlNet pipeline 
    # we'll add support to other backbone models used in our paper soon. Stay tuned!
    pipe = I2VGenXLControlNetAdapterPipeline.from_pretrained("ali-vilab/i2vgen-xl", **pipe_line_args).to(device)
    # need to reload unet from our modified code under dir i2vgenxl. otherwise the default diffuser code will be used
    pipe.unet = I2VGenXLUNet.from_pretrained("ali-vilab/i2vgen-xl", data_type=torch.float16, subfolder="unet").to(device)

    if args.xformers:
        pipe.enable_xformers_memory_efficient_attention()

    

    ### load inference images 
    input_dir = args.input_path 
    condition_folder_path = os.path.join(input_dir, "control_frames")
    
    prompt = args.prompt

    first_frame_pil = Image.open(os.path.join(input_dir, "first_frame.png"))
    condition_frames = sorted(os.listdir(condition_folder_path))[:args.n_sample_frames]
    conditioning_images_pil = [Image.open(os.path.join(condition_folder_path, frame)) for frame in condition_frames]
    
    
    
    ### start inference     
    kwargs = {
        'height': args.height,
        'width': args.width,
        
        'controlnet_conditioning_scale': args.controlnet_conditioning_scale,
        'control_guidance_start': args.control_guidance_start,
        'control_guidance_end': args.control_guidance_end,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.classifier_free_guidance_scale,
        
        'sparse_frames': args.sparse_frames,
        'skip_conv_in': args.skip_conv_in,
        'skip_time_emb': args.skip_time_emb,
        'use_size_512': args.use_size_512,
        'adapter_locations': args.adapter_locations,
        'inference_expert_masks': inference_expert_masks
    }

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            i2vgenxl_outputs = pipe(
                prompt=prompt,
                control_images = conditioning_images_pil,
                negative_prompt=args.negative_prompt,
                image= first_frame_pil,
                generator=generator,
                target_fps = args.output_fps,
                num_frames = args.n_sample_frames,
                output_type="pil",
                **kwargs
            )
            images = i2vgenxl_outputs.frames[0]


    os.makedirs("outputs", exist_ok=True)
    save_as_gif(images, f"outputs/{prompt}.gif", duration=args.video_duration // args.video_length)



if __name__ == "__main__":
    
    args = parse_args()
    run_inference(args)
    
    
    