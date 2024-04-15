import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from einops import rearrange

import torch 
from torch.utils.data import Dataset
import torchvision.transforms as T

import decord
decord.bridge.set_bridge('torch')

# load util functions
from utils.utils import center_crop_and_resize, image_to_tensor

# load util functions for specific backbones
# To train Ctrl-Adapter on a new backbone, you can follow the same structure by 
# creating a utils_xxx function under utils folder, and import here
from utils.utils_sdxl import add_time_ids 
from utils.utils_svd import _resize_with_antialiasing
from utils.utils_i2vgenxl import _resize_bilinear



class VideoLoader(Dataset):
    def __init__(
        self,
        width: int = 512, # we resize the width of training videos to this value
        height: int = 512, # we resize the width of training videos to this value
        n_sample_frames: int = 14, # number of frames to sample. We recommend setting this as the same default value of the backbone model
        output_fps: int = 3, # the input videos are mp4 format, output_fps defines the frame rate we sample for training
        path: str = "./data", # training video folder
        metadata_path: str = None, # a csv file which contains prompts for training videos
        use_empty_prompt = False,
        model_name = None, # we support SDXL right now. you can follow the structure here and adapt to a new backbone model
        **kwargs
    ):

        self.width = width
        self.height = height

        self.model_name = model_name
        self.n_sample_frames = n_sample_frames
        self.output_fps = output_fps 
        self.use_empty_prompt = use_empty_prompt
        
        self.path = path 
        self.metadata_path = metadata_path
        self.video_files = glob(f"{path}/*.mp4") # make sure the training videos are of mp4 format
        self.video_files = sorted(self.video_files)

        # read the csv file for prompts
        if self.metadata_path is not None:
            metadata = pd.read_csv(self.metadata_path)[['filename']]
            metadata_list = list(metadata['filename'].values)
            self.metadata = {metadata_list[i]: metadata_list[i].replace(".mp4", "") for i in range(len(metadata_list)) }
            del metadata, metadata_list

        # make sure self.image_files only contains the images with prompts specified in the csv file
        self.video_files = list(set(self.video_files).intersection(set([path + '/' + l for l in list(self.metadata.keys())])))
        self.video_files = sorted(self.video_files)
        
        # we use fixed seed here (optional)
        random.seed(42)
        random.shuffle(self.video_files)

        
        # please note that we need to set backbone-specific parameters here
        # these parameters are used in the forward function

        if self.model_name == 'hotshotxl':
            self.tokenizer = kwargs['tokenizer'] 
            self.tokenizer_2 = kwargs['tokenizer_2'] 
            self.unet_config = kwargs['unet_config']
            self.unet_add_embedding = kwargs['unet_add_embedding']
            self.og_size = kwargs['og_size']
            self.crop_coords = kwargs['crop_coords']
            self.required_size = kwargs['required_size']
            
        elif self.model_name == 'i2vgenxl':
            self.tokenizer = kwargs['tokenizer'] 
            self.feature_extractor = kwargs['feature_extractor']
            self.image_processor = kwargs['image_processor']
            
        elif self.model_name == 'svd':
            self.tokenizer = kwargs['tokenizer'] 
            self.feature_extractor = kwargs['feature_extractor']
            self.image_processor = kwargs['image_processor']            
        
        
    # this function extracts n_sample_frames from the mp4 file
    def get_frame_batch(self, vr):
        
        n_sample_frames = self.n_sample_frames
        len_vr = len(vr)
        
        input_fps = vr.get_avg_fps()
        every_k_frame = int(np.ceil(input_fps // self.output_fps))
        
        if (len_vr - n_sample_frames*every_k_frame <0): 
            # if the total number of frames in the mp4 file cannot give us n_sample_frame at desired fps, 
            # we randomly sample n_sample_frames from the mp4 file 
            idxs = np.array(sorted(random.sample(list(range(len_vr)), self.n_sample_frames)))
        elif every_k_frame == 0: 
            # if the fps of the mp4 file is too low (lower than our desired fps),
            # we randomly sample n_smaple_frames from the mp4 file
            idxs = np.array(sorted(random.sample(list(range(len_vr)), self.n_sample_frames)))
        else:
            # randomly set a starting frame, and then sample n_sample_frames at desired fps
            start_idx = random.randint(0, (len_vr - n_sample_frames*every_k_frame)) 
            start_idx = int((len_vr - n_sample_frames*every_k_frame)//2)
            idxs = np.arange(start_idx, start_idx + every_k_frame*n_sample_frames, every_k_frame)

        if len(idxs) < self.n_sample_frames:
            # if the sampled frames is still less than n_sample_frames, 
            # we randomly sample n_smaple_frames from the mp4 file
            idxs = np.array(sorted(random.sample(list(range(len_vr)), self.n_sample_frames)))
        
        video = vr.get_batch(idxs) # extract these frames from the mp4 file 
        video = rearrange(video, "f h w c -> f c h w") 
        _, _, h, w = video.shape
        
        # create pil images and pixel_values from these extracted frames
        images_pil = [T.functional.to_pil_image(image) for image in video]
        images_pil = [center_crop_and_resize(img, (self.width, self.height)) for img in images_pil]
        pixel_values = torch.stack([image_to_tensor(x) for x in images_pil])
            
        return images_pil, pixel_values 
        
    
    @staticmethod
    def __getname__(): 
        return 'folder'


    def __len__(self):
        return len(self.video_files)

        
    def __getitem__(self, index):
        while True: 
            try:
                vid_path = self.video_files[index]
                
                # load video prompt
                vid_filename = vid_path.split(self.path)[1][1:] 
                vid_prompt = self.metadata[vid_filename]
                
                # use decord library to read mp4 files
                vr = decord.VideoReader(vid_path)
                if len(vr) < self.n_sample_frames: # skip the videos with number of frames less than n_sample_frames 
                    index = random.randint(0, len(self.video_files))    
                else:
                    images_pil, frames = self.get_frame_batch(vr) # extract n_sample_frames from the mp4 file
                    break
            except:
                index = random.randint(0, len(self.video_files))
        
        # prompt
        text = "" if self.use_empty_prompt else vid_prompt
        
        # We perform some basic data preprocessing and obtain tokenizers in this data loader
        # which can be done by CPUs without using GPUs
        if self.model_name == 'i2vgenxl':

            input_ids = self.tokenizer(text, padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length,).input_ids
            
            image = _resize_bilinear(images_pil[0], (self.feature_extractor.crop_size["width"], self.feature_extractor.crop_size["height"]))
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            image_for_embeddings = self.feature_extractor(images=image, do_normalize=True, do_center_crop=False, do_resize=False, do_rescale=False, return_tensors="pt",).pixel_values
            image_for_latents = self.image_processor.preprocess(images_pil[0])

            return {
                "frames": frames,
                "images_pil": images_pil, 
                "input_ids": input_ids,
                "image_for_embeddings": image_for_embeddings,
                "image_for_latents": image_for_latents,
                }
            
        elif self.model_name == 'svd':
            
            input_ids = self.tokenizer(text, padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length,).input_ids
            
            image = self.image_processor.pil_to_numpy(images_pil[0])
            image = self.image_processor.numpy_to_pt(image)
            
            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0
            
            image_for_embeddings = self.feature_extractor(images=image, do_normalize=True, do_center_crop=False, do_resize=False, do_rescale=False, return_tensors="pt",).pixel_values
            image_for_latents = self.image_processor.preprocess(images_pil[0], height=self.height, width=self.width)
            
            return {
                "frames": frames,
                "images_pil": images_pil, 
                "input_ids": input_ids,
                "image_for_embeddings": image_for_embeddings,
                "image_for_latents": image_for_latents,
                }
            
            
        
class ImageLoader(Dataset):
    def __init__(
        self,
        width: int = 1024, # we resize the width of training images to this value
        height: int = 1024, # we resize the height of training images to this value
        path: str = "./data", # training image folder
        metadata_path: str = None, # a csv file which contains prompts for training images
        use_empty_prompt = False,   
        model_name = None, # we support SDXL right now. you can follow the structure here and adapt to a new backbone model
        **kwargs
    ):

        self.width = width
        self.height = height
        
        self.model_name = model_name
        self.use_empty_prompt = use_empty_prompt
        
        # load training images
        self.path = path 
        self.image_files = glob(f"{path}/*.jpg")
        self.image_files = sorted(self.image_files)
        
        # read the csv file for prompts
        self.metadata_path = metadata_path
        if self.metadata_path is not None: # for webvid dataset
            metadata = pd.read_csv(self.metadata_path)[['filename', 'caption']]
            self.metadata = metadata.set_index('filename')['caption'].to_dict() 
            del metadata 
        
        # make sure self.image_files only contains the images with prompts specified in the csv file
        self.image_files = list(set(self.image_files).intersection(set([path + '/' + l for l in list(self.metadata.keys())])))
        self.image_files = sorted(self.image_files)
        
        # we use fixed seed here (optional)
        random.seed(42)
        random.shuffle(self.image_files)
        
        
        # please note that we need to set backbone-specific parameters here
        # these parameters are used in the forward function
        
        if self.model_name == 'sdxl':
            self.tokenizer = kwargs['tokenizer'] 
            self.tokenizer_2 = kwargs['tokenizer_2'] 
            self.unet_config = kwargs['unet_config']
            self.unet_add_embedding = kwargs['unet_add_embedding']
            self.og_size = kwargs['og_size']
            self.crop_coords = kwargs['crop_coords']
            self.required_size = kwargs['required_size']

        
    @staticmethod
    def __getname__(): 
        return 'folder'


    def __len__(self):
        return len(self.image_files)

        
    def __getitem__(self, index):
        while True: 
            try:
                img_path = self.image_files[index]
                
                # load image prompt
                img_filename = img_path.split(self.path)[1][1:] 
                img_prompt = self.metadata[img_filename]
                
                # load pil image, and then get pixel_values
                image_pil = Image.open(img_path)
                image_pil = center_crop_and_resize(image_pil, (self.width, self.height))
                pixel_values = image_to_tensor(image_pil) # pixel_values are between [-1, 1]
                
                break
                
            except:
                index = random.randint(0, len(self.image_files))
        
        # prompt
        text = "" if self.use_empty_prompt else img_prompt

        # We perform some basic data preprocessing and obtain tokenizers in this data loader
        # which can be done by CPUs without using GPUs
        if self.model_name == 'sdxl':
            
            # modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
            additional_time_ids = add_time_ids(self.unet_config, self.unet_add_embedding, self.og_size, self.crop_coords,
                                               (self.required_size[0], self.required_size[1]), dtype=torch.float32)
            
            input_ids_0 = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length,).input_ids
            input_ids_1 = self.tokenizer_2(text, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length,).input_ids
            
            return {
                "frames": pixel_values.unsqueeze(0),
                "images_pil": [image_pil], 
                "input_ids_0": input_ids_0,
                "input_ids_1": input_ids_1,
                "additional_time_ids": additional_time_ids,
                }
        
