import cv2 
import numpy as np 
from PIL import Image 
import itertools 
from einops import rearrange
from typing import List, Optional

import torch
import torchvision.transforms as T

from transformers import AutoTokenizer

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import VaeImageProcessor

from utils.utils import batch_to_device, import_model_class_from_model_name_or_path



class ControlNetHelper(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self, pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5", use_size_512=True):
        super().__init__()

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.variant = None 
        self.revision = None
        self.weight_dtype = torch.float16
        self.use_size_512 = use_size_512

        ### load vae and controlnet ###
        text_encoder_cls = import_model_class_from_model_name_or_path(
            self.pretrained_model_name_or_path, self.revision) # import correct text encoder class
        
        self.text_encoder = text_encoder_cls.from_pretrained(
            self.pretrained_model_name_or_path, 
            subfolder="text_encoder", 
            revision=self.revision, 
            variant=self.variant)

        ### disable gradients ###
        self.text_encoder.requires_grad = False

        ### Load the tokenizer ###
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path, 
            subfolder="tokenizer",
            revision=self.revision,
            use_fast=False,
        )

        ### image processor for controlnet
        self.vae_scale_factor = 8
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

        # our current checkpoints are trained on 512 * 512 resolution, so by default we first resize and center crop the input videos
        if self.use_size_512: 
            self.conditioning_image_transforms = T.transforms.Compose(
                [
                    T.transforms.Resize(512, interpolation=T.transforms.InterpolationMode.BILINEAR),
                    T.transforms.CenterCrop(512),
                    T.transforms.ToTensor(),
                ]
            )
        else:
            self.conditioning_image_transforms = T.transforms.Compose([T.transforms.ToTensor(),])
            

    @torch.no_grad()
    def add_depth_estimator(self, estimator_ckpt_path=None):
        if estimator_ckpt_path is not None: # use personallized depth estimator (need to set 'estimator_ckpt_path' to the depth estimator checkpoint path)
            from utils.run_depth import DepthMidas
            self.depth_estimator = DepthMidas(estimator_ckpt_path)
        else:
            from transformers import pipeline
            self.depth_estimator = pipeline('depth-estimation')
        self.estimator_ckpt_path = estimator_ckpt_path


    @torch.no_grad()
    def add_normal_estimator(self):
        from controlnet_aux import NormalBaeDetector
        self.normal_estimator = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
        self.normal_estimator = self.normal_estimator.to("cuda")


    @torch.no_grad()
    def add_segmentation_estimator(self):
        from utils.ada_palette import ada_palette
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation, SegformerForSemanticSegmentation
        self.ada_palette = ada_palette
        #self.segmentation_image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        #self.segmentation_image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small").cuda()
        self.segmentation_image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
        self.segmentation_image_segmentor = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640").to("cuda")


    @torch.no_grad()
    def add_softedge_estimator(self):
        from controlnet_aux import PidiNetDetector, HEDdetector
        self.softedge_processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
        self.softedge_processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
        self.softedge_processor = self.softedge_processor.to("cuda")
        

    @torch.no_grad()
    def add_lineart_estimator(self):
        from controlnet_aux import LineartDetector
        self.lineart_processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
        self.lineart_processor = self.lineart_processor.to("cuda")
        

    @torch.no_grad()
    def add_shuffle_estimator(self):
        from controlnet_aux import ContentShuffleDetector
        self.shuffle_processor = ContentShuffleDetector()


    @torch.no_grad()
    def add_scribble_estimator(self):
        from controlnet_aux import HEDdetector
        self.scribble_processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
        self.scribble_processor = self.scribble_processor.to("cuda")


    @torch.no_grad()
    def add_openpose_estimator(self, hand_and_face=False):
        from controlnet_aux import OpenposeDetector
        self.openpose_processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        self.openpose_processor = self.openpose_processor.to("cuda")
        self.openpose_hand_and_face = hand_and_face
        
        
    @torch.no_grad()
    def post_process_conditioning_pil_and_pixel_values(self, conditioning_images_pil, num_Frames=8, img_size=(512, 512)):
        conditioning_images_pil = [cond_img_pil.resize(img_size) for cond_img_pil in conditioning_images_pil]
        processed_conditioning_pixel_values = torch.stack(
            [self.conditioning_image_transforms(image.convert("RGB")) for image in conditioning_images_pil])
        processed_conditioning_images_pil = [conditioning_images_pil[i:i + num_Frames] 
                                             for i in range(0, len(conditioning_images_pil), num_Frames)]
        return processed_conditioning_images_pil, processed_conditioning_pixel_values
        
        
    @torch.no_grad()
    def prepare_conditioning_images(self, batch_images_pils, current_batch_control_types, img_size=(512, 512), num_Frames=8):
        
        if type(current_batch_control_types) == str:
            current_batch_control_types = [current_batch_control_types]
            
        all_conditioning_dict = {}
        
        
        for control_type in current_batch_control_types:
            
            if control_type == 'depth':
                if self.estimator_ckpt_path is None:
                    conditioning_images_pil = []
                    for i in range(len(batch_images_pils)):
                        image = self.depth_estimator(batch_images_pils[i])['depth']
                        image = np.array(image)
                        image = image[:, :, None]
                        image = np.concatenate([image, image, image], axis=2)
                        conditioning_images_pil.append(Image.fromarray(image))
                else:
                    conditioning_images_pil = self.depth_estimator.estimate(batch_images_pils, new_size = img_size)
                    
            if control_type == 'canny':
                all_conditioning_dict['canny'] = {}
                low_threshold, high_threshold = 100, 200
                conditioning_images_pil = []
                for i in range(len(batch_images_pils)):
                    image = cv2.Canny(np.array(batch_images_pils[i]), low_threshold, high_threshold)
                    image = image[:, :, None]
                    image = np.concatenate([image, image, image], axis=2)
                    conditioning_images_pil.append(Image.fromarray(image))
                    
            if control_type == 'normal':
                conditioning_images_pil = [self.normal_estimator(img_pil) for img_pil in batch_images_pils]
                
            if control_type == 'softedge':
                conditioning_images_pil = [self.softedge_processor(img_pil) for img_pil in batch_images_pils]

            if control_type == 'openpose':
                conditioning_images_pil = [self.openpose_processor(img_pil) for img_pil in batch_images_pils]

            if control_type == 'lineart':
                conditioning_images_pil = [self.lineart_processor(img_pil) for img_pil in batch_images_pils]

            if control_type == 'shuffle':
                conditioning_images_pil = [self.shuffle_processor(img_pil) for img_pil in batch_images_pils]
                
            if control_type == 'scribble':
                conditioning_images_pil = [self.scribble_processor(img_pil, scribble=True) for img_pil in batch_images_pils]

            if control_type == 'segmentation':
                all_conditioning_dict['segmentation'] = {}
                conditioning_images_pil = []
                for image in batch_images_pils:
                    #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pixel_values = self.segmentation_image_processor(image, return_tensors="pt").pixel_values
                    with torch.no_grad():
                        outputs = self.segmentation_image_segmentor(pixel_values.cuda())
                        seg = self.segmentation_image_processor.post_process_semantic_segmentation(
                            outputs, target_sizes=[image.size[::-1]])[0]
                        seg = seg.cpu()
                        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
                        for label, color in enumerate(self.ada_palette):
                            color_seg[seg == label, :] = color
    
                    color_seg = color_seg.astype(np.uint8)
                    control_image = Image.fromarray(color_seg)
                    conditioning_images_pil.append(control_image)
                             
            
            processed_conditioning_images_pil, processed_conditioning_pixel_values = \
                self.post_process_conditioning_pil_and_pixel_values(
                    conditioning_images_pil, num_Frames = num_Frames, img_size=img_size)
                
            all_conditioning_dict[control_type] = {
                'conditioning_images_pil': processed_conditioning_images_pil,
                'conditioning_pixel_values': processed_conditioning_pixel_values
                }
            
        return all_conditioning_dict
            
    
    @torch.no_grad()
    def prepare_batch(self, batch, current_batch_control_types = ['depth'], control_condition_estimation=True):
        
        bsz = len(batch['images_pil'])
        num_Frames = len(batch['images_pil'][0])
        img_size = batch['images_pil'][0][0].size

        if control_condition_estimation:
            batch_images_pils = list(itertools.chain(*batch['images_pil']))
            
            kwargs = {
                'img_size': img_size,
                "current_batch_control_types": current_batch_control_types,
                "num_Frames": num_Frames,
                }
            
            all_conditioning_dict = self.prepare_conditioning_images(batch_images_pils, **kwargs)
            batch.update(all_conditioning_dict)
            
        batch_to_device(batch, self.device)
        for k in current_batch_control_types:
            batch_to_device(batch[k], self.device)
                
        batch['frames'] = batch['frames'].detach()
        if len(batch['frames'].shape) == 4:
            batch['frames'] = rearrange(batch['frames'], "(b f) c h w -> b f c h w", b=bsz, f=num_Frames)

        for k in current_batch_control_types:
            if len(batch[k]['conditioning_pixel_values'].shape) == 4:
                batch[k]['conditioning_pixel_values'] = rearrange(
                    batch[k]['conditioning_pixel_values'], "(b f) c h w -> b f c h w", b=bsz, f=num_Frames)

        return batch
        

    ### Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image
    @torch.no_grad()
    def prepare_images(
        self,
        images,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        images_pre_processed = [self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32) for image in images]

        images_pre_processed = torch.cat(images_pre_processed, dim=0)

        repeat_factor = [1] * len(images_pre_processed.shape)
        repeat_factor[0] = batch_size * num_images_per_prompt
        images_pre_processed = images_pre_processed.repeat(*repeat_factor)

        images = images_pre_processed.unsqueeze(0)
        images = images.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            repeat_factor = [1] * len(images.shape)
            repeat_factor[0] = 2
            images = images.repeat(*repeat_factor)

        return images


    ### function to encode input prompt to controlnet
    @torch.no_grad()
    def encode_controlnet_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            attention_mask = None
            
            #with torch.no_grad():
            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                pooled_prompt_embeds = prompt_embeds[1]
                prompt_embeds = prompt_embeds[0]
            else:
                #with torch.no_grad():
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True)
                #pooled_prompt_embeds = self.text_encoder(text_input_ids.to(device))
                pooled_prompt_embeds = prompt_embeds[1]
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            # TODO: L380: self.config.force_zeros_for_empty_prompt
            
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None
            #with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_pooled_prompt_embeds = negative_prompt_embeds[1] ## newly added
            negative_prompt_embeds = negative_prompt_embeds[0]

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        
        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )
            
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    

    ### Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids
    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
     
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

