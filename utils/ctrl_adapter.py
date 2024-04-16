import cv2 
import numpy as np 
from PIL import Image 
import itertools 
from einops import rearrange

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

from transformers import AutoTokenizer

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from utils.adapter_spatial_temporal import AdapterSpatioTemporal
from utils.utils import batch_to_device, import_model_class_from_model_name_or_path



class ControlNetAdapter(ModelMixin, ConfigMixin,):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        backbone_model_name,
        
        num_blocks = 2,
        num_frames = 8,
        num_adapters_per_location = 3,
        
        cross_attention_dim = None,
        adapter_type='spatial_temporal_resnet_transformer', 
        
        ### choose which modules to activate in each ctrl-adapter ###
        add_spatial_resnet = True, 
        add_temporal_resnet = False,
        add_spatial_transformer = True, 
        add_temporal_transformer = False,
        
        ### choose where to insert ctrl-adapters ###
        add_adapter_location_A = False, 
        add_adapter_location_B = False, 
        add_adapter_location_C = False, 
        add_adapter_location_D = False, 
        add_adapter_location_M = False, 
    ):
        super().__init__()
        
        self.add_adapter_location_A = add_adapter_location_A
        self.add_adapter_location_B = add_adapter_location_B
        self.add_adapter_location_C = add_adapter_location_C
        self.add_adapter_location_D = add_adapter_location_D
        self.add_adapter_location_M = add_adapter_location_M

        self.num_adapters_per_location = num_adapters_per_location 

        if add_adapter_location_M:
            mid_block_channels = 1280

        down_blocks_channels = []

        if add_adapter_location_A:
            down_blocks_channels = [320] * self.num_adapters_per_location

        if add_adapter_location_B:
            if self.num_adapters_per_location == 3:
                down_blocks_channels += [320, 640, 640]
            elif self.num_adapters_per_location == 2:
                down_blocks_channels += [320, 640]
            elif self.num_adapters_per_location == 1:
                down_blocks_channels += [640]

        if add_adapter_location_C:
            if self.num_adapters_per_location == 3:
                down_blocks_channels += [640, 1280, 1280]
            elif self.num_adapters_per_location == 2:
                down_blocks_channels += [640, 1280]
            elif self.num_adapters_per_location == 1:
                down_blocks_channels += [1280]

        if add_adapter_location_D:
            if self.num_adapters_per_location == 3:
                down_blocks_channels += [1280] * self.num_adapters_per_location

        self.down_blocks_adapter = nn.ModuleList([])
        
        self.num_adapters = len(down_blocks_channels) 
        self.adapter_type = adapter_type
        
        
        ### down blocks ###
        for i in range(self.num_adapters): 
            config = {"in_channels": down_blocks_channels[i], 
                        "out_channels": down_blocks_channels[i], 
                        "cross_attention_dim": cross_attention_dim,
                        "num_layers": num_blocks,
                        "up": True if backbone_model_name in ['sdxl'] else False,
                        "add_spatial_resnet": add_spatial_resnet, 
                        "add_temporal_resnet": add_temporal_resnet,
                        "add_spatial_transformer": add_spatial_transformer, 
                        "add_temporal_transformer": add_temporal_transformer,
                        }
            self.down_blocks_adapter.append(AdapterSpatioTemporal(**config))
             
            
        ### mid block ###
        if add_adapter_location_M:
            config = {"in_channels": mid_block_channels, 
                        "out_channels": mid_block_channels, 
                        "cross_attention_dim": cross_attention_dim,
                        "num_layers": num_blocks,
                        "up": True if backbone_model_name in ['sdxl'] else False,
                        "add_spatial_resnet": add_spatial_resnet, 
                        "add_temporal_resnet": add_temporal_resnet,
                        "add_spatial_transformer": add_spatial_transformer, 
                        "add_temporal_transformer": add_temporal_transformer,
                        }
            self.mid_block_adapter = AdapterSpatioTemporal(**config)
        else:
            self.mid_block_adapter = None
            
    

    def forward(self, down_block_res_samples, mid_block_res_sample=None, sparsity_masking=None, 
                num_frames=None, timestep=None, encoder_hidden_states=None):

        down_block_ids = []

        if self.add_adapter_location_A:
            selection_map = {3: [0, 1, 2], 2: [0, 2], 1: [2]}
            down_block_ids += selection_map.get(self.num_adapters_per_location, [])
            
        if self.add_adapter_location_B:
            selection_map = {3: [3, 4, 5], 2: [3, 5], 1: [5]}
            down_block_ids += selection_map.get(self.num_adapters_per_location, [])
            
        if self.add_adapter_location_C:
            selection_map = {3: [6, 7, 8], 2: [6, 8], 1: [8]}
            down_block_ids += selection_map.get(self.num_adapters_per_location, [])

        if self.add_adapter_location_D:
            selection_map = {3: [9, 10, 11], 2: [9, 11], 1: [11]}
            down_block_ids += selection_map.get(self.num_adapters_per_location, [])

        
        ### collect down block res samples ###
        adapted_down_block_res_samples = []
        curr_idx = 0
        for i in range(12):
            if i in down_block_ids:
                adapted_down_block_res_samples.append(
                    self.down_blocks_adapter[curr_idx](
                        down_block_res_samples[i], 
                        sparsity_masking=sparsity_masking, 
                        num_frames=num_frames, 
                        timestep=timestep, 
                        encoder_hidden_states=encoder_hidden_states
                        )
                    )
                curr_idx += 1
            else:
                adapted_down_block_res_samples.append(torch.zeros_like(down_block_res_samples[i]))


        ### collect mid block res sample ###
        if mid_block_res_sample is not None and self.mid_block_adapter is not None:
            adapted_mid_block_res_sample = self.mid_block_adapter(
                mid_block_res_sample, 
                sparsity_masking=sparsity_masking, 
                num_frames=num_frames, 
                timestep=timestep, 
                encoder_hidden_states=encoder_hidden_states
                )
        else:
            adapted_mid_block_res_sample = None
            

        return adapted_down_block_res_samples, adapted_mid_block_res_sample



class EqualWeights(ModelMixin, ConfigMixin,):
    
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self, num_experts = 2):
        super().__init__()
        
        self.num_experts = num_experts
        self.holder = torch.tensor([1], requires_grad=False)

    def forward(self, inputs=None):
        logits = torch.zeros([self.num_experts]).unsqueeze(0).cuda()
        return logits
    
    
    
class ControlNetRouter(ModelMixin, ConfigMixin,):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_experts = 2,
        backbone_model_name = None,
        router_type = 'simple_weights',
        embedding_dim = None,
        num_routers = 12,
        add_mid_block_router = True,
    ):
        super().__init__()
        
        self.num_experts = num_experts 
        self.num_routers = num_routers
        self.router_type = router_type 
        self.embedding_dim = embedding_dim
        self.backbone_model_name = backbone_model_name
        self.add_mid_block_router = add_mid_block_router


        ##### down blocks #####
        self.down_blocks_router = nn.ModuleList([])
        for i in range(num_routers): # only process the first 9 down blocks
            if self.router_type == 'equal_weights':
                self.down_blocks_router.append(EqualWeights(self.num_experts))

            
        ##### mid block #####
        if self.add_mid_block_router:
            if self.router_type == 'equal_weights':
                self.mid_block_router = EqualWeights(self.num_experts)    

          
    def forward(self, router_input=None, sparse_mask=None, fixed_weights = None):
        
        if self.router_type == 'equal_weights':
            #assert router_input == None, "router_input should be None for learnable_weights "
            down_block_logits = [self.down_blocks_router[i]() for i in range(self.num_routers)]
            mid_block_logits = self.mid_block_router() if self.backbone_model_name in self.add_mid_block_router else None


        if sparse_mask is not None:
            for i in range(len(sparse_mask)):
                if mid_block_logits is not None:
                    if sparse_mask[i] == 0:
                        mid_block_logits[0, i] -= 1e6 
                if sparse_mask[i] == 0:
                    for j in range(len(down_block_logits)):
                            down_block_logits[j][0, i] -= 1e6
                
            
        down_block_weights = F.softmax(torch.concat(down_block_logits), dim = -1) 
        mid_block_weights = F.softmax(mid_block_logits, dim = -1) if mid_block_logits is not None else None

        if mid_block_weights.dim() == 2:
            mid_block_weights = mid_block_weights.squeeze(0)
        
        return down_block_weights, mid_block_weights



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
    def add_depth_estimator(self):
        from transformers import pipeline
        self.depth_estimator = pipeline('depth-estimation')


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
                conditioning_images_pil = []
                for i in range(len(batch_images_pils)):
                    image = self.depth_estimator(batch_images_pils[i])['depth']
                    image = np.array(image)
                    image = image[:, :, None]
                    image = np.concatenate([image, image, image], axis=2)
                    conditioning_images_pil.append(Image.fromarray(image))
                    
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
        
