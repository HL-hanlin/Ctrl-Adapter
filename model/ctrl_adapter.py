import torch

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from model.adapter_spatial_temporal import AdapterSpatioTemporal

from controlnet.controlnet import zero_module 



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
        
        ### experimental
        num_repeats = 1, 
        out_channels = None,
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

        down_blocks_channels = self.get_down_block_channels()
        down_block_ids = self.get_down_block_ids()
        
        if backbone_model_name in ['sdxl']:
            down_blocks_up_scales = [2] * len(down_block_ids)
            mid_block_up_scale = 2         
        else:
            down_blocks_up_scales = [1] * len(down_block_ids)
            mid_block_up_scale = 1
            
            
        self.down_blocks_adapter = torch.nn.ModuleList([])
        
        self.num_adapters = len(down_blocks_channels) 
        self.adapter_type = adapter_type
        
        self.num_repeats = num_repeats
        

        ### down blocks ###
        if num_repeats > 1:
            self.zero_convs = torch.nn.ModuleList([]) 

        for r in range(num_repeats):
            for i in range(self.num_adapters): 
                config = {"in_channels": down_blocks_channels[i], 
                            "out_channels": down_blocks_channels[i],  
                            "cross_attention_dim": cross_attention_dim,
                            "num_layers": num_blocks,
                            "up_sampling_scale": down_blocks_up_scales[i],
                            "add_spatial_resnet": add_spatial_resnet, 
                            "add_temporal_resnet": add_temporal_resnet,
                            "add_spatial_transformer": add_spatial_transformer, 
                            "add_temporal_transformer": add_temporal_transformer,
                            }
                self.down_blocks_adapter.append(AdapterSpatioTemporal(**config))

                if num_repeats > 1:
                    self.zero_convs.append(
                        zero_module(
                            torch.nn.Conv2d(down_blocks_channels[i], out_channels, kernel_size=1)
                            )
                        )
             
        ### mid block ###
        if add_adapter_location_M:
            config = {"in_channels": mid_block_channels, 
                        "out_channels": mid_block_channels, 
                        "cross_attention_dim": cross_attention_dim,
                        "num_layers": num_blocks,
                        "up_sampling_scale": mid_block_up_scale,
                        "add_spatial_resnet": add_spatial_resnet, 
                        "add_temporal_resnet": add_temporal_resnet,
                        "add_spatial_transformer": add_spatial_transformer, 
                        "add_temporal_transformer": add_temporal_transformer,
                        }
            self.mid_block_adapter = AdapterSpatioTemporal(**config)
        else:
            self.mid_block_adapter = None
            
            
    def get_down_block_ids(self):
        
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
        
        return down_block_ids
    
    
    def get_down_block_channels(self):
        
        down_blocks_channels = []

        if self.add_adapter_location_A:
            down_blocks_channels = [320] * self.num_adapters_per_location

        if self.add_adapter_location_B:
            if self.num_adapters_per_location == 3:
                down_blocks_channels += [320, 640, 640]
            elif self.num_adapters_per_location == 2:
                down_blocks_channels += [320, 640]
            elif self.num_adapters_per_location == 1:
                down_blocks_channels += [640]

        if self.add_adapter_location_C:
            if self.num_adapters_per_location == 3:
                down_blocks_channels += [640, 1280, 1280]
            elif self.num_adapters_per_location == 2:
                down_blocks_channels += [640, 1280]
            elif self.num_adapters_per_location == 1:
                down_blocks_channels += [1280]

        if self.add_adapter_location_D:
            down_blocks_channels += [1280] * self.num_adapters_per_location
        
        return down_blocks_channels
    

    def forward(self, down_block_res_samples, mid_block_res_sample=None, sparsity_masking=None, 
                num_frames=None, timestep=None, encoder_hidden_states=None):

        down_block_ids = self.get_down_block_ids()
 
        ### collect down block res samples ###
        adapted_down_block_res_samples = []
        for r in range(self.num_repeats):
            curr_idx = 0
            for i in range(12):
                if i in down_block_ids:
                    adapted_down_block_res_samples.append(
                        self.down_blocks_adapter[curr_idx + r * len(down_block_ids)](
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
            
        
        if self.num_repeats > 1:
            zero_conv_idx = 0
            adapted_down_block_res_samples_aggregated = []
            for repeat_idx in range(self.num_repeats):
                adapted_down_block_res_samples_agg = 0
                curr_idx = 0
                for i in range(12):
                    if i in down_block_ids:
                        adapted_down_block_res_samples_agg = adapted_down_block_res_samples_agg + self.zero_convs[zero_conv_idx](adapted_down_block_res_samples[curr_idx + 12 * repeat_idx])
                        curr_idx += 1
                        zero_conv_idx += 1
                adapted_down_block_res_samples_aggregated.append(adapted_down_block_res_samples_agg)

            return adapted_down_block_res_samples_aggregated, None

        else:    
            return adapted_down_block_res_samples, adapted_mid_block_res_sample
        
            


if __name__ == "__main__":
    
    from utils.utils import count_params
    
    parameters_list = []    
    
    ### this is the input/output feature size and channel dimensions for I2VGen-XL ###
    # dims = [(64, 64), (64, 64), (64, 64), (32, 32), (32, 32), (32, 32), (16, 16), (16, 16), (16, 16), (8, 8), (8, 8), (8, 8)]
    # channels = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
    
    ### this is the input/output feature size and channel dimensions for SDXL ###
    dims = [(64, 64), (64, 64), (64, 64), (32, 32), (32, 32), (32, 32), (16, 16), (16, 16), (16, 16)]
    channels = [320, 320, 320, 320, 640, 640, 640, 1280, 1280]


    for dim, channel in zip(dims, channels):
            
        transformer_configs = {
            "in_channels": channel, 
            "out_channels": channel, 
            "cross_attention_dim": 2048,
            "num_layers": 1,
            "up": True,
            "add_spatial_resnet": True, 
            "add_temporal_resnet": False, 
            "add_spatial_transformer": True, 
            "add_temporal_transformer": False, 
            }
     
        adapter = AdapterSpatioTemporal(**transformer_configs)
        
        for name, para in adapter.named_parameters():
            parameters_list.append(para)
    
    
    count_params(parameters_list) # quick check of total trainable parameters
        
    
    