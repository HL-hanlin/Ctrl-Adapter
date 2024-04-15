import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional

from diffusers.models.embeddings import TimestepEmbedding, Timesteps


class AdapterSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int, 
        num_layers: int = 1,

        ### choose which modules to activate ###
        add_spatial_resnet: bool = True, 
        add_temporal_resnet: bool = True, 
        add_spatial_transformer: bool = True,
        add_temporal_transformer: bool = True,
                
        ### resnet arguments ###
        eps: float = 1e-6,
        temporal_eps: float = None,
        merge_factor: float = 0.5,
        merge_strategy="learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
        #up: bool = False,
        up_sampling_scale: float = 1.0,
        
        
        ### transformer arguments ###
        cross_attention_dim: int = 1024,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
    ):
        super().__init__()
        
        temb_channels = in_channels 
        
        self.num_attention_heads = in_channels // attention_head_dim 
        self.num_layers = num_layers
        self.up_sampling_scale = up_sampling_scale

        self.add_spatial_resnet = add_spatial_resnet 
        self.add_temporal_resnet = add_temporal_resnet 
        self.add_spatial_transformer = add_spatial_transformer 
        self.add_temporal_transformer = add_temporal_transformer 
        
        self.add_resnet_time_mixer = self.add_spatial_resnet and self.add_temporal_resnet
        self.add_transformer_time_mixer = self.add_spatial_transformer and self.add_temporal_transformer
        
        
        if self.add_spatial_resnet or self.add_temporal_resnet:
            self.resnet_time_proj = Timesteps(out_channels, True, downscale_freq_shift=0)
            self.resnet_time_embedding = TimestepEmbedding(in_channels, in_channels)
            
            
        if self.add_spatial_transformer or self.add_temporal_transformer:
            self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
            self.inner_dim = num_attention_heads * attention_head_dim
            
            if self.add_temporal_transformer:
                self.transformer_time_embedding = TimestepEmbedding(in_channels, self.inner_dim)
                self.transformer_time_proj = Timesteps(in_channels, True, 0)
            
            self.proj_in = nn.Linear(in_channels, self.inner_dim)
            self.proj_out = nn.Linear(self.inner_dim, in_channels)

            
        spatial_resnets, temporal_resnets = [], []
        spatial_attentions, temporal_attentions = [], []
        resnets_time_mixer, transformers_time_mixer = [], []


        for i in range(self.num_layers):
            
            ### spatial resnet ### 
            if self.add_spatial_resnet:
                #from diffusers.models.resnet import ResnetBlock2D
                from model.resnet_block_2d import ResnetBlock2D
                spatial_resnets.append(
                    ResnetBlock2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=eps,
                        use_in_shortcut = True,
                        up = True if i==0 and self.up_sampling_scale > 1 else False, # only the 1st layer in SDXL-based backbone needs upscale 
                    )
                )

            ### temporal resnet ### 
            if self.add_temporal_resnet:
                from diffusers.models.resnet import TemporalResnetBlock
                temporal_resnets.append(
                    TemporalResnetBlock(
                        in_channels=out_channels if self.add_spatial_resnet else in_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=temporal_eps if temporal_eps is not None else eps,
                    )
                )                
                
            ### spatial transformer ###
            if self.add_spatial_transformer:
                from diffusers.models.transformers.transformer_temporal import BasicTransformerBlock
                spatial_attentions.append(        
                    BasicTransformerBlock(
                        self.inner_dim,
                        self.num_attention_heads,
                        attention_head_dim,
                        cross_attention_dim=cross_attention_dim,
                    )
                )
            
            ### temporal transformer ###
            if self.add_temporal_transformer:
                from diffusers.models.transformers.transformer_temporal import TemporalBasicTransformerBlock
                time_mix_inner_dim = self.inner_dim
                temporal_attentions.append(
                    TemporalBasicTransformerBlock(
                        self.inner_dim,
                        time_mix_inner_dim,
                        self.num_attention_heads,
                        attention_head_dim,
                        cross_attention_dim=cross_attention_dim,
                    )
                )

            ### add time_mixer layer if temporal resnet exists ###
            if self.add_resnet_time_mixer:
                from diffusers.models.resnet import AlphaBlender
                resnets_time_mixer.append(
                    AlphaBlender(
                        alpha=merge_factor,
                        merge_strategy=merge_strategy,
                        switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
                    )
                )

            ### add time_mixer layer if temporal transformer exists ###
            if self.add_transformer_time_mixer:
                from diffusers.models.resnet import AlphaBlender
                transformers_time_mixer.append(
                    AlphaBlender(
                        alpha=merge_factor,
                        merge_strategy=merge_strategy,
                        switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
                    )
                )
                
                
        if self.add_spatial_resnet:
            self.spatial_resnets = nn.ModuleList(spatial_resnets)
            
        if self.add_temporal_resnet:
            self.temporal_resnets = nn.ModuleList(temporal_resnets)
            
        if self.add_spatial_transformer:
            self.spatial_attentions = nn.ModuleList(spatial_attentions)
            
        if self.add_temporal_transformer:
            self.temporal_attentions = nn.ModuleList(temporal_attentions)  
            
        if self.add_resnet_time_mixer:
            self.resnets_time_mixer = nn.ModuleList(resnets_time_mixer)
            
        if self.add_transformer_time_mixer:
            self.transformers_time_mixer = nn.ModuleList(transformers_time_mixer)
            
            

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        num_frames: int,
        timestep: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        sparsity_masking = None,
    ) -> torch.FloatTensor:
        
    
        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        
        ### 0. process timestep ###
        if type(timestep) == int or type(timestep) == float:
            timestep = torch.Tensor([timestep], device=hidden_states.device).repeat_interleave(batch_frames, dim=0)
        elif type(timestep) == torch.Tensor and timestep.dim() == 0:
            timestep = torch.Tensor([timestep]).to(device=hidden_states.device).repeat_interleave(batch_frames, dim=0)
        elif type(timestep) == torch.Tensor and timestep.dim() == 1 and len(timestep) == 1:
            timestep = torch.Tensor(timestep).to(device=hidden_states.device).repeat_interleave(batch_frames, dim=0)
        elif type(timestep) == torch.Tensor and timestep.dim() == 2:
            timestep = timestep.squeeze()
        timestep = timestep.to(hidden_states.dtype) # bf 
            
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=hidden_states.dtype, device=hidden_states.device)


        for i in range(self.num_layers):
            
            ### 1.0 prepare resnet blocks ###
            if self.add_spatial_resnet or self.add_temporal_resnet:
                resnet_temb = self.resnet_time_proj(timestep) # bf, 320
                resnet_temb = self.resnet_time_embedding(resnet_temb)  # bf, 1280
                resnet_temb = resnet_temb.to(hidden_states.dtype)

            
            ### 1.1 spatial resnet ###
            if self.add_spatial_resnet:
                _, _, height, width = hidden_states.shape
                output_size = (int(height * self.up_sampling_scale), int(width * self.up_sampling_scale)) if i==0 else None
                hidden_states = self.spatial_resnets[i](hidden_states, resnet_temb, output_size=output_size) # bf c h w -> bf c h w
                _, _, height, width = hidden_states.shape # update height and width again, in case upsampled in spatial resnet
                if self.add_resnet_time_mixer:
                    hidden_states_mix = (hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)) # bf c h w -> b c f h w

            
            ### 1.2 temporal resnet ###
            if self.add_temporal_resnet:
                hidden_states = (hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)) # bf c h w -> b c f h w
                resnet_temb = resnet_temb.reshape(batch_size, num_frames, -1) # bf c -> b f c
                hidden_states = self.temporal_resnets[i](hidden_states, resnet_temb) # b c f h w -> b c f h w
                
                if self.add_resnet_time_mixer:
                    hidden_states = self.resnets_time_mixer[i](x_spatial=hidden_states_mix, x_temporal=hidden_states, image_only_indicator=image_only_indicator,) # -> b c f h w
                
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width) # b c f h w -> bf c h w
                
            
            ### 2.0 prepare transformer blocks ###
            if not self.add_spatial_resnet and not self.add_temporal_resnet and i==0 and self.up_sampling_scale > 1: # for sdxl only 
                hidden_states = F.interpolate(hidden_states, scale_factor=self.up_sampling_scale, mode="nearest")
                _, _, height, width = hidden_states.shape # update height and width again
                
            if self.add_spatial_transformer or self.add_temporal_transformer:
                if encoder_hidden_states.dim() == 2:
                    encoder_hidden_states = encoder_hidden_states.unsqueeze(1)

                if encoder_hidden_states.shape[0] == 1:
                    encoder_hidden_states = encoder_hidden_states.repeat_interleave(batch_frames, dim = 0)
                
                if self.add_temporal_transformer:
                    time_context = encoder_hidden_states # bf 1 c 
                    time_context_first_timestep = time_context[None, :].reshape(batch_size, num_frames, -1, time_context.shape[-1])[:, 0] # 1 1 c
                    time_context = time_context_first_timestep[None, :].broadcast_to(height * width, batch_size, 1, time_context.shape[-1]) # hw 1 1 c
                    time_context = time_context.reshape(height * width * batch_size, 1, time_context.shape[-1]) # hw 1 c
            
                residual = hidden_states # bf c h w 
        
                hidden_states = self.norm(hidden_states) # bf c h w
                inner_dim = hidden_states.shape[1] # c
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim) # bf hw c
                hidden_states = self.proj_in(hidden_states) # bf hw c'

                if self.add_temporal_transformer:
                    num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
                    num_frames_emb = num_frames_emb.repeat(batch_size, 1)
                    num_frames_emb = num_frames_emb.reshape(-1)
                    transformer_t_emb = self.transformer_time_proj(num_frames_emb) #  bf c
                    transformer_t_emb = transformer_t_emb.to(dtype=hidden_states.dtype)
                    emb = self.transformer_time_embedding(transformer_t_emb) # bf c'
                    emb = emb[:, None, :] # bf 1 c'
    

            ### 2.1 spatial transformer ###
            if self.add_spatial_transformer:
                hidden_states = self.spatial_attentions[i](hidden_states, encoder_hidden_states=encoder_hidden_states,) # bf hw c'
                if self.add_transformer_time_mixer:
                    hidden_states_mix = hidden_states
                
                
            
            ### 2.2 temporal transformer ###
            if self.add_temporal_transformer:
                hidden_states = hidden_states + emb # bf hw c'
                hidden_states = self.temporal_attentions[i](hidden_states, num_frames=num_frames, encoder_hidden_states=time_context,) # bf hw c'
                if self.add_transformer_time_mixer:           
                    hidden_states = self.transformers_time_mixer[i](x_spatial=hidden_states_mix, x_temporal=hidden_states, image_only_indicator=image_only_indicator,) # bf hw c'
                        
            
            ### 3. Output ###
            if self.add_spatial_transformer or self.add_temporal_transformer:
                hidden_states = self.proj_out(hidden_states) # bf hw c
                hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous() # bf c h w    
                hidden_states = hidden_states + residual # bf c h w
            
            
        return hidden_states



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
        
    