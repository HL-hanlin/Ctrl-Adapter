import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


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
    
    

class SimpleWeights(ModelMixin, ConfigMixin,):
    
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(self, num_experts = 2):
        super().__init__()
        
        self.num_experts = num_experts
        self.wg = nn.Linear(1, self.num_experts, bias=False)

    def forward(self, inputs=None):
        constant_tensor = torch.Tensor([1]).unsqueeze(0).cuda().to(self.wg.weight.dtype)
        logits = self.wg(constant_tensor) 
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
        use_sparsemax = False,
    ):
        super().__init__()
        
        self.num_experts = num_experts 
        self.num_routers = num_routers
        self.router_type = router_type 
        self.embedding_dim = embedding_dim
        self.backbone_model_name = backbone_model_name
        self.add_mid_block_router = add_mid_block_router
        self.use_sparsemax = use_sparsemax # this is not used in our current version

        ##### down blocks #####
        self.down_blocks_router = nn.ModuleList([])
        for i in range(num_routers): # only process the first 9 down blocks
            if self.router_type == 'equal_weights':
                self.down_blocks_router.append(EqualWeights(self.num_experts))
            elif self.router_type == 'simple_weights':
                self.down_blocks_router.append(SimpleWeights(self.num_experts))

        ##### mid block #####
        if self.add_mid_block_router:
            if self.router_type == 'equal_weights':
                self.mid_block_router = EqualWeights(self.num_experts)    
            elif self.router_type == 'simple_weights':
                self.mid_block_router = SimpleWeights(self.num_experts)
                
          
    def forward(self, router_input=None, sparse_mask=None, fixed_weights = None):
        
        if self.router_type == 'equal_weights':
            #assert router_input == None, "router_input should be None for learnable_weights "
            down_block_logits = [self.down_blocks_router[i]() for i in range(self.num_routers)]
            mid_block_logits = self.mid_block_router() 

        elif self.router_type == 'simple_weights':
            down_block_logits = [self.down_blocks_router[i]() for i in range(self.num_routers)]
            mid_block_logits = self.mid_block_router() 
            
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


