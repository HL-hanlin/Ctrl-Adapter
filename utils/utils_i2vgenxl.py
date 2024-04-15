import torch 
from PIL import Image 
from typing import Any, Dict, List, Optional, Tuple, Union

from diffusers.image_processor import VaeImageProcessor

# The following utilities are taken and adapted from
# https://github.com/ali-vilab/i2vgen-xl/blob/main/utils/transforms.py.
def _convert_pt_to_pil(image: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(image, list) and isinstance(image[0], torch.Tensor):
        image = torch.cat(image, 0)

    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image_numpy = VaeImageProcessor.pt_to_numpy(image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_numpy)
        image = image_pil

    return image


def _resize_bilinear(
    image: Union[torch.Tensor, List[torch.Tensor], Image.Image, List[Image.Image]], resolution: Tuple[int, int]
):
    # First convert the images to PIL in case they are float tensors (only relevant for tests now).
    image = _convert_pt_to_pil(image)

    if isinstance(image, list):
        image = [u.resize(resolution, Image.BILINEAR) for u in image]
    else:
        image = image.resize(resolution, Image.BILINEAR)
    return image


