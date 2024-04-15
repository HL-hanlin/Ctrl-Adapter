import argparse 
from PIL import Image
from typing import List, Union
from io import BytesIO

import torch
from torchvision import transforms

from transformers import PretrainedConfig



def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")
        

def count_params(params):
    total_trainable_params_count = 0 
    for p in params:
        total_trainable_params_count += p.numel()
    print("total_trainable_params_count is: ", total_trainable_params_count)


def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


def enable_grads(model):
    for name, p in model.named_parameters():
        if p.requires_grad == False:
            #print(name, p.requires_grad)
            p.requires_grad = True


def print_trainable_grads(model):
    for name, p in model.named_parameters():
        if p.requires_grad == True:
            print(name, p.requires_grad)


def print_disabled_grads(model):
    for name, p in model.named_parameters():
        if p.requires_grad == False:
            print(name, p.requires_grad)


def images_to_gif_bytes(images: List, duration: int = 1000) -> bytes:
    with BytesIO() as output_buffer:
        # Save the first image
        images[0].save(output_buffer,
                       format='GIF',
                       save_all=True,
                       append_images=images[1:],
                       duration=duration,
                       loop=0)  # 0 means the GIF will loop indefinitely

        # Get the byte array from the buffer
        gif_bytes = output_buffer.getvalue()

    return gif_bytes


def save_as_gif(images: List, file_path: str, duration: int = 1000):
    with open(file_path, "wb") as f:
        f.write(images_to_gif_bytes(images, duration))


def save_concatenated_gif(single_image, output_gif_path, image_list_arrays, duration=500):

    # Ensure all lists are the same length
    if not all(len(lst) == len(image_list_arrays[0]) for lst in image_list_arrays):
        raise ValueError("All image lists must have the same number of elements")

    # Create a list to hold the concatenated frames
    concatenated_frames = []

    # Loop through each index in the list (assuming all lists have the same length)
    for index in range(len(image_list_arrays[0])):
        # Start with the single image
        images = [single_image] + [lst[index] for lst in image_list_arrays]

        # Calculate total width and max height
        total_width = sum(img.size[0] for img in images)
        max_height = max(img.size[1] for img in images)

        # Create a new image with the calculated dimensions
        new_image = Image.new('RGB', (total_width, max_height))

        # Paste each image into the new image
        current_x = 0
        for img in images:
            new_image.paste(img, (current_x, 0))
            current_x += img.size[0]
        
        concatenated_frames.append(new_image)

    # Save the final sequence of frames as a new GIF
    concatenated_frames[0].save(output_gif_path, save_all=True, append_images=concatenated_frames[1:], optimize=False, duration=duration, loop=0)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")
        
        
def center_crop_and_resize(img, output_size=(512, 512)):
    # Load the image
    #img = Image.open(image_path)
    
    # Calculate the aspect ratio of the output image
    aspect_ratio = output_size[0] / output_size[1]
    
    # Get the current size of the image
    original_width, original_height = img.size
    
    # Calculate the aspect ratio of the original image
    original_aspect_ratio = original_width / original_height
    
    # Determine the dimensions to which the image needs to be resized before cropping
    if original_aspect_ratio > aspect_ratio:
        # Image is wider than the desired aspect ratio; resize based on height
        new_height = output_size[1]
        new_width = int(new_height * original_aspect_ratio)
    else:
        # Image is taller than the desired aspect ratio; resize based on width
        new_width = output_size[0]
        new_height = int(new_width / original_aspect_ratio)
    
    # Resize the image
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Calculate the cropping box
    left = (new_width - output_size[0]) / 2
    top = (new_height - output_size[1]) / 2
    right = (new_width + output_size[0]) / 2
    bottom = (new_height + output_size[1]) / 2
    
    # Crop the center
    img_cropped = img_resized.crop((left, top, right, bottom))
    
    return img_cropped


image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def image_to_tensor(img):
    
    with torch.no_grad():
        if img.mode != "RGB":
            img = img.convert("RGB")

        image = image_transforms(img)#.to(accelerator.device)

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if image.shape[0] > 3:
            image = image[:3, :, :]

    return image
