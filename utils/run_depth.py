"""
Compute depth maps for images in the input folder.
the code is adapted from midas library: https://github.com/isl-org/MiDaS/tree/master
"""
import os
import sys
svd_path = os.path.join(os.getcwd(), 'utils/MiDaS')
if svd_path not in sys.path:
    sys.path.append(svd_path)
    
import glob
import torch
import utils
import cv2
import argparse
import time

import numpy as np
from PIL import Image

from imutils.video import VideoStream
from midas.model_loader import default_models, load_model
import midas 



first_execution = True
def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def create_side_by_side(image, depth, grayscale):
    """
    Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
    for better visibility.

    Args:
        image: the RGB image
        depth: the depth map
        grayscale: use a grayscale colormap?

    Returns:
        the image and depth map place side by side
    """
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)


def load_midas_model(model_path, device=None, model_type='dpt_swin2_large_384', optimize=False, height=None, square=False):
    # select device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)
    return model, transform, net_w, net_h



class DepthMidas():
    def __init__(self, model_path='checkpoints/dpt_swin2_large_384.pt', device='cuda', 
                       model_type='dpt_swin2_large_384', optimize=False, height=None, square=False):
        
        self.device = device 
        self.model_path = model_path 
        self.model_type = model_type 
        self.optimize = optimize 
        self.height = height 
        self.square = square 
        self.model, self.transform, self.net_w, self.net_h = midas.model_loader.load_model(device, model_path, model_type, optimize, height, square)
     
    def estimate(self, images_pil, new_size=(512, 512)): 
        depth_pil = []
        for i in range(len(images_pil)):
            rgb_image = images_pil[i].convert('RGB').resize(new_size)
            numpy_image = np.array(rgb_image) / 255.0
            image = self.transform({"image": numpy_image})["image"]
            with torch.no_grad():
                prediction = process(self.device, self.model, self.model_type, image, (self.net_w, self.net_h), numpy_image.shape[1::-1],
                                     self.optimize, False)
                content = create_side_by_side(None, prediction, grayscale=True)
                depth_pil.append(Image.fromarray(content.astype(np.uint8), 'RGB'))
        return depth_pil
        

