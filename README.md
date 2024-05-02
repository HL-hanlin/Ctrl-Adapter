# Ctrl-Adapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model

Official implementation of **Ctrl-Adapter**, an efficient and versatile framework that adds diverse controls
to any image/video diffusion models by adapting pretrained ControlNets.


[![arXiv](https://img.shields.io/badge/ArXiv-2404.09967-orange)](https://arxiv.org/abs/2404.09967) 
[![projectpage](https://img.shields.io/badge/Project-Page-green)](https://ctrl-adapter.github.io/)
[![checkpoints](https://img.shields.io/badge/Model-Checkpoints-blue)](https://huggingface.co/hanlincs/ctrl-adapter)




[Han Lin](https://hl-hanlin.github.io/),
[Jaemin Cho](https://j-min.io),
[Abhay Zala](https://aszala.com/),
[Mohit Bansal](https://www.cs.unc.edu/~mbansal/)




<br>
<img width="800" src="assets/teaser_update.gif"/>
<br>


CTRL-Adapter is an efficient and versatile framework for adding diverse
spatial controls to any image or video diffusion model. It supports a variety of useful
applications, including video control, video control with multiple conditions, video control with
sparse frame conditions, image control, zero-shot transfer to unseen conditions, and video editing.

# 🔥 News
- **Apr. 30, 2024**. Training code released now! It's time to train Ctrl-Adapter on your desired backbone! 🚀🚀
- **Apr. 29, 2024**. **SDXL**, **I2VGen-XL**, and **SVD** inference code and checkpoints are all released!


# 🔧 Setup

### Environment Setup

If you only need to perform inference with our code, please install from ```requirements_inference.txt```. To make our codebase easy to use, the primary libraries that need to be installed are Torch, Diffusers, and Transformers. Specific versions of these libraries are not required; the default versions should work fine :)

If you are planning to conduct training, please install from ```requirements_train.txt``` instead, which contains more dependent libraries needed.


```shell
conda create -n ctrl-adapter python==3.10
conda activate ctrl-adapter
pip install -r requirements_inference.txt # install from this if you only need to perform inference
pip install -r requirements_train.txt # install from this if you plan to do some training
```


Here we list several questions that we believe important when you start using this 

# 🔮 Inference

We provde model checkpoints and inference scripts for Ctrl-Adapter trained on SDXL, I2VGen-XL, and SVD. 
All inference scripts are put under ```./inference_scripts```.

### 📌 Notice Before You Begin

Please note that there is usually no single model that excels at generating images/videos for all motion styles across various control conditions.

Different image/video generation backbones may perform better with specific types of motion. For instance, we have observed that SVD excels at slide motions, while it generally performs worse than I2VGen-XL with complex motions (this is consistent wtih the findings in [DynamiCrafter](https://www.youtube.com/watch?v=0NfmIsNAg-g)). Additionally, using different control conditions can lead to significantly different results in the generated images/videos, and some control conditions may be more informative than others for certain types of motion.



### 📌 Inference Data Structure

We put some sample images/frames for inference under the folder ```./assets/evaluation```. You can add your custom examples following the same file structure illustrated below. 

For model inference, we support two options:
- If you already have condition image/frames extracted from some image/video, you can use inference (w/ extracted condition).

```bash
./assets/evaluation/images
    ├── depth
    │   ├── anime_corgi.png
    ├── raw_input
    │   ├── anime_corgi.png
    ├── captions.json

./assets/evaluation/frames
    ├── depth
    │   ├── newspaper_cat
    │   │   ├── 00000.png
    │   │   ├── 00001.png
    │   │   ...
    │   │   ├── 00015.png
    ├── raw_input
    │   ├── newspaper_cat
    │   │   ├── 00000.png # only the 1st frame is needed for I2V models
    ├── captions.json
```


- If you haven't extracted control conditions and only have the raw image/frames, you can use inference (w/o extracted condition). In this way, our code can automatically extract the control conditions from the input image/frames and then generate corresponding image/video.

```bash
./assets/evaluation/images
    ├── raw_input
    │   ├── anime_corgi.png
    ├── captions.json

./assets/evaluation/frames
    ├── raw_input
    │   ├── newspaper_cat
    │   │   ├── 00000.png
    │   │   ├── 00001.png
    │   │   ...
    │   │   ├── 00015.png
    ├── captions.json
```


### 📌 Run Inference Scripts

Here is a sample command to run inference on SDXL with depth map as control (w/ extracted condition).

```
sh inference_scripts/sdxl/sdxl_inference_depth.sh
```

⚠️  ```--control_guidance_end```: this is the most important parameter that balances generated image/video quality with control strength. If you notice the generated image/video does not follow the spatial control well, you can increase this value; and if you notice the generated image/video quality is not good because the spatial control is too strong, you can decrease this value. Detailed discussion of control strength via this parameter is shown in our paper.

We list the inference scripts for different tasks mentioned in our paper as follows ⬇️

## Controllable Image Generation 

<br>
<img width="800" src="assets/concatenated_image.jpg"/>
<br>



### SDXL


| Control Conditions    | Checkpoints | Inference (w/ extracted condition) | Inference (w/o extracted condition) |
|-----------|------|---------|--------|
| Depth Map      | <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/sdxl_depth">HF link</a> | [command](/inference_scripts/sdxl/sdxl_inference_depth.sh)    | [command](/inference_scripts/sdxl/sdxl_inference_extract_depth_from_raw_image.sh)   |
| Canny Edge     | <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/sdxl_canny">HF link</a> | [command](/inference_scripts/sdxl/sdxl_inference_canny.sh)    | [command](/inference_scripts/sdxl/sdxl_inference_extract_canny_from_raw_images.sh)   |
| Soft Edge     | <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/sdxl_softedge">HF link</a> | [command](/inference_scripts/sdxl/sdxl_inference_softedge.sh)    | [command](/inference_scripts/sdxl/sdxl_inference_extract_softedge_from_raw_images.sh)   |
| Normal Map     | <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/sdxl_normal">HF link</a> | [command](/inference_scripts/sdxl/sdxl_inference_normal.sh)    | [command](/inference_scripts/sdxl/sdxl_inference_extract_normal_from_raw_images.sh)   |
| Segmentation     | <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/sdxl_segmentation">HF link</a> | [command](/inference_scripts/sdxl/sdxl_inference_segmentation.sh)    | [command](/inference_scripts/sdxl/sdxl_inference_extract_segmentation_from_raw_images.sh)   |
| Scribble     | <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/sdxl_scribble">HF link</a> | [command](/inference_scripts/sdxl/sdxl_inference_scribble.sh)    | [command](/inference_scripts/sdxl/sdxl_inference_extract_scribble_from_raw_images.sh)   |
| Lineart     | <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/sdxl_lineart">HF link</a> | [command](/inference_scripts/sdxl/sdxl_inference_lineart.sh)    | [command](/inference_scripts/sdxl/sdxl_inference_extract_lineart_from_raw_images.sh)   |



## Controllable Video Generation

<br>
<div>
    <img width="400" src="assets/girl_and_dog.gif" style="margin-right: 10px;"/>
    <img width="400" src="assets/newspaper_cat.gif"/>
</div>
<br>


### I2VGen-XL

| Control Conditions    | Checkpoints | Inference (w/ extracted condition) | Inference (w/o extracted condition) |
|-----------|------|---------|--------|
| Depth Map      | <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/i2vgenxl_depth">HF link</a>    | [command](/inference_scripts/i2vgenxl/i2vgenxl_inference_depth.sh)       | [command](/inference_scripts/i2vgenxl/i2vgenxl_inference_extract_depth_from_raw_frames.sh)   |
| Canny Edge     | <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/i2vgenxl_canny">HF link</a>    | [command](/inference_scripts/i2vgenxl/i2vgenxl_inference_canny.sh)       | [command](/inference_scripts/i2vgenxl/i2vgenxl_inference_extract_canny_from_raw_frames.sh)   |
| Soft Edge      | <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/i2vgenxl_softedge">HF link</a> | [command](/inference_scripts/i2vgenxl/i2vgenxl_inference_softedge.sh)    | [command](/inference_scripts/i2vgenxl/i2vgenxl_inference_extract_softedge_from_raw_frames.sh)   |



### SVD

| Control Conditions    | Checkpoints | Inference (w/ extracted condition) | Inference (w/o extracted condition) |
|-----------|------|---------|--------|
| Depth Map      | <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/svd_depth">HF link</a>    | [command](/inference_scripts/svd/svd_inference_depth.sh)       | [command](/inference_scripts/svd/svd_inference_extract_depth_from_raw_frames.sh)   |
| Canny Edge     | <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/svd_canny">HF link</a>    | [command](/inference_scripts/svd/svd_inference_canny.sh)       | [command](/inference_scripts/svd/svd_inference_extract_canny_from_raw_frames.sh)   |
| Soft Edge      | <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/svd_softedge">HF link</a> | [command](/inference_scripts/svd/svd_inference_softedge.sh)    | [command](/inference_scripts/svd/svd_inference_extract_softedge_from_raw_frames.sh)   |



## Video Generation with Multi-Condition Control

<br>
<img width="800" src="assets/boy_and_mom.gif"/>
<br>

We currently implemented multi-condition control on **I2VGen-XL**. The following checkpoint are trained on 7 control conditions, including depth, canny, normal, softedge, segmentation, lineart, and openpose. Here are the sample inference scripts that uses depth, canny, segmentation, and openpose as control conditions.

| Adapter Checkpoint    | Router Checkpoint | Inference (w/ extracted condition) | Inference (w/o extracted condition) |
|-----------|------|---------|--------|
| <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/i2vgenxl_multi_control_adapter">HF link</a>      | <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/i2vgenxl_multi_control_router">HF link</a>    | [command](/inference_scripts/i2vgenxl/i2vgenxl_inference_depth_canny_segmentation_openpose.sh)       | [command](/inference_scripts/i2vgenxl/i2vgenxl_inference_extract_depth_canny_segmentation_openpose_from_raw_frames.sh)   |



## Video Generation with Sparse Control

<br>
<img width="400" src="assets/museum.gif"/>
<br>

Here we provide a sample inference script that uses user scribbles as condition, and 4 out of 16 frames for sparse control.

| Control Conditions    | Checkpoint | Inference (w/ extracted condition) |
|-----------|------|---------|
| Scribbles      | <a href="https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main/i2vgenxl_scribble_sparse">HF link</a>    | [command](/inference_scripts/i2vgenxl/i2vgenxl_inference_scribble_sparse.sh)       |



# 🚅 How To Train 

### Step 1: Download Training Data

- For Ctrl-Adapter training on image backbones (e.g., SDXL), we use 300k images from the [LAION POP](https://laion.ai/blog/laion-pop/) dataset. You can download a subset from this dataset [here](https://huggingface.co/datasets/Ejafa/ye-pop).

- For Ctrl-Adapter training on video backbones (e.g., I2VGen-XL, SVD), we download a subset of videos (around 1.5M) from the 10M training set of the [Panda70M](https://snap-research.github.io/Panda-70M/) dataset. You can follow their [instructions](https://github.com/snap-research/Panda-70M/tree/main/dataset_dataloading) to download the dataset. To ensure the videos contain enough movement, we further filter out the videos with [optical flow score](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html) lower than a threshold of 0.3. 

⚠️ The dataset we used here might not be optimal for all control conditions. We recommend that users train our Ctrl-Adapter on the dataset that best suits their use cases.

### Step 2: Prepare Data in Specified Format

We provides some sample training images/videos under folder ```./sample_data```.

- For image dataset, you can put the png/jpg raw images under the folder ```./sample_data/images```, and create a csv file with similar rows/columns format as ```./sample_data/image_captions.csv```. 

- For video dataset, you can put the mp4 raw videos under the folder ```./sample_data/videos```, and create a csv file with similar rows/columns format as ```./sample_data/video_captions.csv```. 

If you want to use different input data format, you can modify ```./utils/data_loader.py``` to suit your need.

If your data is stored in different paths, you can change the ```train_data_path``` and ```train_prompt_path``` in the training configuration files listed under ```./configs```.

In addition, please set the ```DATA_PATH``` in the training configuration files to the path where you want all training checkpoints to be stored.

### Step 3: Control Conditions Extractors

To simplify the start-up process, our codebase automatically performs all control condition extractions during training. This eliminates the need for pre-processing the control images from input images/videos! (Note that this may slightly reduce training speed, but it is generally worthwhile since the Ctrl-Adapter converges quite rapidly.)

Most of our control condition extractors are directly utilized from the transformers or controlnet_aux libraries. You can check ```./model/ctrl_helper.py``` to see the extractors used during training. If you wish to use different condition extractors, you can modify the python script accordingly.

⚠️ One major changes we made is for the depth estimator, we found that the default depth estimator from the transformers library is relatively slow. Therefore, we recommend using ```dpt_swin2_large_384``` from the [MiDaS](https://github.com/isl-org/MiDaS) library for depth estimation. We have already added code in the ```./utils``` folder to utilize this depth estimator. All you need to do is to download the checkpoint ```dpt_swin2_large_384```, and place it under the path ```{DATA_PATH}/ckpts/DepthMidas/dpt_swin2_large_384.pt```.


### Step 4: Set Up Training Scripts

All training configuration files and training scripts are placed under ```./configs``` and ```train_scripts``` respectively. 

Here is the command we used to start training on SDXL with depth map as control condition.

```
sh train_scripts/sdxl/sdxl_train_depth.sh
```

Specifically, in the training scripts, we added hyper-parameters that controls how many training steps we do evaluation. In this way, you can monitor the training process better.

`--save_n_steps`: Save the trained adapter checkpoints every n training steps.

`--validate_every_steps`: Perform evaluation every x training steps. The evaluation data are placed under ```./assets/evaluation```. If you prefer to evaluate different samples, you can replace them by following the same file structure.

```--extract_control_conditions```: If you already have condition image/frames extracted from evaluation image/video (see Inference Data Structure section above), you can set it as ```False```. Otherwise, if you haven't extracted control conditions and only have the raw image/frames, you can set it as ```True```, and our code can automatically extract the control conditions from the evaluation image/frames. The default setting is ```False```.


### Step 5: Train Ctrl-Adapter on a New Backbone Model (Optional)

To train Ctrl-Adapter on a new backbone model, basically here are several steps you need to take:

- Create a new folder (similar as i2vgenxl) and put the unet and pipeline files under this repo.

- Modify ```train.py```, ```inference.py```, ```./utils/data_loader.py``` (and sometimes ```./model/ctrl_adapter.py```). We already highlighted the code in these files where you need to pay attention with ```###```. You can modify/add the code following the instructions. 

- Create new training scripts, inference scripts, and configuration files. 


🚩 Now you are ready to start training!! 😆



# 📝 TODO List
- [x] Release environment setup, inference code, and model checkpoints.
- [x] Release training code.
- [x] Training guideline to adapt our Ctrl-Adapter to new image/video diffusion models.
- [ ] Ctrl-Adapter + DiT-based image/video generation backbones. (**WIP**)
- [ ] Release evaluation code.

💗 Please let us know in the issues or PRs if you're interested in any relevant backbones or down-stream tasks that can be implemented by our Ctrl-Adapter framework! Welcome to collaborate and contribute!

# 📚 BibTeX

🌟 If you find our project useful in your research or application development, citing our paper would be the best support for us! 

```
@misc{lin2024ctrladapter,
      title={Ctrl-Adapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model}, 
      author={Han Lin and Jaemin Cho and Abhay Zala and Mohit Bansal},
      year={2024},
      eprint={2404.09967},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# 🙏 Acknowledgements
The development of Ctrl-Adapter has been greatly inspired by the following amazing works and teams:

- [ControlNet](https://huggingface.co/lllyasviel/ControlNet)
- [SDXL](https://stability.ai/stable-diffusion)
- [I2VGen-XL](https://i2vgen-xl.github.io/)
- [Hotshot-XL](https://github.com/hotshotco/Hotshot-XL)
- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models)

We hope that releasing this model/codebase helps the community to continue pushing these creative tools forward in an open and responsible way.
