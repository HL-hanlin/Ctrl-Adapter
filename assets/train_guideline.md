

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

#### 4.1 Controllable Image/Video Generation

Here is the command we used to start training on SDXL with depth map as control condition. Training scripts on I2VGen-XL and SVD are roughly the same.

```
sh train_scripts/sdxl/sdxl_train_depth.sh
```

Specifically, in the training scripts:

`--yaml_file`: The configuration file for all hyper-parameters related to **training**.

The rest of the hyper-parameters in the training script are for **evaluation**, which can help you monitor the training process better.

`--save_n_steps`: Save the trained adapter checkpoints every n training steps.

`--save_starting_step`: Save the trained adapter checkpoints after such training steps.

`--validate_every_steps`: Perform evaluation every x training steps. The evaluation data are placed under ```./assets/evaluation```. If you prefer to evaluate different samples, you can replace them by following the same file structure.

`--num_inference_steps`: The number of inference steps during inference. We can just set it as the same value as the default inference steps of the backbone model.

```--extract_control_conditions```: If you already have condition image/frames extracted from evaluation image/video (see Inference Data Structure section above), you can set it as ```False```. Otherwise, if you haven't extracted control conditions and only have the raw image/frames, you can set it as ```True```, and our code can automatically extract the control conditions from the evaluation image/frames. The default setting is ```False```.

```--control_guidance_end```: As mentioned above, this is the most important parameter that balances generated image/video quality with control strength. But since we want to see if the training code working or not, we recommend just setting it as 1.0 to give control across all inference steps. You can adjust it to a lower value later after you have a trained model.


#### 4.2 Multi-Condition Control 

Here is the command we used to do multi-condition control training on I2VGen-XL. 

```
sh train_scripts/i2vgenxl/i2vgenxl_train_multi_condition.sh
```

Please note that we currently only support I2VGen-XL for multi-condition control. If you are interested in trying it with other backbones, you can modify the code accordingly. 

In the training configuration file, here are the hyper-parameters specific to multi-condition control: 

`--control_types`: You can put the list of control conditions you want here, such as ```[depth, canny, normal, segmentation]```.

```--router_type```: We currently support equal weight and linear weight in our codebase. The definition of different router types are illustrated in our paper. 

```--multi_source_random_select_control_types``` and ```--max_num_multi_source_train```: Since we need to load the ControlNet for each control type in the above `--control_types` list, the training code will go out-of-memory if there are too many control conditions. Therefore, we add a binary hyper-parameter `--multi_source_random_select_control_types` to randomly select k control conditions within the range `[1, --max_num_multi_source_train]` in each training step. If the training script can run without out-of-memory on your GPUs, you can just set `--multi_source_random_select_control_types` as `False`.


### Step 5: Train Ctrl-Adapter on a New Backbone Model (Optional)

To train Ctrl-Adapter on a new backbone model, basically here are several steps you need to take:

- Create a new folder (similar as i2vgenxl) and put the unet and pipeline files under this repo.

- Modify ```train.py```, ```inference.py```, ```./utils/data_loader.py``` (and sometimes ```./model/ctrl_adapter.py```). We already highlighted the code in these files where you need to pay attention with ```###```. You can modify/add the code following the instructions. 

- Create new training scripts, inference scripts, and configuration files. 


🚩 Now you are ready to start training!! 😆


