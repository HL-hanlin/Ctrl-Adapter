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



# 🔧 Setup

### Environment Setup

To make our codebase easy to use, you primarily need to install Torch, Diffusers, and Transformers. Specific versions of these libraries are not required; the default versions should work fine.

```shell
conda create -n ctrl-adapter python=3.10
conda activate ctrl-adapter
pip install -r requirements.txt
```


# 🔮 Inference

### Video Generation with Condition Control

Here is a sample script that utilizes I2VGen-XL as the backbone model, with a depth map as the control condition. Inference consumes approximately 22GB of GPU memory on a single RTX 4090 GPU. The amount of memory required for inference depends on the backbone model used.

```
sh scripts/depth_ctrladapter_inference.sh
```

# 📝 TODO List
There are lots of ways we are excited about improving Hotshot-XL. For example:

- [x] Release environment setup, inference code for I2VGen-XL, and model checkpoints.
- [ ] Release checkpoints for other models.  (**WIP**)
- [ ] Release training code, and guidline to adapt our Ctrl-Adapter to new image/video diffusion models. 
- [ ] Release evaluation code.

💗 Please let us know in the issues or PRs if you're interested in any relevant backbones or down-stream tasks that can be implemented by our Ctrl-Adapter framework!

# 📚 BibTeX

If you find our project useful in your research, please cite the following paper:

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

- [SDXL](https://stability.ai/stable-diffusion)
- [I2VGen-XL](https://i2vgen-xl.github.io/)
- [Hotshot-XL](https://github.com/hotshotco/Hotshot-XL)
- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models)

We hope that releasing this model/codebase helps the community to continue pushing these creative tools forward in an open and responsible way.
