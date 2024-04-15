"""
This file is mostly copied from diffusers library: 
    https://github.com/huggingface/diffusers/blob/v0.27.2/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L149
We added some additional arguments for ctrl-adapter support

You can ctrl+F and search ### to see the location of code where we make changes
"""

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

### we replaced .. into diffusers. to directly import the following modules from diffusers library ###
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL
#from diffusers.models.unets.unet_i2vgen_xl import I2VGenXLUNet
from i2vgen_xl.models.unets.unet_i2vgen_xl import I2VGenXLUNet ### we use the I2VGenXL UNet from our repo

from diffusers.schedulers import DDIMScheduler
from diffusers.utils import (
    BaseOutput,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

### import libraries related to our ctrl-adapter
from model.ctrl_adapter import ControlNetAdapter
from model.ctrl_router import ControlNetRouter
from model.ctrl_helper import ControlNetHelper

from controlnet.multicontrolnet import MultiControlNetModel
from controlnet.controlnet import ControlNetModel

import torch.nn.functional as F
from einops import rearrange
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.utils.torch_utils import is_compiled_module


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import I2VGenXLPipeline
        >>> from diffusers.utils import export_to_gif, load_image

        >>> pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
        >>> pipeline.enable_model_cpu_offload()

        >>> image_url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"
        >>> image = load_image(image_url).convert("RGB")

        >>> prompt = "Papers were floating in the air on a table in the library"
        >>> negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
        >>> generator = torch.manual_seed(8888)

        >>> frames = pipeline(
        ...     prompt=prompt,
        ...     image=image,
        ...     num_inference_steps=50,
        ...     negative_prompt=negative_prompt,
        ...     guidance_scale=9.0,
        ...     generator=generator
        ... ).frames[0]
        >>> video_path = export_to_gif(frames, "i2v.gif")
        ```
"""


# Copied from diffusers.pipelines.animatediff.pipeline_animatediff.tensor2vid
def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs


@dataclass
class I2VGenXLPipelineOutput(BaseOutput):
    r"""
     Output class for image-to-video pipeline.

     Args:
         frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
             List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised
     PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
    `(batch_size, num_frames, channels, height, width)`
    """

    frames: Union[torch.Tensor, np.ndarray, List[List[PIL.Image.Image]]]
    down_block_weights: List ### output controlnet weights if using multi-condition control
    mid_block_weights: List ### output controlnet weights if using multi-condition control


class I2VGenXLControlNetAdapterPipeline(DiffusionPipeline):
    r"""
    Pipeline for image-to-video generation as proposed in [I2VGenXL](https://i2vgen-xl.github.io/).

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer (`CLIPTokenizer`):
            A [`~transformers.CLIPTokenizer`] to tokenize text.
        unet ([`I2VGenXLUNet`]):
            A [`I2VGenXLUNet`] to denoise the encoded video latents.
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        image_encoder: CLIPVisionModelWithProjection,
        feature_extractor: CLIPImageProcessor,
        unet: I2VGenXLUNet,
        scheduler: DDIMScheduler,
        
        ### controlnet, adapter, helper, and router are newly added
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        adapter: ControlNetAdapter,
        helper: ControlNetHelper,
        router: ControlNetRouter = None,
    ):
        super().__init__()
        
        ### if input controlnet is a list or tuple, construct from multi-controlnet class
        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)
            

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet, ###
            adapter=adapter, ###
            helper=helper, ###
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_resize=False)

        # ### image processor for controlnet
        # self.control_image_processor = VaeImageProcessor(
        #     vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        # )

        self.router = router
        
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    def encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if self.do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            # Apply clip_skip to negative prompt embeds
            if clip_skip is None:
                negative_prompt_embeds = self.text_encoder(
                    uncond_input.input_ids.to(device),
                    attention_mask=attention_mask,
                )
                negative_prompt_embeds = negative_prompt_embeds[0]
            else:
                negative_prompt_embeds = self.text_encoder(
                    uncond_input.input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                negative_prompt_embeds = negative_prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                negative_prompt_embeds = self.text_encoder.text_model.final_layer_norm(negative_prompt_embeds)

        if self.do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def _encode_image(self, image, device, num_videos_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # Normalize the image with CLIP training stats.
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if self.do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def decode_latents(self, latents, decode_chunk_size=None):
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        if decode_chunk_size is not None:
            frames = []
            for i in range(0, latents.shape[0], decode_chunk_size):
                frame = self.vae.decode(latents[i : i + decode_chunk_size]).sample
                frames.append(frame)
            image = torch.cat(frames, dim=0)
        else:
            image = self.vae.decode(latents).sample

        decode_shape = (batch_size, num_frames, -1) + image.shape[2:]
        video = image[None, :].reshape(decode_shape).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()
        return video

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        image,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

    def prepare_image_latents(
        self,
        image,
        device,
        num_frames,
        num_videos_per_prompt,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.sample()
        image_latents = image_latents * self.vae.config.scaling_factor

        # Add frames dimension to image latents
        image_latents = image_latents.unsqueeze(2)

        # Append a position mask for each subsequent frame
        # after the intial image latent frame
        frame_position_mask = []
        for frame_idx in range(num_frames - 1):
            scale = (frame_idx + 1) / (num_frames - 1)
            frame_position_mask.append(torch.ones_like(image_latents[:, :, :1]) * scale)
        if frame_position_mask:
            frame_position_mask = torch.cat(frame_position_mask, dim=2)
            image_latents = torch.cat([image_latents, frame_position_mask], dim=2)

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1, 1)

        if self.do_classifier_free_guidance:
            image_latents = torch.cat([image_latents] * 2)

        return image_latents

    # Copied from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth.TextToVideoSDPipeline.prepare_latents
    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = 704,
        width: Optional[int] = 1280,
        target_fps: Optional[int] = 16,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        num_videos_per_prompt: Optional[int] = 1,
        decode_chunk_size: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = 1,

        ### the following arguments are newly added
        control_images: List[PIL.Image.Image] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        num_images_per_prompt: Optional[int] = 1,
        guess_mode: bool = False,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        sparse_frames = None,
        skip_conv_in = False,
        skip_time_emb = False,
        fixed_controlnet_timestep = -1,
        use_size_512 = True,
        adapter_locations = None,
        inference_expert_masks = None,
        fixed_weights = None,
    ):
        r"""
        The call function to the pipeline for image-to-video generation with [`I2VGenXLPipeline`].

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            target_fps (`int`, *optional*):
                Frames per second. The rate at which the generated images shall be exported to a video after generation. This is also used as a "micro-condition" while generation.
            num_frames (`int`, *optional*):
                The number of video frames to generate.
            num_inference_steps (`int`, *optional*):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            eta (`float`, *optional*):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            num_videos_per_prompt (`int`, *optional*):
                The number of images to generate per prompt.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`pipelines.i2vgen_xl.pipeline_i2vgen_xl.I2VGenXLPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`pipelines.i2vgen_xl.pipeline_i2vgen_xl.I2VGenXLPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated frames.
        """

        ### newly added
        if adapter_locations is None:
            adapter_locations = ['A', 'B', 'C', 'D', 'M']
            
        video_length = num_frames
        
        if video_length > 1 and num_images_per_prompt > 1:
            print(f"Warning - setting num_images_per_prompt = 1 because video_length = {video_length}")
            num_images_per_prompt = 1

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]
        ###
        

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, image, height, width, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = guidance_scale
        
        ### newly added
        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = False
        ###


        # 3.1 Encode input text prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        #if self.do_classifier_free_guidance:
        #    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        ### newly added
        (
            controlnet_prompt_embeds, 
            controlnet_negative_prompt_embeds, 
            pooled_prompt_embeds, 
            negative_pooled_prompt_embeds
        ) =self.helper.encode_controlnet_prompt(
            prompt,
            device,
            1,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )

        # 3.1.5 Prepare image
        if isinstance(controlnet, ControlNetModel):

            assert len(control_images) == video_length * batch_size
       
            images = self.helper.prepare_images(
                    images=control_images,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

            height, width = images.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):

            images = []
            
            for image_ in control_images:
                image_ = self.helper.prepare_images(
                    images=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
            
                images.append(image_)
            
            #image = images
            height, width = images[0].shape[-2:]

        else:
            assert False
        ###


        # 3.2 Encode image prompt
        # 3.2.1 Image encodings.
        # https://github.com/ali-vilab/i2vgen-xl/blob/2539c9262ff8a2a22fa9daecbfd13f0a2dbc32d0/tools/inferences/inference_i2vgen_entrance.py#L114
        cropped_image = _center_crop_wide(image, (width, width))
        cropped_image = _resize_bilinear(
            cropped_image, (self.feature_extractor.crop_size["width"], self.feature_extractor.crop_size["height"])
        )
        image_embeddings = self._encode_image(cropped_image, device, num_videos_per_prompt)

        # 3.2.2 Image latents.
        resized_image = _center_crop_wide(image, (width, height))
        image = self.image_processor.preprocess(resized_image).to(device=device, dtype=image_embeddings.dtype)
        image_latents = self.prepare_image_latents(
            image,
            device=device,
            num_frames=num_frames,
            num_videos_per_prompt=num_videos_per_prompt,
        )

        # 3.3 Prepare additional conditions for the UNet.
        if self.do_classifier_free_guidance:
            fps_tensor = torch.tensor([target_fps, target_fps]).to(device)
        else:
            fps_tensor = torch.tensor([target_fps]).to(device)
        fps_tensor = fps_tensor.repeat(batch_size * num_videos_per_prompt, 1).ravel()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        ### newly added
        # 6.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 6.2 Prepare added time ids & embeddings
        if isinstance(images, list):
            original_size = images[0].shape[-2:]
        else:
            original_size = images.shape[-2:]
        target_size = (height, width)

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self.helper._get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype)

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self.helper._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
            controlnet_prompt_embeds = torch.cat([controlnet_negative_prompt_embeds, controlnet_prompt_embeds])

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        controlnet_prompt_embeds = controlnet_prompt_embeds.to(device)

        
        if isinstance(images, list):
            images = [rearrange(img, "b f c h w -> (b f) c h w") for img in images]
        else:
            images = rearrange(images, "b f c h w -> (b f) c h w")  
            
        if video_length > 1:
            # use repeat_interleave as we need to match the rearrangement above.
            controlnet_prompt_embeds = controlnet_prompt_embeds.repeat_interleave(video_length, dim=0) 

        output_down_block_weights = []
        output_mid_block_weights = []
        ###



        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                

                ### newly added
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input 
                    controlnet_added_cond_kwargs = added_cond_kwargs

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i] 


                # this will be non interlaced when arranged!
                control_model_input = rearrange(control_model_input, "b c f h w -> (b f) c h w") 
                # if we chunked this by 2 - the top 8 frames will be positive for cfg
                # the bottom half will be negative for cfg...

                if video_length > 1:
                    controlnet_added_cond_kwargs = {
                        "text_embeds": controlnet_added_cond_kwargs['text_embeds'].repeat_interleave(video_length, dim=0),
                        "time_ids": controlnet_added_cond_kwargs['time_ids'].repeat_interleave(video_length, dim=0)
                    }

                _, _, control_model_input_h, control_model_input_w = control_model_input.shape
                if (control_model_input_h, control_model_input_w) != (64, 64) and use_size_512:
                    reshaped_control_model_input = F.adaptive_avg_pool2d(control_model_input, (64, 64))
                    reshaped_images = F.adaptive_avg_pool2d(images, (512, 512))
                else:
                    reshaped_control_model_input = control_model_input
                    reshaped_images = images

                # todo - check if video_length > 1 this needs to produce num_frames * batch_size samples...
                with torch.no_grad():

                    if fixed_controlnet_timestep >=0: 
                        controlnet_timesteps = (torch.zeros_like(t) + fixed_controlnet_timestep).long().to(t.device)
                    else:
                        controlnet_timesteps = t

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        reshaped_control_model_input,
                        controlnet_timesteps,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=reshaped_images,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        added_cond_kwargs=controlnet_added_cond_kwargs,
                        return_dict=False,
                        skip_conv_in = skip_conv_in,
                        skip_time_emb = skip_time_emb,
                    )


                # this part is for MoE router
                if self.router is not None:
                    with torch.no_grad():
                        if self.router.router_type == 'timestep_weights':
                            input_t = t.unsqueeze(0) if t.dim()==0 else t
                            down_block_weights, mid_block_weights = self.router(input_t.to(self.router.dtype), 
                                                                                sparse_mask = inference_expert_masks)
                        elif self.router.router_type == 'simple_weights' or self.router.router_type == 'equal_weights':
                            down_block_weights, mid_block_weights = self.router(sparse_mask = inference_expert_masks)

                        elif self.router.router_type == 'embedding_weights':
                            down_block_weights, mid_block_weights = self.router(router_input=image_embeddings[-1].unsqueeze(0).to(self.router.dtype), 
                                                                                sparse_mask = inference_expert_masks)
                        elif self.router.router_type == 'timestep_embedding_weights':
                            input_t = t.unsqueeze(0) if t.dim()==0 else t
                            down_block_weights, mid_block_weights = self.router(router_input=[input_t.to(self.router.dtype), 
                                                                                image_embeddings[-1].unsqueeze(0).to(self.router.dtype)], 
                                                                                sparse_mask = inference_expert_masks)
                        
                    output_down_block_weights.append(down_block_weights.cpu().numpy().tolist())
                    if mid_block_weights is not None:
                        output_mid_block_weights.append(mid_block_weights.cpu().numpy().tolist())
                    else:
                        output_mid_block_weights.append(None)
                    
                    num_routers = self.router.num_routers
                    num_experts = self.router.num_experts
                    
                    
                    # merge the controlnets' features according to router weights
                    if mid_block_weights is not None:
                        mid_block_res_sample_merged = 0
                        idx_e = 0
                        for e in range(num_experts):
                            if inference_expert_masks[e] == True:
                                mid_block_res_sample_merged = mid_block_res_sample_merged + \
                                    mid_block_res_sample[idx_e] * mid_block_weights.repeat_interleave(num_frames, dim = 0)[e]
                                idx_e += 1
                    else:
                        mid_block_res_sample_merged = None
                    
                    down_block_res_samples_merged = [0 for k in range(num_routers)]
                    for k in range(num_routers):
                        idx_e = 0
                        for e in range(num_experts):
                            if inference_expert_masks[e] == True:
                                down_block_res_samples_merged[k] = down_block_res_samples_merged[k] + \
                                    down_block_res_samples[idx_e][k] * down_block_weights[k].repeat_interleave(num_frames, dim = 0)[e] 
                                idx_e += 1                       
                                
                    down_block_res_samples = down_block_res_samples_merged
                    mid_block_res_sample = mid_block_res_sample_merged
                        

                # this part is for sparse control
                # we only give the sparse key frames to our adapter below
                if sparse_frames is not None:
                    sparse_frames = [int(sparse_frames[k]) for k in range(len(sparse_frames))]
                    #print("sparse_frames", sparse_frames)
                    if self.do_classifier_free_guidance:
                        double_sparse_frames = sparse_frames + [(sparse_frames[k] + num_frames) for k in range(len(sparse_frames))]
                    down_block_res_samples = [down_block_res_samples[i][double_sparse_frames, :] for i in range(len(down_block_res_samples))]
                    mid_block_res_sample = mid_block_res_sample[double_sparse_frames, :]


                with torch.no_grad():
                    mid_block_res_sample = mid_block_res_sample.to(self.adapter.dtype) if 'M' in adapter_locations else None

                    adapter_input_num_frames = len(sparse_frames) if sparse_frames is not None else num_frames
                
                    # get the mid block and output block features output from adapters
                    adapted_down_block_res_samples, adapted_mid_block_res_sample = self.adapter(
                        down_block_res_samples = [down_block.to(self.adapter.dtype) for down_block in down_block_res_samples],
                        mid_block_res_sample = mid_block_res_sample, 
                        sparsity_masking=sparse_frames, 
                        num_frames=adapter_input_num_frames, 
                        timestep = t, 
                        encoder_hidden_states = image_embeddings[-1].unsqueeze(0)
                        )

                    
                    # transform from sparse frame to dense frame, since I2VGen-XL UNet needs dense frames as input
                    if sparse_frames is not None:
                        if self.do_classifier_free_guidance:
                            full_n_sample_frames = num_frames * 2
                            full_sparsity_masking = double_sparse_frames
                        else:
                            full_n_sample_frames = num_frames
                            full_sparsity_masking = sparse_frames

                        full_adapted_down_block_res_samples = []
                        for k in range(len(adapted_down_block_res_samples)):
                            _, c, h, w = adapted_down_block_res_samples[k].shape 
                            full_adapted_down_block_res_samples.append(torch.zeros((full_n_sample_frames, c, h, w)).to(device))
                            for j, pos in enumerate(full_sparsity_masking):
                                full_adapted_down_block_res_samples[k][pos] = adapted_down_block_res_samples[k][j]
                        if adapted_mid_block_res_sample is not None:
                            _, c, h, w = adapted_mid_block_res_sample.shape 
                            full_adapted_mid_block_res_sample = torch.zeros((full_n_sample_frames, c, h, w)).to(device)
                            for j, pos in enumerate(full_sparsity_masking):
                                full_adapted_mid_block_res_sample[pos] = adapted_mid_block_res_sample[j]
                        else:
                            full_adapted_mid_block_res_sample = None

                    else:
                        full_adapted_mid_block_res_sample = adapted_mid_block_res_sample
                        full_adapted_down_block_res_samples = adapted_down_block_res_samples

                    if full_adapted_mid_block_res_sample is not None:
                        full_adapted_mid_block_res_sample = rearrange(full_adapted_mid_block_res_sample, "(bs nf) c h w -> bs c nf h w", bs=2)
                    full_adapted_down_block_res_samples = [rearrange(down_block, "(bs nf) c h w -> bs c nf h w", bs=2) 
                                                                for down_block in full_adapted_down_block_res_samples]
                    if cond_scale == 0:
                        full_adapted_down_block_res_samples = None
                ###
                
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    fps=fps_tensor,
                    image_latents=image_latents,
                    image_embeddings=image_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=full_adapted_down_block_res_samples, ### newly added
                    mid_block_additional_residual=full_adapted_mid_block_res_sample, ### newly added
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents
                batch_size, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # reshape latents back
                latents = latents[None, :].reshape(batch_size, frames, channel, width, height).permute(0, 2, 1, 3, 4)
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        #if output_type == "latent":
        #    return I2VGenXLPipelineOutput(frames=latents)

        # video = self.decode_latents(latents, decode_chunk_size=decode_chunk_size)

        # # Convert to tensor
        # if output_type == "tensor":
        #     video = torch.from_numpy(video)

        # if not return_dict:
        #     return video

        # return I2VGenXLPipelineOutput(videos=video)

        video_tensor = self.decode_latents(latents, decode_chunk_size=decode_chunk_size)
        video = tensor2vid(video_tensor, self.image_processor, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        ### additionally return the mid/output block router weights
        #return I2VGenXLPipelineOutput(frames=video, down_block_weights=None, mid_block_weights=None)
        return I2VGenXLPipelineOutput(frames=video, down_block_weights=output_down_block_weights, mid_block_weights=output_mid_block_weights)


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
    image: Union[torch.Tensor, List[torch.Tensor], PIL.Image.Image, List[PIL.Image.Image]], resolution: Tuple[int, int]
):
    # First convert the images to PIL in case they are float tensors (only relevant for tests now).
    image = _convert_pt_to_pil(image)

    if isinstance(image, list):
        image = [u.resize(resolution, PIL.Image.BILINEAR) for u in image]
    else:
        image = image.resize(resolution, PIL.Image.BILINEAR)
    return image


def _center_crop_wide(
    image: Union[torch.Tensor, List[torch.Tensor], PIL.Image.Image, List[PIL.Image.Image]], resolution: Tuple[int, int]
):
    # First convert the images to PIL in case they are float tensors (only relevant for tests now).
    image = _convert_pt_to_pil(image)

    if isinstance(image, list):
        scale = min(image[0].size[0] / resolution[0], image[0].size[1] / resolution[1])
        image = [u.resize((round(u.width // scale), round(u.height // scale)), resample=PIL.Image.BOX) for u in image]

        # center crop
        x1 = (image[0].width - resolution[0]) // 2
        y1 = (image[0].height - resolution[1]) // 2
        image = [u.crop((x1, y1, x1 + resolution[0], y1 + resolution[1])) for u in image]
        return image
    else:
        scale = min(image.size[0] / resolution[0], image.size[1] / resolution[1])
        image = image.resize((round(image.width // scale), round(image.height // scale)), resample=PIL.Image.BOX)
        x1 = (image.width - resolution[0]) // 2
        y1 = (image.height - resolution[1]) // 2
        image = image.crop((x1, y1, x1 + resolution[0], y1 + resolution[1]))
        return image
