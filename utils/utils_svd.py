import torch
import numpy as np 


def add_time_ids_svd(fps, motion_bucket_id, noise_aug_strength, dtype, batch_size,):
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids


def rand_log_normal(u, loc=0., scale=1.):#, device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    #u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def _convert_to_karras(num_intervals=1000) -> torch.FloatTensor:
    
    """Constructs the noise schedule of Karras et al. (2022)."""
    # This code is adapted from EulerDiscreteScheduler in diffusers library: 
    # https://github.com/huggingface/diffusers/blob/v0.27.2/src/diffusers/schedulers/scheduling_euler_discrete.py#L132 
    
    sigma_min = 0.002
    sigma_max = 700

    rho = 7.0  # 7.0 is the value used in the paper
    ramp = np.linspace(0, 1, num_intervals)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


def _sigma_to_t(sigma, log_sigmas):
    
    # This code is adapted from EulerDiscreteScheduler in diffusers library: 
    # https://github.com/huggingface/diffusers/blob/v0.27.2/src/diffusers/schedulers/scheduling_euler_discrete.py#L132 

    # get log sigma
    log_sigma = np.log(np.maximum(sigma, 1e-10))

    # get distribution
    dists = log_sigma - log_sigmas[:, np.newaxis]

    # get sigmas range
    low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
    high_idx = low_idx + 1

    low = log_sigmas[low_idx]
    high = log_sigmas[high_idx]

    # interpolate sigmas
    w = (low - log_sigma) / (low - high)
    w = np.clip(w, 0, 1)

    # transform interpolation to time range
    t = (1 - w) * low_idx + w * high_idx
    t = t.reshape(sigma.shape)
    return t


def sample_svd_sigmas_timesteps(bsz, sigmas_svd, num_inference_steps=25):

    random_indices = torch.rand((bsz ))
    random_indices = torch.Tensor(random_indices * len(sigmas_svd)).to(torch.int32)
    
    u = random_indices / (len(sigmas_svd)-1) * (1 - 1 / num_inference_steps) + 0.001

    sigma_svd = sigmas_svd[random_indices]
    
    return u, sigma_svd 


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


# resizing utils
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output
