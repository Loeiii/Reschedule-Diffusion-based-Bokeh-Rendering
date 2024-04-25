import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from .noise import newton


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

class CustomWeightedLoss(nn.Module):
    def __init__(self, weights = torch.tensor([0.9, 0.9, 1.1]), reduction = 'sum'):
        super(CustomWeightedLoss, self).__init__()
        self.weights = weights
        self.reduction = reduction

    def forward(self, predictions, targets):
        squared_diff = (predictions - targets)**2
        weighted_squared_diff = squared_diff * self.weights.view(1, -1, 1, 1).to(predictions.device)
        if self.reduction == 'mean':
            loss = torch.mean(weighted_squared_diff)
        elif self.reduction == 'sum':
            loss = torch.sum(weighted_squared_diff)
        return loss
    
    def to(self, device):
        # Move the weights tensor to the specified device
        self.weights = self.weights.to(device)
        return super(CustomWeightedLoss, self).to(device)

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def find_first_smaller_index(tensor, target):
    left, right = 0, len(tensor) - 1
    result = -1

    while left <= right:
        mid = (left + right) // 2

        if tensor[mid] < target:
            result = mid
            right = mid - 1
        else:
            left = mid + 1

    return result

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        elif self.loss_type == 'weighted':
            self.loss_func = CustomWeightedLoss()
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas 

        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        noise_pred = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)

        eta = 1.0
        sigma = eta**2  * self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])

        coef1 = (torch.exp(torch.log(1. - self.betas)*(-0.5)) * self.sqrt_one_minus_alphas_cumprod - 
            torch.exp(torch.log((1. - self.alphas_cumprod_prev - sigma))*0.5))

        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        return (torch.exp(torch.log(1. - self.betas[t])*(-0.5)) * x - 
            coef1[t] * noise_pred + 
            torch.exp(torch.log(sigma) * (0.5)) * noise)

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, mode= 'origin'):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            noise = torch.randn_like(x_in)
            # noise = torch.from_numpy(np.load("experiments/0_1_xt.npy")).unsqueeze(0).to(x_in.device)

            if mode == 'origin':
                img = noise
                n_sample = self.num_timesteps
            elif mode == 'piror':
                n_sample = self.num_timesteps
                img = self.q_sample(
                    x_start=x, continuous_sqrt_alpha_cumprod=torch.sqrt(self.alphas_cumprod[-1]).view(-1, 1, 1, 1), noise= noise)
            elif mode == 'dynamic':
                noise_scale = 96 * max(max(x.std(axis = (2, 3)))) ** 2 
                alpha_bar = 1 / (noise_scale + 1)
                start_t = find_first_smaller_index(self.alphas_cumprod, alpha_bar)
                n_sample = start_t
                img = self.q_sample(
                    x_start=x, continuous_sqrt_alpha_cumprod=torch.sqrt(self.alphas_cumprod[start_t]).view(-1, 1, 1, 1), noise= noise)
            elif mode == 'refined':
                noise_scale = 96 * max(max(x.std(axis = (2, 3)))) ** 2 
                alpha_bar = 1 / (noise_scale + 1)
                n_sample = self.num_timesteps
                gamma = 0.1
                beta_end = newton(n_sample, alpha_bar.detach().cpu().numpy() * (1 - np.exp(-n_sample/20)), gamma, 0.01)
                opt = {'schedule':'linear','n_timestep':n_sample, 'linear_start': beta_end - gamma, 'linear_end': beta_end}
                self.set_new_noise_schedule(opt, x_in.device)
                img = self.q_sample(
                    x_start=x, continuous_sqrt_alpha_cumprod=torch.sqrt(self.alphas_cumprod[-1]).view(-1, 1, 1, 1), noise= noise)

            ret_img = noise
            for i in tqdm(reversed(range(0, n_sample)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def bokeh(self, x_in, continous=False, mode = 'origin'):
        return self.p_sample_loop(x_in, continous, mode)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['Gt']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1 - 100)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['Cond'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
