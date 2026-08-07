import torch


class LinearNoiseScheduler:
    def __init__(self, num_timesteps: int, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cum_prod_sqrt = torch.sqrt(self.alphas_cum_prod)
        self.sqrt_one_minus_alphas_cum_prod = torch.sqrt(1.0 - self.alphas_cum_prod)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cum_prod = self.alphas_cum_prod.to(device)
        self.alphas_cum_prod_sqrt = self.alphas_cum_prod_sqrt.to(device)
        self.sqrt_one_minus_alphas_cum_prod = self.sqrt_one_minus_alphas_cum_prod.to(device)
        return self

    def _gather(self, values, t, x_shape):
        out = values[t]
        for _ in range(len(x_shape) - 1):
            out = out.unsqueeze(-1)
        return out

    def add_noise(self, original, noise, t):
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alphas_cum_prod = self.alphas_cum_prod_sqrt[t].reshape(batch_size)
        sqrt_one_minus_alphas_cum_prod = self.sqrt_one_minus_alphas_cum_prod[t].reshape(batch_size)

        for _ in range(len(original_shape) - 1):
            sqrt_alphas_cum_prod = sqrt_alphas_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alphas_cum_prod = sqrt_one_minus_alphas_cum_prod.unsqueeze(-1)

        return sqrt_alphas_cum_prod * original + sqrt_one_minus_alphas_cum_prod * noise

    def get_ddim_timesteps(self, num_inference_steps: int, device=None):
        if num_inference_steps >= self.num_timesteps:
            return torch.arange(
                self.num_timesteps - 1,
                -1,
                -1,
                device=device,
                dtype=torch.long,
            )

        return torch.linspace(
            self.num_timesteps - 1,
            0,
            num_inference_steps,
            device=device,
        ).long()

    def ddim_step(self, xt, noise_pred, t: int, t_prev: int, eta: float = 0.0):
        alpha_prod_t = self._gather(self.alphas_cum_prod, t, xt.shape)
        alpha_prod_t_prev = (
            self._gather(self.alphas_cum_prod, t_prev, xt.shape)
            if t_prev >= 0
            else torch.ones_like(alpha_prod_t)
        )

        pred_x0 = (
            xt - self._gather(self.sqrt_one_minus_alphas_cum_prod, t, xt.shape) * noise_pred
        ) / self._gather(self.alphas_cum_prod_sqrt, t, xt.shape)
        pred_x0 = pred_x0.clamp(-1, 1)

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
        variance = variance * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance.clamp(min=0).sqrt()

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2).clamp(min=0).sqrt() * noise_pred
        prev_sample = alpha_prod_t_prev.sqrt() * pred_x0 + pred_sample_direction

        if eta > 0:
            noise = torch.randn_like(xt)
            prev_sample = prev_sample + std_dev_t * noise

        return prev_sample, pred_x0

    def sample_prev_timestep(self, xt, noise_pred, t):
        x0 = (xt - (self.sqrt_one_minus_alphas_cum_prod[t] * noise_pred)) / torch.sqrt(self.alphas_cum_prod[t])
        x0 = torch.clamp(x0, -1, 1)

        mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alphas_cum_prod[t]))
        mean = mean / torch.sqrt(self.alphas[t])

        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alphas_cum_prod[t - 1]) / (1.0 - self.alphas_cum_prod[t])
            variance = variance * self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn(x0.shape).to(xt.device)
            return mean + sigma * z, x0
