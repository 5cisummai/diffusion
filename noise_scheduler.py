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

    def add_noise(self, original, noise, t):
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alphas_cum_prod = self.alphas_cum_prod_sqrt[t].reshape(batch_size)
        sqrt_one_minus_alphas_cum_prod = self.sqrt_one_minus_alphas_cum_prod[t].reshape(batch_size)

        for _ in range(len(original_shape) - 1):
            sqrt_alphas_cum_prod = sqrt_alphas_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alphas_cum_prod = sqrt_one_minus_alphas_cum_prod.unsqueeze(-1)

        return sqrt_alphas_cum_prod * original + sqrt_one_minus_alphas_cum_prod * noise

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