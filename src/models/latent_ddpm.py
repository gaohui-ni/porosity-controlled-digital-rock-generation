import torch


def make_ddpm_schedule(n_steps: int, beta_start: float, beta_end: float, device: str):
    betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod
