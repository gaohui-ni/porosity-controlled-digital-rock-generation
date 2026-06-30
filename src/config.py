from dataclasses import dataclass
from typing import Tuple

@dataclass
class CFG:
    raw_path: str = "S1.raw"
    raw_shape: Tuple[int, int, int] = (800, 800, 800)
    patch_size: int = 256
    n_samples: int = 1000
    min_porosity: float = 0.0
    max_tries: int = 50

    # down=4 -> latent 64^3 (for fine pores)
    downsample_factor: int = 4  # fixed for this script (4)

    # VQ-VAE
    embedding_dim: int = 32
    num_embeddings: int = 1024
    commitment_cost: float = 0.25
    lr_vae: float = 2e-4
    epochs_vae: int = 80
    batch_size_vae: int = 1
    amp: bool = True

    # VQ-VAE width control (IMPORTANT)
    vae_base_ch: int = 32       # base width
    vae_max_ch: int = 96        # max width (<=96 recommended for down=4 256^3 on 40GB)

    # reconstruction losses
    bce_weight: float = 1.0
    dice_weight: float = 0.2
    pos_weight_mode: str = "batch"  # "fixed" or "batch"
    poro_center: float = 0.13
    pos_weight_clip: float = 20.0

    # DDPM
    lr_ddpm: float = 1e-4
    epochs_ddpm: int = 150
    batch_size_ddpm: int = 1
    n_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # DDPM UNet width
    unet_base: int = 64
    unet_time_dim: int = 128

    # conditioning
    poro_scale: float = 0.02
    poro_clip: float = 5.0

    # runtime
    save_interval: int = 5
    log_interval: int = 20
    save_dir: str = "./ckpt_lat64_vq80_ddpm150_light96"
    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 123

    # latent stats for ddpm
    stats_batches: int = 80

    # sampling
    target_porosity: float = 0.13
    sample_seed: int = 0
    clamp_x: float = 15.0
    eps_clip: float = 5.0

    # IMPORTANT: quantile thresholding to match porosity
    use_quantile_threshold: bool = True

    # grad clip (optional)
    grad_clip: float = 1.0
