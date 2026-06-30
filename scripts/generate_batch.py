import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path

from src.config import CFG
from src.sampling.batch_generate import batch_generate
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Batch generation using trained VQ-VAE + FiLM-conditioned latent DDPM.")
    p.add_argument("--ckpt_dir", required=True, help="Directory containing vqvae_final.pth, unet_final.pth and latent_stats.npz")
    p.add_argument("--out_root", required=True, help="Output root for generated samples")
    p.add_argument("--targets", type=float, nargs="+", default=[0.11, 0.12, 0.13, 0.14, 0.15])
    p.add_argument("--n_per_target", type=int, default=100)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--device", default="cuda")

    # Model/runtime parameters; defaults match the main training script.
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--downsample_factor", type=int, default=4)
    p.add_argument("--embedding_dim", type=int, default=32)
    p.add_argument("--num_embeddings", type=int, default=1024)
    p.add_argument("--commitment_cost", type=float, default=0.25)
    p.add_argument("--vae_base_ch", type=int, default=32)
    p.add_argument("--vae_max_ch", type=int, default=96)
    p.add_argument("--unet_base", type=int, default=64)
    p.add_argument("--unet_time_dim", type=int, default=128)

    p.add_argument("--n_steps", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--poro_center", type=float, default=0.13)
    p.add_argument("--poro_scale", type=float, default=0.02)
    p.add_argument("--poro_clip", type=float, default=5.0)
    p.add_argument("--eps_clip", type=float, default=5.0)
    p.add_argument("--clamp_x", type=float, default=15.0)

    p.add_argument("--no_prob", action="store_true", help="Do not save probability volume in npz; only save seg and metadata")
    p.add_argument("--no_raw", action="store_true", help="Do not save .raw binary files")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CFG()
    cfg.save_dir = args.ckpt_dir
    cfg.device = args.device

    cfg.patch_size = args.patch_size
    cfg.downsample_factor = args.downsample_factor
    cfg.embedding_dim = args.embedding_dim
    cfg.num_embeddings = args.num_embeddings
    cfg.commitment_cost = args.commitment_cost
    cfg.vae_base_ch = args.vae_base_ch
    cfg.vae_max_ch = args.vae_max_ch
    cfg.unet_base = args.unet_base
    cfg.unet_time_dim = args.unet_time_dim

    cfg.n_steps = args.n_steps
    cfg.beta_start = args.beta_start
    cfg.beta_end = args.beta_end
    cfg.poro_center = args.poro_center
    cfg.poro_scale = args.poro_scale
    cfg.poro_clip = args.poro_clip
    cfg.eps_clip = args.eps_clip
    cfg.clamp_x = args.clamp_x

    seeds = list(range(args.seed_start, args.seed_start + args.n_per_target))

    log_dir = Path(args.out_root) / "logs"
    logger = setup_logger(str(log_dir), name="batch_generate")
    set_seed(args.seed_start)

    batch_generate(
        cfg=cfg,
        targets=args.targets,
        seeds=seeds,
        out_root=args.out_root,
        logger=logger,
        save_prob=not args.no_prob,
        save_raw=not args.no_raw,
    )


if __name__ == "__main__":
    main()
