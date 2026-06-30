import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path

from src.config import CFG
from src.training.train_vqvae import train_vqvae
from src.training.train_ddpm import train_ddpm
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train the same VQ-VAE + latent DDPM pipeline on Fontainebleau sandstone.")
    p.add_argument("--stage", choices=["vqvae", "ddpm", "all"], default="all")
    p.add_argument("--raw_path", required=True, help="Prepared uint8 raw volume, 0=solid and 1=pore")
    p.add_argument("--save_dir", default="outputs/fontainebleau_phi0p2045")
    p.add_argument("--device", default="cuda")

    p.add_argument("--raw_shape", type=int, nargs=3, default=[480, 480, 480])
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--min_porosity", type=float, default=0.0)

    p.add_argument("--epochs_vae", type=int, default=80)
    p.add_argument("--epochs_ddpm", type=int, default=150)
    p.add_argument("--batch_vae", type=int, default=1)
    p.add_argument("--batch_ddpm", type=int, default=1)

    p.add_argument("--poro_center", type=float, default=0.2045, help="Use the training-rock porosity as FiLM normalization center")
    p.add_argument("--target_porosity", type=float, default=0.2045)
    p.add_argument("--poro_scale", type=float, default=0.02)
    p.add_argument("--poro_clip", type=float, default=5.0)

    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CFG()
    cfg.raw_path = args.raw_path
    cfg.raw_shape = tuple(args.raw_shape)
    cfg.patch_size = args.patch_size
    cfg.n_samples = args.n_samples
    cfg.min_porosity = args.min_porosity
    cfg.save_dir = args.save_dir
    cfg.device = args.device

    cfg.epochs_vae = args.epochs_vae
    cfg.epochs_ddpm = args.epochs_ddpm
    cfg.batch_size_vae = args.batch_vae
    cfg.batch_size_ddpm = args.batch_ddpm

    cfg.poro_center = args.poro_center
    cfg.target_porosity = args.target_porosity
    cfg.poro_scale = args.poro_scale
    cfg.poro_clip = args.poro_clip
    cfg.seed = args.seed

    if args.no_amp:
        cfg.amp = False
    elif args.amp:
        cfg.amp = True

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(cfg.save_dir, name="train_fontainebleau")
    set_seed(cfg.seed)

    logger.info("==== Fontainebleau training config ====")
    logger.info(str(cfg))

    if args.stage == "vqvae":
        train_vqvae(cfg, logger)
    elif args.stage == "ddpm":
        train_ddpm(cfg, logger)
    else:
        train_vqvae(cfg, logger)
        train_ddpm(cfg, logger)


if __name__ == "__main__":
    main()
