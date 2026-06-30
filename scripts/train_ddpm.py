import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config import CFG
from src.training.train_ddpm import train_ddpm
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def main():
    cfg = CFG()
    set_seed(cfg.seed)
    logger = setup_logger(cfg.save_dir)
    train_ddpm(cfg, logger)


if __name__ == "__main__":
    main()
