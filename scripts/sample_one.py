import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config import CFG
from src.sampling.sample import sample_one
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def main():
    cfg = CFG()
    set_seed(cfg.sample_seed)
    logger = setup_logger(cfg.save_dir)
    sample_one(cfg, logger)


if __name__ == "__main__":
    main()
