import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.summarize_all import main, summarize_results, write_summary


__all__ = ["main", "summarize_results", "write_summary"]


if __name__ == "__main__":
    main()
