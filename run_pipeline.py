"""Official manuscript reproduction entry point.

Use this top-level wrapper for reviewer-facing reproduction:

    python run_pipeline.py --mode full --config configs/main.yaml

The implementation lives in ``scripts/run_pipeline.py`` so that the public
entry point remains stable while the internal workflow can stay modular.
"""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "run_pipeline.py"), run_name="__main__")
