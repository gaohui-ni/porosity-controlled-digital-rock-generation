"""Compatibility wrapper for the historical Fontainebleau batch generator.

Example:

    python scripts/generate_fontainebleau_batch.py \
      --ckpt_dir outputs/fontainebleau_phi0p2045 \
      --out_root generated_fontainebleau_sets \
      --targets 0.2045 0.1743 0.1263 \
      --n_per_target 20 \
      --poro_center 0.13 \
      --device cuda

This is not the unavailable Fuel Fig. 11 epoch-75/EXT3X workflow.
"""

from scripts.generate_batch import main


if __name__ == "__main__":
    main()
