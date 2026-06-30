from scripts.generate_batch import main

# This wrapper is intentionally thin. Use it with Fontainebleau defaults, e.g.:
# python scripts/generate_batch.py \
#   --ckpt_dir outputs/fontainebleau_phi0p2045 \
#   --out_root generated_fontainebleau_sets \
#   --targets 0.2045 0.1743 0.1263 0.0853 \
#   --n_per_target 50 \
#   --poro_center 0.2045 \
#   --device cuda

if __name__ == "__main__":
    main()
