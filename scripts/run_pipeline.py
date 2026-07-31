"""Internal modular implementation for the reproduction pipeline.

Reviewer-facing manuscript reproduction should use the top-level
``run_pipeline.py`` entry point. This module builds and executes the ordered
workflow used by that wrapper.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_scalar(value):
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [parse_scalar(item) for item in body.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"').strip("'")


def load_simple_yaml(path):
    data = {}
    current = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not line.startswith(" "):
                key = line.split(":", 1)[0].strip()
                value = line.split(":", 1)[1].strip()
                if value:
                    data[key] = parse_scalar(value)
                    current = None
                else:
                    data[key] = {}
                    current = key
            elif current is not None:
                key, value = line.strip().split(":", 1)
                data[current][key.strip()] = parse_scalar(value)
    return data


def load_config(path):
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        return load_simple_yaml(path)


def as_str_list(values):
    return [str(v) for v in values]


def run_step(name, command, dry_run=False):
    print(f"\n=== {name} ===")
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def command_report(path):
    return [sys.executable, "src/metrics/summarize_all.py", "--root", str(path)]


def demo_pipeline(cfg):
    demo = cfg["demo"]
    return [
        (
            "Synthetic porosity-control demo",
            [sys.executable, "scripts/demo_quantile_binarization.py"],
        ),
        (
            "Summarize demo outputs",
            command_report(demo["summary_root"]),
        ),
    ]


def main_pipeline(cfg, config_path="configs/main.yaml", checkpoint_dir=None, train_models=True):
    data = cfg["data"]
    train = cfg["training"]
    gen = cfg["generation"]
    eval_cfg = cfg["evaluation"]
    paths = cfg["paths"]
    model = cfg["model"]
    ckpt_dir = checkpoint_dir or paths["checkpoint_dir"]

    common_train = [
        "--raw_path",
        data["raw_path"],
        "--raw_shape",
        *as_str_list(data["raw_shape"]),
        "--save_dir",
        paths["checkpoint_dir"],
        "--device",
        train["device"],
        "--batch_vae",
        str(train["batch_vae"]),
        "--batch_ddpm",
        str(train["batch_ddpm"]),
        "--epochs_vae",
        str(train["epochs_vae"]),
        "--epochs_ddpm",
        str(train["epochs_ddpm"]),
        "--n_samples",
        str(train["n_samples"]),
        "--target_porosity",
        str(gen["targets"][0]),
        "--poro_center",
        str(model["poro_center"]),
        "--poro_scale",
        str(model["poro_scale"]),
    ]

    generate_cmd = [
        sys.executable,
        "scripts/generate_batch.py",
        "--ckpt_dir",
        ckpt_dir,
        "--out_root",
        paths["generated_root"],
        "--targets",
        *as_str_list(gen["targets"]),
        "--n_per_target",
        str(gen["n_per_target"]),
        "--seed_start",
        str(gen["seed_start"]),
        "--device",
        train["device"],
        "--poro_center",
        str(model["poro_center"]),
        "--poro_scale",
        str(model["poro_scale"]),
        "--n_steps",
        str(model["n_steps"]),
    ]

    steps = [
        (
            "Build real porosity groups",
            [
                sys.executable,
                "scripts/build_real_phi_groups.py",
                "--raw_path",
                data["raw_path"],
                "--raw_shape",
                *as_str_list(data["raw_shape"]),
                "--patch",
                str(data["patch_size"]),
                "--stride",
                str(data["stride"]),
                "--targets",
                *as_str_list(gen["targets"]),
                "--n_per_target",
                str(data["real_samples_per_target"]),
                "--out_root",
                paths["real_root"],
            ],
        ),
        ("Generate controlled samples", generate_cmd),
        (
            "Evaluate S2, lineal path, and EDT",
            [
                sys.executable,
                "scripts/evaluate_s2_lineal_edt.py",
                "--real_root",
                paths["real_root"],
                "--gen_root",
                paths["generated_root"],
                "--out_root",
                paths["curve_results"],
                "--targets",
                *as_str_list(gen["targets"]),
                "--r_max",
                str(eval_cfg["r_max"]),
            ],
        ),
        (
            "Evaluate voxel metrics and permeability",
            [
                sys.executable,
                "scripts/evaluate_voxel_and_perm.py",
                "--input_root",
                paths["generated_root"],
                "--output_csv",
                str(Path(paths["table_results"]) / "generated_voxel_perm.csv"),
                "--group_name",
                "gen",
                "--shape",
                *as_str_list([data["patch_size"]] * 3),
                "--voxel_size",
                str(data["voxel_size_m"]),
                "--recursive",
            ],
        ),
        (
            "Evaluate topology",
            [
                sys.executable,
                "scripts/evaluate_coordination_euler.py",
                "--real-root",
                paths["real_root"],
                "--gen-root",
                paths["generated_root"],
                "--out-root",
                paths["topology_results"],
            ],
        ),
        (
            "Evaluate PNM six-panel descriptors",
            [
                sys.executable,
                "scripts/evaluate_pore_network_6panel.py",
                "--config",
                config_path,
            ],
        ),
        (
            "Summarize result files",
            command_report(paths["results_root"]),
        ),
    ]

    if train_models:
        steps[1:1] = [
            (
                "Train VQ-VAE",
                [
                    sys.executable,
                    "scripts/train_256_vqvae_ddpm_lat64_v6_light96_full.py",
                    "--stage",
                    "vqvae",
                    *common_train,
                ],
            ),
            (
                "Train latent DDPM",
                [
                    sys.executable,
                    "scripts/train_256_vqvae_ddpm_lat64_v6_light96_full.py",
                    "--stage",
                    "ddpm",
                    *common_train,
                ],
            ),
        ]

    return steps


def fontainebleau_pipeline(cfg, train_models=True):
    fb_cfg_path = ROOT / "configs" / "fontainebleau_config.yaml"
    fb = load_config(fb_cfg_path)
    train = cfg["training"]
    model = cfg["model"]
    paths = cfg["paths"]

    out_root = "data/generated_fontainebleau_sets"
    save_dir = "outputs/fontainebleau_phi0p2045" if train_models else fb.get("final_checkpoint_dir", "savedmodels/fontainebleau_phi0p2045")

    steps = []
    if train_models:
        steps.append(
            (
            "Train Fontainebleau VQ-VAE and latent DDPM",
            [
                sys.executable,
                "scripts/train_fontainebleau.py",
                "--stage",
                "all",
                "--raw_path",
                fb["raw_path"],
                "--raw_shape",
                *as_str_list(fb["raw_shape"]),
                "--patch_size",
                str(fb["patch_size"]),
                "--save_dir",
                save_dir,
                "--device",
                train["device"],
                "--epochs_vae",
                str(train["epochs_vae"]),
                "--epochs_ddpm",
                str(train["epochs_ddpm"]),
                "--batch_vae",
                str(train["batch_vae"]),
                "--batch_ddpm",
                str(train["batch_ddpm"]),
                "--poro_center",
                str(fb["poro_center"]),
                "--target_porosity",
                str(fb["training_porosity"]),
                "--poro_scale",
                str(fb["poro_scale"]),
            ],
            )
        )

    steps.extend([
        (
            "Generate Fontainebleau validation samples",
            [
                sys.executable,
                "scripts/generate_batch.py",
                "--ckpt_dir",
                save_dir,
                "--out_root",
                out_root,
                "--targets",
                *as_str_list(fb["validation_targets"]),
                "--n_per_target",
                str(fb["n_per_target"]),
                "--seed_start",
                "0",
                "--device",
                train["device"],
                "--poro_center",
                str(fb["poro_center"]),
                "--poro_scale",
                str(fb["poro_scale"]),
                "--n_steps",
                str(model["n_steps"]),
            ],
        ),
        (
            "Evaluate Fontainebleau voxel metrics and permeability",
            [
                sys.executable,
                "scripts/evaluate_voxel_and_perm.py",
                "--input_root",
                out_root,
                "--output_csv",
                str(Path(paths["table_results"]) / "fontainebleau_voxel_perm.csv"),
                "--group_name",
                "fontainebleau",
                "--shape",
                *as_str_list([fb["patch_size"]] * 3),
                "--voxel_size",
                str(float(fb["voxel_size_um"]) * 1.0e-6),
                "--recursive",
            ],
        ),
        (
            "Summarize Fontainebleau result files",
            command_report(paths["results_root"]),
        ),
    ])
    return steps


def full_pipeline(cfg, config_path="configs/main.yaml"):
    return main_pipeline(cfg, config_path=config_path) + fontainebleau_pipeline(cfg)


def final_checkpoint_pipeline(cfg, config_path="configs/main.yaml"):
    paths = cfg["paths"]
    main_final_dir = paths.get("final_checkpoint_dir", "savedmodels/main_sandstone")
    return main_pipeline(
        cfg,
        config_path=config_path,
        checkpoint_dir=main_final_dir,
        train_models=False,
    ) + fontainebleau_pipeline(cfg, train_models=False)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Internal implementation for the digital-rock reproduction pipeline. "
            "Use top-level run_pipeline.py for official manuscript reproduction."
        )
    )
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--mode", choices=["demo", "main", "fontainebleau", "full", "final"], default="demo")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pipeline logic and write the manifest without training, generation, or evaluation.",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    if args.mode == "demo":
        steps = demo_pipeline(cfg)
    elif args.mode == "main":
        steps = main_pipeline(cfg, config_path=args.config)
    elif args.mode == "fontainebleau":
        steps = fontainebleau_pipeline(cfg)
    elif args.mode == "final":
        steps = final_checkpoint_pipeline(cfg, config_path=args.config)
    else:
        steps = full_pipeline(cfg, config_path=args.config)

    manifest = {"mode": args.mode, "config": args.config, "steps": []}
    for name, command in steps:
        manifest["steps"].append({"name": name, "command": command})
        run_step(name, command, dry_run=args.dry_run)

    manifest_path = ROOT / cfg["paths"]["results_root"] / f"pipeline_{args.mode}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote pipeline manifest: {manifest_path}")


if __name__ == "__main__":
    main()
