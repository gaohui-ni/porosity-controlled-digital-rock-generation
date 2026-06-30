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


def full_pipeline(cfg):
    data = cfg["data"]
    train = cfg["training"]
    gen = cfg["generation"]
    eval_cfg = cfg["evaluation"]
    paths = cfg["paths"]
    model = cfg["model"]

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
        paths["checkpoint_dir"],
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

    return [
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
            "Summarize result files",
            command_report(paths["results_root"]),
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run the digital-rock reproduction pipeline.")
    parser.add_argument("--config", default="configs/experiment_main.yaml")
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    steps = demo_pipeline(cfg) if args.mode == "demo" else full_pipeline(cfg)

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
