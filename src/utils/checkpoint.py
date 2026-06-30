import os
import json
import logging
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn

from src.config import CFG

def save_cfg_json(cfg: CFG, logger: logging.Logger):
    path = os.path.join(cfg.save_dir, "run_cfg.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    logger.info(f"[CFG] saved: {path}")

def save_ckpt(path: str, model: nn.Module, optimizer, epoch: int, extra: Optional[dict], logger: logging.Logger):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, path)
    logger.info(f"[AutoSave] {os.path.basename(path)} (epoch={epoch})")

def load_ckpt(path: str, model: nn.Module, optimizer, device: str, logger: logging.Logger):
    if not os.path.exists(path):
        return 0, {}
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(f"[Resume] {os.path.basename(path)} from epoch={ckpt['epoch']}")
    extra = ckpt.get("extra", {}) or {}
    return int(ckpt["epoch"]) + 1, extra
