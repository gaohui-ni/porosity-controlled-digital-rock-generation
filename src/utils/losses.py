import numpy as np
import torch
import torch.nn.functional as F

from src.config import CFG

def soft_dice_loss(prob, target, eps=1e-6):
    num = 2 * (prob * target).sum(dim=(1,2,3,4))
    den = (prob + target).sum(dim=(1,2,3,4)) + eps
    return 1 - (num / den).mean()

def make_pos_weight(cfg: CFG, poro_batch: torch.Tensor, device: str):
    if cfg.pos_weight_mode == "batch":
        p = float(poro_batch.mean().detach().cpu())
    else:
        p = float(cfg.poro_center)
    pw = (1.0 - p) / max(p, 1e-6)
    pw = float(np.clip(pw, 1.0, cfg.pos_weight_clip))
    return torch.tensor([pw], device=device, dtype=torch.float32)
