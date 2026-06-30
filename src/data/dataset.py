import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

class Rock256ProbDataset(Dataset):
    """
    raw: uint8, 0=solid matrix, 1=pore space
    return:
      x: float32 in [0,1], [1,256,256,256]
      poro: float32 scalar in [0,1]
    """
    def __init__(self, raw_path: str, raw_shape: Tuple[int, int, int], patch_size: int,
                 n_samples: int, min_porosity: float, max_tries: int):
        self.raw_shape = tuple(raw_shape)
        self.patch_size = int(patch_size)
        self.n_samples = int(n_samples)
        self.min_porosity = float(min_porosity)
        self.max_tries = int(max_tries)

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"raw not found: {raw_path}")

        self.mm = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=self.raw_shape)

    def __len__(self):
        return self.n_samples

    def _sample_patch(self):
        S = self.patch_size
        X, Y, Z = self.raw_shape
        x0 = np.random.randint(0, X - S + 1)
        y0 = np.random.randint(0, Y - S + 1)
        z0 = np.random.randint(0, Z - S + 1)
        patch = np.array(self.mm[x0:x0+S, y0:y0+S, z0:z0+S], dtype=np.uint8)
        return patch

    def __getitem__(self, idx):
        for i in range(self.max_tries):
            p = self._sample_patch()
            poro = float(np.mean(p == 1))
            if poro >= self.min_porosity or i == self.max_tries - 1:
                x = p.astype(np.float32)
                x = torch.from_numpy(x).unsqueeze(0)  # [1,S,S,S]
                return x, torch.tensor(poro, dtype=torch.float32)
