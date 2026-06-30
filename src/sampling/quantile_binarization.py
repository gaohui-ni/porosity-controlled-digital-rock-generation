import numpy as np


def quantile_binarize(prob_np: np.ndarray, target_porosity: float, seed: int = 0):
    """
    Quantile-based binarization used in sample_one().

    This function is extracted from the original training script without changing
    the core algorithm:
    - find the adaptive threshold by np.partition;
    - assign voxels above threshold to pore phase;
    - randomly fill threshold-tie voxels to match target pore count.
    """
    np.random.seed(seed)

    flat = prob_np.reshape(-1)
    N = flat.size

    target_p = float(target_porosity)
    target_cnt = int(round(target_p * N))

    k = N - target_cnt
    k = np.clip(k, 0, N - 1)

    thr = np.partition(flat, k)[k]

    seg = (prob_np > thr).astype(np.uint8)

    cur_cnt = int(seg.sum())
    need = target_cnt - cur_cnt
    if need > 0:
        eq = (prob_np == thr).reshape(-1)
        idx = np.flatnonzero(eq)
        if idx.size > 0:
            np.random.shuffle(idx)
            seg.reshape(-1)[idx[:need]] = 1
    seg_poro = float(seg.mean())

    return seg, float(thr), seg_poro
