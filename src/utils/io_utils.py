import json
import numpy as np


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npy(path):
    return np.load(path)


def save_npy(path, array):
    np.save(path, array)
