import numpy as np

from src.sampling.tp_aqb import topology_signature, tp_aqb_segment


def test_tp_aqb_returns_exact_pore_count_without_topology_change():
    rng = np.random.default_rng(20260928)
    prob = rng.random((11, 11, 11), dtype=np.float32)
    base = prob >= 0.5
    target_count = int(base.sum()) - 12
    target = target_count / prob.size
    before = topology_signature(base)
    info = {}

    seg = tp_aqb_segment(
        prob,
        target,
        cfg={
            "skeleton": False,
            "throat": False,
            "max_waves": 40,
        },
        info=info,
    )

    assert int(seg.sum()) == target_count
    assert info["infeasible"] is False
    assert info["shortfall"] == 0
    assert topology_signature(seg) == before


def test_tp_aqb_noop_when_threshold_count_is_already_exact():
    prob = np.zeros((7, 7, 7), dtype=np.float32)
    prob[2:5, 2:5, 2:5] = 0.9
    target = float((prob >= 0.5).mean())
    info = {}

    seg = tp_aqb_segment(prob, target, info=info)

    assert np.array_equal(seg, prob >= 0.5)
    assert info["mode"] == "none"
