#!/usr/bin/env python3
"""TP-AQB v2 -- topology / throat preserving AQB.

    v2 = v1 (sha256 d0c507e9c09182a4d66e4409569298156014d0c9cf7a63053533c7201
    656add1) with exactly ONE change: `_delete_to_count` re-derives the
    remaining deficit instead of reusing the pre-ladder `need` at every relax
    level.  See that function's docstring for the bug and the measurement that
    exposed it.  No algorithm parameter, score weight, protection radius,
    ladder order, swap setting or tie-break differs from v1.

The only thing this module is allowed to touch is the map

    (probability volume P, target porosity)  ->  binary volume

It never re-samples, never retrains, never sees the Test set.

WHY THE OBVIOUS IMPLEMENTATION IS WRONG
---------------------------------------
naive AQB keeps the `round(phi*N)` highest-probability voxels.  Because the
model over-produces (OFF lands ~0.55pp above phi_real), that threshold sits
ABOVE 0.5, so the voxels it throws away are the marginal ones -- which are
preferentially the voxels filling pore bodies and necks.  The pore space
fragments into more, smaller pores and Kgeom collapses.

TP-AQB keeps the same preference order (among voxels that are all >= 0.5,
"closest to 0.5" is exactly "lowest probability"), but only ever deletes a
voxel that is (a) a simple point and (b) outside the protected skeleton /
throat set.

TOPOLOGY
--------
Pore phase uses 26-connectivity, solid phase 6-connectivity.  A voxel is a
simple point iff T26 == 1 and T6 == 1 where

    T26(x) = # 26-connected components of pore  in N26(x)
    T6 (x) = # 6-connected  components of solid in N18(x)

`simple_point_flags` implements exactly that, JIT-compiled.  The module does
NOT ask you to trust it: `selftest_simple_point` checks it against a global
ground truth (rebuild the whole volume without x, compare component counts of
both phases and the Euler characteristic) on random voxels of real volumes.

BATCH SAFETY
------------
Deleting a whole wave at once is only valid if the deleted voxels cannot
interact.  Two voxels at Chebyshev distance >= 2 are not in each other's N26,
so each one's local configuration is untouched by the other's removal and
simplicity is preserved simultaneously.  `_pick_wave` enforces that by
blocking the 3x3x3 around every pick.  A global topology check after each wave
is still performed, because "provably safe" is a claim about the code, and the
check is a claim about the actual volume.
"""
import numpy as np
from numba import njit
from scipy import ndimage
from skimage.morphology import skeletonize

CONN6 = ndimage.generate_binary_structure(3, 1)
CONN26 = ndimage.generate_binary_structure(3, 3)
DEFAULT = dict(
    skeleton=True, skeleton_radius=1,
    throat=False, throat_quantile=0.50, throat_radius=1,
    porebody_pref=False,
    w_prob=1.0, w_dist=0.0, w_body=0.0, w_scale=0.0, w_face=0.0,
    face_greedy=False, wave_cap=0,
    target_dS=None,        # if set, choose the deletion set to hit this total
    swap_rounds=0,         # >0 enables the local-swap repair
    swap_tol=0.0,          # stop once |dS remaining| <= this many interfaces
    swap_max_pairs=4000,
    swap_pref_deleted=False,   # prefer restoring voxels the backbone deleted
    max_waves=40, verbose=False,
)


# ======================================================================
# (26,6) simple point test
# ======================================================================
def _offsets():
    off = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz or dy or dx:
                    off.append((dz, dy, dx))
    return off


_OFF26 = _offsets()
_IDX = {o: i for i, o in enumerate(_OFF26)}
# 26-adjacency: offsets differing by at most 1 in every coordinate
_PAIRS26 = [(i, j) for i, a in enumerate(_OFF26) for j, b in enumerate(_OFF26)
            if i < j and max(abs(a[k] - b[k]) for k in range(3)) == 1]
_ADJ26 = np.zeros(26, dtype=np.int64)
for _i, _j in _PAIRS26:
    _ADJ26[_i] |= (1 << _j)
    _ADJ26[_j] |= (1 << _i)
# the 18-neighbourhood: face (manhattan 1) + edge (manhattan 2) neighbours
_OFF18 = [o for o in _OFF26 if sum(abs(c) for c in o) <= 2]
_N18 = len(_OFF18)                                   # 6 + 12 = 18
_PAIRS18 = [(i, j) for i, a in enumerate(_OFF18) for j, b in enumerate(_OFF18)
            if i < j and sum(abs(a[k] - b[k]) for k in range(3)) == 1]
_ADJ18 = np.zeros(_N18, dtype=np.int64)
for _i, _j in _PAIRS18:
    _ADJ18[_i] |= (1 << _j)
    _ADJ18[_j] |= (1 << _i)
# the six FACE neighbours, as bits in the N18 ordering.  These are the only
# positions a newly-deleted voxel can be 6-adjacent to, so they are the
# `required` mask of the background test.
_FACE18 = 0
for _i, _o in enumerate(_OFF18):
    if sum(abs(c) for c in _o) == 1:
        _FACE18 |= (1 << _i)


@njit(cache=True, inline="always")
def _ncomp(mask, adj, required):
    """Components of the set bits of `mask` that intersect `required`.

    `required` is what makes this the classical (26,6) test rather than a
    plausible-looking approximation.  For the background it must be the six
    FACE neighbours: the voxel being deleted can only be 6-adjacent to those,
    so a background component reachable only through edge neighbours does not
    absorb it -- delete such an x and the new background voxel is isolated,
    which is a topology change.  Counting every component in N18 (the obvious
    reading) accepts those voxels and silently increments n_solid_cc.
    """
    rem = mask
    count = 0
    while rem:
        seed = rem & (-rem)
        comp = seed
        frontier = seed
        while frontier:
            b = frontier & (-frontier)
            frontier ^= b
            i = 0
            t = b
            while t > 1:
                t >>= 1
                i += 1
            new = adj[i] & rem & ~comp
            comp |= new
            frontier |= new
        if comp & required:
            count += 1
        rem &= ~comp
    return count


@njit(cache=True)
def simple_and_addable(pore, off26, adj26, off18, adj18, face18):
    """(simple, addable) flags for every interior voxel.

    Both rest on the SAME two neighbourhood conditions:

        T26 = 1  # 26-connected components of pore  in N26(x)
        T6  = 1  # 6-connected components of solid in N18(x) touching a FACE

    They differ only in the gate:
        pore[x] == 1  -> deleting x is topology-preserving      (simple)
        pore[x] == 0  -> adding x to the pore is topological    (addable)

    That the conditions coincide is worth stating, because the add direction
    was first written with its own approximate rule and it was wrong: adding x
    joins x to EVERY pore voxel in N26(x), so "does this merge two components"
    must be asked over all 26 neighbours.  The earlier `_bridges` helper looked
    only at the six FACE neighbours, so a voxel with a pore face-neighbour and
    a pore diagonal neighbour in a different component merged them silently --
    one wave dropped n_pore_cc from 464 to 392 before the wave assertion caught
    it.  Asking T26 == 1 answers merging and isolation at once.

    Voxels on the volume border are reported 0 in both: their neighbourhood is
    cut off, so the safe answer is "do not touch".
    """
    sz, sy, sx = pore.shape
    out_s = np.zeros(pore.shape, dtype=np.uint8)
    out_a = np.zeros(pore.shape, dtype=np.uint8)
    for z in range(1, sz - 1):
        for y in range(1, sy - 1):
            for x in range(1, sx - 1):
                m26 = 0
                for k in range(off26.shape[0]):
                    if pore[z + off26[k, 0], y + off26[k, 1], x + off26[k, 2]]:
                        m26 |= (1 << k)
                if m26 == 0:
                    continue                     # no pore neighbour: isolated
                m18 = 0
                for k in range(off18.shape[0]):
                    if pore[z + off18[k, 0], y + off18[k, 1], x + off18[k, 2]] == 0:
                        m18 |= (1 << k)
                if _ncomp(m26, adj26, -1) != 1:
                    continue
                if _ncomp(m18, adj18, face18) != 1:
                    continue
                if pore[z, y, x]:
                    out_s[z, y, x] = 1
                else:
                    out_a[z, y, x] = 1
    return out_s, out_a


def _args(pore):
    return (np.ascontiguousarray(pore, dtype=np.uint8),
            np.ascontiguousarray(np.array(_OFF26, dtype=np.int64)),
            np.ascontiguousarray(_ADJ26),
            np.ascontiguousarray(np.array(_OFF18, dtype=np.int64)),
            np.ascontiguousarray(_ADJ18),
            np.int64(_FACE18))


def simple_points(pore):
    return simple_and_addable(*_args(pore))[0] != 0


def addable_points(pore):
    """Voxels that can be switched solid -> pore without changing topology."""
    return simple_and_addable(*_args(pore))[1] != 0


def selftest_addable(vol, n_probe=40, seed=0):
    """Check the add direction against the global ground truth.

    `n_solid_cc` is EXEMPT: adding a pore voxel removes a solid voxel, which
    may legitimately split the solid phase -- that is pore growth, not a
    topology violation.  What must never change is the PORE component count
    (no merging, no isolation) and the Euler characteristic (no new tunnels).
    """
    add = addable_points(vol)
    inner = np.zeros(vol.shape, bool); inner[1:-1, 1:-1, 1:-1] = True
    cand = np.argwhere(add & inner)
    if len(cand) == 0:
        return 0, 0
    r = np.random.default_rng(seed)
    probes = cand[r.choice(len(cand), min(n_probe, len(cand)), replace=False)]
    base = topology_signature(vol)
    bad = []
    for (z, y, x) in probes:
        q = vol.copy(); q[z, y, x] = 1
        s = topology_signature(q)
        if (s["n_pore_cc"] != base["n_pore_cc"]
                or s["euler"] != base["euler"]):
            bad.append((int(z), int(y), int(x),
                        (base["n_pore_cc"], base["euler"]),
                        (s["n_pore_cc"], s["euler"])))
    return len(probes), bad


# ======================================================================
# topology ground truth (independent of the local test)
# ======================================================================
def topology_signature(seg):
    """Global topology of a binary volume, computed without any local rule.

    All three functionals must be read in the SAME digital-topology caliber as
    the rule they audit: pore 26-connected, solid 6-connected.  That fixes the
    Euler characteristic too -- it has to be the 26-convention (closed cubes),
    not the 6-convention cubical complex.  The two disagree on exactly the
    configurations the simple-point test is about: two cubes meeting at a
    corner are one contractible object under 26 (chi=+1) and two components
    under 6 (chi=+3).  Auditing a 26-convention cell count with a 6-convention
    chi manufactures "topology changed" verdicts for deletions that changed
    nothing, so the ground truth must not mix calibers.
    """
    pore = np.ascontiguousarray(seg, dtype=bool)
    lab, n_pore = ndimage.label(pore, structure=CONN26)
    if n_pore:
        sizes = np.bincount(lab.ravel())[1:]
        lcc = float(sizes.max()) / float(pore.sum()) if pore.sum() else 0.0
    else:
        lcc = 0.0
    _, n_solid = ndimage.label(~pore, structure=CONN6)
    return {"n_pore_cc": int(n_pore), "n_solid_cc": int(n_solid),
            "lcc_frac": lcc, "euler": euler_char_26(pore)}


def euler_char_26(pore):
    """Euler characteristic in the 26-convention (closed cubes).

    Counts the cells of the cubical complex of the union of closed unit
    voxels: a k-cell belongs to the complex iff it is a face of at least one
    foreground voxel.  Two voxels meeting only at a corner share that corner
    cell, so they form one contractible object (chi=+1) -- which is what
    `n_pore_cc` (26-connected) reports, and why this, not the 6-convention
    version, is the functional that audits the simple-point rule without
    manufacturing phantom changes.

    Cell counts, on P = one-voxel zero pad (a cell belongs iff >=1 incident
    voxel exists; "before/after along the cell's own axis" is what makes the
    corner-sharing case collapse to a single cell):

        V   grid corners touching >=1 voxel          (n+1)^3
        Ex  x-edges: 2 voxels before/after in x      (n, n+1, n+1)
        Fxy xy-faces: 2 voxels before/after in z     (n, n, n+1)
        C   foreground voxels

    Cross-checked against skimage.measure.euler_number(connectivity=3) in
    selftest_topology.py layer 0, on shapes where the two conventions differ.
    """
    P = np.pad(pore.astype(np.int64), 1)

    # 0-cells: corner (i,j,k) is touched by the 8 voxels around it
    V = (P[:-1, :-1, :-1] + P[1:, :-1, :-1] + P[:-1, 1:, :-1] + P[:-1, :-1, 1:]
         + P[1:, 1:, :-1] + P[1:, :-1, 1:] + P[:-1, 1:, 1:] + P[1:, 1:, 1:]) > 0

    # 1-cells: an edge along `ax` is fixed in the other two axes and spans the
    # two voxels immediately before/after it along `ax`
    Ex = (P[1:-1, :-1, :-1] + P[1:-1, 1:, :-1] + P[1:-1, :-1, 1:] + P[1:-1, 1:, 1:]) > 0
    Ey = (P[:-1, 1:-1, :-1] + P[1:, 1:-1, :-1] + P[:-1, 1:-1, 1:] + P[1:, 1:-1, 1:]) > 0
    Ez = (P[:-1, :-1, 1:-1] + P[:-1, 1:, 1:-1] + P[1:, :-1, 1:-1] + P[1:, 1:, 1:-1]) > 0

    # 2-cells: a face spanning `ax0`x`ax1` is 1x1 in those axes and spans the
    # two voxels before/after it along the remaining axis
    Fxy = (P[1:-1, 1:-1, :-1] + P[1:-1, 1:-1, 1:]) > 0
    Fxz = (P[1:-1, :-1, 1:-1] + P[1:-1, 1:, 1:-1]) > 0
    Fyz = (P[:-1, 1:-1, 1:-1] + P[1:, 1:-1, 1:-1]) > 0

    C = int(P[1:-1, 1:-1, 1:-1].sum())
    return int(V.sum()) - int(Ex.sum() + Ey.sum() + Ez.sum()) \
        + int(Fxy.sum() + Fxz.sum() + Fyz.sum()) - C

def euler_char_6(pore):
    """Euler characteristic of the 6-connected pore complex (V-E+F-C)."""
    p = pore.astype(np.int64)
    V = int(p.sum())
    # E: 6-adjacent voxel pairs
    E = 0
    for ax in range(3):
        a = np.take(p, range(0, p.shape[ax] - 1), axis=ax)
        b = np.take(p, range(1, p.shape[ax]), axis=ax)
        E += int((a & b).sum())
    # F: 2x2 squares in each of the three coordinate planes
    F = 0
    for ax0, ax1 in ((0, 1), (0, 2), (1, 2)):
        a = np.take(p, range(0, p.shape[ax0] - 1), axis=ax0)
        a = np.take(a, range(0, p.shape[ax1] - 1), axis=ax1)
        b = np.take(p, range(1, p.shape[ax0]), axis=ax0)
        b = np.take(b, range(0, p.shape[ax1] - 1), axis=ax1)
        c = np.take(p, range(0, p.shape[ax0] - 1), axis=ax0)
        c = np.take(c, range(1, p.shape[ax1]), axis=ax1)
        d = np.take(p, range(1, p.shape[ax0]), axis=ax0)
        d = np.take(d, range(1, p.shape[ax1]), axis=ax1)
        F += int((a & b & c & d).sum())
    # C: 2x2x2 cubes
    C = 0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                s = p[dz:dz + p.shape[0] - 1, dy:dy + p.shape[1] - 1,
                      dx:dx + p.shape[2] - 1]
                cube = s if (dx == 0 and dy == 0 and dz == 0) else cube & s
    C = int(cube.sum())
    return V - E + F - C


def selftest_simple_point(volumes, n_probe=400, seed=0, verbose=True):
    """Check the local rule against the global definition, on real volumes.

    For each probed voxel, delete it from the whole volume and recompute the
    global signature.  `local says simple` must agree with `global topology
    unchanged`, in BOTH directions -- a rule that only errs on the safe side
    would silently cost us deletable voxels.
    """
    rng = np.random.default_rng(seed)
    tot = agree = local_true = global_true = 0
    bad = []
    for v in volumes:
        p = np.ascontiguousarray(v, dtype=np.uint8)
        loc = simple_points(p)
        cand = np.argwhere(p > 0)
        if len(cand) > n_probe:
            cand = cand[rng.choice(len(cand), n_probe, replace=False)]
        base = topology_signature(p)
        for (z, y, x) in cand:
            if not (0 < z < p.shape[0] - 1 and 0 < y < p.shape[1] - 1 and 0 < x < p.shape[2] - 1):
                continue
            q = p.copy(); q[z, y, x] = 0
            sig = topology_signature(q)
            g = (sig["n_pore_cc"] == base["n_pore_cc"]
                 and sig["n_solid_cc"] == base["n_solid_cc"]
                 and sig["euler"] == base["euler"])
            l = bool(loc[z, y, x])
            tot += 1
            local_true += l
            global_true += g
            if l == g:
                agree += 1
            elif len(bad) < 8:
                bad.append((tuple(int(t) for t in (z, y, x)), l, g,
                            base, sig))
    if verbose:
        print(f"  probed {tot} voxels: local==global on {agree} ({agree/max(tot,1):.4%})")
        print(f"    local says simple : {local_true}")
        print(f"    global says safe  : {global_true}")
        for b in bad:
            print(f"    DISAGREE at {b[0]}: local={b[1]} global={b[2]}")
            print(f"       before {b[3]}")
            print(f"       after  {b[4]}")
    return tot, agree, bad


# ======================================================================
# protection masks and score
# ======================================================================
def protection_mask(seg, cfg, info=None):
    """Voxels that must not be deleted: skeleton, and narrow-throat cores.

    The two branches are INDEPENDENT -- an earlier version nested the throat one
    inside `if cfg["skeleton"]`, so `throat=True, skeleton=False` silently
    protected nothing.

    Note also the geometric relation that made the throat knob inert: the
    narrow-throat set is a SUBSET of the skeleton, and dilating a subset by r
    can never exceed dilating the whole set by r.  With
    `throat_radius == skeleton_radius` the throat mask is therefore a subset of
    the skeleton mask by construction and R3 was byte-identical to R2.  The
    throat branch only adds anything when its radius EXCEEDS the skeleton's.
    `info["n_throat_only"]` reports what it actually contributed, so an inert
    knob shows up as a number instead of an assumption.
    """
    prot = np.zeros(seg.shape, dtype=bool)
    skel = None
    if cfg["skeleton"] or cfg["throat"]:
        skel = skeletonize(seg.astype(bool))
        if info is not None:
            info["n_skeleton_voxels"] = int(skel.sum())
    if cfg["skeleton"]:
        if cfg["skeleton_radius"] > 0:
            prot |= ndimage.binary_dilation(skel, structure=CONN26,
                                            iterations=int(cfg["skeleton_radius"]))
        else:
            prot |= skel

    throat_only = 0
    if cfg["throat"]:
        before = prot.copy()
        edt = ndimage.distance_transform_edt(seg)
        r_skel = edt[skel]
        if r_skel.size:
            thr = float(np.quantile(r_skel, cfg["throat_quantile"]))
            narrow = skel & (edt <= thr)
            if info is not None:
                info["n_narrow_throat_voxels"] = int(narrow.sum())
                info["throat_radius_threshold"] = thr
            if cfg["throat_radius"] > 0:
                narrow = ndimage.binary_dilation(narrow, structure=CONN26,
                                                 iterations=int(cfg["throat_radius"]))
            prot |= narrow
        throat_only = int((prot & ~before).sum())
    if info is not None:
        info["n_protected"] = int(prot.sum())
        info["n_throat_only"] = throat_only
        if cfg["throat"] and throat_only == 0:
            info["throat_inert"] = True
    return prot


def _face_solid_count(seg):
    """Number of SOLID face-neighbours of every voxel, out-of-box = not solid.

    This is the quantity that controls surface area.  `surface_area_per_volume`
    counts disagreeing face-adjacent PAIRS, so deleting a pore voxel with k
    solid face-neighbours changes the surface count by exactly (2k - 6): k new
    interfaces appear, and the (6-k) the voxel already exposed disappear.  With
    the pore count pinned exactly, S/V is therefore monotone in the k's of the
    deleted set, and k is a genuine lever on gate D rather than a proxy for one.

    Every simple point has k >= 1: the (26,6) test's T6 == 1 counts 6-connected
    solid components FACE-adjacent to x, so a voxel with no solid face
    neighbour cannot satisfy it.  k=0 voxels are common in the candidate pool
    (~70%) but are never deletable.
    """
    s = (~seg.astype(bool)).astype(np.int64)
    n = np.zeros(seg.shape, np.int64)
    n[1:, :, :] += s[:-1, :, :]; n[:-1, :, :] += s[1:, :, :]
    n[:, 1:, :] += s[:, :-1, :]; n[:, :-1, :] += s[:, 1:, :]
    n[:, :, 1:] += s[:, :, :-1]; n[:, :, :-1] += s[:, :, 1:]
    return n


def _target_order(cur, pool, need, target):
    """Pick `need` voxels from `pool` whose total surface change is closest to
    `target`, and return them ordered for deletion.

    The surface change is exactly additive, dS = 6 - 2k, so choosing the set IS
    choosing the total: the only question is which k's to take.  k = 3 costs
    nothing and is the filler of choice; the non-neutral k's are then balanced
    against each other so the running total lands on `target`.

    `target` is NOT read from any real volume.  The protocol this implements
    uses `target = 0`, i.e. "delete the required voxels without moving OFF's
    own surface" -- preserving the AQB-OFF structure the method starts from.
    A consequence worth stating: if the total really is 0 then
    |sv_TP - sv_real| == |sv_off - sv_real| exactly, so preserving OFF also
    matches REAL exactly as well as OFF does, with no REAL input anywhere.

    Works on counts per distinct contribution value (6 of them at most) rather
    than per voxel: start from the `need` smallest contributions, then raise the
    total toward `target` in +2 steps along adjacent values.  Since every
    contribution is even, an odd target leaves an unavoidable residual of 1.
    """
    k = _face_solid_count(cur)
    cvals = (6 - 2 * k)[pool]
    idx = np.argwhere(pool)
    take = int(min(need, len(idx)))
    vals, av = np.unique(cvals, return_counts=True)
    vals = [int(v) for v in vals]
    av = [int(a) for a in av]

    n_take = {}
    rem = take
    for v, a in zip(vals, av):
        t = min(a, rem)
        n_take[v] = t
        rem -= t
        if rem == 0:
            break
    acc = sum(v * t for v, t in n_take.items())

    if acc < target:
        R = target - acc
        for i in range(len(vals) - 1):
            v, v2 = vals[i], vals[i + 1]
            d = v2 - v
            if d <= 0 or R < d:
                continue
            slack2 = av[i + 1] - n_take.get(v2, 0)
            can = min(n_take.get(v, 0), slack2)
            if can <= 0:
                continue
            move = min(can, R // d)
            if move <= 0:
                continue
            n_take[v] -= move
            n_take[v2] = n_take.get(v2, 0) + move
            acc += d * move
            R -= d * move
            if R < d:
                break

    # gather the chosen voxels, most influential first so the blocking rule
    # cannot silently drop the ones the total depends on
    chosen = []
    for v in sorted(n_take, key=lambda x: -abs(x)):
        c = n_take.get(v, 0)
        if c <= 0:
            continue
        sub = idx[cvals == v][:c]
        chosen.append(sub)
    sel = (np.concatenate(chosen) if chosen else idx[:0])

    # the rest, neutral (k=3) first: if the blocking rule rejects a chosen
    # voxel, a zero-cost substitute keeps the running total where it was
    selset = set(map(tuple, sel.tolist()))
    rest = np.array([t for t in idx.tolist() if tuple(t) not in selset],
                    dtype=idx.dtype) if len(sel) else idx
    if len(rest):
        rc = (6 - 2 * k)[tuple(rest.T)]
        rest = rest[np.argsort(np.abs(rc), kind="stable")]
    out = np.concatenate([sel, rest]) if len(sel) else rest
    return out, acc, take


def surface_count(vol):
    """Interface count, byte-for-byte the definition the gates use
    (`batch_voxel_metrics.surface_area_per_volume` counts the same three
    slabs and divides by the constant box volume).  Kept here as a local
    integer so the swap loop can call it per round without importing a
    metric module into the segmentation path."""
    v = np.asarray(vol, np.uint8)
    return int((v[:-1, :, :] != v[1:, :, :]).sum()
               + (v[:, :-1, :] != v[:, 1:, :]).sum()
               + (v[:, :, :-1] != v[:, :, 1:]).sum())


def _swap_wave(cur, k, simple, addable, prot, need, max_pairs, pref_a=None):
    """One batch of non-interacting (add, delete) pairs.

    Adding voxel `a` (solid -> pore) changes the interface count by 2*k_a - 6;
    deleting voxel `d` (pore -> solid) changes it by 6 - 2*k_d.  A swap does
    both and keeps the pore count, so

        dS_swap = 2*(k_a - k_d)

    To LOWER the surface, pair low-k additions with high-k deletions; to raise
    it, the reverse.  Sorting the two pools in opposite k order and pairing them
    index-by-index puts the largest available |dS| first, and the magnitudes
    decrease down the list, so a single pass can stop as soon as the running
    total lands on `need`.

    `pref_a` is a TIE-BREAK behind the k sort: voxels the backbone itself
    deleted (OFF has them as pore, `seg` does not).  Restoring those puts back
    the pore body and throat walls the backbone removed, rather than carving
    new pore somewhere OFF never had any.  It is a restriction when it can
    supply the wave and a fallback-to-all when it cannot, so it can never make
    the wave infeasible.  It reads only the method's own two volumes.

    There is deliberately NO pore-body-wall tie-break on the delete side.  One
    was written (prefer smaller EDT among equal k) and measured: it is inert by
    construction, because every deletable voxel is a simple point, T6 == 1
    requires a FACE-adjacent solid voxel, and so the EDT of the entire
    candidate pool is identically 1.00 -- 556k candidates on the first crop, all
    at 1.00, in every one of the seven k blocks.  A tie-break with no ties to
    break is not a knob; see the ledger.

    Safety is the same argument `_pick_wave` uses for deletions: everything
    selected is blocked out to Chebyshev 2, so no chosen voxel is in any other
    chosen voxel's 26-neighbourhood and each one's local configuration is
    untouched by the rest.  `a` and `d` therefore need no special casing.
    """
    d_idx = np.argwhere(simple & cur & ~prot)
    a_all = addable & ~cur
    if pref_a is not None:
        a_pref = a_all & pref_a
        a_idx = np.argwhere(a_pref if a_pref.any() else a_all)
    else:
        a_idx = np.argwhere(a_all)
    if len(d_idx) == 0 or len(a_idx) == 0:
        return [], 0, 0
    kd = k[tuple(d_idx.T)].astype(np.int64)
    ka = k[tuple(a_idx.T)].astype(np.int64)
    sgn = -1 if need < 0 else 1

    # primary key LAST in lexsort; secondary first.  `~pref` sorts True-first.
    pa = (~pref_a[tuple(a_idx.T)]) if pref_a is not None else np.zeros(len(a_idx), bool)
    if sgn < 0:                       # lower the surface
        oa = np.lexsort((pa, ka))
        od = np.argsort(-kd, kind="stable")
    else:                             # raise it
        oa = np.lexsort((pa, -ka))
        od = np.argsort(kd, kind="stable")

    blocked = np.zeros(cur.shape, dtype=bool)
    picks = []
    acc = 0
    for i in range(min(len(oa), len(od), max_pairs * 4)):
        if len(picks) >= max_pairs:
            break
        ai, di = oa[i], od[i]
        a = a_idx[ai]; d = d_idx[di]
        contrib = 2 * int(ka[ai] - kd[di])
        if contrib == 0:
            break                      # sorted: nothing better remains
        if abs(need - (acc + contrib)) >= abs(need - acc):
            continue                   # would overshoot; a later pair is smaller
        az, ay, ax = int(a[0]), int(a[1]), int(a[2])
        dz, dy, dx = int(d[0]), int(d[1]), int(d[2])
        if blocked[az, ay, ax] or blocked[dz, dy, dx]:
            continue
        if max(abs(az - dz), abs(ay - dy), abs(ax - dx)) < 2:
            continue                   # the pair would interact with itself
        picks.append((a, d, contrib))
        acc += contrib
        for (p, q, r) in ((az, ay, ax), (dz, dy, dx)):
            blocked[max(0, p - 1):p + 2, max(0, q - 1):q + 2,
                    max(0, r - 1):r + 2] = True
    return picks, acc, len(d_idx) + len(a_idx)


def swap_repair(prob, seg, sv_off, cfg=None, info=None):
    """Greedy local-swap repair toward AQB-OFF's own surface density.

    `sv_off` is measured from the OFF structure derived from P (P >= 0.5).  No
    real crop is read: the target is a property of the method's own starting
    point, not of any ground truth.

    Each round revalidates globally before committing, and stops when either
    (A) |sv - sv_off| is inside tolerance or (B) no further swap improves it.
    """
    c = dict(DEFAULT); c.update(cfg or {})
    info = {} if info is None else info
    N = seg.size
    cur = seg.astype(bool).copy()
    n0 = int(cur.sum())
    before = topology_signature(cur)
    sv = surface_count(cur) / N
    target_dS = (sv_off - sv) * N
    total = 0
    # The add-back preference is "voxels the backbone threw away": OFF has them
    # as pore, `seg` does not.  OFF here is P >= 0.5, derived from the same
    # probability volume the method is already given, so this reads nothing
    # outside the allowed input set.
    pref_a = (prob >= 0.5) & ~cur if c["swap_pref_deleted"] else None
    hist = [dict(round=0, sv=sv, drift=abs(sv - sv_off), swaps=0)]
    for it in range(c["swap_rounds"]):
        if abs(target_dS - total) <= c["swap_tol"]:
            break
        k = _face_solid_count(cur)
        simple = simple_points(cur)
        addable = addable_points(cur)
        prot = protection_mask(cur, c, None)
        picks, acc, npool = _swap_wave(cur, k, simple, addable, prot,
                                       target_dS - total, c["swap_max_pairs"],
                                       pref_a=pref_a)
        if not picks:
            info["swap_stop"] = "no improving pair"
            break
        for (a, d, contrib) in picks:
            cur[tuple(a)] = True
            cur[tuple(d)] = False
        if int(cur.sum()) != n0:
            raise AssertionError(
                f"swap repair changed the pore count: {n0} -> {int(cur.sum())}")
        now = topology_signature(cur)
        if (now["n_pore_cc"] != before["n_pore_cc"]
                or now["n_solid_cc"] != before["n_solid_cc"]
                or now["euler"] != before["euler"]):
            raise AssertionError(
                f"swap wave {it} changed global topology: {before} -> {now}")
        total += acc
        sv = surface_count(cur) / N
        hist.append(dict(round=it + 1, sv=sv, drift=abs(sv - sv_off),
                         swaps=len(picks)))
    info["swap_rounds"] = len(hist) - 1
    info["swap_sv_before"] = hist[0]["sv"]
    info["swap_sv_after"] = hist[-1]["sv"]
    info["swap_drift_before"] = hist[0]["drift"]
    info["swap_drift_after"] = hist[-1]["drift"]
    info["swap_hist"] = hist
    return cur.astype(np.uint8)


def safety_score(prob, seg, prot, cfg):
    """Higher = safer to delete.  Components A/B/C/D of the task spec."""
    cand = seg.astype(bool) & ~prot
    s = cfg["w_prob"] * _unit(-np.abs(prob - 0.5), cand)
    if cfg.get("w_face", 0.0) > 0:
        # Prefer deleting HIGH-k voxels.  The sign is the opposite of the
        # obvious guess and was settled by measurement, not algebra: deleting
        # x costs the k interfaces it had as a pore voxel (one per SOLID face
        # neighbour) and gains the (6-k) it has as a solid voxel (one per PORE
        # face neighbour), so dS = (6-k) - k = 6 - 2k.  Verified exactly on all
        # seven k in selftest_face.py.  Low-k (convex corner) deletions
        # therefore ADD surface; only k >= 4 shrinks it.
        s = s + cfg["w_face"] * _unit(_face_solid_count(seg), cand)
    if cfg["w_dist"] > 0:
        d = ndimage.distance_transform_edt(~prot)
        s = s + cfg["w_dist"] * _unit(d, cand)
    if cfg["w_body"] > 0 or cfg["w_scale"] > 0:
        edt = ndimage.distance_transform_edt(seg)
        if cfg["w_body"] > 0:
            med = float(np.median(edt[cand])) if cand.any() else 1.0
            s = s + cfg["w_body"] * (edt >= med)
        if cfg["w_scale"] > 0:
            s = s + cfg["w_scale"] * _unit(edt, cand)
    return s


def _unit(a, mask):
    """Scale `a` to [0,1] over `mask` so the weights are comparable."""
    a = np.asarray(a, dtype=np.float64)
    if not mask.any():
        return np.zeros_like(a)
    v = a[mask]
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _pick_wave(order, blocked, simple, n_want):
    """Greedily take voxels that are simple, unblocked, and pairwise
    Chebyshev-distant >= 2 -- so they cannot interact when removed together."""
    picks = []
    for (z, y, x) in order:
        if len(picks) >= n_want:
            break
        if blocked[z, y, x] or not simple[z, y, x]:
            continue
        picks.append((z, y, x))
        blocked[max(z - 1, 0):z + 2, max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = True
    return picks


# ======================================================================
# the driver
# ======================================================================
def tp_aqb_segment(prob, phi_target, cfg=None, info=None):
    """Exact-count binary volume from P, deleting only topology-safe voxels."""
    c = dict(DEFAULT); c.update(cfg or {})
    # A flag whose weight is left at zero is a decorative knob: `porebody_pref`
    # gates a score term that `safety_score` only adds `if w_body > 0`, so
    # setting the flag alone changed nothing.  Give it its weight here, in one
    # place, so the flag and the term cannot drift apart again.
    if c["porebody_pref"] and c["w_body"] <= 0:
        c["w_body"] = 1.0
    info = {} if info is None else info
    prob = np.ascontiguousarray(prob, dtype=np.float32)
    N = prob.size
    N_target = int(np.clip(int(round(phi_target * N)), 0, N))
    base = (prob >= 0.5)
    cur = base.copy()
    n0 = int(cur.sum())

    info.update(N=N, N_target=N_target, n_off=n0, mode=None,
                relax_level=0, waves=0, infeasible=False)

    if n0 == N_target:
        info["mode"] = "none"
        return cur.astype(np.uint8)

    if n0 < N_target:
        info["mode"] = "add"
        _add_to_count(prob, cur, N_target, c, info)
    else:
        info["mode"] = "delete"
        _delete_to_count(prob, base, cur, N_target, c, info)

    info["n_final"] = int(cur.sum())
    info["abs_err_voxels"] = abs(info["n_final"] - N_target)
    return cur.astype(np.uint8)


def _delete_to_count(prob, base, cur, N_target, c, info):
    """Relaxation ladder of section 8 -- never the simple-point constraint.

    v2 FIX -- the only behavioural change from v1.  v1 computed

        need = cur.sum() - N_target

    ONCE, before the ladder, and then handed every relax level that same full
    `need`: `got` was reset to 0 at each level, and the only termination test
    was the per-level counter `got >= need`, never `cur.sum() <= N_target`.
    Each level was therefore granted the whole original deficit again, so when
    level 0 fell short the levels below it deleted `need` MORE voxels on top of
    what was already gone.  Measured on the Fontainebleau low-phi targets: the
    ladder removed 2,868,030 voxels where 2,335,062 were required, landing
    532,968 voxels BELOW target (phi 0.0853 -> 0.0534) while `info["infeasible"]`
    simultaneously reported "could not delete enough" and `info["shortfall"]`
    was NEGATIVE -- the code's own two self-reported quantities contradicting
    each other.  In-distribution crops never reached it: there `P >= 0.5`
    already sits at phi_target, `need` is small, level 0 always succeeds, and
    the ladder is never entered.  That is why the frozen 64-crop validation
    passed 64/64 while the same code over-deleted by 18.6% on a low-phi target.

    v2 re-derives the deficit from the ACTUAL current count before every relax
    level and after every wave, never deletes more than that deficit
    (`want = min(remaining, number_of_safe_candidates)`), and refuses to let
    the count pass below N_target.  At level 0 `need - got == cur.sum() -
    N_target` by construction, so the fix is inert wherever v1 was already
    correct; it only changes the outcome where v1 was wrong.
    """
    c = dict(c)
    info["level_trace"] = []
    for level in range(4):
        info["relax_level"] = level
        remaining0 = int(cur.sum()) - N_target
        if remaining0 == 0:
            info["level_trace"].append(dict(level=level, remaining_at_start=0,
                                            remaining_at_end=0, deleted=0))
            break
        if remaining0 < 0:
            raise AssertionError(
                f"already below target on entering level {level}: "
                f"count {int(cur.sum())} < N_target {N_target}")
        prot = protection_mask(cur, c, info if level == 0 else None)
        pool = cur.astype(bool) & ~prot
        if not pool.any():
            _relax(c, level)
            continue
        if c.get("target_dS") is not None and not c.get("face_greedy", False):
            # choose WHAT to delete by its surface contribution rather than by
            # a preference score.  Pool must be restricted to simple points
            # here, because the contribution total is only meaningful over
            # voxels that can actually be deleted.
            simple0 = simple_points(cur)
            pool_t = pool & simple0
            if pool_t.any():
                order, acc, take = _target_order(cur, pool_t, remaining0,
                                                 c["target_dS"])
                info["target_dS"] = c["target_dS"]
                info["target_dS_planned"] = int(acc)
                info["target_dS_n"] = int(take)
            else:
                idx = np.argwhere(pool)
                sc = safety_score(prob, cur, prot, c)
                order = idx[np.argsort(-sc[pool], kind="stable")]
        else:
            sc = safety_score(prob, cur, prot, c)
            # candidates ordered by safety, most-deletable first
            idx = np.argwhere(pool)
            order = idx[np.argsort(-sc[pool], kind="stable")]
        for w in range(c["max_waves"]):
            remaining = int(cur.sum()) - N_target
            if remaining <= 0:
                break
            if c.get("face_greedy", False) and w > 0:
                # Re-rank against the CURRENT volume.  Needed because
                # `_pick_wave` blocks a 3x3x3 around every pick (that is what
                # makes batch deletion safe), and high-k voxels cluster in
                # crevices -- so a static order spends wave 1 having its
                # neighbours blocked and then walks down a stale list, taking
                # low-k voxels it would not have chosen.  Measured: on
                # vc_x006_y070_z544 a static max-k order reached dS=+14,765
                # where the k-supply bound allows -27,020.
                prot_ = protection_mask(cur, c, None)
                pool = cur.astype(bool) & ~prot_
                if not pool.any():
                    break
                sc = safety_score(prob, cur, prot_, c)
                idx = np.argwhere(pool)
                order = idx[np.argsort(-sc[pool], kind="stable")]
            simple = simple_points(cur)
            blocked = np.zeros(cur.shape, dtype=bool)
            # section A2: never delete more than what is still missing.
            # `n_safe` is an upper bound on what `_pick_wave` can return, so
            # this cap is a no-op against v1 whenever remaining > n_safe --
            # but it makes the quota explicit instead of implicit.
            n_safe = int((simple & pool).sum())
            want = min(remaining, n_safe)
            if c.get("wave_cap", 0) > 0:
                want = min(want, c["wave_cap"])
            picks = _pick_wave([tuple(t) for t in order], blocked, simple,
                               want)
            if not picks:
                break
            before = topology_signature(cur)
            for (z, y, x) in picks:
                cur[z, y, x] = 0
            after = topology_signature(cur)
            if (after["n_pore_cc"] != before["n_pore_cc"]
                    or after["n_solid_cc"] != before["n_solid_cc"]
                    or after["euler"] != before["euler"]):
                raise AssertionError(
                    f"wave {w} changed global topology despite the packing "
                    f"rule: {before} -> {after}")
            # section A3's hard invariant.  A `raise`, not an `assert`, so it
            # survives `python -O`.
            n_now = int(cur.sum())
            if n_now < N_target:
                raise AssertionError(
                    f"deletion wave {w} over-deleted: {n_now} < {N_target}")
            info["waves"] += 1
        n_end = int(cur.sum())
        rem_end = n_end - N_target
        info["level_trace"].append(dict(level=level,
                                        remaining_at_start=remaining0,
                                        remaining_at_end=rem_end,
                                        deleted=remaining0 - rem_end))
        if rem_end == 0:
            break                       # exact -- done
        # this protection level could not supply the REST of the deficit:
        # ladder down, carrying the true remaining deficit, not a stale one
        _relax(c, level)
        info["relax_level"] = level + 1
    n_final = int(cur.sum())
    if n_final == N_target:
        info["infeasible"] = False
        info["shortfall"] = 0
    else:
        # n_final > N_target.  Every level was exhausted and the deletable
        # pool still could not supply the remaining deficit.  This is the only
        # legal "infeasible" state and its shortfall is positive by
        # construction -- a negative shortfall is no longer representable.
        info["infeasible"] = True
        info["shortfall"] = n_final - N_target


def _relax(c, level):
    """Section 8: L1 drop pore-body preference, L2 shrink skeleton dilation,
    L3 shrink throat protection.  The simple-point constraint is never
    relaxed, and naive AQB is never substituted."""
    if level == 0:
        c["porebody_pref"] = False
        c["w_body"] = 0.0
        c["w_scale"] = 0.0
    elif level == 1:
        c["skeleton_radius"] = max(int(c["skeleton_radius"]) - 1, 0)
    elif level == 2:
        c["throat_radius"] = max(int(c["throat_radius"]) - 1, 0)
    elif level >= 3:
        c["throat"] = False
        c["skeleton_radius"] = 0


def _add_to_count(prob, cur, N_target, c, info):
    """Section 7: only when OFF under-produces.  Grow pore bodies, never
    bridge two existing pore components and never open a shortcut.

    `addable_points` answers both at once and over all 26 neighbours -- see its
    docstring for why the six-face version of this test merged components.
    """
    need = N_target - int(cur.sum())
    got = 0
    for w in range(c["max_waves"]):
        if got >= need:
            break
        can = addable_points(cur) & ~cur.astype(bool)
        if not can.any():
            break
        sc = _unit(prob, can)         # highest probability first
        idx = np.argwhere(can)
        order = idx[np.argsort(-sc[can], kind="stable")]
        picks = []
        blocked = np.zeros(cur.shape, dtype=bool)
        for (z, y, x) in order:
            if len(picks) >= need - got:
                break
            if blocked[z, y, x]:
                continue
            picks.append((z, y, x))
            blocked[max(z - 1, 0):z + 2, max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = True
        if not picks:
            break
        before = topology_signature(cur)
        for (z, y, x) in picks:
            cur[z, y, x] = 1
        after = topology_signature(cur)
        # n_solid_cc is exempt, by design: adding a pore voxel removes a solid
        # voxel and so may split the solid phase.  That is pore growth, not a
        # topology violation.  Pore components and Euler must not move.
        if (after["n_pore_cc"] != before["n_pore_cc"]
                or after["euler"] != before["euler"]):
            raise AssertionError(f"add wave {w} changed topology: {before} -> {after}")
        got += len(picks)
        info["waves"] += 1
    if got < need:
        info["infeasible"] = True
        info["shortfall"] = need - got
