"""D4RL-v2 normalized-score reference — and the guard that refuses to apply it to the wrong world.

A normalized score ``100 * (ret - ref_min) / (ref_max - ref_min)`` is a *Diffuser-comparable* number
ONLY when the raw return was earned on the same env + dataset the reference constants describe: the
original D4RL **v2** mujoco datasets, rolled out on the **v2** env. genforge has neither here — D4RL
names resolve to Farama **Minari** re-recordings on **v5** envs (``adapter_minari.MINARI_ID_MAP``),
and even the real-v2 replay buffers roll out on v5 (``adapter_d4rl.ENV_ID_MAP``). Applying v2
constants to a v5/Minari return yields a confident-looking percentage that means nothing — the exact
silent-wrong-output defect the design contract forbids. ``d4rl_normalized_score`` makes it impossible to do
silently: it RAISES unless the provenance is genuine v2, naming which half is wrong.
"""
from __future__ import annotations

from typing import Sequence

# (ref_min, ref_max): the random-policy and expert returns D4RL published for the **v2** datasets.
# 0 = random, 100 = expert — valid ONLY for a v2 return on a v2 env. Source: d4rl/infos.py.
D4RL_V2_REF: dict[str, tuple[float, float]] = {
    "hopper":      (-20.272305, 3234.3),
    "walker2d":    (1.629008, 4592.3),
    "halfcheetah": (-280.178953, 12135.0),
}


def _prefix(name: str) -> str:
    p = str(name).split("-", 1)[0].lower()
    if p not in D4RL_V2_REF:
        raise KeyError(f"no D4RL v2 reference for locomotion task {name!r}; known: {sorted(D4RL_V2_REF)}")
    return p


def _is_d4rl_v2(dataset_id: str) -> bool:
    """A genuine D4RL v2 dataset id/filename ends in ``-v2``. Minari re-recordings end ``-v0``
    (e.g. ``mujoco/hopper/medium-v0``); a path stem is checked after dropping ``.hdf5``."""
    stem = dataset_id[:-5] if dataset_id.endswith(".hdf5") else dataset_id
    return stem.endswith("-v2")


def d4rl_normalized_score(raw_return: float, *, name: str, dataset_ids: str | Sequence[str],
                          env_id: str | None = None) -> float:
    """Diffuser-comparable normalized score for ``name``, or RAISE if provenance can't support one.

    The percentage is meaningful only when ``raw_return`` was earned on the D4RL **v2** dataset+env
    the reference describes. Both mismatches genforge actually ships are refused loudly:
      - a Minari re-recording standing in for the v2 dataset (``dataset_ids`` not ``-v2``), and
      - a v5 rollout env standing in for the v2 env (``env_id`` not ``-v2``).
    Pass ``env_id=None`` to skip the env check (e.g. when the caller only knows the dataset).
    """
    prefix = _prefix(name)
    ids = [dataset_ids] if isinstance(dataset_ids, str) else list(dataset_ids)

    bad = [d for d in ids if not _is_d4rl_v2(d)]
    if bad:
        raise ValueError(
            f"{name}: refusing to D4RL-normalize a return earned on {bad} — these are not the D4RL "
            f"v2 dataset the reference (ref_max={D4RL_V2_REF[prefix][1]}) was calibrated on (Minari "
            f"re-recordings end -v0). The percentage would be inflated by construction. Report the "
            f"raw return, or install the real D4RL v2 data."
        )
    if env_id is not None and not str(env_id).endswith("-v2"):
        raise ValueError(
            f"{name}: refusing to D4RL-normalize a return rolled out on {env_id!r} — the v2 "
            f"reference assumes v2 physics, and a v5 rollout is a genuine train/eval dynamics "
            f"mismatch (the return is really lower). Point ENV_ID_MAP at the v2 env (needs "
            f"mujoco_py) before normalizing."
        )
    ref_min, ref_max = D4RL_V2_REF[prefix]
    return 100.0 * (raw_return - ref_min) / (ref_max - ref_min)
