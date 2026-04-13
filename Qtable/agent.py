import numpy as np
import pickle
import os 

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

_Q = None
def compress_obs(obs):
    left     = int(any(obs[0:4]))
    fwd_far  = int(any(obs[4:12:2]))
    fwd_near = int(any(obs[5:12:2]))
    right    = int(any(obs[12:16]))
    ir       = int(obs[16])
    stuck    = int(obs[17])
    imbalance = int(sum(obs[0:4]) > sum(obs[12:16]))
    return (left, fwd_far, fwd_near, right, ir, stuck, imbalance)


def _load():
    global _Q
    if _Q is not None:
        return

    path = os.path.join(os.path.dirname(__file__), "qtable.pkl")
    with open(path, "rb") as f:
        _Q = pickle.load(f)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load()

    s = compress_obs(obs)

    if s in _Q:
        return ACTIONS[int(np.argmax(_Q[s]))]

    # ===== fallback (important!) =====
    if obs[16]:  # IR
        return "FW"

    if obs[17]:  # stuck
        return "L45" if rng.random() < 0.5 else "R45"

    return "FW"