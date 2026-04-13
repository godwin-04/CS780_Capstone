from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

# ================= Dueling DQN ================= #

class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.value = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.feature(x)

        v = self.value(x)
        a = self.advantage(x)

        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


# ================= LOAD MODEL ================= #

_model: Optional[DuelingDQN] = None
_last_action: Optional[int] = None
_repeat_count: int = 0

_MAX_REPEAT = 2
_CLOSE_Q_DELTA = 0.05


def _load_once():
    global _model
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")

    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py."
        )

    m = DuelingDQN()

    sd = torch.load(wpath, map_location="cpu")

    # handle checkpoint dict
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    m.load_state_dict(sd, strict=True)
    m.eval()

    _model = m


# ================= POLICY ================= #

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count

    _load_once()

    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    q = _model(x).squeeze(0).cpu().numpy()

    # ----- Greedy action -----
    best = int(np.argmax(q))

    # ----- Relative gap computation -----
    sorted_idx = np.argsort(-q)
    best_q = float(q[sorted_idx[0]])
    second_q = float(q[sorted_idx[1]])

    gap = best_q - second_q
    scale = abs(best_q) + 1e-6
    rel_gap = gap / scale

    # ----- Action smoothing -----
    REL_GAP_THRESHOLD = 0.1   # robust threshold
    MAX_REPEAT = 2            # prevents getting stuck

    if _last_action is not None:
        if rel_gap < REL_GAP_THRESHOLD:
            if _repeat_count < MAX_REPEAT:
                best = _last_action
                _repeat_count += 1
            else:
                _repeat_count = 0
        else:
            _repeat_count = 0

    _last_action = best

    return ACTIONS[best]