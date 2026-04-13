import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None


# =========================================================
# HEURISTIC LOGITS (same as training)
# =========================================================
def heuristic_logits(obs):
    logits = np.zeros(5, dtype=np.float32)

    stuck = bool(obs[17])
    ir = bool(obs[16])

    fwd_near = any(obs[5:12:2])
    fwd_far = any(obs[4:12:2])

    left_sig = sum(obs[0:4])
    right_sig = sum(obs[12:16])
    any_sig = any(obs[:17])

    if ir:
        logits[2] += 3.0  # FW

    if left_sig >= 3:
        logits[3] += 2.5  # R22
    if right_sig >= 3:
        logits[1] += 2.5  # L22

    if fwd_near or fwd_far:
        logits[2] += 2.0

    if left_sig > right_sig:
        logits[1] += 1.0
    elif right_sig > left_sig:
        logits[3] += 1.0

    if not any_sig:
        logits[2] += 3.0

    return logits


# =========================================================
# LOAD MODEL ONCE
# =========================================================
def _load():
    global _MODEL
    if _MODEL is not None:
        return

    import torch
    import torch.nn as nn

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(18, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
            self.actor = nn.Linear(128, 5)

        def forward(self, x):
            x = self.shared(x)
            return self.actor(x)

    model = ActorCritic()

    submission_dir = os.path.dirname(__file__)
    path = os.path.join(submission_dir, "hybrid_ppo.pth")

    model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    model.eval()

    _MODEL = model


# =========================================================
# FINAL POLICY
# =========================================================
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load()

    import torch

    # ===== HARD SAFETY (critical for leaderboard) =====
    if obs[17]:  # stuck
        return ACTIONS[int(rng.choice([0, 4]))]

    state = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits = _MODEL(state).squeeze(0)

    # ===== HYBRID COMBINATION =====
    h_logits = torch.tensor(heuristic_logits(obs))

    # lower alpha at inference (important)
    alpha = 0.8

    final_logits = logits + alpha * h_logits

    # ===== SMALL RANDOMNESS TO PREVENT DEADLOCKS =====
    if rng.random() < 0.05:
        return ACTIONS[int(rng.integers(0, 5))]

    return ACTIONS[int(torch.argmax(final_logits).item())]