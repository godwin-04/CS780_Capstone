import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None

# =========================================================
# MEMORY (NO ENV DEPENDENCY)
# =========================================================
class SimpleMemory:
    def __init__(self, grid=50):
        self.grid = grid
        self.visited = np.zeros((grid, grid), dtype=np.float32)

    def update(self, obs):
        # approximate position proxy from sensors
        # (robust fallback since env is NOT available in submission)
        x = np.mean(obs[:8])
        y = np.mean(obs[8:16])

        gx = int(np.clip(x * self.grid, 0, self.grid - 1))
        gy = int(np.clip(y * self.grid, 0, self.grid - 1))

        self.visited[gx, gy] += 1
        return gx, gy

    def density(self, gx, gy):
        x1, x2 = max(0, gx - 2), min(self.grid, gx + 3)
        y1, y2 = max(0, gy - 2), min(self.grid, gy + 3)
        return np.mean(self.visited[x1:x2, y1:y2])

    def bias(self, gx, gy):
        def safe(x, y):
            if 0 <= x < self.grid and 0 <= y < self.grid:
                return self.visited[x, y]
            return 10.0

        return np.array([
            safe(gx + 2, gy),
            safe(gx, gy + 2),
            safe(gx, gy - 2)
        ], dtype=np.float32)


_memory = SimpleMemory()
_prev_stuck = 0
_mode = "EXPLORE"
_no_progress = 0


# =========================================================
# HEURISTIC (ROBUST + STABLE)
# =========================================================
def heuristic_logits(obs):
    logits = np.zeros(5, dtype=np.float32)

    ir = bool(obs[16])
    stuck = bool(obs[17])

    left = np.sum(obs[0:4])
    right = np.sum(obs[12:16])
    front = np.sum(obs[4:12])

    if ir:
        logits[2] += 3.0  # FW

    if front > 0:
        logits[2] += 2.0

    if left > right:
        logits[1] += 1.2
    elif right > left:
        logits[3] += 1.2

    if not np.any(obs[:17]):
        logits[2] += 2.0

    if stuck:
        logits[0] += 1.0
        logits[4] += 1.0
        logits[1] += 1.5
        logits[3] += 1.5

    return logits


# =========================================================
# MODEL (MATCH TRAINING = 25 DIM)
# =========================================================
def _load():
    global _MODEL
    if _MODEL is not None:
        return

    class ActorCritic(nn.Module):
        def __init__(self, state_dim=25):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
            self.actor = nn.Linear(128, 5)
            self.critic = nn.Linear(128, 1)

        def forward(self, x):
            x = self.shared(x)
            return self.actor(x), self.critic(x)

    model = ActorCritic(state_dim=25)

    path = os.path.join(os.path.dirname(__file__), "final_model.pth")
    state_dict = torch.load(path, map_location="cpu")

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _MODEL = model


# =========================================================
# FEATURE BUILDER (TRAIN MATCHED 25-DIM)
# =========================================================
def build_features(obs):
    gx, gy = _memory.update(obs)

    density = _memory.density(gx, gy)
    bias = _memory.bias(gx, gy)

    # fake pose proxies (since env not available)
    x = np.mean(obs[:8])
    y = np.mean(obs[8:16])
    ang = 0.0

    return np.concatenate([
        obs,
        [x, y, ang],
        [density],
        bias
    ]).astype(np.float32)


# =========================================================
# STATE MACHINE
# =========================================================
def update_state(obs):
    global _mode, _no_progress, _prev_stuck

    stuck = obs[17]

    if stuck:
        _mode = "ESCAPE"
        _no_progress = 0

    if np.sum(obs[:17]) > 3:
        _mode = "PUSH"

    if _no_progress > 15:
        _mode = "EXPLORE"

    if stuck:
        _prev_stuck += 1
    else:
        _prev_stuck = max(0, _prev_stuck - 1)

    if _prev_stuck > 5:
        _mode = "ESCAPE"


# =========================================================
# ACTION MODES
# =========================================================
def escape_action(rng):
    return ACTIONS[int(rng.choice([0, 1, 3, 4]))]


def push_action(model_logits, h_logits, rng):
    logits = model_logits + 0.6 * h_logits
    logits[2] += 2.0
    return ACTIONS[int(np.argmax(logits))]


def explore_action(model_logits, h_logits, rng):
    logits = model_logits + 0.9 * h_logits
    if rng.random() < 0.07:
        return ACTIONS[int(rng.integers(0, 5))]
    return ACTIONS[int(np.argmax(logits))]


# =========================================================
# FINAL POLICY
# =========================================================
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load()
    update_state(obs)

    # HARD SAFETY
    if obs[17]:
        return escape_action(rng)

    try:
        feat = build_features(obs)
        state = torch.from_numpy(feat).unsqueeze(0)

        with torch.no_grad():
            model_logits = _MODEL(state).squeeze(0).numpy()

    except:
        # fallback safe policy
        model_logits = np.zeros(5, dtype=np.float32)

    h_logits = heuristic_logits(obs)

    # adaptive blending
    if _mode == "EXPLORE":
        alpha = 0.9
    elif _mode == "PUSH":
        alpha = 0.6
    else:
        alpha = 0.3

    combined = model_logits + alpha * h_logits

    if _mode == "ESCAPE":
        return escape_action(rng)

    if _mode == "PUSH":
        return push_action(model_logits, h_logits, rng)

    return explore_action(model_logits, h_logits, rng)