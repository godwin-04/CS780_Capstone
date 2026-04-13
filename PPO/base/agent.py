
import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None  # stores the loaded model


def _load_once():
    """Load the trained PPO model and weights."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights.pth")

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

            # MUST match training exactly
            self.actor = nn.Sequential(
                nn.Linear(128, 5)
            )

            self.critic = nn.Sequential(
                nn.Linear(128, 1)
            )

        def forward(self, x):
            x = self.shared(x)
            logits = self.actor(x)
            return logits

    model = ActorCritic()

    # ✅ ignore critic mismatch safely
    model.load_state_dict(torch.load(wpath, map_location="cpu"), strict=False)

    model.eval()
    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Use the trained PPO model to choose the best action."""
    _load_once()

    import torch

    # ===== Anti-stuck safety (CRITICAL for leaderboard) =====
    if obs[17]:  # stuck flag
        return ACTIONS[int(rng.choice([0, 4]))]

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits = _MODEL(x).squeeze(0).numpy()

    return ACTIONS[int(np.argmax(logits))]