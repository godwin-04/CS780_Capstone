import numpy as np
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    if rng.random() < 0.1:
        return "FW"
    else:
        action = rng.choice([0,1, 3,4])
        return ACTIONS[action]