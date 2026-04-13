import numpy as np

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# =========================================================
# GLOBAL STATE
# =========================================================
_last_action = 2
_turn_streak = 0
_forward_streak = 0

# Escape
_escape_mode = False
_escape_dir = 1
_escape_step = 0

# Push detection
_ir_streak = 0

# Alignment
_align_steps = 0
_align_dir = 0

# Position tracking
_position = np.array([0.0, 0.0])
_angle = 0.0

# Memory
_visited = {}
_bad_edges = set()

# Debug trace
_trace = []

# =========================================================
# UTILS
# =========================================================
def update_position(action):
    global _position, _angle

    if action == 0:
        _angle += 45
    elif action == 4:
        _angle -= 45
    elif action == 1:
        _angle += 22.5
    elif action == 3:
        _angle -= 22.5
    elif action == 2:
        rad = np.deg2rad(_angle)
        _position += np.array([np.cos(rad), np.sin(rad)])

def grid(pos):
    return tuple(np.round(pos / 3).astype(int))

def dir_bin(angle):
    return int((angle % 360) // 45)

def log_step(obs, action):
    _trace.append({
        "pos": _position.copy(),
        "angle": _angle,
        "action": action,
        "ir": obs[16],
        "stuck": obs[17]
    })

# =========================================================
# POLICY
# =========================================================
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _turn_streak, _forward_streak
    global _escape_mode, _escape_dir, _escape_step
    global _ir_streak, _align_steps, _align_dir
    global _visited, _bad_edges

    ir = obs[16]
    stuck = obs[17]

    right = sum(obs[0:4])
    front = sum(obs[4:12])
    left = sum(obs[12:16])

    pos_key = grid(_position)
    _visited[pos_key] = _visited.get(pos_key, 0) + 1

    # =====================================================
    # 1. STUCK HANDLING (LEARN + ESCAPE)
    # =====================================================
    if stuck:
        _bad_edges.add((pos_key, dir_bin(_angle)))

        if not _escape_mode:
            _escape_mode = True
            _escape_dir = 1  # fixed → no oscillation
            _escape_step = 0

        seq = [4,2,4,2,4,2,4,2] if _escape_dir == 1 else [0,2,0,2,0,2,0,2]

        action = seq[_escape_step % 8]
        _escape_step += 1

        update_position(action)
        log_step(obs, action)
        return ACTIONS[action]

    else:
        _escape_mode = False

    # =====================================================
    # 2. PUSH PHASE (LOCK)
    # =====================================================
    if ir:
        _ir_streak += 1
    else:
        _ir_streak = 0

    if _ir_streak >= 2:
        action = 2
        update_position(action)
        log_step(obs, action)
        return "FW"

    # =====================================================
    # 3. ZERO SIGNAL → PURE FORWARD
    # =====================================================
    if obs.sum() == 0:
        action = 2
        update_position(action)
        log_step(obs, action)
        return "FW"

    # =====================================================
    # 4. ALIGNMENT (SAFE CHECK)
    # =====================================================
    if _align_steps > 0:
        _align_steps -= 1
        action = 4 if _align_dir == 1 else 0

        update_position(action)
        log_step(obs, action)
        return ACTIONS[action]

    if left > right and left > 0:
        test_angle = _angle - 90
        if (pos_key, dir_bin(test_angle)) not in _bad_edges:
            _align_steps = 2
            _align_dir = -1
            action = 0
        else:
            action = 2

    elif right > left and right > 0:
        test_angle = _angle + 90
        if (pos_key, dir_bin(test_angle)) not in _bad_edges:
            _align_steps = 2
            _align_dir = 1
            action = 4
        else:
            action = 2

    elif front > 0:
        action = 2

    else:
        action = 2

    # =====================================================
    # 5. BAD EDGE AVOIDANCE
    # =====================================================
    next_edge = (pos_key, dir_bin(_angle))
    if next_edge in _bad_edges and not ir:
        action = 4  # turn away

    # =====================================================
    # 6. ANTI-SPIN
    # =====================================================
    if action != 2:
        _turn_streak += 1
        _forward_streak = 0
    else:
        _forward_streak += 1
        _turn_streak = 0

    if _turn_streak > 3:
        action = 2
        _turn_streak = 0

    # =====================================================
    # 7. UPDATE
    # =====================================================
    update_position(action)
    log_step(obs, action)
    _last_action = action

    return ACTIONS[action]