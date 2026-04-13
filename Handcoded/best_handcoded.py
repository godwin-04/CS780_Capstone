import numpy as np

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# =========================================================
# GLOBAL STATE
# =========================================================
_last_action = 2
_turn_streak = 0

# Escape
_escape_mode = False
_escape_dir = 1
_escape_step = 0
_escape_cooldown = 0

# Wall understanding
_wall_mode = False
_wall_dir = None

# Push detection
_ir_streak = 0

# Target commitment (NEW)
_target_mode = False
_target_dir = None
_target_steps = 0

# Position tracking
_position = np.array([0.0, 0.0])
_angle = 0.0

# Memory
_visited = {}
_bad_edges = set()

# Trace
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
    global _escape_mode, _escape_dir, _escape_step, _escape_cooldown
    global _ir_streak
    global _visited, _bad_edges
    global _wall_mode, _wall_dir
    global _target_mode, _target_dir, _target_steps

    ir = obs[16]
    stuck = obs[17]

    right = sum(obs[0:4])
    front = sum(obs[4:12])
    left = sum(obs[12:16])

    pos_key = grid(_position)
    _visited[pos_key] = _visited.get(pos_key, 0) + 1

    # =====================================================
    # 1. STUCK HANDLING
    # =====================================================
    if stuck:
        d = dir_bin(_angle)

        for dd in [d-1, d, d+1]:
            _bad_edges.add((pos_key, dd % 8))

        _wall_mode = True
        _wall_dir = d

        _target_mode = False  # break commitment

        if not _escape_mode:
            _escape_mode = True
            _escape_dir = 1
            _escape_step = 0

        _escape_cooldown = 6

        seq = [4,2,4,2,4,2,4,2]
        action = seq[_escape_step % 8]
        _escape_step += 1

        update_position(action)
        log_step(obs, action)
        return ACTIONS[action]

    else:
        _escape_mode = False

    # =====================================================
    # 2. ESCAPE COOLDOWN
    # =====================================================
    if _escape_cooldown > 0:
        _escape_cooldown -= 1
        action = 2
        update_position(action)
        log_step(obs, action)
        return "FW"

    # =====================================================
    # 3. PUSH PHASE
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
    # 4. TARGET COMMIT MODE (NEW CORE)
    # =====================================================
    if _target_mode:
        next_edge = (pos_key, dir_bin(_angle))

        # break if dangerous
        if next_edge in _bad_edges or obs.sum() == 0:
            _target_mode = False
        else:
            _target_steps += 1
            action = 2
            update_position(action)
            log_step(obs, action)
            return "FW"

    # =====================================================
    # 5. ZERO SIGNAL
    # =====================================================
    if obs.sum() == 0:
        action = 2
        update_position(action)
        log_step(obs, action)
        return "FW"

    # =====================================================
    # 6. ENTER TARGET MODE
    # =====================================================
    if front > 0:
        target_dir = dir_bin(_angle)

        if (pos_key, target_dir) not in _bad_edges:
            _target_mode = True
            _target_dir = target_dir
            _target_steps = 0

            action = 2
            update_position(action)
            log_step(obs, action)
            return "FW"

    elif left > right and left > 0:
        target_dir = dir_bin(_angle - 90)

        if ((pos_key, target_dir) not in _bad_edges and
            (not _wall_mode or target_dir != _wall_dir)):

            _target_mode = True
            _target_dir = target_dir
            _target_steps = 0

            action = 0
            update_position(action)
            log_step(obs, action)
            return ACTIONS[action]

    elif right > left and right > 0:
        target_dir = dir_bin(_angle + 90)

        if ((pos_key, target_dir) not in _bad_edges and
            (not _wall_mode or target_dir != _wall_dir)):

            _target_mode = True
            _target_dir = target_dir
            _target_steps = 0

            action = 4
            update_position(action)
            log_step(obs, action)
            return ACTIONS[action]

    # =====================================================
    # 7. SAFE DEFAULT
    # =====================================================
    next_edge = (pos_key, dir_bin(_angle))

    if next_edge in _bad_edges:
        action = 4
    else:
        action = 2

    update_position(action)
    log_step(obs, action)
    return ACTIONS[action]