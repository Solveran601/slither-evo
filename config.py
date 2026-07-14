"""
Slither Evo v2 — Configuration
===============================
Per-mode config (team/ffa) with flexible team/agent counts.
Obstacle generation support.
"""

import math

# ═══════════════════════════════════════════════════════════════════════
# SHARED CONSTANTS (apply to all modes)
# ═══════════════════════════════════════════════════════════════════════
WORLD_SIZE = 3000
CELL_SIZE = 500

# Worm physics
SEG_RADIUS = 12
HEAD_RADIUS = 14
SEG_DIST = 14
INITIAL_SEGMENTS = 15
BASE_SPEED = 3.5
SPRINT_SPEED = 7.0
SPRINT_MASS_COST = 0.3

# NN architecture (layer-based perception: 8 rays x 5 layers + 4 extra)
N_INPUT = 44
N_HIDDEN1 = 30
N_HIDDEN2 = 22
N_HIDDEN3 = 16
N_OUTPUT = 4

WEIGHT_COUNT = (
    N_INPUT * N_HIDDEN1 + N_HIDDEN1 +
    N_HIDDEN1 * N_HIDDEN2 + N_HIDDEN2 +
    N_HIDDEN2 * N_HIDDEN3 + N_HIDDEN3 +
    N_HIDDEN3 * N_OUTPUT + N_OUTPUT
)

# Mutation
MUTATION_RATE = 0.15
MUTATION_AMOUNT = 0.25

# Fitness weights (now defined in FitnessEvaluator class in evolution.py)
# These constants kept for backward compat but not used by new evaluator
FITNESS_SURVIVAL = 1.0
FITNESS_PVP = 1.0
FITNESS_SIZE = 1.0
INACTIVITY_THRESHOLD = 100
INACTIVITY_PENALTY = 0.05
FOOD_REWARD = 12.0
ACTIVITY_BONUS_FACTOR = 0.3
STARVATION_PENALTY = 8.0
STARVATION_DIVISOR = 25

# Hall of Fame
HALL_OF_FAME_SIZE = 10
AUTO_SAVE_INTERVAL = 10

# Default mode (can be overridden by per-mode configs)
GAME_MODE = 'team'

# ═══════════════════════════════════════════════════════════════════════
# PER-MODE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
MODE_PARAMS = {
    'team': {
        'GAME_MODE': 'team',
        'N_TEAMS': 50,
        'WORMS_PER_TEAM': 5,
        'MODELS_PER_TEAM': 1,
        'FOOD_COUNT': 4000,
        'ZONE_RADIUS': 1800,
        'ZONE_DAMAGE': 0.5,
        'OBSTACLES_ENABLED': False,
        'OBSTACLE_MAP': 'random_gen',
    },
    'ffa': {
        'GAME_MODE': 'ffa',
        'N_TEAMS': 20,
        'WORMS_PER_TEAM': 1,
        'MODELS_PER_TEAM': 5,
        'FOOD_COUNT': 8000,
        'ZONE_RADIUS': 0,
        'ZONE_DAMAGE': 0,
        'OBSTACLES_ENABLED': False,
        'OBSTACLE_MAP': 'random_gen',
    },
}

# ═══════════════════════════════════════════════════════════════════════
# TEAM NAMES & COLORS
# ═══════════════════════════════════════════════════════════════════════
_BASE_NAMES = [
    'Alpha', 'Shadow', 'Viper', 'Cobra', 'Storm',
    'Blaze', 'Frost', 'Thorn', 'Venom', 'Ghost',
    'Raven', 'Titan', 'Neon', 'Pixel', 'Flux',
]

_EXTRA_NAMES = [
    '[',
]

_BASE_COLORS = [
    '#ff4455', '#ff8822', '#ffdd33', '#44ee55', '#00eebb',
    '#44aaff', '#5577ff', '#bb55ff', '#ff44aa', '#ff6677',
    '#99ff44', '#44ffaa', '#22ddff', '#cc44ff', '#ff88aa',
]


def _hsl_to_hex(h, s, l):
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    if h < 60: r, g, b = c, x, 0
    elif h < 120: r, g, b = x, c, 0
    elif h < 180: r, g, b = 0, c, x
    elif h < 240: r, g, b = 0, x, c
    elif h < 300: r, g, b = x, 0, c
    else: r, g, b = c, 0, x
    return '#{:02x}{:02x}{:02x}'.format(
        int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


# ═══════════════════════════════════════════════════════════════════════
# DYNAMIC COMPUTATION (depends on N_TEAMS / mode)
# ═══════════════════════════════════════════════════════════════════════
N_TEAMS = 1
WORMS_PER_TEAM = 1
MODELS_PER_TEAM = 1
FOOD_COUNT = 100
ZONE_RADIUS = 0
ZONE_DAMAGE = 0
OBSTACLES_ENABLED = False
OBSTACLE_MAP = ''
N_WORMS = 1
TEAM_NAMES = []
TEAM_COLORS = []


def _compute_derived():
    global N_WORMS, CELL_SIZE, GRID_SIZE, TEAM_NAMES, TEAM_COLORS
    N_WORMS = N_TEAMS * WORMS_PER_TEAM
    CELL_SIZE = WORLD_SIZE // 20
    GRID_SIZE = WORLD_SIZE // CELL_SIZE

    # Team names
    names = list(_BASE_NAMES)
    if N_TEAMS > len(names):
        needed = N_TEAMS - len(names)
        for i in range(needed):
            names.append(_EXTRA_NAMES[i % len(_EXTRA_NAMES)])
    else:
        names = names[:N_TEAMS]

    if GAME_MODE == 'ffa':
        names = [f'Agent_{i}' for i in range(N_TEAMS)]

    TEAM_NAMES = names

    # Team colors
    colors = list(_BASE_COLORS)
    if N_TEAMS > len(colors):
        for i in range(len(colors), N_TEAMS):
            hue = (i * 360 / N_TEAMS + 15) % 360
            colors.append(_hsl_to_hex(hue, 0.85, 0.55))
    else:
        colors = colors[:N_TEAMS]
    TEAM_COLORS = colors


def apply_mode_overrides(mode=None):
    """Apply per-mode parameter overrides and recompute derived values."""
    global GAME_MODE, N_TEAMS, WORMS_PER_TEAM, MODELS_PER_TEAM
    global FOOD_COUNT, ZONE_RADIUS, ZONE_DAMAGE
    global OBSTACLES_ENABLED, OBSTACLE_MAP

    if mode is not None:
        GAME_MODE = mode

    params = MODE_PARAMS.get(GAME_MODE, MODE_PARAMS['team'])
    GAME_MODE = params['GAME_MODE']
    N_TEAMS = params['N_TEAMS']
    WORMS_PER_TEAM = params['WORMS_PER_TEAM']
    MODELS_PER_TEAM = params['MODELS_PER_TEAM']
    FOOD_COUNT = params['FOOD_COUNT']
    ZONE_RADIUS = params['ZONE_RADIUS']
    ZONE_DAMAGE = params['ZONE_DAMAGE']
    OBSTACLES_ENABLED = params['OBSTACLES_ENABLED']
    OBSTACLE_MAP = params['OBSTACLE_MAP']

    _compute_derived()


def switch_mode(mode):
    """Switch game mode and recompute everything."""
    apply_mode_overrides(mode)


# Apply defaults at import time
apply_mode_overrides()
