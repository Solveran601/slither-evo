WORLD_SIZE = 10000
CELL_SIZE = 500
GRID_SIZE = WORLD_SIZE // CELL_SIZE

N_TEAMS = 10
WORMS_PER_TEAM = 10
N_WORMS = N_TEAMS * WORMS_PER_TEAM
MODELS_PER_TEAM = 15

FOOD_COUNT = 6000
SEG_RADIUS = 8
HEAD_RADIUS = 14
SEG_DIST = 14
INITIAL_SEGMENTS = 15
BASE_SPEED = 3.5
SPRINT_SPEED = 7.0
SPRINT_MASS_COST = 0.3

N_INPUT = 26
N_HIDDEN1 = 20
N_HIDDEN2 = 14
N_HIDDEN3 = 10
N_OUTPUT = 2
WEIGHT_COUNT = (N_INPUT * N_HIDDEN1 + N_HIDDEN1 +
                N_HIDDEN1 * N_HIDDEN2 + N_HIDDEN2 +
                N_HIDDEN2 * N_HIDDEN3 + N_HIDDEN3 +
                N_HIDDEN3 * N_OUTPUT + N_OUTPUT)  # 1006

# ========== MODE 'ffa' or 'team'
GAME_MODE = 'ffa'

ZONE_RADIUS = 18
ZONE_DAMAGE = 0.5

MUTATION_RATE = 0.15
MUTATION_AMOUNT = 0.25

FITNESS_SURVIVAL = 0.35
FITNESS_PVP = 0.25
FITNESS_SIZE = 0.30

# ========== FITNESS
INACTIVITY_THRESHOLD = 300     # pixels moved below this → smooth penalty
INACTIVITY_PENALTY = 0.3       # fitness multiplier for lazy worms
FOOD_REWARD = 4.0              # per food eaten (×2 vs kills per effort)
ACTIVITY_BONUS_FACTOR = 0.3    # bonus per 1000px traveled (cap 20 * factor)

# Hunger penalty — worms that don't eat enough lose fitness
STARVATION_PENALTY = 2.0       # penalty per missing food unit (×4 stronger)
STARVATION_DIVISOR = 30        # expect food every 30 ticks (stricter)

# ========== FOOD
FOOD_TYPES = {
    'normal': {'color': '#44ee55', 'mass': 0.08, 'points': 1, 'weight': 70, 'label': 'Normal'},
    'golden': {'color': '#ffd700', 'mass': 0.30, 'points': 2, 'weight': 15, 'label': 'Golden'},
    'poison': {'color': '#a855f7', 'mass': -0.20, 'points': -1, 'weight': 10, 'label': 'Poison'},
    'growth': {'color': '#ff69b4', 'mass': 0.00, 'points': 0, 'weight': 5, 'label': 'Growth'},
}

_BASE_NAMES = [
    'Alpha', 'Shadow', 'Viper', 'Cobra', 'Storm',
    'Blaze', 'Frost', 'Thorn', 'Venom', 'Ghost',
    'Raven', 'Titan', 'Neon', 'Pixel', 'Flux'
]
_EXTRA_NAMES = [
    'Fang', 'Claw', 'Spike', 'Horn', 'Scale',
    'Wing', 'Maw', 'Sting', 'Bolt', 'Fuse',
    'Echo', 'Nova', 'Zero', 'Byte', 'Dash',
    'Pulse', 'Drift', 'Glide', 'Surge', 'Crest',
    'Orbit', 'Prism', 'Shard', 'Gleam', 'Haze',
    'Jolt', 'Lynx', 'Myth', 'Nexus', 'Onyx',
]

TEAM_NAMES = list(_BASE_NAMES)
if N_TEAMS > len(TEAM_NAMES):
    needed = N_TEAMS - len(TEAM_NAMES)
    for i in range(needed):
        TEAM_NAMES.append(_EXTRA_NAMES[i % len(_EXTRA_NAMES)])
else:
    TEAM_NAMES = TEAM_NAMES[:N_TEAMS]

def _hsl_to_hex(h, s, l):
    """HSL -> #rrggbb hex string."""
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

_BASE_COLORS = [
    '#ff4455', '#ff8822', '#ffdd33', '#44ee55', '#00eebb',
    '#44aaff', '#5577ff', '#bb55ff', '#ff44aa', '#ff6677',
    '#99ff44', '#44ffaa', '#22ddff', '#cc44ff', '#ff88aa'
]

TEAM_COLORS = list(_BASE_COLORS)
if N_TEAMS > len(TEAM_COLORS):
    for i in range(len(TEAM_COLORS), N_TEAMS):
        hue = (i * 360 / N_TEAMS + 15) % 360
        TEAM_COLORS.append(_hsl_to_hex(hue, 0.85, 0.55))
else:
    TEAM_COLORS = TEAM_COLORS[:N_TEAMS]
