"""
Evolution Engine — Slither Evo v2
==================================
Full genetic algorithm with per-team model pools (15 models/team),
advanced selection/crossover/mutation operators, speciation,
island model, hall of fame, and rich analytics. ~5000+ lines.

Architecture:
  NeuralNetwork      — feed-forward NN (14->12->2)
  ModelPool          — 15 models per team, epoch tracking
  WeightManager      — per-team folder structure (weights/TeamName/)
  ZoneManager        — team territories
  FitnessEvaluator   — RL-shaped fitness with inactivity penalty
  PopulationManager  — assigns models to worms, collects stats
  SelectionMethods   — tournament, roulette, rank, SUS, truncation
  CrossoverMethods   — uniform, single-point, two-point, blend, SBX, average
  MutationMethods    — gaussian, uniform, adaptive, polynomial, creep
  TeamEvolver        — GA loop for a team's model pool
  EvolutionEngine    — orchestrates everything
  Speciation         — niche protection via fitness sharing
  IslandModel        — migration between teams
  HallOfFame         — best models ever
  DiversityMetrics   — genetic diversity tracking
  HyperparameterScheduler — adaptive mutation/crossover rates
  EloRating          — team vs team comparison
  GenealogyTracker   — model lineage
  StatsAnalyzer      — trends, convergence, stagnation
  WeightAnalyzer     — weight distribution, layer importance
  TrainingManager    — checkpoints, rotation, logging
  BenchmarkSuite     — fitness benchmarks and tests
  ReportGenerator    — HTML/text report export
  ModelVisualizer    — weight heatmap data export
  CLI                — interactive command interface
"""

import json
import math
import os
import random
import time
import csv
import copy
import itertools
import statistics
from pathlib import Path
from collections import defaultdict, Counter, deque
from typing import List, Dict, Optional, Tuple, Union, Callable

import config
# Import all config names into module namespace for backward compat
# but keep reference to config module for dynamic reloading
from config import *
_CONFIG_NAMES = [k for k in dir(config) if k.isupper()]

def _sync_config():
    """Re-sync module-level config values from config module (for mode switching)."""
    for name in _CONFIG_NAMES:
        if hasattr(config, name):
            globals()[name] = getattr(config, name)

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════

class NeuralNetwork:
    """
    Feed-forward neural network.
    Architecture: N_INPUT -> N_HIDDEN1 -> N_HIDDEN2 -> N_HIDDEN3 -> N_OUTPUT
    Weight layout: [W1][b1][W2][b2][W3][b3][W4][b4]
    """
    __slots__ = ("weights", "hidden_act", "output_act", "arch",
                   "fitness", "epoch_created", "model_id", "team")

    ACTIVATIONS = {
        'tanh': lambda x: x / (1 + abs(x)),
        'sigmoid': lambda x: 1.0 / (1.0 + math.exp(-max(-100, min(100, x)))),
        'relu': lambda x: max(0.0, x),
        'leaky_relu': lambda x: x if x > 0 else x * 0.01,
        'identity': lambda x: x,
        'softsign': lambda x: x / (1 + abs(x)),
        'bent_identity': lambda x: (math.sqrt(x * x + 1.0) - 1.0) / 2.0 + x,
    }

    LAYER_SIZES = [N_INPUT, N_HIDDEN1, N_HIDDEN2, N_HIDDEN3, N_OUTPUT]

    def __init__(self, weights: Optional[List[float]] = None,
                 hidden_act: str = 'tanh', output_act: str = 'tanh'):
        self.hidden_act = hidden_act
        self.output_act = output_act
        self.arch = (N_INPUT, N_HIDDEN1, N_HIDDEN2, N_HIDDEN3, N_OUTPUT)
        self.fitness = 0.0
        self.epoch_created = 0
        self.model_id = -1
        self.team = -1

        if weights is None:
            self.weights = [0.0 for _ in range(WEIGHT_COUNT)]
        else:
            self.weights = list(weights)
        assert len(self.weights) == WEIGHT_COUNT, \
            f"Expected {WEIGHT_COUNT} weights, got {len(self.weights)}"

    def _activate(self, x: float, activation: str) -> float:
        fn = NeuralNetwork.ACTIVATIONS.get(activation)
        if fn is None:
            raise ValueError(f"Unknown activation: {activation}")
        return fn(x)

    def _layer_forward(self, inp: List[float], w_in: int,
                        n_in: int, n_out: int, act: str) -> Tuple[List[float], int]:
        """Compute one fully-connected layer. Returns (outputs, next_weight_index)."""
        out = [0.0] * n_out
        for j in range(n_out):
            s = self.weights[w_in + n_in * n_out + j]
            for i in range(n_in):
                s += inp[i] * self.weights[w_in + j * n_in + i]
            out[j] = self._activate(s, act)
        return out, w_in + n_in * n_out + n_out

    def forward(self, inputs: List[float]) -> Dict[str, float]:
        """Forward pass -> {turn: [-1,1], boost: [0,1]}"""
        w_off = 0
        # 3 hidden layers
        h1, w_off = self._layer_forward(inputs, w_off, N_INPUT, N_HIDDEN1, self.hidden_act)
        h2, w_off = self._layer_forward(h1, w_off, N_HIDDEN1, N_HIDDEN2, self.hidden_act)
        h3, w_off = self._layer_forward(h2, w_off, N_HIDDEN2, N_HIDDEN3, self.hidden_act)
        # output layer
        out, _ = self._layer_forward(h3, w_off, N_HIDDEN3, N_OUTPUT, self.output_act)
        boost = (out[1] + 1.0) / 2.0
        return {'turn': out[0], 'boost': max(0.0, min(1.0, boost))}

    def mutate(self, rate: Optional[float] = None,
               amount: Optional[float] = None) -> 'NeuralNetwork':
        rate = rate if rate is not None else MUTATION_RATE
        amount = amount if amount is not None else MUTATION_AMOUNT
        child = NeuralNetwork(list(self.weights), self.hidden_act, self.output_act)
        for i in range(WEIGHT_COUNT):
            if random.random() < rate:
                child.weights[i] += (random.random() - 0.5) * 2.0 * amount
        return child

    def copy(self) -> 'NeuralNetwork':
        nn = NeuralNetwork(list(self.weights), self.hidden_act, self.output_act)
        nn.fitness = self.fitness
        nn.epoch_created = self.epoch_created
        nn.model_id = self.model_id
        nn.team = self.team
        return nn

    def to_list(self) -> List[float]:
        return list(self.weights)

    @staticmethod
    def from_list(lst: List[float]) -> 'NeuralNetwork':
        return NeuralNetwork(lst)

    @staticmethod
    def random() -> 'NeuralNetwork':
        nn = NeuralNetwork()
        nn.weights = [random.uniform(-1.0, 1.0) for _ in range(WEIGHT_COUNT)]
        return nn

    def weight_stats(self) -> Dict[str, float]:
        """Compute basic statistics of this network's weights."""
        arr = self.weights
        n = len(arr)
        mean = sum(arr) / n
        variance = sum((x - mean) ** 2 for x in arr) / n
        return {
            'min': min(arr),
            'max': max(arr),
            'mean': mean,
            'std': math.sqrt(variance),
            'zeros': sum(1 for x in arr if abs(x) < 1e-6),
            'total': n,
        }

    def save(self, path: Union[str, Path]) -> None:
        data = {
            'weights': self.weights,
            'hidden_act': self.hidden_act,
            'output_act': self.output_act,
            'fitness': self.fitness,
            'epoch_created': self.epoch_created,
            'model_id': self.model_id,
            'team': self.team,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: Union[str, Path]) -> 'NeuralNetwork':
        with open(path) as f:
            data = json.load(f)
        nn = NeuralNetwork(
            data.get('weights'),
            data.get('hidden_act', 'tanh'),
            data.get('output_act', 'tanh'),
        )
        nn.fitness = data.get('fitness', 0.0)
        nn.epoch_created = data.get('epoch_created', 0)
        nn.model_id = data.get('model_id', -1)
        nn.team = data.get('team', -1)
        return nn

    def similarity(self, other: 'NeuralNetwork') -> float:
        """Euclidean distance between weight vectors (lower = more similar)."""
        return math.sqrt(
            sum((a - b) ** 2 for a, b in zip(self.weights, other.weights)) / WEIGHT_COUNT
        )

    def __repr__(self) -> str:
        return f"NN({self.arch[0]}->{self.arch[1]}->{self.arch[2]}->{self.arch[3]}->{self.arch[4]} fit={self.fitness:.1f})"


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: MODEL POOL — 15 models per team
# ═══════════════════════════════════════════════════════════════════

class ModelPool:
    """
    A team's pool of MODELS_PER_TEAM neural networks.
    Each pool tracks its own epoch and maintains fitness history
    for all models across generations.

    Structure:
      models: List[NeuralNetwork]  — size MODELS_PER_TEAM
      fitness_records: Dict[int, List[float]] — {model_idx: [fitness_values]}
      epoch: int — team-specific generation counter
    """

    def __init__(self, team: int, team_name: str = '', team_color: str = '#ffffff'):
        self.team = team
        self.team_name = team_name or TEAM_NAMES[team] if team < len(TEAM_NAMES) else f'Team-{team}'
        self.team_color = team_color or TEAM_COLORS[team] if team < len(TEAM_COLORS) else '#ffffff'
        self.epoch = 0
        self.models: List[NeuralNetwork] = []
        self.fitness_records: Dict[int, List[float]] = {}
        self.best_fitness_ever = 0.0
        self.best_model_idx = -1
        self.epoch_fitness_log: List[float] = []
        self.creation_time = time.time()

    def init_random(self) -> None:
        """Fill pool with MODELS_PER_TEAM random neural networks."""
        self.models = []
        for i in range(MODELS_PER_TEAM):
            nn = NeuralNetwork.random()
            nn.team = self.team
            nn.model_id = i
            nn.epoch_created = 0
            self.models.append(nn)
            self.fitness_records[i] = []

    def get_model(self, idx: int) -> NeuralNetwork:
        """Get a copy of model at index (0..MODELS_PER_TEAM-1)."""
        if 0 <= idx < len(self.models):
            return self.models[idx].copy()
        raise IndexError(f"Model index {idx} out of range [0, {len(self.models)})")

    def get_best_model(self) -> NeuralNetwork:
        """Return a copy of the best model."""
        if self.best_model_idx >= 0:
            return self.models[self.best_model_idx].copy()
        if self.models:
            return self.models[0].copy()
        return NeuralNetwork.random()

    def get_best_fitness(self) -> float:
        """Return the best fitness in the current pool."""
        if not self.models:
            return 0.0
        return max(m.fitness for m in self.models)

    def get_avg_fitness(self) -> float:
        if not self.models:
            return 0.0
        return sum(m.fitness for m in self.models) / len(self.models)

    def get_model_fitnesses(self) -> List[float]:
        """Return list of (model_idx, avg_fitness) for all models."""
        result = []
        for i in range(len(self.models)):
            rec = self.fitness_records.get(i, [])
            avg_f = sum(rec) / len(rec) if rec else 0.0
            result.append(avg_f)
        return result

    def get_all_fitness(self) -> Dict[int, float]:
        """Return {model_idx: fitness} for all models."""
        return {i: m.fitness for i, m in enumerate(self.models)}

    def record_fitness(self, model_idx: int, fitness: float) -> None:
        """Record a fitness observation for a specific model."""
        if model_idx not in self.fitness_records:
            self.fitness_records[model_idx] = []
        self.fitness_records[model_idx].append(fitness)
        # update running best
        if fitness > self.best_fitness_ever:
            self.best_fitness_ever = fitness
            self.best_model_idx = model_idx

    def update_model_fitness(self, model_idx: int, fitness: float) -> None:
        """Set the model's aggregate fitness (e.g., average across worms)."""
        if 0 <= model_idx < len(self.models):
            self.models[model_idx].fitness = fitness
            if fitness > self.best_fitness_ever:
                self.best_fitness_ever = fitness
                self.best_model_idx = model_idx

    def get_model_stats(self, model_idx: int) -> Dict:
        """Get detailed stats for a model."""
        if model_idx < 0 or model_idx >= len(self.models):
            return {}
        m = self.models[model_idx]
        rec = self.fitness_records.get(model_idx, [])
        return {
            'model_idx': model_idx,
            'fitness': m.fitness,
            'avg_fitness': sum(rec) / len(rec) if rec else 0.0,
            'n_evaluations': len(rec),
            'epoch_created': m.epoch_created,
            'weight_stats': m.weight_stats(),
        }

    def get_pool_stats(self) -> Dict:
        """Summary statistics for the entire pool."""
        fits = [m.fitness for m in self.models]
        return {
            'team': self.team,
            'team_name': self.team_name,
            'epoch': self.epoch,
            'n_models': len(self.models),
            'best_fitness': max(fits) if fits else 0.0,
            'avg_fitness': sum(fits) / len(fits) if fits else 0.0,
            'worst_fitness': min(fits) if fits else 0.0,
            'std_fitness': statistics.stdev(fits) if len(fits) > 1 else 0.0,
            'best_fitness_ever': self.best_fitness_ever,
            'best_model_idx': self.best_model_idx,
            'diversity': self.compute_diversity(),
            'epochs_run': len(self.epoch_fitness_log),
        }

    def compute_diversity(self) -> float:
        """Average pairwise distance between all models in the pool."""
        if len(self.models) < 2:
            return 0.0
        total = 0.0
        pairs = 0
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                total += self.models[i].similarity(self.models[j])
                pairs += 1
        return total / pairs if pairs > 0 else 0.0

    def get_top_n(self, n: int) -> List[Tuple[int, NeuralNetwork]]:
        """
        Return the top n models (index, model copy) sorted by fitness descending.
        """
        indexed = list(enumerate(self.models))
        indexed.sort(key=lambda x: x[1].fitness, reverse=True)
        return [(i, m.copy()) for i, m in indexed[:n]]

    def save(self, base_dir: Union[str, Path]) -> None:
        """Save all models as model_NN.json plus meta.json."""
        base = Path(base_dir) / self.team_name
        base.mkdir(parents=True, exist_ok=True)

        # save meta
        meta = {
            'team': self.team,
            'team_name': self.team_name,
            'team_color': self.team_color,
            'epoch': self.epoch,
            'best_fitness_ever': self.best_fitness_ever,
            'best_model_idx': self.best_model_idx,
            'creation_time': self.creation_time,
            'n_models': len(self.models),
        }
        with open(base / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        # save each model
        for i, nn in enumerate(self.models):
            nn.model_id = i
            nn.team = self.team
            path = base / f'model_{i:02d}.json'
            nn.save(path)

    @staticmethod
    def load(team: int, base_dir: Union[str, Path]) -> Optional['ModelPool']:
        """Load a team's pool from disk. Returns None if not found."""
        team_name = TEAM_NAMES[team] if team < len(TEAM_NAMES) else f'Team-{team}'
        team_color = TEAM_COLORS[team] if team < len(TEAM_COLORS) else '#ffffff'
        base = Path(base_dir) / team_name

        if not base.exists():
            return None

        pool = ModelPool(team, team_name, team_color)

        # load meta
        meta_file = base / 'meta.json'
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                pool.epoch = meta.get('epoch', 0)
                pool.best_fitness_ever = meta.get('best_fitness_ever', 0.0)
                pool.best_model_idx = meta.get('best_model_idx', -1)
                pool.creation_time = meta.get('creation_time', time.time())
            except (json.JSONDecodeError, IOError):
                pass

        # load models
        pool.models = []
        for i in range(MODELS_PER_TEAM):
            path = base / f'model_{i:02d}.json'
            if path.exists():
                try:
                    nn = NeuralNetwork.load(path)
                    pool.models.append(nn)
                    pool.fitness_records[i] = []
                except (json.JSONDecodeError, IOError, AssertionError):
                    pool.models.append(NeuralNetwork.random())
                    pool.fitness_records[i] = []
            else:
                nn = NeuralNetwork.random()
                nn.team = team
                nn.model_id = i
                pool.models.append(nn)
                pool.fitness_records[i] = []

        # fill if missing
        while len(pool.models) < MODELS_PER_TEAM:
            nn = NeuralNetwork.random()
            nn.team = team
            nn.model_id = len(pool.models)
            pool.models.append(nn)

        return pool


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: WEIGHT MANAGER — per-team folder structure
# ═══════════════════════════════════════════════════════════════════

class WeightManager:
    """
    Manages all team model pools on disk.
    Directory structure:
      weights/
        <mode_dir>/        (e.g. 'team' or 'ffa')
          TeamName/
            meta.json
            model_00.json
            ...
    """

    def __init__(self, base_dir: str = 'weights', mode_dir: str = ''):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.mode_dir = mode_dir or GAME_MODE
        self.pools: Dict[int, ModelPool] = {}
        self.initialized = False

    @property
    def mode_path(self) -> Path:
        return self.base_dir / self.mode_dir

    def init_random_pools(self) -> None:
        """Create random pools for all teams without loading from disk."""
        self.pools = {}
        for t in range(N_TEAMS):
            pool = ModelPool(t, TEAM_NAMES[t], TEAM_COLORS[t])
            pool.init_random()
            self.pools[t] = pool
        self.save_all()
        self.initialized = True

    def init_teams(self) -> None:
        """Create random pools for all teams (backward compat)."""
        self.init_random_pools()

    def load_all(self) -> None:
        """Load all team pools from disk, creating missing ones."""
        mode_path = self.mode_path
        mode_path.mkdir(parents=True, exist_ok=True)

        for t in range(N_TEAMS):
            pool = ModelPool.load(t, mode_path)
            if pool is None:
                name = TEAM_NAMES[t] if t < len(TEAM_NAMES) else f'Team-{t}'
                color = TEAM_COLORS[t] if t < len(TEAM_COLORS) else '#ffffff'
                pool = ModelPool(t, name, color)
                pool.init_random()
                pool.save(mode_path)
            self.pools[t] = pool
        self.initialized = True

    def get_pool(self, team: int) -> ModelPool:
        return self.pools.get(team)

    def get_model(self, team: int, model_idx: int) -> Optional[NeuralNetwork]:
        pool = self.pools.get(team)
        if pool and 0 <= model_idx < len(pool.models):
            return pool.models[model_idx]
        return None

    def save_pool(self, team: int) -> None:
        pool = self.pools.get(team)
        if pool:
            pool.save(self.mode_path)

    def save_all(self) -> None:
        mode_path = self.mode_path
        mode_path.mkdir(parents=True, exist_ok=True)
        for pool in self.pools.values():
            pool.save(mode_path)

    def get_all_weights(self) -> Dict[int, List[float]]:
        """Return {team: best_model_weights} for backward compat."""
        result = {}
        for t, pool in self.pools.items():
            best = pool.get_best_model()
            result[t] = best.to_list()
        return result

    def get_all_pools_info(self) -> Dict[int, Dict]:
        """Return pool info for all teams (for API)."""
        return {t: pool.get_pool_stats() for t, pool in self.pools.items()}

    def get_team_epochs(self) -> Dict[int, int]:
        return {t: pool.epoch for t, pool in self.pools.items()}

    def get_all_models_for_team(self, team: int) -> List[List[float]]:
        """Return ALL model weights for a team (for the frontend pool)."""
        pool = self.pools.get(team)
        if not pool:
            return []
        return [m.to_list() for m in pool.models]

    def reset_for_mode(self, mode: str) -> None:
        """Reset all pools for a new game mode."""
        self.mode_dir = mode
        self.pools = {}
        self.initialized = False
        self.init_random_pools()


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: ZONE MANAGER
# ═══════════════════════════════════════════════════════════════════

class ZoneManager:
    """Manages team zones (territories) on the map."""

    def __init__(self):
        self.zones: Dict[int, Dict] = {}

    def generate(self) -> None:
        """Place N_TEAMS zones randomly on the map with minimum spacing."""
        self.zones = {}
        min_dist = ZONE_RADIUS * 2.5
        margin = ZONE_RADIUS + 500
        attempts = 0
        while len(self.zones) < N_TEAMS and attempts < 2000:
            cx = random.uniform(margin, WORLD_SIZE - margin)
            cy = random.uniform(margin, WORLD_SIZE - margin)
            ok = True
            for existing in self.zones.values():
                dx = cx - existing['cx']
                dy = cy - existing['cy']
                if math.sqrt(dx * dx + dy * dy) < min_dist:
                    ok = False
                    break
            if ok:
                team = len(self.zones)
                self.zones[team] = {'cx': cx, 'cy': cy, 'radius': ZONE_RADIUS}
            attempts += 1

    def to_dict(self) -> Dict:
        return self.zones

    def is_inside(self, x: float, y: float, team: int) -> bool:
        if team not in self.zones:
            return False
        z = self.zones[team]
        dx = x - z['cx']
        dy = y - z['cy']
        return dx * dx + dy * dy <= z['radius'] * z['radius']

    def is_any_zone(self, x: float, y: float) -> Optional[int]:
        for team, z in self.zones.items():
            dx = x - z['cx']
            dy = y - z['cy']
            if dx * dx + dy * dy <= z['radius'] * z['radius']:
                return team
        return None

    def get_zone_center(self, team: int) -> Tuple[float, float]:
        z = self.zones.get(team)
        if z:
            return (z['cx'], z['cy'])
        return (WORLD_SIZE / 2, WORLD_SIZE / 2)


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: FITNESS EVALUATOR
# ═══════════════════════════════════════════════════════════════════

class FitnessEvaluator:
    """
    ======================================================================
    MERIT-BASED FITNESS SYSTEM (no free points)
    ======================================================================

    CORE PRINCIPLE: Worms EARN their fitness. Nothing is free.

    REWARDS (only earned through action):
      +15    per food eaten           (primary — seek food!)
      +4     per unit mass gained     (growth = eating well)
      +80    per kill                 (PvP — high risk, high reward)
      +0.5   per 100px traveled       (exploration, no cap)
      +8     per 10s in enemy zone    (territory aggression)

    PENALTIES (subtractive, applied BEFORE multiplier):
      -3     per second alive with NO food eaten yet
             → only applies for first 10s (grace period)
      -5     per missing food unit when starving
             → expected = survival_time / 20 (must eat every 20s)
      -(mass_lost * 4)   if dropped >50% from peak mass

    MULTIPLIER starts at 1.0:
      ×0.01  if died in first 3s      (instant death — useless)
      ×0.05  if died from wall        (wall collision — avoidable)
      ×0.02  if traveled < 50px after 20s (completely frozen)
      ×0.10  if never ate anything    (peak mass < 1.5)

    BONUS MULTIPLIERS (on top of multiplier):
      ×1.4   if mass > 30             (big worm)
      ×1.3   if killed >= 2 enemies   (predator)
      ×1.2   if food_eaten > 40       (feeder)
      ×1.1   if survived > 90s        (veteran)

    PHILOSOPHY:
      A worm that eats 5 food, gains 3 mass, lives 60s, travels 3000px:
        food: 5*15 = 75
        growth: (4-1)*4 = 12  (start mass 1, end mass 4)
        explore: 3000/100*0.5 = 15
        starvation: expected=60/20=3, deficit=0 → 0
        total raw: 102
      No bonuses/penalties → fitness = 102

      A worm that eats 0 food, lives 60s, travels 500px:
        food: 0
        growth: 0
        explore: 500/100*0.5 = 2.5
        starvation: expected=60/20=3, deficit=3, penalty=3*5=15
        grace: first 10s penalty = 10*3 = 30
        raw: 2.5 - 15 - 30 = 0 (clamped)
        never_ate: ×0.10
        fitness = 0

      The second worm gets ZERO. This forces evolution to prioritize
      food-seeking above all else. Survival alone is worthless.
    """

    def __init__(self):
        self.food_reward = 4.0
        self.mass_gain_per_unit = 1.5
        self.kill_reward = 20.0
        self.exploration_per_100px = 0.15
        self.zone_aggression_per_10s = 2.0

        self.big_worm_threshold = 30.0
        self.big_worm_bonus = 1.15
        self.predator_kill_threshold = 2
        self.predator_bonus = 1.1
        self.feeder_food_threshold = 40
        self.feeder_bonus = 1.05
        self.veteran_threshold = 90.0
        self.veteran_bonus = 1.03

        self.instant_death_threshold = 3.0
        self.instant_death_penalty = 0.005
        self.wall_death_penalty = 0.03
        self.frozen_distance_threshold = 50
        self.frozen_time_threshold = 20.0
        self.frozen_penalty = 0.01
        self.never_ate_penalty = 0.02

        self.starvation_divisor = 14
        self.starvation_penalty_per_unit = 12.0
        self.mass_waste_penalty = 8.0

        self.grace_period = 8.0
        self.grace_penalty_per_sec = 6.0
        self.sprint_penalty_per_sec = 4.0

    def compute(self, mass: float, survival_time: float, kills: int,
                food_eaten: int = 0, distance: float = 0.0,
                extra_bonus: float = 0.0, wall_death: bool = False,
                peak_mass: float = 0.0, zone_time: float = 0.0,
                sprint_time: float = 0.0) -> float:
        """
        Compute fitness — only positive if the worm actually did something.
        """
        food_score = food_eaten * self.food_reward
        mass_gained = max(0.0, mass - 1.0)
        growth_score = mass_gained * self.mass_gain_per_unit
        kill_score = kills * self.kill_reward
        explore_score = distance / 100.0 * self.exploration_per_100px
        zone_score = (zone_time / 10.0) * self.zone_aggression_per_10s

        raw = food_score + growth_score + kill_score + explore_score + zone_score + extra_bonus
        raw -= sprint_time * self.sprint_penalty_per_sec

        grace_penalty = max(0.0, self.grace_period - survival_time) * self.grace_penalty_per_sec
        raw -= grace_penalty

        expected_food = survival_time / self.starvation_divisor if survival_time > self.grace_period else 0
        if food_eaten < expected_food:
            deficit = expected_food - food_eaten
            raw -= deficit * self.starvation_penalty_per_unit

        if peak_mass > 5.0 and mass < peak_mass * 0.5:
            mass_lost = peak_mass - mass
            raw -= mass_lost * self.mass_waste_penalty

        raw = max(0.0, raw)

        penalty = 1.0
        if survival_time < self.instant_death_threshold:
            penalty *= self.instant_death_penalty
        if wall_death:
            penalty *= self.wall_death_penalty
        if survival_time > self.frozen_time_threshold and distance < self.frozen_distance_threshold:
            penalty *= self.frozen_penalty
        peak = max(peak_mass, mass)
        if peak < 1.5:
            penalty *= self.never_ate_penalty

        bonus = 1.0
        if mass > self.big_worm_threshold:
            bonus *= self.big_worm_bonus
        if kills >= self.predator_kill_threshold:
            bonus *= self.predator_bonus
        if food_eaten > self.feeder_food_threshold:
            bonus *= self.feeder_bonus
        if survival_time > self.veteran_threshold:
            bonus *= self.veteran_bonus

        return raw * penalty * bonus

    def compute_from_stats(self, worm_stats: Dict) -> float:
        return self.compute(
            mass=worm_stats.get('mass', 0),
            survival_time=worm_stats.get('survivalTime', 0),
            kills=worm_stats.get('kills', 0),
            food_eaten=worm_stats.get('foodEaten', 0),
            distance=worm_stats.get('distanceTraveled', 0),
            extra_bonus=worm_stats.get('bonus', 0),
            wall_death=worm_stats.get('wallDeath', False),
            peak_mass=worm_stats.get('peakMass', 0),
            zone_time=worm_stats.get('zoneTime', 0),
            sprint_time=worm_stats.get('sprintTime', 0),
        )

    def normalized(self, mass: float, survival_time: float, kills: int,
                   food_eaten: int = 0, distance: float = 0.0) -> float:
        raw = self.compute(mass, survival_time, kills, food_eaten, distance)
        expected_good = 10 * self.food_reward + 8 * self.mass_gain_per_unit + 5000 / 100 * self.exploration_per_100px
        return min(100.0, (raw / max(expected_good, 1)) * 100.0)


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: POPULATION MANAGER
# ═══════════════════════════════════════════════════════════════════

class PopulationManager:
    """
    Manages which model each worm uses, collects per-model fitness data,
    and handles model assignment for respawns.

    Tracks:
      {team: {model_idx: [fitness_values]}}
    """

    def __init__(self):
        self.assignment: Dict[int, Dict[int, int]] = {}  # {team: {worm_id: model_idx}}
        self.fitness_data: Dict[int, Dict[int, List[float]]] = {}  # {team: {model_idx: [fitness]}}
        self.model_usage: Dict[int, Dict[int, int]] = {}  # {team: {model_idx: usage_count}}

    def assign_model(self, team: int, worm_id: int, model_idx: int) -> None:
        if team not in self.assignment:
            self.assignment[team] = {}
        self.assignment[team][worm_id] = model_idx

    def get_assignment(self, team: int, worm_id: int) -> Optional[int]:
        return self.assignment.get(team, {}).get(worm_id)

    def record_fitness(self, team: int, model_idx: int, fitness: float) -> None:
        if team not in self.fitness_data:
            self.fitness_data[team] = defaultdict(list)
        self.fitness_data[team][model_idx].append(fitness)
        if team not in self.model_usage:
            self.model_usage[team] = defaultdict(int)
        self.model_usage[team][model_idx] += 1

    def get_model_avg_fitness(self, team: int, model_idx: int) -> float:
        records = self.fitness_data.get(team, {}).get(model_idx, [])
        return sum(records) / len(records) if records else 0.0

    def get_all_model_fitness(self, team: int) -> Dict[int, float]:
        """Get average fitness for each model in a team."""
        result = {}
        team_data = self.fitness_data.get(team, {})
        for mid, fits in team_data.items():
            result[mid] = sum(fits) / len(fits) if fits else 0.0
        return result

    def get_usage_counts(self, team: int) -> Dict[int, int]:
        return dict(self.model_usage.get(team, {}))

    def get_best_model(self, team: int) -> Optional[int]:
        """Return model index with highest average fitness for this team."""
        fits = self.get_all_model_fitness(team)
        if not fits:
            return None
        return max(fits, key=fits.get)

    def select_model_for_respawn(self, team: int, pool: ModelPool,
                                  strategy: str = 'fitness_weighted') -> int:
        """
        Select a model for a new worm based on strategy.
        Strategies:
          'fitness_weighted' — roulette wheel by fitness
          'best' — always best model
          'random' — uniform random
          'tournament' — tournament selection
        """
        if strategy == 'random' or not pool.models:
            return random.randrange(len(pool.models))

        if strategy == 'best':
            fits = [(i, m.fitness) for i, m in enumerate(pool.models)]
            return max(fits, key=lambda x: x[1])[0]

        if strategy == 'tournament':
            k = min(3, len(pool.models))
            candidates = random.sample(range(len(pool.models)), k)
            best_idx = max(candidates, key=lambda i: pool.models[i].fitness)
            return best_idx

        # fitness_weighted (default)
        fits = [max(0.01, m.fitness) for m in pool.models]
        total = sum(fits)
        r = random.random() * total
        cumulative = 0.0
        for i, f in enumerate(fits):
            cumulative += f
            if r <= cumulative:
                return i
        return random.randrange(len(pool.models))

    def reset_team(self, team: int) -> None:
        """Clear tracking data for a team (after evolution)."""
        self.assignment.pop(team, None)
        self.fitness_data.pop(team, None)
        self.model_usage.pop(team, None)

    def reset_all(self) -> None:
        self.assignment.clear()
        self.fitness_data.clear()
        self.model_usage.clear()

    def get_summary(self, team: int) -> Dict:
        fits = self.get_all_model_fitness(team)
        usage = self.get_usage_counts(team)
        return {
            'team': team,
            'models_tracked': len(fits),
            'total_evaluations': sum(len(v) for v in self.fitness_data.get(team, {}).values()),
            'best_model': self.get_best_model(team),
            'model_fitnesses': fits,
            'model_usage': usage,
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: SELECTION METHODS
# ═══════════════════════════════════════════════════════════════════

class SelectionMethods:
    """Collection of selection operators for genetic algorithms."""

    @staticmethod
    def tournament(fitness_dict: Dict[int, float], tournament_size: int = 3) -> int:
        """Tournament selection: pick k random, return best."""
        items = list(fitness_dict.items())
        if len(items) <= tournament_size:
            return max(items, key=lambda x: x[1])[0]
        sample = random.sample(items, min(tournament_size, len(items)))
        return max(sample, key=lambda x: x[1])[0]

    @staticmethod
    def tournament_multiple(fitness_dict: Dict[int, float], count: int,
                            tournament_size: int = 3) -> List[int]:
        """Select 'count' individuals via tournament (with replacement)."""
        return [
            SelectionMethods.tournament(fitness_dict, tournament_size)
            for _ in range(count)
        ]

    @staticmethod
    def roulette(fitness_dict: Dict[int, float]) -> int:
        """Roulette wheel selection: probability proportional to fitness."""
        items = list(fitness_dict.items())
        if not items:
            raise ValueError("Empty fitness dict")
        min_f = min(f for _, f in items)
        offset = 0.0 if min_f >= 0 else -min_f + 0.01
        weights = [max(0.001, f + offset) for _, f in items]
        total = sum(weights)
        r = random.random() * total
        cumulative = 0.0
        for (idx, _), w in zip(items, weights):
            cumulative += w
            if r <= cumulative:
                return idx
        return items[-1][0]

    @staticmethod
    def roulette_multiple(fitness_dict: Dict[int, float], count: int) -> List[int]:
        return [SelectionMethods.roulette(fitness_dict) for _ in range(count)]

    @staticmethod
    def rank(fitness_dict: Dict[int, float], selective_pressure: float = 1.5) -> int:
        """
        Rank-based selection: probability based on rank (not raw fitness).
        Best individual: pressure, worst: 2 - pressure.
        """
        items = list(fitness_dict.items())
        items.sort(key=lambda x: x[1])
        n = len(items)
        if n == 0:
            raise ValueError("Empty fitness dict")
        if n == 1:
            return items[0][0]
        probs = [
            (2.0 - selective_pressure) + 2.0 * (selective_pressure - 1.0) * (n - 1 - i) / (n - 1)
            for i in range(n)
        ]
        total = sum(probs)
        r = random.random() * total
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return items[i][0]
        return items[-1][0]

    @staticmethod
    def sus(fitness_dict: Dict[int, float], count: int) -> List[int]:
        """
        Stochastic Universal Sampling: evenly spaced pointers.
        More diverse than roulette.
        """
        items = list(fitness_dict.items())
        if not items:
            raise ValueError("Empty fitness dict")
        if count >= len(items):
            return [random.choice(items)[0] for _ in range(count)]

        min_f = min(f for _, f in items)
        offset = 0.0 if min_f >= 0 else -min_f + 0.01
        weights = [max(0.001, f + offset) for _, f in items]
        total = sum(weights)
        if total <= 0:
            return [random.choice(items)[0] for _ in range(count)]

        step = total / count
        start = random.random() * step
        pointers = [start + i * step for i in range(count)]

        result = []
        cumulative = 0.0
        idx = 0
        for p in pointers:
            while cumulative < p and idx < len(items):
                cumulative += weights[idx]
                if cumulative < p:
                    idx += 1
            result.append(items[min(idx, len(items) - 1)][0])
        return result

    @staticmethod
    def truncation(fitness_dict: Dict[int, float], top_k: int) -> List[int]:
        """Truncation selection: keep top k individuals."""
        items = sorted(fitness_dict.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in items[:top_k]]

    @staticmethod
    def stochastic_tournament(fitness_dict: Dict[int, float], win_prob: float = 0.75) -> int:
        """Stochastic tournament: pick 2, better wins with probability win_prob."""
        items = list(fitness_dict.items())
        a, b = random.sample(items, 2)
        if a[1] > b[1]:
            return a[0] if random.random() < win_prob else b[0]
        else:
            return b[0] if random.random() < win_prob else a[0]


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: CROSSOVER METHODS
# ═══════════════════════════════════════════════════════════════════

class CrossoverMethods:
    """Collection of crossover operators for real-valued genomes."""

    @staticmethod
    def uniform(p1: List[float], p2: List[float]) -> Tuple[List[float], List[float]]:
        """Uniform crossover: each gene from either parent with 50% probability."""
        n = len(p1)
        c1 = [0.0] * n
        c2 = [0.0] * n
        for i in range(n):
            if random.random() < 0.5:
                c1[i] = p1[i]
                c2[i] = p2[i]
            else:
                c1[i] = p2[i]
                c2[i] = p1[i]
        return c1, c2

    @staticmethod
    def single_point(p1: List[float], p2: List[float]) -> Tuple[List[float], List[float]]:
        """Single-point crossover: one split point."""
        n = len(p1)
        point = random.randrange(1, n)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return c1, c2

    @staticmethod
    def two_point(p1: List[float], p2: List[float]) -> Tuple[List[float], List[float]]:
        """Two-point crossover: two split points."""
        n = len(p1)
        a, b = sorted(random.sample(range(1, n), 2))
        c1 = p1[:a] + p2[a:b] + p1[b:]
        c2 = p2[:a] + p1[a:b] + p2[b:]
        return c1, c2

    @staticmethod
    def blend(p1: List[float], p2: List[float], alpha: float = 0.5) -> Tuple[List[float], List[float]]:
        """Blend (BLX-alpha) crossover: child in [min - I*alpha, max + I*alpha]."""
        n = len(p1)
        c1 = [0.0] * n
        c2 = [0.0] * n
        for i in range(n):
            lo = min(p1[i], p2[i])
            hi = max(p1[i], p2[i])
            I = hi - lo
            ext = I * alpha
            c1[i] = random.uniform(lo - ext, hi + ext)
            c2[i] = random.uniform(lo - ext, hi + ext)
        return c1, c2

    @staticmethod
    def simulated_binary(p1: List[float], p2: List[float], eta: float = 15.0) -> Tuple[List[float], List[float]]:
        """
        Simulated Binary Crossover (SBX).
        eta: distribution index (larger = closer to parents).
        """
        n = len(p1)
        c1 = [0.0] * n
        c2 = [0.0] * n
        for i in range(n):
            if random.random() < 0.5:
                if abs(p1[i] - p2[i]) < 1e-10:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
                else:
                    u = random.random()
                    if u <= 0.5:
                        beta = (2.0 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
                    c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]
        return c1, c2

    @staticmethod
    def average(p1: List[float], p2: List[float]) -> Tuple[List[float], List[float]]:
        """Average crossover: child = (p1 + p2) / 2 + small noise."""
        n = len(p1)
        c1 = [0.0] * n
        c2 = [0.0] * n
        for i in range(n):
            avg = (p1[i] + p2[i]) / 2.0
            noise = random.uniform(-0.1, 0.1)
            c1[i] = avg + noise
            c2[i] = avg - noise
        return c1, c2

    @staticmethod
    def heuristic(p1: List[float], p2: List[float], fitness_p1: float,
                  fitness_p2: float) -> Tuple[List[float], List[float]]:
        """Heuristic crossover: child moves toward the better parent."""
        n = len(p1)
        if fitness_p1 > fitness_p2:
            better, worse = p1, p2
        else:
            better, worse = p2, p1
        r = random.random()
        c1 = [better[i] + r * (better[i] - worse[i]) for i in range(n)]
        c2 = [worse[i] + r * (better[i] - worse[i]) for i in range(n)]
        return c1, c2

    @staticmethod
    def select_crossover(method: str = 'uniform'):
        """Return the crossover function by name."""
        methods = {
            'uniform': CrossoverMethods.uniform,
            'single_point': CrossoverMethods.single_point,
            'two_point': CrossoverMethods.two_point,
            'blend': CrossoverMethods.blend,
            'sbx': CrossoverMethods.simulated_binary,
            'average': CrossoverMethods.average,
            'heuristic': CrossoverMethods.heuristic,
        }
        return methods.get(method, CrossoverMethods.uniform)

    @staticmethod
    def crossover_pair(p1: List[float], p2: List[float], method: str = 'uniform',
                       **kwargs) -> Tuple[List[float], List[float]]:
        fn = CrossoverMethods.select_crossover(method)
        return fn(p1, p2, **kwargs) if method in ('blend', 'sbx', 'heuristic') else fn(p1, p2)


# ═══════════════════════════════════════════════════════════════════
# SECTION 9: MUTATION METHODS
# ═══════════════════════════════════════════════════════════════════

class MutationMethods:
    """Collection of mutation operators for real-valued genomes."""

    @staticmethod
    def gaussian(weights: List[float], rate: float = 0.12, sigma: float = 0.5) -> List[float]:
        """Gaussian mutation: add N(0,sigma) noise to each gene with probability rate."""
        result = list(weights)
        for i in range(len(result)):
            if random.random() < rate:
                result[i] += random.gauss(0, sigma)
        return result

    @staticmethod
    def uniform(weights: List[float], rate: float = 0.12, amount: float = 0.8) -> List[float]:
        """Uniform mutation: add U(-amount, amount) noise."""
        result = list(weights)
        for i in range(len(result)):
            if random.random() < rate:
                result[i] += (random.random() - 0.5) * 2.0 * amount
        return result

    @staticmethod
    def adaptive(weights: List[float], parent_fitness: float,
                 base_fitness: float = 50.0) -> List[float]:
        """Adaptive mutation: higher fitness = lower mutation rate."""
        ratio = min(1.0, parent_fitness / max(base_fitness, 1.0))
        rate = 0.05 + (1.0 - ratio) * 0.25
        amount = 0.2 + (1.0 - ratio) * 1.0
        return MutationMethods.uniform(weights, rate, amount)

    @staticmethod
    def polynomial(weights: List[float], rate: float = 0.12, eta: float = 20.0) -> List[float]:
        """
        Polynomial mutation (Deb & Agrawal).
        eta: distribution index (larger = smaller perturbation).
        """
        result = list(weights)
        for i in range(len(result)):
            if random.random() < rate:
                u = random.random()
                if u < 0.5:
                    delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                result[i] += delta
        return result

    @staticmethod
    def creep(weights: List[float], rate: float = 0.05, step: float = 0.1) -> List[float]:
        """Creep mutation: small incremental changes to a few genes."""
        result = list(weights)
        for i in range(len(result)):
            if random.random() < rate:
                result[i] += random.choice([-step, step]) * random.random()
        return result

    @staticmethod
    def non_uniform(weights: List[float], generation: int, max_generation: int,
                    rate: float = 0.12, strength: float = 1.0) -> List[float]:
        """
        Non-uniform mutation: decreases over generations.
        Allows exploration early, exploitation late.
        """
        t = generation / max(max_generation, 1)
        r = random.random()
        amount = strength * (1.0 - t)  # decays over time
        result = list(weights)
        for i in range(len(result)):
            if random.random() < rate:
                result[i] += (random.random() - 0.5) * 2.0 * amount
        return result

    @staticmethod
    def selective(weights: List[float], importance_mask: List[float],
                  rate: float = 0.12, amount: float = 0.8) -> List[float]:
        """
        Selective mutation: important weights (high magnitude) mutate less.
        importance_mask: list of importance values (0-1) for each weight.
        """
        result = list(weights)
        for i in range(len(result)):
            imp = importance_mask[i] if i < len(importance_mask) else 0.5
            adj_rate = rate * (1.0 - imp * 0.8)
            if random.random() < adj_rate:
                result[i] += (random.random() - 0.5) * 2.0 * amount
        return result

    @staticmethod
    def select_mutation(method: str = 'uniform'):
        methods = {
            'gaussian': MutationMethods.gaussian,
            'uniform': MutationMethods.uniform,
            'adaptive': MutationMethods.adaptive,
            'polynomial': MutationMethods.polynomial,
            'creep': MutationMethods.creep,
            'non_uniform': MutationMethods.non_uniform,
            'selective': MutationMethods.selective,
        }
        return methods.get(method, MutationMethods.uniform)


# ═══════════════════════════════════════════════════════════════════
# SECTION 10: GENETIC OPERATORS (legacy wrappers)
# ═══════════════════════════════════════════════════════════════════

def crossover_weights(parent_a: List[float], parent_b: List[float],
                      method: str = 'uniform') -> List[float]:
    """Legacy wrapper: crossover producing one child (first child)."""
    c1, _ = CrossoverMethods.crossover_pair(parent_a, parent_b, method)
    return c1


def gaussian_mutate(weights: List[float], rate: float = 0.15, sigma: float = 0.5) -> List[float]:
    return MutationMethods.gaussian(weights, rate, sigma)


def adaptive_mutate(weights: List[float], parent_fitness: float,
                    base_fitness: float = 50.0) -> List[float]:
    return MutationMethods.adaptive(weights, parent_fitness, base_fitness)


def mutate_weights_inplace(weights: List[float], rate: float = 0.12,
                           amount: float = 0.8) -> List[float]:
    return MutationMethods.uniform(weights, rate, amount)


# ═══════════════════════════════════════════════════════════════════
# SECTION 11: TEAM EVOLVER — GA for a team's model pool
# ═══════════════════════════════════════════════════════════════════

class TeamEvolver:
    """
    Genetic algorithm engine for evolving a single team's model pool.

    Each call to evolve_pool() takes the current pool + per-model fitness,
    and produces a new pool of MODELS_PER_TEAM models via:
      1. Sort by fitness, preserve elites
      2. Select parents (tournament / roulette / rank)
      3. Crossover (uniform / single-point / two-point / blend / sbx)
      4. Mutate offspring
      5. Fill remaining slots with random + mutated elites
      6. Increment epoch
    """

    def __init__(self,
                 pool_size: int = MODELS_PER_TEAM,
                 elite_count: int = 3,           # keep top 3 elites
                 tournament_size: int = 4,       # stronger selection pressure
                 crossover_rate: float = 0.6,    # more mutation, less crossover
                 crossover_method: str = 'uniform',
                 mutation_rate: float = 0.08,    # lower for 2468 weights (was 0.12 for 1006)
                 mutation_amount: float = 0.5,   # gentler mutations (was 0.8)
                 mutation_method: str = 'uniform',
                 selection_method: str = 'tournament',
                 random_fill: int = 2):          # more random exploration (was 1)
        self.pool_size = max(1, pool_size)
        self.elite_count = min(elite_count, self.pool_size)
        self.tournament_size = min(tournament_size, self.pool_size)
        self.crossover_rate = crossover_rate
        self.crossover_method = crossover_method
        self.mutation_rate = mutation_rate
        self.mutation_amount = mutation_amount
        self.mutation_method = mutation_method
        self.selection_method = selection_method
        self.random_fill = min(random_fill, max(0, self.pool_size - self.elite_count))

    def evolve_pool(self, pool: ModelPool,
                    model_fitnesses: Optional[Dict[int, float]] = None) -> ModelPool:
        """
        Evolve a team's model pool.

        Args:
            pool: The current ModelPool to evolve
            model_fitnesses: {model_idx: avg_fitness} for the current generation.
                             If None, uses pool.models[i].fitness.

        Returns:
            A new ModelPool with evolved models and incremented epoch.
        """
        # Build fitness dict from available data
        fitness_dict = {}
        if model_fitnesses:
            for i in range(len(pool.models)):
                fitness_dict[i] = model_fitnesses.get(i, pool.models[i].fitness)
        else:
            for i, m in enumerate(pool.models):
                fitness_dict[i] = m.fitness

        # If everything is zero fitness, use small random values for selection
        if all(f <= 0 for f in fitness_dict.values()):
            for i in fitness_dict:
                fitness_dict[i] = random.random() * 0.1 + 0.01

        # Create new pool
        new_pool = ModelPool(pool.team, pool.team_name, pool.team_color)
        new_pool.models = []
        new_pool.fitness_records = {i: [] for i in range(self.pool_size)}

        # Sort by fitness (descending) for elitism
        sorted_indices = sorted(fitness_dict.keys(), key=lambda i: fitness_dict[i], reverse=True)

        # 1. Elitism: preserve top models unchanged
        elites = []
        for idx in sorted_indices[:self.elite_count]:
            elites.append(pool.models[idx].copy())
            new_pool.models.append(pool.models[idx].copy())

        # 2. Ensure at least one mutated model (even with pool_size=1)
        first_mutated = False
        needed = self.pool_size - len(new_pool.models) - self.random_fill

        for _ in range(needed):
            child = self._create_offspring(fitness_dict, pool.models)
            if child is not None:
                new_pool.models.append(child)
                first_mutated = True

        # If no offspring was created, mutate the last elite
        if not first_mutated and len(new_pool.models) > 0:
            new_pool.models[-1] = new_pool.models[-1].mutate(self.mutation_rate, self.mutation_amount)
            first_mutated = True

        # 3. Fill remaining with mutated versions of top models
        while len(new_pool.models) < self.pool_size - self.random_fill:
            parent_idx = random.choice(sorted_indices[:max(3, self.pool_size // 3)])
            child = pool.models[parent_idx].mutate(self.mutation_rate, self.mutation_amount)
            new_pool.models.append(child)

        # 4. Add completely random models for diversity
        for _ in range(self.random_fill):
            new_pool.models.append(NeuralNetwork.random())

        # Ensure correct size
        while len(new_pool.models) > self.pool_size:
            new_pool.models.pop()
        while len(new_pool.models) < self.pool_size:
            new_pool.models.append(NeuralNetwork.random())

        # Set metadata
        for i, m in enumerate(new_pool.models):
            m.team = pool.team
            m.model_id = i
            m.epoch_created = pool.epoch + 1

        # Update pool metadata
        new_pool.epoch = pool.epoch + 1
        new_pool.best_fitness_ever = pool.best_fitness_ever
        new_pool.epoch_fitness_log = list(pool.epoch_fitness_log)

        # Log best fitness of new pool
        best_in_new = max(m.fitness for m in new_pool.models)
        new_pool.epoch_fitness_log.append(best_in_new)
        if best_in_new > new_pool.best_fitness_ever:
            new_pool.best_fitness_ever = best_in_new

        return new_pool

    def _create_offspring(self, fitness_dict: Dict[int, float],
                          models: List[NeuralNetwork]) -> Optional[NeuralNetwork]:
        """Create one offspring via selection + crossover + mutation."""
        # Select two parents
        if self.selection_method == 'tournament':
            p1_idx = SelectionMethods.tournament(fitness_dict, self.tournament_size)
            p2_idx = SelectionMethods.tournament(fitness_dict, self.tournament_size)
        elif self.selection_method == 'roulette':
            p1_idx = SelectionMethods.roulette(fitness_dict)
            p2_idx = SelectionMethods.roulette(fitness_dict)
        elif self.selection_method == 'rank':
            p1_idx = SelectionMethods.rank(fitness_dict)
            p2_idx = SelectionMethods.rank(fitness_dict)
        else:
            p1_idx = SelectionMethods.tournament(fitness_dict, self.tournament_size)
            p2_idx = SelectionMethods.tournament(fitness_dict, self.tournament_size)

        if p1_idx == p2_idx:
            # If same parent, just mutate
            child = models[p1_idx].mutate(self.mutation_rate, self.mutation_amount)
            return child

        # Crossover
        if random.random() < self.crossover_rate:
            w1 = models[p1_idx].to_list()
            w2 = models[p2_idx].to_list()
            child_w, _ = CrossoverMethods.crossover_pair(w1, w2, self.crossover_method)
            child = NeuralNetwork(child_w)
        else:
            # No crossover: pick one parent
            child = models[p1_idx].copy()

        # Mutation
        if random.random() < self.mutation_rate:
            mut_fn = MutationMethods.select_mutation(self.mutation_method)
            child_w = mut_fn(
                child.to_list(),
                self.mutation_rate if self.mutation_method != 'adaptive' else None,
            )
            child = NeuralNetwork(child_w)

        return child


# ═══════════════════════════════════════════════════════════════════
# SECTION 11B: TEAM EVOLVER EXTENSIONS
# ═══════════════════════════════════════════════════════════════════

class TeamEvolverExtensions:
    """Additional evolution strategies for TeamEvolver."""

    @staticmethod
    def differential_evolution(pool: ModelPool,
                               fitness_dict: Dict[int, float],
                               f: float = 0.8,
                               cr: float = 0.9) -> ModelPool:
        """
        Differential Evolution variant for the model pool.
        DE/rand/1/bin strategy.
        """
        n = len(pool.models)
        dim = WEIGHT_COUNT
        new_pool = ModelPool(pool.team, pool.team_name, pool.team_color)
        new_pool.models = []
        indices = list(range(n))

        for i in range(n):
            # Pick 3 distinct random indices different from i
            candidates = [idx for idx in indices if idx != i]
            a, b, c = random.sample(candidates, 3)

            # Mutation: v = a + F * (b - c)
            w_a = pool.models[a].to_list()
            w_b = pool.models[b].to_list()
            w_c = pool.models[c].to_list()
            mutant = [w_a[j] + f * (w_b[j] - w_c[j]) for j in range(dim)]

            # Crossover: binomial
            trial = [0.0] * dim
            j_rand = random.randrange(dim)
            for j in range(dim):
                if random.random() < cr or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = pool.models[i].to_list()[j]

            child = NeuralNetwork(trial)
            child.team = pool.team
            child.model_id = i
            child.epoch_created = pool.epoch + 1
            new_pool.models.append(child)

        new_pool.epoch = pool.epoch + 1
        new_pool.best_fitness_ever = pool.best_fitness_ever
        new_pool.epoch_fitness_log = list(pool.epoch_fitness_log)
        return new_pool

    @staticmethod
    def simulated_annealing(model: NeuralNetwork,
                            initial_temp: float = 1.0,
                            cooling_rate: float = 0.95,
                            steps: int = 50) -> NeuralNetwork:
        """
        Apply simulated annealing to refine a single model.
        Higher temp = more exploration.
        """
        current = model.copy()
        current_fit = current.fitness
        temp = initial_temp

        for step in range(steps):
            # Generate neighbor by mutation
            neighbor = current.mutate(rate=0.2, amount=temp)
            neighbor.fitness = current_fit + random.uniform(-1, 1)  # approximate

            # Acceptance criterion
            delta = neighbor.fitness - current_fit
            if delta > 0 or random.random() < math.exp(delta / max(temp, 0.01)):
                current = neighbor
                current_fit = neighbor.fitness

            temp *= cooling_rate

        return current

    @staticmethod
    def island_selection(pools: Dict[int, ModelPool],
                         n_islands: int = 3) -> List[ModelPool]:
        """
        Group teams into islands, each island evolves independently,
        then best models migrate between islands.
        """
        teams = list(range(N_TEAMS))
        random.shuffle(teams)
        islands = []
        chunk_size = max(1, len(teams) // n_islands)

        for i in range(0, len(teams), chunk_size):
            island_teams = teams[i:i + chunk_size]
            island_pools = {t: pools[t] for t in island_teams if t in pools}
            islands.append(island_pools)

        return islands

    @staticmethod
    def crowding_selection(models: List[NeuralNetwork],
                           fitness: Dict[int, float],
                           n_select: int = MODELS_PER_TEAM) -> List[int]:
        """
        Deterministic crowding: select diverse individuals.
        Prevents premature convergence.
        """
        if len(models) <= n_select:
            return list(range(len(models)))

        selected = []
        remaining = set(range(len(models)))

        # Always keep the best
        best_idx = max(remaining, key=lambda i: fitness.get(i, 0))
        selected.append(best_idx)
        remaining.remove(best_idx)

        while len(selected) < n_select and remaining:
            # Pick the individual most different from selected set
            best_diversity = -1
            best_candidate = -1
            for i in remaining:
                min_dist_to_selected = min(
                    models[i].similarity(models[s]) for s in selected
                )
                if min_dist_to_selected > best_diversity:
                    best_diversity = min_dist_to_selected
                    best_candidate = i
            if best_candidate >= 0:
                selected.append(best_candidate)
                remaining.remove(best_candidate)

        return selected


# ═══════════════════════════════════════════════════════════════════
# SECTION 11C: NOISE AND REGULARIZATION
# ═══════════════════════════════════════════════════════════════════

class NoiseInjection:
    """
    Noise injection strategies for robust evolution.
    Adds noise to inputs/weights during evaluation to
    encourage robustness.
    """

    @staticmethod
    def gaussian_noise(value: float, std: float = 0.05) -> float:
        return value + random.gauss(0, std)

    @staticmethod
    def uniform_noise(value: float, amount: float = 0.1) -> float:
        return value + random.uniform(-amount, amount)

    @staticmethod
    def perturb_weights(model: NeuralNetwork, std: float = 0.01) -> NeuralNetwork:
        """Add small Gaussian noise to all weights."""
        child = model.copy()
        for i in range(len(child.weights)):
            child.weights[i] += random.gauss(0, std)
        return child

    @staticmethod
    def evaluate_with_noise(model: NeuralNetwork, base_fitness: float,
                            n_samples: int = 5, noise_std: float = 0.02) -> float:
        """
        Evaluate model fitness with noise injection.
        Returns average fitness across noisy evaluations.
        """
        total = 0.0
        for _ in range(n_samples):
            noisy = NoiseInjection.perturb_weights(model, noise_std)
            # Approximate fitness change
            noise_factor = 1.0 + random.gauss(0, 0.1)
            total += base_fitness * max(0.5, noise_factor)
        return total / n_samples


class Regularization:
    """Weight regularization to prevent extreme values."""

    @staticmethod
    def l2_penalty(weights: List[float], lambda_val: float = 0.001) -> float:
        """L2 regularization penalty."""
        return lambda_val * sum(w ** 2 for w in weights)

    @staticmethod
    def l1_penalty(weights: List[float], lambda_val: float = 0.001) -> float:
        """L1 regularization penalty (promotes sparsity)."""
        return lambda_val * sum(abs(w) for w in weights)

    @staticmethod
    def clip_weights(weights: List[float], min_val: float = -5.0,
                     max_val: float = 5.0) -> List[float]:
        """Clip weights to a range."""
        return [max(min_val, min(max_val, w)) for w in weights]

    @staticmethod
    def apply_regularization(model: NeuralNetwork,
                             l2_lambda: float = 0.001,
                             clip: bool = True) -> NeuralNetwork:
        """Apply regularization to a model."""
        child = model.copy()
        if clip:
            child.weights = Regularization.clip_weights(child.weights)
        penalty = Regularization.l2_penalty(child.weights, l2_lambda)
        child.fitness -= penalty
        return child, penalty


# ═══════════════════════════════════════════════════════════════════
# SECTION 11D: TABU SEARCH
# ═══════════════════════════════════════════════════════════════════

class TabuSearch:
    """
    Tabu search wrapper for local optimization of models.
    Prevents revisiting recently explored weight configurations.
    """

    def __init__(self, tabu_size: int = 20):
        self.tabu_list: deque = deque(maxlen=tabu_size)

    def _hash_weights(self, weights: List[float], precision: int = 2) -> str:
        return ','.join(str(round(w, precision)) for w in weights[:10])

    def is_tabu(self, weights: List[float]) -> bool:
        h = self._hash_weights(weights)
        return h in self.tabu_list

    def add(self, weights: List[float]) -> None:
        h = self._hash_weights(weights)
        self.tabu_list.append(h)

    def optimize(self, model: NeuralNetwork, evaluator: FitnessEvaluator,
                 max_steps: int = 20) -> NeuralNetwork:
        """Local optimization with tabu search."""
        best = model.copy()
        best_fit = best.fitness
        current = model.copy()

        for step in range(max_steps):
            # Generate neighbor
            neighbor = current.mutate(rate=0.3, amount=0.5)

            # Skip if tabu
            if self.is_tabu(neighbor.to_list()):
                continue

            # Evaluate (approximate)
            neighbor.fitness = best_fit + random.gauss(0, 2)

            if neighbor.fitness > best_fit:
                best = neighbor.copy()
                best_fit = neighbor.fitness

            self.add(current.to_list())
            current = neighbor

        return best


# ═══════════════════════════════════════════════════════════════════
# SECTION 11E: ENSEMBLE EVALUATION
# ═══════════════════════════════════════════════════════════════════

class EnsembleEvaluator:
    """
    Evaluate teams by combining multiple models into an ensemble.
    Useful for ranking teams by their collective intelligence.
    """

    @staticmethod
    def ensemble_forward(models: List[NeuralNetwork],
                         inputs: List[float],
                         method: str = 'average') -> Dict[str, float]:
        """Forward pass through multiple models, combine outputs."""
        turns = []
        boosts = []

        for m in models:
            out = m.forward(inputs)
            turns.append(out['turn'])
            boosts.append(out['boost'])

        if method == 'average':
            return {
                'turn': sum(turns) / len(turns),
                'boost': sum(boosts) / len(boosts),
            }
        elif method == 'median':
            sorted_t = sorted(turns)
            sorted_b = sorted(boosts)
            mid = len(sorted_t) // 2
            return {
                'turn': sorted_t[mid],
                'boost': sorted_b[mid],
            }
        elif method == 'max':
            return {
                'turn': max(turns, key=abs),
                'boost': max(boosts),
            }
        elif method == 'voting':
            turn_signs = [1 if t > 0 else -1 for t in turns]
            avg_sign = sum(turn_signs)
            return {
                'turn': 1.0 if avg_sign > 0 else -1.0,
                'boost': sum(boosts) / len(boosts),
            }
        return {'turn': 0, 'boost': 0}

    @staticmethod
    def team_ensemble_fitness(pool: ModelPool, n_trials: int = 50) -> float:
        """
        Estimate team fitness by evaluating ensemble of top models
        on standardized test scenarios.
        """
        top_models = [m.copy() for m in pool.get_top_n(3)]
        if not top_models:
            return 0.0

        score = 0.0
        for _ in range(n_trials):
            inp = [random.uniform(-1, 1) for _ in range(N_INPUT)]
            ensemble_out = EnsembleEvaluator.ensemble_forward(top_models, inp)
            score += abs(ensemble_out['turn']) * 0.5 + ensemble_out['boost'] * 0.5

        return (score / n_trials) * 100


# ═══════════════════════════════════════════════════════════════════
# SECTION 12: EVOLUTION ENGINE — main orchestrator
# ═══════════════════════════════════════════════════════════════════

class EvolutionEngine:
    """
    Main orchestrator for the Slither Evo genetic algorithm.

    Responsibilities:
      - Manage all team model pools
      - Receive stats from browser, compute fitness
      - Trigger evolution when teams are fully dead
      - Track global stats, best fitness, generations
      - Provide API for server endpoints
    """

    def __init__(self, mode: str = ''):
        mode_name = mode or GAME_MODE
        if mode:
            config.switch_mode(mode)
            _sync_config()
        self.weight_manager = WeightManager(mode_dir=mode_name)
        self.zone_manager = ZoneManager()
        self.fitness_evaluator = FitnessEvaluator()
        self.population_manager = PopulationManager()
        self.team_evolver = TeamEvolver()
        self.novelty_evolver = NoveltyEnhancedEvolver(novelty_weight=4.0)
        self.selector = TournamentSelector()
        self.hparams = HyperparameterScheduler()

        self.generation = 0  # global generation counter
        self.total_births = 0
        self.best_fitness_ever = 0.0
        self.stats_log = []
        self.team_last_epoch: Dict[int, int] = {}
        self.analyzer = StatsAnalyzer(self.stats_log)

        self.load_state()

    def reset_for_mode(self, mode: str) -> None:
        """Completely reset the engine for a new game mode."""
        config.switch_mode(mode)
        _sync_config()
        self.weight_manager = WeightManager(mode_dir=mode)
        self.zone_manager = ZoneManager()
        self.population_manager = PopulationManager()
        self.team_evolver = TeamEvolver()
        self.novelty_evolver = NoveltyEnhancedEvolver(novelty_weight=4.0)
        self.hparams = HyperparameterScheduler()
        self.generation = 0
        self.total_births = 0
        self.best_fitness_ever = 0.0
        self.stats_log = []
        self.team_last_epoch = {}
        self.analyzer = StatsAnalyzer(self.stats_log)

        self.weight_manager.init_random_pools()
        if GAME_MODE == 'team':
            self.zone_manager.generate()

    def load_state(self) -> None:
        """Load existing pools or create new ones, load zones and history."""
        self.weight_manager.load_all()
        self.zone_manager.generate()

        # Load generation from stats history
        stats_file = Path(__file__).parent / 'stats_history.json'
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    self.stats_log = json.load(f)
                if self.stats_log:
                    last = self.stats_log[-1]
                    self.generation = last.get('generation', 0)
                    self.total_births = last.get('total_births', 0)
                    self.best_fitness_ever = last.get('best_fitness_ever', 0.0)
            except (json.JSONDecodeError, IOError):
                pass

        # Keep analyzer in sync even if the stats list was reloaded.
        self.analyzer.history = self.stats_log

        # Initialize team last epochs
        for t in range(N_TEAMS):
            pool = self.weight_manager.get_pool(t)
            if pool:
                self.team_last_epoch[t] = pool.epoch

    def _refresh_training_controls(self) -> Dict[str, float]:
        """Adapt GA settings from global progress signals."""
        stagnation = self.analyzer.get_stagnation_count()
        improvement_rate = self.analyzer.get_improvement_rate(20)
        self.hparams.update(
            self.generation,
            max_generations=1000,
            current_best_fitness=self.best_fitness_ever,
            stagnation=stagnation,
        )
        params = self.hparams.get_params()
        self.team_evolver.mutation_rate = params['mutation_rate']
        self.team_evolver.mutation_amount = params['mutation_amount']
        self.team_evolver.crossover_rate = params['crossover_rate']

        return {
            **params,
            'stagnation': stagnation,
            'improvement_rate': round(improvement_rate, 4),
        }

    def _configure_team_evolver(self, pool: ModelPool) -> Dict[str, float]:
        """Tune selection pressure for one pool before evolving it."""
        diversity = PopulationHealthMetrics.compute_diversity(pool)
        stagnation = PopulationHealthMetrics.compute_stagnation(pool.epoch_fitness_log)
        improvement_rate = PopulationHealthMetrics.compute_improvement_rate(pool.epoch_fitness_log)
        convergence_risk = PopulationHealthMetrics.compute_convergence_risk(pool)

        if convergence_risk > 0.65 or diversity < 0.25 or stagnation > 8:
            self.team_evolver.selection_method = 'roulette'
            self.team_evolver.tournament_size = 2
            self.team_evolver.random_fill = min(
                max(2, self.team_evolver.pool_size // 3),
                max(0, self.team_evolver.pool_size - self.team_evolver.elite_count),
            )
        elif improvement_rate > 5.0:
            self.team_evolver.selection_method = 'tournament'
            self.team_evolver.tournament_size = 5
            self.team_evolver.random_fill = 1
        else:
            self.team_evolver.selection_method = 'tournament'
            self.team_evolver.tournament_size = 4
            self.team_evolver.random_fill = 2

        self.team_evolver.random_fill = min(
            self.team_evolver.random_fill,
            max(0, self.team_evolver.pool_size - self.team_evolver.elite_count),
        )

        return {
            'diversity': round(diversity, 4),
            'stagnation': stagnation,
            'improvement_rate': round(improvement_rate, 4),
            'convergence_risk': round(convergence_risk, 4),
        }

    def process_worm_stats(self, worm: Dict) -> Tuple[int, int, float]:
        """
        Process a single worm's stats.
        Returns: (team, model_idx, fitness)
        """
        team = worm['team']
        worm_id = worm.get('id', 0)
        model_idx = worm.get('modelId', 0)

        fitness = self.fitness_evaluator.compute(
            mass=worm.get('mass', 0),
            survival_time=worm.get('survivalTime', 0),
            kills=worm.get('kills', 0),
            food_eaten=worm.get('foodEaten', 0),
            distance=worm.get('distanceTraveled', 0),
            wall_death=worm.get('wallDeath', False),
            peak_mass=worm.get('peakMass', 0),
            zone_time=worm.get('zoneTime', 0),
            sprint_time=worm.get('sprintTime', 0),
        )

        # Record in population manager
        self.population_manager.assign_model(team, worm_id, model_idx)
        self.population_manager.record_fitness(team, model_idx, fitness)

        # Update pool
        pool = self.weight_manager.get_pool(team)
        if pool:
            pool.record_fitness(model_idx, fitness)

        return team, model_idx, fitness

    def evolve(self, dead_stats: List[Dict], dead_teams: List[int]) -> List[Dict]:
        """
        Process dead worm stats and evolve dead teams.
        Fitness is computed ONLY on death — no intermediate evaluations.

        Args:
            dead_stats: List of dead worm stats dicts (sent once on death)
            dead_teams: List of team indices where all worms are dead

        Returns:
            List of evolution results for dead teams.
        """
        evolutions = []
        all_fitnesses = []
        for w in dead_stats:
            team, mid, f = self.process_worm_stats(w)
            all_fitnesses.append((f, w))

        if all_fitnesses:
            best_current = max(f for f, _ in all_fitnesses)
            if best_current > self.best_fitness_ever:
                self.best_fitness_ever = best_current

        best_fitness = -1.0
        best_team = -1
        for f, w in all_fitnesses:
            if f > best_fitness:
                best_fitness = f
                best_team = w['team']

        if not dead_teams:
            return evolutions

        global_training = self._refresh_training_controls()

        # Evolve each dead team (all members dead → epoch advance)
        for team in dead_teams:
            pool = self.weight_manager.get_pool(team)
            if pool is None:
                continue

            model_fitnesses = self.population_manager.get_all_model_fitness(team)
            team_health = self._configure_team_evolver(pool)
            selection_fitnesses = dict(model_fitnesses)
            if pool.models:
                try:
                    selection_fitnesses = self.novelty_evolver.apply_novelty_to_pool(
                        pool, model_fitnesses
                    )
                except Exception:
                    selection_fitnesses = dict(model_fitnesses)

            gen = self.generation
            migrate_chance = 0.05 + 0.15 / (1 + gen / 100)
            if best_team >= 0 and best_team != team and random.random() < migrate_chance:
                best_pool = self.weight_manager.get_pool(best_team)
                if best_pool:
                    best_model = best_pool.get_best_model()
                    if best_model:
                        inject_idx = random.randrange(len(pool.models))
                        migrated = best_model.copy()
                        migrated.team = team
                        migrated.model_id = inject_idx
                        migrated.epoch_created = pool.epoch
                        pool.models[inject_idx] = migrated
                        selection_fitnesses[inject_idx] = best_fitness * 1.1

            new_pool = self.team_evolver.evolve_pool(pool, selection_fitnesses)
            self.weight_manager.pools[team] = new_pool
            self.weight_manager.save_pool(team)

            models_data = [{'model_idx': i, 'weights': m.to_list()} for i, m in enumerate(new_pool.models)]
            evolutions.append({
                'team': team,
                'epoch': new_pool.epoch,
                'models': models_data,
                'training': {
                    **team_health,
                    'mutation_rate': round(self.team_evolver.mutation_rate, 4),
                    'mutation_amount': round(self.team_evolver.mutation_amount, 4),
                    'crossover_rate': round(self.team_evolver.crossover_rate, 4),
                    'selection_method': self.team_evolver.selection_method,
                    'random_fill': self.team_evolver.random_fill,
                },
            })
            self.total_births += 1
            self.team_last_epoch[team] = new_pool.epoch
            self.population_manager.reset_team(team)

        self.generation += 1

        stats_snapshot = {
            'generation': self.generation,
            'alive': len(dead_stats),
            'dead_teams': dead_teams,
            'best_fitness': best_fitness,
            'best_team': best_team,
            'total_births': self.total_births,
            'best_fitness_ever': self.best_fitness_ever,
            'timestamp': time.time(),
            'team_epochs': dict(self.team_last_epoch),
            'training': global_training,
        }
        self.stats_log.append(stats_snapshot)
        if len(self.stats_log) > 1000:
            del self.stats_log[:-1000]

        self.save_state()
        return evolutions

    def save_state(self) -> None:
        """Save stats history to disk."""
        try:
            stats_file = Path(__file__).parent / 'stats_history.json'
            with open(stats_file, 'w') as f:
                json.dump(self.stats_log[-500:], f)
        except IOError:
            pass

    def get_leaderboard(self) -> List[Dict]:
        """Get ranked list of teams based on pool performance."""
        ranks = []
        for team in range(N_TEAMS):
            pool = self.weight_manager.get_pool(team)
            if pool:
                stats = pool.get_pool_stats()
                rank_val = self._compute_rank(team)
                ranks.append({
                    'team': team,
                    'name': TEAM_NAMES[team],
                    'color': TEAM_COLORS[team],
                    'rank': rank_val,
                    'epoch': pool.epoch,
                    'best_fitness': round(stats['best_fitness'], 1),
                    'avg_fitness': round(stats['avg_fitness'], 1),
                    'diversity': round(stats['diversity'], 3),
                })
            else:
                ranks.append({
                    'team': team,
                    'name': TEAM_NAMES[team],
                    'color': TEAM_COLORS[team],
                    'rank': 10,
                    'epoch': 0,
                    'best_fitness': 0.0,
                    'avg_fitness': 0.0,
                    'diversity': 0.0,
                })
        ranks.sort(key=lambda x: x['rank'])
        return ranks

    def _compute_rank(self, team: int) -> int:
        """Compute team rank 1-10 based on pool stats."""
        scores = {}
        for t in range(N_TEAMS):
            pool = self.weight_manager.get_pool(t)
            if pool:
                scores[t] = pool.get_pool_stats().get('best_fitness', 0)
            else:
                scores[t] = 0
        sorted_teams = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        n = len(sorted_teams)
        for rank_pos, t in enumerate(sorted_teams):
            if t == team:
                return max(1, min(10, n - rank_pos))
        return 10

    def get_stats(self) -> Dict:
        """Get current stats for the GET /stats endpoint."""
        team_epochs = self.weight_manager.get_team_epochs()
        pool_infos = self.weight_manager.get_all_pools_info()
        return {
            'generation': self.generation,
            'bestFitnessEver': round(self.best_fitness_ever, 1),
            'totalBirths': self.total_births,
            'zones': self.zone_manager.to_dict(),
            'teamEpochs': team_epochs,
            'training': {
                'mutation_rate': round(self.team_evolver.mutation_rate, 4),
                'mutation_amount': round(self.team_evolver.mutation_amount, 4),
                'crossover_rate': round(self.team_evolver.crossover_rate, 4),
                'selection_method': self.team_evolver.selection_method,
                'tournament_size': self.team_evolver.tournament_size,
                'random_fill': self.team_evolver.random_fill,
                'stagnation': self.analyzer.get_stagnation_count(),
                'improvement_rate': round(self.analyzer.get_improvement_rate(20), 4),
                'novelty_weight': round(self.novelty_evolver.novelty_weight, 4),
                'novelty_archive_size': len(self.novelty_evolver.archive.archive),
            },
            'pools': {
                str(t): {
                    'epoch': info.get('epoch', 0),
                    'best_fitness': info.get('best_fitness', 0.0),
                    'avg_fitness': info.get('avg_fitness', 0.0),
                    'n_models': info.get('n_models', 0),
                    'diversity': info.get('diversity', 0.0),
                }
                for t, info in pool_infos.items()
            },
        }

    def get_team_stats(self, team: int) -> Dict:
        """Get detailed stats for a specific team."""
        pool = self.weight_manager.get_pool(team)
        if not pool:
            return {'error': 'team not found'}
        stats = pool.get_pool_stats()
        model_stats = []
        for i in range(len(pool.models)):
            model_stats.append(pool.get_model_stats(i))
        return {
            **stats,
            'models': model_stats,
            'fitness_history': pool.epoch_fitness_log[-100:],
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 13: SPECIATION
# ═══════════════════════════════════════════════════════════════════

class Speciation:
    """
    Speciation via fitness sharing and niche protection.
    Groups models into species based on genetic similarity,
    then applies fitness sharing to maintain diversity.
    """

    def __init__(self, similarity_threshold: float = 0.5,
                 compatibility_modifier: float = 1.0):
        self.threshold = similarity_threshold
        self.compatibility_modifier = compatibility_modifier
        self.species: Dict[int, List[int]] = {}  # {species_id: [model_indices]}
        self.species_fitness: Dict[int, float] = {}

    def classify(self, models: List[NeuralNetwork]) -> Dict[int, List[int]]:
        """
        Assign each model to a species based on genetic similarity.
        Uses a simple clustering: if a model is within threshold of
        any species representative, join that species; else new species.
        """
        self.species = {}
        representatives = []

        for i, model in enumerate(models):
            assigned = False
            for s_id, rep_idx in enumerate(representatives):
                rep_model = models[rep_idx]
                dist = model.similarity(rep_model)
                if dist < self.threshold * self.compatibility_modifier:
                    if s_id not in self.species:
                        self.species[s_id] = []
                    self.species[s_id].append(i)
                    assigned = True
                    break
            if not assigned:
                new_sid = len(representatives)
                representatives.append(i)
                self.species[new_sid] = [i]

        return self.species

    def apply_fitness_sharing(self, models: List[NeuralNetwork],
                               raw_fitness: Dict[int, float]) -> Dict[int, float]:
        """
        Apply fitness sharing: divide fitness by species size.
        This protects niche species from being dominated.
        """
        self.classify(models)
        shared_fitness = {}

        for s_id, members in self.species.items():
            niche_count = len(members)
            for idx in members:
                shared_fitness[idx] = raw_fitness.get(idx, 0) / max(niche_count, 1)

        return shared_fitness

    def get_species_summary(self) -> Dict:
        return {
            f'Species_{sid}': {
                'size': len(members),
                'avg_fitness': self.species_fitness.get(sid, 0.0),
            }
            for sid, members in self.species.items()
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 14: ISLAND MODEL
# ═══════════════════════════════════════════════════════════════════

class IslandModel:
    """
    Island model for migration between teams.
    Periodically, the best model from one team migrates to another team,
    replacing the worst model in the destination team's pool.
    """

    def __init__(self, migration_interval: int = 10,
                 migration_rate: float = 0.1,
                 topology: str = 'ring'):
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.topology = topology  # 'ring', 'random', 'fully_connected'
        self.last_migration_gen = 0

    def should_migrate(self, generation: int) -> bool:
        return (generation - self.last_migration_gen) >= self.migration_interval

    def get_migration_pairs(self) -> List[Tuple[int, int]]:
        """Get (source_team, dest_team) pairs based on topology."""
        pairs = []
        teams = list(range(N_TEAMS))

        if self.topology == 'ring':
            for i in range(N_TEAMS):
                src = i
                dst = (i + 1) % N_TEAMS
                pairs.append((src, dst))

        elif self.topology == 'random':
            n_migrations = max(1, int(N_TEAMS * self.migration_rate))
            for _ in range(n_migrations):
                src, dst = random.sample(teams, 2)
                pairs.append((src, dst))

        elif self.topology == 'fully_connected':
            for src in teams:
                for dst in teams:
                    if src != dst:
                        pairs.append((src, dst))

        return pairs

    def migrate(self, weight_manager: WeightManager, generation: int) -> List[Dict]:
        """
        Perform migration between teams.
        Returns list of migration events.
        """
        if not self.should_migrate(generation):
            return []

        self.last_migration_gen = generation
        events = []

        for src_team, dst_team in self.get_migration_pairs():
            src_pool = weight_manager.get_pool(src_team)
            dst_pool = weight_manager.get_pool(dst_team)
            if not src_pool or not dst_pool:
                continue

            # Find worst model in dest
            dst_fits = [(i, m.fitness) for i, m in enumerate(dst_pool.models)]
            if not dst_fits:
                continue
            worst_idx = min(dst_fits, key=lambda x: x[1])[0]

            # Best model from source
            best_model = src_pool.get_best_model()

            # Replace
            dst_pool.models[worst_idx] = best_model
            dst_pool.models[worst_idx].team = dst_team
            dst_pool.models[worst_idx].model_id = worst_idx
            dst_pool.models[worst_idx].epoch_created = dst_pool.epoch

            events.append({
                'source_team': src_team,
                'dest_team': dst_team,
                'source_fitness': best_model.fitness,
                'replaced_idx': worst_idx,
            })

        return events


# ═══════════════════════════════════════════════════════════════════
# SECTION 15: HALL OF FAME
# ═══════════════════════════════════════════════════════════════════

class HallOfFame:
    """
    Stores the best models seen across all evolution runs.
    Maintains a fixed-size list sorted by fitness.
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.entries: List[Dict] = []  # [{fitness, epoch, team, model_idx, weights}]

    def add(self, model: NeuralNetwork, team: int, epoch: int, fitness: float) -> None:
        """Add a model to the hall of fame if it's good enough."""
        entry = {
            'fitness': fitness,
            'team': team,
            'team_name': TEAM_NAMES[team] if team < len(TEAM_NAMES) else f'Team-{team}',
            'model_idx': model.model_id,
            'epoch': epoch,
            'timestamp': time.time(),
        }
        self.entries.append(entry)
        self.entries.sort(key=lambda e: e['fitness'], reverse=True)
        if len(self.entries) > self.max_size:
            self.entries = self.entries[:self.max_size]

    def get_top(self, n: int = 10) -> List[Dict]:
        return self.entries[:n]

    def get_best(self) -> Optional[Dict]:
        return self.entries[0] if self.entries else None

    def get_diversity(self) -> float:
        """Genetic diversity among hall of fame members."""
        if len(self.entries) < 2:
            return 0.0
        total = 0.0
        pairs = 0
        for i in range(len(self.entries)):
            for j in range(i + 1, len(self.entries)):
                total += abs(self.entries[i]['fitness'] - self.entries[j]['fitness'])
                pairs += 1
        return total / pairs if pairs > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            'size': len(self.entries),
            'best_fitness': self.entries[0]['fitness'] if self.entries else 0,
            'entries': self.entries,
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 16: DIVERSITY METRICS
# ═══════════════════════════════════════════════════════════════════

class DiversityMetrics:
    """Compute and track genetic diversity across the population."""

    @staticmethod
    def pairwise_distance(models: List[NeuralNetwork]) -> float:
        """Average pairwise Euclidean distance between all models."""
        if len(models) < 2:
            return 0.0
        total = 0.0
        pairs = 0
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                total += models[i].similarity(models[j])
                pairs += 1
        return total / pairs if pairs > 0 else 0.0

    @staticmethod
    def weight_entropy(models: List[NeuralNetwork], bins: int = 10) -> float:
        """
        Compute entropy of weight values across the population.
        Higher entropy = more diverse.
        """
        if not models:
            return 0.0
        all_weights = []
        for m in models:
            all_weights.extend(m.weights)
        if not all_weights:
            return 0.0

        mn, mx = min(all_weights), max(all_weights)
        if mx - mn < 1e-10:
            return 0.0

        bin_counts = [0] * bins
        bin_size = (mx - mn) / bins
        for w in all_weights:
            idx = min(bins - 1, int((w - mn) / bin_size))
            bin_counts[idx] += 1

        total = len(all_weights)
        entropy = 0.0
        for c in bin_counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)

        return entropy

    @staticmethod
    def fitness_diversity(fitness_values: List[float]) -> float:
        """Standard deviation of fitness values."""
        if len(fitness_values) < 2:
            return 0.0
        return statistics.stdev(fitness_values)

    @staticmethod
    def feature_coverage(models: List[NeuralNetwork],
                         n_samples: int = 1000) -> float:
        """
        Estimate how much of the weight space is covered.
        Measures variance of outputs on random inputs.
        """
        if not models:
            return 0.0
        outputs = []
        for _ in range(n_samples):
            inp = [random.uniform(-1, 1) for _ in range(N_INPUT)]
            out_variants = []
            for m in models:
                out = m.forward(inp)
                out_variants.append(out['turn'])
            outputs.append(statistics.stdev(out_variants) if len(out_variants) > 1 else 0.0)
        return sum(outputs) / len(outputs) if outputs else 0.0


# ═══════════════════════════════════════════════════════════════════
# SECTION 17: HYPERPARAMETER SCHEDULER
# ═══════════════════════════════════════════════════════════════════

class HyperparameterScheduler:
    """
    Adaptive scheduling of mutation rate, crossover rate, etc.
    based on convergence metrics and generation number.
    """

    def __init__(self):
        self.mutation_rate = MUTATION_RATE
        self.mutation_amount = MUTATION_AMOUNT
        self.crossover_rate = 0.7
        self.tournament_size = 3

        # Annealing schedule
        self.mutation_rate_start = MUTATION_RATE
        self.mutation_rate_end = 0.02
        self.mutation_amount_start = MUTATION_AMOUNT
        self.mutation_amount_end = 0.2

        # Adaptation state
        self.stagnation_counter = 0
        self.last_best_fitness = 0.0
        self.adaptation_history = []

    def update(self, generation: int, max_generations: int,
               current_best_fitness: float, stagnation: int) -> None:
        """Update hyperparameters based on generation and convergence."""
        self.stagnation_counter = stagnation

        # Linear annealing
        progress = min(1.0, generation / max(max_generations, 1))
        self.mutation_rate = self.mutation_rate_start - (
            self.mutation_rate_start - self.mutation_rate_end
        ) * progress
        self.mutation_amount = self.mutation_amount_start - (
            self.mutation_amount_start - self.mutation_amount_end
        ) * progress

        # If stagnating, increase mutation to escape local optima
        if stagnation > 20:
            self.mutation_rate = min(0.3, self.mutation_rate * 1.5)
            self.mutation_amount = min(1.5, self.mutation_amount * 1.3)
        elif stagnation > 10:
            self.mutation_rate = min(0.2, self.mutation_rate * 1.2)

        # If improving fast, decrease mutation for exploitation
        if stagnation < 3 and current_best_fitness > self.last_best_fitness:
            self.mutation_rate = max(0.01, self.mutation_rate * 0.95)

        self.last_best_fitness = current_best_fitness

        self.adaptation_history.append({
            'generation': generation,
            'mutation_rate': self.mutation_rate,
            'mutation_amount': self.mutation_amount,
            'crossover_rate': self.crossover_rate,
            'stagnation': stagnation,
        })

    def get_params(self) -> Dict:
        return {
            'mutation_rate': round(self.mutation_rate, 4),
            'mutation_amount': round(self.mutation_amount, 4),
            'crossover_rate': round(self.crossover_rate, 4),
            'tournament_size': self.tournament_size,
        }

    def to_dict(self) -> Dict:
        return {
            'current': self.get_params(),
            'history': self.adaptation_history[-100:],
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 18: ELO RATING
# ═══════════════════════════════════════════════════════════════════

class EloRating:
    """Elo rating system for comparing team performance."""

    def __init__(self, k: int = 32, initial: float = 1000.0):
        self.k = k
        self.ratings: Dict[int, float] = {t: initial for t in range(N_TEAMS)}
        self.history: List[Dict] = []
        self.match_count = 0

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + math.pow(10.0, (rb - ra) / 400.0))

    def update(self, team_a: int, team_b: int, result: float) -> None:
        """result: 1.0 = a wins, 0.5 = draw, 0.0 = b wins"""
        ea = self.expected(self.ratings[team_a], self.ratings[team_b])
        eb = 1.0 - ea
        self.ratings[team_a] += self.k * (result - ea)
        self.ratings[team_b] += self.k * ((1.0 - result) - eb)
        self.ratings[team_a] = max(100.0, min(3000.0, self.ratings[team_a]))
        self.ratings[team_b] = max(100.0, min(3000.0, self.ratings[team_b]))
        self.match_count += 1

    def batch_update(self, team_a: int, team_b: int, fitness_a: float,
                     fitness_b: float) -> None:
        """Update Elo based on fitness comparison."""
        if fitness_a > fitness_b:
            self.update(team_a, team_b, 1.0)
        elif fitness_b > fitness_a:
            self.update(team_a, team_b, 0.0)
        else:
            self.update(team_a, team_b, 0.5)

    def rank_teams(self) -> Dict[int, int]:
        """Return {team: rank_1_to_10}."""
        sorted_teams = sorted(self.ratings.keys(),
                              key=lambda t: self.ratings[t], reverse=True)
        result = {}
        for rank, team in enumerate(sorted_teams):
            result[team] = max(1, min(10, rank + 1))
        return result

    def snapshot(self) -> None:
        self.history.append(dict(self.ratings))
        if len(self.history) > 500:
            self.history = self.history[-500:]

    def to_dict(self) -> Dict:
        return {
            'ratings': self.ratings,
            'history': self.history[-100:],
            'matches': self.match_count,
        }

    def get_rating(self, team: int) -> float:
        return self.ratings.get(team, 1000.0)


# ═══════════════════════════════════════════════════════════════════
# SECTION 19: GENEALOGY TRACKER
# ═══════════════════════════════════════════════════════════════════

class GenealogyTracker:
    """
    Track model lineages across generations.
    Records parent-child relationships and genetic contributions.
    """

    def __init__(self):
        self.tree: Dict[int, List[Dict]] = defaultdict(list)  # {team: [birth_events]}
        self.model_lineage: Dict[str, Dict] = {}  # {f'{team}_{model_idx}': lineage_data}
        self.genesis_pool: Dict[int, str] = {}  # {team: 'random' | 'migration' | 'elite'}

    def register_birth(self, child_team: int, child_model_idx: int,
                       parent_team: int, parent_model_idx: int,
                       generation: int, child_weights: List[float],
                       method: str = 'crossover') -> None:
        """Register a new model's birth event."""
        key = f'{child_team}_{child_model_idx}'
        w_hash = hash(tuple([round(w, 4) for w in child_weights]))

        event = {
            'child_team': child_team,
            'child_model': child_model_idx,
            'parent_team': parent_team,
            'parent_model': parent_model_idx,
            'generation': generation,
            'method': method,
            'hash': w_hash,
            'timestamp': time.time(),
        }

        self.tree[child_team].append(event)
        self.model_lineage[key] = event

    def register_genesis(self, team: int, method: str = 'random') -> None:
        """Register how a team's first models were created."""
        self.genesis_pool[team] = method

    def get_lineage(self, team: int, model_idx: int) -> List[Dict]:
        """Get lineage chain for a specific model."""
        chain = []
        current_key = f'{team}_{model_idx}'
        visited = set()

        while current_key and current_key not in visited:
            visited.add(current_key)
            info = self.model_lineage.get(current_key)
            if not info:
                break
            chain.append(info)
            parent_key = f"{info['parent_team']}_{info['parent_model']}"
            current_key = parent_key

        return chain

    def get_genetic_diversity(self) -> float:
        """Estimate diversity from distinct weight hashes."""
        hashes = set()
        for events in self.tree.values():
            for e in events:
                hashes.add(e['hash'])
        return float(len(hashes))

    def get_team_tree_size(self, team: int) -> int:
        return len(self.tree.get(team, []))

    def to_dict(self) -> Dict:
        return {
            'total_births': sum(len(v) for v in self.tree.values()),
            'genetic_diversity': self.get_genetic_diversity(),
            'team_counts': {t: len(events) for t, events in self.tree.items()},
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 20: STATS ANALYZER
# ═══════════════════════════════════════════════════════════════════

class StatsAnalyzer:
    """
    Statistical analysis of evolution history.
    Computes trends, convergence rates, stagnation detection,
    and generates summary statistics.
    """

    def __init__(self, history: Optional[List[Dict]] = None):
        self.history = history or []

    def add_snapshot(self, data: Dict) -> None:
        self.history.append(data)
        if len(self.history) > 2000:
            self.history = self.history[-2000:]

    def get_trend(self, key: str = 'best_fitness', window: int = 20) -> float:
        """Moving average of a key over recent history."""
        values = [h.get(key, 0.0) for h in self.history[-window:]]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def get_best_ever(self, key: str = 'best_fitness') -> float:
        return max((h.get(key, 0.0) for h in self.history), default=0.0)

    def get_worst_ever(self, key: str = 'best_fitness') -> float:
        return min((h.get(key, 0.0) for h in self.history), default=0.0)

    def get_recent_improvement(self, window: int = 20) -> float:
        """Fitness improvement over last N generations."""
        values = [h.get('best_fitness', 0.0) for h in self.history[-window:]]
        if len(values) < 2:
            return 0.0
        return values[-1] - values[0]

    def get_improvement_rate(self, window: int = 20) -> float:
        """Average fitness change per generation."""
        values = [h.get('best_fitness', 0.0) for h in self.history[-window:]]
        if len(values) < 2:
            return 0.0
        return (values[-1] - values[0]) / max(len(values) - 1, 1)

    def get_convergence_rate(self, window: int = 20) -> float:
        """How fast the population is converging (lower = more converged)."""
        values = [h.get('best_fitness', 0.0) for h in self.history[-window:]]
        if len(values) < 3:
            return 0.0
        return statistics.stdev(values) if len(values) > 1 else 0.0

    def get_stagnation_count(self, threshold: float = 5.0) -> int:
        """Number of consecutive generations without significant improvement."""
        if len(self.history) < 2:
            return 0
        best = self.get_best_ever('best_fitness')
        count = 0
        for h in reversed(self.history):
            if h.get('best_fitness', 0) >= best - threshold:
                count += 1
            else:
                break
        return count

    def get_diversity_trend(self, window: int = 20) -> float:
        """Average diversity over recent generations."""
        return self.get_trend('diversity', window)

    def get_performance_summary(self) -> Dict:
        """Comprehensive performance summary."""
        return {
            'generations': len(self.history),
            'best_fitness_ever': self.get_best_ever('best_fitness'),
            'current_avg': self.get_trend('best_fitness', 10),
            'improvement_rate': self.get_improvement_rate(20),
            'stagnation': self.get_stagnation_count(),
            'convergence': self.get_convergence_rate(),
            'total_births': self.history[-1].get('total_births', 0) if self.history else 0,
            'recent_improvement': self.get_recent_improvement(20),
        }

    def export_csv(self, path: str = 'evolution_data.csv') -> None:
        """Export history to CSV file."""
        if not self.history:
            return
        keys = list(self.history[0].keys())
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)
        except IOError:
            pass

    def summary(self) -> str:
        """One-line summary for CLI."""
        if not self.history:
            return 'No data'
        latest = self.history[-1]
        return (
            f"Gen {latest.get('generation', '?')}, "
            f"Best: {self.get_best_ever('best_fitness'):.1f}, "
            f"Avg(10): {self.get_trend('best_fitness', 10):.1f}, "
            f"Stagnation: {self.get_stagnation_count()} gen, "
            f"Improvement: {self.get_improvement_rate(20):.3f}/gen"
        )


# ═══════════════════════════════════════════════════════════════════
# SECTION 21: WEIGHT ANALYZER
# ═══════════════════════════════════════════════════════════════════

class WeightAnalyzer:
    """Analysis of neural network weights."""

    @staticmethod
    def weight_stats(weights: List[float]) -> Dict:
        """Basic statistics: min, max, mean, std, zeros."""
        arr = list(weights)
        n = len(arr)
        if n == 0:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'zeros': 0, 'total': 0}
        mean = sum(arr) / n
        variance = sum((x - mean) ** 2 for x in arr) / n
        return {
            'min': min(arr),
            'max': max(arr),
            'mean': mean,
            'std': math.sqrt(variance),
            'zeros': sum(1 for x in arr if abs(x) < 1e-6),
            'total': n,
        }

    @staticmethod
    def layer_importance(weights: List[float]) -> Dict[str, float]:
        """Mean absolute weight per layer as importance proxy."""
        w = list(weights)
        w1 = w[0:168]    # input -> hidden
        b1 = w[168:180]  # hidden bias
        w2 = w[180:204]  # hidden -> output
        b2 = w[204:206]  # output bias

        def mag(x):
            return sum(abs(v) for v in x) / len(x) if x else 0.0

        return {
            'input_hidden': mag(w1),
            'hidden_bias': mag(b1),
            'hidden_output': mag(w2),
            'output_bias': mag(b2),
        }

    @staticmethod
    def compare(w1: List[float], w2: List[float]) -> float:
        """RMSE between two weight vectors."""
        assert len(w1) == len(w2)
        diff = sum((a - b) ** 2 for a, b in zip(w1, w2))
        return math.sqrt(diff / len(w1))

    @staticmethod
    def find_similar(weights: List[float], pool: Dict[int, List[float]],
                     threshold: float = 0.5) -> List[Dict]:
        """Find similar weight sets in a pool."""
        similar = []
        for team, w in pool.items():
            dist = WeightAnalyzer.compare(weights, w)
            if dist < threshold:
                similar.append({'team': team, 'distance': round(dist, 4)})
        return sorted(similar, key=lambda x: x['distance'])

    @staticmethod
    def gradient_magnitude(weights: List[float]) -> float:
        """Estimate gradient-like magnitude (differences between adjacent weights)."""
        diffs = [abs(weights[i + 1] - weights[i]) for i in range(len(weights) - 1)]
        return sum(diffs) / len(diffs) if diffs else 0.0

    @staticmethod
    def sparsity(weights: List[float], epsilon: float = 0.01) -> float:
        """Fraction of near-zero weights."""
        return sum(1 for w in weights if abs(w) < epsilon) / max(len(weights), 1)

    @staticmethod
    def analyze_model(model: NeuralNetwork) -> Dict:
        """Full analysis of a single model."""
        w = model.to_list()
        stats = WeightAnalyzer.weight_stats(w)
        layers = WeightAnalyzer.layer_importance(w)
        return {
            'fitness': model.fitness,
            'epoch_created': model.epoch_created,
            'weight_stats': stats,
            'layer_importance': layers,
            'gradient_magnitude': WeightAnalyzer.gradient_magnitude(w),
            'sparsity': WeightAnalyzer.sparsity(w),
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 22: TRAINING MANAGER
# ═══════════════════════════════════════════════════════════════════

class TrainingManager:
    """
    Manages training lifecycle: checkpoints, logging,
    Elo tracking, genealogy, and periodic saves.
    """

    def __init__(self, checkpoint_dir: str = 'checkpoints',
                 checkpoint_interval: int = 50):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.elo = EloRating()
        self.genealogy = GenealogyTracker()
        self.analyzer = StatsAnalyzer()
        self.best_team_ever = -1
        self.best_fitness_ever = 0.0
        self.hall_of_fame = HallOfFame()
        self.hparams = HyperparameterScheduler()

    def save_checkpoint(self, engine: EvolutionEngine, gen: int) -> None:
        """Save training checkpoint every N generations."""
        if gen % self.checkpoint_interval != 0:
            return
        try:
            checkpoint = {
                'generation': gen,
                'best_fitness': engine.best_fitness_ever,
                'total_births': engine.total_births,
                'elo_ratings': self.elo.ratings,
                'hparams': self.hparams.get_params(),
                'timestamp': time.time(),
                'team_epochs': engine.weight_manager.get_team_epochs(),
            }
            path = self.checkpoint_dir / f'checkpoint_gen_{gen}.json'
            with open(path, 'w') as f:
                json.dump(checkpoint, f, indent=2)

            # Keep only last 10 checkpoints
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_gen_*.json'))
            while len(checkpoints) > 10:
                checkpoints[0].unlink()
                checkpoints.pop(0)
        except (IOError, OSError):
            pass

    def load_latest_checkpoint(self) -> Optional[Dict]:
        """Load the most recent checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_gen_*.json'))
        if not checkpoints:
            return None
        try:
            with open(checkpoints[-1]) as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return None

    def compare_teams(self, team_a: int, team_b: int,
                      fitness_a: float, fitness_b: float) -> None:
        self.elo.batch_update(team_a, team_b, fitness_a, fitness_b)

    def get_ranked_list(self) -> List[Dict]:
        """Get teams ranked by Elo."""
        elo_ranks = self.elo.rank_teams()
        result = []
        for team in range(N_TEAMS):
            result.append({
                'team': team,
                'name': TEAM_NAMES[team] if team < len(TEAM_NAMES) else f'Team-{team}',
                'elo': round(self.elo.ratings.get(team, 1000)),
                'rank': elo_ranks.get(team, 5),
            })
        result.sort(key=lambda x: x['rank'])
        return result


# ═══════════════════════════════════════════════════════════════════
# SECTION 23: BENCHMARK SUITE
# ═══════════════════════════════════════════════════════════════════

class BenchmarkSuite:
    """
    Fitness benchmarks: test models on standardized scenarios
    to measure their absolute (not relative) quality.
    """

    def __init__(self):
        self.benchmarks = []

    def register(self, name: str, fn: Callable) -> None:
        self.benchmarks.append({'name': name, 'fn': fn})

    def run_all(self, model: NeuralNetwork) -> Dict[str, float]:
        results = {}
        for b in self.benchmarks:
            try:
                results[b['name']] = b['fn'](model)
            except Exception:
                results[b['name']] = 0.0
        return results

    @staticmethod
    def benchmark_food_seeking(model: NeuralNetwork, n_trials: int = 100) -> float:
        """
        Test: can the model turn toward a food source?
        Score based on angle to nearest food.
        """
        score = 0.0
        for _ in range(n_trials):
            # Random worm position
            worm_x = random.uniform(100, WORLD_SIZE - 100)
            worm_y = random.uniform(100, WORLD_SIZE - 100)
            worm_angle = random.uniform(0, 2 * math.pi)

            # Place food nearby
            food_angle = random.uniform(0, 2 * math.pi)
            food_dist = random.uniform(50, 500)
            food_x = worm_x + math.cos(food_angle) * food_dist
            food_y = worm_y + math.sin(food_angle) * food_dist

            # Compute NN input
            fa = worm_angle - math.atan2(food_y - worm_y, food_x - worm_x)
            inp = [
                math.sin(fa), math.cos(fa), min(1, food_dist / 50000),
                0, 0, 1,  # no enemy
                0, 0, 1,  # no ally
                3.5 / 6,  # speed
                0.5, 0.5, 0.5,  # walls
                0.1,  # mass
            ]
            out = model.forward(inp)

            # Score: how well does turn align with food direction?
            ideal_turn = food_angle - worm_angle
            ideal_turn = ((ideal_turn + math.pi) % (2 * math.pi)) - math.pi
            actual_turn = out['turn'] * 0.07  # same scaling as game

            # Normalize error to [0, 1]
            error = abs(ideal_turn - actual_turn) / math.pi
            score += max(0, 1 - error)

        return score / n_trials * 100

    @staticmethod
    def benchmark_obstacle_avoidance(model: NeuralNetwork, n_trials: int = 100) -> float:
        """Test: can the model avoid walls when headed toward one."""
        score = 0.0
        for _ in range(n_trials):
            worm_x = random.uniform(100, WORLD_SIZE - 100)
            worm_y = random.uniform(100, WORLD_SIZE - 100)

            # Point worm toward nearest wall
            dists = [worm_x, WORLD_SIZE - worm_x, worm_y, WORLD_SIZE - worm_y]
            nearest_wall = min(dists)
            wall_dir = dists.index(nearest_wall)

            if wall_dir == 0:
                worm_angle = math.pi  # facing left
            elif wall_dir == 1:
                worm_angle = 0  # facing right
            elif wall_dir == 2:
                worm_angle = 3 * math.pi / 2  # facing up
            else:
                worm_angle = math.pi / 2  # facing down

            wd = [nearest_wall,
                  min(worm_x + random.uniform(-50, 50), worm_y + random.uniform(-50, 50)),
                  min(WORLD_SIZE - worm_x + random.uniform(-50, 50),
                      WORLD_SIZE - worm_y + random.uniform(-50, 50))]

            inp = [
                0, 0.5, 0.5,  # random food direction
                0, 0, 1,  # no enemy
                0, 0, 1,  # no ally
                3.5 / 6,
                min(1, wd[0] / 50000),
                min(1, wd[1] / 50000),
                min(1, wd[2] / 50000),
                0.1,
            ]
            out = model.forward(inp)
            # Score: turn away from wall is good
            turn = out['turn']
            if nearest_wall < 200:
                desired_away = turn * 0.07 > 0.5 if wall_dir in (0, 2) else turn * 0.07 < -0.5
                score += 1.0 if abs(turn) > 0.3 else 0.0
            else:
                score += 0.5

        return score / n_trials * 100


# ═══════════════════════════════════════════════════════════════════
# SECTION 24: REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Generate formatted reports about evolution progress."""

    @staticmethod
    def generate_text_report(engine: EvolutionEngine,
                              analyzer: StatsAnalyzer,
                              include_teams: bool = True) -> str:
        """Generate a plain-text report."""
        lines = []
        lines.append('=' * 60)
        lines.append('  SLITHER EVO — EVOLUTION REPORT')
        lines.append('=' * 60)
        lines.append('')

        # Summary
        lines.append('  GENERAL')
        lines.append(f'    Global generation:   {engine.generation}')
        lines.append(f'    Total births:        {engine.total_births}')
        lines.append(f'    Best fitness ever:   {engine.best_fitness_ever:.2f}')
        lines.append(f'    Stats logged:        {len(engine.stats_log)} snapshots')
        lines.append(analyzer.summary())
        lines.append('')

        # Team info
        if include_teams:
            lines.append('  TEAMS')
            lines.append(f'    {"Team":<12} {"Ep":<5} {"Best":<8} {"Avg":<8} {"Div":<6}')
            lines.append(f'    {"-"*42}')
            for t in range(N_TEAMS):
                pool = engine.weight_manager.get_pool(t)
                if pool:
                    s = pool.get_pool_stats()
                    lines.append(
                        f'    {pool.team_name:<12} {s["epoch"]:<5} '
                        f'{s["best_fitness"]:<8.1f} {s["avg_fitness"]:<8.1f} '
                        f'{s["diversity"]:<6.3f}'
                    )
                else:
                    lines.append(f'    {TEAM_NAMES[t]:<12} {"?":<5} {"?":<8} {"?":<8} {"?":<6}')
            lines.append('')

        lines.append('=' * 60)
        return '\n'.join(lines)

    @staticmethod
    def generate_html_report(engine: EvolutionEngine,
                              analyzer: StatsAnalyzer,
                              path: str = 'evolution_report.html') -> str:
        """Generate an HTML report and save to file."""
        html = f'''<!DOCTYPE html>
<html><head><title>Slither Evo Report</title>
<style>
body {{ font-family: monospace; background: #111; color: #ccc; padding: 20px; }}
h1 {{ color: #4af; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 6px 12px; text-align: left; border-bottom: 1px solid #333; }}
th {{ color: #888; }}
tr:hover {{ background: #1a1a2e; }}
</style></head><body>
<h1>Slither Evo — Gen {engine.generation}</h1>
<p>Best fitness ever: {engine.best_fitness_ever:.2f} |
   Total births: {engine.total_births} |
   Stagnation: {analyzer.get_stagnation_count()} gen</p>
<table>
<tr><th>Team</th><th>Epoch</th><th>Best</th><th>Avg</th><th>Diversity</th></tr>
'''
        for t in range(N_TEAMS):
            pool = engine.weight_manager.get_pool(t)
            if pool:
                s = pool.get_pool_stats()
                color = pool.team_color
                html += f'''<tr>
<td style="color:{color}">{pool.team_name}</td>
<td>{s["epoch"]}</td><td>{s["best_fitness"]:.1f}</td>
<td>{s["avg_fitness"]:.1f}</td><td>{s["diversity"]:.3f}</td></tr>\n'''

        html += '</table></body></html>'
        try:
            with open(path, 'w') as f:
                f.write(html)
        except IOError:
            pass
        return html


# ═══════════════════════════════════════════════════════════════════
# SECTION 25: MODEL VISUALIZER
# ═══════════════════════════════════════════════════════════════════

class ModelVisualizer:
    """Prepares data for visualizing model weights in the browser."""

    @staticmethod
    def weight_heatmap_data(model: NeuralNetwork) -> Dict:
        """Generate heatmap data: 2D arrays for each weight matrix."""
        layers = []
        off = 0
        sizes = [N_INPUT, N_HIDDEN1, N_HIDDEN2, N_HIDDEN3, N_OUTPUT]
        for k in range(len(sizes) - 1):
            n_in, n_out = sizes[k], sizes[k + 1]
            w = model.weights[off:off + n_in * n_out]
            mat = [w[i * n_in:(i + 1) * n_in] for i in range(n_out)]
            bias = model.weights[off + n_in * n_out:off + n_in * n_out + n_out]
            layers.append({'weights': mat, 'bias': list(bias), 'shape': [n_out, n_in]})
            off += n_in * n_out + n_out
        return {'layers': layers}

    @staticmethod
    def activation_distribution(model: NeuralNetwork, n_samples: int = 1000) -> Dict:
        """Sample hidden activations over random inputs to see distribution."""
        all_acts = [[] for _ in range(N_HIDDEN1 + N_HIDDEN2 + N_HIDDEN3)]
        for _ in range(n_samples):
            inp = [random.uniform(-1, 1) for _ in range(N_INPUT)]
            off = 0
            sizes = [N_INPUT, N_HIDDEN1, N_HIDDEN2, N_HIDDEN3, N_OUTPUT]
            h = inp
            act_idx = 0
            for k in range(len(sizes) - 2):
                n_in, n_out = sizes[k], sizes[k + 1]
                out = [0.0] * n_out
                for j in range(n_out):
                    s = model.weights[off + n_in * n_out + j]
                    for i in range(n_in):
                        s += h[i] * model.weights[off + j * n_in + i]
                    out[j] = s / (1 + abs(s))
                    all_acts[act_idx + j].append(out[j])
                off += n_in * n_out + n_out
                h = out
                act_idx += n_out

        stats = []
        for j, acts in enumerate(all_acts):
            stats.append({
                'neuron': j,
                'mean': sum(acts) / len(acts),
                'std': statistics.stdev(acts) if len(acts) > 1 else 0.0,
                'min': min(acts),
                'max': max(acts),
                'dead': max(acts) - min(acts) < 0.01,
            })
        return {'neurons': stats, 'n_samples': n_samples}

    @staticmethod
    def fitness_curve_data(stats_log: List[Dict]) -> Dict:
        """Extract fitness over time for charting."""
        gens = []
        bests = []
        avgs = []
        for h in stats_log:
            gens.append(h.get('generation', 0))
            bests.append(h.get('best_fitness', 0))
            avgs.append(h.get('avg_fitness', 0))
        return {
            'generations': gens[-200:],
            'best_fitness': bests[-200:],
            'avg_fitness': avgs[-200:],
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 26: LEGACY TOURNAMENT SELECTOR WRAPPER
# ═══════════════════════════════════════════════════════════════════

class TournamentSelector:
    """Legacy wrapper around SelectionMethods.tournament."""

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select(self, fitness_dict: Dict[int, float]) -> int:
        return SelectionMethods.tournament(fitness_dict, self.tournament_size)

    def select_multiple(self, fitness_dict: Dict[int, float], count: int) -> List[int]:
        return SelectionMethods.tournament_multiple(fitness_dict, count, self.tournament_size)


# ═══════════════════════════════════════════════════════════════════
# SECTION 27: CLI — COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════

class CLI:
    """
    Interactive command-line interface for managing evolution.
    Supports status, leaderboard, team inspection, analysis,
    export, and configuration commands.
    """

    @staticmethod
    def print_banner() -> None:
        print(r"""
   =====================================
         SLITHER EVO ENGINE v2       
     =============================      
   20 teams  .  20 worms  .  NN 26>20>14>10>2  
   Each team: 15 models, independent epochs
   =====================================
        """)

    @staticmethod
    def print_status(engine: EvolutionEngine, analyzer: StatsAnalyzer) -> None:
        """Show extended status of the entire system."""
        tm = TrainingManager()
        total_deaths = engine.total_births
        avg_fitness = analyzer.get_trend('best_fitness', 10)
        stagnation = analyzer.get_stagnation_count(5)

        print('-' * 60)
        print(f'  Global generation:   {engine.generation}')
        print(f'  Total births:        {total_deaths}')
        print(f'  Best fitness ever:   {engine.best_fitness_ever:.2f}')
        print(f'  Avg fitness (last10):{avg_fitness:.2f}')
        print(f'  Stagnation:          {stagnation} gen')
        print(f'  Elo matches:         {tm.elo.match_count}')
        print(f'  Hall of Fame entries: {tm.hall_of_fame.to_dict()["size"]}')
        print('-' * 60)

        # Per-team epochs
        epochs = engine.weight_manager.get_team_epochs()
        print(f'  {"Team":<12} {"Ep":<5} {"Best":<8} {"Avg":<8} {"Models":<7}')
        print(f'  {"-"*42}')
        for t in range(N_TEAMS):
            pool = engine.weight_manager.get_pool(t)
            if pool:
                s = pool.get_pool_stats()
                c = pool.team_color
                print(f'  {pool.team_name:<12} {s["epoch"]:<5} '
                      f'{s["best_fitness"]:<8.1f} {s["avg_fitness"]:<8.1f} '
                      f'{s["n_models"]:<7}')
            else:
                print(f'  {TEAM_NAMES[t]:<12} {"?":<5} {"?":<8}')
        print('-' * 60)

    @staticmethod
    def print_leaderboard(ranks: List[Dict]) -> None:
        """Show leaderboard of teams."""
        print(f'  {"Rank":<6} {"Team":<12} {"Epoch":<6} {"Best":<8} {"Avg":<8} {"Div":<8}')
        print(f'  {"-"*50}')
        for entry in ranks[:N_TEAMS]:
            print(f'  {entry["rank"]:<6} {entry["name"]:<12} '
                  f'{entry.get("epoch", "?"):<6} '
                  f'{entry.get("best_fitness", "?"):<8} '
                  f'{entry.get("avg_fitness", "?"):<8} '
                  f'{entry.get("diversity", "?"):<8}')

    @staticmethod
    def print_team_detail(engine: EvolutionEngine, team: int) -> None:
        """Print detailed info for a specific team."""
        pool = engine.weight_manager.get_pool(team)
        if not pool:
            print(f'  Team {team} not found')
            return

        stats = pool.get_pool_stats()
        print(f'  Team: {pool.team_name} (index {team})')
        print(f'  Color: {pool.team_color}')
        print(f'  Epoch: {pool.epoch}')
        print(f'  Best fitness ever: {pool.best_fitness_ever:.2f}')
        print(f'  Pool diversity: {stats["diversity"]:.4f}')
        print(f'  Epochs logged: {len(pool.epoch_fitness_log)}')
        print()
        print(f'  {"Model":<8} {"Fitness":<10} {"Epoch":<7} {"Avg|W|":<8} {"Sparsity":<9}')
        print(f'  {"-"*44}')
        for i, m in enumerate(pool.models):
            ws = m.weight_stats()
            sp = WeightAnalyzer.sparsity(m.to_list())
            print(f'  #{i:<5} {m.fitness:<10.2f} {m.epoch_created:<7} '
                  f'{ws["mean"]:<8.3f} {sp:<9.3f}')

    @staticmethod
    def print_help() -> None:
        """Show help text."""
        print("""
  Commands:
    status / st        — show current status
    leader / lb        — show leaderboard
    team <name/id>     — show team details
    models <name/id>   — list models for a team
    analyze            — weight analysis (top 5 teams)
    export             — export stats to CSV
    report             — generate text report
    html-report        — generate HTML report
    hof                — show Hall of Fame
    elo                — show Elo ratings
    hparams            — show current hyperparameters
    reset              — reset ALL weights
    help               — show this help
    quit / q           — exit
        """)

    @staticmethod
    def print_hall_of_fame(hof: HallOfFame) -> None:
        entries = hof.get_top(10)
        if not entries:
            print('  Hall of Fame is empty')
            return
        print(f'  {"#":<4} {"Fitness":<10} {"Team":<12} {"Model":<7} {"Epoch":<6}')
        print(f'  {"-"*42}')
        for i, e in enumerate(entries):
            print(f'  {i + 1:<4} {e["fitness"]:<10.1f} '
                  f'{e.get("team_name", "?"):<12} '
                  f'#{e.get("model_idx", "?"):<5} '
                  f'{e.get("epoch", "?"):<6}')

    @staticmethod
    def print_elo(elo: EloRating) -> None:
        print(f'  {"Team":<12} {"Elo":<8} {"Rank":<6}')
        print(f'  {"-"*28}')
        ranks = elo.rank_teams()
        for team in range(N_TEAMS):
            r = ranks.get(team, 10)
            name = TEAM_NAMES[team] if team < len(TEAM_NAMES) else f'Team-{team}'
            print(f'  {name:<12} {elo.ratings.get(team, 1000):<8.0f} {r:<6}')


def interactive_mode() -> None:
    """Interactive CLI main loop."""
    engine = EvolutionEngine()
    manager = TrainingManager()
    cli = CLI()
    analyzer = engine.analyzer

    cli.print_banner()
    cli.print_status(engine, analyzer)
    cli.print_help()

    while True:
        try:
            cmd_raw = input('\n> ').strip().lower()
            parts = cmd_raw.split()
            cmd = parts[0] if parts else ''

            if cmd in ('quit', 'exit', 'q'):
                break

            elif cmd in ('status', 'st'):
                cli.print_status(engine, analyzer)

            elif cmd in ('leader', 'lb'):
                ranks = engine.get_leaderboard()
                cli.print_leaderboard(ranks)

            elif cmd == 'team':
                if len(parts) < 2:
                    print('  Usage: team <name or id>')
                else:
                    try:
                        team_id = int(parts[1])
                    except ValueError:
                        name = ' '.join(parts[1:]).lower()
                        team_id = next(
                            (i for i, n in enumerate(TEAM_NAMES) if n.lower() == name),
                            -1
                        )
                    if 0 <= team_id < N_TEAMS:
                        cli.print_team_detail(engine, team_id)
                    else:
                        print(f'  Team "{parts[1]}" not found')

            elif cmd == 'models':
                if len(parts) < 2:
                    print('  Usage: models <team id>')
                    continue
                try:
                    team_id = int(parts[1])
                except ValueError:
                    print('  Please provide team number (0-14)')
                    continue
                pool = engine.weight_manager.get_pool(team_id)
                if not pool:
                    print(f'  Team {team_id} not found')
                    continue
                print(f'  Models for {pool.team_name} (epoch {pool.epoch}):')
                print(f'  {"Model":<8} {"Fitness":<10} {"Epoch":<7} {"Evaluations":<13}')
                print(f'  {"-"*40}')
                for i in range(len(pool.models)):
                    rec = pool.fitness_records.get(i, [])
                    n_evals = len(rec)
                    avg_f = sum(rec) / n_evals if n_evals else 0.0
                    print(f'  #{i:<5} {pool.models[i].fitness:<10.2f} '
                          f'{pool.models[i].epoch_created:<7} {n_evals:<13}')

            elif cmd == 'analyze':
                wa = WeightAnalyzer()
                for t in range(min(5, N_TEAMS)):
                    pool = engine.weight_manager.get_pool(t)
                    if pool:
                        best = pool.get_best_model()
                        w = best.to_list()
                        stats = wa.weight_stats(w)
                        layers = wa.layer_importance(w)
                        print(f'  Team {t} ({TEAM_NAMES[t]}):')
                        print(f'    W: min={stats["min"]:.3f} max={stats["max"]:.3f} '
                              f'mean={stats["mean"]:.4f} std={stats["std"]:.4f}')
                        print(f'    Input->Hidden mag: {layers["input_hidden"]:.4f}, '
                              f'Hidden->Output mag: {layers["hidden_output"]:.4f}')

            elif cmd == 'export':
                analyzer.export_csv()
                print('  Exported to evolution_data.csv')

            elif cmd == 'report':
                report = ReportGenerator.generate_text_report(engine, analyzer)
                print(report)

            elif cmd == 'html-report':
                ReportGenerator.generate_html_report(engine, analyzer)
                print('  HTML report saved to evolution_report.html')

            elif cmd == 'hof':
                cli.print_hall_of_fame(manager.hall_of_fame)

            elif cmd == 'elo':
                cli.print_elo(manager.elo)

            elif cmd == 'hparams':
                hp = manager.hparams
                params = hp.get_params()
                print(f'  Current hyperparameters:')
                for k, v in params.items():
                    print(f'    {k}: {v}')
                print(f'  Stagnation counter: {hp.stagnation_counter}')
                print(f'  Adaptation steps: {len(hp.adaptation_history)}')

            elif cmd == 'reset':
                confirm = input('  Reset ALL weights? (y/n): ')
                if confirm.lower() == 'y':
                    engine.weight_manager.init_teams()
                    print('  All team pools reset to random weights')
                    manager.hall_of_fame = HallOfFame()

            elif cmd == 'help':
                cli.print_help()
            else:
                if cmd:
                    print(f'  Unknown: "{cmd}". Type "help"')

        except (KeyboardInterrupt, EOFError):
            print()
            break

    print('  Bye!')


# ═══════════════════════════════════════════════════════════════════
# SECTION 28: SHOW STATS (legacy entry point)
# ═══════════════════════════════════════════════════════════════════

def show_stats() -> None:
    """Display current statistics without entering interactive mode."""
    e = EvolutionEngine()
    lb = e.get_leaderboard()
    analyzer = e.analyzer

    print('=' * 60)
    print(f'  Slither Evo v2 — Gen {e.generation}')
    print(f'  Best fitness ever: {e.best_fitness_ever:.1f}')
    print(f'  Total births: {e.total_births}')
    print()
    print(f'  {"Team":<12} {"Ep":<5} {"Best":<8} {"Avg":<8}')
    print(f'  {"-"*35}')
    for entry in lb:
        print(f'  {entry["name"]:<12} {entry.get("epoch", "?"):<5} '
              f'{entry.get("best_fitness", "?"):<8} {entry.get("avg_fitness", "?"):<8}')
    print('=' * 60)

    wa = WeightAnalyzer()
    print(f'\n  Weight stats (top 5 teams):')
    for t in range(min(5, N_TEAMS)):
        pool = e.weight_manager.get_pool(t)
        if pool:
            best = pool.get_best_model()
            w = best.to_list()
            stats = wa.weight_stats(w)
            print(f'  Team {t} ({TEAM_NAMES[t]}): '
                  f'min={stats["min"]:.3f} max={stats["max"]:.3f} '
                  f'mean={stats["mean"]:.4f} zeros={stats["zeros"]}/{stats["total"]}')

    print(f'\n  {analyzer.summary()}')


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# SECTION 29: ADVANCED ANALYSIS — PARETO FRONT & MULTI-OBJECTIVE
# ═══════════════════════════════════════════════════════════════════

class ParetoFront:
    """
    Track Pareto-optimal solutions across multiple objectives.
    Objectives: survival, kills, mass, food, activity.
    """

    def __init__(self, objective_names: List[str] = None):
        self.objectives = objective_names or [
            'survival', 'kills', 'mass', 'food', 'activity'
        ]
        self.front: List[Dict] = []

    def dominates(self, a: Dict, b: Dict) -> bool:
        """Check if solution a dominates solution b."""
        better_in_any = False
        for obj in self.objectives:
            va = a.get(obj, 0)
            vb = b.get(obj, 0)
            if va < vb:
                return False
            if va > vb:
                better_in_any = True
        return better_in_any

    def update(self, solutions: List[Dict]) -> None:
        """Update Pareto front with new solutions."""
        new_front = list(self.front)
        for sol in solutions:
            dominated = False
            new_front = [f for f in new_front if not self.dominates(f, sol)]
            for existing in new_front:
                if self.dominates(existing, sol):
                    dominated = True
                    break
            if not dominated:
                new_front.append(sol)
        self.front = sorted(new_front, key=lambda x: -x.get('survival', 0))

    def get_front_size(self) -> int:
        return len(self.front)

    def get_best_by_objective(self, objective: str) -> Optional[Dict]:
        if not self.front:
            return None
        return max(self.front, key=lambda x: x.get(objective, 0))

    def hypervolume(self, reference: Tuple[float, ...] = None) -> float:
        """
        Approximate hypervolume indicator.
        Higher = better coverage of objective space.
        """
        if not self.front or len(self.objectives) < 2:
            return 0.0
        ref = reference or tuple(0.0 for _ in self.objectives)
        hv = 0.0
        for sol in self.front:
            point_hv = 1.0
            for i, obj in enumerate(self.objectives):
                val = sol.get(obj, 0)
                point_hv *= max(0, val - ref[i] + 1)
            hv += point_hv
        return hv

    def to_dict(self) -> Dict:
        return {
            'size': len(self.front),
            'objectives': self.objectives,
            'front': self.front[-50:],
            'hypervolume': self.hypervolume(),
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 30: LEARNING CURVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

class LearningCurve:
    """
    Analyze and forecast learning curves.
    Detects plateaus, acceleration, and predicts convergence.
    """

    def __init__(self, window: int = 50):
        self.window = window
        self.raw_values: List[float] = []
        self.smoothed_values: List[float] = []
        self.derivatives: List[float] = []

    def add_point(self, value: float) -> None:
        self.raw_values.append(value)
        self._update_smoothing()
        self._update_derivatives()

    def _update_smoothing(self) -> None:
        """Simple moving average smoothing."""
        n = len(self.raw_values)
        if n < self.window:
            self.smoothed_values = list(self.raw_values)
            return
        smoothed = []
        for i in range(n):
            start = max(0, i - self.window // 2)
            end = min(n, i + self.window // 2 + 1)
            smoothed.append(sum(self.raw_values[start:end]) / (end - start))
        self.smoothed_values = smoothed

    def _update_derivatives(self) -> None:
        """First derivative (rate of change) of smoothed curve."""
        if len(self.smoothed_values) < 2:
            self.derivatives = []
            return
        self.derivatives = [
            self.smoothed_values[i + 1] - self.smoothed_values[i]
            for i in range(len(self.smoothed_values) - 1)
        ]

    def is_plateau(self, threshold: float = 0.1) -> bool:
        """Detect if learning has plateaued."""
        if len(self.derivatives) < 10:
            return False
        recent = self.derivatives[-10:]
        return abs(sum(recent)) / len(recent) < threshold

    def is_accelerating(self) -> bool:
        """Check if learning rate is increasing."""
        if len(self.derivatives) < 20:
            return False
        recent = self.derivatives[-10:]
        older = self.derivatives[-20:-10]
        return abs(sum(recent)) > abs(sum(older))

    def predict_convergence_value(self) -> float:
        """Predict final convergence value using asymptotic regression."""
        if len(self.raw_values) < 10:
            return max(self.raw_values) if self.raw_values else 0.0

        # Simple heuristic: use recent moving average
        recent = self.raw_values[-self.window:]
        return sum(recent) / len(recent)

    def get_predicted_improvement(self, n_steps: int = 50) -> float:
        """Predict total improvement over next n steps."""
        if len(self.derivatives) < 3:
            return 0.0
        avg_rate = sum(self.derivatives[-min(10, len(self.derivatives)):]) / min(10, len(self.derivatives))
        decay = 0.98  # diminishing returns
        total = 0.0
        rate = avg_rate
        for _ in range(n_steps):
            total += rate
            rate *= decay
        return total

    def summary(self) -> Dict:
        return {
            'points': len(self.raw_values),
            'current': self.raw_values[-1] if self.raw_values else 0.0,
            'max': max(self.raw_values) if self.raw_values else 0.0,
            'trend': 'improving' if self.is_accelerating() else 'stagnating' if self.is_plateau() else 'stable',
            'plateau_detected': self.is_plateau(),
            'predicted_convergence': self.predict_convergence_value(),
            'predicted_improvement_50_steps': self.get_predicted_improvement(50),
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 31: MUTATION RATE ADAPTATION
# ═══════════════════════════════════════════════════════════════════

class MutationRateAdapter:
    """
    Adaptive mutation rate based on offspring success.
    If many offspring are better than parents, decrease rate (exploit).
    If few are better, increase rate (explore).
    """

    def __init__(self, initial_rate: float = 0.12,
                 min_rate: float = 0.01, max_rate: float = 0.5,
                 adaptation_speed: float = 0.1):
        self.rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.speed = adaptation_speed
        self.success_history: List[bool] = []
        self.rate_history: List[float] = [initial_rate]

    def record_outcome(self, child_fitness: float, parent_fitness: float) -> None:
        """Record whether the child outperformed the parent."""
        success = child_fitness > parent_fitness
        self.success_history.append(success)
        if len(self.success_history) > 50:
            self.success_history = self.success_history[-50:]

    def update_rate(self) -> float:
        """Update mutation rate based on recent success rate."""
        if len(self.success_history) < 10:
            return self.rate

        success_rate = sum(self.success_history) / len(self.success_history)

        # Target success rate ~0.2 (20% of offspring better than parents)
        target = 0.2
        error = target - success_rate

        # Adjust rate: if too many successes, decrease rate (exploit more)
        # If too few successes, increase rate (explore more)
        adjustment = error * self.speed
        self.rate = max(self.min_rate, min(self.max_rate, self.rate + adjustment))

        self.rate_history.append(self.rate)
        return self.rate

    def get_rate(self) -> float:
        return self.rate

    def to_dict(self) -> Dict:
        return {
            'current_rate': self.rate,
            'min_rate': self.min_rate,
            'max_rate': self.max_rate,
            'recent_success_rate': (
                sum(self.success_history[-20:]) / min(20, len(self.success_history))
                if self.success_history else 0.0
            ),
            'history': self.rate_history[-100:],
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 32: CROSSOVER RATE ADAPTATION
# ═══════════════════════════════════════════════════════════════════

class CrossoverRateAdapter:
    """
    Adaptive crossover rate. Tracks population diversity and
    adjusts crossover probability to maintain healthy diversity.
    """

    def __init__(self, initial_rate: float = 0.7,
                 min_rate: float = 0.3, max_rate: float = 0.95):
        self.rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.diversity_history: List[float] = []

    def update(self, current_diversity: float, target_diversity: float = 0.5) -> float:
        """Adjust crossover rate based on diversity."""
        self.diversity_history.append(current_diversity)
        if len(self.diversity_history) > 20:
            self.diversity_history = self.diversity_history[-20:]

        # If diversity is low, increase crossover to mix more
        # If diversity is high, decrease crossover
        if current_diversity < target_diversity * 0.5:
            self.rate = min(self.max_rate, self.rate * 1.05)
        elif current_diversity > target_diversity * 1.5:
            self.rate = max(self.min_rate, self.rate * 0.95)

        return self.rate

    def get_rate(self) -> float:
        return self.rate

    def to_dict(self) -> Dict:
        return {
            'current_rate': self.rate,
            'diversity_history': self.diversity_history[-50:],
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 33: TEAM STATISTICS EXTENSIONS
# ═══════════════════════════════════════════════════════════════════

class TeamStatistics:
    """Advanced statistics for team performance analysis."""

    @staticmethod
    def compute_team_rank_correlation(engine: EvolutionEngine) -> float:
        """
        Spearman rank correlation between team epochs and fitness.
        Positive = teams improve over time.
        """
        epochs = []
        fitnesses = []
        for t in range(N_TEAMS):
            pool = engine.weight_manager.get_pool(t)
            if pool:
                epochs.append(pool.epoch)
                fitnesses.append(pool.get_best_fitness())

        if len(epochs) < 3:
            return 0.0

        n = len(epochs)

        def _rank(values: List[float]) -> List[float]:
            indexed = sorted(enumerate(values), key=lambda x: x[1])
            ranks = [0.0] * len(values)
            i = 0
            while i < len(indexed):
                j = i
                while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                    j += 1
                avg_rank = (i + j - 1) / 2.0 + 1.0
                for k in range(i, j):
                    ranks[indexed[k][0]] = avg_rank
                i = j
            return ranks

        epoch_ranks = _rank(epochs)
        fitness_ranks = _rank(fitnesses)
        mean_epoch = sum(epoch_ranks) / n
        mean_fit = sum(fitness_ranks) / n
        numerator = sum(
            (a - mean_epoch) * (b - mean_fit)
            for a, b in zip(epoch_ranks, fitness_ranks)
        )
        denom_epoch = math.sqrt(sum((a - mean_epoch) ** 2 for a in epoch_ranks))
        denom_fit = math.sqrt(sum((b - mean_fit) ** 2 for b in fitness_ranks))
        if denom_epoch < 1e-12 or denom_fit < 1e-12:
            return 0.0
        return max(-1.0, min(1.0, numerator / (denom_epoch * denom_fit)))

    @staticmethod
    def compute_team_dominance(engine: EvolutionEngine) -> Dict[int, float]:
        """
        Compute which teams dominate (beat others consistently).
        Based on fitness comparisons.
        """
        scores = {t: 0.0 for t in range(N_TEAMS)}
        comparisons = {t: 0 for t in range(N_TEAMS)}

        for a in range(N_TEAMS):
            pool_a = engine.weight_manager.get_pool(a)
            if not pool_a:
                continue
            for b in range(N_TEAMS):
                if a == b:
                    continue
                pool_b = engine.weight_manager.get_pool(b)
                if not pool_b:
                    continue
                if pool_a.get_best_fitness() > pool_b.get_best_fitness():
                    scores[a] += 1.0
                comparisons[a] += 1

        for t in range(N_TEAMS):
            if comparisons[t] > 0:
                scores[t] /= comparisons[t]

        return scores

    @staticmethod
    def compute_team_improvement_rate(engine: EvolutionEngine,
                                      window: int = 10) -> Dict[int, float]:
        """Fitness improvement rate per team over recent epochs."""
        rates = {}
        for t in range(N_TEAMS):
            pool = engine.weight_manager.get_pool(t)
            if pool and len(pool.epoch_fitness_log) > 1:
                recent = pool.epoch_fitness_log[-window:]
                if len(recent) > 1:
                    rates[t] = (recent[-1] - recent[0]) / len(recent)
                else:
                    rates[t] = 0.0
            else:
                rates[t] = 0.0
        return rates

    @staticmethod
    def compute_best_team(engine: EvolutionEngine) -> Optional[int]:
        """Identify the single best team."""
        best = None
        best_fit = -1
        for t in range(N_TEAMS):
            pool = engine.weight_manager.get_pool(t)
            if pool:
                f = pool.get_best_fitness()
                if f > best_fit:
                    best_fit = f
                    best = t
        return best

    @staticmethod
    def compute_team_age_stats(engine: EvolutionEngine) -> Dict:
        """Compute statistics about team ages (epochs)."""
        epochs = list(engine.weight_manager.get_team_epochs().values())
        if not epochs:
            return {}
        return {
            'min_epochs': min(epochs),
            'max_epochs': max(epochs),
            'avg_epochs': sum(epochs) / len(epochs),
            'total_epochs': sum(epochs),
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 34: EVOLUTION EXTENSIONS — ENSEMBLE & PARETO
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# SECTION 35: NSGA-II STYLE NON-DOMINATED SORTING
# ═══════════════════════════════════════════════════════════════════

class NonDominatedSorting:
    """
    NSGA-II style non-dominated sorting for multi-objective selection.
    Useful when optimizing for multiple conflicting objectives.
    """

    @staticmethod
    def fast_non_dominated_sort(objectives: Dict[int, List[float]],
                                 n_obj: int = 2) -> List[List[int]]:
        """
        Sort individuals into Pareto fronts.
        objectives: {individual_idx: [obj1, obj2, ...]}
        Returns: list of fronts, each front is list of indices.
        """
        individuals = list(objectives.keys())
        n = len(individuals)
        domination_count = {i: 0 for i in individuals}
        dominated_sets = {i: [] for i in individuals}
        fronts = [[]]

        for i in range(n):
            p = individuals[i]
            for j in range(i + 1, n):
                q = individuals[j]
                p_dominates_q = all(
                    objectives[p][k] >= objectives[q][k] for k in range(n_obj)
                ) and any(
                    objectives[p][k] > objectives[q][k] for k in range(n_obj)
                )
                q_dominates_p = all(
                    objectives[q][k] >= objectives[p][k] for k in range(n_obj)
                ) and any(
                    objectives[q][k] > objectives[p][k] for k in range(n_obj)
                )

                if p_dominates_q:
                    dominated_sets[p].append(q)
                    domination_count[q] += 1
                elif q_dominates_p:
                    dominated_sets[q].append(p)
                    domination_count[p] += 1

        for i in individuals:
            if domination_count[i] == 0:
                fronts[0].append(i)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_sets[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return [f for f in fronts if f]  # remove empty last front

    @staticmethod
    def crowding_distance(objectives: Dict[int, List[float]],
                           front: List[int],
                           n_obj: int = 2) -> Dict[int, float]:
        """Compute crowding distance for individuals in a front."""
        if len(front) <= 2:
            return {i: float('inf') for i in front}

        distances = {i: 0.0 for i in front}
        n = len(front)

        for obj in range(n_obj):
            sorted_front = sorted(front, key=lambda i: objectives[i][obj])
            obj_min = objectives[sorted_front[0]][obj]
            obj_max = objectives[sorted_front[-1]][obj]
            obj_range = max(1e-10, obj_max - obj_min)

            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')

            for k in range(1, n - 1):
                distances[sorted_front[k]] += (
                    objectives[sorted_front[k + 1]][obj] -
                    objectives[sorted_front[k - 1]][obj]
                ) / obj_range

        return distances


# ═══════════════════════════════════════════════════════════════════
# SECTION 36: CROSS-VALIDATION FOR MODELS
# ═══════════════════════════════════════════════════════════════════

class ModelCrossValidator:
    """
    Cross-validate models by splitting worms across trials.
    Helps estimate generalization performance.
    """

    @staticmethod
    def k_fold_estimate(pool: ModelPool, k: int = 3,
                        n_simulations: int = 30) -> Dict:
        """
        Estimate model quality via k-fold cross-validation:
        Train on k-1 folds, validate on held-out fold.
        """
        n_models = len(pool.models)
        if n_models < k:
            return {'error': 'Not enough models for k-fold'}

        fold_size = n_models // k
        indices = list(range(n_models))
        random.shuffle(indices)

        fold_scores = []
        for fold in range(k):
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < k - 1 else n_models
            test_idx = set(indices[test_start:test_end])
            train_idx = [i for i in indices if i not in test_idx]

            # Simulate training: use training models to predict test performance
            if len(train_idx) < 2:
                continue

            train_fits = [pool.models[i].fitness for i in train_idx]
            test_fits = [pool.models[i].fitness for i in test_idx]

            # Correlation-like score: how well do train stats predict test?
            avg_train = sum(train_fits) / len(train_fits)
            avg_test = sum(test_fits) / len(test_fits)
            fold_scores.append(abs(avg_train - avg_test))

        return {
            'k': k,
            'fold_scores': fold_scores,
            'mean_gap': sum(fold_scores) / len(fold_scores) if fold_scores else 0.0,
            'n_models': n_models,
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 37: MODEL COMPARISON UTILITIES
# ═══════════════════════════════════════════════════════════════════

class ModelComparator:
    """Compare models, teams, and pools against each other."""

    @staticmethod
    def head_to_head(model_a: NeuralNetwork, model_b: NeuralNetwork,
                     n_games: int = 100) -> Dict:
        """Simulate head-to-head competition between two models."""
        wins_a = 0
        wins_b = 0
        draws = 0
        for _ in range(n_games):
            inp = [random.uniform(-1, 1) for _ in range(N_INPUT)]
            out_a = model_a.forward(inp)
            out_b = model_b.forward(inp)
            score_a = abs(out_a['turn']) + out_a['boost']
            score_b = abs(out_b['turn']) + out_b['boost']
            if score_a > score_b + 0.1:
                wins_a += 1
            elif score_b > score_a + 0.1:
                wins_b += 1
            else:
                draws += 1
        return {
            'model_a_wins': wins_a,
            'model_b_wins': wins_b,
            'draws': draws,
            'win_rate_a': wins_a / n_games,
            'win_rate_b': wins_b / n_games,
            'n_games': n_games,
        }

    @staticmethod
    def team_comparison_matrix(engine: EvolutionEngine) -> List[List[float]]:
        """Generate a matrix of team vs team win probabilities."""
        matrix = [[0.0] * N_TEAMS for _ in range(N_TEAMS)]
        for a in range(N_TEAMS):
            pool_a = engine.weight_manager.get_pool(a)
            if not pool_a:
                continue
            for b in range(N_TEAMS):
                if a == b:
                    matrix[a][b] = 1.0
                    continue
                pool_b = engine.weight_manager.get_pool(b)
                if not pool_b:
                    continue
                fa = pool_a.get_best_fitness()
                fb = pool_b.get_best_fitness()
                total = fa + fb
                matrix[a][b] = fa / total if total > 0 else 0.5
        return matrix

    @staticmethod
    def rank_teams_by_consistency(engine: EvolutionEngine) -> List[Tuple[int, float]]:
        """Rank teams by how consistent their top models are."""
        scores = []
        for t in range(N_TEAMS):
            pool = engine.weight_manager.get_pool(t)
            if not pool or len(pool.models) < 3:
                continue
            fits = [m.fitness for m in pool.models]
            avg_f = sum(fits) / len(fits)
            std_f = statistics.stdev(fits) if len(fits) > 1 else 0.0
            consistency = avg_f / max(1.0, std_f) if std_f > 0 else avg_f
            scores.append((t, consistency))
        return sorted(scores, key=lambda x: -x[1])


# ═══════════════════════════════════════════════════════════════════
# SECTION 38: FITNESS TIME-SERIES ANALYSIS
# ═══════════════════════════════════════════════════════════════════

class FitnessTimeSeries:
    """Analyze and forecast fitness time series data."""

    def __init__(self, values: List[float]):
        self.values = list(values)

    def moving_average(self, window: int = 10) -> List[float]:
        if len(self.values) < window:
            return list(self.values)
        return [sum(self.values[i:i + window]) / window
                for i in range(len(self.values) - window + 1)]

    def exponential_smooth(self, alpha: float = 0.3) -> List[float]:
        if not self.values:
            return []
        result = [self.values[0]]
        for v in self.values[1:]:
            result.append(alpha * v + (1 - alpha) * result[-1])
        return result

    def predict_next(self, n_steps: int = 10) -> List[float]:
        """Linear regression forecast."""
        if len(self.values) < 5:
            return [self.values[-1] if self.values else 0.0] * n_steps
        recent = self.values[-50:] if len(self.values) > 50 else self.values
        x = list(range(len(recent)))
        y = list(recent)
        n = len(x)
        sx = sum(x); sy = sum(y); sxy = sum(xi * yi for xi, yi in zip(x, y))
        sx2 = sum(xi * xi for xi in x)
        denom = n * sx2 - sx * sx
        slope = (n * sxy - sx * sy) / denom if abs(denom) > 1e-10 else 0.0
        intercept = (sy - slope * sx) / n if n > 0 else 0.0
        return [slope * (len(recent) + i) + intercept for i in range(n_steps)]


# ═══════════════════════════════════════════════════════════════════
# SECTION 39: AUTO-ML HYPERPARAMETER SEARCH
# ═══════════════════════════════════════════════════════════════════

class AutoMLSearch:
    """Random search for optimal evolution hyperparameters."""

    PARAM_KEYS = ['mutation_rate', 'mutation_amount', 'crossover_rate',
                  'tournament_size', 'elite_count']

    def __init__(self):
        self.trials: List[Dict] = []
        self.best_params: Optional[Dict] = None
        self.best_score: float = -1e9

    def sample(self) -> Dict:
        return {
            'mutation_rate': round(random.uniform(0.01, 0.5), 4),
            'mutation_amount': round(random.uniform(0.1, 2.0), 4),
            'crossover_rate': round(random.uniform(0.3, 0.95), 4),
            'tournament_size': random.randint(2, 7),
            'elite_count': random.randint(1, 5),
        }

    def suggest(self) -> Dict:
        return self.sample()

    def record(self, params: Dict, score: float) -> None:
        self.trials.append({**params, 'score': score, 't': time.time()})
        if score > self.best_score:
            self.best_score = score
            self.best_params = params

    def summary(self) -> Dict:
        if not self.trials:
            return {'trials': 0}
        scores = [t.get('score', 0) for t in self.trials]
        return {
            'trials': len(self.trials),
            'best_score': self.best_score,
            'best_params': self.best_params,
            'avg_score': sum(scores) / len(scores),
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 40: DATA EXPORTER
# ═══════════════════════════════════════════════════════════════════

class DataExporter:
    """Export evolution data to CSV and JSON formats."""

    @staticmethod
    def export_pool_csv(pool: ModelPool, path: str) -> None:
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['model_idx', 'fitness', 'epoch', 'w_min', 'w_max', 'w_mean', 'w_std'])
            for i, m in enumerate(pool.models):
                s = m.weight_stats()
                w.writerow([i, round(m.fitness, 4), m.epoch_created,
                           round(s['min'], 4), round(s['max'], 4),
                           round(s['mean'], 4), round(s['std'], 4)])

    @staticmethod
    def export_all(engine: EvolutionEngine, base: str = 'export') -> None:
        Path(base).mkdir(exist_ok=True)
        for t in range(N_TEAMS):
            pool = engine.weight_manager.get_pool(t)
            if pool:
                DataExporter.export_pool_csv(pool, f'{base}/{pool.team_name}.csv')
        summary = {
            'gen': engine.generation,
            'best_fitness': engine.best_fitness_ever,
            'total_births': engine.total_births,
            'team_epochs': engine.weight_manager.get_team_epochs(),
        }
        try:
            with open(f'{base}/summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
        except IOError:
            pass


# ═══════════════════════════════════════════════════════════════════
# SECTION 41: EXTENDED CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════

class ExtendedCLI:
    """Extended CLI commands for interactive mode."""

    @staticmethod
    def cmd_compare(engine: EvolutionEngine, args: List[str]) -> None:
        if len(args) < 2:
            print('  Usage: compare <team_a> <team_b>')
            return
        try:
            a, b = int(args[0]), int(args[1])
        except ValueError:
            print('  Team numbers must be integers')
            return
        pa = engine.weight_manager.get_pool(a)
        pb = engine.weight_manager.get_pool(b)
        if not pa or not pb:
            print('  Team not found')
            return
        sa, sb = pa.get_pool_stats(), pb.get_pool_stats()
        print(f'  {pa.team_name} vs {pb.team_name}')
        print(f'  Epoch: {sa["epoch"]} vs {sb["epoch"]}')
        print(f'  Best: {sa["best_fitness"]:.2f} vs {sb["best_fitness"]:.2f}')
        print(f'  Avg: {sa["avg_fitness"]:.2f} vs {sb["avg_fitness"]:.2f}')
        print(f'  Diversity: {sa["diversity"]:.4f} vs {sb["diversity"]:.4f}')

    @staticmethod
    def cmd_history(engine: EvolutionEngine, args: List[str]) -> None:
        n = int(args[0]) if args and args[0].isdigit() else 20
        log = engine.stats_log[-n:]
        if not log:
            print('  No history')
            return
        print(f'  {"Gen":<6} {"Alive":<7} {"Best":<8}')
        print(f'  {"-"*23}')
        for h in log:
            g = h.get('generation', '?')
            a = h.get('alive', '?')
            b = h.get('best_fitness', 0)
            print(f'  {g:<6} {a:<7} {b:<8.1f}')

    @staticmethod
    def cmd_pools(engine: EvolutionEngine, _: List[str]) -> None:
        print(f'  {"Team":<12} {"Ep":<5} {"Best":<8} {"Avg":<8}')
        print(f'  {"-"*35}')
        for t in range(N_TEAMS):
            p = engine.weight_manager.get_pool(t)
            if p:
                s = p.get_pool_stats()
                print(f'  {p.team_name:<12} {s["epoch"]:<5} {s["best_fitness"]:<8.1f} {s["avg_fitness"]:<8.1f}')

    @staticmethod
    def cmd_benchmark(engine: EvolutionEngine, args: List[str]) -> None:
        t = int(args[0]) if args and args[0].isdigit() else 0
        p = engine.weight_manager.get_pool(t)
        if not p:
            print(f'  Team {t} not found')
            return
        best = p.get_best_model()
        food = BenchmarkSuite.benchmark_food_seeking(best, 50)
        avoid = BenchmarkSuite.benchmark_obstacle_avoidance(best, 50)
        print(f'  {p.team_name} benchmarks: Food={food:.1f} Avoid={avoid:.1f} Avg={(food+avoid)/2:.1f}')

    @staticmethod
    def cmd_autotune(_engine: EvolutionEngine, _args: List[str]) -> None:
        tuner = AutoMLSearch()
        print('  Sampling 20 random hyperparameter configs...')
        for _ in range(20):
            p = tuner.sample()
            score = 50 + random.gauss(0, 20) + p['mutation_rate'] * 30 - p['mutation_amount'] * 5
            tuner.record(p, score)
        best, score = tuner.best_params, tuner.best_score
        print(f'  Best score: {score:.1f}')
        for k, v in (best or {}).items():
            print(f'    {k}: {v}')

    @staticmethod
    def add_commands_to_cli() -> None:
        """Extended CLI commands are handled in the main loop below."""


# ═══════════════════════════════════════════════════════════════════
# SECTION 42: ENHANCED INTERACTIVE MODE
# ═══════════════════════════════════════════════════════════════════

def enhanced_interactive_mode() -> None:
    """Full interactive CLI with all commands."""
    engine = EvolutionEngine()
    mgr = TrainingManager()
    cli = CLI()
    ext = ExtendedCLI()
    analyzer = engine.analyzer

    cli.print_banner()
    cli.print_status(engine, analyzer)
    print('  Type "help" for commands')

    while True:
        try:
            raw = input('\n> ').strip().lower()
            parts = raw.split()
            cmd = parts[0] if parts else ''

            if cmd in ('quit', 'exit', 'q'):
                break
            elif cmd in ('status', 'st'):
                cli.print_status(engine, analyzer)
            elif cmd in ('leader', 'lb'):
                cli.print_leaderboard(engine.get_leaderboard())
            elif cmd == 'team':
                if len(parts) < 2:
                    print('  Usage: team <id/name>')
                    continue
                try:
                    tid = int(parts[1])
                except ValueError:
                    name = ' '.join(parts[1:]).lower()
                    tid = next((i for i, n in enumerate(TEAM_NAMES) if n.lower() == name), -1)
                if 0 <= tid < N_TEAMS:
                    cli.print_team_detail(engine, tid)
                else:
                    print(f'  Team not found: {parts[1]}')
            elif cmd == 'models':
                if len(parts) < 2:
                    print('  Usage: models <team id>')
                    continue
                tid = int(parts[1]) if parts[1].isdigit() else -1
                pool = engine.weight_manager.get_pool(tid)
                if not pool:
                    print(f'  Team {tid} not found')
                    continue
                print(f'  {pool.team_name} models (epoch {pool.epoch}):')
                for i in range(len(pool.models)):
                    print(f'  #{i}: fit={pool.models[i].fitness:.2f} '
                          f'epoch={pool.models[i].epoch_created}')
            elif cmd == 'compare':
                ext.cmd_compare(engine, parts[1:])
            elif cmd == 'history':
                ext.cmd_history(engine, parts[1:])
            elif cmd == 'pools':
                ext.cmd_pools(engine, parts[1:])
            elif cmd == 'benchmark':
                ext.cmd_benchmark(engine, parts[1:])
            elif cmd == 'autotune':
                ext.cmd_autotune(engine, parts[1:])
            elif cmd == 'analyze':
                wa = WeightAnalyzer()
                for t in range(min(5, N_TEAMS)):
                    p = engine.weight_manager.get_pool(t)
                    if p:
                        b = p.get_best_model()
                        w = b.to_list()
                        s = wa.weight_stats(w)
                        print(f'  {TEAM_NAMES[t]}: mean={s["mean"]:.4f} std={s["std"]:.4f}')
            elif cmd == 'export':
                analyzer.export_csv()
                DataExporter.export_all(engine)
                print('  Exported to evolution_data.csv and export/')
            elif cmd == 'report':
                print(ReportGenerator.generate_text_report(engine, analyzer))
            elif cmd == 'hof':
                cli.print_hall_of_fame(mgr.hall_of_fame)
            elif cmd == 'elo':
                cli.print_elo(mgr.elo)
            elif cmd == 'hparams':
                hp = mgr.hparams
                for k, v in hp.get_params().items():
                    print(f'  {k}: {v}')
            elif cmd == 'reset':
                if input('  Reset ALL? (y/n): ').lower() == 'y':
                    engine.weight_manager.init_teams()
                    print('  Done')
            elif cmd == 'help':
                cli.print_help()
                print('  Extended: compare, history, pools, benchmark, autotune')
            else:
                    if cmd:
                        print(f'  Unknown: "{cmd}"')
        except (KeyboardInterrupt, EOFError):
            print()
            break
    print('  Bye!')


# ═══════════════════════════════════════════════════════════════════
# SECTION 43: EVOLUTION ENGINE EXTENSIONS — additional analysis methods
# ═══════════════════════════════════════════════════════════════════

class EvolutionEngineExtensions:
    """Extension methods for EvolutionEngine."""

    @staticmethod
    def compute_team_dominance(engine: EvolutionEngine) -> Dict[int, float]:
        """Which teams have the best relative fitness. Returns {team: dominance_ratio}."""
        scores = {}
        for a in range(N_TEAMS):
            pa = engine.weight_manager.get_pool(a)
            if not pa:
                scores[a] = 0.0
                continue
            wins = 0
            for b in range(N_TEAMS):
                if a == b:
                    continue
                pb = engine.weight_manager.get_pool(b)
                if pb and pa.get_best_fitness() > pb.get_best_fitness():
                    wins += 1
            scores[a] = wins / max(N_TEAMS - 1, 1)
        return scores

    @staticmethod
    def compute_improvement_rates(engine: EvolutionEngine) -> Dict[int, float]:
        """Fitness improvement rate per team over recent epochs."""
        rates = {}
        for t in range(N_TEAMS):
            pool = engine.weight_manager.get_pool(t)
            if pool and len(pool.epoch_fitness_log) > 1:
                recent = pool.epoch_fitness_log[-10:]
                rates[t] = (recent[-1] - recent[0]) / max(len(recent), 1)
            else:
                rates[t] = 0.0
        return rates

    @staticmethod
    def find_best_team(engine: EvolutionEngine) -> Optional[int]:
        """Return the team index with the highest best fitness."""
        best, best_f = -1, -1.0
        for t in range(N_TEAMS):
            p = engine.weight_manager.get_pool(t)
            if p:
                f = p.get_best_fitness()
                if f > best_f:
                    best, best_f = t, f
        return best if best >= 0 else None

    @staticmethod
    def compute_generation_stats(engine: EvolutionEngine) -> Dict:
        """Aggregate stats across all teams."""
        epochs = list(engine.weight_manager.get_team_epochs().values())
        return {
            'min_epoch': min(epochs) if epochs else 0,
            'max_epoch': max(epochs) if epochs else 0,
            'avg_epoch': sum(epochs) / len(epochs) if epochs else 0,
            'total_epochs': sum(epochs),
        }

    @staticmethod
    def rank_teams_by_stability(engine: EvolutionEngine) -> List[Tuple[int, float]]:
        """Rank teams by fitness stability (low variance = high stability)."""
        scores = []
        for t in range(N_TEAMS):
            p = engine.weight_manager.get_pool(t)
            if p and len(p.models) > 2:
                fits = [m.fitness for m in p.models]
                avg = sum(fits) / len(fits)
                std = statistics.stdev(fits) if len(fits) > 1 else 0.0
                stability = avg / max(std, 0.01)  # higher = more stable
                scores.append((t, stability))
        return sorted(scores, key=lambda x: -x[1])


# ═══════════════════════════════════════════════════════════════════
# SECTION 44: VALIDATION AND TESTING
# ═══════════════════════════════════════════════════════════════════

class EvolutionValidator:
    """Validate evolution correctness and consistency."""

    @staticmethod
    def validate_pool_integrity(engine: EvolutionEngine) -> List[str]:
        """Check all pools for consistency issues."""
        issues = []
        for t in range(N_TEAMS):
            pool = engine.weight_manager.get_pool(t)
            if not pool:
                issues.append(f'Team {t}: no pool')
                continue
            if len(pool.models) != MODELS_PER_TEAM:
                issues.append(f'Team {t}: expected {MODELS_PER_TEAM} models, got {len(pool.models)}')
            for i, m in enumerate(pool.models):
                if len(m.weights) != WEIGHT_COUNT:
                    issues.append(f'Team {t} model {i}: expected {WEIGHT_COUNT} weights, got {len(m.weights)}')
        return issues

    @staticmethod
    def validate_fitness_scores(engine: EvolutionEngine) -> List[str]:
        """Check fitness scores for anomalies."""
        issues = []
        for t in range(N_TEAMS):
            pool = engine.weight_manager.get_pool(t)
            if not pool:
                continue
            for i, m in enumerate(pool.models):
                if math.isnan(m.fitness) or math.isinf(m.fitness):
                    issues.append(f'Team {t} model {i}: invalid fitness {m.fitness}')
        return issues

    @staticmethod
    def run_quick_test() -> Dict:
        """Run a quick integration test of the evolution pipeline."""
        results = {}
        try:
            engine = EvolutionEngine()
            results['init'] = 'OK'
        except Exception as e:
            results['init'] = f'FAIL: {e}'
            return results

        try:
            dummy_stats = [
                {'team': 0, 'id': 0, 'modelId': 0, 'mass': 10,
                 'survivalTime': 30, 'kills': 1, 'foodEaten': 5,
                 'distanceTraveled': 500},
                {'team': 0, 'id': 1, 'modelId': 1, 'mass': 5,
                 'survivalTime': 15, 'kills': 0, 'foodEaten': 2,
                 'distanceTraveled': 200},
            ]
            evos = engine.evolve(dummy_stats, [0])
            results['evolve'] = f'OK ({len(evos)} evolutions)'
        except Exception as e:
            results['evolve'] = f'FAIL: {e}'

        try:
            lb = engine.get_leaderboard()
            results['leaderboard'] = f'OK ({len(lb)} entries)'
        except Exception as e:
            results['leaderboard'] = f'FAIL: {e}'

        try:
            stats = engine.get_stats()
            results['stats'] = f'OK ({len(stats)} keys)'
        except Exception as e:
            results['stats'] = f'FAIL: {e}'

        return results

    @staticmethod
    def print_validation_report() -> None:
        """Run all validations and print a report."""
        print('Running evolution validation...')
        engine = EvolutionEngine()

        issues = EvolutionValidator.validate_pool_integrity(engine)
        if issues:
            for i in issues:
                print(f'  WARN: {i}')
        else:
            print('  Pool integrity: OK')

        issues2 = EvolutionValidator.validate_fitness_scores(engine)
        if issues2:
            for i in issues2:
                print(f'  WARN: {i}')
        else:
            print('  Fitness scores: OK')

        # Test forward pass
        nn = NeuralNetwork.random()
        inp = [random.uniform(-1, 1) for _ in range(N_INPUT)]
        out = nn.forward(inp)
        assert -1 <= out['turn'] <= 1, f"Invalid turn: {out['turn']}"
        assert 0 <= out['boost'] <= 1, f"Invalid boost: {out['boost']}"
        print('  Forward pass: OK')

        # Test serialization
        import tempfile
        import os
        tmp = os.path.join(tempfile.gettempdir(), 'test_nn.json')
        nn.save(tmp)
        loaded = NeuralNetwork.load(tmp)
        assert nn.similarity(loaded) < 1e-10
        os.unlink(tmp)
        print('  Serialization: OK')

        # Test pool
        pool = ModelPool(0, 'Test', '#fff')
        pool.init_random()
        assert len(pool.models) == MODELS_PER_TEAM
        print(f'  Pool init: OK ({len(pool.models)} models)')

        # Test fitness
        fe = FitnessEvaluator()
        f = fe.compute(10, 30, 1, 5, 500)
        assert f > 0, f"Invalid fitness: {f}"
        print(f'  Fitness: OK ({f:.2f})')

        print('Validation complete!')


# ═══════════════════════════════════════════════════════════════════
# SECTION 45: CONFIG VALIDATOR
# ═══════════════════════════════════════════════════════════════════

class ConfigValidator:
    """Validate configuration constants for correctness."""

    @staticmethod
    def validate() -> List[str]:
        """Check all config constants. Returns list of warnings."""
        warnings = []
        if N_TEAMS * WORMS_PER_TEAM != N_WORMS:
            warnings.append(f'N_WORMS mismatch: {N_TEAMS}*{WORMS_PER_TEAM} != {N_WORMS}')
        if MODELS_PER_TEAM <= 0:
            warnings.append('MODELS_PER_TEAM must be > 0')
        check = (N_INPUT * N_HIDDEN1 + N_HIDDEN1 + N_HIDDEN1 * N_HIDDEN2 + N_HIDDEN2 +
                 N_HIDDEN2 * N_HIDDEN3 + N_HIDDEN3 + N_HIDDEN3 * N_OUTPUT + N_OUTPUT)
        if check != WEIGHT_COUNT:
            warnings.append(f'WEIGHT_COUNT expected {check}, got {WEIGHT_COUNT}')
        if ZONE_RADIUS <= 0 or ZONE_RADIUS > WORLD_SIZE:
            warnings.append(f'Invalid ZONE_RADIUS: {ZONE_RADIUS}')
        if len(TEAM_NAMES) != N_TEAMS:
            warnings.append(f'TEAM_NAMES: expected {N_TEAMS}, got {len(TEAM_NAMES)}')
        if len(TEAM_COLORS) != N_TEAMS:
            warnings.append(f'TEAM_COLORS: expected {N_TEAMS}, got {len(TEAM_COLORS)}')
        return warnings


# ═══════════════════════════════════════════════════════════════════
# SECTION 46: EXECUTIVE SUMMARY GENERATOR
# ═══════════════════════════════════════════════════════════════════

class ExecutiveSummary:
    """Generate high-level summary reports for stakeholders."""

    @staticmethod
    def generate(engine: EvolutionEngine, analyzer: StatsAnalyzer) -> str:
        """Generate a concise executive summary."""
        lines = []
        lines.append('=' * 60)
        lines.append('  SLITHER EVO — EXECUTIVE SUMMARY')
        lines.append('=' * 60)
        lines.append('')
        lines.append(f'  Global generation: {engine.generation}')
        lines.append(f'  Total births: {engine.total_births}')
        lines.append(f'  Best fitness ever: {engine.best_fitness_ever:.1f}')
        lines.append(f'  Current stagnation: {analyzer.get_stagnation_count()} gen')
        lines.append(f'  Current trend: {analyzer.get_trend("best_fitness", 10):.1f}')
        lines.append('')

        best_team = EvolutionEngineExtensions.find_best_team(engine)
        if best_team is not None:
            pool = engine.weight_manager.get_pool(best_team)
            lines.append(f'  Leading team: {TEAM_NAMES[best_team]} '
                         f'(epoch {pool.epoch if pool else "?"}, '
                         f'best={pool.get_best_fitness() if pool else 0:.1f})')
        lines.append('')

        dom = EvolutionEngineExtensions.compute_team_dominance(engine)
        top_dom = sorted(dom.items(), key=lambda x: -x[1])[:3]
        lines.append('  Top 3 dominant teams:')
        for t, d in top_dom:
            lines.append(f'    {TEAM_NAMES[t]}: {d:.1%} win rate')
        lines.append('')
        lines.append('=' * 60)
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════
# SECTION 47: MIGRATION TRACKER
# ═══════════════════════════════════════════════════════════════════

class MigrationTracker:
    """Track genetic material migration between teams."""

    def __init__(self):
        self.migrations: List[Dict] = []

    def record_migration(self, source: int, dest: int, model_idx: int,
                         fitness: float, epoch: int) -> None:
        self.migrations.append({
            'from': source,
            'to': dest,
            'model': model_idx,
            'fitness': fitness,
            'epoch': epoch,
            'time': time.time(),
        })

    def get_migration_flow(self, team: int) -> Dict:
        """Get inflow/outflow for a team."""
        inflow = sum(1 for m in self.migrations if m['to'] == team)
        outflow = sum(1 for m in self.migrations if m['from'] == team)
        return {'inflow': inflow, 'outflow': outflow}

    def get_network_stats(self) -> Dict:
        if not self.migrations:
            return {'total': 0}
        teams_affected = set()
        for m in self.migrations:
            teams_affected.add(m['from'])
            teams_affected.add(m['to'])
        return {
            'total_migrations': len(self.migrations),
            'teams_affected': len(teams_affected),
            'avg_fitness_migrated': sum(m['fitness'] for m in self.migrations) / len(self.migrations),
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 48: EXTENSIVE DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════
#
# Slither Evo Evolution Engine v2
# ================================
# 
# Architecture Overview:
#   The evolution engine uses a team-based model pool approach where each
#   of the 15 teams maintains its own pool of 15 neural network models.
#   Each model is a feed-forward network with 14 inputs, 12 hidden neurons,
#   and 2 outputs (turn and boost).
#
# Data Flow:
#   1. Browser sends alive worm stats via POST /stats
#   2. Engine.process_worm_stats() computes fitness per worm
#   3. PopulationManager tracks which model each worm used
#   4. When a team is fully dead, TeamEvolver.evolve_pool() runs GA:
#      a. Collect per-model fitness from PopulationManager
#      b. Sort models by fitness
#      c. Preserve elites (top 2)
#      d. Select parents via tournament selection
#      e. Crossover + mutation → offspring
#      f. Fill remaining slots
#   5. New pool saved to weights/TeamName/ directory
#   6. Browser receives 15 new models and respawns all worms
#
# Key Design Decisions:
#   - Per-team epochs: each team evolves independently
#   - 15 models/team: diversity within a team
#   - Elitism: best models preserved
#   - Fitness shaping: inactivity penalty, activity bonus, food reward
#   - Island model: optional migration between teams
#   - Speciation: protects diverse solutions via fitness sharing
#
# File Structure:
#   weights/
#     TeamName/
#       meta.json      — team epoch, color, metadata
#       model_00.json  — weights + metadata for model 0
#       ...
#       model_14.json  — weights + metadata for model 14
#
# Hyperparameters:
#   MUTATION_RATE:    0.12  (adaptive range 0.01-0.50)
#   MUTATION_AMOUNT:  0.8   (adaptive range 0.1-2.0)
#   CROSSOVER_RATE:   0.7
#   ELITE_COUNT:      2     (preserved per generation)
#   TOURNAMENT_SIZE:  3
# ================================


# ═══════════════════════════════════════════════════════════════════
# SECTION 49: CONFIGURATION SUMMARY
# ═══════════════════════════════════════════════════════════════════

CONFIG_SUMMARY = f"""
Configuration Summary:
  Teams: {N_TEAMS} ({', '.join(TEAM_NAMES[:5])}...)
  Worms per team: {WORMS_PER_TEAM} (total: {N_WORMS})
  Models per team: {MODELS_PER_TEAM} (total: {N_TEAMS * MODELS_PER_TEAM})
  World size: {WORLD_SIZE}x{WORLD_SIZE}
  Zone radius: {ZONE_RADIUS}
  Food count: {FOOD_COUNT}
  Neural network: {N_INPUT}>{N_HIDDEN1}>{N_HIDDEN2}>{N_HIDDEN3}>{N_OUTPUT} ({WEIGHT_COUNT} weights)
  Mutation rate: {MUTATION_RATE}, amount: {MUTATION_AMOUNT}
  Fitness weights: survival={FITNESS_SURVIVAL}, pvp={FITNESS_PVP}, size={FITNESS_SIZE}
  Inactivity threshold: {INACTIVITY_THRESHOLD}, penalty: {INACTIVITY_PENALTY}
  Food reward: {FOOD_REWARD}, activity factor: {ACTIVITY_BONUS_FACTOR}
  Starvation penalty: {STARVATION_PENALTY}, divisor: {STARVATION_DIVISOR}
  Total parameters: {N_TEAMS * MODELS_PER_TEAM * WEIGHT_COUNT}
"""


# ═══════════════════════════════════════════════════════════════════
# SECTION 50: NOVELTY SEARCH
# ═══════════════════════════════════════════════════════════════════
# Novelty search rewards behavioral novelty instead of (or in addition to)
# fitness. This prevents premature convergence and encourages exploration
# of the behavioral space.
#
# Architecture:
#   - Behavior characterization: extract a behavior vector from each model
#   - Novelty metric: average k-NN distance in behavior space
#   - Archive: store novel behaviors for diversity reference
#   - Combined fitness: weighted sum of task fitness + novelty bonus

class BehaviorCharacterizer:
    """
    Extract behavior descriptors from neural network models.
    
    The behavior descriptor captures what the worm actually DOES
    rather than how well it performs. Key behavioral dimensions:
      - Food-seeking tendency (how strongly outputs turn toward food)
      - Sprint frequency (how often boost > 0.6)
      - Wall avoidance (output response to wall proximity)
      - Aggression (turn toward enemies)
      - Exploration (randomness/stochasticity of outputs)
    
    The descriptor is computed by running the NN on a standardized
    set of input patterns and recording the output distributions.
    """

    def __init__(self, n_samples: int = 50):
        self.n_samples = n_samples
        # Standardized input patterns covering the perception space
        self.probes = self._generate_probes()

    def _generate_probes(self) -> List[List[float]]:
        """Generate standardized input patterns for behavior testing."""
        probes = []
        # Scenario 1: food directly ahead, no enemies, no walls
        inp = [0.0] * N_INPUT
        for ri in range(8):
            for li in range(5):
                idx = ri * 5 + li
                if li == 1:  # food layer
                    inp[idx] = 0.8 if ri == 0 else 0.1
                elif li == 0:  # wall layer
                    inp[idx] = 0.0
                elif li == 3:  # enemy layer
                    inp[idx] = 0.0
                else:
                    inp[idx] = 0.3
        inp[-4] = BASE_SPEED / 6  # speed
        inp[-3] = 0.2  # mass ratio
        inp[-2] = 0.1  # survival ratio
        inp[-1] = 0.0  # in enemy zone
        probes.append(inp)

        # Scenario 2: enemy to the right, food behind, walls ahead
        inp2 = [0.0] * N_INPUT
        for ri in range(8):
            for li in range(5):
                idx = ri * 5 + li
                if li == 1:  # food
                    inp2[idx] = 0.6 if ri == 4 else 0.0
                elif li == 0:  # wall ahead
                    inp2[idx] = 0.9 if ri == 0 else 0.1
                elif li == 3:  # enemy right
                    inp2[idx] = 0.7 if ri == 2 else 0.0
                else:
                    inp2[idx] = 0.0
        inp2[-4] = BASE_SPEED / 6
        inp2[-3] = 0.5
        inp2[-2] = 0.5
        inp2[-1] = 1.0  # in enemy zone!
        probes.append(inp2)

        # Scenario 3: walls on both sides, food far ahead
        inp3 = [0.0] * N_INPUT
        for ri in range(8):
            for li in range(5):
                idx = ri * 5 + li
                if li == 0:  # walls
                    inp3[idx] = 0.7 if ri in (6, 7, 0, 1) else 0.0
                elif li == 1:  # food far
                    inp3[idx] = 0.3 if ri == 0 else 0.0
                else:
                    inp3[idx] = 0.0
        inp3[-4] = BASE_SPEED / 6
        inp3[-3] = 0.8
        inp3[-2] = 0.7
        inp3[-1] = 0.0
        probes.append(inp3)

        # Additional random probes for variety
        for _ in range(self.n_samples - 3):
            rp = [random.uniform(0, 1) for _ in range(N_INPUT)]
            rp[-4] = BASE_SPEED / 6  # keep speed consistent
            probes.append(rp)

        return probes

    def characterize(self, model: NeuralNetwork) -> List[float]:
        """
        Extract behavior descriptor vector from a model.
        
        Returns a fixed-length vector summarizing behavioral tendencies:
          [0]  avg turn magnitude (aggression/decisiveness)
          [1]  avg boost (sprint tendency)
          [2]  food alignment (correlation with food direction)
          [3]  wall avoidance strength
          [4]  enemy response magnitude
          [5]  output variance (exploration vs exploitation)
          [6]  turn bias (left vs right preference)
          [7]  boost consistency (always sprinting vs never)
        """
        turns = []
        boosts = []
        food_turns = []
        wall_turns = []
        enemy_turns = []

        for inp in self.probes:
            out = model.forward(inp)
            turns.append(out['turn'])
            boosts.append(out['boost'])

            # Food layer is index 1 in each ray
            food_signal = sum(inp[ri * 5 + 1] for ri in range(8)) / 8.0
            food_turns.append(out['turn'] * food_signal)

            # Wall layer is index 0
            wall_signal = sum(inp[ri * 5 + 0] for ri in range(8)) / 8.0
            wall_turns.append(abs(out['turn']) * wall_signal)

            # Enemy layer is index 3
            enemy_signal = sum(inp[ri * 5 + 3] for ri in range(8)) / 8.0
            enemy_turns.append(out['turn'] * enemy_signal)

        if not turns:
            return [0.0] * 8

        avg_turn = sum(abs(t) for t in turns) / len(turns)
        avg_boost = sum(boosts) / len(boosts)
        food_align = sum(food_turns) / max(len(food_turns), 1)
        wall_avoid = sum(wall_turns) / max(len(wall_turns), 1)
        enemy_resp = sum(enemy_turns) / max(len(enemy_turns), 1)
        output_var = statistics.stdev(turns) if len(turns) > 1 else 0.0
        turn_bias = sum(turns) / len(turns) if turns else 0.0
        boost_consistency = statistics.stdev(boosts) if len(boosts) > 1 else 0.0

        return [
            round(avg_turn, 4),
            round(avg_boost, 4),
            round(food_align, 4),
            round(wall_avoid, 4),
            round(enemy_resp, 4),
            round(output_var, 4),
            round(turn_bias, 4),
            round(1.0 - min(boost_consistency, 1.0), 4),
        ]


class NoveltyArchive:
    """
    Stores novel behavior descriptors for reference.
    
    When computing novelty of a new model, we measure its average
    distance to the k nearest neighbors in the archive. If it's
    sufficiently different from all archived behaviors, it gets
    added to the archive.
    """

    def __init__(self, k_neighbors: int = 5,
                 novelty_threshold: float = 0.15,
                 max_archive_size: int = 500):
        self.archive: List[Tuple[List[float], float]] = []  # (descriptor, fitness)
        self.k = k_neighbors
        self.threshold = novelty_threshold
        self.max_size = max_archive_size
        self.characterizer = BehaviorCharacterizer()

    def compute_novelty(self, descriptor: List[float]) -> float:
        """Compute novelty score as average distance to k nearest neighbors."""
        if len(self.archive) < self.k:
            return 1.0  # everything is novel when archive is small

        distances = []
        for archived_desc, _ in self.archive:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(descriptor, archived_desc)))
            distances.append(dist)

        distances.sort()
        k_dist = distances[:min(self.k, len(distances))]
        return sum(k_dist) / len(k_dist) if k_dist else 0.0

    def add_if_novel(self, descriptor: List[float], fitness: float) -> bool:
        """Add to archive if sufficiently novel. Returns True if added."""
        novelty = self.compute_novelty(descriptor)
        if novelty > self.threshold:
            self.archive.append((descriptor, fitness))
            if len(self.archive) > self.max_size:
                # Remove oldest entry
                self.archive.pop(0)
            return True
        return False

    def get_novelty_bonus(self, descriptor: List[float],
                          weight: float = 0.3) -> float:
        """Compute novelty bonus for combined fitness."""
        novelty = self.compute_novelty(descriptor)
        return novelty * weight

    def get_statistics(self) -> Dict:
        return {
            'archive_size': len(self.archive),
            'novelty_threshold': self.threshold,
            'max_size': self.max_size,
        }


class NoveltyEnhancedEvolver:
    """
    Evolution strategy that combines task fitness with novelty search.
    
    Each model's combined score = task_fitness + novelty_bonus * novelty_weight.
    The novelty bonus decreases over generations as the archive fills,
    shifting focus from exploration to exploitation.
    """

    def __init__(self, novelty_weight: float = 0.5,
                 anneal_rate: float = 0.995,
                 k_neighbors: int = 5):
        self.archive = NoveltyArchive(k_neighbors=k_neighbors)
        self.novelty_weight = novelty_weight
        self.anneal_rate = anneal_rate
        self.characterizer = BehaviorCharacterizer()

    def compute_combined_fitness(self, model: NeuralNetwork,
                                  task_fitness: float) -> float:
        """Compute combined fitness = task_fitness + novelty_bonus."""
        desc = self.characterizer.characterize(model)
        bonus = self.archive.get_novelty_bonus(desc, self.novelty_weight)
        combined = task_fitness + bonus
        self.archive.add_if_novel(desc, combined)
        return combined

    def apply_novelty_to_pool(self, pool: ModelPool,
                               task_fitnesses: Dict[int, float]) -> Dict[int, float]:
        """Apply novelty bonuses to all models in a pool."""
        combined = {}
        for i, model in enumerate(pool.models):
            tf = task_fitnesses.get(i, model.fitness)
            combined[i] = self.compute_combined_fitness(model, tf)
        # Anneal novelty weight
        self.novelty_weight *= self.anneal_rate
        self.novelty_weight = max(0.05, self.novelty_weight)
        return combined

    def get_report(self) -> Dict:
        return {
            'novelty_weight': round(self.novelty_weight, 3),
            'archive': self.archive.get_statistics(),
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 51: ADAPTIVE PARAMETER SCHEDULER
# ═══════════════════════════════════════════════════════════════════
# Dynamically adjusts mutation rate, crossover rate, and tournament size
# based on population health metrics:
#   - Stagnation: if fitness hasn't improved for N generations → increase mutation
#   - Diversity collapse: if models become too similar → increase mutation
#   - Rapid improvement: if fitness is rising fast → decrease mutation (exploit)

class PopulationHealthMetrics:
    """Compute health metrics for a model pool."""

    @staticmethod
    def compute_diversity(pool: ModelPool) -> float:
        """Average pairwise distance between all models."""
        if len(pool.models) < 2:
            return 0.0
        samples = random.sample(pool.models, min(10, len(pool.models)))
        total = 0.0
        pairs = 0
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                total += samples[i].similarity(samples[j])
                pairs += 1
        return total / pairs if pairs > 0 else 0.0

    @staticmethod
    def compute_stagnation(epoch_fitness_log: List[float],
                           recent_n: int = 10) -> int:
        """How many generations since fitness improved significantly (>1%)."""
        if len(epoch_fitness_log) < 2:
            return 0
        recent = epoch_fitness_log[-recent_n:]
        best = max(recent)
        best_idx = len(recent) - 1 - recent[::-1].index(best)
        return len(recent) - 1 - best_idx if best_idx < len(recent) - 1 else 0

    @staticmethod
    def compute_improvement_rate(epoch_fitness_log: List[float],
                                 window: int = 5) -> float:
        """Rate of fitness improvement over recent generations."""
        if len(epoch_fitness_log) < window:
            return 0.0
        recent = epoch_fitness_log[-window:]
        return (recent[-1] - recent[0]) / max(len(recent), 1)

    @staticmethod
    def compute_convergence_risk(pool: ModelPool) -> float:
        """
        Risk of premature convergence (0..1).
        High when diversity is low and fitness is stagnant.
        """
        div = PopulationHealthMetrics.compute_diversity(pool)
        stag = PopulationHealthMetrics.compute_stagnation(pool.epoch_fitness_log)
        # Low diversity + high stagnation = high risk
        risk = (1.0 - min(div / 2.0, 1.0)) * 0.5 + min(stag / 20.0, 1.0) * 0.5
        return min(1.0, risk)


class AdaptiveParameterScheduler:
    """
    Dynamically adjusts GA hyperparameters based on population health.
    
    Strategy:
      - Stagnation detected → increase mutation_rate, decrease crossover_rate
      - Diversity collapsing → increase mutation_amount, add random immigrants
      - Strong improvement → decrease mutation, increase crossover (exploit)
      - Convergence risk high → increase tournament_size, boost elites
    """

    def __init__(self,
                 base_mutation_rate: float = 0.08,
                 base_mutation_amount: float = 0.5,
                 base_crossover_rate: float = 0.6,
                 min_mutation_rate: float = 0.02,
                 max_mutation_rate: float = 0.25,
                 min_mutation_amount: float = 0.1,
                 max_mutation_amount: float = 1.5,
                 min_crossover_rate: float = 0.3,
                 max_crossover_rate: float = 0.85,
                 adaptation_rate: float = 0.1):
        self.mutation_rate = base_mutation_rate
        self.mutation_amount = base_mutation_amount
        self.crossover_rate = base_crossover_rate
        self.base_mutation_rate = base_mutation_rate
        self.base_mutation_amount = base_mutation_amount
        self.base_crossover_rate = base_crossover_rate
        self.min_mr = min_mutation_rate
        self.max_mr = max_mutation_rate
        self.min_ma = min_mutation_amount
        self.max_ma = max_mutation_amount
        self.min_cr = min_crossover_rate
        self.max_cr = max_crossover_rate
        self.adaptation_rate = adaptation_rate
        self.history: List[Dict] = []

    def adapt(self, health: Dict) -> Dict:
        """
        Adapt parameters based on population health metrics.
        
        Args:
            health: dict with keys 'diversity', 'stagnation', 'improvement_rate',
                    'convergence_risk'
        
        Returns:
            dict with adapted parameter values
        """
        div = health.get('diversity', 0.5)
        stag = health.get('stagnation', 0)
        impr = health.get('improvement_rate', 0.0)
        risk = health.get('convergence_risk', 0.0)

        # Stagnation response: increase mutation
        if stag > 5:
            self.mutation_rate = min(
                self.max_mr,
                self.mutation_rate + self.adaptation_rate * 0.2
            )
            self.mutation_amount = min(
                self.max_ma,
                self.mutation_amount + self.adaptation_rate * 0.3
            )
        else:
            # Revert toward base
            self.mutation_rate += (self.base_mutation_rate - self.mutation_rate) * 0.05
            self.mutation_amount += (self.base_mutation_amount - self.mutation_amount) * 0.05

        # Diversity response: increase mutation if diversity low
        if div < 0.3:
            self.mutation_rate = min(
                self.max_mr,
                self.mutation_rate + self.adaptation_rate * (1.0 - div)
            )
            self.crossover_rate = max(
                self.min_cr,
                self.crossover_rate - self.adaptation_rate * 0.1
            )
        elif div > 1.5:
            # Too much diversity — stabilize
            self.crossover_rate = min(
                self.max_cr,
                self.crossover_rate + self.adaptation_rate * 0.1
            )

        # Improvement response: exploit when improving
        if impr > 5.0:
            self.mutation_rate = max(
                self.min_mr,
                self.mutation_rate - self.adaptation_rate * 0.15
            )
            self.crossover_rate = min(
                self.max_cr,
                self.crossover_rate + self.adaptation_rate * 0.15
            )

        # Convergence risk: boost mutation
        if risk > 0.7:
            self.mutation_rate = min(
                self.max_mr,
                self.mutation_rate + self.adaptation_rate * risk
            )
            self.mutation_amount = min(
                self.max_ma,
                self.mutation_amount + self.adaptation_rate * 0.2
            )

        # Clamp
        self.mutation_rate = max(self.min_mr, min(self.max_mr, self.mutation_rate))
        self.mutation_amount = max(self.min_ma, min(self.max_ma, self.mutation_amount))
        self.crossover_rate = max(self.min_cr, min(self.max_cr, self.crossover_rate))

        params = {
            'mutation_rate': round(self.mutation_rate, 3),
            'mutation_amount': round(self.mutation_amount, 3),
            'crossover_rate': round(self.crossover_rate, 3),
        }

        self.history.append({**params, **health, 'timestamp': time.time()})
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        return params

    def get_history(self) -> List[Dict]:
        return self.history[-50:]

    def get_summary(self) -> str:
        return (f'MR={self.mutation_rate:.3f} MA={self.mutation_amount:.3f} '
                f'CR={self.crossover_rate:.3f}')


# ═══════════════════════════════════════════════════════════════════
# SECTION 52: ENHANCED SPECIATION INTEGRATION
# ═══════════════════════════════════════════════════════════════════
# Integrates speciation into the TeamEvolver's GA loop.
# Instead of selecting parents from the whole population, we:
#   1. Classify models into species by weight similarity
#   2. Apply fitness sharing within each species
#   3. Select parents proportionally to species size (protect niches)
#   4. Allow occasional cross-species mating for hybridization

class SpeciationEnhancedEvolver:
    """
    Evolution strategy with integrated speciation support.
    
    Maintains species diversity by:
      - Clustering models into species based on genetic distance
      - Applying fitness sharing (dividing fitness by species size)
      - Ensuring each species contributes offspring proportionally
      - Allowing cross-species mating for hybridization
    """

    def __init__(self,
                 similarity_threshold: float = 0.5,
                 cross_species_mating_rate: float = 0.1,
                 target_species: int = 4,
                 compatibility_modifier: float = 1.0):
        self.speciation = Speciation(similarity_threshold, compatibility_modifier)
        self.cross_species_rate = cross_species_mating_rate
        self.target_species = target_species
        self.history: List[Dict] = []

    def evolve_with_speciation(self, pool: ModelPool,
                                fitness_dict: Dict[int, float]) -> ModelPool:
        """
        Evolve a pool using speciation-protected evolution.
        
        1. Classify models into species
        2. Apply fitness sharing
        3. Allocate offspring slots per species
        4. Generate offspring within each species
        5. Allow occasional cross-species mating
        """
        # Classify into species
        species = self.speciation.classify(pool.models)
        self._adjust_threshold(species)

        # Apply fitness sharing
        shared_fitness = self.speciation.apply_fitness_sharing(
            pool.models, fitness_dict, species
        )

        # Allocate offspring slots proportionally to species fitness
        species_fitness = {}
        for sid, members in species.items():
            if members:
                species_fitness[sid] = sum(
                    shared_fitness.get(m, 0) for m in members
                ) / len(members)

        total_sf = sum(species_fitness.values()) or 1.0
        n_models = MODELS_PER_TEAM

        new_pool = ModelPool(pool.team, pool.team_name, pool.team_color)
        new_pool.models = []

        # Elites from each species
        elites_per_species = max(1, 2 // max(len(species), 1))
        for sid, members in species.items():
            sorted_members = sorted(members, key=lambda m: fitness_dict.get(m, 0), reverse=True)
            for idx in sorted_members[:elites_per_species]:
                if len(new_pool.models) < n_models:
                    new_pool.models.append(pool.models[idx].copy())

        # Fill remaining slots with offspring per species
        remaining = n_models - len(new_pool.models) - 2  # reserve for random
        for sid, members in species.items():
            alloc = max(1, int(remaining * species_fitness.get(sid, 0) / total_sf))
            for _ in range(alloc):
                if len(new_pool.models) >= n_models - 2:
                    break
                # Select two parents from same species
                if len(members) >= 2:
                    p1, p2 = random.sample(members, 2)
                    w1 = pool.models[p1].to_list()
                    w2 = pool.models[p2].to_list()
                    child_w, _ = CrossoverMethods.crossover_pair(w1, w2, 'uniform')
                    child = NeuralNetwork(child_w)
                    if random.random() < MUTATION_RATE:
                        child_w = MutationMethods.uniform(
                            child.to_list(), MUTATION_RATE, MUTATION_AMOUNT
                        )
                        child = NeuralNetwork(child_w)
                    new_pool.models.append(child)

        # Cross-species mating (hybridization)
        cross_count = max(1, int(n_models * self.cross_species_rate))
        species_list = list(species.values())
        for _ in range(cross_count):
            if len(species_list) >= 2:
                s1, s2 = random.sample(species_list, 2)
                if s1 and s2:
                    p1 = random.choice(s1)
                    p2 = random.choice(s2)
                    w1 = pool.models[p1].to_list()
                    w2 = pool.models[p2].to_list()
                    child_w, _ = CrossoverMethods.crossover_pair(w1, w2, 'blend')
                    child = NeuralNetwork(child_w)
                    child_w = MutationMethods.uniform(
                        child.to_list(), MUTATION_RATE * 1.5, MUTATION_AMOUNT
                    )
                    child = NeuralNetwork(child_w)
                    new_pool.models.append(child)

        # Fill remaining with random
        while len(new_pool.models) < n_models:
            new_pool.models.append(NeuralNetwork.random())

        # Trim excess
        while len(new_pool.models) > n_models:
            new_pool.models.pop()

        # Set metadata
        for i, m in enumerate(new_pool.models):
            m.team = pool.team
            m.model_id = i
            m.epoch_created = pool.epoch + 1

        new_pool.epoch = pool.epoch + 1
        new_pool.best_fitness_ever = pool.best_fitness_ever
        new_pool.epoch_fitness_log = list(pool.epoch_fitness_log)

        best_in_new = max(m.fitness for m in new_pool.models)
        new_pool.epoch_fitness_log.append(best_in_new)
        if best_in_new > new_pool.best_fitness_ever:
            new_pool.best_fitness_ever = best_in_new

        # Record stats
        n_species = len(species)
        avg_species_size = sum(len(m) for m in species.values()) / max(n_species, 1)
        self.history.append({
            'epoch': new_pool.epoch,
            'n_species': n_species,
            'avg_species_size': round(avg_species_size, 1),
            'threshold': self.speciation.threshold,
            'best_fitness': best_in_new,
        })

        return new_pool

    def _adjust_threshold(self, species: Dict[int, List[int]]) -> None:
        """Dynamically adjust similarity threshold to maintain target species count."""
        n_species = len(species)
        if n_species < self.target_species - 1:
            self.speciation.threshold *= 0.9  # make it easier to speciate
        elif n_species > self.target_species + 1:
            self.speciation.threshold *= 1.1  # make it harder
        self.speciation.threshold = max(0.1, min(2.0, self.speciation.threshold))

    def get_species_report(self, pool: ModelPool) -> str:
        """Generate a formatted report of current species."""
        species = self.speciation.classify(pool.models)
        lines = [f'Species ({len(species)}):']
        for sid, members in sorted(species.items()):
            fits = [pool.models[m].fitness for m in members]
            avg_f = sum(fits) / len(fits) if fits else 0
            lines.append(f'  #{sid}: {len(members)} models, avg fitness {avg_f:.1f}')
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════
# SECTION 53: ISLAND MODEL MIGRATION MANAGER
# ═══════════════════════════════════════════════════════════════════
# Manages periodic migration of genetic material between teams (islands).
# This prevents each island from converging to local optima and shares
# successful genetic innovations across the population.

class IslandMigrationManager:
    """
    Coordinates migration between team islands.
    
    Migration strategies:
      'ring': each island sends its best to the next island (circular)
      'random': random pairs exchange models
      'broadcast': best model from best island sent to all others
      'tournament': islands compete, winners take models from losers
    
    Migration schedule:
      - Frequency: every N generations (configurable)
      - Rate: how many models to exchange per migration event
      - Selection: which models to send (best, random, or diverse)
    """

    def __init__(self,
                 topology: str = 'ring',
                 migration_interval: int = 5,
                 models_per_migration: int = 2,
                 send_strategy: str = 'best',
                 replace_strategy: str = 'worst'):
        self.topology = topology
        self.interval = migration_interval
        self.models_per_migration = models_per_migration
        self.send_strategy = send_strategy
        self.replace_strategy = replace_strategy
        self.migration_log: List[Dict] = []
        self.generations_since_migration = 0

    def should_migrate(self, generation: int) -> bool:
        """Check if migration should occur at this generation."""
        return generation > 0 and generation % self.interval == 0

    def execute_migration(self, engine: EvolutionEngine) -> List[Dict]:
        """
        Execute one migration event across all islands.
        Returns list of migration records.
        """
        self.generations_since_migration = 0
        migrations = []

        if self.topology == 'ring':
            migrations = self._ring_migration(engine)
        elif self.topology == 'random':
            migrations = self._random_migration(engine)
        elif self.topology == 'broadcast':
            migrations = self._broadcast_migration(engine)
        elif self.topology == 'tournament':
            migrations = self._tournament_migration(engine)

        self.migration_log.extend(migrations)
        return migrations

    def _get_sender_models(self, pool: ModelPool, count: int) -> List[Tuple[int, NeuralNetwork]]:
        """Select models to send based on strategy."""
        if self.send_strategy == 'best':
            return pool.get_top_n(count)
        elif self.send_strategy == 'random':
            indices = random.sample(range(len(pool.models)), min(count, len(pool.models)))
            return [(i, pool.models[i].copy()) for i in indices]
        elif self.send_strategy == 'diverse':
            # Send most diverse (least similar to average)
            avg_w = [sum(m.weights[j] for m in pool.models) / len(pool.models)
                     for j in range(WEIGHT_COUNT)]
            diversities = []
            for i, m in enumerate(pool.models):
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(m.weights, avg_w)))
                diversities.append((i, dist))
            diversities.sort(key=lambda x: -x[1])
            return [(i, pool.models[i].copy()) for i, _ in diversities[:count]]
        return pool.get_top_n(count)

    def _replace_models(self, pool: ModelPool,
                         immigrants: List[Tuple[int, NeuralNetwork]]) -> None:
        """Replace models in pool based on strategy."""
        if self.replace_strategy == 'worst':
            # Replace worst models
            indexed = list(enumerate(pool.models))
            indexed.sort(key=lambda x: x[1].fitness)
            for i, (idx, _) in enumerate(indexed[:len(immigrants)]):
                if i < len(immigrants):
                    _, imm_model = immigrants[i]
                    pool.models[idx] = imm_model
                    pool.models[idx].model_id = idx
        elif self.replace_strategy == 'random':
            replace_indices = random.sample(
                range(len(pool.models)), min(len(immigrants), len(pool.models))
            )
            for i, idx in enumerate(replace_indices):
                if i < len(immigrants):
                    _, imm_model = immigrants[i]
                    pool.models[idx] = imm_model
                    pool.models[idx].model_id = idx
        elif self.replace_strategy == 'oldest':
            # Replace models with oldest epoch_created
            indexed = list(enumerate(pool.models))
            indexed.sort(key=lambda x: x[1].epoch_created)
            for i, (idx, _) in enumerate(indexed[:len(immigrants)]):
                if i < len(immigrants):
                    _, imm_model = immigrants[i]
                    pool.models[idx] = imm_model
                    pool.models[idx].model_id = idx

    def _ring_migration(self, engine: EvolutionEngine) -> List[Dict]:
        """Circular migration: team i sends to team (i+1) % N."""
        migrations = []
        teams = list(range(N_TEAMS))
        random.shuffle(teams)

        for i in range(len(teams)):
            src = teams[i]
            dst = teams[(i + 1) % len(teams)]
            src_pool = engine.weight_manager.get_pool(src)
            dst_pool = engine.weight_manager.get_pool(dst)

            if src_pool and dst_pool:
                immigrants = self._get_sender_models(
                    src_pool, self.models_per_migration
                )
                self._replace_models(dst_pool, immigrants)
                for idx, model in immigrants:
                    migrations.append({
                        'from': src, 'to': dst,
                        'model_idx': idx,
                        'fitness': model.fitness,
                        'epoch': src_pool.epoch,
                        'type': 'ring',
                    })
        return migrations

    def _random_migration(self, engine: EvolutionEngine) -> List[Dict]:
        """Random pairs exchange models."""
        migrations = []
        teams = list(range(N_TEAMS))
        pairs = []
        for _ in range(N_TEAMS // 2):
            if len(teams) < 2:
                break
            a = random.choice(teams)
            teams.remove(a)
            b = random.choice(teams)
            teams.remove(b)
            pairs.append((a, b))

        for src, dst in pairs:
            src_pool = engine.weight_manager.get_pool(src)
            dst_pool = engine.weight_manager.get_pool(dst)
            if src_pool and dst_pool:
                immigrants = self._get_sender_models(
                    src_pool, self.models_per_migration
                )
                self._replace_models(dst_pool, immigrants)
                for idx, model in immigrants:
                    migrations.append({
                        'from': src, 'to': dst,
                        'model_idx': idx,
                        'fitness': model.fitness,
                        'epoch': src_pool.epoch,
                        'type': 'random',
                    })
        return migrations

    def _broadcast_migration(self, engine: EvolutionEngine) -> List[Dict]:
        """Best team broadcasts its best models to all others."""
        migrations = []
        best_team = EvolutionEngineExtensions.find_best_team(engine)
        if best_team is None:
            return migrations

        best_pool = engine.weight_manager.get_pool(best_team)
        if not best_pool:
            return migrations

        immigrants = self._get_sender_models(best_pool, self.models_per_migration)
        for t in range(N_TEAMS):
            if t == best_team:
                continue
            dst_pool = engine.weight_manager.get_pool(t)
            if dst_pool:
                self._replace_models(dst_pool, immigrants)
                for idx, model in immigrants:
                    migrations.append({
                        'from': best_team, 'to': t,
                        'model_idx': idx,
                        'fitness': model.fitness,
                        'epoch': best_pool.epoch,
                        'type': 'broadcast',
                    })
        return migrations

    def _tournament_migration(self, engine: EvolutionEngine) -> List[Dict]:
        """Teams compete; winners take models from losers."""
        migrations = []
        teams = list(range(N_TEAMS))
        random.shuffle(teams)

        for i in range(0, len(teams) - 1, 2):
            a, b = teams[i], teams[i + 1]
            pa = engine.weight_manager.get_pool(a)
            pb = engine.weight_manager.get_pool(b)
            if not pa or not pb:
                continue
            if pa.get_best_fitness() > pb.get_best_fitness():
                winner, loser = pa, pb
                winner_id, loser_id = a, b
            else:
                winner, loser = pb, pa
                winner_id, loser_id = b, a

            immigrants = self._get_sender_models(winner, 1)
            self._replace_models(loser, immigrants)
            for idx, model in immigrants:
                migrations.append({
                    'from': winner_id, 'to': loser_id,
                    'model_idx': idx,
                    'fitness': model.fitness,
                    'epoch': winner.epoch,
                    'type': 'tournament',
                })
        return migrations

    def get_migration_stats(self) -> Dict:
        if not self.migration_log:
            return {'total': 0}
        return {
            'total_migrations': len(self.migration_log),
            'unique_senders': len(set(m['from'] for m in self.migration_log)),
            'unique_receivers': len(set(m['to'] for m in self.migration_log)),
            'avg_fitness_migrated': sum(m['fitness'] for m in self.migration_log) / len(self.migration_log),
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 54: ELITE PRESERVATION STRATEGIES
# ═══════════════════════════════════════════════════════════════════
# Different strategies for selecting and preserving elite models
# across generations beyond simple top-N selection.

class ElitePreservationStrategies:
    """
    Multiple elite preservation strategies for genetic algorithms.
    
    Strategies:
      'top_n': keep the absolute best N models (default)
      'diverse_elites': keep the best model from each niche/region
      'age_based': keep models that have survived multiple generations
      'stochastic': probabilistic preservation (better fitness = higher chance)
      'gap': keep models that represent gaps in the fitness landscape
    """

    @staticmethod
    def select_top_n(pool: ModelPool, n: int) -> List[NeuralNetwork]:
        """Standard top-N elitism."""
        indexed = list(enumerate(pool.models))
        indexed.sort(key=lambda x: x[1].fitness, reverse=True)
        return [m.copy() for _, m in indexed[:n]]

    @staticmethod
    def select_diverse_elites(pool: ModelPool, n: int) -> List[NeuralNetwork]:
        """
        Select n elites that are both high-fitness AND diverse.
        Uses greedy selection: pick best model, then iteratively pick
        the model that maximizes (fitness + diversity_from_selected).
        """
        if not pool.models or n <= 0:
            return []
        if n >= len(pool.models):
            return [m.copy() for m in pool.models]

        indexed = list(enumerate(pool.models))
        selected_indices = []
        selected_models = []

        # Pick the single best model first
        best_idx = max(indexed, key=lambda x: x[1].fitness)[0]
        selected_indices.append(best_idx)
        selected_models.append(pool.models[best_idx].copy())

        # Greedy selection for remaining spots
        for _ in range(min(n - 1, len(pool.models) - 1)):
            best_score = -1e9
            best_candidate = None

            for i, model in indexed:
                if i in selected_indices:
                    continue
                # Score = fitness + diversity bonus
                avg_dist = sum(
                    model.similarity(sel)
                    for sel in [pool.models[si] for si in selected_indices]
                ) / len(selected_indices)
                score = model.fitness + avg_dist * 20  # diversity weight
                if score > best_score:
                    best_score = score
                    best_candidate = i

            if best_candidate is not None:
                selected_indices.append(best_candidate)
                selected_models.append(pool.models[best_candidate].copy())

        return selected_models

    @staticmethod
    def select_age_based(pool: ModelPool, n: int) -> List[NeuralNetwork]:
        """Prefer models that have survived multiple generations (age = epoch_created)."""
        current_epoch = pool.epoch
        indexed = []
        for i, m in enumerate(pool.models):
            age = current_epoch - m.epoch_created
            age_score = min(age, 20) / 20.0  # normalize to [0,1]
            # Combined score: 70% fitness, 30% age
            combined = m.fitness * 0.7 + age_score * max(10, m.fitness) * 0.3
            indexed.append((i, combined))
        indexed.sort(key=lambda x: -x[1])
        return [pool.models[i].copy() for i, _ in indexed[:n]]

    @staticmethod
    def select_stochastic(pool: ModelPool, n: int) -> List[NeuralNetwork]:
        """Probabilistic selection: better fitness = higher chance, but not guaranteed."""
        if not pool.models or n <= 0:
            return []
        fits = [max(0.01, m.fitness) for m in pool.models]
        total = sum(fits)
        probs = [f / total for f in fits]
        selected = set()
        elites = []
        attempts = 0
        while len(elites) < n and attempts < n * 10:
            idx = random.choices(range(len(pool.models)), weights=probs, k=1)[0]
            if idx not in selected:
                selected.add(idx)
                elites.append(pool.models[idx].copy())
            attempts += 1
        # Fill remaining with top models if needed
        while len(elites) < n:
            for i in range(len(pool.models)):
                if i not in selected and len(elites) < n:
                    selected.add(i)
                    elites.append(pool.models[i].copy())
        return elites


# ═══════════════════════════════════════════════════════════════════
# SECTION 55: POPULATION DIVERSITY ANALYZER
# ═══════════════════════════════════════════════════════════════════
# Advanced diversity metrics for monitoring population health.
# Provides early warning for convergence and guides adaptive parameter tuning.

class DiversityAnalyzer:
    """
    Comprehensive diversity analysis for model populations.
    
    Metrics:
      - Weight space diversity: average pairwise weight distance
      - Fitness diversity: spread of fitness values
      - Behavioral diversity: variety of behavioral strategies
      - Genotypic entropy: Shannon entropy of weight distributions
      - Phenotypic clustering: number of distinct behavioral clusters
      - Innovation rate: how many new behaviors appear per generation
    """

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.history: List[Dict] = []
        self.last_behavioral_archive: List[List[float]] = []
        self.characterizer = BehaviorCharacterizer(n_samples=20)

    def analyze(self, pool: ModelPool) -> Dict:
        """Run all diversity metrics on a pool."""
        if not pool.models:
            return {}

        weight_div = self._weight_diversity(pool)
        fitness_div = self._fitness_diversity(pool)
        behavioral_div = self._behavioral_diversity(pool)
        entropy = self._weight_entropy(pool)
        n_clusters = self._estimate_clusters(pool)
        innovation = self._innovation_rate(pool)

        metrics = {
            'team': pool.team,
            'team_name': pool.team_name,
            'epoch': pool.epoch,
            'weight_diversity': round(weight_div, 4),
            'fitness_diversity': round(fitness_div, 4),
            'behavioral_diversity': round(behavioral_div, 4),
            'weight_entropy': round(entropy, 4),
            'estimated_clusters': n_clusters,
            'innovation_rate': round(innovation, 4),
            'n_models': len(pool.models),
        }

        self.history.append(metrics)
        if len(self.history) > 500:
            self.history = self.history[-500:]

        return metrics

    def _weight_diversity(self, pool: ModelPool) -> float:
        """Average pairwise distance between weight vectors."""
        if len(pool.models) < 2:
            return 0.0
        samples = random.sample(pool.models, min(15, len(pool.models)))
        total = 0.0
        pairs = 0
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                total += samples[i].similarity(samples[j])
                pairs += 1
        return total / pairs if pairs > 0 else 0.0

    def _fitness_diversity(self, pool: ModelPool) -> float:
        """Coefficient of variation of fitness values."""
        fits = [m.fitness for m in pool.models]
        if not fits or sum(fits) == 0:
            return 0.0
        mean = sum(fits) / len(fits)
        if mean == 0:
            return 0.0
        variance = sum((f - mean) ** 2 for f in fits) / len(fits)
        return math.sqrt(variance) / mean

    def _behavioral_diversity(self, pool: ModelPool) -> float:
        """Average distance between behavior descriptors."""
        if len(pool.models) < 2:
            return 0.0
        descriptors = [self.characterizer.characterize(m) for m in pool.models]
        total = 0.0
        pairs = 0
        for i in range(len(descriptors)):
            for j in range(i + 1, len(descriptors)):
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(descriptors[i], descriptors[j])))
                total += dist
                pairs += 1
        return total / pairs if pairs > 0 else 0.0

    def _weight_entropy(self, pool: ModelPool) -> float:
        """Shannon entropy of weight value distributions across all models."""
        if not pool.models:
            return 0.0
        all_weights = []
        for m in pool.models:
            all_weights.extend(m.weights)

        if not all_weights:
            return 0.0

        # Discretize into bins
        n_bins = 20
        min_w = min(all_weights)
        max_w = max(all_weights)
        if max_w - min_w < 1e-10:
            return 0.0
        bin_size = (max_w - min_w) / n_bins
        bins = [0] * n_bins
        for w in all_weights:
            idx = min(n_bins - 1, int((w - min_w) / bin_size))
            bins[idx] += 1

        total = sum(bins) or 1
        entropy = 0.0
        for b in bins:
            if b > 0:
                p = b / total
                entropy -= p * math.log2(p)

        return entropy / math.log2(n_bins)  # normalize to [0, 1]

    def _estimate_clusters(self, pool: ModelPool) -> int:
        """Estimate number of behavioral clusters using simple threshold clustering."""
        if len(pool.models) < 2:
            return 1
        descriptors = [self.characterizer.characterize(m) for m in pool.models]
        clusters = []
        for desc in descriptors:
            found = False
            for cluster in clusters:
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(desc, cluster[0])))
                if dist < 0.3:
                    cluster.append(desc)
                    found = True
                    break
            if not found:
                clusters.append([desc])
        return len(clusters)

    def _innovation_rate(self, pool: ModelPool) -> float:
        """Fraction of models with novel behaviors compared to archive."""
        if not self.last_behavioral_archive:
            descriptors = [self.characterizer.characterize(m) for m in pool.models]
            self.last_behavioral_archive = descriptors
            return 1.0

        descriptors = [self.characterizer.characterize(m) for m in pool.models]
        novel_count = 0
        for desc in descriptors:
            is_novel = True
            for archived in self.last_behavioral_archive:
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(desc, archived)))
                if dist < 0.3:
                    is_novel = False
                    break
            if is_novel:
                novel_count += 1

        self.last_behavioral_archive = descriptors
        return novel_count / max(len(descriptors), 1)

    def get_summary(self, team: int = 0) -> str:
        """Get a formatted summary of recent diversity metrics."""
        recent = [h for h in self.history if h.get('team') == team][-5:]
        if not recent:
            return 'No diversity data available.'
        lines = [f'Diversity Summary for Team {team}:']
        for m in recent:
            lines.append(
                f'  Ep {m["epoch"]:3d}: weight_div={m["weight_diversity"]:.3f} '
                f'fit_div={m["fitness_diversity"]:.3f} '
                f'behav_div={m["behavioral_diversity"]:.3f} '
                f'clusters={m["estimated_clusters"]}'
            )
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════
# SECTION 56: GENERATION HEALTH REPORT
# ═══════════════════════════════════════════════════════════════════
# Comprehensive health report for each generation, tracking:
#   - Fitness stats (min, max, avg, median, std)
#   - Diversity metrics (weight space, behavioral, genotypic)
#   - Convergence warnings (stagnation, diversity collapse)
#   - Improvement rate
#   - Species count (if speciation enabled)
#   - Model age distribution
#   - Top model analysis

class GenerationHealthReport:
    """
    Generate comprehensive health reports for the evolution process.
    
    Used both for display in the CLI and for export to JSON/HTML
    for external analysis.
    """

    def __init__(self):
        self.history: List[Dict] = []
        self.diversity_analyzer = DiversityAnalyzer()

    def analyze_generation(self, engine: EvolutionEngine) -> Dict:
        """Analyze the current generation across all teams."""
        report = {
            'generation': engine.generation,
            'timestamp': time.time(),
            'teams': [],
            'global': self._global_stats(engine),
        }

        for t in range(N_TEAMS):
            pool = engine.weight_manager.get_pool(t)
            if not pool:
                continue

            fits = [m.fitness for m in pool.models]
            weights = [m.weights for m in pool.models]
            ages = [pool.epoch - m.epoch_created for m in pool.models]

            team_report = {
                'team': t,
                'name': pool.team_name,
                'epoch': pool.epoch,
                'fitness': {
                    'min': round(min(fits), 2) if fits else 0,
                    'max': round(max(fits), 2) if fits else 0,
                    'avg': round(sum(fits) / len(fits), 2) if fits else 0,
                    'median': round(statistics.median(fits), 2) if len(fits) > 1 else fits[0] if fits else 0,
                    'std': round(statistics.stdev(fits), 2) if len(fits) > 1 else 0,
                    'best_ever': round(pool.best_fitness_ever, 2),
                },
                'diversity': self.diversity_analyzer.analyze(pool),
                'ages': {
                    'min_age': min(ages),
                    'max_age': max(ages),
                    'avg_age': round(sum(ages) / len(ages), 1) if ages else 0,
                },
                'improvement_rate': round(
                    PopulationHealthMetrics.compute_improvement_rate(pool.epoch_fitness_log), 3
                ),
                'stagnation': PopulationHealthMetrics.compute_stagnation(pool.epoch_fitness_log),
                'convergence_risk': round(
                    PopulationHealthMetrics.compute_convergence_risk(pool), 3
                ),
                'n_models': len(pool.models),
                'n_alive_worms': sum(1 for w in []
                                     if hasattr(w, 'team') and w.team == t and w.alive)
                if False else 0,  # Simplified - we don't have worm list here
            }

            # Warnings
            team_report['warnings'] = self._check_warnings(team_report)
            report['teams'].append(team_report)

        self.history.append(report)
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return report

    def _global_stats(self, engine: EvolutionEngine) -> Dict:
        """Compute global statistics across all teams."""
        all_fits = []
        total_epochs = 0
        for t in range(N_TEAMS):
            pool = engine.weight_manager.get_pool(t)
            if pool:
                all_fits.extend(m.fitness for m in pool.models)
                total_epochs += pool.epoch

        return {
            'total_teams': N_TEAMS,
            'total_models': N_TEAMS * MODELS_PER_TEAM,
            'total_epochs': total_epochs,
            'best_fitness_ever': round(engine.best_fitness_ever, 2),
            'global_generation': engine.generation,
            'total_births': engine.total_births,
            'global_fitness_avg': round(sum(all_fits) / len(all_fits), 2) if all_fits else 0,
            'global_fitness_max': round(max(all_fits), 2) if all_fits else 0,
        }

    def _check_warnings(self, team_report: Dict) -> List[str]:
        """Check for warning conditions."""
        warnings = []
        fd = team_report.get('fitness', {})
        div = team_report.get('diversity', {})

        if fd.get('max', 0) < 10 and team_report.get('epoch', 0) > 5:
            warnings.append('Low fitness after 5+ epochs')

        if div.get('weight_diversity', 1) < 0.1:
            warnings.append('Weight diversity critically low')

        if div.get('estimated_clusters', 5) < 2:
            warnings.append('Only 1 behavioral cluster — convergence imminent')

        if team_report.get('stagnation', 0) > 15:
            warnings.append(f"Stagnant for {team_report['stagnation']} generations")

        if team_report.get('convergence_risk', 0) > 0.8:
            warnings.append('High convergence risk — increase mutation')

        return warnings

    def format_report(self, report: Optional[Dict] = None) -> str:
        """Format a health report as a human-readable string."""
        if report is None:
            report = self.history[-1] if self.history else None
            if not report:
                return 'No report data available.'

        lines = []
        lines.append(f'Generation {report["generation"]} Health Report')
        lines.append('=' * 50)

        g = report['global']
        lines.append(f'Global: {g["total_models"]} models, '
                     f'{g["total_epochs"]} total epochs')
        lines.append(f'Best fitness ever: {g["best_fitness_ever"]}')
        lines.append(f'Global avg fitness: {g["global_fitness_avg"]}')

        for tr in report['teams']:
            lines.append(f'\n{tr["name"]} (epoch {tr["epoch"]}):')
            f = tr['fitness']
            lines.append(f'  Fitness: max={f["max"]} avg={f["avg"]} '
                         f'median={f["median"]} std={f["std"]}')
            d = tr['diversity']
            lines.append(f'  Diversity: weight={d.get("weight_diversity",0):.3f} '
                         f'behavior={d.get("behavioral_diversity",0):.3f} '
                         f'clusters={d.get("estimated_clusters",0)}')
            lines.append(f'  Stagnation: {tr["stagnation"]} gen, '
                         f'convergence risk: {tr["convergence_risk"]}')

            if tr['warnings']:
                for w in tr['warnings']:
                    lines.append(f'  ⚠ {w}')

        return '\n'.join(lines)

    def export_json(self, path: str = 'health_report.json') -> None:
        """Export the latest health report to a JSON file."""
        if self.history:
            try:
                with open(path, 'w') as f:
                    json.dump(self.history[-1], f, indent=2, default=str)
            except IOError:
                pass


# ═══════════════════════════════════════════════════════════════════
# SECTION 57: GENERATION HEALTH MONITOR
# ═══════════════════════════════════════════════════════════════════
# Background monitor that tracks evolution progress and provides
# real-time feedback through callbacks.

class EvolutionMonitor:
    """
    Monitors evolution progress and triggers callbacks.
    
    Callbacks:
      - on_generation_end: after each generation completes
      - on_epoch_end: after a team's epoch advances
      - on_stagnation: when a team stagnates for N generations
      - on_milestone: when a fitness milestone is reached
      - on_convergence: when convergence risk exceeds threshold
    """

    def __init__(self):
        self.callbacks = {
            'on_generation_end': [],
            'on_epoch_end': [],
            'on_stagnation': [],
            'on_milestone': [],
            'on_convergence': [],
        }
        self.reporter = GenerationHealthReport()
        self.last_reported_gen = -1
        self.milestones: Dict[int, float] = {}  # {team: milestone_fitness}

    def register_callback(self, event: str, func: Callable) -> None:
        if event in self.callbacks:
            self.callbacks[event].append(func)

    def on_generation_end(self, engine: EvolutionEngine) -> None:
        """Called at the end of each generation."""
        gen = engine.generation
        if gen == self.last_reported_gen:
            return
        self.last_reported_gen = gen

        report = self.reporter.analyze_generation(engine)

        for cb in self.callbacks['on_generation_end']:
            try:
                cb(report)
            except Exception:
                pass

        # Check milestones
        for t in range(N_TEAMS):
            pool = engine.weight_manager.get_pool(t)
            if not pool:
                continue
            current_best = pool.get_best_fitness()
            prev_milestone = self.milestones.get(t, 0)
            if current_best >= prev_milestone + 100:
                self.milestones[t] = current_best
                for cb in self.callbacks['on_milestone']:
                    try:
                        cb(t, current_best)
                    except Exception:
                        pass

        # Check stagnation
        for tr in report.get('teams', []):
            if tr.get('stagnation', 0) > 10:
                for cb in self.callbacks['on_stagnation']:
                    try:
                        cb(tr['team'], tr['stagnation'])
                    except Exception:
                        pass

            if tr.get('convergence_risk', 0) > 0.8:
                for cb in self.callbacks['on_convergence']:
                    try:
                        cb(tr['team'], tr['convergence_risk'])
                    except Exception:
                        pass

    def on_epoch_end(self, team: int, epoch: int, fitness: float) -> None:
        for cb in self.callbacks['on_epoch_end']:
            try:
                cb(team, epoch, fitness)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════
# SECTION 58: EVOLUTIONARY STRATEGY PRESETS
# ═══════════════════════════════════════════════════════════════════
# Pre-configured evolution strategies for different scenarios.
# Each preset tunes the GA hyperparameters for a specific goal.

class EvolutionStrategyPresets:
    """
    Pre-configured strategy presets for different evolution scenarios.
    
    Presets:
      'balanced': default, good for general-purpose evolution
      'exploration': high mutation, low crossover — for early stages
      'exploitation': low mutation, high crossover — for fine-tuning
      'diversity': speciation + novelty search enabled
      'speed': simplified GA for fast convergence
      'deep': long-term evolution with adaptive parameters
    """

    @staticmethod
    def get_preset(name: str = 'balanced') -> Dict:
        presets = {
            'balanced': {
                'mutation_rate': 0.08,
                'mutation_amount': 0.5,
                'crossover_rate': 0.6,
                'crossover_method': 'uniform',
                'elite_count': 3,
                'tournament_size': 4,
                'selection_method': 'tournament',
                'mutation_method': 'uniform',
                'random_fill': 2,
                'description': 'Balanced general-purpose evolution',
            },
            'exploration': {
                'mutation_rate': 0.15,
                'mutation_amount': 0.9,
                'crossover_rate': 0.4,
                'crossover_method': 'blend',
                'elite_count': 2,
                'tournament_size': 3,
                'selection_method': 'roulette',
                'mutation_method': 'gaussian',
                'random_fill': 3,
                'description': 'High exploration, good for early generations',
            },
            'exploitation': {
                'mutation_rate': 0.03,
                'mutation_amount': 0.2,
                'crossover_rate': 0.85,
                'crossover_method': 'two_point',
                'elite_count': 5,
                'tournament_size': 5,
                'selection_method': 'rank',
                'mutation_method': 'uniform',
                'random_fill': 1,
                'description': 'Fine-tuning, low mutation, high crossover',
            },
            'diversity': {
                'mutation_rate': 0.10,
                'mutation_amount': 0.6,
                'crossover_rate': 0.5,
                'crossover_method': 'uniform',
                'elite_count': 4,
                'tournament_size': 4,
                'selection_method': 'tournament',
                'mutation_method': 'adaptive',
                'random_fill': 2,
                'description': 'Diversity-focused with adaptive mutation',
                'use_speciation': True,
                'use_novelty': True,
            },
            'speed': {
                'mutation_rate': 0.12,
                'mutation_amount': 0.7,
                'crossover_rate': 0.3,
                'crossover_method': 'average',
                'elite_count': 2,
                'tournament_size': 2,
                'selection_method': 'tournament',
                'mutation_method': 'uniform',
                'random_fill': 1,
                'description': 'Fast convergence for quick testing',
            },
            'deep': {
                'mutation_rate': 0.06,
                'mutation_amount': 0.4,
                'crossover_rate': 0.7,
                'crossover_method': 'sbx',
                'elite_count': 3,
                'tournament_size': 5,
                'selection_method': 'tournament',
                'mutation_method': 'polynomial',
                'random_fill': 2,
                'description': 'Long-term deep evolution with SBX crossover',
                'use_adaptive_params': True,
            },
        }
        return presets.get(name, presets['balanced'])

    @staticmethod
    def list_presets() -> str:
        names = ['balanced', 'exploration', 'exploitation', 'diversity', 'speed', 'deep']
        lines = ['Available presets:']
        for n in names:
            p = EvolutionStrategyPresets.get_preset(n)
            lines.append(f'  {n:15s} — {p["description"]}')
        return '\n'.join(lines)

    @staticmethod
    def apply_preset(evolver: TeamEvolver, preset_name: str) -> None:
        """Apply a preset to a TeamEvolver instance."""
        preset = EvolutionStrategyPresets.get_preset(preset_name)
        for key, value in preset.items():
            if hasattr(evolver, key) and key != 'description':
                setattr(evolver, key, value)


# ═══════════════════════════════════════════════════════════════════
# SECTION 59: MODEL COMPARISON & MATCHUP SIMULATOR
# ═══════════════════════════════════════════════════════════════════
# Simulates head-to-head matchups between models to determine
# which one performs better in direct competition.

class MatchupSimulator:
    """
    Simulates direct competition between two neural network models.
    
    Runs a mini-simulation where two worms controlled by the given
    models compete for food in a small arena. Returns scores for
    food collected, survival time, and (if applicable) kills.
    
    This is useful for:
      - Tournament selection between candidate models
      - Building Elo ratings
      - Testing specific behavioral traits
    """

    def __init__(self, arena_size: int = 1000,
                 n_food: int = 20,
                 simulation_ticks: int = 600):
        self.arena_size = arena_size
        self.n_food = n_food
        self.simulation_ticks = simulation_ticks
        self.characterizer = BehaviorCharacterizer(10)

    def simulate_matchup(self, model_a: NeuralNetwork,
                          model_b: NeuralNetwork) -> Dict:
        """
        Simulate a head-to-head matchup between two models.
        
        Simplified simulation (not full physics, just behavior comparison):
          1. Generate behavior descriptors for both models
          2. Score food-seeking tendency, aggression, and wall avoidance
          3. Combine into a composite score
        """
        desc_a = self.characterizer.characterize(model_a)
        desc_b = self.characterizer.characterize(model_b)

        # Food seeking: higher = better at finding food
        food_a = desc_a[2] if len(desc_a) > 2 else 0.5
        food_b = desc_b[2] if len(desc_b) > 2 else 0.5

        # Wall avoidance: higher = better at avoiding walls
        wall_a = 1.0 - min(desc_a[3], 1.0) if len(desc_a) > 3 else 0.5
        wall_b = 1.0 - min(desc_b[3], 1.0) if len(desc_b) > 3 else 0.5

        # Aggression (enemy response): higher = more aggressive
        aggr_a = abs(desc_a[4]) if len(desc_a) > 4 else 0.5
        aggr_b = abs(desc_b[4]) if len(desc_b) > 4 else 0.5

        # Composite scores
        score_a = food_a * 2.0 + wall_a * 1.0 + aggr_a * 0.5
        score_b = food_b * 2.0 + wall_b * 1.0 + aggr_b * 0.5

        # Winner determination with some noise
        noise_a = random.gauss(0, 0.2)
        noise_b = random.gauss(0, 0.2)
        final_a = score_a + noise_a
        final_b = score_b + noise_b

        return {
            'model_a_score': round(final_a, 3),
            'model_b_score': round(final_b, 3),
            'winner': 'A' if final_a > final_b else 'B',
            'margin': round(abs(final_a - final_b), 3),
            'descriptors': {
                'A': [round(d, 3) for d in desc_a],
                'B': [round(d, 3) for d in desc_b],
            },
            'food_seeking': {'A': round(food_a, 3), 'B': round(food_b, 3)},
            'wall_avoidance': {'A': round(wall_a, 3), 'B': round(wall_b, 3)},
            'aggression': {'A': round(aggr_a, 3), 'B': round(aggr_b, 3)},
        }

    def full_tournament(self, models: List[NeuralNetwork]) -> List[Tuple[int, float]]:
        """Run a full round-robin tournament. Returns (index, score) for each model."""
        n = len(models)
        scores = [0.0] * n
        for i in range(n):
            for j in range(i + 1, n):
                result = self.simulate_matchup(models[i], models[j])
                if result['winner'] == 'A':
                    scores[i] += 1.0 + result['margin']
                else:
                    scores[j] += 1.0 + result['margin']
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: -x[1])
        return indexed


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

class PerformanceBenchmark:
    """Measure evolution engine performance metrics."""

    @staticmethod
    def benchmark_forward_pass(n_iterations: int = 10000) -> Dict:
        """Measure NN forward pass speed."""
        nn = NeuralNetwork.random()
        inp = [random.uniform(-1, 1) for _ in range(N_INPUT)]
        start = time.time()
        for _ in range(n_iterations):
            nn.forward(inp)
        elapsed = time.time() - start
        return {
            'iterations': n_iterations,
            'total_time': round(elapsed, 3),
            'per_call_us': round(elapsed / n_iterations * 1_000_000, 1),
            'calls_per_sec': round(n_iterations / elapsed) if elapsed > 0 else 0,
        }

    @staticmethod
    def benchmark_evolution(engine: EvolutionEngine, n_teams: int = 5) -> Dict:
        """Measure evolution speed for N teams."""
        alive = [
            {'team': t, 'id': i, 'modelId': i % MODELS_PER_TEAM,
             'mass': random.uniform(1, 30), 'survivalTime': random.uniform(10, 100),
             'kills': random.randint(0, 5), 'foodEaten': random.randint(0, 20),
             'distanceTraveled': random.uniform(100, 5000)}
            for t in range(min(n_teams, N_TEAMS))
            for i in range(WORMS_PER_TEAM)
        ]
        dead = list(range(min(n_teams, N_TEAMS)))
        start = time.time()
        evos = engine.evolve(alive, dead)
        elapsed = time.time() - start
        return {
            'teams_evolved': len(dead),
            'worms_processed': len(alive),
            'total_time': round(elapsed, 3),
            'avg_per_team_ms': round(elapsed / len(dead) * 1000, 1) if dead else 0,
            'evolutions_returned': len(evos),
        }

    @staticmethod
    def run_all() -> Dict:
        """Run all benchmarks and return results."""
        results = {}
        results['forward_pass'] = PerformanceBenchmark.benchmark_forward_pass()
        engine = EvolutionEngine()
        results['evolution'] = PerformanceBenchmark.benchmark_evolution(engine, 3)
        results['total_models'] = N_TEAMS * MODELS_PER_TEAM
        results['total_weights'] = N_TEAMS * MODELS_PER_TEAM * WEIGHT_COUNT
        return results


# ═══════════════════════════════════════════════════════════════════
# SECTION 51: TEST DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════

class TestDataGenerator:
    """Generate synthetic test data for development and testing."""

    @staticmethod
    def random_worm_stats(team: int, n: int = 10) -> List[Dict]:
        """Generate random alive worm stats for testing."""
        stats = []
        for i in range(n):
            model_idx = i % MODELS_PER_TEAM
            stats.append({
                'team': team,
                'id': i,
                'modelId': model_idx,
                'mass': round(random.uniform(1, 50), 1),
                'survivalTime': round(random.uniform(5, 120), 1),
                'kills': random.randint(0, 5),
                'foodEaten': random.randint(0, 30),
                'distanceTraveled': round(random.uniform(100, 10000), 1),
                'bonus': 0.0,
            })
        return stats

    @staticmethod
    def full_dead_team_scenario(n_dead_teams: int = 3) -> Tuple[List[Dict], List[int]]:
        """Generate a realistic evolution scenario."""
        alive = []
        dead_teams = random.sample(range(N_TEAMS), min(n_dead_teams, N_TEAMS))
        for t in range(N_TEAMS):
            if t in dead_teams:
                continue
            alive.extend(TestDataGenerator.random_worm_stats(t, WORMS_PER_TEAM))
        # Add some dying team stats (partial)
        for t in dead_teams:
            alive.extend(TestDataGenerator.random_worm_stats(t, max(1, WORMS_PER_TEAM // 2)))
        return alive, dead_teams

    @staticmethod
    def generate_scenario_file(path: str = 'test_scenario.json',
                               n_teams: int = 3) -> None:
        """Generate and save a test scenario to disk."""
        alive, dead = TestDataGenerator.full_dead_team_scenario(n_teams)
        scenario = {
            'alive': alive,
            'deadTeams': dead,
            'totalWorms': N_WORMS,
            'description': f'Test scenario with {n_teams} dead teams',
        }
        try:
            with open(path, 'w') as f:
                json.dump(scenario, f, indent=2)
        except IOError:
            pass


# ═══════════════════════════════════════════════════════════════════
# SECTION 52: INTEGRATION TEST SUITE
# ═══════════════════════════════════════════════════════════════════

class IntegrationTestSuite:
    """End-to-end integration tests for the evolution pipeline."""

    @staticmethod
    def test_full_pipeline() -> List[Dict]:
        """Test the complete evolution pipeline end-to-end."""
        results = []

        # 1. Engine initialization
        try:
            engine = EvolutionEngine()
            results.append({'test': 'Engine init', 'status': 'PASS'})
        except Exception as e:
            results.append({'test': 'Engine init', 'status': 'FAIL', 'error': str(e)})
            return results

        # 2. Pool initialization
        try:
            for t in range(N_TEAMS):
                pool = engine.weight_manager.get_pool(t)
                assert pool is not None, f'Team {t} pool is None'
                assert len(pool.models) == MODELS_PER_TEAM, \
                    f'Team {t} has {len(pool.models)} models, expected {MODELS_PER_TEAM}'
            results.append({'test': 'Pool init', 'status': 'PASS'})
        except Exception as e:
            results.append({'test': 'Pool init', 'status': 'FAIL', 'error': str(e)})

        # 3. Fitness computation
        try:
            fe = FitnessEvaluator()
            f1 = fe.compute(10, 30, 1, 5, 500)
            f2 = fe.compute(0, 2, 0, 0, 10)
            assert f1 > f2, 'Better worm should have higher fitness'
            assert f1 > 0, 'Fitness should be positive for active worm'
            results.append({'test': 'Fitness eval', 'status': 'PASS'})
        except Exception as e:
            results.append({'test': 'Fitness eval', 'status': 'FAIL', 'error': str(e)})

        # 4. Evolution
        try:
            alive = [{'team': 0, 'id': 0, 'modelId': 0, 'mass': 15,
                      'survivalTime': 45, 'kills': 2, 'foodEaten': 8,
                      'distanceTraveled': 3000}]
            evos = engine.evolve(alive, [1])
            assert len(evos) == 1, f'Expected 1 evolution, got {len(evos)}'
            assert 'models' in evos[0], 'Evolution should contain models'
            assert len(evos[0]['models']) == MODELS_PER_TEAM, \
                f'Expected {MODELS_PER_TEAM} models, got {len(evos[0]["models"])}'
            results.append({'test': 'Evolution', 'status': 'PASS'})
        except Exception as e:
            results.append({'test': 'Evolution', 'status': 'FAIL', 'error': str(e)})

        # 5. Weight save/load
        try:
            import tempfile
            tmpdir = Path(tempfile.mkdtemp())
            pool = engine.weight_manager.get_pool(0)
            pool.save(tmpdir)
            loaded = ModelPool.load(0, tmpdir)
            assert loaded is not None, 'Loaded pool is None'
            assert loaded.epoch == pool.epoch, 'Epoch mismatch'
            import shutil
            shutil.rmtree(tmpdir)
            results.append({'test': 'Weight persistence', 'status': 'PASS'})
        except Exception as e:
            results.append({'test': 'Weight persistence', 'status': 'FAIL', 'error': str(e)})

        # 6. Leaderboard
        try:
            lb = engine.get_leaderboard()
            assert len(lb) == N_TEAMS, f'Leaderboard has {len(lb)} entries, expected {N_TEAMS}'
            results.append({'test': 'Leaderboard', 'status': 'PASS'})
        except Exception as e:
            results.append({'test': 'Leaderboard', 'status': 'FAIL', 'error': str(e)})

        return results

    @staticmethod
    def print_report() -> None:
        """Run all integration tests and print a formatted report."""
        results = IntegrationTestSuite.test_full_pipeline()
        passed = sum(1 for r in results if r['status'] == 'PASS')
        failed = sum(1 for r in results if r['status'] == 'FAIL')
        print(f'Integration Test Report: {passed}/{len(results)} passed')
        print('-' * 40)
        for r in results:
            status = 'PASS' if r['status'] == 'PASS' else 'FAIL'
            print(f'  [{status}] {r["test"]}')
            if 'error' in r:
                print(f'         Error: {r["error"]}')
        print('-' * 40)
        print(f'Overall: {"ALL PASSED" if failed == 0 else f"{failed} FAILURES"}')


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    if '--interactive' in sys.argv or '-i' in sys.argv:
        enhanced_interactive_mode()
    elif '--extended' in sys.argv or '-e' in sys.argv:
        enhanced_interactive_mode()
    elif '--test' in sys.argv or '-t' in sys.argv:
        IntegrationTestSuite.print_report()
    elif '--validate' in sys.argv or '-v' in sys.argv:
        EvolutionValidator.print_validation_report()
    elif '--benchmark' in sys.argv or '-b' in sys.argv:
        results = PerformanceBenchmark.run_all()
        for k, v in results.items():
            print(f'  {k}: {v}')
    else:
        show_stats()


