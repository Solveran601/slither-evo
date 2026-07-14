"""
Obstacle Generator — creates walls and blocks for game maps.
Maps: random_gen, arena, maze, or custom JSON.

Each obstacle is a dict:
  {'type': 'wall', 'x1':..., 'y1':..., 'x2':..., 'y2':..., 'thickness':...}
  {'type': 'block', 'x':..., 'y':..., 'radius':...}
"""

import json
import math
import random
from pathlib import Path


def generate(map_name: str, world_size: int) -> list:
    """Generate obstacles for the given map name."""
    # First try to load as JSON file
    map_path = Path(__file__).parent / 'maps' / f'{map_name}.json'
    if map_path.exists():
        try:
            with open(map_path) as f:
                data = json.load(f)
            return _parse_map_data(data, world_size)
        except (json.JSONDecodeError, IOError):
            pass

    # Built-in generators
    generators = {
        'random_gen': _generate_random,
        'arena': _generate_arena,
        'maze': _generate_maze,
        'empty': lambda ws: [],
    }
    gen = generators.get(map_name, generators['random_gen'])
    return gen(world_size)


def _parse_map_data(data: dict, world_size: int) -> list:
    """Parse a map config dict into obstacle list."""
    obstacles = []

    for w in data.get('walls', []):
        obstacles.append({
            'type': 'wall',
            'x1': w.get('x1', 0),
            'y1': w.get('y1', 0),
            'x2': w.get('x2', world_size),
            'y2': w.get('y2', world_size),
            'thickness': w.get('thickness', 20),
        })

    for b in data.get('blocks', []):
        obstacles.append({
            'type': 'block',
            'x': b.get('x', world_size // 2),
            'y': b.get('y', world_size // 2),
            'radius': b.get('radius', 100),
        })

    if data.get('maze'):
        obstacles.extend(_generate_maze_internal(
            world_size,
            data.get('maze_cells', 8),
            data.get('maze_wall_thickness', 15),
        ))

    if data.get('random_walls', 0) > 0:
        obstacles.extend(_generate_random_walls(
            world_size,
            data['random_walls'],
            data.get('min_wall_length', 500),
            data.get('max_wall_length', 3000),
        ))

    return obstacles


def _generate_random(world_size: int) -> list:
    """Generate random obstacles for variety."""
    obstacles = []
    margin = world_size * 0.1
    inner = world_size - margin * 2

    num_walls = random.randint(8, 15)
    obstacles.extend(_generate_random_walls(
        world_size, num_walls, 600, 2500, margin
    ))

    num_blocks = random.randint(3, 8)
    max_block_attempts = num_blocks * 10
    for _ in range(max_block_attempts):
        if len([o for o in obstacles if o['type'] == 'block']) >= num_blocks:
            break
        bx = margin + random.random() * inner
        by = margin + random.random() * inner
        br = random.randint(60, 180)
        # Check overlap with all obstacles
        overlap = False
        for o in obstacles:
            if o['type'] == 'wall':
                dx, dy = o['x2'] - o['x1'], o['y2'] - o['y1']
                length_sq = dx * dx + dy * dy
                if length_sq < 1:
                    continue
                t = max(0, min(1, ((bx - o['x1']) * dx + (by - o['y1']) * dy) / length_sq))
                px = o['x1'] + t * dx
                py = o['y1'] + t * dy
                d = math.hypot(bx - px, by - py)
                if d < br + o['thickness'] / 2 + 50:
                    overlap = True
                    break
            elif o['type'] == 'block':
                d = math.hypot(bx - o['x'], by - o['y'])
                if d < br + o['radius'] + 100:
                    overlap = True
                    break
        if overlap:
            continue
        obstacles.append({
            'type': 'block',
            'x': bx, 'y': by,
            'radius': br,
        })

    return obstacles


def _generate_arena(world_size: int) -> list:
    """Generate arena-style map with walls around edges and in center."""
    obstacles = []
    m = world_size * 0.05

    obstacles.append({
        'type': 'wall', 'x1': m, 'y1': m,
        'x2': m, 'y2': world_size - m, 'thickness': 30,
    })
    obstacles.append({
        'type': 'wall', 'x1': world_size - m, 'y1': m,
        'x2': world_size - m, 'y2': world_size - m, 'thickness': 30,
    })
    obstacles.append({
        'type': 'wall', 'x1': m, 'y1': m,
        'x2': world_size - m, 'y2': m, 'thickness': 30,
    })
    obstacles.append({
        'type': 'wall', 'x1': m, 'y1': world_size - m,
        'x2': world_size - m, 'y2': world_size - m, 'thickness': 30,
    })

    cx = world_size // 2
    cy = world_size // 2
    obstacles.append({
        'type': 'block', 'x': cx, 'y': cy,
        'radius': world_size * 0.08,
    })

    for angle in range(0, 360, 45):
        a = math.radians(angle)
        dist = world_size * 0.15
        obstacles.append({
            'type': 'block',
            'x': cx + math.cos(a) * dist,
            'y': cy + math.sin(a) * dist,
            'radius': random.randint(40, 80),
        })

    return obstacles


def _generate_maze(world_size: int) -> list:
    """Generate a maze using recursive backtracker."""
    cells = max(6, world_size // 1200)
    wall_thickness = max(10, world_size // 300)
    return _generate_maze_internal(world_size, cells, wall_thickness)


def _generate_maze_internal(world_size: int, cells: int,
                            wall_thickness: int) -> list:
    """Internal maze generator using recursive backtracker."""
    cell_w = world_size / cells
    cell_h = world_size / cells
    obstacles = []

    grid = [[0] * cells for _ in range(cells)]
    # 0 = not visited, bit flags: 1=top, 2=right, 4=bottom, 8=left

    def carve(cx, cy):
        grid[cy][cx] |= 16  # visited
        dirs = [(0, -1, 1, 4), (1, 0, 2, 8), (0, 1, 4, 1), (-1, 0, 8, 2)]
        random.shuffle(dirs)
        for dx, dy, wall_bit, rev_bit in dirs:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cells and 0 <= ny < cells and not (grid[ny][nx] & 16):
                grid[cy][cx] |= wall_bit
                grid[ny][nx] |= rev_bit
                carve(nx, ny)

    carve(0, 0)

    for y in range(cells):
        for x in range(cells):
            px = x * cell_w
            py = y * cell_h
            if not (grid[y][x] & 1):
                obstacles.append({
                    'type': 'wall',
                    'x1': px, 'y1': py,
                    'x2': px + cell_w, 'y2': py,
                    'thickness': wall_thickness,
                })
            if not (grid[y][x] & 2):
                obstacles.append({
                    'type': 'wall',
                    'x1': px + cell_w, 'y1': py,
                    'x2': px + cell_w, 'y2': py + cell_h,
                    'thickness': wall_thickness,
                })
            if not (grid[y][x] & 4):
                obstacles.append({
                    'type': 'wall',
                    'x1': px, 'y1': py + cell_h,
                    'x2': px + cell_w, 'y2': py + cell_h,
                    'thickness': wall_thickness,
                })
            if not (grid[y][x] & 8):
                obstacles.append({
                    'type': 'wall',
                    'x1': px, 'y1': py,
                    'x2': px, 'y2': py + cell_h,
                    'thickness': wall_thickness,
                })

    return obstacles


def _segments_overlap(ox1, oy1, ox2, oy2, thickness, obstacles, min_gap=50):
    """Check if a new wall segment overlaps existing obstacles."""
    for o in obstacles:
        if o['type'] == 'wall':
            # Minimum distance between two segments
            def seg_dist(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
                dx, dy = ax2 - ax1, ay2 - ay1
                ex, ey = bx2 - bx1, by2 - by1
                len_a = dx * dx + dy * dy
                len_b = ex * ex + ey * ey
                if len_a < 1 or len_b < 1:
                    return float('inf')
                # Check closest points on two segments
                r = dx * ex + dy * ey
                s = (dx * (ay1 - by1) + dy * (ax1 - bx1)) * -1
                denom = len_a * len_b - r * r
                if abs(denom) < 1e-6:
                    # Parallel - check distance from point to segment
                    t = max(0, min(1, ((bx1 - ax1) * dx + (by1 - ay1) * dy) / len_a))
                    px = ax1 + t * dx
                    py = ay1 + t * dy
                    return math.hypot(bx1 - px, by1 - py)
                t = max(0, min(1, (r * (by1 - ay1) * -1 + len_b * ((bx1 - ax1) * dy - (by1 - ay1) * dx)) / denom))
                u = max(0, min(1, (dx * (by1 - ay1) - dy * (bx1 - ax1)) / denom))
                if t < 0 or t > 1 or u < 0 or u > 1:
                    # Check endpoints
                    def pt_seg(px, py, ax, ay, ex, ey):
                        el = ex * ex + ey * ey
                        if el < 1:
                            return math.hypot(px - ax, py - ay)
                        t = max(0, min(1, ((px - ax) * ex + (py - ay) * ey) / el))
                        return math.hypot(px - (ax + t * ex), py - (ay + t * ey))
                    return min(
                        pt_seg(bx1, by1, ax1, ay1, dx, dy),
                        pt_seg(bx2, by2, ax1, ay1, dx, dy),
                        pt_seg(ax1, ay1, bx1, by1, ex, ey),
                        pt_seg(ax2, ay2, bx1, by1, ex, ey),
                    )
                px = ax1 + t * dx
                py = ay1 + t * dy
                return math.hypot(bx1 + u * ex - px, by1 + u * ey - py)

            half = (thickness + o['thickness']) / 2 + min_gap
            d = seg_dist(ox1, oy1, ox2, oy2, o['x1'], o['y1'], o['x2'], o['y2'])
            if d < half:
                return True

        elif o['type'] == 'block':
            # Check if segment passes near a block center
            dx, dy = ox2 - ox1, oy2 - oy1
            length = dx * dx + dy * dy
            if length < 1:
                continue
            t = max(0, min(1, ((o['x'] - ox1) * dx + (o['y'] - oy1) * dy) / length))
            px = ox1 + t * dx
            py = oy1 + t * dy
            d = math.hypot(o['x'] - px, o['y'] - py)
            if d < o['radius'] + thickness / 2 + min_gap:
                return True

    return False


def _generate_random_walls(world_size: int, count: int,
                           min_len: int, max_len: int,
                           margin: float = 0) -> list:
    """Generate random wall segments without overlap."""
    if margin == 0:
        margin = world_size * 0.05
    obstacles = []
    inner = world_size - margin * 2
    max_attempts = count * 10

    for _ in range(max_attempts):
        if len(obstacles) >= count:
            break

        length = random.randint(min_len, max_len)
        angle = random.random() * math.pi
        cx = margin + random.random() * inner
        cy = margin + random.random() * inner

        x1 = cx - math.cos(angle) * length / 2
        y1 = cy - math.sin(angle) * length / 2
        x2 = cx + math.cos(angle) * length / 2
        y2 = cy + math.sin(angle) * length / 2

        x1 = max(0, min(world_size, x1))
        y1 = max(0, min(world_size, y1))
        x2 = max(0, min(world_size, x2))
        y2 = max(0, min(world_size, y2))

        thickness = random.randint(10, 25)

        if _segments_overlap(x1, y1, x2, y2, thickness, obstacles):
            continue

        obstacles.append({
            'type': 'wall',
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'thickness': thickness,
        })

    return obstacles
