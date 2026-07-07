"""
Slither Evo v2 — HTTP server
Per-team model pools, epoch tracking, team endpoints.
"""

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from pathlib import Path

from config import *
from evolution import EvolutionEngine

PORT = 8765
STATIC_DIR = Path(__file__).parent

engine = EvolutionEngine()


class Handler(BaseHTTPRequestHandler):

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_file(self, path):
        try:
            filepath = STATIC_DIR / path
            if not filepath.exists() or not str(filepath).startswith(str(STATIC_DIR)):
                self.send_error(404)
                return
            ext = filepath.suffix.lower()
            mime = {
                '.html': 'text/html', '.js': 'application/javascript',
                '.css': 'text/css', '.json': 'application/json',
                '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml',
            }.get(ext, 'application/octet-stream')
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            with open(filepath, 'rb') as f:
                self.wfile.write(f.read())
        except (IOError, OSError):
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/':
            self._send_file('index.html')

        elif path == '/config':
            self._send_json({
                'worldSize': WORLD_SIZE,
                'nTeams': N_TEAMS,
                'wormsPerTeam': WORMS_PER_TEAM,
                'modelsPerTeam': MODELS_PER_TEAM,
                'nWorms': N_WORMS,
                'foodCount': FOOD_COUNT,
                'baseSpeed': BASE_SPEED,
                'sprintSpeed': SPRINT_SPEED,
                'sprintMassCost': SPRINT_MASS_COST,
                'segRadius': SEG_RADIUS,
                'headRadius': HEAD_RADIUS,
                'segmentDist': SEG_DIST,
                'initialSegments': INITIAL_SEGMENTS,
                'teamNames': TEAM_NAMES,
                'teamColors': TEAM_COLORS,
                'zoneRadius': ZONE_RADIUS,
                'zoneDamage': ZONE_DAMAGE,
                'nInput': N_INPUT,
                'nHidden1': N_HIDDEN1,
                'nHidden2': N_HIDDEN2,
                'nHidden3': N_HIDDEN3,
                'nOutput': N_OUTPUT,
                'weightCount': WEIGHT_COUNT,
                'gameMode': GAME_MODE,
            })

        elif path == '/zones':
            self._send_json(engine.zone_manager.zones)

        elif path == '/weights':
            # Backward compat: return best model per team
            self._send_json(engine.weight_manager.get_all_weights())

        elif path == '/leaderboard':
            self._send_json(engine.get_leaderboard())

        elif path == '/stats':
            stats = engine.get_stats()
            self._send_json(stats)

        elif path == '/fitness_history':
            import json
            hist = []
            try:
                with open('stats_history.json', 'r') as sf:
                    hist = json.load(sf)
            except:
                pass
            if isinstance(hist, list):
                if len(hist) > 500:
                    hist = hist[-500:]
            self._send_json({'history': hist})

        elif path == '/history':
            self._send_json(engine.stats_log[-200:])

        elif path == '/hof':
            # Hall of Fame
            hof_path = STATIC_DIR / 'weights' / '_hall_of_fame' / 'best.json'
            if hof_path.exists():
                with open(hof_path) as f:
                    self._send_json(json.load(f))
            else:
                self._send_json({'error': 'no hall of fame yet'}, 404)

        elif path == '/teams':
            # Return all team info: epochs, pool stats
            infos = engine.weight_manager.get_all_pools_info()
            result = {}
            for t, info in infos.items():
                result[str(t)] = {
                    'team': info.get('team'),
                    'team_name': info.get('team_name', TEAM_NAMES[t]),
                    'team_color': TEAM_COLORS[t],
                    'epoch': info.get('epoch', 0),
                    'best_fitness': info.get('best_fitness', 0.0),
                    'avg_fitness': info.get('avg_fitness', 0.0),
                    'diversity': info.get('diversity', 0.0),
                    'n_models': info.get('n_models', MODELS_PER_TEAM),
                }
            self._send_json(result)

        elif path.startswith('/team/'):
            # /team/<id> or /team/<name>
            team_identifier = path[6:]
            try:
                team_id = int(team_identifier)
            except ValueError:
                name = team_identifier.lower()
                team_id = next(
                    (i for i, n in enumerate(TEAM_NAMES) if n.lower() == name),
                    -1
                )
            if 0 <= team_id < N_TEAMS:
                detail = engine.get_team_stats(team_id)
                if detail and 'error' not in detail:
                    self._send_json(detail)
                else:
                    self._send_json({'error': 'team not found'}, 404)
            else:
                self._send_json({'error': f'invalid team: {team_identifier}'}, 404)

        elif path.startswith('/weights/'):
            # /weights/TeamName/model_NN.json
            fname = path.lstrip('/')
            fpath = STATIC_DIR / fname
            if fpath.exists() and 'weights' in fpath.parts:
                self._send_file(fname)
            else:
                self.send_error(404)

        else:
            self._send_file(path.lstrip('/'))

    def do_POST(self):
        path = urlparse(self.path).path

        if path == '/stats':
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self._send_json({'error': 'empty body'}, 400)
                return
            try:
                data = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                self._send_json({'error': 'invalid json'}, 400)
                return

            try:
                alive = data.get('alive', [])
                dead_teams = data.get('deadTeams', [])
                evolutions = engine.evolve(alive, dead_teams)

                response = {
                    'evolutions': evolutions,
                    'generation': engine.generation,
                    'ranks': engine.get_leaderboard(),
                    'teamEpochs': engine.weight_manager.get_team_epochs(),
                }

                pool_info = {}
                for t in range(N_TEAMS):
                    pool = engine.weight_manager.get_pool(t)
                    if pool:
                        pool_info[str(t)] = {
                            'epoch': pool.epoch,
                            'best_fitness': pool.get_best_fitness(),
                            'avg_fitness': pool.get_avg_fitness(),
                        }
                response['poolInfo'] = pool_info

                self._send_json(response)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        elif path == '/mode':
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self._send_json({'error': 'empty body'}, 400)
                return
            try:
                data = json.loads(self.rfile.read(length))
                mode = data.get('mode', '')
                if mode not in ('ffa', 'team'):
                    self._send_json({'error': f'invalid mode: {mode}'}, 400)
                    return
                global GAME_MODE
                GAME_MODE = mode
                print(f"  [mode] switched to {mode}")
                if mode == 'ffa':
                    # FFA: clear zones
                    engine.zone_manager.zones = {}
                else:
                    engine.zone_manager.generate()
                self._send_json({'gameMode': GAME_MODE})
            except Exception as e:
                self._send_json({'error': str(e)}, 400)

        else:
            self._send_json({'error': 'not found'}, 404)

    def log_message(self, fmt, *args):
        if '200' in str(args[1]) or '404' in str(args[1]):
            return
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")


def auto_save():
    """Auto-save all pools every 10 generations."""
    try:
        from pathlib import Path
        weights_dir = Path(__file__).parent / 'weights'
        weights_dir.mkdir(exist_ok=True)
        best_fit = -1e9
        best_data = None
        for t in range(N_TEAMS):
            pool = engine.weight_manager.get_pool(t)
            if pool:
                pool.save(weights_dir)
                bf = pool.get_best_fitness()
                if bf > best_fit:
                    best_fit = bf
                    bm = pool.get_best_model()
                    if bm:
                        best_data = {'team': TEAM_NAMES[t], 'team_id': t,
                                     'epoch': pool.epoch, 'fitness': bf,
                                     'weights': list(bm.weights)}
        # Hall of Fame
        if best_data:
            hof_dir = weights_dir / '_hall_of_fame'
            hof_dir.mkdir(exist_ok=True)
            import json
            with open(hof_dir / 'best.json', 'w') as f:
                json.dump(best_data, f)
        print(f"  [auto-save] gen {engine.generation}")
    except Exception as e:
        print(f"  [auto-save error] {e}")

def main():
    print(f"Slither Evo v2 — http://0.0.0.0:{PORT}")
    print(f"   Local:  http://127.0.0.1:{PORT}")
    print(f"   Mode:   {GAME_MODE}  (POST /mode to change)")
    print(f"   Teams:  {N_TEAMS}, Worms: {N_WORMS}, Models/team: {MODELS_PER_TEAM}")
    print(f"   Gen:    {engine.generation}, Best fitness: {engine.best_fitness_ever:.1f}")
    print(f"   NN:     {N_INPUT}>{N_HIDDEN1}>{N_HIDDEN2}>{N_HIDDEN3}>{N_OUTPUT} ({WEIGHT_COUNT} weights)")
    print(f"   Ctrl+C to stop")

    server = HTTPServer(('', PORT), Handler)
    last_save_gen = engine.generation
    import time
    try:
        while True:
            server.handle_request()
            if engine.generation - last_save_gen >= 10:
                auto_save()
                last_save_gen = engine.generation
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == '__main__':
    main()
