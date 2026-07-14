# Slither Evo v2

AI evolution simulator for slither-style worms with a browser UI and a Python training server.

## What it does

- Evolves neural-network controlled worms in real time.
- Supports Team mode and FFA mode.
- Preserves the best models across runs.
- Tracks team fitness, diversity, epochs, and training health.
- Uses only the Python standard library.

## Run

```bash
python server.py
```

Open `http://127.0.0.1:8765` in your browser.

## Training upgrades in this version

- Adaptive mutation and crossover settings.
- Novelty search to keep behaviors diverse.
- Real model migration between teams.
- Stable save/load so model age and history survive restarts.

## Neural network

Each worm uses a feed-forward network:

```text
Input (44) -> Hidden1 (30) -> Hidden2 (22) -> Hidden3 (16) -> Output (4)
```

- 8 rays x 5 perception layers, plus 4 extra state features.
- Output channels cover turn, boost, and two extra behavior heads used by the browser and analytics.

## API

- `GET /config`
- `GET /weights`
- `GET /leaderboard`
- `GET /stats`
- `GET /fitness_history`
- `GET /hof`
- `GET /teams`
- `GET /team/<name>`
- `GET /zones`
- `GET /history`
- `POST /stats`
- `POST /mode`

## Project structure

- `server.py` - HTTP server and API
- `config.py` - simulation settings
- `evolution.py` - evolution engine and training logic
- `index.html` - browser frontend
- `obstacles.py` - obstacle generation
- `config_gui.py` - config editor
- `weights/` - saved models

## Notes

- Existing weight files remain compatible.
- The engine now keeps model metadata more accurately when saving.
- The browser still works with the same local server URL.
