"""
Slither Evo Config GUI
======================
Standalone configuration tool — edit all simulator parameters visually.
Run:  python config_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
import ast
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.py')
EVOLUTION_PATH = os.path.join(SCRIPT_DIR, 'evolution.py')

# ── Parsing helpers ─────────────────────────────────────────────────

def _parse_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return f.read()

def _write_config(text):
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write(text)

def _parse_value(source, name, default=None):
    m = re.search(rf'^{name}\s*=\s*(.+?)(?:\s*#.*)?$', source, re.MULTILINE)
    if m:
        try:
            return ast.literal_eval(m.group(1).strip())
        except:
            return m.group(1).strip()
    return default

def _set_value(source, name, value):
    if isinstance(value, str) and not value.startswith(("'", '"')):
        value = repr(value)
    pattern = rf'^({name}\s*=\s*).*$'
    replacement = rf'\g<1>{value}'
    return re.sub(pattern, replacement, source, flags=re.MULTILINE)

def _parse_mode_params(source):
    m = re.search(r'MODE_PARAMS\s*=\s*\{', source)
    if not m:
        return {}, source
    start = m.start()
    depth, i = 0, start
    while i < len(source):
        if source[i] == '{': depth += 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0: break
        i += 1
    block = source[start:i + 1]
    try:
        params = ast.literal_eval(block.split('=', 1)[1].strip())
    except:
        return {}, source
    return params, source[:start] + source[i + 1:]

def _set_mode_params(source, params):
    def fmt_val(v):
        if isinstance(v, bool): return str(v)
        return repr(v)
    lines = ['MODE_PARAMS = {']
    for mode, settings in params.items():
        lines.append(f"    {fmt_val(mode)}: {{")
        for k, v in settings.items():
            lines.append(f"        {fmt_val(k)}: {fmt_val(v)},")
        lines.append('    },')
    lines.append('}')
    return re.sub(
        r'MODE_PARAMS\s*=\s*\{.*?\n\}',
        '\n'.join(lines),
        source, flags=re.DOTALL
    )

# ── Theme ───────────────────────────────────────────────────────────

BG = '#0a0a15'
BG2 = '#111120'
FG = '#ccc'
FG2 = '#888'
ACCENT = '#44aaff'
ACCENT2 = '#4f8'
INPUT_BG = '#1a1a2e'
INPUT_FG = '#ddd'
BORDER = '#2a2a3a'
BTN_BG = '#2a2a3a'
BTN_FG = '#ccc'

FONT = ('Segoe UI', 10)
FONT_SM = ('Segoe UI', 8)
FONT_BOLD = ('Segoe UI', 10, 'bold')
FONT_TAB = ('Segoe UI', 11)

# ── GUI ────────────────────────────────────────────────────────────

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind('<Enter>', self._show)
        widget.bind('<Leave>', self._hide)

    def _show(self, _):
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + 18
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f'+{x}+{y}')
        tk.Label(self.tip, text=self.text, bg='#222', fg=FG,
                 font=FONT_SM, padx=10, pady=5,
                 wraplength=280, justify='left').pack()

    def _hide(self, _):
        if self.tip: self.tip.destroy(); self.tip = None


class ConfigGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Slither Evo — Configuration')
        self.root.geometry('960x760')
        self.root.configure(bg=BG)
        try:
            self.root.iconbitmap(default=os.path.join(SCRIPT_DIR, 'icon.ico'))
        except:
            pass

        self.entries = {}
        self.color_btns = {}
        self.mode_entries = {}
        self.fitness_entries = {}
        self.team_name_entries = []
        self.team_color_btns = []
        self.original_text = _parse_config()
        self.evo_text = open(EVOLUTION_PATH, 'r', encoding='utf-8').read()

        self._build_ui()
        self._load_values()

    def _make_entry(self, parent, w=10):
        e = tk.Entry(parent, width=w, font=FONT, bg=INPUT_BG, fg=INPUT_FG,
                     insertbackground=ACCENT, relief='flat', bd=3,
                     highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=ACCENT)
        return e

    def _make_spin(self, parent, f, t, w=10):
        s = tk.Spinbox(parent, from_=f, to=t, width=w, font=FONT,
                       bg=INPUT_BG, fg=INPUT_FG, buttonbackground=BTN_BG,
                       relief='flat', bd=3, highlightthickness=1,
                       highlightbackground=BORDER, highlightcolor=ACCENT)
        return s

    def _row(self, parent, label, key, default='', tooltip='',
             unit='', minv=None, maxv=None, kind='entry'):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill='x', pady=1)
        lbl = tk.Label(f, text=label, width=28, anchor='w',
                       font=FONT, fg=FG, bg=BG)
        lbl.pack(side='left', padx=(10, 4))
        if tooltip:
            ToolTip(lbl, tooltip)
        v = tk.StringVar(value=str(default))
        if kind == 'check':
            w = tk.Checkbutton(f, variable=v, onvalue='True', offvalue='False',
                               bg=BG, fg=FG, selectcolor=BG, activebackground=BG,
                               activeforeground=FG, font=FONT)
            v.set('True' if default else 'False')
        elif minv is not None and maxv is not None:
            w = self._make_spin(f, minv, maxv)
            w.config(textvariable=v)
        else:
            w = self._make_entry(f)
            w.config(textvariable=v)
        w.pack(side='left', padx=2)
        if unit:
            tk.Label(f, text=unit, fg=FG2, font=FONT_SM, bg=BG).pack(side='left')
        self.entries[key] = v
        return v

    def _color_btn(self, parent, label, key, default='#ff4455', tooltip=''):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill='x', pady=1)
        lbl = tk.Label(f, text=label, width=28, anchor='w',
                       font=FONT, fg=FG, bg=BG)
        lbl.pack(side='left', padx=(10, 4))
        if tooltip:
            ToolTip(lbl, tooltip)
        v = tk.StringVar(value=default)
        btn = tk.Label(f, text='  ██████  ', bg=default, fg='#fff',
                       font=FONT, cursor='hand2',
                       relief='solid', bd=1, padx=6, pady=2)
        btn.pack(side='left', padx=2)

        def pick():
            c = colorchooser.askcolor(btn.cget('bg'), title=label)
            if c and c[1]:
                v.set(c[1])
                btn.config(bg=c[1])
        btn.bind('<Button-1>', lambda e: pick())
        self.entries[key] = v
        self.color_btns[key] = btn
        return v

    def _section(self, parent, title):
        f = tk.LabelFrame(parent, text=title, bg=BG2, fg=ACCENT,
                          font=FONT_BOLD, padx=10, pady=6,
                          relief='flat', bd=0,
                          highlightthickness=1, highlightbackground=BORDER)
        f.pack(fill='x', padx=6, pady=3)
        return f

    def _build_ui(self):
        # ── Header ──
        hdr = tk.Frame(self.root, bg=BG2, height=50)
        hdr.pack(fill='x')
        tk.Label(hdr, text='⚙  SLITHER EVO  ·  CONFIGURATION', font=('Segoe UI', 14, 'bold'),
                 fg=ACCENT, bg=BG2).pack(side='left', padx=16, pady=10)
        tk.Label(hdr, text=f'config.py  ·  evolution.py', font=FONT_SM,
                 fg=FG2, bg=BG2).pack(side='right', padx=16, pady=10)

        # ── Notebook ──
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', background=BTN_BG, foreground=FG2,
                        padding=[14, 5], font=FONT_TAB)
        style.map('TNotebook.Tab', background=[('selected', BG2)],
                  foreground=[('selected', ACCENT)])
        style.layout('TNotebook.Tab', [
            ('Notebook.tab', {
                'sticky': 'nswe',
                'children': [
                    ('Notebook.padding', {
                        'side': 'top',
                        'children': [
                            ('Notebook.label', {'side': 'top', 'sticky': ''})
                        ]
                    })
                ]
            })
        ])

        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=8, pady=4)

        self._build_world_tab(nb)
        self._build_nn_tab(nb)
        self._build_modes_tab(nb)
        self._build_fitness_tab(nb)
        self._build_teams_tab(nb)

        # ── Bottom bar ──
        bar = tk.Frame(self.root, bg=BG2, height=44)
        bar.pack(fill='x')

        self.status_var = tk.StringVar(value='Ready')
        tk.Label(bar, textvariable=self.status_var, font=FONT_SM,
                 fg=FG2, bg=BG2).pack(side='left', padx=14)

        for txt, cmd, style in [
            ('Reset to Defaults', self._reset, '#a33'),
            ('Reload from File', self._reload, '#555'),
            ('Save & Apply', self._save, '#2a6'),
        ]:
            btn = tk.Button(bar, text=txt, font=FONT_BOLD, fg='#fff',
                            bg=style, activebackground='#555',
                            activeforeground='#fff', relief='flat',
                            padx=14, pady=4, cursor='hand2',
                            command=cmd)
            btn.pack(side='right', padx=4)

    def _build_world_tab(self, nb):
        tab = tk.Frame(nb, bg=BG)
        nb.add(tab, text='🌍 World')

        sf = self._section(tab, 'World')
        self._row(sf, 'World Size', 'WORLD_SIZE', 8000, 'World width/height in pixels', 'px', 1000, 32000)
        self._row(sf, 'Cell Size', 'CELL_SIZE', 500, 'Grid cell size for spatial partitioning', 'px', 50, 2000)

        sf = self._section(tab, 'Worm Physics')
        self._row(sf, 'Segment Radius', 'SEG_RADIUS', 8, 'Base radius of body segments', 'px', 2, 30)
        self._row(sf, 'Head Radius', 'HEAD_RADIUS', 14, 'Base radius of head', 'px', 4, 50)
        self._row(sf, 'Segment Distance', 'SEG_DIST', 14, 'Target distance between segments', 'px', 4, 40)
        self._row(sf, 'Initial Segments', 'INITIAL_SEGMENTS', 15, 'Number of segments at mass=1', '', 3, 100)
        self._row(sf, 'Base Speed', 'BASE_SPEED', 3.5, 'Movement speed (px/frame)', 'px', 0.5, 20)
        self._row(sf, 'Sprint Speed', 'SPRINT_SPEED', 7.0, 'Sprint speed multiplier', 'px', 1, 30)
        self._row(sf, 'Sprint Mass Cost', 'SPRINT_MASS_COST', 0.3, 'Mass lost per frame while sprinting', '', 0, 5)

    def _build_nn_tab(self, nb):
        tab = tk.Frame(nb, bg=BG)
        nb.add(tab, text='🧠 Neural Network')

        sf = self._section(tab, 'Architecture')
        self._row(sf, 'Input Neurons', 'N_INPUT', 44, 'Number of input neurons (8 rays × 5 layers + extras)', '', 4, 200)
        self._row(sf, 'Hidden Layer 1', 'N_HIDDEN1', 30, 'Neurons in first hidden layer', '', 2, 200)
        self._row(sf, 'Hidden Layer 2', 'N_HIDDEN2', 22, 'Neurons in second hidden layer', '', 2, 200)
        self._row(sf, 'Hidden Layer 3', 'N_HIDDEN3', 16, 'Neurons in third hidden layer', '', 2, 200)
        self._row(sf, 'Output Neurons', 'N_OUTPUT', 4, 'Output: turn, boost, direction', '', 2, 20)

        sf = self._section(tab, 'Mutation')
        self._row(sf, 'Mutation Rate', 'MUTATION_RATE', 0.15, 'Probability of mutating each weight (0-1)', '', 0, 1)
        self._row(sf, 'Mutation Amount', 'MUTATION_AMOUNT', 0.25, 'Magnitude of weight mutations (0-2)', '', 0, 2)

    def _build_modes_tab(self, nb):
        tab = tk.Frame(nb, bg=BG)
        nb.add(tab, text='🎮 Game Modes')

        main_f = tk.Frame(tab, bg=BG)
        main_f.pack(fill='both', expand=True)

        for mode in ('team', 'ffa'):
            sf = self._section(main_f, f'Mode: {mode.upper()}')
            self.mode_entries[mode] = {}
            defaults = {
                'team':  {'N_TEAMS': 4, 'WORMS_PER_TEAM': 1, 'MODELS_PER_TEAM': 1, 'FOOD_COUNT': 4000,
                          'ZONE_RADIUS': 1800, 'ZONE_DAMAGE': 0.5},
                'ffa':   {'N_TEAMS': 20, 'WORMS_PER_TEAM': 1, 'MODELS_PER_TEAM': 5, 'FOOD_COUNT': 8000,
                          'ZONE_RADIUS': 0, 'ZONE_DAMAGE': 0},
            }
            d = defaults[mode]
            for key, (label, tip, mn, mx) in {
                'N_TEAMS': ('Number of Teams', 'How many teams/agents in this mode', 1, 50),
                'WORMS_PER_TEAM': ('Worms per Team', 'Number of worms in each team', 1, 50),
                'MODELS_PER_TEAM': ('Models per Team', 'Neural network pool (brains) per team', 1, 50),
                'FOOD_COUNT': ('Food Count', 'Number of food items on the map', 0, 50000),
                'ZONE_RADIUS': ('Zone Radius', 'Territory zone radius (0 = disabled)', 0, 10000),
                'ZONE_DAMAGE': ('Zone Damage', 'Damage/sec in enemy zone', 0, 10),
            }.items():
                v = self._row(sf, label, f'mode_{mode}_{key}', d.get(key, 0), tip, '', mn, mx)
                self.mode_entries[mode][key] = v

            # Obstacles group
            of = tk.Frame(sf, bg=BG)
            of.pack(fill='x', pady=3)
            tk.Label(of, text='Obstacles', width=28, anchor='w', font=FONT,
                     fg=FG, bg=BG).pack(side='left', padx=(10, 4))
            v = self._row(of, '', f'mode_{mode}_OBSTACLES_ENABLED',
                         mode == 'team', 'Enable obstacles (walls, blocks) on the map',
                         kind='check')
            self.mode_entries[mode]['OBSTACLES_ENABLED'] = v
            v = self._row(of, '', f'mode_{mode}_OBSTACLE_MAP',
                         'random_gen',
                         'Preset: random_gen, maze, rings, cross, or custom JSON',
                         kind='entry')
            self.mode_entries[mode]['OBSTACLE_MAP'] = v

    def _build_fitness_tab(self, nb):
        tab = tk.Frame(nb, bg=BG)
        nb.add(tab, text='🏆 Fitness')

        c = tk.Canvas(tab, bg=BG, highlightthickness=0)
        scroll = tk.Scrollbar(tab, orient='vertical', command=c.yview, bg=BG)
        inner = tk.Frame(c, bg=BG)
        inner.bind('<Configure>', lambda e: c.configure(scrollregion=c.bbox('all')))
        c.create_window((0, 0), window=inner, anchor='nw')
        c.configure(yscrollcommand=scroll.set)
        c.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

        def _on_mw(e):
            c.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        c.bind_all('<MouseWheel>', _on_mw, add='+')

        sf = self._section(inner, 'Rewards')
        rewards = [
            ('food_reward', 'Food Reward', 15.0, 'Points per food eaten', 0, 100),
            ('mass_gain_per_unit', 'Mass Gain Reward', 4.0, 'Points per unit mass gained', 0, 50),
            ('kill_reward', 'Kill Reward', 80.0, 'Points per kill', 0, 500),
            ('exploration_per_100px', 'Exploration (per 100px)', 0.5, 'Points per 100px traveled', 0, 20),
            ('zone_aggression_per_10s', 'Zone Aggression (per 10s)', 8.0, 'Points per 10s in enemy zone', 0, 50),
        ]
        for key, label, default, tip, mn, mx in rewards:
            v = self._row(sf, label, f'fit_{key}', default, tip, '', mn, mx)
            self.fitness_entries[key] = v

        sf2 = self._section(inner, 'Bonuses')
        bonuses = [
            ('big_worm_threshold', 'Big Worm (mass)', 30.0, 'Mass threshold for big worm bonus', 0, 500),
            ('big_worm_bonus', 'Big Worm Bonus ×', 1.4, 'Fitness multiplier for big worms', 1, 5),
            ('predator_kill_threshold', 'Predator (kills)', 2, 'Kills needed for predator bonus', 0, 50),
            ('predator_bonus', 'Predator Bonus ×', 1.3, 'Fitness multiplier for predators', 1, 5),
            ('feeder_food_threshold', 'Feeder (food)', 40, 'Food eaten needed for feeder bonus', 0, 500),
            ('feeder_bonus', 'Feeder Bonus ×', 1.2, 'Fitness multiplier for feeders', 1, 5),
            ('veteran_threshold', 'Veteran (seconds)', 90.0, 'Survival time for veteran bonus', 0, 600),
            ('veteran_bonus', 'Veteran Bonus ×', 1.1, 'Fitness multiplier for veterans', 1, 5),
        ]
        for key, label, default, tip, mn, mx in bonuses:
            v = self._row(sf2, label, f'fit_{key}', default, tip, '', mn, mx)
            self.fitness_entries[key] = v

        sf3 = self._section(inner, 'Penalties')
        penalties = [
            ('instant_death_threshold', 'Instant Death (s)', 3.0, 'Death within this time = ×0.01', 0, 30),
            ('instant_death_penalty', 'Instant Death ×', 0.01, 'Multiplier if died too fast', 0, 1),
            ('wall_death_penalty', 'Wall Death ×', 0.05, 'Multiplier if died from wall', 0, 1),
            ('frozen_distance_threshold', 'Frozen (px)', 50, 'Distance threshold for frozen penalty', 0, 1000),
            ('frozen_time_threshold', 'Frozen (s)', 20.0, 'Time threshold for frozen detection', 0, 120),
            ('frozen_penalty', 'Frozen ×', 0.02, 'Multiplier if frozen in place', 0, 1),
            ('never_ate_penalty', 'Never Ate ×', 0.10, 'Multiplier if never ate food', 0, 1),
            ('starvation_divisor', 'Starvation (s/food)', 20.0, 'Expected seconds between eating', 1, 200),
            ('starvation_penalty_per_unit', 'Starvation Penalty', 5.0, 'Points per missing food unit', 0, 100),
            ('mass_waste_penalty', 'Mass Waste Penalty', 4.0, 'Points per unit mass lost from peak', 0, 50),
            ('grace_period', 'Grace Period (s)', 10.0, 'Seconds before starvation penalties start', 0, 60),
            ('grace_penalty_per_sec', 'Grace Penalty (per s)', 3.0, 'Points per second during grace period', 0, 50),
        ]
        for key, label, default, tip, mn, mx in penalties:
            v = self._row(sf3, label, f'fit_{key}', default, tip, '', mn, mx)
            self.fitness_entries[key] = v

    def _build_teams_tab(self, nb):
        tab = tk.Frame(nb, bg=BG)
        nb.add(tab, text='👥 Teams')

        sf = self._section(tab, 'Base Names & Colors (first 15)')
        tk.Label(sf, text='Extra names cycle automatically if more teams are needed.',
                 font=FONT_SM, fg=FG2, bg=BG2).pack(anchor='w', padx=8, pady=(0, 4))

        cv = tk.Canvas(sf, bg=BG2, highlightthickness=0, height=360)
        scroll = tk.Scrollbar(sf, orient='vertical', command=cv.yview, bg=BG)
        inner = tk.Frame(cv, bg=BG2)
        inner.bind('<Configure>', lambda e: cv.configure(scrollregion=cv.bbox('all')))
        cv.create_window((0, 0), window=inner, anchor='nw')
        cv.configure(yscrollcommand=scroll.set)
        cv.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

        def _on_mw(e):
            cv.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        cv.bind_all('<MouseWheel>', _on_mw, add='+')

        names = ['Alpha', 'Shadow', 'Viper', 'Cobra', 'Storm',
                 'Blaze', 'Frost', 'Thorn', 'Venom', 'Ghost',
                 'Raven', 'Titan', 'Neon', 'Pixel', 'Flux']
        colors = ['#ff4455', '#ff8822', '#ffdd33', '#44ee55', '#00eebb',
                  '#44aaff', '#5577ff', '#bb55ff', '#ff44aa', '#ff6677',
                  '#99ff44', '#44ffaa', '#22ddff', '#cc44ff', '#ff88aa']

        self.team_name_entries = []
        self.team_color_btns = []
        for i in range(15):
            f = tk.Frame(inner, bg=BG2)
            f.pack(fill='x', pady=1)
            tk.Label(f, text=f'{i+1}.', width=3, anchor='e',
                     font=FONT_SM, fg=FG2, bg=BG2).pack(side='left')
            e = self._make_entry(f, 16)
            e.insert(0, names[i])
            e.pack(side='left', padx=2)
            self.team_name_entries.append(e)

            cv_var = tk.StringVar(value=colors[i])
            btn = tk.Label(f, text='  ██████  ', bg=colors[i], fg='#fff',
                           font=FONT, cursor='hand2',
                           relief='solid', bd=1, padx=6, pady=2)
            btn.pack(side='left', padx=2)

            def make_pick(v, b):
                def pick():
                    c = colorchooser.askcolor(b.cget('bg'), title='Pick Color')
                    if c and c[1]:
                        v.set(c[1])
                        b.config(bg=c[1])
                return pick
            btn.bind('<Button-1>', lambda e, v=cv_var, b=btn: make_pick(v, b)())
            self.team_color_btns.append((cv_var, btn))

        sf2 = self._section(tab, 'Extra Names')
        tk.Label(sf2, text='Comma-separated. Cycles if more teams than base names.',
                 font=FONT_SM, fg=FG2, bg=BG2).pack(anchor='w', padx=8, pady=(0, 4))
        e = self._make_entry(sf2, 60)
        e.insert(0, 'Fang, Claw, Spike, Horn, Scale, Wing, Maw, Sting, Bolt, Fuse, Echo, Nova, Zero, Byte, Dash, Pulse, Drift, Glide, Surge, Crest, Orbit, Prism, Shard, Gleam, Haze, Jolt, Lynx, Myth, Nexus, Onyx')
        e.pack(fill='x', padx=8, pady=2)
        self.extra_names_entry = e

    def _load_values(self):
        source = self.original_text
        for key in ('WORLD_SIZE', 'CELL_SIZE', 'SEG_RADIUS', 'HEAD_RADIUS',
                     'SEG_DIST', 'INITIAL_SEGMENTS', 'BASE_SPEED', 'SPRINT_SPEED',
                     'SPRINT_MASS_COST', 'N_INPUT', 'N_HIDDEN1', 'N_HIDDEN2',
                     'N_HIDDEN3', 'N_OUTPUT', 'MUTATION_RATE', 'MUTATION_AMOUNT'):
            val = _parse_value(source, key)
            if val is not None and key in self.entries:
                self.entries[key].set(str(val))

        mode_params, _ = _parse_mode_params(source)
        for mode in ('team', 'ffa'):
            mp = mode_params.get(mode, {})
            for key in ('N_TEAMS', 'WORMS_PER_TEAM', 'MODELS_PER_TEAM', 'FOOD_COUNT',
                        'ZONE_RADIUS', 'ZONE_DAMAGE', 'OBSTACLES_ENABLED', 'OBSTACLE_MAP'):
                ek = f'mode_{mode}_{key}'
                if ek in self.entries:
                    val = mp.get(key)
                    if val is not None:
                        self.entries[ek].set(str(val))

        extra = _parse_value(source, '_EXTRA_NAMES')
        if extra:
            self.extra_names_entry.delete(0, 'end')
            self.extra_names_entry.insert(0, ', '.join(extra))

        for key in self.fitness_entries:
            m = re.search(rf'self\.{key}\s*=\s*([\d.]+)', self.evo_text)
            if m:
                self.fitness_entries[key].set(m.group(1))

    def _save(self):
        source = _parse_config()

        for key in ('WORLD_SIZE', 'CELL_SIZE', 'SEG_RADIUS', 'HEAD_RADIUS',
                     'SEG_DIST', 'INITIAL_SEGMENTS', 'BASE_SPEED', 'SPRINT_SPEED',
                     'SPRINT_MASS_COST', 'N_INPUT', 'N_HIDDEN1', 'N_HIDDEN2',
                     'N_HIDDEN3', 'N_OUTPUT', 'MUTATION_RATE', 'MUTATION_AMOUNT'):
            if key in self.entries:
                val = self.entries[key].get().strip()
                try: parsed = ast.literal_eval(val)
                except: parsed = val
                source = _set_value(source, key, parsed)

        mode_params, _ = _parse_mode_params(source)
        for mode in ('team', 'ffa'):
            if mode not in mode_params:
                mode_params[mode] = {}
            for key in ('N_TEAMS', 'WORMS_PER_TEAM', 'MODELS_PER_TEAM', 'FOOD_COUNT',
                        'ZONE_RADIUS', 'ZONE_DAMAGE', 'OBSTACLES_ENABLED', 'OBSTACLE_MAP'):
                ek = f'mode_{mode}_{key}'
                if ek in self.entries:
                    val = self.entries[ek].get().strip()
                    try: parsed = ast.literal_eval(val)
                    except: parsed = val
                    mode_params[mode][key] = parsed
        source = _set_mode_params(source, mode_params)

        extra_str = self.extra_names_entry.get().strip()
        extra_list = [n.strip() for n in extra_str.split(',') if n.strip()]
        def _replace_extra(m):
            return '_EXTRA_NAMES = [\n    ' + ',\n    '.join(repr(n) for n in extra_list) + ',\n]'
        source = re.sub(
            r'^_EXTRA_NAMES\s*=\s*\[.*?^\]',
            _replace_extra,
            source, flags=re.MULTILINE | re.DOTALL
        )

        try:
            _write_config(source)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to write config.py:\n{e}')
            return

        evo_text = open(EVOLUTION_PATH, 'r', encoding='utf-8').read()
        for key in self.fitness_entries:
            val = self.fitness_entries[key].get().strip()
            try: parsed = ast.literal_eval(val)
            except: parsed = val
            evo_text = re.sub(
                rf'(self\.{key}\s*=\s*)[\d.]+',
                rf'\g<1>{parsed}',
                evo_text
            )
        try:
            with open(EVOLUTION_PATH, 'w', encoding='utf-8') as f:
                f.write(evo_text)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to write evolution.py:\n{e}')
            return

        # Auto-restart server via /reset
        try:
            import urllib.request
            req = urllib.request.Request('http://localhost:8765/reset', method='POST', data=b'{}')
            urllib.request.urlopen(req, timeout=10)
            self.status_var.set('✓ Saved & applied! Server restarted.')
            messagebox.showinfo('Saved & Applied', 'Configuration saved and server reset successfully!')
        except Exception:
            self.status_var.set('✓ Saved! Restart the server manually (python server.py)')
            messagebox.showinfo('Saved', 'Configuration saved!\n\nStart or restart the server for changes to take effect:\n  python server.py')

    def _reload(self):
        self.original_text = _parse_config()
        self.evo_text = open(EVOLUTION_PATH, 'r', encoding='utf-8').read()
        self._load_values()
        self.status_var.set('Reloaded from file.')

    def _reset(self):
        if not messagebox.askyesno('Reset', 'Reset all values to defaults?'):
            return
        self._load_defaults()
        self.status_var.set('Defaults loaded (not saved yet).')

    def _load_defaults(self):
        defaults = {
            'WORLD_SIZE': 8000, 'CELL_SIZE': 500, 'SEG_RADIUS': 8, 'HEAD_RADIUS': 14,
            'SEG_DIST': 14, 'INITIAL_SEGMENTS': 15, 'BASE_SPEED': 3.5, 'SPRINT_SPEED': 7.0,
            'SPRINT_MASS_COST': 0.3, 'N_INPUT': 44, 'N_HIDDEN1': 30, 'N_HIDDEN2': 22,
            'N_HIDDEN3': 16, 'N_OUTPUT': 4, 'MUTATION_RATE': 0.15, 'MUTATION_AMOUNT': 0.25,
        }
        for key, val in defaults.items():
            if key in self.entries:
                self.entries[key].set(str(val))
        mode_defaults = {
            'team': {'WORMS_PER_TEAM': 1, 'MODELS_PER_TEAM': 1, 'FOOD_COUNT': 4000,
                     'ZONE_RADIUS': 1800, 'ZONE_DAMAGE': 0.5, 'OBSTACLES_ENABLED': True,
                     'OBSTACLE_MAP': 'random_gen'},
            'ffa': {'WORMS_PER_TEAM': 1, 'MODELS_PER_TEAM': 5, 'FOOD_COUNT': 8000,
                    'ZONE_RADIUS': 0, 'ZONE_DAMAGE': 0, 'OBSTACLES_ENABLED': True,
                    'OBSTACLE_MAP': 'random_gen'},
        }
        for mode, params in mode_defaults.items():
            for key, val in params.items():
                ek = f'mode_{mode}_{key}'
                if ek in self.entries:
                    self.entries[ek].set(str(val))
        fit_defaults = {
            'food_reward': 15.0, 'mass_gain_per_unit': 4.0, 'kill_reward': 80.0,
            'exploration_per_100px': 0.5, 'zone_aggression_per_10s': 8.0,
            'big_worm_threshold': 30.0, 'big_worm_bonus': 1.4,
            'predator_kill_threshold': 2, 'predator_bonus': 1.3,
            'feeder_food_threshold': 40, 'feeder_bonus': 1.2,
            'veteran_threshold': 90.0, 'veteran_bonus': 1.1,
            'instant_death_threshold': 3.0, 'instant_death_penalty': 0.01,
            'wall_death_penalty': 0.05, 'frozen_distance_threshold': 50,
            'frozen_time_threshold': 20.0, 'frozen_penalty': 0.02,
            'never_ate_penalty': 0.10, 'starvation_divisor': 20.0,
            'starvation_penalty_per_unit': 5.0, 'mass_waste_penalty': 4.0,
            'grace_period': 10.0, 'grace_penalty_per_sec': 3.0,
        }
        for key, val in fit_defaults.items():
            if key in self.fitness_entries:
                self.fitness_entries[key].set(str(val))

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    ConfigGUI().run()
