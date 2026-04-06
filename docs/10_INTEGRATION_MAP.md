# MAPA DE INTEGRACIÓN — QUIÉN LLAMA A QUIÉN

## Flujo de datos principal (por tick)

```
environment.tick()
│
├─ [Paso 1] agent.perceive(env, tick)         → actualiza shared_data (known_map, discovered_resources,
│  ─ collector.perceive()                         last_seen_enemies)
│  ─ guard.perceive()
│
├─ [Paso 3] hunter.communicate(alive_hunters)  → comparte local_memory entre hunters
│
├─ [Paso 4] DECISIONES
│  ├─ collector.decide(env, tick, alive_collectors)
│  │    ├── .perceive()                         → visible_enemies, _, visible_allies
│  │    ├── .build_state(enemies, allies)       → state tuple (10 vars)
│  │    ├── ._get_game_phase()                  → "EARLY"|"MID"|"LATE"
│  │    │   └── shared_data['explored_count']
│  │    ├── .calculate_heuristic_biases(state, phase, ctx)  → {action: float}
│  │    ├── collector_ql.update(prev_s, prev_a, reward, s)  → actualiza Q_table
│  │    ├── collector_ql.get_action(state, biases)          → action string
│  │    └── .execute_ACTION()                   → A* path
│  │        └── find_path(pos, goal, env, cost_fn, known_map)
│  │
│  ├─ guard.decide(env, tick, all_collectors, alive_guards)
│  │    ├── .perceive()                         → enemies, allies, collectors
│  │    ├── .build_state(enemies, allies, collectors)  → state tuple (12 vars)
│  │    ├── .calculate_heuristic_biases(state, phase, ctx)
│  │    ├── guard_ql.update(prev_s, prev_a, 0.0, s)
│  │    ├── guard_ql.get_action(state, biases)
│  │    └── .execute_ACTION()                   → A* path o acción de combate
│  │
│  └─ hunter.decide(env, all_agents, tick)
│       ├── .perceive() → actualiza local_memory
│       ├── .calculate_risk(), .calculate_opportunity()
│       ├── .evaluate_actions()                 → gene-weighted FSM
│       └── .calculate_movement()               → next_position
│
├─ [Paso 5] MOVIMIENTO — mueve todos según next_position
│
├─ [Paso 6] RECOLECCIÓN
│  └─ collector.collect_resource(cell)
│       env → collector.receive_reward('collect')
│               └── collector_ql.update(prev_s, prev_a, +5, prev_s)
│
├─ [Paso 7] DEPÓSITO
│  └─ collector.deposit_resources()
│       env → collector.receive_reward('deliver_resources')
│           → env._check_build_kits(alive_collectors)
│               └─ collector.receive_build_kit()  → has_build_kit = True
│
├─ [Paso 8] CONSTRUCCIÓN
│  └─ if c.wants_to_build and c.build_target == c.position:
│       Tower(pos) → cells[x][y].tower = tower; env.towers.append(t)
│       env → collector.receive_reward('build_tower')
│
├─ [Paso 9] COMBATE
│  └─ env._collect_all_attacks()
│       ├─ hunters atacan collectors/guards  → (attacker, target, dmg)
│       ├─ guards atacan hunters             → (attacker, target, dmg)
│       └─ tower.update(env, known_map, ...)  → [(tower, target)]
│  └─ env._resolve_combat()
│       └─ env._kill_agent(target)
│           ├─ collector: receive_reward('lose_resources') + receive_reward('die')
│           │             notifica guards cercanos: receive_reward('collector_dies_nearby')
│           ├─ guard:     receive_reward('die')
│           └─ hunter:    hunter.die() → respawn_timer=15
│
├─ [Paso 10] RESPAWN
│  └─ hunter.try_respawn() → genes mutados, spawn en borde
│
├─ [Paso 11] RISK MAP
│  └─ env._update_risk_map() → decay + difusión + enemies + towers
│
├─ [Paso 12] explored_count
│  └─ shared_data['explored_count'] = sum(known_map[x][y]['explored'])
│
├─ [Paso 13] RL DECAY
│  └─ collector_ql.decay_epsilon()
│  └─ guard_ql.decay_epsilon()
│
└─ [Paso 14] VICTORIA
   └─ _check_win_conditions()
```

---

## Shared Data — Flujo de referencias

```
Environment.__init__()
├── self.known_map            ─────────────────────────────────┐
├── self.discovered_resources ─────────────────────────────┐   │  (referencias)
├── self.last_seen_enemies    ─────────────────────────┐   │   │
├── self.risk_map             ─────────────────────┐   │   │   │
│                                                  │   │   │   │
shared_data = {                                    │   │   │   │
    'known_map': ──────────────────────────────────┼───┼───┼───┘
    'discovered_resources': ───────────────────────┼───┼───┘
    'last_seen_enemies': ──────────────────────────┼───┘
    'risk_map': ───────────────────────────────────┘
    'explored_count': int (actualizado cada tick)
}

Collector.__init__(shared_data)          Guard.__init__(shared_data)
├── self.shared_data = shared_data       ├── self.shared_data = shared_data
├── self.known_map = shared_data[...]    ├── self.known_map = shared_data[...]
├── self.discovered_resources = ...      ├── self.discovered_resources = ...
├── self.last_seen_enemies = ...         ├── self.last_seen_enemies = ...
└── self.risk_map = ...                  └── self.risk_map = ...

# Modificar known_map desde un Collector es visible para todos los Guards y viceversa.
```

---

## A* — Interfaz

```python
# pathfinding/astar.py
find_path(start, goal, grid, cost_function, known_map) -> list[(x,y)]

# 'grid' = objeto Environment
# Usa: grid.cells[x][y].type para saber si es obstáculo
#      grid.width, grid.height para límites
#      known_map para fog of war (solo mueve por celdas exploradas o límite)
# cost_function(current_pos, neighbor_pos) -> float
```

---

## Q-learning — Flujo de objetos

```
Environment                           Trainer
├── collector_ql: QLearning  ◄────────── self.collector_ql
│   (shared entre collectors)
├── guard_ql: QLearning      ◄────────── self.guard_ql
    (shared entre guards)

# En entrenamiento: el Trainer crea las instancias y las inyecta:
env = Environment(collector_ql=trainer.collector_ql, ...)

# En juego: main.py crea instancias, carga .pkl, las inyecta:
collector_ql.load("data/collector_qtable.pkl")
env = Environment(collector_ql=collector_ql, ...)
```

---

## Dónde modificar cada comportamiento

### Si quieres que el recolector sea más agresivo explorando
→ `agents/collector.py:calculate_heuristic_biases()`
→ Subir `HEUR_EXPLORE_*` en `utils/constants.py`

### Si quieres que los guardias escorten menos
→ `agents/guard.py:calculate_heuristic_biases()`
→ Bajar `HEUR_ESCORT_VULNERABLE`, subir `HEUR_SCOUT_*`
→ Subir `REWARD_REDUNDANT_ESCORT` (penaliza más la redundancia)

### Si quieres que los hunters sean más difíciles
→ `utils/constants.py`: `HUNTER_SPEED`, `HUNTER_HP`, reducir `HUNTER_RESPAWN_TICKS`

### Si quieres agregar una nueva acción al guardia
```
1. agents/guard.py → ACTIONS list: añadir 'NEW_ACTION'
2. agents/guard.py → execute_new_action() method
3. agents/guard.py → decide(): añadir elif action == 'NEW_ACTION'
4. agents/guard.py → calculate_heuristic_biases(): añadir bias para 'NEW_ACTION'
5. environment.py  → _GUARD_ACTIONS: añadir 'NEW_ACTION'
6. (opcional) utils/constants.py: añadir HEUR_NEW_ACTION_* y REWARD_NEW_ACTION
```

### Si quieres cambiar la dificultad del entrenamiento
→ `utils/constants.py`: `TRAINING_EPISODES_PHASE_*`, `TRAINING_MAX_TICKS_PER_EPISODE`
→ `training/curriculum.py`: cambiar `num_hunters` por fase

### Si el render del juego no muestra algo bien
→ `main.py:render()` — función standalone (la que se usa)
→ `environment.py:Environment.render()` — método alternativo (no usado en main.py)

---

## Herencia de datos entre sesiones

```
Entrenamiento (train.py):
  collector_ql, guard_ql → aprenden → se guardan en data/*.pkl

Juego (main.py):
  se cargan data/*.pkl → collector_ql, guard_ql ya conocen buenas acciones
  siguen aprendiendo con epsilon bajo (0.08)
```

Los agentes no "recuerdan" episodios pasados como entidades individuales — la memoria está en la Q-table compartida, no en los objetos Collector/Guard (que se destruyen entre episodios).
