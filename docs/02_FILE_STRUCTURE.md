# ESTRUCTURA DE ARCHIVOS

```
project/
│
├── main.py                    # Entry point con pygame. Crea QLearning, carga Q-tables,
│                              # instancia Environment, loop de render + tick.
│                              # Maneja clicks de usuario (targets para hunters).
│
├── environment.py             # Motor central. Clase Environment + clase Cell.
│                              # Gestiona grid, agentes, recursos, combate, risk map,
│                              # build kits, respawn, victory check.
│                              # Clase Cell: position, type, agents[], tower, resource_amount
│
├── train.py                   # Entry point de entrenamiento headless.
│                              # Llama Trainer.train_curriculum(CURRICULUM).
│
├── agents/
│   ├── collector.py           # Clase Collector (Equipo A).
│   │                          # Q-learning 10-vars + heurísticas fase-aware.
│   │                          # Acciones: EXPLORE, GO_TO_RESOURCE, RETURN_TO_BASE, FLEE, BUILD_TOWER
│   │
│   ├── guard.py               # Clase Guard (Equipo A).
│   │                          # Q-learning 12-vars + heurísticas autónomas.
│   │                          # Acciones: PATROL, ESCORT, ATTACK, INTERCEPT,
│   │                          #           DEFEND_ZONE, INVESTIGATE, SCOUT
│   │
│   ├── hunter.py              # Clase Hunter (Equipo B).
│   │                          # FSM con 9 estados + genes evolutivos.
│   │                          # Respawn con genes mutados tras 15 ticks.
│   │                          # Sin Q-learning.
│   │
│   └── tower.py               # Clase Tower (Equipo A).
│                              # Estática, automática. Revela fog, reduce riesgo, dispara.
│
├── rl/
│   ├── __init__.py
│   └── q_learning.py          # Clase QLearning tabular.
│                              # Q_table = defaultdict(defaultdict(float))
│                              # Métodos: get_action(), update(), decay_epsilon(),
│                              #          save(), load()
│
├── training/
│   ├── __init__.py
│   ├── curriculum.py          # Lista CURRICULUM con 4 fases (0→4→10→15 hunters).
│   └── trainer.py             # Clase Trainer. train_curriculum(), _train_phase(),
│                              # _save_qtables(). Headless, sin pygame.
│
├── evolution/
│   ├── __init__.py
│   └── genetic_system.py      # Clase Genes (gamma, beta, delta, alpha).
│                              # mutate(), random_spawn_position().
│                              # Usada solo por Hunter.
│
├── pathfinding/
│   ├── __init__.py
│   └── astar.py               # Función find_path(start, goal, grid, cost_fn, known_map).
│                              # Usada por Collector._astar() y Guard._astar().
│                              # El 'grid' es el objeto Environment.
│
├── utils/
│   ├── __init__.py
│   └── constants.py           # Todas las constantes del sistema (≈260 líneas).
│                              # Ver 09_CONSTANTS_REFERENCE.md para detalle.
│
└── data/                      # Directorio auto-creado por Trainer.
    ├── collector_qtable.pkl   # Q-table latest de recolectores
    ├── guard_qtable.pkl       # Q-table latest de guardias
    ├── collector_qtable_phase1.pkl  # Por fase
    └── ...
```

---

## Dependencias entre archivos (imports)

```
main.py
  └── environment.py
  └── rl/q_learning.py
  └── utils/constants.py

environment.py
  └── agents/collector.py
  └── agents/guard.py
  └── agents/hunter.py
  └── agents/tower.py
  └── rl/q_learning.py
  └── evolution/genetic_system.py  (solo random_spawn_position)
  └── utils/constants.py

agents/collector.py
  └── pathfinding/astar.py
  └── utils/constants.py

agents/guard.py
  └── pathfinding/astar.py
  └── utils/constants.py

agents/hunter.py
  └── evolution/genetic_system.py
  └── utils/constants.py  (implícito)

agents/tower.py
  └── utils/constants.py  (implícito)

training/trainer.py
  └── environment.py
  └── rl/q_learning.py
  └── utils/constants.py

training/curriculum.py
  └── utils/constants.py
```

---

## Notas importantes para modificaciones

- **`environment.py` importa constantes de render** (CELL_SIZE, COLOR_*, etc.) aunque no use pygame en headless. No hay problema — son solo tuplas/ints.
- **`astar.py` recibe `grid` = objeto Environment** (no un array). Usa `grid.cells[x][y].type` y `grid.width/height`.
- **Shared data**: todos los agentes del Equipo A comparten el mismo dict `shared_data` por referencia. Mutarlo en un agente lo hace visible para todos los demás.
- **Q-learning compartido**: todos los collectors comparten UNA instancia de QLearning. Ídem guards. Una sola Q-table por tipo de agente.
