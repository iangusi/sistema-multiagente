# AGENTS/COLLECTOR.PY

## Datos del agente

```python
HP = 2, speed = 1, vision_range = 3
carrying_capacity = 10
COLLECT_RATE = 5  # máx por tick
No ataca (attack_range = 0)
```

## Acciones disponibles

```python
ACTIONS = ['EXPLORE', 'GO_TO_RESOURCE', 'RETURN_TO_BASE', 'FLEE', 'BUILD_TOWER']
```

---

## Estado RL — 10 variables

```python
state = (
    enemy_near,       # 0/1 — enemigo en vision_range
    carrying,         # 0/1 — lleva algún recurso
    is_full,          # 0/1 — carrying >= capacity
    risk_level,       # 0/1/2 — LOW(<0.3)/MED/HIGH(>0.7) del risk_map actual
    guard_near,       # 0/1 — Guard visible en vision_range
    tower_near,       # 0/1 — Tower visible en vision_range
    has_build_kit,    # 0/1
    resources_known,  # 0/1 — discovered_resources no vacío
    near_base,        # 0/1 — distancia Manhattan <= 5 a BASE_POSITION
    explored_level,   # 0/1/2 — LOW(<30%)/MED(30-65%)/HIGH(>65%) del mapa
)
# Total estados posibles: 2^8 * 3 * 3 ≈ 2304
```

**Construido en:** `build_state(visible_enemies, visible_allies)`
- `explored_level` usa `shared_data['explored_count']` (actualizado por Environment cada tick).

---

## Heurísticas — calculate_heuristic_biases(state, phase, game_context)

### game_context (construido en decide())
```python
game_context = {
    'carrying_resources':   self.carrying_resources,
    'carrying_capacity':    self.carrying_capacity,
    'explored_percent':     float,     # explored_count / 1600
    'tower_count':          int,       # torres en known_map
    'num_alive_collectors': int,
    'current_action_streak': int,      # ticks haciendo la misma acción
    'ticks_since_progress': int,       # ticks sin progreso real
    'guard_near':           0|1,       # extraído del state
}
```

### Reglas de emergencia (siempre dominan)

| Condición | Efecto |
|---|---|
| `enemy_near` | FLEE += 200 |
| `enemy_near AND NOT guard_near` | FLEE += 120 adicional |
| `phase=="EARLY" AND enemy_near` | FLEE += 80 adicional |
| `is_full` | RETURN_TO_BASE += 90 |
| `carrying` (parcial) | RETURN_TO_BASE += 40 * fill_ratio |
| `NOT carrying` | RETURN_TO_BASE += -80 |
| `NOT has_build_kit` | BUILD_TOWER = -200 (forzado) |

### Reglas estratégicas

| Condición | Efecto |
|---|---|
| `resources_known` | GO_TO_RESOURCE += 60 |
| `NOT carrying AND resources_known` | GO_TO_RESOURCE += 40 |
| `explored_level==0` | EXPLORE += 50 |
| `NOT resources_known` | EXPLORE += 40 |
| `NOT guard_near AND NOT tower_near` | EXPLORE += -60 |
| `phase=="EARLY"` | EXPLORE += 40 |
| `has_build_kit AND tower_count >= max_t` | BUILD_TOWER += -80 |
| `has_build_kit AND num_alive_collectors<=1` | BUILD_TOWER += -150 |
| `action_streak > 12` | current_action -= 8*(streak-12) |
| `ticks_since_progress > 12` | GO_TO_RESOURCE += prog*3, EXPLORE += prog*2 |

---

## Acciones — Implementación

### execute_explore(all_collectors)
- Busca celdas **frontera** (inexploradas adyacentes a exploradas).
- Score: `100 - dist*0.5 - risk*20 - other_heading*15`
- Penaliza celdas donde otros collectors se dirigen.

### execute_go_to_resource(all_collectors)
- Busca en `discovered_resources`.
- Score: `amount - dist*0.5 - risk*10 - others_going*5`
- Fallback: `execute_explore()` si no hay recursos conocidos.

### execute_return_to_base()
- A* hacia `BASE_POSITION`.

### execute_flee(visible_enemies, visible_allies)
- Opciones: torres (+30 base), guards (+20 base), base (+10 base).
- Penaliza cada opción por proximidad a enemigos.

### execute_build_tower()
- `_select_build_cell()`: score = `risk*10 + resource_proximity - redundancy*15 - dist*0.2`
- `build_target` persiste entre ticks hasta que la torre se construye.
- Al llegar: `wants_to_build = True` → environment construye la torre.

---

## Función de costo A*

```python
cost = 1.0 + risk_map[nx][ny]  # (o RISK_UNEXPLORED=0.3 si no explorado)
cost += max(0, (3 - dist_to_known_enemy) * 1.5)  # penalizar cercanía enemigos
min = 0.1
```

---

## Recompensas

### Desde environment (eventos)
```
deliver_resources → +10
collect           → +5
explore           → +3
build_tower       → +8
danger_zone       → -2
lose_resources    → -5
die               → -10
```

### Desde decide() (paso a paso)
```python
# _compute_step_reward(action, guard_near):
if prev_action == 'RETURN_TO_BASE' and carrying_resources == 0: reward += -4
if ticks_since_progress > 12: reward += -5
```

---

## Tracking de progreso (anti-estancamiento)

```python
# Atributos en __init__:
self.current_action_streak = 0      # ticks repitiendo la misma acción
self.ticks_since_progress  = 0      # ticks sin recolectar ni explorar
self._prev_resources_carried = 0
self._prev_explored_count    = 0

# Actualizado en _update_progress_tracking(action):
# - streak++ si action == current_action, else = 0
# - si carrying_resources cambió o explored_count creció > 2 → reset ticks_since_progress
```

---

## Señalización de construcción de torres

```python
# El collector señaliza al environment:
self.wants_to_build = True    # quiere construir esta posición
self.build_target   = (x, y)  # posición objetivo

# El environment (en paso 8 del tick):
if c.wants_to_build and c.has_build_kit and c.build_target == c.position:
    # Construye la torre y limpia: c.has_build_kit=False, c.build_target=None
```

---

## Flujo decide()

```
perceive() → build_state() → _get_game_phase() → game_context dict
→ calculate_heuristic_biases(state, phase, ctx)
→ _update_progress_tracking()
→ q_learning.update(prev_state, prev_action, step_reward, state)  # reward de paso
→ q_learning.get_action(state, biases)  # ε-greedy
→ execute_ACTION()
→ next_position = path.pop(0) o self.position (idle)
```
