# AGENTS/GUARD.PY

## Datos del agente

```python
HP = 1, speed = 1, vision_range = 3
attack_range = 2, attack_cooldown = 2, damage = 1
No muere y respawnea (muerte permanente como el collector).
```

## Acciones disponibles

```python
ACTIONS = ['PATROL', 'ESCORT', 'ATTACK', 'INTERCEPT', 'DEFEND_ZONE', 'INVESTIGATE', 'SCOUT']
#                                                                                     ^ NUEVA
```

---

## Estado RL — 12 variables

```python
state = (
    enemy_near,               # 0/1 — Hunter visible
    enemy_in_range,           # 0/1 — Hunter a distancia <= attack_range (2)
    collector_near,           # 0/1 — Collector visible
    collector_vulnerable,     # 0/1 — Collector sin guardia/torre cerca Y con enemigo visible
    risk_level,               # 0/1/2 — LOW(<0.3)/MED/HIGH(>0.7)
    cooldown_ready,           # 0/1 — current_cooldown == 0
    distance_to_enemy,        # 0/1/2 — NEAR(<=3)/MID(<=6)/FAR
    numerical_advantage,      # 0/1/2 — LOW(<0.9)/EVEN/HIGH(>1.1) ratio aliados/enemigos
    guards_near_collector,    # 0/1/2 — 0,1, o 2+ guards a distancia<=5 del collector más cercano
    collector_near_base,      # 0/1 — collector más cercano a distancia<=5 de BASE_POSITION
    unexplored_frontier_near, # 0/1 — hay frontera a distancia<=8 del guardia
    collector_carrying,       # 0/1 — el collector más cercano lleva recursos
)
# Total estados posibles: ≈ 110,592 (muchos rarísimos o nunca visitados)
```

**Nota:** `collector_vulnerable` requiere que haya enemigos visibles Y que ningún guard/tower esté a rango del collector.

---

## Heurísticas — calculate_heuristic_biases(state, phase, game_context)

### game_context (construido en decide())
```python
game_context = {
    'tower_count':           int,    # torres en known_map
    'recent_enemy_sighting': bool,   # last_seen_enemies con tick reciente (<= 5 ticks)
    'collector_has_kit':     bool,   # algún visible_collector tiene has_build_kit
    'guards_near_my_collector': 0|1|2,  # = state[8]
    'collector_near_base':   0|1,    # = state[9]
    'escorted_collector_collected': False,  # simplificado
}
```

### Reglas de heurísticas

| Acción | Condición | Bias |
|---|---|---|
| ATTACK | `enemy_in_range AND cooldown_ready` | +100 |
| ATTACK | `numerical_advantage==2 (HIGH)` | +40 |
| ATTACK | `numerical_advantage==0 (LOW)` | -60 |
| INTERCEPT | `enemy_near AND NOT enemy_in_range` | +80 |
| INTERCEPT | `collector_vulnerable` | +50 |
| ESCORT | `collector_vulnerable AND collector_carrying` | +100 |
| ESCORT | `collector_has_kit` | +130 |
| ESCORT | `guards_near_collector >= 2` | -60 (redundante) |
| ESCORT | `collector_near_base` | -40 (innecesario) |
| SCOUT | `unexplored_frontier_near` | +60 |
| SCOUT | `phase=="EARLY"` | +40 |
| SCOUT | `NOT collector_vulnerable` | +30 |
| DEFEND_ZONE | `tower_count > 0` | +50 |
| DEFEND_ZONE | `risk_level >= 2` | +40 |
| INVESTIGATE | `recent_enemy_sighting` | +60 |
| PATROL | siempre | +20 (base) |

---

## Acciones — Implementación

### execute_escort(collectors, visible_enemies)
- Score del collector: `has_kit*100 + carrying*50 - dist - risk*10`
- Si hay enemigo visible: goal = punto medio entre enemigo y collector.
- Si no: goal = posición del collector.

### execute_attack(target)
- Verifica rango <= 2 y cooldown == 0. Retorna bool.
- El daño real lo aplica environment en `_collect_all_attacks()`.
- Si no puede atacar, el path lleva hacia el target.

### execute_intercept(enemy)
- Predice dirección del enemigo (compara con last_seen_enemies).
- Prueba k=1,2,3 ticks adelante; elige punto con mejor score (`-dist_guard - dist_ruta`).

### execute_defend_zone()
- Score: recursos conocidos (`amount*0.5 + risk*5 - dist*0.3`) o posiciones de enemigos recientes (`10 - dist*0.2`).

### execute_investigate()
- Va a la última posición del enemigo más recientemente visto (`max(last_seen_enemies, key=tick)`).

### execute_patrol()
- Waypoints en frontera del mapa explorado (hasta 4 puntos, distribuidos).
- Se regeneran al completar un ciclo completo.

### execute_scout() ← NUEVA
```python
def execute_scout(self):
    frontier_cells = self._find_frontier_cells()  # inexploradas adyacentes a exploradas
    # Para cada cell: score = -dist*2 - risk*30 + unexplored_neighbors*15
    # Bonus: +20 si hay collector a distancia < 8
    # Va al mejor cell con _astar()
    # Fallback: execute_patrol() si no hay fronteras
```

---

## Métodos auxiliares nuevos

### _find_frontier_cells()
Retorna celdas **inexploradas** que tienen al menos un vecino explorado.
El guardia se mueve a estas celdas para revelar territorio nuevo con su visión.

### _count_unexplored_neighbors(cell)
Cuenta los 8 vecinos (incluyendo diagonales) de una celda que no están explorados.
Usado para priorizar fronteras con mayor potencial de revelación.

### _has_frontier_nearby()
Busca si hay frontera a distancia <= 8 del guardia.
Retorna 0 o 1. Usado para calcular `unexplored_frontier_near` en `build_state()`.

### _nearest_alive_collector(visible_collectors)
Retorna el collector vivo más cercano de la lista de visibles, o None.

### _get_game_phase()
Igual que Collector: usa `shared_data['explored_count']` para determinar EARLY/MID/LATE.

---

## Función de costo A*

```python
cost = 1.0 + risk_map[nx][ny]
cost += max(0, (3 - dist_to_known_enemy) * 1.5)  # penalizar cercanía enemigos
cost -= max(0, (TOWER_RANGE - dist_to_tower) * 0.3)  # bonus cerca de torres aliadas
min = 0.1
```

---

## Recompensas

### Desde environment (eventos)
```
kill_hunter            → +8
protect_collector      → +10
intercept              → +6
defend_zone            → +5
die                    → -10
collector_dies_nearby  → -8
bad_decision           → -3
```

### Recompensas de autonomía (nuevas)
```
sole_escort              → +12  (soy el ÚNICO guardia escoltando)
redundant_escort         → -4   (ya hay 2+ guardias con ese collector)
collector_safe_no_escort → -1   (escolto collector en la base)
scout_new_cells          → +4
scout_find_resource      → +7
escort_collector_collects→ +3   (collector escoltado recogió recursos)
guard_idle               → -2
```

---

## Flujo decide()

```
perceive() → build_state() → _get_game_phase() → game_context dict
→ calculate_heuristic_biases(state, phase, ctx)
→ q_learning.update(prev_state, prev_action, 0.0, state)
→ q_learning.get_action(state, biases)  # ε-greedy
→ execute_ACTION() según acción elegida
→ next_position = path.pop(0) o self.position (idle)
```

**Diferencia con Collector:** el guardia NO calcula `step_reward` en `decide()`. Sus recompensas vienen principalmente de eventos del environment.

---

## Atributo _all_collectors

```python
# En decide():
self._all_collectors = [c for c in all_collectors if c.is_alive]
# Usado en execute_scout() para encontrar collectors vivos
# Se actualiza cada tick
```
