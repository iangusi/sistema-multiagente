# ENVIRONMENT.PY — MOTOR DEL JUEGO

## Clase Cell

```python
class Cell:
    position: (x, y)
    type: str          # 'empty' | 'obstacle' | 'resource' | 'base' | 'tower'
    agents: list       # agentes actualmente en esta celda
    tower: Tower|None  # referencia a la Torre si type=='tower'
    resource_amount: int
```

---

## Clase Environment — Constructor

```python
Environment(
    num_hunters_override=None,  # int o None → usa NUM_HUNTERS de constants
    collector_ql=None,          # QLearning inyectado (trainer) o None (crea interno)
    guard_ql=None,              # QLearning inyectado (trainer) o None (crea interno)
    headless=False,             # True = modo entrenamiento sin render
)
```

### Atributos principales

```python
self.width, self.height         # 40, 40
self.cells[x][y]                # grid 2D de objetos Cell
self.known_map[x][y]            # dict: {explored, last_seen, last_known_type}
self.discovered_resources       # lista de {position, amount, last_seen}
self.last_seen_enemies          # lista de {agent, position, type, tick}
self.risk_map[x][y]             # float 0–5
self.shared_data                # dict con referencias a los 4 anteriores + explored_count

self.collectors, self.guards    # listas de agentes Equipo A
self.hunters, self.towers       # listas de agentes B y torres

self.collector_ql, self.guard_ql  # instancias QLearning

self.base_resources             # recursos depositados acumulados
self.total_resources            # suma de todos los recursos del mapa al inicio
self.win_target                 # total_resources * 0.80
self.current_tick               # tick actual
self.game_over, self.winner     # bool, 'A'|'B'|None
self.last_event                 # string del último evento (para sidebar)
self.headless                   # bool
```

### shared_data (dict compartido por todos los agentes A)

```python
shared_data = {
    'known_map':            self.known_map,
    'discovered_resources': self.discovered_resources,
    'last_seen_enemies':    self.last_seen_enemies,
    'risk_map':             self.risk_map,
    'explored_count':       int,  # actualizado cada tick
}
```

---

## Ciclo tick() — Orden de pasos

```
1.  PERCEPCIÓN — agent.perceive(self, tick) para todos los agentes A + torres
2.  (torres perciben dentro de _collect_all_attacks, no aquí)
3.  COMUNICACIÓN — hunter.communicate(alive_hunters)
4.  DECISIONES — c.decide(), g.decide(), h.decide() → calcula next_position
5.  MOVIMIENTO SIMULTÁNEO — mueve todos los agentes (actualiza cells[].agents)
6.  RECOLECCIÓN — collectors recogen si están en celda resource
7.  DEPÓSITO — collectors depositan si están en BASE_POSITION
8.  CONSTRUCCIÓN DE TORRES — si c.wants_to_build y c.build_target==c.position
9.  COMBATE — _collect_all_attacks() luego _resolve_combat()
10. RESPAWN — hunters muertos intentan respawn tras 15 ticks
11. RISK MAP — _update_risk_map()
12. explored_count — sum de known_map[x][y]['explored']
13. RL DECAY — collector_ql.decay_epsilon(), guard_ql.decay_epsilon()
14. VICTORIA — _check_win_conditions()
15. tick += 1
```

---

## Combate

### _collect_all_attacks()
Recolecta ataques SIN aplicar daño. Retorna `[(attacker, target, damage), ...]`.
- Hunters → collectors/guards si en rango y cooldown=0
- Guards → hunters en rango y cooldown=0 (busca el más cercano)
- Torres → `tower.update()` devuelve sus ataques

### _resolve_combat()
- Acumula daño por target
- Aplica `current_hp -= total_dmg` simultáneamente
- Para cada killed: `_kill_agent(agent)`
  - Collector muerto: `receive_reward('lose_resources')`, `receive_reward('die')`, notifica a guards cercanos
  - Guard muerto: `receive_reward('die')`
  - Hunter muerto: `agent.die()` → inicia timer de respawn

---

## Risk Map — _update_risk_map()

```
1. Decay:   risk[x][y] *= 0.98
2. Difusión: suma 10% del riesgo de vecinos (sobre copia)
3. Enemigos: +RISK_ENEMY_WEIGHT * factor_distancia en radio 3 de cada enemigo reciente
4. Torres:  -RISK_TOWER_REDUCTION en radio attack_range de cada torre
5. Clamp: [0.0, 5.0]
```

---

## Build Kits — _check_build_kits()

```python
total_kits = base_resources // BUILD_KIT_COST  # cada 10 depositados
new_kits = total_kits - _kits_given

# Reglas (si no pasan → no dar kit):
if len(alive_collectors) <= 1: return      # no dar al último recolector
if len(towers) >= max_towers: return       # cap dinámico de torres
max_towers = max(2, int(explored_count * MAX_TOWERS_PER_EXPLORATION))

# Si pasa: dar kit al primer collector en BASE_POSITION sin kit
```

---

## get_game_phase()

```python
def get_game_phase(self) -> 'EARLY' | 'MID' | 'LATE':
    explored_pct = explored_count / (40*40)
    # EARLY < 0.30, MID [0.30, 0.65), LATE >= 0.65
```

---

## Rendering

Hay **dos** sistemas de render (actualmente coexisten):

1. **`main.py:render(screen, env, font_sm, font_md)`** — función standalone usada en la partida real.
2. **`environment.py:Environment.render(screen)`** — método interno de la clase (render alternativo).

En `main.py` se usa la función standalone (1). El método interno (2) existe pero no se llama desde main.py.

---

## Inicialización del mapa

- `_place_obstacles()`: 30 obstáculos aleatorios, no en zona de base (radio 3).
- `_place_resources()`: 20 nodos de recursos, 5–15 por nodo, no en radio 2 de base.
- `_spawn_near_base(offset)`: posición libre cerca de base para spawn de collectors/guards.
- Zona inicial de base se revela en known_map y risk=0.
