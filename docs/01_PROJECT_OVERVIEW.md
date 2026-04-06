# PROJECT OVERVIEW

## Objetivo del juego

Simulación multi-agente con dos equipos compitiendo en un grid 40×40:

- **Equipo A** (RL + heurísticas): recolectar el 80% de los recursos del mapa y depositarlos en la base.
- **Equipo B** (algoritmos genéticos): eliminar a todos los agentes del Equipo A.

**Condición de victoria:**
- Equipo A: `base_resources >= total_resources * 0.80`
- Equipo B: no quedan collectors ni guards vivos

---

## Agentes

### Equipo A
| Agente | Archivo | Cantidad | Rol |
|---|---|---|---|
| Collector | `agents/collector.py` | 5 | Explora, recolecta recursos, construye torres |
| Guard | `agents/guard.py` | 5 | Protege collectors, patrulla, hace scouting |
| Tower | `agents/tower.py` | variable | Defensa estática, reduce riesgo, ataca cazadores |

### Equipo B
| Agente | Archivo | Cantidad | Rol |
|---|---|---|---|
| Hunter | `agents/hunter.py` | 15 (default) | Ataca agentes A; mata = buffs de velocidad/HP; muere = respawn con genes mutados |

---

## Mecánicas clave

### Fog of War
- Equipo A comparte un `known_map[x][y]` mutable con campos `explored`, `last_seen`, `last_known_type`.
- Solo ven lo que han explorado. Cazadores siempre visibles en el render (el jugador los controla).

### Risk Map
- `risk_map[x][y]` float 0–5. Decae por tick (`*0.98`), sube cerca de enemigos, baja cerca de torres.
- Los agentes usan este mapa para costear rutas A* y discretizar `risk_level` en el estado RL.

### Build Kits
- Por cada 10 recursos depositados en base → se genera 1 build kit.
- Se entrega al primer collector en BASE_POSITION sin kit.
- Reglas: no dar si queda 1 solo collector, no dar si torres ≥ cap dinámico.

### Combate simultáneo
- Todos los ataques se recolectan primero (`_collect_all_attacks()`) y el daño se aplica después (`_resolve_combat()`).
- Cazadores: daño letal (9999). Guards: daño=1, cooldown=2. Torres: daño=1, cooldown=2.

---

## Cómo ejecutar

```bash
# Entrenar agentes (sin pygame, a máxima velocidad)
python train.py

# Jugar con agentes entrenados
python main.py

# Jugar sin entrenamiento previo (Q-tables vacías, epsilon alto)
python main.py   # si no existe data/*.pkl, inicia en blanco
```

---

## Constantes globales más importantes

```
MAP_WIDTH/HEIGHT = 40
BASE_POSITION    = (5, 5)
NUM_COLLECTORS   = 5
NUM_GUARDS       = 5
NUM_HUNTERS      = 15
WIN_RESOURCE_PERCENT = 0.80
MAX_TICKS        = 5000
```

Ver [09_CONSTANTS_REFERENCE.md](09_CONSTANTS_REFERENCE.md) para el listado completo.
