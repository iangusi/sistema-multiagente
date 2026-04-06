# AGENTS — HUNTER Y TOWER (Equipo B / Estructuras)

> **IMPORTANTE:** Hunter y Tower NO usan Q-learning. No modificar para cambiar el comportamiento RL de los agentes A.

---

## HUNTER (agents/hunter.py)

### Stats base
```python
HP = 1, speed = 1, vision_range = 5
attack_range = 1, attack_cooldown = 1, damage = 9999 (letal)
comm_radius = 10, memory_duration = 8 ticks
```

### On-kill (buffs acumulables)
```python
Matar collector → speed += 0.25 (cap: base*2.0)
Matar guard     → HP    += 1    (cap: base*3)
```

### Respawn
- Tras morir: `respawn_timer = 15`
- Cada tick: `try_respawn()` decrementa timer
- Al revivir: genes se mutan, stats se resetean al base, spawna en borde del mapa

### FSM — 9 estados
```
ATTACK       — atacar target
CHASE        — perseguir target
FLEE         — alejarse de amenaza (guards/torres cercanas)
GROUP        — moverse hacia centroide de aliados
STALK        — mantenerse a safe_distance=5 del target
FLANK        — posición lateral al target
WAIT_FOR_REINFORCEMENTS — esperar aliados
WANDER       — explorar mapa interior
RETREAT      — moverse hacia borde del mapa
```

### Genes (evolution/genetic_system.py)
```python
class Genes:
    gamma: float  # agresividad (0-1)
    beta:  float  # persecución (0-1)
    delta: float  # evasión (0-1)
    alpha: float  # cohesión (0-1)
    mutation_rate = 0.1
```
Los genes pesan las acciones del FSM en `evaluate_actions()`.

### decide() — flujo
```
perceive() → communicate(alive_hunters)
→ calculate_risk(), calculate_opportunity()
→ evaluate_actions()  # gene-weighted scoring
→ select_target()
→ calculate_movement()  # ejecuta la acción ganadora
```

### Comunicación
Dentro de `comm_radius=10`: comparte `local_memory` con aliados vivos.
Los hunters construyen un mapa mental de enemigos vistos recientemente.

### Targeting del usuario
```python
h.user_target_position = (gx, gy)  # fijado por environment.handle_click()
```
Si está seteado, el hunter prioriza moverse hacia esa posición.

---

## TOWER (agents/tower.py)

### Stats
```python
vision_range = 4, attack_range = 4
attack_cooldown = 2, damage = 1
```

### update(grid, known_map, last_seen_enemies, risk_map, current_tick)

Ejecutado en `environment._collect_all_attacks()` UNA VEZ por tick:

```
1. Decrementar cooldown
2. Revelar celdas en vision_range → actualizar known_map
3. Detectar hunters → actualizar last_seen_enemies
4. Reducir risk_map en attack_range (seguridad pasiva)
5. Si cooldown==0 y hunter en attack_range → attack(nearest_hunter)
```

Retorna `[(tower, target)]` o `[]`.

### Por qué no se llama en el paso 1 de tick()
Las torres se procesan en el paso 9 (combate) para evitar doble decremento de cooldown.

---

## Cómo afectan los hunters al Equipo A

| Evento | Consecuencia en A |
|---|---|
| Hunter ve collector | Lo persigue y ataca (daño=9999, muerte instantánea) |
| Hunter ve guard | Puede atacarlo o evitarlo según genes |
| Hunter muere | respawn_timer=15, genes mutados → vuelve más adaptado |
| Hunter mata collector | speed +0.25 (más difícil de atrapar) |
| Hunter mata guard | HP +1 (más resistente) |

---

## Modificar el Equipo B

Si quieres **ajustar dificultad** de los hunters:
- Velocidad: `HUNTER_SPEED` en constants.py (base) + `HUNTER_MAX_SPEED`, `HUNTER_SPEED_INCREMENT`
- HP: `HUNTER_HP`, `HUNTER_MAX_HP`, `HUNTER_HP_INCREMENT`
- Respawn: `HUNTER_RESPAWN_TICKS`
- Número: `NUM_HUNTERS` en constants.py o `num_hunters_override` al Environment

Si quieres **ajustar el comportamiento** de los genes:
- Valores iniciales: `GENE_GAMMA_INIT`, `GENE_BETA_INIT`, `GENE_DELTA_INIT`, `GENE_ALPHA_INIT`
- Tasa de mutación: `GENE_MUTATION_RATE`
- La lógica de scoring está en `hunter.py:evaluate_actions()`

Si quieres **ajustar las torres**:
- `TOWER_ATTACK_RANGE`, `TOWER_ATTACK_COOLDOWN`, `TOWER_DAMAGE`, `TOWER_VISION`
