# SISTEMA DE Q-LEARNING (rl/q_learning.py)

## Arquitectura

- **Una instancia por tipo de agente**: todos los collectors comparten `collector_ql`, todos los guards comparten `guard_ql`.
- **Q_table**: `defaultdict(lambda: defaultdict(float))` → `Q[state_tuple][action_str] = float`
- Los estados no vistos se inicializan a 0.0 automáticamente.

---

## Clase QLearning

```python
QLearning(
    actions: list[str],     # acciones posibles del agente
    alpha: float,           # learning rate (0.2)
    gamma: float,           # discount factor (0.9)
    epsilon: float,         # exploración inicial (0.20)
    epsilon_decay: float,   # por tick (0.998)
    epsilon_min: float,     # piso mínimo (0.08)
)
```

### Métodos

```python
get_action(state, heuristic_biases=None) -> str
    # epsilon-greedy: con prob epsilon → acción aleatoria
    # con prob 1-epsilon → argmax(Q[s][a] + bias[a])

get_best_action(state, heuristic_biases=None) -> str
    # sin exploración (epsilon=0), útil para evaluación

update(state, action, reward, next_state)
    # Q[s][a] += alpha * (reward + gamma * max(Q[s']) - Q[s][a])

decay_epsilon()
    # epsilon = max(epsilon_min, epsilon * epsilon_decay)
    # Llamado una vez por tick desde environment.tick()

save(filepath: str)
    # pickle.dump(dict(Q_table), f)

load(filepath: str) -> bool
    # Carga Q_table desde disco. Retorna True si existe.
```

---

## Flujo de actualización

El Q-learning se actualiza en **dos momentos** por agente por tick:

### 1. En decide() — reward de paso (0 o pequeño negativo)
```python
# Al inicio de decide(), antes de elegir nueva acción:
step_reward = _compute_step_reward(prev_action, context)
q_learning.update(prev_state, prev_action, step_reward, current_state)
# step_reward incluye: REWARD_RETURN_EMPTY, REWARD_STAGNATION
```

### 2. En environment — eventos positivos y negativos
```python
# Llamados desde environment.tick() cuando ocurren eventos:
agent.receive_reward('collect')          # +5
agent.receive_reward('deliver_resources') # +10
agent.receive_reward('build_tower')       # +8
agent.receive_reward('die')              # -10
agent.receive_reward('lose_resources')   # -5
# etc.

# receive_reward() usa prev_state como next_state (aproximación):
q_learning.update(prev_state, prev_action, reward, prev_state)
```

---

## Biases heurísticos

La función `get_action()` acepta `heuristic_biases: dict[str, float]`.

```python
score[action] = Q[state][action] + biases.get(action, 0.0)
```

Esto permite que las heurísticas **orienten** al Q-learning sin sustituirlo:
- Si el Q-learning aprendió bien, los biases son un complemento menor.
- Al inicio del entrenamiento (Q-table vacía), los biases **son** el comportamiento.

---

## Tamaño de las Q-tables

| Agente | Estado posible | Acciones | Max states |
|---|---|---|---|
| Collector | 2^8 × 3 × 3 | 5 | ~2,304 |
| Guard | 2^8 × 3^3 × 2^4 | 7 | ~110,592 |

En la práctica, solo se visitan los estados que ocurren durante el entrenamiento.

---

## Parámetros actuales

```python
QL_ALPHA         = 0.2    # learning rate
QL_GAMMA         = 0.9    # discount
QL_EPSILON       = 0.20   # exploración inicial (en entrenamiento)
QL_EPSILON_DECAY = 0.998  # decay por tick
QL_EPSILON_MIN   = 0.08   # piso mínimo

# En main.py (modo juego, ya entrenado):
epsilon_inicial  = QL_EPSILON_MIN  # = 0.08 (casi greedy)
```

---

## Save / Load de Q-tables

```python
# Guardar:
ql.save("data/collector_qtable.pkl")

# Cargar:
success = ql.load("data/collector_qtable.pkl")
# Si no existe → retorna False, Q_table queda vacía

# En main.py (al inicio):
collector_ql.load(f"{QTABLE_SAVE_PATH}collector_qtable.pkl")
guard_ql.load(f"{QTABLE_SAVE_PATH}guard_qtable.pkl")
```

---

## Cómo los agentes acceden a la Q-table

Los agentes reciben la instancia compartida en su constructor:
```python
# En environment.__init__:
c = Collector(pos, self.shared_data, self.collector_ql)
g = Guard(pos, self.shared_data, self.guard_ql)

# En el agente:
self.q_learning = q_learning  # referencia al objeto compartido
self.q_learning.get_action(state, biases)
self.q_learning.update(...)
```

Modificar la Q-table desde un agente la modifica para todos los agentes del mismo tipo.

---

## Cómo resetear el aprendizaje

```python
# Opción 1: borrar los .pkl
# (simplemente borrar data/collector_qtable.pkl y data/guard_qtable.pkl)

# Opción 2: en código
ql.Q_table = defaultdict(lambda: defaultdict(float))
ql.epsilon = QL_EPSILON
```

---

## Modificar el sistema RL

Para **agregar una nueva variable al estado**:
1. `agents/collector.py:build_state()` → agregar variable al tuple
2. `agents/collector.py:calculate_heuristic_biases()` → desempaquetar la nueva variable
3. Actualizar comentario del tamaño de estados
4. Resetear Q-tables (el cambio de shape invalida las tablas existentes)

Para **agregar una nueva recompensa**:
1. `utils/constants.py` → agregar `REWARD_NUEVA = X`
2. `agents/collector.py:get_reward()` → agregar `'evento': REWARD_NUEVA`
3. `environment.py` → llamar `agent.receive_reward('evento')` cuando el evento ocurra
