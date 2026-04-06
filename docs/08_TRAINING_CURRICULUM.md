# SISTEMA DE ENTRENAMIENTO POR CURRÍCULO

## Archivos involucrados

```
train.py                    # Punto de entrada
training/
    __init__.py             # vacío
    curriculum.py           # Define CURRICULUM (lista de 4 fases)
    trainer.py              # Clase Trainer — lógica de entrenamiento
data/
    collector_qtable.pkl    # Q-table latest (cargada por main.py)
    guard_qtable.pkl
    collector_qtable_phase1.pkl  # Por fase (backup)
    ...
```

---

## Cómo ejecutar

```bash
# Entrenar completo (~1800 episodios, puede tardar varios minutos)
python train.py

# Resultado: Q-tables en data/
# Luego jugar:
python main.py
```

---

## Currículo (training/curriculum.py)

```python
CURRICULUM = [
    {
        "phase": 1,
        "name": "Exploración y recolección básica",
        "num_hunters": 0,                    # Sin enemigos
        "episodes": 300,
        "success_metric": "win_rate >= 0.99",
    },
    {
        "phase": 2,
        "name": "Amenaza baja",
        "num_hunters": 4,
        "episodes": 400,
        "success_metric": "win_rate >= 0.95",
    },
    {
        "phase": 3,
        "name": "Amenaza media",
        "num_hunters": 10,
        "episodes": 500,
        "success_metric": "win_rate >= 0.90",
    },
    {
        "phase": 4,
        "name": "Dificultad completa",
        "num_hunters": 15,                   # = NUM_HUNTERS
        "episodes": 600,
        "success_metric": "win_rate >= 0.85",
    },
]
```

**Total: 1800 episodios**, dificultad progresiva.

---

## Clase Trainer (training/trainer.py)

### Constructor

```python
trainer = Trainer()
# Crea collector_ql y guard_ql con epsilon=QL_EPSILON (0.20) para explorar
```

### train_curriculum(curriculum)

```
Para cada fase:
  1. Si fase > 1: epsilon = max(epsilon_actual, QL_EPSILON * 0.5)
     → re-explora parcialmente con las nuevas condiciones
  2. _train_phase(phase_config) → retorna métricas
  3. _save_qtables(phase_num) → guarda por fase y como 'latest'
  4. Imprime win_rate, avg_resources, avg_collectors_alive
```

### _train_phase(phase_config)

```python
for ep in range(episodes):
    env = Environment(
        num_hunters_override=num_hunters,
        collector_ql=self.collector_ql,  # Q-table compartida entre episodios
        guard_ql=self.guard_ql,
        headless=True,
    )
    for tick in range(TRAINING_MAX_TICKS_PER_EPISODE):  # 3000
        env.tick()
        if env.game_over: break

    # Métricas del episodio
    wins += (env.winner == 'A')
    total_resources_pct += env.base_resources / env.win_target

    # Decay epsilon UNA VEZ por episodio (no por tick, para que no caiga tan rápido)
    self.collector_ql.decay_epsilon()
    self.guard_ql.decay_epsilon()
```

### Métricas retornadas

```python
{
    'phase': int,
    'win_rate': float,           # wins / episodes
    'avg_resources': float,      # % promedio de recursos colectados vs win_target
    'avg_collectors_alive': float,
}
```

---

## Modo headless — ¿Qué cambia?

- `Environment(headless=True)` solo guarda el flag en `self.headless`.
- `tick()` NO llama a `render()` (ya era así — render es externo en main.py).
- En la práctica, headless solo significa que el trainer no llama `render()`.
- **No se importa pygame** durante el entrenamiento (las constantes de color son simples tuples).

---

## Inyección de Q-tables

Las Q-tables se crean en `Trainer.__init__` y se pasan a `Environment`:
```python
env = Environment(collector_ql=self.collector_ql, guard_ql=self.guard_ql)
```

Dentro de environment, los agentes reciben la misma instancia:
```python
Collector(pos, shared_data, collector_ql)  # → self.q_learning = collector_ql
```

**La Q-table se actualiza en-place durante el episodio** → aprendizaje acumulativo entre episodios.

---

## Guardado de Q-tables

```python
def _save_qtables(self, phase):
    # Por fase (backup):
    data/collector_qtable_phase{N}.pkl
    data/guard_qtable_phase{N}.pkl

    # Latest (usado por main.py):
    data/collector_qtable.pkl
    data/guard_qtable.pkl
```

El formato es `pickle.dump(dict(Q_table))`. Al cargar:
```python
loaded = pickle.load(f)
self.Q_table = defaultdict(lambda: defaultdict(float), loaded)
```

---

## Cómo modificar el currículo

### Agregar una fase
```python
# En training/curriculum.py, agregar un dict más a CURRICULUM
{
    "phase": 5,
    "name": "Presión extrema",
    "num_hunters": 20,
    "episodes": 800,
    "success_metric": "win_rate >= 0.70",
}
```

### Cambiar episodios por fase
```python
# En utils/constants.py:
TRAINING_EPISODES_PHASE_1 = 300  # cambiar aquí
# curriculum.py los lee automáticamente
```

### Entrenar solo una fase específica
```python
# En train.py, cambiar:
trainer.train_curriculum(CURRICULUM[2:3])  # solo fase 3
```

### Retomar desde Q-tables existentes
```python
# En train.py:
trainer.collector_ql.load("data/collector_qtable_phase2.pkl")
trainer.guard_ql.load("data/guard_qtable_phase2.pkl")
trainer.train_curriculum(CURRICULUM[2:])  # continuar desde fase 3
```

---

## Output del entrenamiento

```
============================================================
FASE 1: Exploración y recolección básica
Cazadores: 0
Episodios:  300
============================================================
  Ep   50/300 | WR=100.00% | ε=0.182 | Q: C=89 G=147
  Ep  100/300 | WR= 98.00% | ε=0.165 | Q: C=134 G=231
  ...

Fase 1 completada.
  Win rate:             98.67%
  Avg resources:        87.3%
  Avg collectors alive: 4.82
  Q-table sizes: C=412  G=867
  Q-tables guardadas en 'data/' (fase 1)
```
