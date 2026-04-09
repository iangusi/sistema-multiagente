"""
Genera Q-tables pre-inicializadas con conocimiento experto.

Estado del RECOLECTOR (5 variables):
  (build_dist, explore_dist, resource_dist, can_carry, hunter_near)
  build_dist:    0=sin kit, 1=kit cerca, 2=kit lejos
  explore_dist:  0=sin frontera, 1=cerca, 2=lejos
  resource_dist: 0=sin recursos, 1=cerca, 2=lejos
  can_carry:     0=lleno, 1=puede cargar más
  hunter_near:   0=no, 1=sí

Acciones: EXPLORE, GO_TO_RESOURCE, RETURN_TO_BASE, FLEE, BUILD_TOWER

Estado del GUARDIA (4 variables):
  (ally_in_danger, i_am_in_danger, hunter_near, can_attack)
  Todos binarios (0/1)

Acciones: EXPLORE, ATTACK, DEFEND, FLEE

Uso:
    python init_qtables.py

Los archivos generados sobreescriben cualquier Q-table existente en data/.
"""

import itertools
import os
import pickle
from collections import defaultdict

QTABLE_SAVE_PATH = "data/"

Q_BEST   = 6.0
Q_SECOND = 1.5

# ---------------------------------------------------------------------------
# RECOLECTOR
# ---------------------------------------------------------------------------

COLLECTOR_ACTIONS = ['EXPLORE', 'GO_TO_RESOURCE', 'RETURN_TO_BASE', 'FLEE', 'BUILD_TOWER']


def _collector_best_actions(state):
    """
    Retorna [(accion, q_value)] en orden de prioridad para el estado dado.
    Lógica:
      1. FLEE si hay cazador
      2. RETURN_TO_BASE si inventario lleno (can_carry=0)
      3. BUILD_TOWER si tiene kit (build_dist>0) y sin cazador
      4. GO_TO_RESOURCE si hay recurso y puede cargar
      5. EXPLORE en otros casos
    """
    build_dist, explore_dist, resource_dist, can_carry, hunter_near = state
    result = []

    # 1. HUIR domina cuando hay cazador
    if hunter_near:
        result.append(('FLEE', Q_BEST))
        return result

    # 2. IR_A_BASE cuando el inventario está lleno
    if can_carry == 0:
        result.append(('RETURN_TO_BASE', Q_BEST))
        return result

    # 3. CONSTRUIR si tiene kit cerca
    if build_dist == 1:
        result.append(('BUILD_TOWER', Q_BEST))
        if resource_dist == 1:
            result.append(('GO_TO_RESOURCE', Q_SECOND))
        return result
    
    # 4 CONSTRUIR si tiene kit (aunque lejos)
    if build_dist == 2:
        result.append(('BUILD_TOWER', Q_SECOND))
        if resource_dist == 1:
            result.append(('GO_TO_RESOURCE', Q_BEST))
        return result

    # 5. IR_POR_RECURSO si hay recursos conocidos y puede cargar
    if resource_dist > 0:
        result.append(('GO_TO_RESOURCE', Q_BEST))
        if explore_dist > 0:
            result.append(('EXPLORE', Q_SECOND))
        return result

    # 6. EXPLORAR en otros casos
    result.append(('EXPLORE', Q_BEST))

    return result


def build_collector_qtable():
    Q = defaultdict(lambda: defaultdict(float))

    build_vals    = [0, 1, 2]
    explore_vals  = [0, 1, 2]
    resource_vals = [0, 1, 2]
    carry_vals    = [0, 1]
    hunter_vals   = [0, 1]

    valid = 0
    for combo in itertools.product(
        build_vals, explore_vals, resource_vals, carry_vals, hunter_vals
    ):
        state = combo
        for action, q_val in _collector_best_actions(state):
            if Q[state][action] < q_val:
                Q[state][action] = q_val
        valid += 1

    print(f"Collector: {valid} estados válidos inicializados.")
    return dict(Q)


# ---------------------------------------------------------------------------
# GUARDIA
# ---------------------------------------------------------------------------

GUARD_ACTIONS = ['EXPLORE', 'ATTACK', 'DEFEND', 'FLEE']


def _guard_best_actions(state):
    """
    Retorna [(accion, q_value)] en orden de prioridad.
    Lógica:
      1. FLEE si está en peligro personal sin poder atacar
      2. DEFEND si aliado en peligro
      3. ATTACK si puede atacar
      4. EXPLORE en otros casos
    """
    ally_in_danger, i_am_in_danger, hunter_near, can_attack = state
    result = []

    # 1. HUIR si en peligro personal
    if (i_am_in_danger or hunter_near) and not can_attack:
        result.append(('FLEE', Q_BEST))
        return result
   
    # 2. ATACAR domina si puede hacerlo
    if hunter_near and can_attack:
        result.append(('ATTACK', Q_BEST))
        if ally_in_danger:
            result.append(('DEFEND', Q_SECOND))
        return result
    
    # 3. DEFENDER si aliado en peligro
    if ally_in_danger:
        result.append(('DEFEND', Q_BEST))
        return result

    # 4. EXPLORAR por defecto
    result.append(('EXPLORE', Q_BEST))
    return result


def build_guard_qtable():
    Q = defaultdict(lambda: defaultdict(float))

    valid = 0
    for combo in itertools.product([0, 1], [0, 1], [0, 1], [0, 1]):

        state = combo
        for action, q_val in _guard_best_actions(state):
            if Q[state][action] < q_val:
                Q[state][action] = q_val
        valid += 1

    print(f"Guard:     {valid} estados válidos inicializados.")
    return dict(Q)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    os.makedirs(QTABLE_SAVE_PATH, exist_ok=True)

    print("Generando Q-tables con conocimiento experto...")
    print()

    collector_q = build_collector_qtable()
    guard_q     = build_guard_qtable()

    files = {
        f"{QTABLE_SAVE_PATH}collector_qtable.pkl": collector_q,
        f"{QTABLE_SAVE_PATH}guard_qtable.pkl":     guard_q,
    }
    for path, qtable in files.items():
        with open(path, 'wb') as f:
            pickle.dump(qtable, f)

    print()
    print("Archivos generados:")
    for fname in os.listdir(QTABLE_SAVE_PATH):
        if fname.endswith('.pkl'):
            path = os.path.join(QTABLE_SAVE_PATH, fname)
            print(f"  {path}  ({os.path.getsize(path):,} bytes)")

    print()
    print("Listo. Ejecuta  python train.py  para afinar con entrenamiento.")


if __name__ == "__main__":
    main()
