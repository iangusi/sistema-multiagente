"""
Genera Q-tables pre-inicializadas con conocimiento experto.

Enumera todas las combinaciones de estado válidas para collectors y guards,
asigna Q-values positivos a la acción óptima según reglas del dominio, y
guarda los archivos .pkl listos para que el entrenamiento afine desde ahí.

Uso:
    python init_qtables.py

Los archivos generados sobreescriben cualquier Q-table existente en data/.
Luego corre python train.py normalmente.
"""

import itertools
import os
import pickle
from collections import defaultdict

QTABLE_SAVE_PATH = "data/"

# Q-value para la acción óptima. Debe estar en el rango de las recompensas
# reales (collect=+5, deliver=+10) para que el entrenamiento pueda ajustarlo
# sin quedar atrapado en un prior demasiado rígido.
Q_BEST   = 6.0
Q_SECOND = 1.0   # segunda mejor opción (para casos con trade-off claro)

# ---------------------------------------------------------------------------
# COLLECTOR
# ---------------------------------------------------------------------------
# Estado (10 variables):
#   (enemy_near, carrying, is_full, risk_level, guard_near, tower_near,
#    has_build_kit, resources_known, near_base, explored_level)
#
# enemy_near       : 0/1
# carrying         : 0/1   (carrying_resources > 0)
# is_full          : 0/1   (carrying >= capacity)
# risk_level       : 0=LOW / 1=MED / 2=HIGH
# guard_near       : 0/1
# tower_near       : 0/1
# has_build_kit    : 0/1
# resources_known  : 0/1
# near_base        : 0/1   (distancia Manhattan a base <= 5)
# explored_level   : 0=LOW(<30%) / 1=MID(30-65%) / 2=HIGH(>65%)

COLLECTOR_ACTIONS = ['EXPLORE', 'GO_TO_RESOURCE', 'RETURN_TO_BASE', 'FLEE', 'BUILD_TOWER']


def _collector_best_actions(state):
    """
    Retorna lista [(accion, q_value)] en orden de prioridad para este estado.
    Solo incluye las acciones con Q-value > 0.
    """
    (enemy_near, carrying, is_full, risk_level, guard_near,
     tower_near, has_build_kit, resources_known, near_base,
     explored_level) = state

    result = []

    # 1. FLEE: máxima prioridad cuando hay enemigo cerca
    if enemy_near:
        result.append(('FLEE', Q_BEST))
        # Si hay guardia/torre cerca, puede intentar GO_TO_RESOURCE igualmente
        if guard_near or tower_near:
            if resources_known and not is_full:
                result.append(('GO_TO_RESOURCE', Q_SECOND))
        return result

    # 2. RETURN_TO_BASE: cuando cargado al máximo
    if is_full:
        result.append(('RETURN_TO_BASE', Q_BEST))
        return result

    # 3. RETURN_TO_BASE parcial: si está cerca de la base y lleva algo
    if near_base and carrying:
        result.append(('RETURN_TO_BASE', Q_BEST))
        if resources_known:
            result.append(('GO_TO_RESOURCE', Q_SECOND))
        return result

    # 4. GO_TO_RESOURCE: recurso conocido, no cargado al máximo, sin enemigo
    if resources_known:
        result.append(('GO_TO_RESOURCE', Q_BEST))
        # BUILD_TOWER como segunda opción si tiene kit y mapa medio-explorado
        if has_build_kit and explored_level >= 1:
            result.append(('BUILD_TOWER', Q_SECOND))
        return result

    # 5. BUILD_TOWER: tiene kit, mapa parcialmente explorado, no hay recurso visible
    if has_build_kit and explored_level >= 1:
        result.append(('BUILD_TOWER', Q_BEST))
        result.append(('EXPLORE', Q_SECOND))
        return result

    # 6. EXPLORE: sin recursos conocidos, mapa sin cubrir
    result.append(('EXPLORE', Q_BEST))
    return result


def build_collector_qtable():
    """Construye y retorna el dict de Q-table del collector."""
    Q = defaultdict(lambda: defaultdict(float))

    enemy_near_vals    = [0, 1]
    carrying_vals      = [0, 1]
    is_full_vals       = [0, 1]
    risk_level_vals    = [0, 1, 2]
    guard_near_vals    = [0, 1]
    tower_near_vals    = [0, 1]
    has_kit_vals       = [0, 1]
    res_known_vals     = [0, 1]
    near_base_vals     = [0, 1]
    explored_lvl_vals  = [0, 1, 2]

    valid = 0
    for combo in itertools.product(
        enemy_near_vals, carrying_vals, is_full_vals, risk_level_vals,
        guard_near_vals, tower_near_vals, has_kit_vals, res_known_vals,
        near_base_vals, explored_lvl_vals
    ):
        enemy_near, carrying, is_full = combo[0], combo[1], combo[2]

        # Estado imposible: no lleva nada pero está lleno
        if carrying == 0 and is_full == 1:
            continue

        state = combo
        for action, q_val in _collector_best_actions(state):
            Q[state][action] = q_val
        valid += 1

    print(f"Collector: {valid} estados válidos inicializados.")
    return dict(Q)


# ---------------------------------------------------------------------------
# GUARD
# ---------------------------------------------------------------------------
# Estado (12 variables):
#   (enemy_near, enemy_in_range, collector_near, collector_vulnerable,
#    risk_level, cooldown_ready, distance_to_enemy, numerical_advantage,
#    guards_near_collector, collector_near_base, unexplored_frontier_near,
#    collector_carrying)
#
# enemy_near              : 0/1
# enemy_in_range          : 0/1
# collector_near          : 0/1
# collector_vulnerable    : 0/1
# risk_level              : 0/1/2
# cooldown_ready          : 0/1
# distance_to_enemy       : 0=NEAR / 1=MID / 2=FAR
# numerical_advantage     : 0=LOW / 1=EVEN / 2=HIGH
# guards_near_collector   : 0/1/2
# collector_near_base     : 0/1
# unexplored_frontier_near: 0/1
# collector_carrying      : 0/1

GUARD_ACTIONS = ['PATROL', 'ESCORT', 'ATTACK', 'INTERCEPT',
                 'DEFEND_ZONE', 'INVESTIGATE', 'SCOUT']


def _guard_best_actions(state):
    """
    Retorna lista [(accion, q_value)] en orden de prioridad para este estado.
    """
    (enemy_near, enemy_in_range, collector_near, collector_vulnerable,
     risk_level, cooldown_ready, distance_to_enemy, numerical_advantage,
     guards_near_collector, collector_near_base, unexplored_frontier_near,
     collector_carrying) = state

    result = []

    # 1. ATTACK: enemigo al alcance y cooldown listo
    if enemy_in_range and cooldown_ready:
        result.append(('ATTACK', Q_BEST))
        # Si tiene ventaja numérica, atacar es muy bueno
        if numerical_advantage == 2:
            result.append(('ATTACK', Q_BEST))   # ya añadido; no importa duplicar
        # Puede escoltear después de atacar si el collector es vulnerable
        if collector_vulnerable:
            result.append(('ESCORT', Q_SECOND))
        return result

    # 2. INTERCEPT: enemigo cerca pero fuera de alcance
    if enemy_near and not enemy_in_range:
        if collector_vulnerable:
            # Collector en peligro: interceptar ES urgente
            result.append(('INTERCEPT', Q_BEST))
            result.append(('ESCORT', Q_SECOND))
        elif numerical_advantage >= 1:
            result.append(('INTERCEPT', Q_BEST))
        else:
            # Desventaja numérica: no interceptar, escoltear/defender
            if collector_near:
                result.append(('ESCORT', Q_BEST))
            else:
                result.append(('DEFEND_ZONE', Q_BEST))
        return result

    # --- Sin enemigos activos ---

    # 3. ESCORT: collector vulnerable y cerca (aunque no haya enemigo visible,
    #    puede haberlo pronto en zonas de riesgo)
    if collector_vulnerable and collector_near:
        result.append(('ESCORT', Q_BEST))
        return result

    # 4. ESCORT: collector cargando recursos sin escolta
    if collector_near and collector_carrying and guards_near_collector == 0:
        result.append(('ESCORT', Q_BEST))
        if unexplored_frontier_near:
            result.append(('SCOUT', Q_SECOND))
        return result

    # 5. SCOUT: hay frontera inexplorada cerca y el collector no necesita escolta
    if unexplored_frontier_near and guards_near_collector >= 1:
        result.append(('SCOUT', Q_BEST))
        if collector_near:
            result.append(('ESCORT', Q_SECOND))
        return result

    if unexplored_frontier_near and not collector_near:
        result.append(('SCOUT', Q_BEST))
        return result

    # 6. ESCORT preventivo: collector cerca sin escolta suficiente
    if collector_near and guards_near_collector == 0:
        result.append(('ESCORT', Q_BEST))
        return result

    # 7. DEFEND_ZONE: zona de alto riesgo
    if risk_level >= 2:
        result.append(('DEFEND_ZONE', Q_BEST))
        if collector_near:
            result.append(('ESCORT', Q_SECOND))
        return result

    # 8. ESCORT moderado: collector cerca con escolta parcial
    if collector_near and guards_near_collector == 1:
        result.append(('ESCORT', Q_BEST))
        result.append(('PATROL', Q_SECOND))
        return result

    # 9. PATROL: comportamiento por defecto
    result.append(('PATROL', Q_BEST))
    return result


def build_guard_qtable():
    """Construye y retorna el dict de Q-table del guard."""
    Q = defaultdict(lambda: defaultdict(float))

    enemy_near_vals       = [0, 1]
    enemy_in_range_vals   = [0, 1]
    collector_near_vals   = [0, 1]
    coll_vuln_vals        = [0, 1]
    risk_level_vals       = [0, 1, 2]
    cooldown_vals         = [0, 1]
    dist_enemy_vals       = [0, 1, 2]
    num_adv_vals          = [0, 1, 2]
    guards_near_coll_vals = [0, 1, 2]
    coll_near_base_vals   = [0, 1]
    frontier_near_vals    = [0, 1]
    coll_carrying_vals    = [0, 1]

    valid = 0
    for combo in itertools.product(
        enemy_near_vals, enemy_in_range_vals, collector_near_vals,
        coll_vuln_vals, risk_level_vals, cooldown_vals, dist_enemy_vals,
        num_adv_vals, guards_near_coll_vals, coll_near_base_vals,
        frontier_near_vals, coll_carrying_vals
    ):
        (enemy_near, enemy_in_range, collector_near, collector_vulnerable,
         risk_level, cooldown_ready, distance_to_enemy, numerical_advantage,
         guards_near_collector, collector_near_base, unexplored_frontier_near,
         collector_carrying) = combo

        # Estados físicamente imposibles
        # enemy_in_range=1 requiere enemy_near=1
        if enemy_in_range == 1 and enemy_near == 0:
            continue
        # collector_vulnerable=1 requiere collector_near=1
        if collector_vulnerable == 1 and collector_near == 0:
            continue
        # distance_to_enemy=0 (NEAR) requiere enemy_near=1
        if distance_to_enemy == 0 and enemy_near == 0:
            continue
        # enemy_in_range=1 implica distance_to_enemy=0 (NEAR)
        if enemy_in_range == 1 and distance_to_enemy != 0:
            continue
        # collector_carrying=1 requiere collector_near=1
        if collector_carrying == 1 and collector_near == 0:
            continue
        # guards_near_collector > 0 requiere collector_near=1 (al menos uno)
        if guards_near_collector > 0 and collector_near == 0:
            continue

        state = combo
        for action, q_val in _guard_best_actions(state):
            # Evitar sobreescribir con valor menor
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

    # Guardar como latest (cargado por trainer como checkpoint)
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
    print()
    print("Nota: el trainer cargará estas Q-tables como checkpoint y mantendrá")
    print("epsilon=0.20 para explorar variantes. Los Q-values se ajustarán")
    print("según las recompensas reales del entorno.")


if __name__ == "__main__":
    main()
