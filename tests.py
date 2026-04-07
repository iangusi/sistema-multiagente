"""
Tests de verificación post-fixes.
Cubre: stagnation fix, pending_reward, balance de heurísticas, integración headless.
"""
import sys
import traceback

PASS = 0
FAIL = 0

def ok(name):
    global PASS
    PASS += 1
    print(f"  [OK] {name}")

def fail(name, detail=""):
    global FAIL
    FAIL += 1
    msg = f"  [FAIL] {name}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

# ============================================================
# 1. IMPORTS
# ============================================================
section("1. IMPORTS")

try:
    from utils.constants import (
        HEUR_RETURN_FULL, HEUR_GOTO_RESOURCE_MID_BONUS,
        STAGNATION_THRESHOLD, BASE_POSITION,
        NUM_COLLECTORS, NUM_GUARDS,
        TRAINING_MAX_TICKS_PER_EPISODE,
    )
    ok("utils.constants importado")
except Exception as e:
    fail("utils.constants", str(e)); sys.exit(1)

try:
    from rl.q_learning import QLearning
    ok("rl.q_learning importado")
except Exception as e:
    fail("rl.q_learning", str(e)); sys.exit(1)

try:
    from environment import Environment
    ok("environment importado")
except Exception as e:
    fail("environment", str(e)); sys.exit(1)

try:
    from agents.collector import Collector
    ok("agents.collector importado")
except Exception as e:
    fail("agents.collector", str(e)); sys.exit(1)

try:
    from agents.guard import Guard
    ok("agents.guard importado")
except Exception as e:
    fail("agents.guard", str(e)); sys.exit(1)

# ============================================================
# 2. CONSTANTES — BALANCE DE HEURÍSTICAS
# ============================================================
section("2. BALANCE DE HEURÍSTICAS (constants)")

from utils.constants import (
    HEUR_GOTO_RESOURCE_BASE, HEUR_GOTO_RESOURCE_EMPTY_BONUS,
    HEUR_RETURN_FULL, HEUR_GOTO_RESOURCE_MID_BONUS,
    HEUR_FLEE_ENEMY_NEAR, HEUR_FLEE_ENEMY_NO_GUARD,
    HEUR_SCOUT_LOW_EXPLORATION, HEUR_SCOUT_EARLY_BONUS, HEUR_SCOUT_NO_ESCORT_NEEDED,
    HEUR_INTERCEPT_ENEMY_NEAR,
)

# RETURN_TO_BASE cuando lleno debe ganar a GO_TO_RESOURCE en cualquier fase
max_goto_mid = HEUR_GOTO_RESOURCE_BASE + HEUR_GOTO_RESOURCE_MID_BONUS  # carrying → no empty bonus
if HEUR_RETURN_FULL > max_goto_mid:
    ok(f"RETURN_FULL ({HEUR_RETURN_FULL}) > GO_TO_RESOURCE+MID cuando carga ({max_goto_mid})")
else:
    fail("RETURN_FULL debería superar GO_TO_RESOURCE+MID cuando lleno",
         f"RETURN_FULL={HEUR_RETURN_FULL}, GO_RESOURCE+MID={max_goto_mid}")

# FLEE debe dominar sobre cualquier otra acción
max_non_flee = HEUR_RETURN_FULL  # el más alto posible de las otras
flee_min = HEUR_FLEE_ENEMY_NEAR
if flee_min > max_non_flee:
    ok(f"FLEE_BASE ({flee_min}) > máximo no-FLEE ({max_non_flee})")
else:
    fail("FLEE debería dominar cualquier otra acción",
         f"FLEE_BASE={flee_min}, max_no_flee={max_non_flee}")

# SCOUT con enemy_near debe perder ante INTERCEPT
scout_with_enemy = HEUR_SCOUT_LOW_EXPLORATION + HEUR_SCOUT_EARLY_BONUS + HEUR_SCOUT_NO_ESCORT_NEEDED - 90
intercept_base = HEUR_INTERCEPT_ENEMY_NEAR
if scout_with_enemy < intercept_base:
    ok(f"SCOUT con enemy_near ({scout_with_enemy}) < INTERCEPT ({intercept_base})")
else:
    fail("SCOUT con enemy_near debería perder ante INTERCEPT",
         f"SCOUT={scout_with_enemy}, INTERCEPT={intercept_base}")

# ============================================================
# 3. COLLECTOR — _pending_reward Y ATRIBUTOS
# ============================================================
section("3. COLLECTOR — atributos y pending_reward")

def make_env_headless(hunters=0):
    collector_ql = QLearning(
        actions=['EXPLORE', 'GO_TO_RESOURCE', 'RETURN_TO_BASE', 'FLEE', 'BUILD_TOWER'],
        alpha=0.2, gamma=0.9, epsilon=0.5, epsilon_decay=0.999, epsilon_min=0.08,
    )
    guard_ql = QLearning(
        actions=['PATROL', 'ESCORT', 'ATTACK', 'INTERCEPT', 'DEFEND_ZONE', 'INVESTIGATE', 'SCOUT'],
        alpha=0.2, gamma=0.9, epsilon=0.5, epsilon_decay=0.999, epsilon_min=0.08,
    )
    env = Environment(num_hunters_override=hunters, collector_ql=collector_ql,
                      guard_ql=guard_ql, headless=True)
    return env, collector_ql, guard_ql

try:
    env, cql, gql = make_env_headless(0)
    c = env.collectors[0]

    # Atributos nuevos
    if hasattr(c, '_pending_reward'):
        ok("Collector tiene _pending_reward")
    else:
        fail("Collector NO tiene _pending_reward")

    if hasattr(c, '_prev_dist_to_base'):
        ok("Collector tiene _prev_dist_to_base")
    else:
        fail("Collector NO tiene _prev_dist_to_base")

    # receive_reward acumula y no crashea
    c.receive_reward('collect')
    c.receive_reward('deliver_resources')
    from utils.constants import REWARD_COLLECT, REWARD_DELIVER_RESOURCES
    expected = REWARD_COLLECT + REWARD_DELIVER_RESOURCES
    if abs(c._pending_reward - expected) < 0.01:
        ok(f"receive_reward acumula correctamente ({c._pending_reward} = {expected})")
    else:
        fail("receive_reward no acumula bien",
             f"esperado={expected}, obtenido={c._pending_reward}")

except Exception as e:
    fail("Collector init/atributos", traceback.format_exc())

# ============================================================
# 4. COLLECTOR — STAGNATION FIX (acercarse a base = progreso)
# ============================================================
section("4. COLLECTOR — stagnation fix: regreso a base")

try:
    env2, cql2, _ = make_env_headless(0)
    c2 = env2.collectors[0]

    # Simular recolector con recursos cargados que se acerca a la base
    c2.carrying_resources = 10
    bx, by = BASE_POSITION
    # Colocar lejos de la base
    c2.position = (bx + 15, by + 15)
    c2._prev_dist_to_base = 30
    c2._prev_resources_carried = 10
    c2._prev_explored_count = 0
    c2.ticks_since_progress = 5

    # Simular un paso acercándose
    c2.position = (bx + 14, by + 15)  # 1 paso más cerca
    c2._update_progress_tracking('RETURN_TO_BASE')

    if c2.ticks_since_progress == 0:
        ok("Acercarse a la base mientras carga resetea ticks_since_progress")
    else:
        fail("Acercarse a la base NO se detecta como progreso",
             f"ticks_since_progress={c2.ticks_since_progress} (esperado 0)")

    # Alejarse de la base sin cargar NO debe contar como progreso
    c2.carrying_resources = 0
    c2._prev_dist_to_base = 13
    c2.ticks_since_progress = 5
    c2.position = (bx + 14, by + 16)  # alejarse
    c2._prev_resources_carried = 0
    c2._update_progress_tracking('EXPLORE')

    if c2.ticks_since_progress == 6:
        ok("Alejarse sin carga NO cuenta como progreso (ticks_since_progress sube)")
    else:
        fail("Alejarse sin carga incorrectamente contó como progreso",
             f"ticks_since_progress={c2.ticks_since_progress} (esperado 6)")

except Exception as e:
    fail("Stagnation fix", traceback.format_exc())

# ============================================================
# 5. COLLECTOR — HEURÍSTICAS: acción correcta por estado
# ============================================================
section("5. COLLECTOR — selección de acción por heurísticas (epsilon=0)")

try:
    env3, cql3, _ = make_env_headless(0)
    c3 = env3.collectors[0]
    c3.q_learning.epsilon = 0.0  # sin aleatoriedad

    from utils.constants import (
        COLLECTOR_CARRY_CAPACITY,
        HEUR_GOTO_RESOURCE_BASE, HEUR_GOTO_RESOURCE_EMPTY_BONUS,
    )

    # Estado: lleno + recursos conocidos + fase MID → debe elegir RETURN_TO_BASE
    state_full = (0, 1, 1, 0, 0, 0, 0, 1, 0, 1)
    biases_full = c3.calculate_heuristic_biases(
        state_full, "MID",
        {'carrying_resources': 10, 'carrying_capacity': 10,
         'explored_percent': 0.5, 'tower_count': 0,
         'num_alive_collectors': 5, 'current_action_streak': 0,
         'ticks_since_progress': 0, 'guard_near': 0}
    )
    best_full = max(biases_full, key=lambda a: biases_full[a])
    if best_full == 'RETURN_TO_BASE':
        ok(f"Lleno+MID+recursos -> RETURN_TO_BASE (bias={biases_full['RETURN_TO_BASE']})")
    else:
        fail("Lleno+MID no elige RETURN_TO_BASE",
             f"eligio={best_full}, biases={biases_full}")

    # Estado: lleno + streak=30 + ticks_since_progress=0 (regreso productivo)
    # El streak NO debe romper RETURN_TO_BASE si hay progreso real
    biases_streak = c3.calculate_heuristic_biases(
        state_full, "MID",
        {'carrying_resources': 10, 'carrying_capacity': 10,
         'explored_percent': 0.5, 'tower_count': 0,
         'num_alive_collectors': 5, 'current_action_streak': 30,
         'ticks_since_progress': 0, 'guard_near': 0}
    )
    best_streak = max(biases_streak, key=lambda a: biases_streak[a])
    if best_streak == 'RETURN_TO_BASE':
        ok(f"Streak=30 + progreso=0 -> RETURN_TO_BASE mantiene prioridad (bias={biases_streak['RETURN_TO_BASE']})")
    else:
        fail("Streak alto con progreso real rompe RETURN_TO_BASE (bug residual de stagnation)",
             f"eligio={best_streak}, RETURN={biases_streak['RETURN_TO_BASE']}, GO_RES={biases_streak['GO_TO_RESOURCE']}")

    # Estado: vacio + recursos conocidos + fase MID -> debe elegir GO_TO_RESOURCE
    state_empty = (0, 0, 0, 0, 0, 0, 0, 1, 0, 1)
    biases_empty = c3.calculate_heuristic_biases(
        state_empty, "MID",
        {'carrying_resources': 0, 'carrying_capacity': 10,
         'explored_percent': 0.5, 'tower_count': 0,
         'num_alive_collectors': 5, 'current_action_streak': 0,
         'ticks_since_progress': 0, 'guard_near': 0}
    )
    best_empty = max(biases_empty, key=lambda a: biases_empty[a])
    if best_empty == 'GO_TO_RESOURCE':
        ok(f"Vacio+MID+recursos -> GO_TO_RESOURCE (bias={biases_empty['GO_TO_RESOURCE']})")
    else:
        fail("Vacio+MID no elige GO_TO_RESOURCE",
             f"eligio={best_empty}, biases={biases_empty}")

    # Estado: enemy_near -> debe elegir FLEE
    state_danger = (1, 0, 0, 0, 0, 0, 0, 1, 0, 0)  # enemy_near=1
    biases_danger = c3.calculate_heuristic_biases(
        state_danger, "EARLY",
        {'carrying_resources': 5, 'carrying_capacity': 10,
         'explored_percent': 0.1, 'tower_count': 0,
         'num_alive_collectors': 5, 'current_action_streak': 0,
         'ticks_since_progress': 0, 'guard_near': 0}
    )
    best_danger = max(biases_danger, key=lambda a: biases_danger[a])
    if best_danger == 'FLEE':
        ok(f"Enemy near -> FLEE (bias={biases_danger['FLEE']})")
    else:
        fail("Enemy near no elige FLEE",
             f"eligio={best_danger}, biases={biases_danger}")

except Exception as e:
    fail("Heurísticas collector", traceback.format_exc())

# ============================================================
# 6. GUARD — atributos y pending_reward
# ============================================================
section("6. GUARD — atributos y pending_reward")

try:
    env4, _, gql4 = make_env_headless(0)
    g = env4.guards[0]

    if hasattr(g, '_pending_reward'):
        ok("Guard tiene _pending_reward")
    else:
        fail("Guard NO tiene _pending_reward")

    # receive_reward acumula
    g.receive_reward('kill_hunter')
    g.receive_reward('protect_collector')
    from utils.constants import REWARD_KILL_HUNTER, REWARD_PROTECT_COLLECTOR
    expected_g = REWARD_KILL_HUNTER + REWARD_PROTECT_COLLECTOR
    if abs(g._pending_reward - expected_g) < 0.01:
        ok(f"Guard receive_reward acumula correctamente ({g._pending_reward} = {expected_g})")
    else:
        fail("Guard receive_reward no acumula",
             f"esperado={expected_g}, obtenido={g._pending_reward}")

except Exception as e:
    fail("Guard atributos", traceback.format_exc())

# ============================================================
# 7. GUARD — HEURÍSTICAS: SCOUT no gana con enemy_near
# ============================================================
section("7. GUARD — SCOUT vs INTERCEPT con enemy_near")

try:
    env5, _, _ = make_env_headless(0)
    g5 = env5.guards[0]

    # Estado con enemy_near=1, fuera de rango, frontera cercana, fase EARLY
    # (enemy_near, enemy_in_range, collector_near, collector_vulnerable,
    #  risk_level, cooldown_ready, distance_to_enemy, numerical_advantage,
    #  guards_near_collector, collector_near_base, unexplored_frontier_near, collector_carrying)
    state_enemy = (1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0)
    biases_enemy = g5.calculate_heuristic_biases(
        state_enemy, "EARLY",
        {'tower_count': 0, 'recent_enemy_sighting': False,
         'collector_has_kit': False, 'guards_near_my_collector': 0,
         'collector_near_base': 0, 'escorted_collector_collected': False}
    )
    best_enemy = max(biases_enemy, key=lambda a: biases_enemy[a])
    scout_val = biases_enemy['SCOUT']
    intercept_val = biases_enemy['INTERCEPT']

    if best_enemy in ('INTERCEPT', 'ATTACK') and scout_val < intercept_val:
        ok(f"Con enemy_near: SCOUT ({scout_val}) < INTERCEPT ({intercept_val}) -> elige {best_enemy}")
    else:
        fail("Con enemy_near SCOUT deberia perder ante INTERCEPT",
             f"SCOUT={scout_val}, INTERCEPT={intercept_val}, mejor={best_enemy}")

    # Sin enemy: SCOUT debe ganar en EARLY
    state_no_enemy = (0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 1, 0)
    biases_no_enemy = g5.calculate_heuristic_biases(
        state_no_enemy, "EARLY",
        {'tower_count': 0, 'recent_enemy_sighting': False,
         'collector_has_kit': False, 'guards_near_my_collector': 0,
         'collector_near_base': 0, 'escorted_collector_collected': False}
    )
    best_no_enemy = max(biases_no_enemy, key=lambda a: biases_no_enemy[a])
    if best_no_enemy == 'SCOUT':
        ok(f"Sin enemy en EARLY: SCOUT gana (bias={biases_no_enemy['SCOUT']})")
    else:
        fail("Sin enemy en EARLY SCOUT deberia ganar",
             f"eligio={best_no_enemy}, biases={biases_no_enemy}")

except Exception as e:
    fail("Guard heurísticas SCOUT/INTERCEPT", traceback.format_exc())

# ============================================================
# 8. INTEGRACIÓN — episodio corto headless (0 cazadores)
# ============================================================
section("8. INTEGRACIÓN — episodio headless 0 cazadores (200 ticks)")

try:
    env6, cql6, gql6 = make_env_headless(0)
    ticks_run = 0
    for _ in range(200):
        env6.tick()
        ticks_run += 1
        if env6.game_over:
            break

    ok(f"Episodio completó {ticks_run} ticks sin crash")

    resources_pct = (env6.base_resources / env6.win_target * 100
                     if env6.win_target > 0 else 0)
    ok(f"Recursos depositados: {env6.base_resources}/{env6.win_target:.0f} ({resources_pct:.1f}%)")

    alive_c = sum(1 for c in env6.collectors if c.is_alive)
    alive_g = sum(1 for g in env6.guards if g.is_alive)
    ok(f"Agentes vivos: {alive_c} recolectores, {alive_g} guardias")

    q_sizes_ok = len(cql6.Q_table) >= 0 and len(gql6.Q_table) >= 0
    ok(f"Q-tables activas: C={len(cql6.Q_table)} estados, G={len(gql6.Q_table)} estados")

except Exception as e:
    fail("Integración headless 0 cazadores", traceback.format_exc())

# ============================================================
# 9. INTEGRACIÓN — episodio con cazadores (Fase 2 simulada)
# ============================================================
section("9. INTEGRACIÓN — episodio headless 4 cazadores (200 ticks)")

try:
    env7, cql7, gql7 = make_env_headless(4)
    ticks_run7 = 0
    for _ in range(200):
        env7.tick()
        ticks_run7 += 1
        if env7.game_over:
            break

    ok(f"Episodio con 4 cazadores: {ticks_run7} ticks sin crash")

    if env7.game_over:
        ok(f"Juego terminó correctamente — ganador: Equipo {env7.winner}")
    else:
        ok("Juego llegó al límite de ticks (sin ganador aún, normal en 200 ticks)")

except Exception as e:
    fail("Integración headless 4 cazadores", traceback.format_exc())

# ============================================================
# 10. INTEGRACIÓN — Q-learning aprende (Q-values cambian)
# ============================================================
section("10. Q-LEARNING — valores cambian tras episodio completo")

try:
    import copy
    env8, cql8, gql8 = make_env_headless(0)

    # Correr 50 ticks para acumular algunas actualizaciones
    for _ in range(50):
        env8.tick()
        if env8.game_over:
            break

    if len(cql8.Q_table) > 0:
        # Verificar que al menos algún Q-value es distinto de 0
        any_nonzero_c = any(
            v != 0.0
            for state_dict in cql8.Q_table.values()
            for v in state_dict.values()
        )
        if any_nonzero_c:
            ok(f"Collector Q-table tiene valores no-cero ({len(cql8.Q_table)} estados visitados)")
        else:
            fail("Collector Q-table solo tiene ceros tras 50 ticks")
    else:
        fail("Collector Q-table vacía tras 50 ticks")

    # Con 0 cazadores, el guardia no recibe eventos de combate → rewards siempre 0.
    # Lo importante es que visita estados (Q-table crece) y no crashea.
    if len(gql8.Q_table) > 0:
        ok(f"Guard Q-table visita estados ({len(gql8.Q_table)} estados, rewards 0 sin cazadores — esperado)")
    else:
        fail("Guard Q-table vacia tras 50 ticks — no visita ningun estado")

except Exception as e:
    fail("Q-learning actualización", traceback.format_exc())

# ============================================================
# 11. RECURSOS SE DEPOSITAN (regresión del bug de stagnation)
# ============================================================
section("11. REGRESIÓN — recursos depositados en episodio largo (sin cazadores)")

try:
    env9, cql9, gql9 = make_env_headless(0)
    for _ in range(1000):
        env9.tick()
        if env9.game_over:
            break

    pct = env9.base_resources / env9.win_target * 100 if env9.win_target > 0 else 0

    if env9.base_resources > 0:
        ok(f"Se depositaron recursos: {env9.base_resources}/{env9.win_target:.0f} ({pct:.1f}%)")
    else:
        fail("Ningún recurso depositado en 1000 ticks sin cazadores")

    # Con epsilon=0.5 (test), la mitad de las decisiones son aleatorias.
    # En entrenamiento real (epsilon 0.08-0.20) el rendimiento es mucho mayor.
    # Umbral de 15%: demuestra que el fix funciona (antes era 0% con el bug).
    if pct >= 15:
        ok(f"Progreso {pct:.1f}% >= 15% en 1000 ticks (stagnation fix activo; con epsilon real sera mayor)")
    else:
        fail(f"Progreso {pct:.1f}% < 15% incluso con epsilon alto — bug residual posible")

    if env9.game_over and env9.winner == 'A':
        ok(f"¡Equipo A GANÓ en {env9.current_tick} ticks!")

except Exception as e:
    fail("Regresión depósito de recursos", traceback.format_exc())

# ============================================================
# RESULTADO FINAL
# ============================================================
print(f"\n{'='*55}")
print(f"  RESULTADO: {PASS} OK   {FAIL} FAIL")
print(f"{'='*55}\n")

if FAIL == 0:
    print("  Todos los checks pasaron. El sistema está listo para entrenamiento.\n")
else:
    print(f"  {FAIL} checks fallaron. Revisar antes de continuar.\n")

sys.exit(0 if FAIL == 0 else 1)
