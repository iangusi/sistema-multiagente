"""
Tests de verificación del sistema multi-agente.
Valida imports, constantes, atributos de agentes, heurísticas e integración headless.
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
        HEUR_FLEE_HUNTER, HEUR_RETURN_FULL, HEUR_BUILD_HAS_KIT,
        HEUR_GOTO_RESOURCE_BASE,
        HEUR_GUARD_ATTACK, HEUR_GUARD_FLEE_DANGER,
        HEUR_GUARD_DEFEND_ALLY, HEUR_GUARD_EXPLORE_BASE,
        NUM_COLLECTORS, NUM_GUARDS,
        BASE_POSITION,
        TRAINING_MAX_TICKS_PER_EPISODE,
        REWARD_COLLECT, REWARD_DELIVER_RESOURCES,
        REWARD_KILL_HUNTER,
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
# 2. CONSTANTES — JERARQUÍA DE HEURÍSTICAS
# ============================================================
section("2. JERARQUÍA DE HEURÍSTICAS (constants)")

# FLEE debe dominar sobre RETURN_TO_BASE y GO_TO_RESOURCE
if HEUR_FLEE_HUNTER > HEUR_RETURN_FULL:
    ok(f"FLEE ({HEUR_FLEE_HUNTER}) > RETURN_FULL ({HEUR_RETURN_FULL})")
else:
    fail("FLEE debe superar RETURN_FULL",
         f"FLEE={HEUR_FLEE_HUNTER}, RETURN_FULL={HEUR_RETURN_FULL}")

if HEUR_FLEE_HUNTER > HEUR_GOTO_RESOURCE_BASE:
    ok(f"FLEE ({HEUR_FLEE_HUNTER}) > GO_TO_RESOURCE ({HEUR_GOTO_RESOURCE_BASE})")
else:
    fail("FLEE debe superar GO_TO_RESOURCE",
         f"FLEE={HEUR_FLEE_HUNTER}, GO_RESOURCE={HEUR_GOTO_RESOURCE_BASE}")

# BUILD_HAS_KIT debe superar RETURN_FULL (kit = oportunidad táctica)
if HEUR_BUILD_HAS_KIT > HEUR_RETURN_FULL:
    ok(f"BUILD_HAS_KIT ({HEUR_BUILD_HAS_KIT}) > RETURN_FULL ({HEUR_RETURN_FULL})")
else:
    fail("BUILD_HAS_KIT debe superar RETURN_FULL",
         f"BUILD_HAS_KIT={HEUR_BUILD_HAS_KIT}, RETURN_FULL={HEUR_RETURN_FULL}")

# Guardia: ATTACK domina sobre FLEE
if HEUR_GUARD_ATTACK > HEUR_GUARD_FLEE_DANGER:
    ok(f"Guard ATTACK ({HEUR_GUARD_ATTACK}) > FLEE ({HEUR_GUARD_FLEE_DANGER})")
else:
    fail("Guard ATTACK debe superar FLEE",
         f"ATTACK={HEUR_GUARD_ATTACK}, FLEE={HEUR_GUARD_FLEE_DANGER}")

# Guardia: DEFEND_ALLY > EXPLORE (base)
if HEUR_GUARD_DEFEND_ALLY > HEUR_GUARD_EXPLORE_BASE:
    ok(f"Guard DEFEND ({HEUR_GUARD_DEFEND_ALLY}) > EXPLORE ({HEUR_GUARD_EXPLORE_BASE})")
else:
    fail("Guard DEFEND debe superar EXPLORE",
         f"DEFEND={HEUR_GUARD_DEFEND_ALLY}, EXPLORE={HEUR_GUARD_EXPLORE_BASE}")

# ============================================================
# HELPER: crea entorno headless con QLearning propio
# ============================================================

def make_env_headless(hunters=0):
    collector_ql = QLearning(
        actions=['EXPLORE', 'GO_TO_RESOURCE', 'RETURN_TO_BASE', 'FLEE', 'BUILD_TOWER'],
        alpha=0.2, gamma=0.9, epsilon=0.5, epsilon_decay=0.999, epsilon_min=0.08,
    )
    guard_ql = QLearning(
        actions=['EXPLORE', 'ATTACK', 'DEFEND', 'FLEE'],
        alpha=0.2, gamma=0.9, epsilon=0.5, epsilon_decay=0.999, epsilon_min=0.08,
    )
    env = Environment(
        num_hunters_override=hunters,
        collector_ql=collector_ql,
        guard_ql=guard_ql,
        headless=True,
    )
    return env, collector_ql, guard_ql

# ============================================================
# 3. COLLECTOR — atributos básicos y receive_reward
# ============================================================
section("3. COLLECTOR — atributos y pending_reward")

try:
    env, cql, gql = make_env_headless(0)
    c = env.collectors[0]

    if hasattr(c, '_pending_reward'):
        ok("Collector tiene _pending_reward")
    else:
        fail("Collector NO tiene _pending_reward")

    if hasattr(c, 'is_alive') and c.is_alive:
        ok("Collector nace vivo")
    else:
        fail("Collector no tiene is_alive o nace muerto")

    if hasattr(c, 'carrying_resources') and c.carrying_resources == 0:
        ok("Collector inicia sin recursos")
    else:
        fail("Collector no tiene carrying_resources o inicia con recursos")

    # receive_reward acumula sin crashear
    c.receive_reward('collect')
    c.receive_reward('deliver_resources')
    expected = REWARD_COLLECT + REWARD_DELIVER_RESOURCES
    if abs(c._pending_reward - expected) < 0.01:
        ok(f"receive_reward acumula correctamente ({c._pending_reward} == {expected})")
    else:
        fail("receive_reward no acumula bien",
             f"esperado={expected}, obtenido={c._pending_reward}")

    # Evento desconocido devuelve 0
    before = c._pending_reward
    c.receive_reward('evento_inexistente')
    if abs(c._pending_reward - before) < 0.01:
        ok("Evento desconocido en receive_reward no modifica pending_reward")
    else:
        fail("Evento desconocido modificó pending_reward incorrectamente")

except Exception as e:
    fail("Collector atributos/receive_reward", traceback.format_exc())

# ============================================================
# 4. GUARD — atributos básicos y receive_reward
# ============================================================
section("4. GUARD — atributos y pending_reward")

try:
    env2, _, gql2 = make_env_headless(0)
    g = env2.guards[0]

    if hasattr(g, '_pending_reward'):
        ok("Guard tiene _pending_reward")
    else:
        fail("Guard NO tiene _pending_reward")

    if hasattr(g, 'is_alive') and g.is_alive:
        ok("Guard nace vivo")
    else:
        fail("Guard no tiene is_alive o nace muerto")

    # receive_reward acumula
    before = g._pending_reward
    g.receive_reward('kill_hunter')
    if g._pending_reward >= before + REWARD_KILL_HUNTER:
        ok(f"Guard receive_reward kill_hunter acumula (delta >= {REWARD_KILL_HUNTER})")
    else:
        fail("Guard receive_reward kill_hunter no acumula",
             f"antes={before}, despues={g._pending_reward}")

except Exception as e:
    fail("Guard atributos/receive_reward", traceback.format_exc())

# ============================================================
# 5. COLLECTOR — heurísticas correctas según estado
# ============================================================
section("5. COLLECTOR — selección de acción por heurísticas")

try:
    env3, cql3, _ = make_env_headless(0)
    c3 = env3.collectors[0]

    # Estado: hunter_near=1 -> FLEE debe dominar
    # (build_dist, explore_dist, resource_dist, can_carry, hunter_near)
    state_danger = (0, 1, 1, 1, 1)
    biases_danger = c3.calculate_heuristic_biases(state_danger, [])
    best = max(biases_danger, key=lambda a: biases_danger[a])
    if best == 'FLEE':
        ok(f"hunter_near=1 -> FLEE (bias={biases_danger['FLEE']})")
    else:
        fail("hunter_near=1 no elige FLEE",
             f"eligio={best}, biases={biases_danger}")

    # Estado: inventario lleno (can_carry=0), sin hunter -> RETURN_TO_BASE
    state_full = (0, 1, 1, 0, 0)
    biases_full = c3.calculate_heuristic_biases(state_full, [])
    best_full = max(biases_full, key=lambda a: biases_full[a])
    if best_full == 'RETURN_TO_BASE':
        ok(f"can_carry=0, sin hunter -> RETURN_TO_BASE (bias={biases_full['RETURN_TO_BASE']})")
    else:
        fail("can_carry=0 no elige RETURN_TO_BASE",
             f"eligio={best_full}, biases={biases_full}")

    # Estado: vacío, recurso cerca (can_carry=1, resource_dist=1, sin hunter) -> GO_TO_RESOURCE
    state_empty = (0, 1, 1, 1, 0)
    biases_empty = c3.calculate_heuristic_biases(state_empty, [])
    best_empty = max(biases_empty, key=lambda a: biases_empty[a])
    if best_empty == 'GO_TO_RESOURCE':
        ok(f"can_carry=1, resource_dist=1 -> GO_TO_RESOURCE (bias={biases_empty['GO_TO_RESOURCE']})")
    else:
        fail("vacío+recurso cercano no elige GO_TO_RESOURCE",
             f"eligio={best_empty}, biases={biases_empty}")

    # Estado: tiene kit (build_dist=1), vacío, sin hunter -> BUILD_TOWER
    state_kit = (1, 1, 0, 1, 0)
    biases_kit = c3.calculate_heuristic_biases(state_kit, [])
    best_kit = max(biases_kit, key=lambda a: biases_kit[a])
    if best_kit == 'BUILD_TOWER':
        ok(f"build_dist=1, sin hunter -> BUILD_TOWER (bias={biases_kit['BUILD_TOWER']})")
    else:
        fail("tiene kit y no elige BUILD_TOWER",
             f"eligio={best_kit}, biases={biases_kit}")

except Exception as e:
    fail("Heurísticas collector", traceback.format_exc())

# ============================================================
# 6. GUARD — heurísticas correctas según estado
# ============================================================
section("6. GUARD — selección de acción por heurísticas")

try:
    env4, _, _ = make_env_headless(0)
    g4 = env4.guards[0]

    # (ally_in_danger, i_am_in_danger, hunter_near, can_attack)

    # can_attack=1 -> ATTACK domina
    state_attack = (0, 0, 1, 1)
    biases_attack = g4.calculate_heuristic_biases(state_attack)
    best_attack = max(biases_attack, key=lambda a: biases_attack[a])
    if best_attack == 'ATTACK':
        ok(f"can_attack=1 -> ATTACK (bias={biases_attack['ATTACK']})")
    else:
        fail("can_attack=1 no elige ATTACK",
             f"eligio={best_attack}, biases={biases_attack}")

    # i_am_in_danger=1, can_attack=0 -> FLEE
    state_flee = (0, 1, 1, 0)
    biases_flee = g4.calculate_heuristic_biases(state_flee)
    best_flee = max(biases_flee, key=lambda a: biases_flee[a])
    if best_flee == 'FLEE':
        ok(f"i_am_in_danger=1, can_attack=0 -> FLEE (bias={biases_flee['FLEE']})")
    else:
        fail("i_am_in_danger=1 no elige FLEE",
             f"eligio={best_flee}, biases={biases_flee}")

    # ally_in_danger=1, i_am_in_danger=0, can_attack=0 -> DEFEND
    state_defend = (1, 0, 0, 0)
    biases_defend = g4.calculate_heuristic_biases(state_defend)
    best_defend = max(biases_defend, key=lambda a: biases_defend[a])
    if best_defend == 'DEFEND':
        ok(f"ally_in_danger=1, sin peligro propio -> DEFEND (bias={biases_defend['DEFEND']})")
    else:
        fail("ally_in_danger=1 sin peligro propio no elige DEFEND",
             f"eligio={best_defend}, biases={biases_defend}")

    # Sin peligro ni cazador -> EXPLORE (comportamiento base)
    state_idle = (0, 0, 0, 0)
    biases_idle = g4.calculate_heuristic_biases(state_idle)
    best_idle = max(biases_idle, key=lambda a: biases_idle[a])
    if best_idle == 'EXPLORE':
        ok(f"sin amenazas -> EXPLORE (bias={biases_idle['EXPLORE']})")
    else:
        fail("sin amenazas no elige EXPLORE",
             f"eligio={best_idle}, biases={biases_idle}")

except Exception as e:
    fail("Heurísticas guard", traceback.format_exc())

# ============================================================
# 7. INTEGRACIÓN — episodio corto headless (0 cazadores)
# ============================================================
section("7. INTEGRACIÓN — episodio headless 0 cazadores (200 ticks)")

try:
    env5, cql5, gql5 = make_env_headless(0)
    ticks_run = 0
    for _ in range(200):
        env5.tick()
        ticks_run += 1
        if env5.game_over:
            break

    ok(f"Episodio completó {ticks_run} ticks sin crash")

    alive_c = sum(1 for c in env5.collectors if c.is_alive)
    alive_g = sum(1 for g in env5.guards if g.is_alive)
    ok(f"Agentes vivos: {alive_c} recolectores, {alive_g} guardias")

    ok(f"Q-tables activas: C={len(cql5.Q_table)} estados, G={len(gql5.Q_table)} estados")

except Exception as e:
    fail("Integración headless 0 cazadores", traceback.format_exc())

# ============================================================
# 8. INTEGRACIÓN — episodio con cazadores
# ============================================================
section("8. INTEGRACIÓN — episodio headless 4 cazadores (200 ticks)")

try:
    env6, cql6, gql6 = make_env_headless(4)
    ticks_run6 = 0
    for _ in range(200):
        env6.tick()
        ticks_run6 += 1
        if env6.game_over:
            break

    ok(f"Episodio con 4 cazadores: {ticks_run6} ticks sin crash")

    if env6.game_over:
        ok(f"Juego terminó correctamente — ganador: Equipo {env6.winner}")
    else:
        ok("Juego llegó al límite de ticks sin ganador (normal en 200 ticks)")

except Exception as e:
    fail("Integración headless 4 cazadores", traceback.format_exc())

# ============================================================
# 9. Q-LEARNING — valores cambian tras episodio
# ============================================================
section("9. Q-LEARNING — Q-values se actualizan tras 50 ticks")

try:
    env7, cql7, gql7 = make_env_headless(0)

    for _ in range(50):
        env7.tick()
        if env7.game_over:
            break

    if len(cql7.Q_table) > 0:
        any_nonzero_c = any(
            v != 0.0
            for state_dict in cql7.Q_table.values()
            for v in state_dict.values()
        )
        if any_nonzero_c:
            ok(f"Collector Q-table tiene valores no-cero ({len(cql7.Q_table)} estados)")
        else:
            fail("Collector Q-table solo tiene ceros tras 50 ticks")
    else:
        fail("Collector Q-table vacía tras 50 ticks")

    if len(gql7.Q_table) > 0:
        ok(f"Guard Q-table visita estados ({len(gql7.Q_table)} estados)")
    else:
        fail("Guard Q-table vacía tras 50 ticks — no visita ningún estado")

except Exception as e:
    fail("Q-learning actualización", traceback.format_exc())

# ============================================================
# 10. RECURSOS SE DEPOSITAN (regresión: los recolectores no se atascan)
# ============================================================
section("10. REGRESIÓN — recursos depositados en episodio largo (sin cazadores)")

try:
    env8, cql8, gql8 = make_env_headless(0)
    for _ in range(1000):
        env8.tick()
        if env8.game_over:
            break

    pct = (env8.base_resources / env8.win_target * 100
           if env8.win_target > 0 else 0)

    if env8.base_resources > 0:
        ok(f"Se depositaron recursos: {env8.base_resources}/{env8.win_target:.0f} ({pct:.1f}%)")
    else:
        fail("Ningún recurso depositado en 1000 ticks sin cazadores")

    # Con epsilon=0.5 (exploratorio) el progreso mínimo esperado es 15%.
    # En entrenamiento real (epsilon bajo) es mucho mayor.
    if pct >= 15:
        ok(f"Progreso {pct:.1f}% >= 15% en 1000 ticks")
    else:
        fail(f"Progreso {pct:.1f}% < 15% — posible bug de estancamiento",
             "Los recolectores no están depositando recursos en la base")

    if env8.game_over and env8.winner == 'A':
        ok(f"¡Equipo A ganó en {env8.current_tick} ticks!")

except Exception as e:
    fail("Regresión depósito de recursos", traceback.format_exc())

# ============================================================
# 11. ESTADO — dimensiones correctas del vector de estado
# ============================================================
section("11. DIMENSIÓN DEL VECTOR DE ESTADO")

try:
    env9, _, _ = make_env_headless(0)
    # Correr 1 tick para que los agentes tengan prev_state
    env9.tick()

    c9 = env9.collectors[0]
    if c9.prev_state is not None:
        if len(c9.prev_state) == 5:
            ok(f"Collector state tiene 5 variables: {c9.prev_state}")
        else:
            fail(f"Collector state debería tener 5 variables, tiene {len(c9.prev_state)}",
                 str(c9.prev_state))
    else:
        fail("Collector no generó prev_state tras 1 tick")

    g9 = env9.guards[0]
    if g9.prev_state is not None:
        if len(g9.prev_state) == 4:
            ok(f"Guard state tiene 4 variables: {g9.prev_state}")
        else:
            fail(f"Guard state debería tener 4 variables, tiene {len(g9.prev_state)}",
                 str(g9.prev_state))
    else:
        fail("Guard no generó prev_state tras 1 tick")

except Exception as e:
    fail("Dimensión del estado", traceback.format_exc())

# ============================================================
# 12. MUERTE — agente muerto no actúa
# ============================================================
section("12. MUERTE — agente muerto no modifica estado")

try:
    env10, _, _ = make_env_headless(0)
    c10 = env10.collectors[0]

    pos_before = c10.position
    c10.die()

    if not c10.is_alive:
        ok("die() marca is_alive=False")
    else:
        fail("die() no marcó is_alive=False")

    if c10.carrying_resources == 0:
        ok("die() limpia carrying_resources")
    else:
        fail(f"die() no limpió carrying_resources: {c10.carrying_resources}")

    # Ejecutar algunos ticks y verificar que el agente muerto no se mueve
    for _ in range(5):
        env10.tick()

    if c10.position == pos_before:
        ok("Agente muerto no cambia de posición")
    else:
        fail("Agente muerto cambió de posición",
             f"antes={pos_before}, después={c10.position}")

except Exception as e:
    fail("Muerte del agente", traceback.format_exc())

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
