from utils.constants import (
    GUARD_HP,
    GUARD_SPEED,
    GUARD_VISION,
    GUARD_ATTACK_RANGE,
    GUARD_ATTACK_COOLDOWN,
    GUARD_DAMAGE,
    TOWER_ATTACK_RANGE,
    BASE_POSITION,
    MAP_WIDTH,
    MAP_HEIGHT,
    PHASE_EARLY_THRESHOLD,
    PHASE_MID_THRESHOLD,
    HEUR_ESCORT_VULNERABLE,
    HEUR_ESCORT_HAS_KIT,
    HEUR_ESCORT_REDUNDANT,
    HEUR_ESCORT_COLLECTOR_AT_BASE,
    HEUR_ATTACK_IN_RANGE,
    HEUR_ATTACK_ADVANTAGE,
    HEUR_ATTACK_DISADVANTAGE,
    HEUR_INTERCEPT_ENEMY_NEAR,
    HEUR_INTERCEPT_COLLECTOR_DANGER,
    HEUR_SCOUT_LOW_EXPLORATION,
    HEUR_SCOUT_EARLY_BONUS,
    HEUR_SCOUT_NO_ESCORT_NEEDED,
    HEUR_DEFEND_HAS_TOWERS,
    HEUR_DEFEND_HIGH_RISK,
    HEUR_INVESTIGATE_RECENT,
    HEUR_PATROL_BASE,
    REWARD_KILL_HUNTER,
    REWARD_PROTECT_COLLECTOR,
    REWARD_INTERCEPT,
    REWARD_DEFEND_ZONE,
    REWARD_GUARD_DIE,
    REWARD_COLLECTOR_DIES_NEARBY,
    REWARD_COLLECTOR_DIES_WHILE_ADJACENT,
    REWARD_BAD_DECISION,
    REWARD_SOLE_ESCORT,
    REWARD_REDUNDANT_ESCORT,
    REWARD_COLLECTOR_SAFE_NO_ESCORT,
    REWARD_SCOUT_NEW_CELLS,
    REWARD_SCOUT_FIND_RESOURCE,
    REWARD_ESCORT_COLLECTOR_COLLECTS,
    REWARD_GUARD_IDLE,
    RISK_UNEXPLORED,
)
from pathfinding.astar import find_path

# Acciones disponibles del guardia (con SCOUT agregado)
ACTIONS = ['PATROL', 'ESCORT', 'ATTACK', 'INTERCEPT', 'DEFEND_ZONE', 'INVESTIGATE', 'SCOUT']

# Umbrales de distancia al enemigo
_DIST_NEAR = 3
_DIST_MID  = 6

# Umbrales de riesgo
_RISK_LOW  = 0.3
_RISK_HIGH = 0.7


class Guard:
    """
    Guardia del Equipo A.

    Toma decisiones mediante Q-learning compartido (tabla separada de recolectores)
    + heurísticas. Sin asignación centralizada de roles: el guardia decide solo.
    Estado RL expandido: 12 variables.
    Acción SCOUT añadida para exploración activa.
    """

    def __init__(self, position, shared_data, q_learning):
        # Posición y stats
        self.position        = position
        self.hp              = GUARD_HP
        self.current_hp      = GUARD_HP
        self.speed           = GUARD_SPEED
        self.vision_range    = GUARD_VISION
        self.attack_range    = GUARD_ATTACK_RANGE
        self.attack_cooldown = GUARD_ATTACK_COOLDOWN
        self.damage          = GUARD_DAMAGE
        self.current_cooldown = 0
        self.is_alive        = True

        # Memoria compartida (referencias)
        self.shared_data         = shared_data
        self.known_map           = shared_data['known_map']
        self.discovered_resources = shared_data['discovered_resources']
        self.last_seen_enemies   = shared_data['last_seen_enemies']
        self.risk_map            = shared_data['risk_map']

        # Q-learning compartido con todos los guardias
        self.q_learning = q_learning

        # Estado interno
        self.current_action = 'PATROL'
        self.current_target = None
        self.current_path   = []
        self.prev_state     = None
        self.prev_action    = None

        # Posición siguiente calculada en decide()
        self.next_position  = position

        # Referencia al grid actual (actualizada al inicio de decide)
        self._last_grid     = None

        # Para execute_patrol
        self._patrol_waypoints = []
        self._patrol_idx       = 0

        # Referencia temporal a todos los colectores (actualizada en decide)
        self._all_collectors = []

        # Recompensas de eventos pendientes (aplicadas en el siguiente decide())
        self._pending_reward = 0.0

    # ===================================================================
    # PERCEPCIÓN
    # ===================================================================

    def perceive(self, grid, current_tick):
        """
        Observa celdas dentro de vision_range=3.
        Actualiza known_map y last_seen_enemies.
        Retorna (visible_enemies, visible_allies, visible_collectors).
        """
        px, py = self.position
        visible_enemies    = []
        visible_allies     = []
        visible_collectors = []

        for dx in range(-self.vision_range, self.vision_range + 1):
            for dy in range(-self.vision_range, self.vision_range + 1):
                if abs(dx) + abs(dy) > self.vision_range:
                    continue
                nx, ny = px + dx, py + dy
                if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                    continue

                cell = grid.cells[nx][ny]

                self.known_map[nx][ny]['explored']       = True
                self.known_map[nx][ny]['last_seen']       = current_tick
                self.known_map[nx][ny]['last_known_type'] = cell.type

                for agent in cell.agents:
                    if not agent.is_alive:
                        continue
                    atype = agent.__class__.__name__
                    if atype == 'Hunter':
                        visible_enemies.append(agent)
                        entry = next(
                            (e for e in self.last_seen_enemies
                             if e.get('agent') is agent),
                            None
                        )
                        if entry:
                            entry['position'] = agent.position
                            entry['tick']     = current_tick
                        else:
                            self.last_seen_enemies.append({
                                'agent':    agent,
                                'position': agent.position,
                                'type':     'Hunter',
                                'tick':     current_tick,
                            })
                    elif atype == 'Collector':
                        visible_collectors.append(agent)
                        visible_allies.append(agent)
                    elif atype == 'Guard' and agent is not self:
                        visible_allies.append(agent)

                if cell.tower is not None:
                    visible_allies.append(cell.tower)

        return visible_enemies, visible_allies, visible_collectors

    # ===================================================================
    # ESTADO RL (12 variables)
    # ===================================================================

    def build_state(self, visible_enemies, visible_allies, visible_collectors):
        """
        Tupla discreta de estado (12 variables):
          (enemy_near, enemy_in_range, collector_near, collector_vulnerable,
           risk_level, cooldown_ready, distance_to_enemy, numerical_advantage,
           guards_near_collector, collector_near_base, unexplored_frontier_near,
           collector_carrying)

        risk_level:           0=LOW, 1=MED, 2=HIGH
        distance_to_enemy:    0=NEAR, 1=MID, 2=FAR
        numerical_advantage:  0=LOW, 1=EVEN, 2=HIGH
        guards_near_collector: 0, 1, 2+ (capped)
        """
        px, py = self.position

        enemy_near     = 1 if visible_enemies else 0
        enemy_in_range = int(
            any(
                abs(e.position[0] - px) + abs(e.position[1] - py) <= self.attack_range
                for e in visible_enemies
            )
        )
        collector_near = 1 if visible_collectors else 0

        # collector_vulnerable: recolector sin guardia NI torre cerca
        guards_in_allies = [a for a in visible_allies if a.__class__.__name__ == 'Guard']
        towers_in_allies = [a for a in visible_allies if a.__class__.__name__ == 'Tower']
        collector_vulnerable = 0
        for c in visible_collectors:
            cx, cy = c.position
            guarded = any(
                abs(g.position[0] - cx) + abs(g.position[1] - cy) <= GUARD_ATTACK_RANGE
                for g in guards_in_allies if g is not self
            )
            towered = any(
                abs(t.position[0] - cx) + abs(t.position[1] - cy) <= TOWER_ATTACK_RANGE
                for t in towers_in_allies
            )
            if not guarded and not towered and visible_enemies:
                collector_vulnerable = 1
                break

        r = self.risk_map[px][py]
        if r < _RISK_LOW:
            risk_level = 0
        elif r < _RISK_HIGH:
            risk_level = 1
        else:
            risk_level = 2

        cooldown_ready = 1 if self.current_cooldown == 0 else 0

        if visible_enemies:
            min_dist = min(
                abs(e.position[0] - px) + abs(e.position[1] - py)
                for e in visible_enemies
            )
            if min_dist <= _DIST_NEAR:
                distance_to_enemy = 0
            elif min_dist <= _DIST_MID:
                distance_to_enemy = 1
            else:
                distance_to_enemy = 2
        else:
            distance_to_enemy = 2

        n_enemies = len(visible_enemies)
        n_allies  = len([a for a in visible_allies if a.__class__.__name__ != 'Tower'])
        if n_enemies == 0:
            numerical_advantage = 2
        else:
            ratio = n_allies / n_enemies
            if ratio < 0.9:
                numerical_advantage = 0
            elif ratio > 1.1:
                numerical_advantage = 2
            else:
                numerical_advantage = 1

        # NUEVAS variables
        nearest_collector = self._nearest_alive_collector(visible_collectors)
        if nearest_collector:
            guards_near = sum(
                1 for g in visible_allies
                if g is not self
                and g.__class__.__name__ == 'Guard'
                and (abs(g.position[0] - nearest_collector.position[0])
                     + abs(g.position[1] - nearest_collector.position[1])) <= 5
            )
            guards_near_collector = min(guards_near, 2)

            dist_coll_base = (abs(nearest_collector.position[0] - BASE_POSITION[0])
                              + abs(nearest_collector.position[1] - BASE_POSITION[1]))
            collector_near_base = 1 if dist_coll_base <= 5 else 0
            collector_carrying  = 1 if nearest_collector.carrying_resources > 0 else 0
        else:
            guards_near_collector = 0
            collector_near_base   = 0
            collector_carrying    = 0

        unexplored_frontier_near = self._has_frontier_nearby()

        return (
            enemy_near, enemy_in_range, collector_near, collector_vulnerable,
            risk_level, cooldown_ready, distance_to_enemy, numerical_advantage,
            guards_near_collector, collector_near_base, unexplored_frontier_near,
            collector_carrying
        )

    # ===================================================================
    # HEURISTIC BIASES
    # ===================================================================

    def calculate_heuristic_biases(self, state, phase, game_context):
        """
        Retorna dict {action: float} con sesgos heurísticos.
        Los guardias deciden solos su rol — sin asignación centralizada.
        """
        (enemy_near, enemy_in_range, collector_near, collector_vulnerable,
         risk_level, cooldown_ready, distance_to_enemy, numerical_advantage,
         guards_near_collector, collector_near_base, unexplored_frontier_near,
         collector_carrying) = state

        biases = {a: 0.0 for a in ACTIONS}

        # -- ATTACK (emergencia — ataca si puede) -----------------------
        if enemy_in_range and cooldown_ready:
            biases['ATTACK'] += HEUR_ATTACK_IN_RANGE
        if numerical_advantage == 2:
            biases['ATTACK'] += HEUR_ATTACK_ADVANTAGE
        elif numerical_advantage == 0:
            biases['ATTACK'] += HEUR_ATTACK_DISADVANTAGE

        # -- INTERCEPT --------------------------------------------------
        if enemy_near and not enemy_in_range:
            biases['INTERCEPT'] += HEUR_INTERCEPT_ENEMY_NEAR
        if collector_vulnerable:
            biases['INTERCEPT'] += HEUR_INTERCEPT_COLLECTOR_DANGER

        # -- ESCORT -----------------------------------------------------
        if collector_vulnerable and collector_carrying:
            biases['ESCORT'] += HEUR_ESCORT_VULNERABLE
        # Siempre escortar kit-holders: van a construir y son vulnerables solos
        if game_context.get('collector_has_kit', False):
            biases['ESCORT'] += HEUR_ESCORT_HAS_KIT
        if guards_near_collector >= 2:
            biases['ESCORT'] += HEUR_ESCORT_REDUNDANT  # -20: reducido para permitir agrupación
        if collector_near_base:
            biases['ESCORT'] += HEUR_ESCORT_COLLECTOR_AT_BASE

        # -- SCOUT ------------------------------------------------------
        if unexplored_frontier_near:
            biases['SCOUT'] += HEUR_SCOUT_LOW_EXPLORATION
        if phase == "EARLY":
            biases['SCOUT'] += HEUR_SCOUT_EARLY_BONUS
        if not collector_vulnerable:
            biases['SCOUT'] += HEUR_SCOUT_NO_ESCORT_NEEDED
        # Penalización gradual según distancia al enemigo — no bloquear exploración lejana
        if enemy_near:
            if distance_to_enemy == 0:    # enemigo muy cerca (<= 3 celdas)
                biases['SCOUT'] -= 80
            elif distance_to_enemy == 1:  # distancia media (4-6 celdas)
                biases['SCOUT'] -= 40
            # distance_to_enemy == 2 (> 6 celdas): no penalizar

        # -- DEFEND_ZONE ------------------------------------------------
        if game_context.get('tower_count', 0) > 0:
            biases['DEFEND_ZONE'] += HEUR_DEFEND_HAS_TOWERS
        # Priorizar INTERCEPT en combate directo, DEFEND_ZONE solo en riesgo sin combate
        if enemy_near and distance_to_enemy == 0:
            biases['INTERCEPT'] += 50
            biases['DEFEND_ZONE'] -= 40
        elif risk_level >= 2:
            biases['DEFEND_ZONE'] += HEUR_DEFEND_HIGH_RISK

        # -- INVESTIGATE ------------------------------------------------
        ticks_ago = game_context.get('recent_enemy_ticks_ago', float('inf'))
        if ticks_ago <= 3:
            biases['INVESTIGATE'] += HEUR_INVESTIGATE_RECENT + 40  # muy reciente
        elif ticks_ago <= 8:
            biases['INVESTIGATE'] += HEUR_INVESTIGATE_RECENT       # moderadamente reciente

        # -- PATROL -----------------------------------------------------
        biases['PATROL'] += HEUR_PATROL_BASE
        if risk_level >= 2:
            biases['PATROL'] += 30  # patrullar más activamente en zona peligrosa
        if collector_near_base:
            biases['PATROL'] += 20  # proteger si hay collector cerca de base

        return biases

    # ===================================================================
    # TOMA DE DECISIÓN
    # ===================================================================

    def decide(self, grid, current_tick, all_collectors, all_guards=None):
        """
        Ciclo completo por tick:
        percibir → estado → biases → acción → ejecutar.
        """
        self._last_grid      = grid
        self._all_collectors = [c for c in all_collectors if c.is_alive]

        if not self.is_alive:
            return

        if self.current_cooldown > 0:
            self.current_cooldown -= 1

        visible_enemies, visible_allies, visible_collectors = self.perceive(
            grid, current_tick
        )
        state = self.build_state(visible_enemies, visible_allies, visible_collectors)

        # Construir game_context para heurísticas
        tower_count = sum(
            1 for x in range(MAP_WIDTH) for y in range(MAP_HEIGHT)
            if self.known_map[x][y].get('last_known_type') == 'tower'
        )
        recent_enemy_ticks_ago = min(
            (current_tick - e['tick'] for e in self.last_seen_enemies),
            default=float('inf')
        )
        collector_has_kit = any(
            getattr(c, 'has_build_kit', False) for c in visible_collectors
        )

        phase = self._get_game_phase()
        game_context = {
            'tower_count':              tower_count,
            'recent_enemy_ticks_ago':   recent_enemy_ticks_ago,
            'collector_has_kit':        collector_has_kit,
            'guards_near_my_collector': state[8],
            'collector_near_base':      state[9],
            'escorted_collector_collected': False,  # simplificación
        }

        biases = self.calculate_heuristic_biases(state, phase, game_context)

        # Q-update con transición anterior + eventos pendientes del tick anterior
        if self.prev_state is not None and self.prev_action is not None:
            step_reward = self._pending_reward
            self._pending_reward = 0.0
            self.q_learning.update(self.prev_state, self.prev_action, step_reward, state)

        action = self.q_learning.get_action(state, biases)
        self.current_action = action
        self.prev_state     = state
        self.prev_action    = action

        # Ejecutar acción
        if action == 'ATTACK':
            target = self._closest_enemy(visible_enemies)
            if target:
                self.execute_attack(target)
                if not self.current_path:
                    self.current_path = self._astar(target.position)
        elif action == 'ESCORT':
            self.current_path = self.execute_escort(all_collectors, visible_enemies)
        elif action == 'INTERCEPT':
            target = self._closest_enemy(visible_enemies)
            if target:
                self.current_path = self.execute_intercept(target)
        elif action == 'DEFEND_ZONE':
            self.current_path = self.execute_defend_zone()
        elif action == 'INVESTIGATE':
            self.current_path = self.execute_investigate()
        elif action == 'SCOUT':
            self.current_path = self.execute_scout()
        elif action == 'PATROL':
            self.current_path = self.execute_patrol()

        if self.current_path:
            self.next_position = self.current_path.pop(0)
        else:
            self.next_position = self.position

        return self.next_position

    # ===================================================================
    # EJECUCIÓN DE ACCIONES
    # ===================================================================

    def execute_escort(self, collectors, visible_enemies):
        """
        Selecciona el recolector a proteger y se posiciona entre él y el
        enemigo más cercano.
        """
        px, py = self.position
        best_c     = None
        best_score = float('-inf')

        for c in collectors:
            if not c.is_alive:
                continue
            cx, cy   = c.position
            dist     = abs(cx - px) + abs(cy - py)
            risk     = self.risk_map[cx][cy]
            carrying = getattr(c, 'carrying_resources', 0)
            has_kit  = 1 if getattr(c, 'has_build_kit', False) else 0
            score    = has_kit * 100 + carrying * 50 - dist - risk * 10
            # Prioridad máxima si el collector está en peligro inmediato
            if any(
                abs(e.position[0] - cx) + abs(e.position[1] - cy) <= 5
                for e in visible_enemies
            ):
                score += 200
            if score > best_score:
                best_score = score
                best_c     = c

        if best_c is None:
            return self.execute_patrol()

        if visible_enemies:
            enemy = min(
                visible_enemies,
                key=lambda e: (abs(e.position[0] - best_c.position[0])
                               + abs(e.position[1] - best_c.position[1]))
            )
            ex, ey = enemy.position
            cx, cy = best_c.position
            mid_x  = (ex + cx) // 2
            mid_y  = (ey + cy) // 2
            goal   = (
                max(0, min(MAP_WIDTH  - 1, mid_x)),
                max(0, min(MAP_HEIGHT - 1, mid_y)),
            )
        else:
            goal = best_c.position

        self.current_target = goal
        return self._astar(goal)

    def execute_attack(self, target):
        """
        Verifica si el ataque es posible.
        El ataque real se resuelve en _collect_all_attacks() del environment.
        """
        if target is None or not target.is_alive:
            return False
        px, py = self.position
        dist   = abs(target.position[0] - px) + abs(target.position[1] - py)
        return dist <= self.attack_range and self.current_cooldown == 0

    def execute_intercept(self, enemy):
        """Predice posición futura del enemigo y va al punto de corte."""
        if enemy is None:
            return self.execute_patrol()

        ex, ey    = enemy.position
        prev_entry = next(
            (e for e in self.last_seen_enemies if e.get('agent') is enemy),
            None
        )
        if prev_entry and prev_entry['position'] != (ex, ey):
            dx = ex - prev_entry['position'][0]
            dy = ey - prev_entry['position'][1]
        else:
            px_self, py_self = self.position
            dx = ex - px_self
            dy = ey - py_self

        norm = max(1, abs(dx) + abs(dy))
        ddx  = dx / norm
        ddy  = dy / norm

        px_self, py_self = self.position
        best_intercept   = (ex, ey)
        best_score       = float('-inf')

        for k in range(1, 4):
            ix = int(ex + ddx * k)
            iy = int(ey + ddy * k)
            ix = max(0, min(MAP_WIDTH  - 1, ix))
            iy = max(0, min(MAP_HEIGHT - 1, iy))
            d_guard = abs(ix - px_self) + abs(iy - py_self)
            d_route = abs(ix - ex)    + abs(iy - ey)
            score   = -d_guard - d_route
            if score > best_score:
                best_score     = score
                best_intercept = (ix, iy)

        self.current_target = best_intercept
        return self._astar(best_intercept)

    def execute_defend_zone(self):
        """Elige la zona con más recursos/torres y mayor actividad enemiga."""
        px, py     = self.position
        best_pos   = None
        best_score = float('-inf')

        for res in self.discovered_resources:
            rx, ry = res['position']
            dist   = abs(rx - px) + abs(ry - py)
            risk   = self.risk_map[rx][ry]
            score  = res['amount'] * 0.5 + risk * 5.0 - dist * 0.3
            if score > best_score:
                best_score = score
                best_pos   = (rx, ry)

        for entry in self.last_seen_enemies:
            ex, ey = entry['position']
            dist   = abs(ex - px) + abs(ey - py)
            score  = 10.0 - dist * 0.2
            if score > best_score:
                best_score = score
                best_pos   = (ex, ey)

        if best_pos is None:
            return self.execute_patrol()

        self.current_target = best_pos
        return self._astar(best_pos)

    def execute_investigate(self):
        """Va a la última posición conocida del enemigo más reciente."""
        if not self.last_seen_enemies:
            return self.execute_patrol()

        most_recent = max(self.last_seen_enemies, key=lambda e: e['tick'])
        goal = most_recent['position']
        self.current_target = goal
        return self._astar(goal)

    def execute_scout(self):
        """
        Va al borde del mapa explorado para expandir visión.
        Prioriza fronteras con más celdas inexploradas alrededor y
        cercanas a recolectores (para que los beneficie).
        """
        frontier_cells = self._find_frontier_cells()

        if not frontier_cells:
            return self.execute_patrol()

        px, py     = self.position
        best_cell  = None
        best_score = float('-inf')

        for cell in frontier_cells:
            cx, cy = cell
            dist   = abs(cx - px) + abs(cy - py)
            risk   = self.risk_map[cx][cy] if self.known_map[cx][cy].get('explored', False) \
                     else RISK_UNEXPLORED
            score  = -dist * 2 - risk * 30
            score += self._count_unexplored_neighbors(cell) * 15

            # Preferir fronteras en dirección de recolectores
            for c in self._all_collectors:
                if (abs(cx - c.position[0]) + abs(cy - c.position[1])) < 8:
                    score += 20

            if score > best_score:
                best_score = score
                best_cell  = cell

        if best_cell:
            self.current_target = best_cell
            return self._astar(best_cell)

        return self.execute_patrol()

    def execute_patrol(self):
        """
        Recorre waypoints en zonas parcialmente exploradas.
        Regenera la lista cuando se llega al último waypoint.
        """
        px, py = self.position

        if (self._patrol_waypoints
                and self.position == self._patrol_waypoints[self._patrol_idx]):
            self._patrol_idx = (self._patrol_idx + 1) % len(self._patrol_waypoints)
            if self._patrol_idx == 0:
                self._patrol_waypoints = []

        if not self._patrol_waypoints:
            self._build_patrol_waypoints(px, py)

        if not self._patrol_waypoints:
            return []

        goal = self._patrol_waypoints[self._patrol_idx]
        self.current_target = goal
        return self._astar(goal)

    def _build_patrol_waypoints(self, px, py):
        """Genera hasta 4 waypoints en la frontera de la zona explorada."""
        waypoints = []
        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                if self.known_map[x][y].get('explored', False):
                    continue
                is_frontier = any(
                    0 <= x + dx < MAP_WIDTH and 0 <= y + dy < MAP_HEIGHT
                    and self.known_map[x + dx][y + dy].get('explored', False)
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                )
                if is_frontier:
                    waypoints.append((x, y))

        waypoints.sort(key=lambda p: abs(p[0] - px) + abs(p[1] - py))
        step = max(1, len(waypoints) // 4)
        self._patrol_waypoints = waypoints[::step][:4] or [BASE_POSITION]
        self._patrol_idx = 0

    # ===================================================================
    # COMBATE
    # ===================================================================

    def can_attack(self, target):
        """True si el target está en rango y el cooldown es 0."""
        if not self.is_alive or self.current_cooldown > 0:
            return False
        if not hasattr(target, 'is_alive') or not target.is_alive:
            return False
        dist = (abs(target.position[0] - self.position[0])
                + abs(target.position[1] - self.position[1]))
        return dist <= self.attack_range

    # ===================================================================
    # RECOMPENSAS
    # ===================================================================

    def get_reward(self, event):
        """Mapea un evento a su recompensa según constants."""
        rewards = {
            'kill_hunter':              REWARD_KILL_HUNTER,
            'protect_collector':        REWARD_PROTECT_COLLECTOR,
            'intercept':                REWARD_INTERCEPT,
            'defend_zone':              REWARD_DEFEND_ZONE,
            'die':                      REWARD_GUARD_DIE,
            'collector_dies_nearby':         REWARD_COLLECTOR_DIES_NEARBY,
            'collector_dies_while_adjacent': REWARD_COLLECTOR_DIES_WHILE_ADJACENT,
            'bad_decision':             REWARD_BAD_DECISION,
            'sole_escort':              REWARD_SOLE_ESCORT,
            'redundant_escort':         REWARD_REDUNDANT_ESCORT,
            'collector_safe_no_escort': REWARD_COLLECTOR_SAFE_NO_ESCORT,
            'scout_new_cells':          REWARD_SCOUT_NEW_CELLS,
            'scout_find_resource':      REWARD_SCOUT_FIND_RESOURCE,
            'escort_collector_collects': REWARD_ESCORT_COLLECTOR_COLLECTS,
            'guard_idle':               REWARD_GUARD_IDLE,
        }
        return rewards.get(event, 0.0)

    def receive_reward(self, event):
        """
        Acumula la recompensa del evento para aplicarla en el siguiente decide(),
        donde el next_state real ya estará disponible.
        """
        self._pending_reward += self.get_reward(event)

    # ===================================================================
    # MUERTE
    # ===================================================================

    def die(self):
        """Muerte permanente."""
        self.is_alive = False

    # ===================================================================
    # PATHFINDING INTERNO
    # ===================================================================

    def _astar(self, goal):
        """Wrapper de find_path con known_map y cost_function del guardia."""
        return find_path(
            self.position,
            goal,
            self._last_grid,
            self._cost_function,
            known_map=self.known_map,
        )

    def _cost_function(self, _, neighbor_pos):
        """
        cost = 1.0 + risk + cercanía_a_enemigos - cercanía_a_torres aliadas
        """
        nx, ny   = neighbor_pos
        explored = self.known_map[nx][ny].get('explored', False)
        cost     = 1.0 + (self.risk_map[nx][ny] if explored else RISK_UNEXPLORED)

        for entry in self.last_seen_enemies:
            ep   = entry['position']
            dist = abs(ep[0] - nx) + abs(ep[1] - ny)
            if dist <= 3:
                cost += max(0.0, (3 - dist) * 1.5)

        if self._last_grid is not None:
            for tx in range(max(0, nx - TOWER_ATTACK_RANGE),
                            min(MAP_WIDTH, nx + TOWER_ATTACK_RANGE + 1)):
                for ty in range(max(0, ny - TOWER_ATTACK_RANGE),
                                min(MAP_HEIGHT, ny + TOWER_ATTACK_RANGE + 1)):
                    if (self.known_map[tx][ty].get('explored', False)
                            and self.known_map[tx][ty].get('last_known_type') == 'tower'):
                        dist = abs(tx - nx) + abs(ty - ny)
                        if dist <= TOWER_ATTACK_RANGE:
                            cost -= max(0.0, (TOWER_ATTACK_RANGE - dist) * 0.3)

        return max(0.1, cost)

    # ===================================================================
    # UTILIDADES
    # ===================================================================

    def _closest_enemy(self, visible_enemies):
        """Retorna el enemigo más cercano o None."""
        if not visible_enemies:
            return None
        px, py = self.position
        return min(
            visible_enemies,
            key=lambda e: abs(e.position[0] - px) + abs(e.position[1] - py)
        )

    def _nearest_alive_collector(self, visible_collectors):
        """Retorna el recolector vivo más cercano entre los visibles, o None."""
        alive = [c for c in visible_collectors if c.is_alive]
        if not alive:
            return None
        px, py = self.position
        return min(
            alive,
            key=lambda c: abs(c.position[0] - px) + abs(c.position[1] - py)
        )

    def _find_frontier_cells(self):
        """
        Retorna celdas inexploradas adyacentes a celdas exploradas
        (frontera del mapa conocido). El guardia va allí para revelar territorio.
        """
        cells = []
        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                if self.known_map[x][y].get('explored', False):
                    continue
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT:
                        if self.known_map[nx][ny].get('explored', False):
                            cells.append((x, y))
                            break
        return cells

    def _count_unexplored_neighbors(self, cell):
        """Cuenta vecinos inexplorados de una celda (para scorer de SCOUT)."""
        cx, cy = cell
        count  = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT:
                if not self.known_map[nx][ny].get('explored', False):
                    count += 1
        return count

    def _has_frontier_nearby(self):
        """
        Detecta si hay frontera inexplorada a distancia <= 8 del guardia.
        Usado para la variable de estado 'unexplored_frontier_near'.
        """
        px, py = self.position
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                nx, ny = px + dx, py + dy
                if not (0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT):
                    continue
                if not self.known_map[nx][ny].get('explored', False):
                    for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nnx, nny = nx + ddx, ny + ddy
                        if 0 <= nnx < MAP_WIDTH and 0 <= nny < MAP_HEIGHT:
                            if self.known_map[nnx][nny].get('explored', False):
                                return 1
        return 0

    def _get_game_phase(self):
        """Determina la fase del juego según cobertura del mapa."""
        explored_count = self.shared_data.get('explored_count', 0)
        explored_pct   = explored_count / (MAP_WIDTH * MAP_HEIGHT)
        if explored_pct < PHASE_EARLY_THRESHOLD:
            return "EARLY"
        elif explored_pct < PHASE_MID_THRESHOLD:
            return "MID"
        return "LATE"
