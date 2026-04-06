from utils.constants import (
    COLLECTOR_HP,
    COLLECTOR_SPEED,
    COLLECTOR_VISION,
    COLLECTOR_CARRY_CAPACITY,
    COLLECT_RATE,
    BASE_POSITION,
    MAP_WIDTH,
    MAP_HEIGHT,
    PHASE_EARLY_THRESHOLD,
    PHASE_MID_THRESHOLD,
    HEUR_FLEE_ENEMY_NEAR,
    HEUR_FLEE_ENEMY_NO_GUARD,
    HEUR_RETURN_FULL,
    HEUR_RETURN_PARTIAL_FACTOR,
    HEUR_RETURN_EMPTY_PENALTY,
    HEUR_GOTO_RESOURCE_BASE,
    HEUR_GOTO_RESOURCE_EMPTY_BONUS,
    HEUR_GOTO_RESOURCE_NO_GUARD,
    HEUR_GOTO_RESOURCE_MID_BONUS,
    HEUR_EXPLORE_LOW_COVERAGE,
    HEUR_EXPLORE_NO_RESOURCES_KNOWN,
    HEUR_EXPLORE_NO_GUARD_PENALTY,
    HEUR_EXPLORE_EARLY_BONUS,
    HEUR_BUILD_HAS_KIT,
    HEUR_BUILD_TOO_MANY_TOWERS,
    HEUR_BUILD_LAST_COLLECTOR,
    HEUR_BUILD_LOW_EXPLORATION,
    STAGNATION_THRESHOLD,
    STAGNATION_PENALTY_RATE,
    MAX_TOWERS_PER_EXPLORATION,
    EARLY_GAME_FLEE_BONUS,
    REWARD_DELIVER_RESOURCES,
    REWARD_COLLECT,
    REWARD_EXPLORE,
    REWARD_BUILD_TOWER,
    REWARD_DANGER_ZONE,
    REWARD_LOSE_RESOURCES,
    REWARD_COLLECTOR_DIE,
    REWARD_IDLE,
    REWARD_RETURN_EMPTY,
    REWARD_APPROACH_RESOURCE,
    REWARD_STAGNATION,
    REWARD_EXPLORE_WITH_ESCORT,
    RISK_UNEXPLORED,
)
from pathfinding.astar import find_path

# Acciones disponibles del recolector
ACTIONS = ['EXPLORE', 'GO_TO_RESOURCE', 'RETURN_TO_BASE', 'FLEE', 'BUILD_TOWER']

# Umbrales del mapa de riesgo para discretización
_RISK_LOW  = 0.3
_RISK_HIGH = 0.7


class Collector:
    """
    Recolector del Equipo A.

    Toma decisiones mediante Q-learning compartido + heurísticas.
    Pathfinding A* restringido a su known_map (fog of war real).
    Muerte permanente (sin respawn).
    Estado RL expandido: 10 variables.
    """

    def __init__(self, position, shared_data, q_learning):
        # Posición y stats
        self.position          = position
        self.hp                = COLLECTOR_HP
        self.current_hp        = COLLECTOR_HP
        self.speed             = COLLECTOR_SPEED
        self.vision_range      = COLLECTOR_VISION
        self.carrying_resources = 0
        self.carrying_capacity = COLLECTOR_CARRY_CAPACITY
        self.has_build_kit     = False
        self.is_alive          = True

        # Memoria compartida (referencias)
        self.shared_data         = shared_data
        self.known_map           = shared_data['known_map']
        self.discovered_resources = shared_data['discovered_resources']
        self.last_seen_enemies   = shared_data['last_seen_enemies']
        self.risk_map            = shared_data['risk_map']

        # Q-learning compartido con todos los recolectores
        self.q_learning = q_learning

        # Estado interno
        self.current_action = 'EXPLORE'
        self.current_target = None
        self.current_path   = []
        self.prev_state     = None
        self.prev_action    = None

        # Posición siguiente calculada en decide()
        self.next_position  = position

        # Señalización de construcción de torre
        self.wants_to_build = False
        self.build_target   = None

        # Referencia al grid actual (actualizada al inicio de decide)
        self._last_grid = None

        # Tracking de progreso (anti-estancamiento)
        self.current_action_streak  = 0
        self.ticks_since_progress   = 0
        self._prev_resources_carried = 0
        self._prev_explored_count    = 0

    # ===================================================================
    # PERCEPCIÓN
    # ===================================================================

    def perceive(self, grid, current_tick):
        """
        Observa celdas dentro de vision_range.
        Actualiza known_map, discovered_resources y last_seen_enemies.
        Retorna (visible_enemies, visible_resources, visible_allies).
        """
        px, py = self.position
        visible_enemies   = []
        visible_resources = []
        visible_allies    = []

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

                if cell.type == 'resource' and cell.resource_amount > 0:
                    existing = next(
                        (r for r in self.discovered_resources
                         if r['position'] == (nx, ny)),
                        None
                    )
                    if existing:
                        existing['amount']    = cell.resource_amount
                        existing['last_seen'] = current_tick
                    else:
                        self.discovered_resources.append({
                            'position': (nx, ny),
                            'amount':   cell.resource_amount,
                            'last_seen': current_tick,
                        })
                    visible_resources.append({'position': (nx, ny), 'amount': cell.resource_amount})
                else:
                    self.discovered_resources[:] = [
                        r for r in self.discovered_resources
                        if r['position'] != (nx, ny) or cell.type == 'resource'
                    ]

                for agent in cell.agents:
                    if agent is self or not agent.is_alive:
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
                    elif atype in ('Collector', 'Guard'):
                        visible_allies.append(agent)

                if cell.tower is not None:
                    visible_allies.append(cell.tower)

        return visible_enemies, visible_resources, visible_allies

    # ===================================================================
    # ESTADO RL (10 variables)
    # ===================================================================

    def build_state(self, visible_enemies, visible_allies):
        """
        Construye la tupla de estado discreta para Q-learning (10 variables):
          (enemy_near, carrying, is_full, risk_level, guard_near, tower_near,
           has_build_kit, resources_known, near_base, explored_level)

        risk_level:     0=LOW, 1=MED, 2=HIGH
        explored_level: 0=LOW(<30%), 1=MED(30-65%), 2=HIGH(>65%)
        """
        px, py = self.position

        enemy_near = 1 if visible_enemies else 0
        carrying   = 1 if self.carrying_resources > 0 else 0
        is_full    = 1 if self.carrying_resources >= self.carrying_capacity else 0

        r = self.risk_map[px][py]
        if r < _RISK_LOW:
            risk_level = 0
        elif r < _RISK_HIGH:
            risk_level = 1
        else:
            risk_level = 2

        guard_near = int(
            any(a.__class__.__name__ == 'Guard' for a in visible_allies)
        )
        tower_near = int(
            any(a.__class__.__name__ == 'Tower' for a in visible_allies)
        )
        has_kit = 1 if self.has_build_kit else 0

        # Nuevas variables
        resources_known = 1 if self.discovered_resources else 0

        dist_base = abs(px - BASE_POSITION[0]) + abs(py - BASE_POSITION[1])
        near_base = 1 if dist_base <= 5 else 0

        explored_count = self.shared_data.get('explored_count', 0)
        explored_pct   = explored_count / (MAP_WIDTH * MAP_HEIGHT)
        if explored_pct < PHASE_EARLY_THRESHOLD:
            explored_level = 0
        elif explored_pct < PHASE_MID_THRESHOLD:
            explored_level = 1
        else:
            explored_level = 2

        return (enemy_near, carrying, is_full, risk_level, guard_near,
                tower_near, has_kit, resources_known, near_base, explored_level)

    # ===================================================================
    # HEURISTIC BIASES
    # ===================================================================

    def calculate_heuristic_biases(self, state, phase, game_context):
        """
        Retorna dict {action: float} con sesgos heurísticos sobre Q-values.
        Las heurísticas son "reglas de emergencia" que complementan al RL.
        """
        (enemy_near, carrying, is_full, risk_level, guard_near,
         tower_near, has_build_kit, resources_known, near_base,
         explored_level) = state

        biases = {a: 0.0 for a in ACTIONS}

        # -- FLEE (emergencia — siempre domina si aplica) ---------------
        if enemy_near:
            biases['FLEE'] += HEUR_FLEE_ENEMY_NEAR
            if not guard_near:
                biases['FLEE'] += HEUR_FLEE_ENEMY_NO_GUARD
        if phase == "EARLY" and enemy_near:
            biases['FLEE'] += EARLY_GAME_FLEE_BONUS

        # -- RETURN_TO_BASE ---------------------------------------------
        if is_full:
            biases['RETURN_TO_BASE'] += HEUR_RETURN_FULL
        elif carrying:
            fill_ratio = game_context['carrying_resources'] / game_context['carrying_capacity']
            biases['RETURN_TO_BASE'] += int(HEUR_RETURN_PARTIAL_FACTOR * fill_ratio)
        else:
            biases['RETURN_TO_BASE'] += HEUR_RETURN_EMPTY_PENALTY

        # -- GO_TO_RESOURCE ---------------------------------------------
        if resources_known:
            biases['GO_TO_RESOURCE'] += HEUR_GOTO_RESOURCE_BASE
            if not carrying:
                biases['GO_TO_RESOURCE'] += HEUR_GOTO_RESOURCE_EMPTY_BONUS
            if risk_level >= 2 and not guard_near:
                biases['GO_TO_RESOURCE'] += HEUR_GOTO_RESOURCE_NO_GUARD
            if phase == "MID":
                biases['GO_TO_RESOURCE'] += HEUR_GOTO_RESOURCE_MID_BONUS

        # -- EXPLORE ----------------------------------------------------
        if explored_level == 0:
            biases['EXPLORE'] += HEUR_EXPLORE_LOW_COVERAGE
        if not resources_known:
            biases['EXPLORE'] += HEUR_EXPLORE_NO_RESOURCES_KNOWN
        if not guard_near and not tower_near:
            biases['EXPLORE'] += HEUR_EXPLORE_NO_GUARD_PENALTY
        if phase == "EARLY":
            biases['EXPLORE'] += HEUR_EXPLORE_EARLY_BONUS

        # -- BUILD_TOWER ------------------------------------------------
        if has_build_kit:
            biases['BUILD_TOWER'] += HEUR_BUILD_HAS_KIT
            explored_pct = game_context.get('explored_percent', 0.0)
            max_t = max(2, int(explored_pct * MAP_WIDTH * MAP_HEIGHT
                               * MAX_TOWERS_PER_EXPLORATION))
            if game_context['tower_count'] >= max_t:
                biases['BUILD_TOWER'] += HEUR_BUILD_TOO_MANY_TOWERS
            if game_context['num_alive_collectors'] <= 1:
                biases['BUILD_TOWER'] += HEUR_BUILD_LAST_COLLECTOR
            if explored_level == 0:
                biases['BUILD_TOWER'] += HEUR_BUILD_LOW_EXPLORATION
        else:
            biases['BUILD_TOWER'] = -200.0

        # -- ANTI-ESTANCAMIENTO ----------------------------------------
        streak = game_context['current_action_streak']
        if streak > STAGNATION_THRESHOLD and self.current_action in biases:
            biases[self.current_action] -= (
                STAGNATION_PENALTY_RATE * (streak - STAGNATION_THRESHOLD)
            )

        stagnation = game_context['ticks_since_progress']
        if stagnation > STAGNATION_THRESHOLD:
            biases['GO_TO_RESOURCE'] += stagnation * 3
            biases['EXPLORE']        += stagnation * 2

        return biases

    # ===================================================================
    # EJECUCIÓN DE ACCIONES
    # ===================================================================

    def execute_explore(self, all_collectors):
        """
        Selecciona la celda frontera no explorada con mayor score.
        Prioriza celdas con guardias o torres cercanas.
        """
        px, py = self.position
        best_cell  = None
        best_score = float('-inf')

        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                if self.known_map[x][y].get('explored', False):
                    continue
                is_frontier = any(
                    0 <= x + dx < MAP_WIDTH and 0 <= y + dy < MAP_HEIGHT
                    and self.known_map[x + dx][y + dy].get('explored', False)
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                )
                if not is_frontier:
                    continue

                dist = abs(x - px) + abs(y - py)
                risk = (self.risk_map[x][y] if self.known_map[x][y].get('explored', False)
                        else RISK_UNEXPLORED)
                other_heading = sum(
                    1 for c in all_collectors
                    if c is not self and c.is_alive and c.current_target == (x, y)
                )
                score = 100.0 - dist * 0.5 - risk * 20.0 - other_heading * 15.0
                if score > best_score:
                    best_score = score
                    best_cell  = (x, y)

        if best_cell is None:
            return []
        self.current_target = best_cell
        return self._astar(best_cell)

    def execute_go_to_resource(self, all_collectors):
        """
        Selecciona el recurso óptimo de discovered_resources.
        """
        px, py = self.position
        best_resource = None
        best_score    = float('-inf')

        for res in self.discovered_resources:
            rx, ry = res['position']
            dist   = abs(rx - px) + abs(ry - py)
            risk   = (self.risk_map[rx][ry] if self.known_map[rx][ry].get('explored', False)
                      else RISK_UNEXPLORED)
            others_going = sum(
                1 for c in all_collectors
                if c is not self and c.is_alive and c.current_target == (rx, ry)
            )
            score = res['amount'] - dist * 0.5 - risk * 10.0 - others_going * 5.0
            if score > best_score:
                best_score    = score
                best_resource = res

        if best_resource is None:
            return self.execute_explore(all_collectors)

        self.current_target = best_resource['position']
        return self._astar(best_resource['position'])

    def execute_return_to_base(self):
        """Pathfinding A* hacia BASE_POSITION."""
        self.current_target = BASE_POSITION
        return self._astar(BASE_POSITION)

    def execute_flee(self, visible_enemies, visible_allies):
        """
        Evalúa opciones de escape y elige la de mayor score.
        """
        px, py = self.position
        options = []

        towers = [a for a in visible_allies if a.__class__.__name__ == 'Tower']
        for t in towers:
            d = abs(t.position[0] - px) + abs(t.position[1] - py)
            options.append({'pos': t.position, 'score': -d + 30.0})

        guards = [a for a in visible_allies if a.__class__.__name__ == 'Guard']
        for g in guards:
            d = abs(g.position[0] - px) + abs(g.position[1] - py)
            options.append({'pos': g.position, 'score': -d + 20.0})

        d_base = abs(BASE_POSITION[0] - px) + abs(BASE_POSITION[1] - py)
        options.append({'pos': BASE_POSITION, 'score': -d_base + 10.0})

        for opt in options:
            ox, oy = opt['pos']
            for enemy in visible_enemies:
                enemy_dist = abs(enemy.position[0] - ox) + abs(enemy.position[1] - oy)
                opt['score'] -= max(0.0, 10.0 - enemy_dist) * 5.0

        if options:
            best = max(options, key=lambda o: o['score'])
            self.current_target = best['pos']
            return self._astar(best['pos'])

        return []

    def execute_build_tower(self):
        """
        Selecciona la mejor celda libre para construir una torre.
        Nunca construye si es el último recolector vivo.
        """
        if self.build_target is None:
            cell = self._select_build_cell()
            if cell is None:
                return []
            self.build_target = cell

        self.current_target = self.build_target

        if self.position == self.build_target:
            self.wants_to_build = True
            return []

        return self._astar(self.build_target)

    def _select_build_cell(self):
        """
        Devuelve la mejor celda explorada vacía para colocar una torre.
        """
        px, py = self.position
        best_cell  = None
        best_score = float('-inf')

        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                if not self.known_map[x][y].get('explored', False):
                    continue
                if self.known_map[x][y]['last_known_type'] not in ('empty', 'resource'):
                    continue

                risk = self.risk_map[x][y]

                resource_proximity = sum(
                    max(0.0, 5.0 - (abs(r['position'][0] - x) + abs(r['position'][1] - y)))
                    for r in self.discovered_resources
                )

                redundancy = sum(
                    1 for tx in range(max(0, x - 4), min(MAP_WIDTH, x + 5))
                    for ty in range(max(0, y - 4), min(MAP_HEIGHT, y + 5))
                    if self.known_map[tx][ty].get('explored', False)
                    and self.known_map[tx][ty]['last_known_type'] == 'tower'
                )

                dist = abs(x - px) + abs(y - py)
                score = risk * 10.0 + resource_proximity - redundancy * 15.0 - dist * 0.2
                if score > best_score:
                    best_score = score
                    best_cell  = (x, y)

        return best_cell

    # ===================================================================
    # RECOLECCIÓN Y DEPÓSITO
    # ===================================================================

    def collect_resource(self, cell):
        """
        Recoge recursos de la celda actual.
        Retorna cantidad recolectada.
        """
        if cell.resource_amount <= 0:
            return 0
        free      = self.carrying_capacity - self.carrying_resources
        collected = min(COLLECT_RATE, cell.resource_amount, free)
        self.carrying_resources  += collected
        cell.resource_amount     -= collected
        return collected

    def deposit_resources(self):
        """Deposita todos los recursos en la base. Retorna cantidad depositada."""
        amount = self.carrying_resources
        self.carrying_resources = 0
        return amount

    def receive_build_kit(self):
        """El environment entrega un build kit al recolector."""
        self.has_build_kit = True

    # ===================================================================
    # RECOMPENSAS
    # ===================================================================

    def get_reward(self, event):
        """Mapea un evento a su recompensa según constants."""
        rewards = {
            'deliver_resources':  REWARD_DELIVER_RESOURCES,
            'collect':            REWARD_COLLECT,
            'explore':            REWARD_EXPLORE,
            'build_tower':        REWARD_BUILD_TOWER,
            'danger_zone':        REWARD_DANGER_ZONE,
            'lose_resources':     REWARD_LOSE_RESOURCES,
            'die':                REWARD_COLLECTOR_DIE,
            'idle':               REWARD_IDLE,
            'return_empty':       REWARD_RETURN_EMPTY,
            'approach_resource':  REWARD_APPROACH_RESOURCE,
            'stagnation':         REWARD_STAGNATION,
            'explore_with_escort': REWARD_EXPLORE_WITH_ESCORT,
        }
        return rewards.get(event, 0.0)

    def receive_reward(self, event):
        """
        Registra la recompensa en la Q-table para el último (state, action).
        """
        if self.prev_state is None or self.prev_action is None:
            return
        reward = self.get_reward(event)
        self.q_learning.update(self.prev_state, self.prev_action, reward, self.prev_state)

    def _compute_step_reward(self, action, guard_near):
        """Calcula recompensa de paso basada en tracking de progreso."""
        reward = 0.0
        if action == 'RETURN_TO_BASE' and self.carrying_resources == 0:
            reward += REWARD_RETURN_EMPTY
        if self.ticks_since_progress > STAGNATION_THRESHOLD:
            reward += REWARD_STAGNATION
        return reward

    # ===================================================================
    # MUERTE
    # ===================================================================

    def die(self):
        """Muerte permanente. Pierde recursos y build kit."""
        self.is_alive           = False
        self.carrying_resources = 0
        self.has_build_kit      = False

    # ===================================================================
    # PATHFINDING INTERNO
    # ===================================================================

    def _astar(self, goal):
        """Wrapper de find_path con known_map (fog of war real)."""
        return find_path(
            self.position,
            goal,
            self._last_grid,
            self._cost_function,
            known_map=self.known_map,
        )

    def _cost_function(self, _, neighbor_pos):
        """
        cost = 1.0 + risk + cercanía_a_enemigos - cercanía_a_torres/guardias
        """
        nx, ny   = neighbor_pos
        explored = self.known_map[nx][ny].get('explored', False)
        cost     = 1.0 + (self.risk_map[nx][ny] if explored else RISK_UNEXPLORED)

        for entry in self.last_seen_enemies:
            ep   = entry['position']
            dist = abs(ep[0] - nx) + abs(ep[1] - ny)
            if dist <= 3:
                cost += max(0.0, (3 - dist) * 1.5)

        return max(0.1, cost)

    # ===================================================================
    # UTILIDADES
    # ===================================================================

    def _get_game_phase(self):
        """Determina la fase del juego según cobertura del mapa."""
        explored_count = self.shared_data.get('explored_count', 0)
        explored_pct   = explored_count / (MAP_WIDTH * MAP_HEIGHT)
        if explored_pct < PHASE_EARLY_THRESHOLD:
            return "EARLY"
        elif explored_pct < PHASE_MID_THRESHOLD:
            return "MID"
        return "LATE"

    def _count_known_towers(self):
        """Cuenta torres conocidas en el mapa."""
        return sum(
            1 for x in range(MAP_WIDTH) for y in range(MAP_HEIGHT)
            if self.known_map[x][y].get('last_known_type') == 'tower'
        )

    def _update_progress_tracking(self, action):
        """Actualiza contadores de estancamiento."""
        # Streak de acción repetida
        if action == self.current_action:
            self.current_action_streak += 1
        else:
            self.current_action_streak = 0

        # Detectar progreso real (recursos o exploración)
        explored_count = self.shared_data.get('explored_count', 0)
        has_progress = (
            self.carrying_resources != self._prev_resources_carried
            or explored_count > self._prev_explored_count + 2
        )
        if has_progress:
            self.ticks_since_progress        = 0
            self._prev_resources_carried     = self.carrying_resources
            self._prev_explored_count        = explored_count
        else:
            self.ticks_since_progress += 1

    # ===================================================================
    # TOMA DE DECISIÓN
    # ===================================================================

    def decide(self, grid, current_tick, all_collectors):
        """
        Ciclo completo por tick:
        percibir → estado → biases → acción (ε-greedy) → ejecutar → path
        """
        self._last_grid = grid
        if not self.is_alive:
            return

        visible_enemies, _, visible_allies = self.perceive(grid, current_tick)
        state = self.build_state(visible_enemies, visible_allies)

        # Construir game_context para heurísticas
        explored_count  = self.shared_data.get('explored_count', 0)
        explored_pct    = explored_count / (MAP_WIDTH * MAP_HEIGHT)
        tower_count     = self._count_known_towers()
        num_alive_colls = sum(1 for c in all_collectors if c.is_alive)
        guard_near      = state[4]
        phase           = self._get_game_phase()

        game_context = {
            'carrying_resources':  self.carrying_resources,
            'carrying_capacity':   self.carrying_capacity,
            'explored_percent':    explored_pct,
            'tower_count':         tower_count,
            'num_alive_collectors': num_alive_colls,
            'current_action_streak': self.current_action_streak,
            'ticks_since_progress':  self.ticks_since_progress,
            'guard_near':           guard_near,
        }

        biases = self.calculate_heuristic_biases(state, phase, game_context)

        # Actualizar tracking ANTES del Q-update (usa acción anterior)
        self._update_progress_tracking(self.current_action)

        # Q-update con transición anterior + recompensa de paso
        if self.prev_state is not None and self.prev_action is not None:
            step_reward = self._compute_step_reward(self.prev_action, guard_near)
            self.q_learning.update(self.prev_state, self.prev_action, step_reward, state)

        action = self.q_learning.get_action(state, biases)
        self.current_action = action
        self.prev_state     = state
        self.prev_action    = action

        self.wants_to_build = False

        if action == 'EXPLORE':
            self.current_path = self.execute_explore(all_collectors)
        elif action == 'GO_TO_RESOURCE':
            self.current_path = self.execute_go_to_resource(all_collectors)
        elif action == 'RETURN_TO_BASE':
            self.current_path = self.execute_return_to_base()
        elif action == 'FLEE':
            self.current_path = self.execute_flee(visible_enemies, visible_allies)
        elif action == 'BUILD_TOWER':
            self.current_path = self.execute_build_tower()

        if self.current_path:
            self.next_position = self.current_path.pop(0)
        else:
            self.next_position = self.position

        return self.next_position
