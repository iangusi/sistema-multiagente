from utils.constants import (
    GUARD_HP,
    GUARD_SPEED,
    GUARD_VISION,
    GUARD_ATTACK_RANGE,
    GUARD_ATTACK_COOLDOWN,
    GUARD_DAMAGE,
    BASE_POSITION,
    MAP_WIDTH,
    MAP_HEIGHT,
    HEUR_GUARD_ATTACK,
    HEUR_GUARD_FLEE_DANGER,
    HEUR_GUARD_DEFEND_ALLY,
    HEUR_GUARD_EXPLORE_BASE,
    REWARD_GUARD_CELL_EXPLORED,
    REWARD_GUARD_APPROACH_EXPLORE,
    REWARD_GUARD_EXPLORE_DANGER,
    REWARD_KILL_HUNTER,
    REWARD_KILL_HUNTER_DEFEND,
    REWARD_GUARD_HUNTER_NO_DANGER,
    REWARD_GUARD_APPROACH_HUNTER,
    REWARD_GUARD_HUNTER_IN_DANGER,
    REWARD_GUARD_FLEE_DANGER,
    REWARD_GUARD_FLEE_NO_ATTACK,
    REWARD_GUARD_CANT_ATTACK,
    REWARD_GUARD_APPROACH_ALLY,
    REWARD_GUARD_DIE,
    RISK_UNEXPLORED,
)
from pathfinding.astar import find_path
from pathfinding.astar_secure import find_path as find_path_secure

# Acciones disponibles del guardia
ACTIONS = ['EXPLORE', 'ATTACK', 'DEFEND', 'FLEE']

# Radio para calcular "cazador cerca" de un aliado en la función de peligro
_DANGER_RANGE = 4

# Radio para considerar que un guardia protege a un recolector
_GUARD_PROTECT_RANGE = 4


class Guard:
    """
    Guardia del Equipo A.

    Estado RL (4 variables):
      (ally_in_danger, i_am_in_danger, hunter_near, can_attack)

    Acciones: EXPLORE, ATTACK, DEFEND, FLEE

    Peligro de cada aliado:
      - Recolector: 0 + 3×(cazadores a ≤4 celdas) − 2×(guardias a ≤4 celdas)
      - Guardia:   −1 + 1×(cazadores a ≤4 celdas)
    Muerte permanente.
    """

    def __init__(self, position, shared_data, q_learning):
        self.position         = position
        self.hp               = GUARD_HP
        self.current_hp       = GUARD_HP
        self.speed            = GUARD_SPEED
        self.vision_range     = GUARD_VISION
        self.attack_range     = GUARD_ATTACK_RANGE
        self.attack_cooldown  = GUARD_ATTACK_COOLDOWN
        self.damage           = GUARD_DAMAGE
        self.current_cooldown = 0
        self.is_alive         = True

        # Memoria compartida
        self.shared_data          = shared_data
        self.known_map            = shared_data['known_map']
        self.discovered_resources = shared_data['discovered_resources']
        self.last_seen_enemies    = shared_data['last_seen_enemies']
        self.risk_map             = shared_data['risk_map']

        # Q-learning compartido
        self.q_learning = q_learning

        # Estado interno
        self.current_action = 'EXPLORE'
        self.current_target = None
        self.current_path   = []
        self.prev_state     = None
        self.prev_action    = None
        self.next_position  = position

        # Referencia al grid
        self._last_grid      = None

        # Datos de aliados actualizados en decide()
        self._all_collectors = []
        self._all_guards     = []

        # Para cálculo de recompensas de acercamiento
        self._prev_dist_explore = None
        self._prev_dist_enemy   = None
        self._prev_dist_ally    = None
        self._cells_explored_this_tick = 0

        # Recompensas de eventos pendientes
        self._pending_reward = 0.0

    # ===================================================================
    # PERCEPCIÓN
    # ===================================================================

    def perceive(self, grid, current_tick):
        """
        Observa celdas dentro de vision_range.
        Retorna (visible_enemies, visible_allies, visible_collectors).
        """
        px, py = self.position
        visible_enemies    = []
        visible_allies     = []
        visible_collectors = []
        self._cells_explored_this_tick = 0

        for dx in range(-self.vision_range, self.vision_range + 1):
            for dy in range(-self.vision_range, self.vision_range + 1):
                if abs(dx) + abs(dy) > self.vision_range:
                    continue
                nx, ny = px + dx, py + dy
                if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                    continue

                cell = grid.cells[nx][ny]
                if not self.known_map[nx][ny].get('explored', False):
                    self._cells_explored_this_tick += 1
                self.known_map[nx][ny]['explored']        = True
                self.known_map[nx][ny]['last_seen']        = current_tick
                self.known_map[nx][ny]['last_known_type']  = cell.type

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
    # CÁLCULO DE PELIGRO
    # ===================================================================

    def _hunter_positions_recent(self, current_tick):
        """Posiciones de cazadores vistos en los últimos DANGER_RANGE ticks."""
        return [
            e['position'] for e in self.last_seen_enemies
            if current_tick - e['tick'] <= _DANGER_RANGE
        ]

    def _compute_danger(self, agent, hunter_positions, all_guards, is_collector):
        """
        Calcula el peligro de un agente aliado.
        Busca cazadores en radio 4 alrededor de la POSICIÓN DEL ALIADO.
        Recolector: 0 + 3×(cazadores_a_≤4_celdas) - 2×(guardias_a_≤4_celdas)
        Guardia:   −1 + 1×(cazadores_a_≤4_celdas)
        """
        ax, ay = agent.position

        # Contar cazadores EN RANGO 4 DE LA POSICIÓN DEL ALIADO
        cazadores_cerca = sum(
            1 for hp in self.last_seen_enemies  # Usar last_seen_enemies, no hunter_positions param
            if abs(hp['position'][0] - ax) + abs(hp['position'][1] - ay) <= _DANGER_RANGE
        )

        if is_collector:
            guardias_cerca = sum(
                1 for g in all_guards
                if g.is_alive
                and abs(g.position[0] - ax) + abs(g.position[1] - ay) <= _GUARD_PROTECT_RANGE
            )
            return 0 + 3 * cazadores_cerca - 2 * guardias_cerca
        else:
            # Guardia
            return -1 + 1 * cazadores_cerca
    
    def _compute_all_dangers(self, all_collectors, all_guards, hunter_positions):
        """
        Retorna dict {agent: danger} para todos los aliados vivos.
        """
        dangers = {}
        for c in all_collectors:
            if c.is_alive:
                dangers[c] = self._compute_danger(c, hunter_positions, all_guards, True)
        for g in all_guards:
            if g.is_alive:
                dangers[g] = self._compute_danger(g, hunter_positions, all_guards, False)
        return dangers

    # ===================================================================
    # ESTADO RL (4 variables)
    # ===================================================================

    def build_state(self, visible_enemies, all_collectors, all_guards, current_tick):
        """
        Tupla discreta de 4 variables:
          (ally_in_danger, i_am_in_danger, hunter_near, can_attack)

        ally_in_danger: 1 si algún aliado tiene peligro > 0
        i_am_in_danger: 1 si el guardia tiene peligro > 0
        hunter_near:    1 si hay cazador visible
        can_attack:     1 si hay cazador en rango y cooldown == 0
        """
        px, py = self.position
        hunter_positions = self._hunter_positions_recent(current_tick)
        dangers = self._compute_all_dangers(all_collectors, all_guards, hunter_positions)

        my_danger = dangers.get(self, self._compute_danger(
            self, hunter_positions, all_guards, False
        ))

        ally_in_danger = int(
            any(d > 0 for agent, d in dangers.items() if agent is not self)
        )
        i_am_in_danger = int(my_danger > 0)
        hunter_near    = 1 if visible_enemies else 0

        can_attack = int(self.current_cooldown == 0)

        return (ally_in_danger, i_am_in_danger, hunter_near, can_attack)

    # ===================================================================
    # HEURISTIC BIASES
    # ===================================================================

    def calculate_heuristic_biases(self, state):
        """
        Sesgos heurísticos para las 4 acciones según el estado.
        """
        ally_in_danger, i_am_in_danger, hunter_near, can_attack = state

        biases = {a: 0.0 for a in ACTIONS}

        # HUIR: domina si está en peligro personal
        if i_am_in_danger:
            biases['FLEE'] += HEUR_GUARD_FLEE_DANGER

        # DEFENDER: prioridad si aliado en peligro y el guardia no está en peligro
        if ally_in_danger:
            biases['DEFEND'] += HEUR_GUARD_DEFEND_ALLY

        if not ally_in_danger:
            if can_attack and hunter_near:
                biases['ATTACK'] += HEUR_GUARD_ATTACK
        
        # EXPLORAR: comportamiento base
        biases['EXPLORE'] += HEUR_GUARD_EXPLORE_BASE

        return biases

    # ===================================================================
    # TOMA DE DECISIÓN
    # ===================================================================

    def decide(self, grid, current_tick, all_collectors, all_guards=None):
        """
        Ciclo completo por tick:
        percibir → estado → biases → acción → ejecutar
        """
        self._last_grid      = grid
        self._all_collectors = [c for c in all_collectors if c.is_alive]
        self._all_guards     = [g for g in (all_guards or []) if g.is_alive]

        if not self.is_alive:
            return

        if self.current_cooldown > 0:
            self.current_cooldown -= 1

        visible_enemies, visible_allies, visible_collectors = self.perceive(
            grid, current_tick
        )

        state = self.build_state(
            visible_enemies, self._all_collectors, self._all_guards, current_tick
        )
        ally_in_danger, i_am_in_danger, hunter_near, can_attack = state

        biases = self.calculate_heuristic_biases(state)

        # Q-update con recompensa de paso + eventos pendientes
        if self.prev_state is not None and self.prev_action is not None:
            step_reward = self._compute_step_reward(
                self.prev_action, state, visible_enemies, current_tick
            )
            step_reward += self._pending_reward
            self._pending_reward = 0.0
            self.q_learning.update(self.prev_state, self.prev_action, step_reward, state)

        action = self.q_learning.get_action(state, biases)
        self.current_action = action
        self.prev_state     = state
        self.prev_action    = action

        # Ejecutar acción
        if action == 'ATTACK':
            # Prioridad: 1) visible_enemies (en vision_range)
            #           2) Si no visible, buscar enemy registrado por aliados (last_seen_enemies)
            target = None
            if visible_enemies:
                target = self._closest_enemy(visible_enemies)
            else:
                # Buscar cazador más recientemente visto (aunque no esté en vision_range ahora)
                if self.last_seen_enemies:
                    most_recent = max(
                        self.last_seen_enemies,
                        key=lambda e: e['tick']
                    )
                    target_pos = most_recent['position']
                    self.current_path = self._astar(target_pos)
                    self.current_target = target_pos
                    if self.current_path:
                        self.next_position = self.current_path.pop(0)
                    else:
                        self.next_position = self.position
                    return self.next_position
            
            if target:
                self.current_path = self._astar(target.position)
            else:
                self.current_path = []
        elif action == 'DEFEND':
            self.current_path = self.execute_defend(current_tick)
        elif action == 'FLEE':
            self.current_path = self.execute_flee()
        else:  # EXPLORE
            best_explore = self._find_best_explore_cell()
            self.current_path = self.execute_explore(best_explore)

        # Actualizar distancias previas para siguiente tick
        self._update_prev_distances(visible_enemies, current_tick)

        if self.current_path:
            self.next_position = self.current_path.pop(0)
        else:
            self.next_position = self.position

        return self.next_position

    # ===================================================================
    # RECOMPENSAS POR TICK
    # ===================================================================

    def _compute_step_reward(self, action, state, visible_enemies, current_tick):
        """
        Calcula recompensa de paso basada en acción y estado.
        """
        ally_in_danger, i_am_in_danger, hunter_near, can_attack = state
        reward = 0.0
        px, py = self.position

        # Celdas que este guardia descubrió en su perceive() de este tick
        if self._cells_explored_this_tick > 0:
            reward += REWARD_GUARD_CELL_EXPLORED * self._cells_explored_this_tick

        # Cazadores cerca: recompensa/penalización según situación
        if hunter_near:
            if i_am_in_danger:
                reward += REWARD_GUARD_HUNTER_IN_DANGER
            else:
                reward += REWARD_GUARD_HUNTER_NO_DANGER
            if not can_attack:
                reward += REWARD_GUARD_CANT_ATTACK

        # Recompensas por acción específica
        if action == 'EXPLORE':
            if ally_in_danger:
                reward += REWARD_GUARD_EXPLORE_DANGER
            else:
                # Acercamiento a celda de exploración
                best = self._find_best_explore_cell()
                if best is not None and self._prev_dist_explore is not None:
                    curr = abs(best[0] - px) + abs(best[1] - py)
                    if curr < self._prev_dist_explore:
                        reward += REWARD_GUARD_APPROACH_EXPLORE

        elif action == 'ATTACK':
            if visible_enemies and self.current_cooldown == 0:
                closest = self._closest_enemy(visible_enemies)
                if closest and self._prev_dist_enemy is not None:
                    curr = abs(closest.position[0] - px) + abs(closest.position[1] - py)
                    if curr < self._prev_dist_enemy:
                        reward += REWARD_GUARD_APPROACH_HUNTER

        elif action == 'FLEE':
            if i_am_in_danger:
                reward += REWARD_GUARD_FLEE_DANGER
            elif not can_attack:
                reward += REWARD_GUARD_FLEE_NO_ATTACK

        elif action == 'DEFEND':
            # Acercamiento al aliado más en peligro
            most_dangerous = self._most_dangerous_ally(current_tick)
            if most_dangerous is not None and self._prev_dist_ally is not None:
                ap = most_dangerous.position
                curr = abs(ap[0] - px) + abs(ap[1] - py)
                if curr < self._prev_dist_ally:
                    reward += REWARD_GUARD_APPROACH_ALLY

        return reward

    def _update_prev_distances(self, visible_enemies, current_tick):
        px, py = self.position

        best_explore = self._find_best_explore_cell()
        if best_explore is not None:
            self._prev_dist_explore = abs(best_explore[0] - px) + abs(best_explore[1] - py)
        else:
            self._prev_dist_explore = None

        if visible_enemies:
            closest = self._closest_enemy(visible_enemies)
            if closest:
                self._prev_dist_enemy = (abs(closest.position[0] - px)
                                         + abs(closest.position[1] - py))
            else:
                self._prev_dist_enemy = None
        else:
            self._prev_dist_enemy = None

        most_danger = self._most_dangerous_ally(current_tick)
        if most_danger is not None:
            ap = most_danger.position
            self._prev_dist_ally = abs(ap[0] - px) + abs(ap[1] - py)
        else:
            self._prev_dist_ally = None

    # ===================================================================
    # EJECUCIÓN DE ACCIONES
    # ===================================================================

    def execute_explore(self, best_explore_cell):
        if best_explore_cell is None:
            return []
        self.current_target = best_explore_cell
        return self._astar(best_explore_cell)

    def execute_defend(self, current_tick):
        """
        Se mueve hacia el aliado con más peligro.
        Si el aliado más peligroso es el guardia mismo, se queda.
        """
        most_dangerous = self._most_dangerous_ally(current_tick)
        if most_dangerous is None:
            return []
        self.current_target = most_dangerous.position
        return self._astar(most_dangerous.position)

    def execute_flee(self):
        """
        Busca la ruta más segura hacia un guardia o torre conocida.
        Si no hay opciones, se queda en posición actual.
        """
        px, py = self.position
        options = []

        for g in self._all_guards:
            if g is not self:
                d = abs(g.position[0] - px) + abs(g.position[1] - py)
                options.append((d, g.position))

        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                if self.known_map[x][y].get('last_known_type') == 'tower':
                    d = abs(x - px) + abs(y - py)
                    options.append((d, (x, y)))

        if options:
            options.sort()
            target = options[0][1]
            self.current_target = target
            return self._astar_secure(target)
        else:
            # Sin opciones: se queda en posición actual
            return []

    def execute_attack(self, target):
        """Verifica ataque (el hit real lo resuelve el environment)."""
        if target is None or not target.is_alive:
            return False
        px, py = self.position
        dist = abs(target.position[0] - px) + abs(target.position[1] - py)
        return dist <= self.attack_range and self.current_cooldown == 0

    # ===================================================================
    # COMBATE
    # ===================================================================

    def can_attack(self, target):
        if not self.is_alive or self.current_cooldown > 0:
            return False
        if not hasattr(target, 'is_alive') or not target.is_alive:
            return False
        return True

    # ===================================================================
    # RECOMPENSAS DE EVENTO
    # ===================================================================

    def get_reward(self, event):
        rewards = {
            'kill_hunter': REWARD_KILL_HUNTER,
            'die':         REWARD_GUARD_DIE,
        }
        return rewards.get(event, 0.0)

    def receive_reward(self, event):
        r = self.get_reward(event)
        if event == 'kill_hunter' and self.prev_state is not None and self.prev_state[0]:
            r += REWARD_KILL_HUNTER_DEFEND
        self._pending_reward += r

    # ===================================================================
    # MUERTE
    # ===================================================================

    def die(self):
        self.is_alive = False

    # ===================================================================
    # PATHFINDING
    # ===================================================================

    def _astar(self, goal):
        return find_path(
            self.position,
            goal,
            self._last_grid,
            self._cost_function,
            known_map=self.known_map,
        )
    
    def _astar_secure(self, goal):
        return find_path_secure(
            self.position,
            goal,
            self._last_grid,
            self._cost_function,
            known_map=self.known_map,
        )

    def _cost_function(self, _, neighbor_pos):
        """
        Costo = 1 + risk_map[x][y].
        Valores negativos (cerca de guardias/torres) = rutas más baratas (preferidas).
        Valores positivos (cerca de cazadores) = rutas más caras (evitadas).
        """
        nx, ny = neighbor_pos
        cost = 1.0 + self.risk_map[nx][ny]
        return max(0.1, cost)

    # ===================================================================
    # UTILIDADES
    # ===================================================================

    def _closest_enemy(self, visible_enemies):
        if not visible_enemies:
            return None
        px, py = self.position
        return min(
            visible_enemies,
            key=lambda e: abs(e.position[0] - px) + abs(e.position[1] - py)
        )

    def _most_dangerous_ally(self, current_tick):
        """
        Retorna el aliado (no yo mismo) con mayor nivel de peligro > 0.
        Retorna None si ninguno está en peligro.
        """
        # Ya no necesita hunter_positions porque _compute_danger usa self.last_seen_enemies
        best_agent  = None
        best_danger = 0  # solo nos interesan los que tienen peligro > 0

        for c in self._all_collectors:
            d = self._compute_danger(c, None, self._all_guards, True)
            if d > best_danger:
                best_danger = d
                best_agent  = c

        for g in self._all_guards:
            if g is not self:
                d = self._compute_danger(g, None, self._all_guards, False)
                if d > best_danger:
                    best_danger = d
                    best_agent  = g

        return best_agent

    def _find_best_explore_cell(self):
        """
        Mejor celda frontera a explorar.
        Criterios:
          + Celdas desconocidas alrededor (más = mejor)
          - Distancia al guardia (menos = mejor)
          + Distancia mínima a otros aliados (más lejos = más valioso)
          - Riesgo (más negativo = más seguro = mejor)
        """
        px, py = self.position
        all_allies = self._all_collectors + [
            g for g in self._all_guards if g is not self
        ]

        best_cell  = None
        best_score = float('-inf')

        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                if self.known_map[x][y].get('explored', False):
                    continue
                is_frontier = any(
                    0 <= x + ddx < MAP_WIDTH and 0 <= y + ddy < MAP_HEIGHT
                    and self.known_map[x + ddx][y + ddy].get('explored', False)
                    for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                )
                if not is_frontier:
                    continue

                unknown_count = sum(
                    1
                    for ddx in range(-4, 5) for ddy in range(-4, 5)
                    if abs(ddx) + abs(ddy) <= 4
                    and 0 <= x + ddx < MAP_WIDTH and 0 <= y + ddy < MAP_HEIGHT
                    and not self.known_map[x + ddx][y + ddy].get('explored', False)
                )

                dist_to_me = abs(x - px) + abs(y - py)

                if all_allies:
                    min_ally_dist = min(
                        abs(x - a.position[0]) + abs(y - a.position[1])
                        for a in all_allies
                    )
                else:
                    min_ally_dist = MAP_WIDTH + MAP_HEIGHT

                risk = self.risk_map[x][y]

                score = (unknown_count * 5.0
                         - dist_to_me * 0.5
                         + min_ally_dist * 0.3
                         - risk * 2.0)

                if score > best_score:
                    best_score = score
                    best_cell  = (x, y)

        return best_cell

