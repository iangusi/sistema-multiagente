from utils.constants import (
    COLLECTOR_HP,
    COLLECTOR_SPEED,
    COLLECTOR_VISION,
    COLLECTOR_CARRY_CAPACITY,
    COLLECT_RATE,
    BASE_POSITION,
    MAP_WIDTH,
    MAP_HEIGHT,
    HEUR_FLEE_HUNTER,
    HEUR_RETURN_FULL,
    HEUR_BUILD_HAS_KIT,
    HEUR_GOTO_RESOURCE_BASE,
    REWARD_APPROACH_EXPLORE,
    REWARD_CELL_EXPLORED,
    REWARD_APPROACH_RESOURCE,
    REWARD_COLLECT,
    REWARD_GOING_TO_BASE_RES,
    REWARD_DELIVER_RESOURCES,
    REWARD_APPROACH_BUILD,
    REWARD_BUILD_NO_KIT,
    REWARD_BUILD_TOWER,
    REWARD_GUARD_NEARBY,
    REWARD_TOWER_NEARBY,
    REWARD_FLEE_HUNTER,
    REWARD_COLLECTOR_DIE,
    REWARD_BASE_NO_RES,
    REWARD_RESOURCE_FULL,
    REWARD_EXPLORE_WITH_KIT,
    REWARD_BAD_ACTION_HUNTER,
    REWARD_FLEE_NO_HUNTER,
    REWARD_APPROACH_BASE,
)
from pathfinding.astar_secure import find_path

# Acciones disponibles del recolector
ACTIONS = ['EXPLORE', 'GO_TO_RESOURCE', 'RETURN_TO_BASE', 'FLEE', 'BUILD_TOWER']

# Umbrales de distancia para discretización del estado
_DIST_BUILD_CLOSE    = 5   # build target se considera "cerca"
_DIST_EXPLORE_CLOSE  = 8   # celda explorar se considera "cerca"
_DIST_RESOURCE_CLOSE = 5   # recurso se considera "cerca"

# Radio de "cerca" para peligro en el cálculo de aliados
_ALLY_NEAR_RANGE = 5


class Collector:
    """
    Recolector del Equipo A.

    Estado RL (5 variables):
      (build_dist, explore_dist, resource_dist, can_carry, hunter_near)

    Acciones: EXPLORE, GO_TO_RESOURCE, RETURN_TO_BASE, FLEE, BUILD_TOWER

    Pathfinding A* con mapa de riesgo (costo = 1 + risk_map).
    Muerte permanente.
    """

    def __init__(self, position, shared_data, q_learning):
        self.position           = position
        self.hp                 = COLLECTOR_HP
        self.current_hp         = COLLECTOR_HP
        self.speed              = COLLECTOR_SPEED
        self.vision_range       = COLLECTOR_VISION
        self.carrying_resources = 0
        self.carrying_capacity  = COLLECTOR_CARRY_CAPACITY
        self.has_build_kit      = False
        self.is_alive           = True

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

        # Señalización de construcción
        self.wants_to_build = False
        self.build_target   = None

        # Referencia al grid (actualizada en decide)
        self._last_grid = None

        # Para cálculo de recompensas de acercamiento
        self._prev_dist_explore  = None
        self._prev_dist_resource = None
        self._prev_dist_build    = None
        self._prev_dist_base     = None
        self._cells_explored_this_tick = 0

        # Recompensas de eventos pendientes
        self._pending_reward = 0.0

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
    # CÁLCULO DE OBJETIVOS (pre-estado)
    # ===================================================================

    def _find_best_build_cell(self):
        """
        Mejor celda para construir una torre: maximiza cobertura de riesgo
        positivo, recursos y frontera inexplorada en radio 4.
        """
        px, py = self.position
        best_cell  = None
        best_score = float('-inf')

        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                if not self.known_map[x][y].get('explored', False):
                    continue
                ctype = self.known_map[x][y]['last_known_type']
                if ctype not in ('empty', 'resource'):
                    continue

                # Riesgo positivo en radio: torres son más útiles donde hay amenaza
                risk_coverage = 0.0
                resource_coverage = 0
                frontier_coverage = 0
                tower_redundancy  = 0

                for ddx in range(-7, 8):
                    for ddy in range(-7, 8):
                        nx2, ny2 = x + ddx, y + ddy
                        if not (0 <= nx2 < MAP_WIDTH and 0 <= ny2 < MAP_HEIGHT):
                            continue
                        if self.known_map[nx2][ny2].get('last_known_type') == 'tower':
                            tower_redundancy += 1
                        if abs(ddx) + abs(ddy) <= 4:
                            risk_coverage += max(0.0, self.risk_map[nx2][ny2])
                            if self.known_map[nx2][ny2].get('last_known_type') == 'resource':
                                resource_coverage += 1
                            if not self.known_map[nx2][ny2].get('explored', False):
                                frontier_coverage += 1

                dist  = abs(x - px) + abs(y - py)
                score = (risk_coverage * 3.0
                         + resource_coverage * 3.0
                         + frontier_coverage * 1.0
                         - tower_redundancy * 5.0
                         - dist * 0.5)
                if score > best_score:
                    best_score = score
                    best_cell  = (x, y)

        return best_cell

    def _find_best_explore_cell(self, all_collectors, all_guards):
        """
        Mejor celda frontera a explorar.
        Criterios:
          + Celdas desconocidas alrededor (más = mejor)
          - Distancia al recolector (menos = mejor)
          + Distancia mínima a otros aliados (más = mejor, evita duplicar cobertura)
          - Riesgo en la celda (más negativo = más seguro = mejor)
        """
        px, py = self.position

        # Aliados vivos (para penalizar celdas que ya están cubiertas)
        alive_allies = (
            [c for c in all_collectors if c.is_alive and c is not self]
            + [g for g in all_guards if g.is_alive]
        )

        best_cell  = None
        best_score = float('-inf')

        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                if self.known_map[x][y].get('explored', False):
                    continue
                # Solo fronteras (adyacente a explorado)
                is_frontier = any(
                    0 <= x + ddx < MAP_WIDTH and 0 <= y + ddy < MAP_HEIGHT
                    and self.known_map[x + ddx][y + ddy].get('explored', False)
                    for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                )
                if not is_frontier:
                    continue

                # Celdas desconocidas en radio 4
                unknown_count = sum(
                    1
                    for ddx in range(-4, 5) for ddy in range(-4, 5)
                    if abs(ddx) + abs(ddy) <= 4
                    and 0 <= x + ddx < MAP_WIDTH and 0 <= y + ddy < MAP_HEIGHT
                    and not self.known_map[x + ddx][y + ddy].get('explored', False)
                )

                dist_to_me = abs(x - px) + abs(y - py)

                # Distancia al aliado más cercano (más lejos = más valioso explorar ahí)
                if alive_allies:
                    min_ally_dist = min(
                        abs(x - a.position[0]) + abs(y - a.position[1])
                        for a in alive_allies
                    )
                else:
                    min_ally_dist = MAP_WIDTH + MAP_HEIGHT

                risk = self.risk_map[x][y]

                score = (unknown_count * 5.0
                         - dist_to_me * 0.5
                         + min_ally_dist * 0.1
                         - risk * 1.0)   # más negativo = más seguro = mejor

                if score > best_score:
                    best_score = score
                    best_cell  = (x, y)

        return best_cell

    def _find_closest_resource(self):
        """Recurso conocido más cercano al recolector, o None."""
        if not self.discovered_resources:
            return None
        px, py = self.position
        return min(
            self.discovered_resources,
            key=lambda r: abs(r['position'][0] - px) + abs(r['position'][1] - py)
        )

    # ===================================================================
    # ESTADO RL (5 variables)
    # ===================================================================

    def build_state(self, visible_enemies, visible_allies,
                    best_build_cell, best_explore_cell, best_resource):
        """
        Tupla discreta de 5 variables para Q-learning:
          (build_dist, explore_dist, resource_dist, can_carry, hunter_near)

        build_dist:    0=sin kit, 1=kit cerca(<=5), 2=kit lejos(>5)
        explore_dist:  0=sin frontera, 1=cerca(<=8), 2=lejos(>8)
        resource_dist: 0=sin recursos, 1=cerca(<=5), 2=lejos(>5)
        can_carry:     0=lleno, 1=puede cargar más
        hunter_near:   0=no, 1=sí
        """
        px, py = self.position

        # Build
        if not self.has_build_kit or best_build_cell is None:
            build_dist = 0
        else:
            d = abs(best_build_cell[0] - px) + abs(best_build_cell[1] - py)
            build_dist = 1 if d <= _DIST_BUILD_CLOSE else 2

        # Explore
        if best_explore_cell is None:
            explore_dist = 0
        else:
            d = abs(best_explore_cell[0] - px) + abs(best_explore_cell[1] - py)
            explore_dist = 1 if d <= _DIST_EXPLORE_CLOSE else 2

        # Resource
        if best_resource is None:
            resource_dist = 0
        else:
            d = abs(best_resource['position'][0] - px) + abs(best_resource['position'][1] - py)
            resource_dist = 1 if d <= _DIST_RESOURCE_CLOSE else 2

        can_carry   = 0 if self.carrying_resources >= self.carrying_capacity else 1
        hunter_near = 1 if visible_enemies else 0

        return (build_dist, explore_dist, resource_dist, can_carry, hunter_near)

    # ===================================================================
    # HEURISTIC BIASES
    # ===================================================================

    def calculate_heuristic_biases(self, state):
        """
        Sesgos heurísticos sobre Q-values según el estado actual.
        """
        build_dist, explore_dist, resource_dist, can_carry, hunter_near = state

        biases = {a: 0.0 for a in ACTIONS}
        '''
        # HUIR: domina cuando hay cazador visible
        if hunter_near:
            biases['FLEE'] += HEUR_FLEE_HUNTER
        # IR_A_BASE: domina cuando el inventario está lleno
        elif can_carry == 0:
            biases['RETURN_TO_BASE'] += HEUR_RETURN_FULL
        # CONSTRUIR: se promueve si tiene kit
        elif build_dist > 0:
            biases['BUILD_TOWER'] += HEUR_BUILD_HAS_KIT
        # IR_POR_RECURSO: se promueve si hay recursos conocidos y puede cargar
        elif resource_dist > 0:
            biases['GO_TO_RESOURCE'] += HEUR_GOTO_RESOURCE_BASE
        '''
        return biases

    # ===================================================================
    # RECOMPENSAS POR TICK
    # ===================================================================

    def _compute_step_reward(self, action, visible_allies, hunter_near,
                              best_explore_cell, best_resource, best_build_cell):
        """
        Calcula recompensa de paso basada en el estado actual y la acción tomada.
        """
        reward = 0.0
        px, py = self.position

        # Celdas que este recolector descubrió en su perceive() de este tick
        if self._cells_explored_this_tick > 0:
            reward += REWARD_CELL_EXPLORED * self._cells_explored_this_tick

        guard_near = any(a.__class__.__name__ == 'Guard' for a in visible_allies)
        tower_near = any(a.__class__.__name__ == 'Tower' for a in visible_allies)

        # Presencia de aliados cerca
        if guard_near:
            reward += REWARD_GUARD_NEARBY
        if tower_near:
            reward += REWARD_TOWER_NEARBY

        # --- Recompensas por acción ---
        if action == 'FLEE':
            if hunter_near:
                reward += REWARD_FLEE_HUNTER
            else:
                reward += REWARD_FLEE_NO_HUNTER

        elif action == 'RETURN_TO_BASE':
            if self.carrying_resources > 0:
                reward += REWARD_GOING_TO_BASE_RES
                bx, by = BASE_POSITION
                curr_dist_base = abs(bx - px) + abs(by - py)
                if self._prev_dist_base is not None and curr_dist_base < self._prev_dist_base:
                    reward += REWARD_APPROACH_BASE
            else:
                reward += REWARD_BASE_NO_RES
            if hunter_near:
                reward += REWARD_BAD_ACTION_HUNTER

        elif action == 'GO_TO_RESOURCE':
            if self.carrying_resources >= self.carrying_capacity:
                reward += REWARD_RESOURCE_FULL
            elif best_resource is not None and self._prev_dist_resource is not None:
                rx, ry = best_resource['position']
                curr_dist = abs(rx - px) + abs(ry - py)
                if curr_dist < self._prev_dist_resource:
                    reward += REWARD_APPROACH_RESOURCE
            if hunter_near:
                reward += REWARD_BAD_ACTION_HUNTER

        elif action == 'EXPLORE':
            if self.has_build_kit:
                reward += REWARD_EXPLORE_WITH_KIT
            if best_explore_cell is not None and self._prev_dist_explore is not None:
                ex, ey = best_explore_cell
                curr_dist = abs(ex - px) + abs(ey - py)
                if curr_dist < self._prev_dist_explore:
                    reward += REWARD_APPROACH_EXPLORE
            if hunter_near:
                reward += REWARD_BAD_ACTION_HUNTER

        elif action == 'BUILD_TOWER':
            if not self.has_build_kit:
                reward += REWARD_BUILD_NO_KIT
            elif best_build_cell is not None and self._prev_dist_build is not None:
                bx2, by2 = best_build_cell
                curr_dist = abs(bx2 - px) + abs(by2 - py)
                if curr_dist < self._prev_dist_build:
                    reward += REWARD_APPROACH_BUILD

        return reward

    def _update_prev_distances(self, best_explore_cell, best_resource, best_build_cell):
        """Guarda distancias actuales para el siguiente tick."""
        px, py = self.position
        if best_explore_cell is not None:
            ex, ey = best_explore_cell
            self._prev_dist_explore = abs(ex - px) + abs(ey - py)
        else:
            self._prev_dist_explore = None

        if best_resource is not None:
            rx, ry = best_resource['position']
            self._prev_dist_resource = abs(rx - px) + abs(ry - py)
        else:
            self._prev_dist_resource = None

        if best_build_cell is not None and self.has_build_kit:
            bx2, by2 = best_build_cell
            self._prev_dist_build = abs(bx2 - px) + abs(by2 - py)
        else:
            self._prev_dist_build = None

        bx, by = BASE_POSITION
        self._prev_dist_base = abs(bx - px) + abs(by - py)

    # ===================================================================
    # EJECUCIÓN DE ACCIONES
    # ===================================================================

    def execute_explore(self, best_explore_cell):
        if best_explore_cell is None:
            return []
        self.current_target = best_explore_cell
        return self._astar(best_explore_cell)

    def execute_go_to_resource(self, best_resource, all_collectors):
        """Va al recurso más cercano conocido."""
        if best_resource is None:
            # Sin recursos: explorar
            return []
        self.current_target = best_resource['position']
        return self._astar(best_resource['position'])

    def execute_return_to_base(self):
        self.current_target = BASE_POSITION
        return self._astar(BASE_POSITION)

    def execute_flee(self, visible_enemies, all_guards):
        """
        Huye hacia la ruta MÁS SEGURA.
        Considera TODOS los aliados disponibles (no solo visibles):
        - Otros recolectores vivos
        - Guardias vivos
        - Torres conocidas (en known_map)
        
        Prioridad: Torre > Guardia > Recolector > Base
        Selecciona el destino con la ruta de menor costo acumulado en risk_map.
        """
        px, py = self.position
        options = []

        # 1. Torres conocidas (en known_map, sin importar si están visibles)
        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                if self.known_map[x][y].get('last_known_type') == 'tower':
                    options.append({'type': 'tower', 'pos': (x, y), 'priority': 1})

        # 2. Guardias vivos (todos, no solo visibles)
        for g in all_guards:
            if g.is_alive:
                options.append({'type': 'guard', 'pos': g.position, 'priority': 2})

        if not options:
            return []

        # Ordenar por prioridad: torres primero
        options.sort(key=lambda o: o['priority'])

        # Evaluar el costo de A* para cada opción
        best_option = None
        best_path = []
        best_cost = float('inf')

        for opt in options:
            target_pos = opt['pos']
            # Calcular la ruta usando A* (que respeta risk_map)
            path = self._astar(target_pos)
            if path:
                # Costo acumulado de la ruta según risk_map
                route_cost = 0.0
                for path_pos in path:
                    px_path, py_path = path_pos
                    route_cost += self.risk_map[px_path][py_path]
                
                # Penalizar si el destino está cerca del cazador visible
                for e in visible_enemies:
                    ed = abs(target_pos[0] - e.position[0]) + abs(target_pos[1] - e.position[1])
                    route_cost += max(0.0, (5.0 - ed) * 2.0)
                
                if route_cost < best_cost:
                    best_cost = route_cost
                    best_option = opt
                    best_path = path
            else:
                # Si A* falla, ignorar esta opción
                continue

        if best_option:
            self.current_target = best_option['pos']
            return best_path
        else:
            # Ninguna opción viable: quedarse quieto
            return []

    def execute_build_tower(self, best_build_cell):
        """Se mueve hacia la mejor celda de construcción."""
        if best_build_cell is None:
            return []

        if self.build_target is None:
            self.build_target = best_build_cell

        self.current_target = self.build_target

        if self.position == self.build_target:
            self.wants_to_build = True
            return []

        return self._astar(self.build_target)

    # ===================================================================
    # RECOLECCIÓN Y DEPÓSITO
    # ===================================================================

    def collect_resource(self, cell):
        if cell.resource_amount <= 0:
            return 0
        free      = self.carrying_capacity - self.carrying_resources
        collected = min(COLLECT_RATE, cell.resource_amount, free)
        self.carrying_resources += collected
        cell.resource_amount    -= collected
        return collected

    def deposit_resources(self):
        amount = self.carrying_resources
        self.carrying_resources = 0
        return amount

    def receive_build_kit(self):
        self.has_build_kit = True

    # ===================================================================
    # RECOMPENSAS DE EVENTO
    # ===================================================================

    def get_reward(self, event):
        rewards = {
            'collect':           REWARD_COLLECT,
            'deliver_resources': REWARD_DELIVER_RESOURCES,
            'build_tower':       REWARD_BUILD_TOWER,
            'die':               REWARD_COLLECTOR_DIE,
        }
        return rewards.get(event, 0.0)

    def receive_reward(self, event):
        self._pending_reward += self.get_reward(event)

    # ===================================================================
    # MUERTE
    # ===================================================================

    def die(self):
        self.is_alive           = False
        self.carrying_resources = 0
        self.has_build_kit      = False

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

    def _cost_function(self, _, neighbor_pos):
        """
        Costo = 1 + risk_map[x][y].
        Valores negativos (cerca de guardias/torres) = rutas más baratas.
        Valores positivos (cerca de cazadores) = rutas más caras.
        """
        nx, ny = neighbor_pos
        cost = 1.0 + self.risk_map[nx][ny]
        return max(0.1, cost)

    # ===================================================================
    # TOMA DE DECISIÓN
    # ===================================================================

    def decide(self, grid, current_tick, all_collectors, all_guards=None):
        """
        Ciclo completo por tick:
        percibir → calcular objetivos → construir estado → biases → acción
        """
        self._last_grid = grid
        if not self.is_alive:
            return

        if all_guards is None:
            all_guards = []

        visible_enemies, _, visible_allies = self.perceive(grid, current_tick)

        # Pre-computar objetivos (usados tanto para estado como para ejecución)
        best_build_cell   = self._find_best_build_cell() if self.has_build_kit else None
        best_explore_cell = self._find_best_explore_cell(all_collectors, all_guards)
        best_resource     = self._find_closest_resource()

        # Regla directa: si está en la mejor celda de construcción, señalizar
        self.wants_to_build = False
        if self.has_build_kit and best_build_cell is not None and self.position == best_build_cell:
            self.wants_to_build = True
            # También actualizar build_target para que el environment lo valide
            self.build_target = best_build_cell

        state = self.build_state(
            visible_enemies, visible_allies,
            best_build_cell, best_explore_cell, best_resource
        )

        hunter_near = state[4]
        biases = self.calculate_heuristic_biases(state)

        # Q-update con transición anterior + recompensa de paso + eventos pendientes
        if self.prev_state is not None and self.prev_action is not None:
            step_reward = self._compute_step_reward(
                self.prev_action, visible_allies, hunter_near,
                best_explore_cell, best_resource, best_build_cell
            )
            step_reward += self._pending_reward
            self._pending_reward = 0.0
            self.q_learning.update(self.prev_state, self.prev_action, step_reward, state)

        action = self.q_learning.get_action(state, biases)
        self.current_action = action
        self.prev_state     = state
        self.prev_action    = action

        # Guardar distancias para siguiente tick
        self._update_prev_distances(best_explore_cell, best_resource, best_build_cell)

        # Ejecutar acción
        if action == 'EXPLORE':
            self.current_path = self.execute_explore(best_explore_cell)
        elif action == 'GO_TO_RESOURCE':
            self.current_path = self.execute_go_to_resource(best_resource, all_collectors)
        elif action == 'RETURN_TO_BASE':
            self.current_path = self.execute_return_to_base()
        elif action == 'FLEE':
            self.current_path = self.execute_flee(visible_enemies, all_guards)
        elif action == 'BUILD_TOWER':
            self.current_path = self.execute_build_tower(best_build_cell)

        if self.current_path:
            self.next_position = self.current_path.pop(0)
        else:
            self.next_position = self.position

        return self.next_position
