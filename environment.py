import random

from utils.constants import (
    MAP_WIDTH, MAP_HEIGHT,
    BASE_POSITION,
    NUM_OBSTACLES,
    NUM_RESOURCE_NODES, RESOURCE_MIN_AMOUNT, RESOURCE_MAX_AMOUNT,
    WIN_RESOURCE_PERCENT,
    BUILD_KIT_COST,
    NUM_COLLECTORS, NUM_GUARDS, NUM_HUNTERS,
    HUNTER_MIN_SPAWN_DISTANCE,
    HUNTER_MIN_SPAWN_DISTANCE_ALLY,
    RISK_GUARD, RISK_TOWER, RISK_COLLECTOR, RISK_HUNTER,
    RISK_PROPAGATION_RANGE, RISK_MEMORY_TICKS, RISK_DECAY,
    QL_ALPHA, QL_GAMMA, QL_EPSILON, QL_EPSILON_DECAY, QL_EPSILON_MIN,
    MAX_TOWERS_PER_EXPLORATION,
    CELL_SIZE, SIDEBAR_WIDTH, WINDOW_WIDTH, WINDOW_HEIGHT, FPS,
    COLOR_BG, COLOR_GRID_LINE, COLOR_FOG, COLOR_EXPLORED,
    COLOR_RESOURCE, COLOR_BASE, COLOR_OBSTACLE,
    COLOR_COLLECTOR, COLOR_COLLECTOR_WITH_KIT,
    COLOR_GUARD, COLOR_HUNTER, COLOR_HUNTER_DEAD,
    COLOR_TOWER, COLOR_SIDEBAR_BG, COLOR_TEXT, COLOR_TEXT_HIGHLIGHT,
)
from agents.collector import Collector
from agents.guard import Guard
from agents.hunter import Hunter
from agents.tower import Tower
from rl.q_learning import QLearning
from evolution.genetic_system import random_spawn_position

# Acciones de RL por tipo de agente
_COLLECTOR_ACTIONS = ['EXPLORE', 'GO_TO_RESOURCE', 'RETURN_TO_BASE', 'FLEE', 'BUILD_TOWER']
_GUARD_ACTIONS     = ['EXPLORE', 'ATTACK', 'DEFEND', 'FLEE']


# ---------------------------------------------------------------------------
# Celda del grid
# ---------------------------------------------------------------------------

class Cell:
    """Una celda del grid 2D."""
    __slots__ = ('position', 'type', 'agents', 'tower', 'resource_amount')

    def __init__(self, position):
        self.position        = position
        self.type            = 'empty'
        self.agents          = []          # agentes que ocupan la celda ahora
        self.tower           = None        # Torre si la hay
        self.resource_amount = 0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Environment:
    """
    Motor principal de la simulación multi-agente.

    Gestiona el grid, agentes, recursos, combate, fog of war,
    risk map, build kits, respawn de cazadores y condiciones de victoria.
    """

    def __init__(self, num_hunters_override=None, collector_ql=None,
                 guard_ql=None, headless=False):
        # Modo headless (sin rendering) para entrenamiento
        self.headless = headless

        # -----------------------------------------------------------------
        # 1. Grid de celdas
        # -----------------------------------------------------------------
        self.width  = MAP_WIDTH
        self.height = MAP_HEIGHT
        self.cells  = [
            [Cell((x, y)) for y in range(MAP_HEIGHT)]
            for x in range(MAP_WIDTH)
        ]

        # -----------------------------------------------------------------
        # 2. Obstáculos
        # -----------------------------------------------------------------
        self._place_obstacles()

        # -----------------------------------------------------------------
        # 3. Recursos
        # -----------------------------------------------------------------
        self.total_resources = 0
        self._place_resources()

        # -----------------------------------------------------------------
        # 4. Base
        # -----------------------------------------------------------------
        bx, by = BASE_POSITION
        self.cells[bx][by].type = 'base'

        # -----------------------------------------------------------------
        # 5. Shared data del Equipo A (todo por referencia)
        # -----------------------------------------------------------------
        self.known_map = [
            [{'explored': False, 'last_seen': -1, 'last_known_type': 'empty'}
             for _ in range(MAP_HEIGHT)]
            for _ in range(MAP_WIDTH)
        ]
        # Revelar zona inicial de la base
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = bx + dx, by + dy
                if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT:
                    self.known_map[nx][ny]['explored']       = True
                    self.known_map[nx][ny]['last_seen']       = 0
                    self.known_map[nx][ny]['last_known_type'] = self.cells[nx][ny].type

        self.discovered_resources = []
        self.last_seen_enemies    = []

        self.risk_map = [
            [0.0 for _ in range(MAP_HEIGHT)]
            for _ in range(MAP_WIDTH)
        ]

        self.shared_data = {
            'known_map':           self.known_map,
            'discovered_resources': self.discovered_resources,
            'last_seen_enemies':   self.last_seen_enemies,
            'risk_map':            self.risk_map,
            'explored_count':      0,       # actualizado cada tick
            'claimed_build_targets': set(), # celdas reservadas por recolectores con kit
        }

        # -----------------------------------------------------------------
        # 6. Q-learning (dos instancias separadas o inyectadas desde afuera)
        # -----------------------------------------------------------------
        if collector_ql is not None:
            self.collector_ql = collector_ql
        else:
            self.collector_ql = QLearning(
                actions=_COLLECTOR_ACTIONS,
                alpha=QL_ALPHA, gamma=QL_GAMMA,
                epsilon=QL_EPSILON, epsilon_decay=QL_EPSILON_DECAY,
                epsilon_min=QL_EPSILON_MIN,
            )

        if guard_ql is not None:
            self.guard_ql = guard_ql
        else:
            self.guard_ql = QLearning(
                actions=_GUARD_ACTIONS,
                alpha=QL_ALPHA, gamma=QL_GAMMA,
                epsilon=QL_EPSILON, epsilon_decay=QL_EPSILON_DECAY,
                epsilon_min=QL_EPSILON_MIN,
            )

        # -----------------------------------------------------------------
        # 7–11. Agentes
        # -----------------------------------------------------------------
        self.collectors: list[Collector] = []
        self.guards:     list[Guard]     = []
        self.hunters:    list[Hunter]    = []
        self.towers:     list[Tower]     = []

        # Collectors y guards spawnean cerca de la base
        for i in range(NUM_COLLECTORS):
            pos = self._spawn_near_base(i)
            c = Collector(pos, self.shared_data, self.collector_ql)
            self.collectors.append(c)
            self.cells[pos[0]][pos[1]].agents.append(c)

        for i in range(NUM_GUARDS):
            pos = self._spawn_near_base(NUM_COLLECTORS + i)
            g = Guard(pos, self.shared_data, self.guard_ql)
            self.guards.append(g)
            self.cells[pos[0]][pos[1]].agents.append(g)

        # Hunters spawnean en bordes del mapa, lejos de la base
        num_h = num_hunters_override if num_hunters_override is not None else NUM_HUNTERS
        for _ in range(num_h):
            pos = random_spawn_position(MAP_WIDTH, MAP_HEIGHT,
                                        BASE_POSITION, HUNTER_MIN_SPAWN_DISTANCE)
            h = Hunter(pos)
            self.hunters.append(h)
            self.cells[pos[0]][pos[1]].agents.append(h)

        # -----------------------------------------------------------------
        # 12–18. Estado global
        # -----------------------------------------------------------------
        self.base_resources      = 0
        self.build_kits_available = 0
        self._kits_given          = 0
        self.current_tick         = 0
        self.user_target_position = None
        self.game_over            = False
        self.winner               = None
        self.last_event           = ''

        # win_target: recurso mínimo para ganar (usado por el trainer)
        self.win_target = self.total_resources * WIN_RESOURCE_PERCENT

    # =====================================================================
    # GAME LOOP PRINCIPAL
    # =====================================================================

    def tick(self):
        """Ejecuta UN tick completo de la simulación en orden estricto."""
        if self.game_over:
            return

        alive_collectors = [c for c in self.collectors if c.is_alive]
        alive_guards     = [g for g in self.guards     if g.is_alive]
        alive_hunters    = [h for h in self.hunters    if h.is_alive]
        all_team_a       = alive_collectors + alive_guards

        # 1. PERCEPCIÓN
        for agent in all_team_a:
            agent.perceive(self, self.current_tick)

        # 2. TORRES: La visión y el ataque de torres se procesan en un único
        # tower.update() llamado en el paso 9 (combate), para evitar doble
        # decremento de cooldown. Aquí no se llama nada.

        # 3. COMUNICACIÓN de cazadores
        for hunter in alive_hunters:
            hunter.communicate(alive_hunters)

        # 4. DECISIONES — cada agente calcula su siguiente posición
        intended_moves: dict = {}
        for c in alive_collectors:
            new_pos = c.decide(self, self.current_tick, alive_collectors, alive_guards)
            intended_moves[c] = new_pos if new_pos is not None else c.position
        for g in alive_guards:
            # Se pasa self.collectors (todos, vivos y muertos) para que el guardia
            # pueda calcular las bajas y ajustar su comportamiento de escolta.
            new_pos = g.decide(self, self.current_tick, self.collectors, alive_guards)
            intended_moves[g] = new_pos if new_pos is not None else g.position
        all_agents_for_hunters = alive_collectors + alive_guards + alive_hunters
        for h in alive_hunters:
            h.decide(self, all_agents_for_hunters, self.current_tick)
            intended_moves[h] = h.next_position

        # 5. MOVIMIENTO SIMULTÁNEO
        for agent, new_pos in intended_moves.items():
            if new_pos is None:
                continue
            old_pos = agent.position
            if old_pos != new_pos:
                self.cells[old_pos[0]][old_pos[1]].agents.remove(agent)
                agent.position = new_pos
                self.cells[new_pos[0]][new_pos[1]].agents.append(agent)

        # Actualizar alive lists después de movimiento (posiciones cambiaron)
        alive_collectors = [c for c in self.collectors if c.is_alive]
        alive_guards     = [g for g in self.guards     if g.is_alive]
        alive_hunters    = [h for h in self.hunters    if h.is_alive]

        # 6. RECOLECCIÓN
        for c in alive_collectors:
            cx, cy = c.position
            cell = self.cells[cx][cy]
            if cell.type == 'resource' and cell.resource_amount > 0:
                amount = c.collect_resource(cell)
                if amount > 0:
                    c.receive_reward('collect')
                    self.last_event = f'Recolector recoge {amount} recursos'
                    # Marcar como explorada la nueva posición de recurso
                    self.known_map[cx][cy]['explored'] = True
                    self.known_map[cx][cy]['last_known_type'] = 'resource'
                if cell.resource_amount <= 0:
                    cell.type = 'empty'
                    # Limpiar de discovered_resources
                    self.discovered_resources[:] = [
                        r for r in self.discovered_resources
                        if r['position'] != (cx, cy)
                    ]

        # 7. DEPÓSITO EN BASE
        for c in alive_collectors:
            if c.position == BASE_POSITION:
                deposited = c.deposit_resources()
                if deposited > 0:
                    self.base_resources += deposited
                    c.receive_reward('deliver_resources')
                    self.last_event = f'Depósito: +{deposited} (total {self.base_resources})'
                    self._check_build_kits(alive_collectors)

        # 8. CONSTRUCCIÓN DE TORRES
        for c in alive_collectors:
            if c.wants_to_build and c.has_build_kit and c.build_target == c.position:
                bpos = c.position
                bx, by = bpos
                cell = self.cells[bx][by]
                if cell.type == 'empty' and cell.tower is None:
                    t = Tower(bpos)
                    self.towers.append(t)
                    cell.tower = t
                    cell.type  = 'tower'
                    self.known_map[bx][by]['last_known_type'] = 'tower'
                    c.has_build_kit  = False
                    c.wants_to_build = False
                    c.build_target   = None   # resetear para futura construcción
                    c.receive_reward('build_tower')
                    self.last_event = f'Torre construida en {bpos}'

        # 9. COMBATE — recolectar todos los ataques ANTES de aplicar daño
        attacks = self._collect_all_attacks(alive_collectors, alive_guards, alive_hunters)
        pending_damage: dict = {}  # agent -> damage acumulado
        for attacker, target, dmg in attacks:
            pending_damage[target] = pending_damage.get(target, 0) + dmg

        # 10. APLICAR DAÑO y procesar muertes
        self._resolve_combat(attacks, pending_damage)

        # 11. RESPAWN de cazadores muertos (distancia mínima a base y a aliados vivos)
        ally_agents = [a for a in self.collectors + self.guards if a.is_alive]
        for h in self.hunters:
            if not h.is_alive:
                revived = h.try_respawn(
                    MAP_WIDTH, MAP_HEIGHT,
                    BASE_POSITION, HUNTER_MIN_SPAWN_DISTANCE,
                    ally_agents, HUNTER_MIN_SPAWN_DISTANCE_ALLY,
                )
                if revived:
                    self.cells[h.position[0]][h.position[1]].agents.append(h)
                    self.last_event = 'Cazador reaparece'

        # 12. ACTUALIZAR RISK MAP
        self._update_risk_map()

        # 13. ACTUALIZAR explored_count en shared_data (usado por agentes para fase/estado)
        self.shared_data['explored_count'] = sum(
            1 for x in range(MAP_WIDTH) for y in range(MAP_HEIGHT)
            if self.known_map[x][y]['explored']
        )

        # 14. (epsilon decay se gestiona en el trainer, una vez por episodio)

        # 15. VERIFICAR VICTORIA
        self._check_win_conditions()

        # 16. Avanzar tick
        self.current_tick += 1

    # =====================================================================
    # COMBATE
    # =====================================================================

    def _collect_all_attacks(self, alive_collectors, alive_guards, alive_hunters):
        """
        Recoge todos los ataques posibles en este tick.
        Retorna lista de (attacker, target, damage).
        NO aplica daño aún.
        """
        attacks = []

        # Cazadores atacan a colectores y guardias
        for h in alive_hunters:
            if h.current_cooldown > 0:
                continue
            target = h.current_target
            if target is None or not getattr(target, 'is_alive', False):
                # Buscar víctima adyacente si no tiene target explícito
                hx, hy = h.position
                for agent in self.cells[hx][hy].agents:
                    if agent is h:
                        continue
                    if (agent.__class__.__name__ in ('Collector', 'Guard')
                            and agent.is_alive):
                        target = agent
                        break
            if target is None or not getattr(target, 'is_alive', False):
                continue
            dist = (abs(target.position[0] - h.position[0])
                    + abs(target.position[1] - h.position[1]))
            if dist <= h.attack_range:
                attacks.append((h, target, h.damage))
                h.current_cooldown = h.attack_cooldown  # aplicar cooldown aquí

        # Guardias atacan a cazadores
        for g in alive_guards:
            if g.current_cooldown > 0:
                continue
            # Buscar cazador más cercano en rango
            best_target = None
            best_dist   = float('inf')
            for h in alive_hunters:
                dist = (abs(h.position[0] - g.position[0])
                        + abs(h.position[1] - g.position[1]))
                if dist <= g.attack_range and dist < best_dist:
                    best_dist   = dist
                    best_target = h
            if best_target is not None:
                attacks.append((g, best_target, g.damage))
                g.current_cooldown = g.attack_cooldown  # aplicar cooldown aquí

        # Torres: única llamada a update() por tick.
        # Actualiza visión/fog, detecta enemigos, reduce riesgo y dispara si puede.
        for tower in self.towers:
            tower_attacks = tower.update(
                self, self.known_map, self.last_seen_enemies,
                self.risk_map, self.current_tick
            )
            for _, target in tower_attacks:
                if target is not None and target.is_alive:
                    attacks.append((tower, target, tower.damage))

        return attacks

    def _resolve_combat(self, attacks, pending_damage):
        """
        Aplica daño simultáneamente a todos los targets golpeados.
        Luego procesa muertes y efectos on-kill.
        """
        killed_this_tick = set()

        for target, total_dmg in pending_damage.items():
            if not target.is_alive:
                continue
            target.current_hp -= total_dmg
            if target.current_hp <= 0:
                killed_this_tick.add(target)

        for target in killed_this_tick:
            # Encontrar a quién dar el crédito del kill (primero atacante que golpeó)
            killers = [att for att, tgt, _ in attacks if tgt is target]
            if killers:
                killer = killers[0]
                target_type = target.__class__.__name__
                if hasattr(killer, 'on_kill'):
                    killer.on_kill(target_type)
                # Recompensa explícita para guardias que matan cazadores
                if target_type == 'Hunter' and killer.__class__.__name__ == 'Guard':
                    killer.receive_reward('kill_hunter')
                self.last_event = f'{killer.__class__.__name__} eliminó a {target_type}'

            # Ejecutar muerte
            self._kill_agent(target)

    def _kill_agent(self, agent):
        """Marca al agente como muerto y lo remueve del grid."""
        atype = agent.__class__.__name__
        px, py = agent.position
        if agent in self.cells[px][py].agents:
            self.cells[px][py].agents.remove(agent)

        if atype == 'Collector':
            agent.receive_reward('die')
            agent.die()
        elif atype == 'Guard':
            agent.receive_reward('die')
            agent.die()
        elif atype == 'Hunter':
            agent.die()

    # =====================================================================
    # RISK MAP
    # =====================================================================

    def _update_risk_map(self):
        """
        Actualiza el mapa de riesgo con decaimiento gradual hacia cero.

        Algoritmo por tick:
          1. Decaimiento: risk_map[x][y] *= RISK_DECAY  (0.5^4 ≈ 0 en 4 ticks)
          2. Sumar contribuciones de las fuentes actuales propagadas en radio R.

        Valores base en la casilla del agente:
          Guardia: -1, Torre: -3, Recolector: +0.5, Cazador: +2

        Los cazadores se consideran fuente solo si fueron vistos en los últimos
        RISK_MEMORY_TICKS ticks; sin ese avistamiento, su contribución no se suma
        y la celda decae sola hacia cero.
        """
        R = RISK_PROPAGATION_RANGE

        # 1. Decaimiento multiplicativo → tiende a 0 en ~4 ticks sin fuente
        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                self.risk_map[x][y] *= RISK_DECAY

        # 2. Recopilar fuentes activas
        sources = []

        for guard in self.guards:
            if guard.is_alive:
                sources.append((guard.position[0], guard.position[1], RISK_GUARD))

        for tower in self.towers:
            sources.append((tower.position[0], tower.position[1], RISK_TOWER))

        for collector in self.collectors:
            if collector.is_alive:
                sources.append((collector.position[0], collector.position[1], RISK_COLLECTOR))

        for entry in self.last_seen_enemies:
            if self.current_tick - entry['tick'] <= RISK_MEMORY_TICKS:
                sources.append((entry['position'][0], entry['position'][1], RISK_HUNTER))

        # 3. Propagar cada fuente en radio R con decaimiento lineal de distancia
        for sx, sy, value in sources:
            for dx in range(-R, R + 1):
                for dy in range(-R, R + 1):
                    dist = abs(dx) + abs(dy)
                    if dist > R:
                        continue
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT:
                        factor = 1.0 - dist / (R + 1)
                        self.risk_map[nx][ny] += value * factor

    # =====================================================================
    # BUILD KITS
    # =====================================================================

    def _check_build_kits(self, alive_collectors):
        """
        Genera build kits al depositar recursos.
        Kits totales generados = base_resources // BUILD_KIT_COST
        Reglas mejoradas:
        - No asignar kit si solo queda 1 recolector vivo
        - No asignar si torres >= cap dinámico
        """
        total_kits = self.base_resources // BUILD_KIT_COST
        new_kits   = total_kits - self._kits_given
        if new_kits <= 0:
            return

        # No dar kit si solo queda 1 recolector
        if len(alive_collectors) <= 1:
            return

        # Cap dinámico de torres
        explored_count = self.shared_data.get('explored_count', 0)
        max_towers = max(2, int(explored_count * MAX_TOWERS_PER_EXPLORATION))
        if len(self.towers) >= max_towers:
            return

        for _ in range(new_kits):
            for c in alive_collectors:
                if c.is_alive and c.position == BASE_POSITION and not c.has_build_kit:
                    c.receive_build_kit()
                    self._kits_given += 1
                    self.build_kits_available = sum(
                        1 for col in self.collectors if col.has_build_kit
                    )
                    self.last_event = 'Build kit asignado a recolector'
                    break

    # =====================================================================
    # CONDICIONES DE VICTORIA
    # =====================================================================

    def _check_win_conditions(self):
        """
        Equipo A gana: base_resources >= total_resources * WIN_RESOURCE_PERCENT
        Equipo B gana: no quedan collectors ni guards vivos
        """
        if self.total_resources > 0:
            if self.base_resources >= self.total_resources * WIN_RESOURCE_PERCENT:
                self.game_over = True
                self.winner    = 'A'
                return

        alive_a = (
            any(c.is_alive for c in self.collectors)
            or any(g.is_alive for g in self.guards)
        )
        if not alive_a:
            self.game_over = True
            self.winner    = 'B'

    # =====================================================================
    # FASE DEL JUEGO
    # =====================================================================

    def get_game_phase(self):
        """
        Retorna 'EARLY', 'MID' o 'LATE' según el porcentaje de mapa explorado.
        """
        from utils.constants import PHASE_EARLY_THRESHOLD, PHASE_MID_THRESHOLD
        explored_count = self.shared_data.get('explored_count', 0)
        explored_pct   = explored_count / (MAP_WIDTH * MAP_HEIGHT)
        if explored_pct < PHASE_EARLY_THRESHOLD:
            return 'EARLY'
        elif explored_pct < PHASE_MID_THRESHOLD:
            return 'MID'
        return 'LATE'

    # =====================================================================
    # INPUT DEL USUARIO
    # =====================================================================

    def handle_click(self, screen_x, screen_y):
        """
        Convierte coordenadas de pantalla a posición del grid
        y propaga user_target_position a todos los cazadores vivos.
        """
        gx = screen_x // CELL_SIZE
        gy = screen_y // CELL_SIZE
        if not (0 <= gx < MAP_WIDTH and 0 <= gy < MAP_HEIGHT):
            return
        self.user_target_position = (gx, gy)
        for h in self.hunters:
            if h.is_alive:
                h.user_target_position = (gx, gy)

    # =====================================================================
    # RENDERING (PYGAME)
    # =====================================================================

    def render(self, screen):
        """
        Dibuja el estado completo de la simulación en la pantalla pygame.
        Capas: fondo → fog → explorado → recursos → base → torres → grid
               → agentes → sidebar.
        """
        import pygame  # importación diferida para no requerir pygame en tests

        # 1. Fondo
        screen.fill(COLOR_BG)

        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                rx = x * CELL_SIZE
                ry = y * CELL_SIZE
                cell = self.cells[x][y]
                info = self.known_map[x][y]

                # 2. Fog of war / explorado
                if not info['explored']:
                    pygame.draw.rect(screen, COLOR_FOG,
                                     (rx, ry, CELL_SIZE, CELL_SIZE))
                    continue

                pygame.draw.rect(screen, COLOR_EXPLORED,
                                 (rx, ry, CELL_SIZE, CELL_SIZE))

                # 3. Obstáculos
                if cell.type == 'obstacle':
                    pygame.draw.rect(screen, COLOR_OBSTACLE,
                                     (rx, ry, CELL_SIZE, CELL_SIZE))
                    continue

                # 4. Recursos
                if cell.type == 'resource' and cell.resource_amount > 0:
                    radius = max(2, min(CELL_SIZE // 2 - 1,
                                        cell.resource_amount // 2))
                    cx_px = rx + CELL_SIZE // 2
                    cy_px = ry + CELL_SIZE // 2
                    pygame.draw.circle(screen, COLOR_RESOURCE,
                                       (cx_px, cy_px), radius)

                # 5. Base
                if cell.type == 'base':
                    pygame.draw.rect(screen, COLOR_BASE,
                                     (rx + 1, ry + 1, CELL_SIZE - 2, CELL_SIZE - 2))

                # 6. Torres
                if cell.tower is not None:
                    pygame.draw.rect(screen, COLOR_TOWER,
                                     (rx + 2, ry + 2, CELL_SIZE - 4, CELL_SIZE - 4))
                    # Rango semi-transparente (surface con alpha)
                    tr = cell.tower.attack_range
                    surf_size = (tr * 2 + 1) * CELL_SIZE
                    range_surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
                    range_surf.fill((0, 200, 80, 20))
                    screen.blit(range_surf,
                                (rx - tr * CELL_SIZE, ry - tr * CELL_SIZE))

        # 7. Líneas del grid (encima de celdas)
        for x in range(MAP_WIDTH + 1):
            pygame.draw.line(screen, COLOR_GRID_LINE,
                             (x * CELL_SIZE, 0),
                             (x * CELL_SIZE, MAP_HEIGHT * CELL_SIZE))
        for y in range(MAP_HEIGHT + 1):
            pygame.draw.line(screen, COLOR_GRID_LINE,
                             (0, y * CELL_SIZE),
                             (MAP_WIDTH * CELL_SIZE, y * CELL_SIZE))

        # 8. Agentes
        for c in self.collectors:
            if not c.is_alive:
                continue
            rx = c.position[0] * CELL_SIZE + CELL_SIZE // 2
            ry = c.position[1] * CELL_SIZE + CELL_SIZE // 2
            color = COLOR_COLLECTOR_WITH_KIT if c.has_build_kit else COLOR_COLLECTOR
            pygame.draw.circle(screen, color, (rx, ry), CELL_SIZE // 2 - 1)
            # Barra de recursos
            if c.carrying_resources > 0:
                ratio = c.carrying_resources / c.carrying_capacity
                bar_w = int((CELL_SIZE - 2) * ratio)
                pygame.draw.rect(screen, (255, 255, 0),
                                 (c.position[0] * CELL_SIZE + 1,
                                  c.position[1] * CELL_SIZE + CELL_SIZE - 3,
                                  bar_w, 2))

        for g in self.guards:
            if not g.is_alive:
                continue
            rx = g.position[0] * CELL_SIZE + CELL_SIZE // 2
            ry = g.position[1] * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(screen, COLOR_GUARD, (rx, ry), CELL_SIZE // 2 - 1)

        for h in self.hunters:
            color = COLOR_HUNTER if h.is_alive else COLOR_HUNTER_DEAD
            rx = h.position[0] * CELL_SIZE + CELL_SIZE // 2
            ry = h.position[1] * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(screen, color, (rx, ry), CELL_SIZE // 2 - 1)

        # Marcador del click del usuario
        if self.user_target_position:
            ux, uy = self.user_target_position
            pygame.draw.rect(screen, (255, 255, 0),
                             (ux * CELL_SIZE, uy * CELL_SIZE, CELL_SIZE, CELL_SIZE), 2)

        # 9. Sidebar
        self._render_sidebar(screen, pygame)

    def _render_sidebar(self, screen, pygame):
        """Dibuja el panel lateral derecho con estadísticas."""
        sx = MAP_WIDTH * CELL_SIZE
        pygame.draw.rect(screen, COLOR_SIDEBAR_BG,
                         (sx, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))

        font_big   = pygame.font.SysFont('monospace', 14, bold=True)
        font_small = pygame.font.SysFont('monospace', 12)

        lines = []
        lines.append(('== SIMULACIÓN ==', COLOR_TEXT_HIGHLIGHT))
        lines.append((f'Tick: {self.current_tick}', COLOR_TEXT))
        lines.append(('', COLOR_TEXT))

        # Progreso de recursos
        pct = (self.base_resources / self.total_resources * 100
               if self.total_resources > 0 else 0)
        lines.append(('Recursos:', COLOR_TEXT_HIGHLIGHT))
        lines.append((f'  {self.base_resources}/{self.total_resources}  '
                       f'({pct:.1f}% / 80%)', COLOR_TEXT))

        # Agentes
        lines.append(('', COLOR_TEXT))
        alive_c = sum(1 for c in self.collectors if c.is_alive)
        alive_g = sum(1 for g in self.guards     if g.is_alive)
        alive_h = sum(1 for h in self.hunters    if h.is_alive)
        dead_h  = sum(1 for h in self.hunters    if not h.is_alive)
        lines.append(('Equipo A:', COLOR_TEXT_HIGHLIGHT))
        lines.append((f'  Recolectores: {alive_c}/{NUM_COLLECTORS}', COLOR_TEXT))
        lines.append((f'  Guardias:     {alive_g}/{NUM_GUARDS}', COLOR_TEXT))
        lines.append((f'  Torres:       {len(self.towers)}', COLOR_TEXT))
        lines.append(('Equipo B:', COLOR_TEXT_HIGHLIGHT))
        lines.append((f'  Cazadores:    {alive_h}/{NUM_HUNTERS}', COLOR_TEXT))
        lines.append((f'  Esperando:    {dead_h}', COLOR_TEXT))

        # Build kits
        lines.append(('', COLOR_TEXT))
        kits = sum(1 for c in self.collectors if c.has_build_kit)
        lines.append((f'Build kits:  {kits}', COLOR_TEXT))

        # Genes promedio de cazadores vivos
        alive_hunters = [h for h in self.hunters if h.is_alive]
        if alive_hunters:
            avg_gamma = sum(h.genes.gamma for h in alive_hunters) / len(alive_hunters)
            avg_beta  = sum(h.genes.beta  for h in alive_hunters) / len(alive_hunters)
            lines.append(('', COLOR_TEXT))
            lines.append(('Genes (prom):', COLOR_TEXT_HIGHLIGHT))
            lines.append((f'  γ={avg_gamma:.2f}  β={avg_beta:.2f}', COLOR_TEXT))

        # Epsilon RL
        lines.append(('', COLOR_TEXT))
        lines.append((f'ε col: {self.collector_ql.epsilon:.3f}', COLOR_TEXT))
        lines.append((f'ε grd: {self.guard_ql.epsilon:.3f}', COLOR_TEXT))

        # Último evento
        lines.append(('', COLOR_TEXT))
        lines.append(('Último evento:', COLOR_TEXT_HIGHLIGHT))
        # Partir evento largo en líneas
        ev = self.last_event
        while len(ev) > 28:
            lines.append(('  ' + ev[:26], COLOR_TEXT))
            ev = ev[26:]
        lines.append(('  ' + ev, COLOR_TEXT))

        # Estado del juego
        if self.game_over:
            lines.append(('', COLOR_TEXT))
            msg = f'FIN: GANA {"Equipo A" if self.winner == "A" else "Equipo B"}'
            lines.append((msg, COLOR_TEXT_HIGHLIGHT))

        # Renderizar líneas
        y_offset = 8
        for text, color in lines:
            surf = font_small.render(text, True, color)
            screen.blit(surf, (sx + 8, y_offset))
            y_offset += 16
            if y_offset > WINDOW_HEIGHT - 16:
                break

        # Barra de progreso de recursos
        bar_x  = sx + 8
        bar_y  = 56
        bar_w  = SIDEBAR_WIDTH - 16
        bar_h  = 8
        pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h))
        if self.total_resources > 0:
            fill = int(bar_w * min(1.0, self.base_resources / self.total_resources))
            pygame.draw.rect(screen, (0, 200, 80), (bar_x, bar_y, fill, bar_h))
        # Marca del 80%
        mark_x = bar_x + int(bar_w * 0.8)
        pygame.draw.line(screen, (255, 255, 0),
                         (mark_x, bar_y - 2), (mark_x, bar_y + bar_h + 2), 2)

    # =====================================================================
    # HELPERS DE INICIALIZACIÓN
    # =====================================================================

    def _place_obstacles(self):
        """Genera obstáculos aleatorios, nunca sobre la base ni recursos."""
        placed = 0
        attempts = 0
        while placed < NUM_OBSTACLES and attempts < NUM_OBSTACLES * 20:
            attempts += 1
            x = random.randint(0, MAP_WIDTH  - 1)
            y = random.randint(0, MAP_HEIGHT - 1)
            # No sobre la base ni la zona inicial
            bx, by = BASE_POSITION
            if abs(x - bx) <= 3 and abs(y - by) <= 3:
                continue
            if self.cells[x][y].type != 'empty':
                continue
            self.cells[x][y].type = 'obstacle'
            placed += 1

    def _place_resources(self):
        """Coloca NUM_RESOURCE_NODES nodos de recursos en celdas vacías."""
        placed = 0
        attempts = 0
        bx, by = BASE_POSITION
        while placed < NUM_RESOURCE_NODES and attempts < NUM_RESOURCE_NODES * 30:
            attempts += 1
            x = random.randint(0, MAP_WIDTH  - 1)
            y = random.randint(0, MAP_HEIGHT - 1)
            if self.cells[x][y].type != 'empty':
                continue
            # No demasiado cerca de la base
            if abs(x - bx) <= 2 and abs(y - by) <= 2:
                continue
            amount = random.randint(RESOURCE_MIN_AMOUNT, RESOURCE_MAX_AMOUNT)
            self.cells[x][y].type            = 'resource'
            self.cells[x][y].resource_amount = amount
            self.total_resources             += amount
            placed += 1

    def _spawn_near_base(self, offset):
        """Retorna una posición libre cerca de la base para spawn inicial."""
        bx, by = BASE_POSITION
        for r in range(1, 6):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = bx + dx, by + dy
                    if not (0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT):
                        continue
                    cell = self.cells[nx][ny]
                    if cell.type in ('empty', 'base') and not cell.agents:
                        return (nx, ny)
        return BASE_POSITION
