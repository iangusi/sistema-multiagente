import random

from utils.constants import (
    HUNTER_HP,
    HUNTER_SPEED,
    HUNTER_VISION,
    HUNTER_ATTACK_RANGE,
    HUNTER_ATTACK_COOLDOWN,
    HUNTER_DAMAGE,
    HUNTER_COMMUNICATION_RADIUS,
    HUNTER_MEMORY_DURATION,
    HUNTER_SPEED_INCREMENT,
    HUNTER_HP_INCREMENT,
    HUNTER_MAX_SPEED,
    HUNTER_MAX_HP,
    HUNTER_RESPAWN_TICKS,
    GUARD_ATTACK_RANGE,
    TOWER_ATTACK_RANGE,
    USER_INFLUENCE_WEIGHT,
    BASE_POSITION,
    MAP_WIDTH,
    MAP_HEIGHT,
)
from evolution.genetic_system import Genes, mutate, random_spawn_position
from pathfinding.astar import find_path

# -----------------------------------------------------------------------
# Acciones disponibles del cazador
# -----------------------------------------------------------------------
ACTIONS = [
    'ATTACK',
    'CHASE',
    'FLEE',
    'GROUP',
    'STALK',
    'FLANK',
    'WAIT_FOR_REINFORCEMENTS',
    'WANDER',
    'RETREAT',
]

# Distancia de seguridad que STALK mantiene respecto al objetivo
_STALK_SAFE_DISTANCE = TOWER_ATTACK_RANGE + 1   # 5


class Hunter:
    """
    Cazador del Equipo B.

    Toma decisiones mediante FSM + scoring heurístico ponderado por genes.
    NO usa Q-learning. Evoluciona (muta genes) al reaparecer.
    Muerte temporal: respawnea tras HUNTER_RESPAWN_TICKS ticks.
    """

    def __init__(self, position, genes=None):
        # Posición y stats base
        self.position        = position
        self.base_hp         = HUNTER_HP
        self.base_speed      = HUNTER_SPEED
        self.current_hp      = HUNTER_HP
        self.current_speed   = HUNTER_SPEED
        self.vision_range    = HUNTER_VISION
        self.attack_range    = HUNTER_ATTACK_RANGE
        self.attack_cooldown = HUNTER_ATTACK_COOLDOWN
        self.damage          = HUNTER_DAMAGE
        self.comm_radius     = HUNTER_COMMUNICATION_RADIUS

        # Genes de personalidad
        self.genes = genes if genes is not None else Genes()

        # Estado FSM
        self.current_state  = 'WANDER'
        self.current_target = None
        self.current_path   = []
        self.current_cooldown = 0

        # Memoria local: lista de dicts {position, type, last_seen_tick, agent}
        self.local_memory = []

        # Comunicación: enemigos reportados por aliados este tick
        self._received_reports = []

        # Tracking para fitness y evolución
        self.kills       = 0
        self.ticks_alive = 0
        self.deaths      = 0

        # Estado de vida
        self.is_alive      = True
        self.respawn_timer = 0

        # Influencia del usuario
        self.user_target_position = None

        # Próxima posición calculada (se aplica en el step de movimiento)
        self.next_position = position

    # ===================================================================
    # PERCEPCIÓN
    # ===================================================================

    def perceive(self, all_agents, current_tick):
        """
        Detecta todos los agentes dentro de vision_range.

        Actualiza local_memory (nuevos avistamientos + caducidad).
        Retorna (visible_enemies, visible_allies).
        """
        px, py = self.position
        visible_enemies = []
        visible_allies  = []

        for agent in all_agents:
            if agent is self or not agent.is_alive:
                continue
            ax, ay = agent.position
            dist = abs(ax - px) + abs(ay - py)
            if dist > self.vision_range:
                continue

            agent_type = agent.__class__.__name__
            if agent_type == 'Hunter':
                visible_allies.append(agent)
            else:
                visible_enemies.append(agent)
                # Actualizar memoria local
                entry = next(
                    (e for e in self.local_memory if e.get('agent') is agent),
                    None
                )
                if entry is not None:
                    entry['position']       = agent.position
                    entry['last_seen_tick'] = current_tick
                else:
                    self.local_memory.append({
                        'agent':          agent,
                        'position':       agent.position,
                        'type':           agent_type,
                        'last_seen_tick': current_tick,
                    })

        # Caducar entradas viejas
        self.local_memory = [
            e for e in self.local_memory
            if current_tick - e['last_seen_tick'] <= HUNTER_MEMORY_DURATION
        ]

        # Integrar reportes recibidos de aliados en la memoria
        for report in self._received_reports:
            existing = next(
                (e for e in self.local_memory if e.get('agent') is report.get('agent')),
                None
            )
            if existing is None:
                self.local_memory.append(report)
            else:
                if report['last_seen_tick'] > existing['last_seen_tick']:
                    existing['position']       = report['position']
                    existing['last_seen_tick'] = report['last_seen_tick']
        self._received_reports.clear()

        return visible_enemies, visible_allies

    # ===================================================================
    # COMUNICACIÓN LOCAL
    # ===================================================================

    def communicate(self, allied_hunters):
        """
        Intercambia información con aliados dentro de comm_radius.

        Envía: posiciones de enemigos en memoria, target actual, estado.
        Recibe: sus reportes (se inyectan en _received_reports).
        """
        px, py = self.position
        for ally in allied_hunters:
            if ally is self or not ally.is_alive:
                continue
            ax, ay = ally.position
            if abs(ax - px) + abs(ay - py) > self.comm_radius:
                continue

            # Compartir nuestra memoria con el aliado
            for entry in self.local_memory:
                ally._received_reports.append(dict(entry))

            # Recibir memoria del aliado
            for entry in ally.local_memory:
                self._received_reports.append(dict(entry))

    # ===================================================================
    # EVALUACIÓN DE RIESGO Y OPORTUNIDAD
    # ===================================================================

    def calculate_risk(self, visible_enemies, visible_allies):
        """
        risk =
            proximidad_a_guardias
          + estar_dentro_de_rango_de_ataque
          + desventaja_numérica (enemigos > aliados)
          + presencia_torres_cercanas
        """
        px, py = self.position
        risk = 0.0

        guards = [e for e in visible_enemies if e.__class__.__name__ == 'Guard']
        towers = [e for e in visible_enemies if e.__class__.__name__ == 'Tower']

        # Proximidad a guardias y si estamos en su rango de ataque
        for g in guards:
            dist = abs(g.position[0] - px) + abs(g.position[1] - py)
            proximity = max(0.0, 1.0 - dist / (GUARD_ATTACK_RANGE + 2))
            risk += proximity
            if dist <= GUARD_ATTACK_RANGE + 1:   # zona de peligro extendida
                risk += 1.5

        # Presencia de torres y si estamos en su zona de amenaza
        for t in towers:
            dist = abs(t.position[0] - px) + abs(t.position[1] - py)
            risk += 1.0
            if dist <= TOWER_ATTACK_RANGE + 1:
                risk += 2.0

        # Desventaja numérica
        n_enemies = len(guards) + len(towers)
        n_allies  = len(visible_allies)
        if n_enemies > 0 and n_allies < n_enemies:
            risk += (n_enemies - n_allies) * 0.5

        return risk

    def calculate_opportunity(self, visible_enemies, visible_allies):
        """
        opportunity =
            enemigo_aislado (sin guardias/torres cerca)
          + ventaja_numérica (aliados > enemigos)
          + falta_de_defensa
        """
        collectors = [
            e for e in visible_enemies
            if e.__class__.__name__ == 'Collector'
        ]
        guards  = [e for e in visible_enemies if e.__class__.__name__ == 'Guard']
        towers  = [e for e in visible_enemies if e.__class__.__name__ == 'Tower']

        opportunity = 0.0

        # Enemigos aislados (recolectores sin guardias/torres muy cerca)
        for c in collectors:
            cx, cy = c.position
            protected = any(
                abs(g.position[0] - cx) + abs(g.position[1] - cy) <= GUARD_ATTACK_RANGE
                for g in guards
            ) or any(
                abs(t.position[0] - cx) + abs(t.position[1] - cy) <= TOWER_ATTACK_RANGE
                for t in towers
            )
            if not protected:
                opportunity += 2.0
            else:
                opportunity += 0.5

        # Ventaja numérica
        n_threats = len(guards) + len(towers)
        n_allies  = len(visible_allies)
        if n_allies > n_threats:
            opportunity += (n_allies - n_threats) * 0.5

        # Falta de defensa general
        if not guards and not towers:
            opportunity += 1.5

        return opportunity

    # ===================================================================
    # SCORING DE ACCIONES
    # ===================================================================

    def _reward_potential(self, target):
        """Valor de eliminar este objetivo (determina buff que recibirá el cazador)."""
        if target is None:
            return 0.0
        t = target.__class__.__name__
        if t == 'Collector':
            return 20.0
        if t == 'Guard':
            return 30.0
        return 10.0

    def _numerical_advantage(self, visible_enemies, visible_allies):
        n_threats = sum(
            1 for e in visible_enemies
            if e.__class__.__name__ in ('Guard', 'Tower')
        )
        n_allies = len(visible_allies)
        if n_threats == 0:
            return n_allies * 0.5
        ratio = n_allies / n_threats
        if ratio < 1.0:
            return -50.0
        if ratio > 1.0:
            return 50.0
        return 0.0

    def evaluate_actions(self, risk, opportunity, visible_enemies, visible_allies):
        """
        Calcula un score para cada acción según genes, riesgo, oportunidad y contexto.

        Retorna dict {action: float}.
        """
        g = self.genes
        target = self.current_target
        reward_pot = self._reward_potential(target)
        num_adv    = self._numerical_advantage(visible_enemies, visible_allies)
        n_allies   = len(visible_allies)
        px, py     = self.position

        # Proximidad al objetivo (normalizada)
        target_proximity = 0.0
        if target is not None and hasattr(target, 'is_alive') and target.is_alive:
            dist = abs(target.position[0] - px) + abs(target.position[1] - py)
            target_proximity = max(0.0, 10.0 - dist)

        scores = {}

        # ATTACK: lanzar ataque directo
        scores['ATTACK'] = (
            g.gamma * opportunity
            + num_adv
            + reward_pot
            - risk * 80.0
        )

        # CHASE: perseguir objetivo
        scores['CHASE'] = (
            g.beta * target_proximity
            + opportunity
            - risk
        )

        # FLEE: huir del peligro
        scores['FLEE'] = (
            g.delta * risk
            - n_allies * 30.0
        )

        # GROUP: agruparse con aliados
        scores['GROUP'] = (
            g.alpha * n_allies
            + (risk * 2.0 if n_allies > 0 else 0.0)  # necesidad de refuerzo
        )

        # STALK: seguir a distancia segura
        scores['STALK'] = (
            opportunity * 0.5
            - risk
        )

        # FLANK: buscar posición lateral
        scores['FLANK'] = (
            num_adv
            + opportunity * 0.3
            - risk * 0.5
        )

        # WAIT_FOR_REINFORCEMENTS: esperar aliados
        scores['WAIT_FOR_REINFORCEMENTS'] = (
            opportunity
            - n_allies * 20.0    # menos útil si ya hay aliados
        )

        # WANDER: explorar sin objetivo claro
        scores['WANDER'] = (
            max(0.0, 5.0 - risk)
            + (0.0 if visible_enemies else 3.0)  # más atractivo sin enemigos
        )

        # RETREAT: retirarse activamente
        scores['RETREAT'] = (
            risk * 2.0
            - opportunity
        )

        return scores

    # ===================================================================
    # SELECCIÓN DE OBJETIVO
    # ===================================================================

    def select_target(self, visible_enemies):
        """
        score_target =
            - target.hp
            - distancia
            + valor_objetivo
            - protección_cercana

        Prioridad: recolector con recursos > con kit > aislado > guardia aislado.
        """
        if not visible_enemies:
            return None

        px, py = self.position
        guards = [e for e in visible_enemies if e.__class__.__name__ == 'Guard']
        towers = [e for e in visible_enemies if e.__class__.__name__ == 'Tower']

        def _protection(agent):
            ax, ay = agent.position
            g_count = sum(
                1 for g in guards
                if abs(g.position[0] - ax) + abs(g.position[1] - ay) <= GUARD_ATTACK_RANGE
            )
            t_count = sum(
                1 for t in towers
                if abs(t.position[0] - ax) + abs(t.position[1] - ay) <= TOWER_ATTACK_RANGE
            )
            return (g_count + t_count) * 10.0

        def _value(agent):
            atype = agent.__class__.__name__
            if atype == 'Collector':
                base = 10.0
                if getattr(agent, 'carrying', 0) > 0:
                    base += 15.0   # lleva recursos
                if getattr(agent, 'has_build_kit', False):
                    base += 10.0   # lleva kit
                return base
            if atype == 'Guard':
                return 8.0
            return 0.0

        candidates = [
            e for e in visible_enemies
            if e.__class__.__name__ in ('Collector', 'Guard')
        ]
        if not candidates:
            return None

        def _score(agent):
            dist = abs(agent.position[0] - px) + abs(agent.position[1] - py)
            return (
                -getattr(agent, 'current_hp', 1)
                - dist
                + _value(agent)
                - _protection(agent)
            )

        return max(candidates, key=_score)

    # ===================================================================
    # TOMA DE DECISIÓN
    # ===================================================================

    def decide(self, grid, all_agents, current_tick):
        """
        Ciclo completo de decisión por tick:
        percibir → riesgo/oportunidad → scoring → acción → movimiento.
        """
        if not self.is_alive:
            return

        self.ticks_alive += 1

        # Reducir cooldown de ataque
        if self.current_cooldown > 0:
            self.current_cooldown -= 1

        # 1. Percibir
        visible_enemies, visible_allies = self.perceive(all_agents, current_tick)

        # 2. Seleccionar objetivo (antes de scoring)
        new_target = self.select_target(visible_enemies)
        if new_target is not None:
            self.current_target = new_target
        elif self.current_target is not None:
            # Mantener objetivo de memoria si sigue en memoria local
            mem_agent = next(
                (e['agent'] for e in self.local_memory
                 if e['agent'] is self.current_target),
                None
            )
            if mem_agent is None:
                self.current_target = None

        # 3. Calcular riesgo y oportunidad
        risk        = self.calculate_risk(visible_enemies, visible_allies)
        opportunity = self.calculate_opportunity(visible_enemies, visible_allies)

        # 4. Scoring de acciones
        scores = self.evaluate_actions(risk, opportunity, visible_enemies, visible_allies)

        # 5. Elegir acción de mayor score
        best_action = max(scores, key=lambda a: scores[a])
        self.current_state = best_action

        # 6. Calcular movimiento
        self.next_position = self.calculate_movement(grid, visible_enemies, visible_allies)

    # ===================================================================
    # CÁLCULO DE MOVIMIENTO
    # ===================================================================

    def _build_visible_cells(self, grid):
        """
        Construye el set de celdas conocidas por el cazador para pathfinding.
        Incluye visión directa + celdas reportadas en memoria local.
        """
        px, py = self.position
        visible = set()
        for dx in range(-self.vision_range, self.vision_range + 1):
            for dy in range(-self.vision_range, self.vision_range + 1):
                if abs(dx) + abs(dy) <= self.vision_range:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < grid.width and 0 <= ny < grid.height:
                        visible.add((nx, ny))
        for entry in self.local_memory:
            ep = entry['position']
            if 0 <= ep[0] < grid.width and 0 <= ep[1] < grid.height:
                visible.add(ep)
        return visible

    def _cost_function(self, neighbor_pos):
        """
        Costo de moverse a una celda para el cazador:
            1.0 base
          + zona_amenaza_torres (si en rango de torre conocida)
          + zona_amenaza_guardias (si guardia visible cerca)
          - cercanía_a_aliados (clamped a mínimo 0.1)
        """
        nx, ny = neighbor_pos
        cost = 1.0

        for entry in self.local_memory:
            etype = entry['type']
            ep    = entry['position']
            dist  = abs(ep[0] - nx) + abs(ep[1] - ny)
            if etype == 'Tower' and dist <= TOWER_ATTACK_RANGE + 1:
                cost += 3.0
            elif etype == 'Guard' and dist <= GUARD_ATTACK_RANGE + 1:
                cost += 2.0

        return max(0.1, cost)

    def _apply_user_influence(self, direction):
        """
        Ajusta la dirección del movimiento en función del user_target_position.
        direction: (dx, dy) con valores en {-1, 0, 1}.
        Retorna nueva posición objetivo.
        """
        if self.user_target_position is None:
            return direction
        px, py = self.position
        tx, ty = self.user_target_position
        user_dx = 0 if tx == px else (1 if tx > px else -1)
        user_dy = 0 if ty == py else (1 if ty > py else -1)
        dx = direction[0] * (1 - USER_INFLUENCE_WEIGHT) + user_dx * USER_INFLUENCE_WEIGHT
        dy = direction[1] * (1 - USER_INFLUENCE_WEIGHT) + user_dy * USER_INFLUENCE_WEIGHT
        # Convertir a dirección discreta dominante
        if abs(dx) >= abs(dy):
            return (1 if dx > 0 else (-1 if dx < 0 else 0), 0)
        else:
            return (0, 1 if dy > 0 else (-1 if dy < 0 else 0))

    def _step_toward(self, goal, grid):
        """
        Usa find_path para obtener la ruta y retorna la siguiente posición.
        Si no hay ruta, se queda en su posición.
        """
        if self.position == goal:
            return self.position
        visible_cells = self._build_visible_cells(grid)
        cost_fn = lambda _, nxt: self._cost_function(nxt)
        path = find_path(
            self.position, goal, grid, cost_fn,
            visible_cells=visible_cells
        )
        if path:
            # Avanzar hasta current_speed pasos
            steps = min(int(self.current_speed), len(path))
            return path[steps - 1]
        return self.position

    def _step_away_from(self, danger_pos, grid):
        """Retorna la posición que más se aleja de danger_pos."""
        px, py = self.position
        dx = px - danger_pos[0]
        dy = py - danger_pos[1]
        # Normalizar
        if dx == 0 and dy == 0:
            return self.position
        if abs(dx) >= abs(dy):
            step = (1 if dx > 0 else -1, 0)
        else:
            step = (0, 1 if dy > 0 else -1)
        step = self._apply_user_influence(step)
        nx = max(0, min(grid.width  - 1, px + step[0]))
        ny = max(0, min(grid.height - 1, py + step[1]))
        if grid.cells[nx][ny].type != 'obstacle':
            return (nx, ny)
        return self.position

    def calculate_movement(self, grid, visible_enemies, visible_allies):
        """
        Determina la siguiente posición según la acción actual.
        Aplica user_influence al vector de dirección final.
        """
        action = self.current_state
        px, py = self.position

        # -- ATTACK: moverse hasta estar en rango y atacar --
        if action == 'ATTACK':
            if self.current_target is not None and self.current_target.is_alive:
                dist = abs(self.current_target.position[0] - px) + \
                       abs(self.current_target.position[1] - py)
                if dist > self.attack_range:
                    return self._step_toward(self.current_target.position, grid)
            return self.position

        # -- CHASE: perseguir objetivo directamente --
        if action == 'CHASE':
            if self.current_target is not None and self.current_target.is_alive:
                return self._step_toward(self.current_target.position, grid)
            return self.position

        # -- FLEE: alejarse del peligro más cercano --
        if action == 'FLEE':
            threats = [
                e for e in visible_enemies
                if e.__class__.__name__ in ('Guard', 'Tower')
            ]
            if threats:
                closest = min(
                    threats,
                    key=lambda e: abs(e.position[0] - px) + abs(e.position[1] - py)
                )
                return self._step_away_from(closest.position, grid)
            return self.position

        # -- GROUP: moverse hacia el centroide de aliados --
        if action == 'GROUP':
            if visible_allies:
                cx = int(sum(a.position[0] for a in visible_allies) / len(visible_allies))
                cy = int(sum(a.position[1] for a in visible_allies) / len(visible_allies))
                return self._step_toward((cx, cy), grid)
            return self.position

        # -- STALK: seguir al objetivo manteniéndose fuera de zona de peligro --
        if action == 'STALK':
            if self.current_target is not None and self.current_target.is_alive:
                tx, ty = self.current_target.position
                dist = abs(tx - px) + abs(ty - py)
                if dist > _STALK_SAFE_DISTANCE:
                    # Acercarse pero sin entrar en zona peligrosa
                    return self._step_toward((tx, ty), grid)
                elif dist < _STALK_SAFE_DISTANCE - 1:
                    return self._step_away_from((tx, ty), grid)
            return self.position

        # -- FLANK: buscar posición lateral respecto al objetivo --
        if action == 'FLANK':
            if self.current_target is not None and self.current_target.is_alive:
                tx, ty = self.current_target.position
                # Candidatos laterales: perpendiculares a la línea cazador-target
                main_dx = tx - px
                main_dy = ty - py
                if abs(main_dx) >= abs(main_dy):
                    # movimiento horizontal → flanquear en vertical
                    laterals = [(tx, ty - 2), (tx, ty + 2)]
                else:
                    laterals = [(tx - 2, ty), (tx + 2, ty)]
                # Escoger lateral más seguro (menos guardianes cerca)
                best_lateral = None
                best_dist = float('inf')
                for lx, ly in laterals:
                    if 0 <= lx < grid.width and 0 <= ly < grid.height:
                        d = abs(lx - px) + abs(ly - py)
                        if d < best_dist:
                            best_dist = d
                            best_lateral = (lx, ly)
                if best_lateral:
                    return self._step_toward(best_lateral, grid)
            return self.position

        # -- WAIT_FOR_REINFORCEMENTS: no moverse, ya comunica en communicate() --
        if action == 'WAIT_FOR_REINFORCEMENTS':
            return self.position

        # -- WANDER: explorar con sesgo hacia el interior del mapa cuando no
        #            hay memoria de objetivos, para que los cazadores no queden
        #            indefinidamente en los bordes sin encontrar enemigos.
        if action == 'WANDER':
            # Sin objetivos en memoria → derivar gradualmente hacia el interior.
            # Cada cazador elige un punto aleatorio en el cuadrante interior
            # (25%-75% del mapa) para distribuirlos y evitar una avalancha sobre
            # la base. Solo aplica el 40% de los ticks para mantener variedad.
            if not self.local_memory and random.random() < 0.40:
                cx = random.randint(MAP_WIDTH  // 4, 3 * MAP_WIDTH  // 4)
                cy = random.randint(MAP_HEIGHT // 4, 3 * MAP_HEIGHT // 4)
                return self._step_toward((cx, cy), grid)

            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            random.shuffle(directions)
            # Aplicar influencia del usuario
            chosen = self._apply_user_influence(directions[0])
            nx = max(0, min(grid.width  - 1, px + chosen[0]))
            ny = max(0, min(grid.height - 1, py + chosen[1]))
            if grid.cells[nx][ny].type != 'obstacle':
                return (nx, ny)
            # Intentar otra dirección si hay obstáculo
            for d in directions[1:]:
                nx = max(0, min(grid.width  - 1, px + d[0]))
                ny = max(0, min(grid.height - 1, py + d[1]))
                if grid.cells[nx][ny].type != 'obstacle':
                    return (nx, ny)
            return self.position

        # -- RETREAT: retirarse hacia el borde del mapa --
        if action == 'RETREAT':
            # Moverse hacia el borde más cercano
            dist_left   = px
            dist_right  = grid.width  - 1 - px
            dist_top    = py
            dist_bottom = grid.height - 1 - py
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
            if min_dist == dist_left:
                goal = (0, py)
            elif min_dist == dist_right:
                goal = (grid.width - 1, py)
            elif min_dist == dist_top:
                goal = (px, 0)
            else:
                goal = (px, grid.height - 1)
            return self._step_toward(goal, grid)

        return self.position

    # ===================================================================
    # COMBATE
    # ===================================================================

    def can_attack(self, target):
        """True si el target está en rango y el cooldown es 0."""
        if not self.is_alive or self.current_cooldown > 0:
            return False
        if not hasattr(target, 'is_alive') or not target.is_alive:
            return False
        dist = abs(target.position[0] - self.position[0]) + \
               abs(target.position[1] - self.position[1])
        return dist <= self.attack_range

    def on_kill(self, target_type):
        """
        Aplica buff on-kill y actualiza contador de kills.

        Recolector eliminado → +speed (cap HUNTER_MAX_SPEED)
        Guardia eliminado    → +HP    (cap HUNTER_MAX_HP)
        """
        self.kills += 1
        if target_type == 'Collector':
            self.current_speed = min(
                HUNTER_MAX_SPEED,
                self.current_speed + HUNTER_SPEED_INCREMENT
            )
        elif target_type == 'Guard':
            self.current_hp = min(
                HUNTER_MAX_HP,
                self.current_hp + HUNTER_HP_INCREMENT
            )

    # ===================================================================
    # MUERTE Y RESPAWN
    # ===================================================================

    def die(self):
        """
        Marca el cazador como muerto, inicia el timer de respawn y
        resetea los buffs a los stats base.
        """
        self.is_alive       = False
        self.respawn_timer  = HUNTER_RESPAWN_TICKS
        self.deaths        += 1
        # Resetear buffs — vuelven a base stats al morir
        self.current_hp    = self.base_hp
        self.current_speed = self.base_speed
        self.current_target = None
        self.current_path   = []
        self.current_state  = 'WANDER'

    def try_respawn(self, map_width, map_height,
                    base_pos=None, min_dist=0):
        """
        Reduce el timer de respawn. Si llega a 0, revive al cazador:
        - Nueva posición en borde del mapa (respetando distancia mínima a la base)
        - Genes mutados respecto a la vida anterior
        - Stats reseteados a base

        Retorna True si el cazador revivió en este tick.
        """
        if self.is_alive:
            return False

        self.respawn_timer -= 1
        if self.respawn_timer <= 0:
            self.genes       = mutate(self.genes)
            self.position    = random_spawn_position(map_width, map_height,
                                                     base_pos, min_dist)
            self.next_position = self.position
            self.is_alive    = True
            self.current_hp  = self.base_hp
            self.current_speed = self.base_speed
            self.current_cooldown = 0
            self.local_memory = []
            self._received_reports = []
            self.current_target = None
            self.current_path   = []
            self.current_state  = 'WANDER'
            return True

        return False
