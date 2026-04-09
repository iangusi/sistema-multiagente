import random

from utils.constants import (
    TOWER_VISION,
    TOWER_ATTACK_RANGE,
    TOWER_ATTACK_COOLDOWN,
    TOWER_DAMAGE,
)


class Tower:
    """
    Torre defensiva del Equipo A.

    Determinista: no toma decisiones estratégicas, no se mueve.
    Ataca al cazador más cercano dentro de su rango cada vez que el
    cooldown llega a 0.

    También actualiza el fog of war (known_map), registra enemigos vistos
    (last_seen_enemies) y reduce el riesgo en su zona (risk_map).
    """

    def __init__(self, position):
        self.position       = position                    # (x, y) fija
        self.vision_range   = TOWER_VISION                # 4
        self.attack_range   = TOWER_ATTACK_RANGE          # 4
        self.attack_cooldown = TOWER_ATTACK_COOLDOWN      # 2
        self.damage         = TOWER_DAMAGE                # 1
        self.current_cooldown = 0
        self.is_active      = True

    # ------------------------------------------------------------------
    # Ciclo principal
    # ------------------------------------------------------------------

    def update(self, grid, known_map, last_seen_enemies, risk_map, current_tick):
        """
        Ejecuta el ciclo completo de la torre en orden estricto:

        1. Reducir cooldown.
        2. Revelar celdas en vision_range → actualizar known_map.
        3. Detectar cazadores visibles → actualizar last_seen_enemies.
        4. Actualizar risk_map (reducir riesgo en su zona).
        5. Si hay cazador en attack_range y cooldown == 0 → atacar al más cercano.

        Parámetros
        ----------
        grid              : grid del environment (.cells[x][y])
        known_map         : referencia compartida known_map[x][y]
        last_seen_enemies : referencia compartida list de dicts
        risk_map          : referencia compartida risk_map[x][y]
        current_tick      : int — tick actual de la simulación

        Retorna
        -------
        list[(Tower, Agent)] : ataques declarados este tick (0 o 1 elemento)
        """
        attacks = []

        # 1. Reducir cooldown
        if self.current_cooldown > 0:
            self.current_cooldown -= 1

        # 2. Revelar celdas en vision_range → actualizar known_map
        visible = self.get_cells_in_range(self.vision_range, grid.width, grid.height)
        for (cx, cy) in visible:
            cell = grid.cells[cx][cy]
            known_map[cx][cy]['explored'] = True
            known_map[cx][cy]['last_seen'] = current_tick
            known_map[cx][cy]['last_known_type'] = cell.type

        # 3. Detectar cazadores visibles → actualizar last_seen_enemies
        for (cx, cy) in visible:
            cell = grid.cells[cx][cy]
            for agent in cell.agents:
                if agent.__class__.__name__ == 'Hunter' and agent.is_alive:
                    # Actualizar o insertar entrada en last_seen_enemies
                    entry = next(
                        (e for e in last_seen_enemies
                         if e.get('agent') is agent),
                        None
                    )
                    if entry is not None:
                        entry['position'] = agent.position
                        entry['tick'] = current_tick
                    else:
                        last_seen_enemies.append({
                            'agent':    agent,
                            'position': agent.position,
                            'type':     'Hunter',
                            'tick':     current_tick,
                        })

        # 4. (La reducción de riesgo de torres se gestiona en _update_risk_map()
        #     del environment, que reconstruye el mapa completo cada tick.)

        # 5. Atacar cazador más cercano en attack_range si cooldown == 0
        if self.current_cooldown == 0:
            cells_in_attack = self.get_cells_in_range(
                self.attack_range, grid.width, grid.height
            )
            enemies_in_range = []
            for (cx, cy) in cells_in_attack:
                for agent in grid.cells[cx][cy].agents:
                    if agent.__class__.__name__ == 'Hunter' and agent.is_alive:
                        enemies_in_range.append(agent)

            target = self.select_target(enemies_in_range)
            if target is not None:
                attacks.append((self, target))
                self.current_cooldown = self.attack_cooldown

        return attacks

    # ------------------------------------------------------------------
    # Utilidades geométricas
    # ------------------------------------------------------------------

    def get_cells_in_range(self, range_val, map_width, map_height):
        """
        Retorna lista de (x, y) dentro de distancia Manhattan <= range_val
        desde la posición de la torre, dentro de los límites del mapa.
        """
        px, py = self.position
        cells = []
        for dx in range(-range_val, range_val + 1):
            for dy in range(-range_val, range_val + 1):
                if abs(dx) + abs(dy) <= range_val:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < map_width and 0 <= ny < map_height:
                        cells.append((nx, ny))
        return cells

    def select_target(self, enemies_in_range):
        """
        Elige al cazador más cercano (distancia Manhattan).
        En caso de empate elige aleatoriamente entre los empatados.

        Parámetros
        ----------
        enemies_in_range : list[Agent] — cazadores vivos dentro de attack_range

        Retorna
        -------
        Agent o None
        """
        if not enemies_in_range:
            return None

        px, py = self.position
        min_dist = min(
            abs(e.position[0] - px) + abs(e.position[1] - py)
            for e in enemies_in_range
        )
        closest = [
            e for e in enemies_in_range
            if abs(e.position[0] - px) + abs(e.position[1] - py) == min_dist
        ]
        return random.choice(closest)
