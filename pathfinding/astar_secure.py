import heapq


def get_neighbors(pos, grid_width, grid_height):
    """Retorna vecinos válidos en 4 direcciones (no diagonal)."""
    x, y = pos
    candidates = [
        (x + 1, y),
        (x - 1, y),
        (x,     y + 1),
        (x,     y - 1),
    ]
    return [
        (nx, ny)
        for nx, ny in candidates
        if 0 <= nx < grid_width and 0 <= ny < grid_height
    ]


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_path(start, goal, grid, cost_function, known_map=None, visible_cells=None,
              max_step_cost=None):
    """
    Encuentra un camino desde start hasta goal usando A*.

    Parámetros
    ----------
    start          : (x, y) posición inicial
    goal           : (x, y) posición destino
    grid           : grid del environment — debe exponer .width, .height
                     y .cells[x][y].type
    cost_function  : callable(pos_actual, pos_vecina) -> float
                     Costo de moverse entre dos posiciones adyacentes.
    known_map      : Para Equipo A (recolectores, guardias).
                     known_map[x][y]['explored'] == True habilita la celda.
                     No se puede pasar junto con visible_cells.
    visible_cells  : Para Cazadores. Set de (x, y) que el cazador conoce
                     (visión directa + comunicación).
                     No se puede pasar junto con known_map.
    max_step_cost  : float | None
                     Umbral de peligro por casilla individual.
                     Si cost_function devuelve un valor > max_step_cost para
                     una celda vecina, esa celda se trata como infranqueable
                     (bloqueo duro), independientemente del costo acumulado
                     total de la ruta.
                     Útil para garantizar que ningún paso individual acerque
                     al agente a una zona de peligro inmediato (p.ej. celda
                     adyacente a un cazador).
                     El goal nunca es bloqueado por este umbral (el agente
                     siempre puede intentar llegar a su destino).
                     Si None (defecto), no se aplica ningún bloqueo duro.

    Modos de accesibilidad (excluyentes):
      - known_map    → solo celdas exploradas que no sean obstáculo
      - visible_cells → solo celdas en el set que no sean obstáculo
      - ninguno      → todo el grid excepto obstáculos (debug/fallback)

    Retorna
    -------
    list[(x, y)]  : ruta desde start (excluido) hasta goal (incluido).
                    Si el goal no es alcanzable, retorna ruta parcial hacia
                    la celda transitable más cercana al goal encontrada.
                    Lista vacía si start == goal o no existe ruta posible.
    """
    if start == goal:
        return []

    grid_w = grid.width
    grid_h = grid.height

    def is_traversable(pos):
        """
        Reglas de accesibilidad en orden estricto:
        1. Obstáculos → siempre rechazados.
        2. known_map  → solo celdas exploradas (Equipo A).
        3. visible_cells → solo celdas en el set (Cazadores).
        4. Sin restricción (debug).
        """
        x, y = pos
        # 1. Obstáculos — nunca transitables por ningún agente
        if grid.cells[x][y].type == "obstacle":
            return False
        # 2. Equipo A: solo celdas exploradas del fog of war
        if known_map is not None:
            return known_map[x][y].get('explored', False)
        # 3. Cazadores: solo celdas dentro de su percepción conocida
        if visible_cells is not None:
            return pos in visible_cells
        # 4. Sin restricción (debug / fallback)
        return True

    # open_heap: (f, g, pos)
    # MODIFICACIÓN: Reducir peso de heurística para priorizar SEGURIDAD
    # f = g + 0.3*h  (en lugar de g + h)
    # Esto hace que prefiera rutas seguras aunque sean más largas
    open_heap = []
    h_start = _manhattan(start, goal)
    heapq.heappush(open_heap, (h_start * 0, 0, start))

    came_from = {start: None}
    g_score = {start: 0.0}

    # Seguimiento del mejor nodo parcial (más cercano al goal que hayamos visitado)
    best_partial = start
    best_partial_h = h_start

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        # Entrada obsoleta en el heap → ignorar
        if g > g_score.get(current, float('inf')):
            continue

        if current == goal:
            return _reconstruct(came_from, goal)

        for neighbor in get_neighbors(current, grid_w, grid_h):
            # El goal es siempre válido como DESTINO final, aunque no pase
            # is_traversable (el agente quiere llegar allí aunque no lo conozca).
            # Como nodo INTERMEDIO solo se permiten celdas transitables.
            if neighbor != goal and not is_traversable(neighbor):
                continue

            step_cost = cost_function(current, neighbor)
            # El costo mínimo por celda es 0.1 (nunca negativo ni cero)
            step_cost = max(0.1, step_cost)

            # Bloqueo duro por peligro local: si la celda es demasiado
            # peligrosa individualmente, se descarta como paso intermedio
            # aunque la ruta total acumulada pareciera aceptable.
            # El goal nunca se bloquea (el agente siempre puede intentar llegar).
            if max_step_cost is not None and neighbor != goal and step_cost > max_step_cost:
                continue
            tentative_g = g + step_cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                h = _manhattan(neighbor, goal)
                # MODIFICACIÓN: Reducir peso de h para priorizar seguridad (costo g)
                # f = g + 0.3*h (en lugar de g + h)
                heapq.heappush(
                    open_heap,
                    (tentative_g + h * 0.1, tentative_g, neighbor)
                )

                # Actualizar mejor nodo parcial (solo nodos transitables)
                if h < best_partial_h and (neighbor == goal or is_traversable(neighbor)):
                    best_partial_h = h
                    best_partial = neighbor

    # Goal no alcanzable: retornar path parcial hacia la celda más cercana
    if best_partial != start:
        return _reconstruct(came_from, best_partial)

    return []


def _reconstruct(came_from, node):
    """Reconstruye la ruta desde came_from hasta node, excluyendo el origen."""
    path = []
    current = node
    while came_from[current] is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path
