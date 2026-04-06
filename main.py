import sys
import pygame

from environment import Environment
from rl.q_learning import QLearning
from utils.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT, CELL_SIZE, SIDEBAR_WIDTH, FPS,
    MAP_WIDTH, MAP_HEIGHT, BASE_POSITION,
    COLOR_BG, COLOR_FOG, COLOR_EXPLORED, COLOR_GRID_LINE,
    COLOR_RESOURCE, COLOR_BASE, COLOR_OBSTACLE,
    COLOR_COLLECTOR, COLOR_COLLECTOR_WITH_KIT,
    COLOR_GUARD, COLOR_HUNTER, COLOR_HUNTER_DEAD,
    COLOR_TOWER, COLOR_SIDEBAR_BG, COLOR_TEXT, COLOR_TEXT_HIGHLIGHT,
    NUM_COLLECTORS, NUM_GUARDS, NUM_HUNTERS,
    QL_ALPHA, QL_GAMMA, QL_EPSILON_MIN, QL_EPSILON_DECAY,
    QTABLE_SAVE_PATH,
)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(screen, env, font_sm, font_md):
    """
    Dibuja el estado completo de la simulación.

    El jugador ve:
    - Todo lo que el Equipo A ha explorado (fog of war)
    - Cazadores siempre visibles (el jugador los controla con clicks)
    - Sidebar con estadísticas en tiempo real
    """
    screen.fill(COLOR_BG)

    # ------------------------------------------------------------------
    # Grid: fog / explorado / obstáculos / recursos / base
    # ------------------------------------------------------------------
    for x in range(MAP_WIDTH):
        for y in range(MAP_HEIGHT):
            rx = x * CELL_SIZE
            ry = y * CELL_SIZE
            cell = env.cells[x][y]
            info = env.known_map[x][y]

            if not info['explored']:
                pygame.draw.rect(screen, COLOR_FOG,
                                 (rx, ry, CELL_SIZE, CELL_SIZE))
                continue

            # Celda explorada
            pygame.draw.rect(screen, COLOR_EXPLORED,
                             (rx, ry, CELL_SIZE, CELL_SIZE))

            ctype = cell.type

            if ctype == 'obstacle':
                pygame.draw.rect(screen, COLOR_OBSTACLE,
                                 (rx, ry, CELL_SIZE, CELL_SIZE))
                continue

            if ctype == 'base' or (x, y) == BASE_POSITION:
                pygame.draw.rect(screen, COLOR_BASE,
                                 (rx + 1, ry + 1, CELL_SIZE - 2, CELL_SIZE - 2))

            if ctype == 'resource' and cell.resource_amount > 0:
                radius = max(2, min(CELL_SIZE // 2 - 1, cell.resource_amount // 2))
                pygame.draw.circle(screen, COLOR_RESOURCE,
                                   (rx + CELL_SIZE // 2, ry + CELL_SIZE // 2),
                                   radius)

    # ------------------------------------------------------------------
    # Torres (siempre visibles si exploradas)
    # ------------------------------------------------------------------
    for tower in env.towers:
        tx, ty = tower.position
        if not env.known_map[tx][ty]['explored']:
            continue
        rx = tx * CELL_SIZE
        ry = ty * CELL_SIZE
        # Rango semi-transparente
        tr = tower.attack_range
        range_surf = pygame.Surface(((tr * 2 + 1) * CELL_SIZE,
                                     (tr * 2 + 1) * CELL_SIZE), pygame.SRCALPHA)
        range_surf.fill((0, 200, 80, 18))
        screen.blit(range_surf,
                    (rx - tr * CELL_SIZE, ry - tr * CELL_SIZE))
        # Cuerpo de la torre
        pygame.draw.rect(screen, COLOR_TOWER,
                         (rx + 2, ry + 2, CELL_SIZE - 4, CELL_SIZE - 4))
        # Indicador de cooldown (línea inferior proporcional)
        if tower.attack_cooldown > 0:
            cd_ratio = 1.0 - tower.current_cooldown / tower.attack_cooldown
            cd_w = int((CELL_SIZE - 4) * cd_ratio)
            if cd_w > 0:
                pygame.draw.rect(screen, (200, 255, 200),
                                 (rx + 2, ry + CELL_SIZE - 4, cd_w, 2))

    # ------------------------------------------------------------------
    # Líneas del grid
    # ------------------------------------------------------------------
    for x in range(MAP_WIDTH + 1):
        pygame.draw.line(screen, COLOR_GRID_LINE,
                         (x * CELL_SIZE, 0),
                         (x * CELL_SIZE, MAP_HEIGHT * CELL_SIZE))
    for y in range(MAP_HEIGHT + 1):
        pygame.draw.line(screen, COLOR_GRID_LINE,
                         (0, y * CELL_SIZE),
                         (MAP_WIDTH * CELL_SIZE, y * CELL_SIZE))

    # ------------------------------------------------------------------
    # Agentes del Equipo A (sólo dentro de zona explorada)
    # ------------------------------------------------------------------
    for c in env.collectors:
        if not c.is_alive:
            continue
        cx_px = c.position[0] * CELL_SIZE + CELL_SIZE // 2
        cy_px = c.position[1] * CELL_SIZE + CELL_SIZE // 2
        color = COLOR_COLLECTOR_WITH_KIT if c.has_build_kit else COLOR_COLLECTOR
        pygame.draw.circle(screen, color, (cx_px, cy_px), CELL_SIZE // 2 - 1)
        # Barra de recursos
        if c.carrying_resources > 0:
            ratio = c.carrying_resources / c.carrying_capacity
            bar_w = int((CELL_SIZE - 2) * ratio)
            pygame.draw.rect(screen, (255, 220, 0),
                             (c.position[0] * CELL_SIZE + 1,
                              c.position[1] * CELL_SIZE + CELL_SIZE - 3,
                              bar_w, 2))

    for g in env.guards:
        if not g.is_alive:
            continue
        gx_px = g.position[0] * CELL_SIZE + CELL_SIZE // 2
        gy_px = g.position[1] * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, COLOR_GUARD, (gx_px, gy_px), CELL_SIZE // 2 - 1)
        # Indicador de cooldown
        if g.current_cooldown > 0:
            pygame.draw.circle(screen, (100, 100, 255),
                               (gx_px, gy_px), CELL_SIZE // 2 - 1, 1)

    # ------------------------------------------------------------------
    # Cazadores — SIEMPRE visibles (el jugador los controla)
    # ------------------------------------------------------------------
    for h in env.hunters:
        hx_px = h.position[0] * CELL_SIZE + CELL_SIZE // 2
        hy_px = h.position[1] * CELL_SIZE + CELL_SIZE // 2
        color = COLOR_HUNTER if h.is_alive else COLOR_HUNTER_DEAD
        pygame.draw.circle(screen, color, (hx_px, hy_px), CELL_SIZE // 2 - 1)
        # HP extra visible (punto blanco por cada HP > 1)
        if h.is_alive and h.current_hp > 1:
            for i in range(min(h.current_hp - 1, 2)):
                pygame.draw.circle(screen, (255, 255, 255),
                                   (hx_px - 3 + i * 4, hy_px - CELL_SIZE // 2 + 2), 1)

    # ------------------------------------------------------------------
    # Marcador del click del usuario
    # ------------------------------------------------------------------
    if env.user_target_position:
        ux, uy = env.user_target_position
        mx = ux * CELL_SIZE
        my = uy * CELL_SIZE
        pygame.draw.line(screen, (255, 255, 0),
                         (mx + 2, my + 2), (mx + CELL_SIZE - 3, my + CELL_SIZE - 3), 2)
        pygame.draw.line(screen, (255, 255, 0),
                         (mx + CELL_SIZE - 3, my + 2), (mx + 2, my + CELL_SIZE - 3), 2)

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    _render_sidebar(screen, env, font_sm, font_md)

    pygame.display.flip()


def _render_sidebar(screen, env, font_sm, font_md):
    """Panel lateral derecho con estadísticas de la simulación."""
    sx = MAP_WIDTH * CELL_SIZE
    pygame.draw.rect(screen, COLOR_SIDEBAR_BG,
                     (sx, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))

    # Título
    title = font_md.render('SIMULACIÓN', True, COLOR_TEXT_HIGHLIGHT)
    screen.blit(title, (sx + 8, 6))

    y = 28
    line_h = 16

    def draw_line(text, color=COLOR_TEXT, bold=False):
        nonlocal y
        f = font_md if bold else font_sm
        surf = f.render(text, True, color)
        screen.blit(surf, (sx + 8, y))
        y += line_h

    def draw_sep():
        nonlocal y
        pygame.draw.line(screen, (60, 60, 70),
                         (sx + 4, y + 4), (sx + SIDEBAR_WIDTH - 8, y + 4))
        y += 10

    # --- Tick ---
    draw_line(f'Tick: {env.current_tick}', COLOR_TEXT_HIGHLIGHT)
    draw_sep()

    # --- Recursos ---
    draw_line('Recursos depositados:', bold=True)
    total_r = max(1, env.total_resources)
    pct = env.base_resources / total_r * 100
    draw_line(f'  {env.base_resources} / {env.total_resources}')
    draw_line(f'  ({pct:.1f}%  meta: 80%)')

    # Barra de progreso
    bar_x = sx + 8
    bar_w = SIDEBAR_WIDTH - 16
    bar_h = 8
    pygame.draw.rect(screen, (50, 50, 55), (bar_x, y, bar_w, bar_h))
    fill_w = int(bar_w * min(1.0, env.base_resources / total_r))
    if fill_w > 0:
        pygame.draw.rect(screen, (0, 180, 80), (bar_x, y, fill_w, bar_h))
    # Marca del 80 %
    mark_x = bar_x + int(bar_w * 0.8)
    pygame.draw.line(screen, (255, 220, 0),
                     (mark_x, y - 2), (mark_x, y + bar_h + 2), 2)
    y += bar_h + 6
    draw_sep()

    # --- Equipo A ---
    draw_line('── Equipo A ──', COLOR_TEXT_HIGHLIGHT, bold=True)
    alive_c = sum(1 for c in env.collectors if c.is_alive)
    alive_g = sum(1 for g in env.guards     if g.is_alive)
    kits    = sum(1 for c in env.collectors if c.has_build_kit)
    draw_line(f'  Recolectores: {alive_c} / {NUM_COLLECTORS}')
    draw_line(f'  Guardias:     {alive_g} / {NUM_GUARDS}')
    draw_line(f'  Torres:       {len(env.towers)}')
    draw_line(f'  Build kits:   {kits}')
    draw_sep()

    # --- Equipo B ---
    draw_line('── Equipo B ──', COLOR_TEXT_HIGHLIGHT, bold=True)
    alive_h  = sum(1 for h in env.hunters if h.is_alive)
    respawn_h = sum(1 for h in env.hunters if not h.is_alive
                    and h.respawn_timer > 0)
    draw_line(f'  Cazadores:   {alive_h} / {NUM_HUNTERS}')
    draw_line(f'  Respawn en:  {respawn_h}')
    draw_sep()

    # --- Genes promedio ---
    alive_hunters = [h for h in env.hunters if h.is_alive]
    if alive_hunters:
        n = len(alive_hunters)
        ag = sum(h.genes.gamma for h in alive_hunters) / n
        ab = sum(h.genes.beta  for h in alive_hunters) / n
        ad = sum(h.genes.delta for h in alive_hunters) / n
        aa = sum(h.genes.alpha for h in alive_hunters) / n
        draw_line('── Genes (prom) ──', COLOR_TEXT_HIGHLIGHT, bold=True)
        draw_line(f'  Agresividad:  {ag:.2f}')
        draw_line(f'  Persecución:  {ab:.2f}')
        draw_line(f'  Evasión:      {ad:.2f}')
        draw_line(f'  Cohesión:     {aa:.2f}')
        draw_sep()

    # --- RL epsilon ---
    draw_line('── RL ──', COLOR_TEXT_HIGHLIGHT, bold=True)
    draw_line(f'  ε col: {env.collector_ql.epsilon:.3f}')
    draw_line(f'  ε grd: {env.guard_ql.epsilon:.3f}')
    draw_sep()

    # --- Último evento ---
    draw_line('── Último evento ──', COLOR_TEXT_HIGHLIGHT, bold=True)
    ev = env.last_event or '—'
    # Partir en trozos de 26 chars para no salir del sidebar
    while ev:
        draw_line('  ' + ev[:26])
        ev = ev[26:]
        if y > WINDOW_HEIGHT - line_h * 3:
            break


def _show_victory(screen, winner, font_big, font_sm):
    """Pantalla de fin de juego. Retorna cuando el usuario hace click o cierra."""
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, 0))

    if winner == 'A':
        msg   = 'EQUIPO A GANA'
        color = (80, 200, 120)
        sub   = 'Recursos recolectados: 80 %'
    else:
        msg   = 'EQUIPO B GANA'
        color = (220, 80, 80)
        sub   = 'Todos los agentes del Equipo A eliminados'

    cx = WINDOW_WIDTH  // 2
    cy = WINDOW_HEIGHT // 2

    title_surf = font_big.render(msg, True, color)
    screen.blit(title_surf, title_surf.get_rect(center=(cx, cy - 30)))

    sub_surf = font_sm.render(sub, True, (200, 200, 200))
    screen.blit(sub_surf, sub_surf.get_rect(center=(cx, cy + 10)))

    hint_surf = font_sm.render('Click para cerrar', True, (150, 150, 150))
    screen.blit(hint_surf, hint_surf.get_rect(center=(cx, cy + 50)))

    pygame.display.flip()

    # Esperar click o cierre
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                return


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    pygame.display.set_caption('Simulación Multi-Agente')

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock  = pygame.time.Clock()

    font_sm  = pygame.font.SysFont('monospace', 12)
    font_md  = pygame.font.SysFont('monospace', 13, bold=True)
    font_big = pygame.font.SysFont('monospace', 28, bold=True)

    # Crear Q-learning e intentar cargar Q-tables preentrenadas
    collector_ql = QLearning(
        actions=['EXPLORE', 'GO_TO_RESOURCE', 'RETURN_TO_BASE', 'FLEE', 'BUILD_TOWER'],
        alpha=QL_ALPHA, gamma=QL_GAMMA,
        epsilon=QL_EPSILON_MIN,       # epsilon bajo: ya está entrenado
        epsilon_decay=QL_EPSILON_DECAY,
        epsilon_min=QL_EPSILON_MIN,
    )
    guard_ql = QLearning(
        actions=['PATROL', 'ESCORT', 'ATTACK', 'INTERCEPT',
                 'DEFEND_ZONE', 'INVESTIGATE', 'SCOUT'],
        alpha=QL_ALPHA, gamma=QL_GAMMA,
        epsilon=QL_EPSILON_MIN,
        epsilon_decay=QL_EPSILON_DECAY,
        epsilon_min=QL_EPSILON_MIN,
    )
    collector_ql.load(f"{QTABLE_SAVE_PATH}collector_qtable.pkl")
    guard_ql.load(f"{QTABLE_SAVE_PATH}guard_qtable.pkl")

    env = Environment(collector_ql=collector_ql, guard_ql=guard_ql)

    running = True
    while running:
        # --- Eventos ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # Sólo procesar clicks dentro del área del grid
                if mx < MAP_WIDTH * CELL_SIZE:
                    env.handle_click(mx, my)

        if not running:
            break

        # --- Tick de simulación ---
        env.tick()

        # --- Render ---
        render(screen, env, font_sm, font_md)

        clock.tick(FPS)

        # --- Fin de partida ---
        if env.game_over:
            render(screen, env, font_sm, font_md)   # último frame antes de overlay
            _show_victory(screen, env.winner, font_big, font_sm)
            running = False

    pygame.quit()
    sys.exit(0)


if __name__ == '__main__':
    main()
