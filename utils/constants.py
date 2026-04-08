# ============================================================
# MAPA
# ============================================================
MAP_WIDTH = 40
MAP_HEIGHT = 40
BASE_POSITION = (5, 5)  # alejado de los bordes para que los cazadores tarden más en llegar

# ============================================================
# OBSTACULOS
# ============================================================
NUM_OBSTACLES = 30

# ============================================================
# RECURSOS
# ============================================================
NUM_RESOURCE_NODES = 20
RESOURCE_MIN_AMOUNT = 5
RESOURCE_MAX_AMOUNT = 15
COLLECT_RATE = 5  # máximo por tick
WIN_RESOURCE_PERCENT = 0.80  # 80% para ganar

# ============================================================
# BUILD KITS
# ============================================================
BUILD_KIT_COST = 10  # cada 10 recursos depositados se genera un kit

# ============================================================
# AGENTES INICIALES
# ============================================================
NUM_COLLECTORS = 5
NUM_GUARDS = 5
NUM_HUNTERS = 15

# ============================================================
# RECOLECTOR
# ============================================================
COLLECTOR_HP = 2
COLLECTOR_SPEED = 1
COLLECTOR_VISION = 3
COLLECTOR_CARRY_CAPACITY = 10
COLLECTOR_ATTACK_RANGE = 0  # no ataca
COLLECTOR_ATTACK_COOLDOWN = 0

# ============================================================
# GUARDIA
# ============================================================
GUARD_HP = 1
GUARD_SPEED = 1
GUARD_VISION = 3
GUARD_ATTACK_RANGE = 2
GUARD_ATTACK_COOLDOWN = 2
GUARD_DAMAGE = 1

# ============================================================
# CAZADOR
# ============================================================
HUNTER_HP = 1
HUNTER_SPEED = 1
HUNTER_VISION = 5
HUNTER_ATTACK_RANGE = 1
HUNTER_ATTACK_COOLDOWN = 1
HUNTER_DAMAGE = 9999  # siempre letal
HUNTER_COMMUNICATION_RADIUS = 10
HUNTER_MEMORY_DURATION = 8  # ticks antes de olvidar

# Progresión on-kill
HUNTER_SPEED_INCREMENT = 0.25
HUNTER_HP_INCREMENT = 1
HUNTER_MAX_SPEED = 2.0  # base * 2
HUNTER_MAX_HP = 3       # base * 3

# Respawn
HUNTER_RESPAWN_TICKS = 15
HUNTER_MIN_SPAWN_DISTANCE = 15  # distancia mínima Manhattan desde la base al respawnear

# ============================================================
# TORRE
# ============================================================
TOWER_VISION = 4
TOWER_ATTACK_RANGE = 4
TOWER_ATTACK_COOLDOWN = 2
TOWER_DAMAGE = 1

# ============================================================
# GENES (VALORES INICIALES PARA CAZADORES)
# ============================================================
GENE_GAMMA_INIT = 0.5    # agresividad
GENE_BETA_INIT = 0.5     # persecución
GENE_DELTA_INIT = 0.5    # evasión
GENE_ALPHA_INIT = 0.5    # cohesión
GENE_MUTATION_RATE = 0.1  # ε para mutación

# ============================================================
# FASES ESTRATÉGICAS
# ============================================================
PHASE_EARLY_THRESHOLD = 0.30    # < 30% explorado = EARLY
PHASE_MID_THRESHOLD = 0.65      # 30-65% = MID, > 65% = LATE

# ============================================================
# HEURÍSTICAS RECOLECTOR
# ============================================================
HEUR_FLEE_ENEMY_NEAR = 200
HEUR_FLEE_ENEMY_NO_GUARD = 120

HEUR_RETURN_FULL = 125
HEUR_RETURN_PARTIAL_FACTOR = 40
HEUR_RETURN_EMPTY_PENALTY = -80

HEUR_GOTO_RESOURCE_BASE = 60
HEUR_GOTO_RESOURCE_EMPTY_BONUS = 40
HEUR_GOTO_RESOURCE_NO_GUARD = 20   # bonus: animar recolección en zona peligrosa sin guardia
HEUR_GOTO_RESOURCE_MID_BONUS = 30

HEUR_EXPLORE_LOW_COVERAGE = 50
HEUR_EXPLORE_NO_RESOURCES_KNOWN = 40
HEUR_EXPLORE_NO_GUARD_PENALTY = -60
HEUR_EXPLORE_EARLY_BONUS = 40

HEUR_BUILD_HAS_KIT = 100
HEUR_BUILD_TOO_MANY_TOWERS = -80
HEUR_BUILD_LAST_COLLECTOR = -150
HEUR_BUILD_LOW_EXPLORATION = 40    # bonus: construir defensas en EARLY es crítico

# Anti-estancamiento
STAGNATION_THRESHOLD = 12
STAGNATION_PENALTY_RATE = 8
MAX_TOWERS_PER_EXPLORATION = 0.08

# Protocolo early game
EARLY_GAME_SAFE_RADIUS = 8
EARLY_GAME_FLEE_BONUS = 80

# ============================================================
# HEURÍSTICAS GUARDIA
# ============================================================
HEUR_ESCORT_VULNERABLE = 100
HEUR_ESCORT_HAS_KIT = 130
HEUR_ESCORT_REDUNDANT = -20   # reducido: no dispersar guardias tan agresivamente
HEUR_ESCORT_COLLECTOR_AT_BASE = -40

HEUR_ATTACK_IN_RANGE = 100
HEUR_ATTACK_ADVANTAGE = 40
HEUR_ATTACK_DISADVANTAGE = -60

HEUR_INTERCEPT_ENEMY_NEAR = 80
HEUR_INTERCEPT_COLLECTOR_DANGER = 50

HEUR_SCOUT_LOW_EXPLORATION = 60
HEUR_SCOUT_EARLY_BONUS = 40
HEUR_SCOUT_NO_ESCORT_NEEDED = 30

HEUR_DEFEND_HAS_TOWERS = 50
HEUR_DEFEND_HIGH_RISK = 40

HEUR_INVESTIGATE_RECENT = 60

HEUR_PATROL_BASE = 20

# ============================================================
# Q-LEARNING
# ============================================================
QL_ALPHA = 0.25
QL_GAMMA = 0.9
QL_EPSILON = 0.20
QL_EPSILON_DECAY = 0.995
QL_EPSILON_MIN = 0.04  # bajo para explotar la política aprendida

# ============================================================
# RECOMPENSAS RL – RECOLECTOR
# ============================================================
REWARD_DELIVER_RESOURCES = 10
REWARD_COLLECT = 5
REWARD_EXPLORE = 3
REWARD_BUILD_TOWER = 8
REWARD_DANGER_ZONE = -2
REWARD_LOSE_RESOURCES = -5
REWARD_COLLECTOR_DIE = -10
REWARD_IDLE = -3
REWARD_RETURN_EMPTY = -4
REWARD_APPROACH_RESOURCE = 1
REWARD_STAGNATION = -5
REWARD_EXPLORE_WITH_ESCORT = 5

# ============================================================
# RECOMPENSAS RL – GUARDIA
# ============================================================
REWARD_KILL_HUNTER = 15
REWARD_PROTECT_COLLECTOR = 10
REWARD_INTERCEPT = 6
REWARD_DEFEND_ZONE = 5
REWARD_GUARD_DIE = -10
REWARD_COLLECTOR_DIES_NEARBY = -8
REWARD_COLLECTOR_DIES_WHILE_ADJACENT = -15
REWARD_BAD_DECISION = -3
REWARD_SOLE_ESCORT = 12
REWARD_REDUNDANT_ESCORT = -4
REWARD_COLLECTOR_SAFE_NO_ESCORT = -1
REWARD_SCOUT_NEW_CELLS = 4
REWARD_SCOUT_FIND_RESOURCE = 7
REWARD_ESCORT_COLLECTOR_COLLECTS = 3
REWARD_GUARD_IDLE = -2

# ============================================================
# ENTRENAMIENTO POR CURRÍCULO
# ============================================================
TRAINING_EPISODES_PHASE_1 = 800   # más tiempo para converger sin hunters
TRAINING_EPISODES_PHASE_2 = 600
TRAINING_EPISODES_PHASE_3 = 800
TRAINING_EPISODES_PHASE_4 = 1000
TRAINING_MAX_TICKS_PER_EPISODE = 3000
TRAINING_EPSILON_RESET_FACTOR = 0.6  # reset moderado → menos choque al subir dificultad
QTABLE_SAVE_PATH = "data/"

# ============================================================
# MAPA DE RIESGO
# ============================================================
RISK_DECAY = 0.98         # por tick
RISK_DIFFUSION = 0.1      # factor de difusión a vecinos
RISK_ENEMY_WEIGHT = 1.0
RISK_TOWER_REDUCTION = 0.5
RISK_UNEXPLORED = 0.3     # riesgo base de lo desconocido

# ============================================================
# USER INPUT
# ============================================================
USER_INFLUENCE_WEIGHT = 0.4  # peso del click del usuario sobre cazadores

# ============================================================
# SIMULACIÓN
# ============================================================
TICK_RATE = 2  # ticks por segundo
MAX_TICKS = 5000

# ============================================================
# RENDERING (PYGAME)
# ============================================================
CELL_SIZE = 16
SIDEBAR_WIDTH = 260
WINDOW_WIDTH = MAP_WIDTH * CELL_SIZE + SIDEBAR_WIDTH  # 900
WINDOW_HEIGHT = MAP_HEIGHT * CELL_SIZE  # 640
FPS = TICK_RATE

# Colores (R, G, B)
COLOR_BG = (30, 30, 30)
COLOR_GRID_LINE = (50, 50, 50)
COLOR_FOG = (15, 15, 15)
COLOR_EXPLORED = (45, 45, 50)
COLOR_RESOURCE = (255, 200, 50)
COLOR_BASE = (80, 130, 255)
COLOR_OBSTACLE = (90, 60, 40)

COLOR_COLLECTOR = (0, 220, 220)
COLOR_COLLECTOR_WITH_KIT = (0, 255, 180)
COLOR_GUARD = (60, 100, 255)
COLOR_HUNTER = (220, 50, 50)
COLOR_HUNTER_DEAD = (100, 30, 30)
COLOR_TOWER = (0, 200, 80)
COLOR_TOWER_RANGE = (0, 200, 80, 40)  # semi-transparente

COLOR_SIDEBAR_BG = (20, 20, 25)
COLOR_TEXT = (200, 200, 200)
COLOR_TEXT_HIGHLIGHT = (255, 255, 100)
