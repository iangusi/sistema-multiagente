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
# CONTROL DE TORRES (usado por environment._check_build_kits)
# ============================================================
MAX_TOWERS_PER_EXPLORATION = 0.08  # max_torres = celdas_exploradas × factor

# ============================================================
# HEURÍSTICAS RECOLECTOR
# ============================================================
HEUR_FLEE_HUNTER = 200          # cazador visible → HUIR domina
HEUR_RETURN_FULL = 130          # inventario lleno → IR_A_BASE
HEUR_BUILD_HAS_KIT = 150        # tiene kit → CONSTRUIR
HEUR_GOTO_RESOURCE_BASE = 100   # recurso conocido → IR_POR_RECURSO


# ============================================================
# HEURÍSTICAS GUARDIA
# ============================================================
HEUR_GUARD_ATTACK = 200         # puede atacar → ATACAR domina
HEUR_GUARD_FLEE_DANGER = 100    # en peligro → HUIR
HEUR_GUARD_DEFEND_ALLY = 80     # aliado en peligro → DEFENDER
HEUR_GUARD_EXPLORE_BASE = 20    # comportamiento base → EXPLORAR

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
REWARD_APPROACH_EXPLORE = 1       # acercarse a celda de exploración
REWARD_CELL_EXPLORED = 1          # celda nueva explorada
REWARD_APPROACH_RESOURCE = 1      # acercarse al recurso más cercano
REWARD_COLLECT = 5                # recolectar recurso
REWARD_GOING_TO_BASE_RES = 1      # ir a base cargando recursos
REWARD_DELIVER_RESOURCES = 5      # depositar recursos en base
REWARD_APPROACH_BUILD = 2         # acercarse a celda de construcción
REWARD_BUILD_NO_KIT = -1          # acción CONSTRUIR sin tener kit
REWARD_BUILD_TOWER = 5            # torre creada exitosamente
REWARD_GUARD_NEARBY = 1           # guardia aliado visible
REWARD_TOWER_NEARBY = 1           # torre aliada visible
REWARD_FLEE_HUNTER = 5            # huir con cazador cerca
REWARD_HUNTER_NEAR = -3           # cazador visible (penaliza estar en peligro)
REWARD_COLLECTOR_DIE = -10        # muerte del recolector
REWARD_BASE_NO_RES = -1           # ir a base sin recursos
REWARD_RESOURCE_FULL = -1         # ir a recurso estando lleno
REWARD_EXPLORE_WITH_KIT = -1      # explorar teniendo kit de construcción
REWARD_BAD_ACTION_HUNTER = -1     # explorar/recurso/base con cazador cerca
REWARD_FLEE_NO_HUNTER = -1        # huir sin cazador cerca
REWARD_APPROACH_BASE = 1          # acercarse a base cargando recursos

# ============================================================
# RECOMPENSAS RL – GUARDIA
# ============================================================
REWARD_GUARD_CELL_EXPLORED = 1    # celda nueva explorada
REWARD_GUARD_APPROACH_EXPLORE = 1 # acercarse a explorar sin aliado en peligro
REWARD_GUARD_EXPLORE_DANGER = -1  # explorar cuando aliado corre peligro
REWARD_KILL_HUNTER = 3            # eliminar cazador
REWARD_KILL_HUNTER_DEFEND = 2    # bonus por matar cazador mientras aliado en peligro
REWARD_GUARD_HUNTER_NO_DANGER = 1 # cazador cerca sin peligro propio
REWARD_GUARD_APPROACH_HUNTER = 1  # acercarse a cazador pudiendo atacar
REWARD_GUARD_HUNTER_IN_DANGER = -1 # cazador cerca con peligro propio
REWARD_GUARD_FLEE_DANGER = 2      # huir estando en peligro
REWARD_GUARD_FLEE_NO_ATTACK = 1   # huir sin poder atacar
REWARD_GUARD_CANT_ATTACK = -2     # cazador cerca sin poder atacar
REWARD_GUARD_APPROACH_ALLY = 1    # acercarse a aliado en peligro
REWARD_GUARD_DIE = -10            # muerte del guardia

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
# Valores base en la casilla del agente (se propagan en radio RISK_PROPAGATION_RANGE)
RISK_GUARD     = -1.0   # guardia reduce riesgo
RISK_TOWER     = -3.0   # torre reduce mucho el riesgo
RISK_COLLECTOR =  0.5   # recolector sube ligeramente el riesgo (atrae cazadores)
RISK_HUNTER    =  2.0   # cazador sube mucho el riesgo
RISK_PROPAGATION_RANGE = 4   # celdas de propagación alrededor
RISK_MEMORY_TICKS      = 4   # ticks antes de ignorar avistamiento de cazador
# Decaimiento por tick: 0.5^4 ≈ 0.06 → la celda tiende a 0 en ~4 ticks sin fuente
RISK_DECAY             = 0.5
RISK_UNEXPLORED        = 0.0  # celdas desconocidas: riesgo neutro

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
