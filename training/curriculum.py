"""
Define las fases de entrenamiento con dificultad creciente.
Cada fase hereda las Q-tables de la fase anterior.
"""
from utils.constants import (
    TRAINING_EPISODES_PHASE_1,
    TRAINING_EPISODES_PHASE_2,
    TRAINING_EPISODES_PHASE_3,
    TRAINING_EPISODES_PHASE_4,
    NUM_HUNTERS,
)

CURRICULUM = [
    {
        "phase": 1,
        "name": "Exploración y recolección básica",
        "num_hunters": 0,
        "episodes": TRAINING_EPISODES_PHASE_1,
        "description": (
            "Sin enemigos. Los recolectores aprenden a explorar, ir a recursos, "
            "volver a base cuando llenos y no volver vacíos. "
            "Los guardias aprenden a patrullar y hacer scouting."
        ),
        "success_metric": "win_rate >= 0.99",
    },
    {
        "phase": 2,
        "name": "Amenaza baja",
        "num_hunters": 4,
        "episodes": TRAINING_EPISODES_PHASE_2,
        "description": (
            "Pocos cazadores. Los recolectores aprenden a huir y valorar guardias. "
            "Los guardias aprenden a escoltar y atacar."
        ),
        "success_metric": "win_rate >= 0.95",
    },
    {
        "phase": 3,
        "name": "Amenaza media",
        "num_hunters": 10,
        "episodes": TRAINING_EPISODES_PHASE_3,
        "description": (
            "Presión real. Los agentes aprenden construcción estratégica, "
            "coordinación y distribución de guardias sin sobre-escoltar."
        ),
        "success_metric": "win_rate >= 0.90",
    },
    {
        "phase": 4,
        "name": "Dificultad completa",
        "num_hunters": NUM_HUNTERS,
        "episodes": TRAINING_EPISODES_PHASE_4,
        "description": (
            "Escenario real del juego con todos los cazadores. "
            "Afinación final bajo máxima presión."
        ),
        "success_metric": "win_rate >= 0.85",
    },
]
