import random

from utils.constants import (
    GENE_GAMMA_INIT,
    GENE_BETA_INIT,
    GENE_DELTA_INIT,
    GENE_ALPHA_INIT,
    GENE_MUTATION_RATE,
)


class Genes:
    """
    Representa la personalidad de un cazador mediante cuatro genes flotantes.

    gamma : agresividad  — aumenta score de ATTACK
    beta  : persecución  — aumenta score de CHASE
    delta : evasión      — aumenta score de FLEE
    alpha : cohesión     — aumenta score de GROUP

    Rango de cada gen: [0.0, 1.0]
    """

    def __init__(
        self,
        gamma=GENE_GAMMA_INIT,
        beta=GENE_BETA_INIT,
        delta=GENE_DELTA_INIT,
        alpha=GENE_ALPHA_INIT,
    ):
        self.gamma = gamma  # agresividad
        self.beta  = beta   # persecución
        self.delta = delta  # evasión
        self.alpha = alpha  # cohesión

    def to_dict(self):
        """Serializa los genes a un diccionario."""
        return {
            "gamma": self.gamma,
            "beta":  self.beta,
            "delta": self.delta,
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, data):
        """Crea una instancia de Genes a partir de un diccionario."""
        return cls(
            gamma=data["gamma"],
            beta=data["beta"],
            delta=data["delta"],
            alpha=data["alpha"],
        )

    def __repr__(self):
        return (
            f"Genes(γ={self.gamma:.3f}, β={self.beta:.3f}, "
            f"δ={self.delta:.3f}, α={self.alpha:.3f})"
        )


def mutate(genes):
    """
    Retorna nuevos Genes con cada valor perturbado por una cantidad aleatoria
    uniforme en [-GENE_MUTATION_RATE, +GENE_MUTATION_RATE], clampeado a [0.0, 1.0].

    No muta in-place: el objeto original queda sin cambios.

    Parámetros
    ----------
    genes : Genes — genes actuales del cazador (de la vida anterior)

    Retorna
    -------
    Genes — nueva instancia con genes mutados
    """
    def _mutate_gene(value):
        delta = random.uniform(-GENE_MUTATION_RATE, GENE_MUTATION_RATE)
        return max(0.0, min(1.0, value + delta))

    return Genes(
        gamma=_mutate_gene(genes.gamma),
        beta=_mutate_gene(genes.beta),
        delta=_mutate_gene(genes.delta),
        alpha=_mutate_gene(genes.alpha),
    )


def calculate_fitness(kills, ticks_alive, deaths):
    """
    Calcula el fitness de un cazador para métricas de evolución.

        fitness = kills * 5 + ticks_alive - deaths

    Parámetros
    ----------
    kills       : int — enemigos eliminados en esta vida
    ticks_alive : int — ticks que sobrevivió
    deaths      : int — número de muertes acumuladas (normalmente 1 al calcular)

    Retorna
    -------
    float — valor de fitness
    """
    return kills * 5 + ticks_alive - deaths


def random_spawn_position(map_width, map_height, base_pos=None, min_dist=0,
                           exclude_zones=None):
    """
    Retorna una posición aleatoria en los bordes del mapa (zona de spawn
    de cazadores), respetando distancias mínimas a zonas de exclusión.

    Parámetros
    ----------
    map_width     : int — ancho del mapa (columnas)
    map_height    : int — alto del mapa (filas)
    base_pos      : tuple | None — posición de la base del Equipo A
    min_dist      : int — distancia Manhattan mínima desde base_pos (0 = sin límite)
    exclude_zones : list[((x,y), min_d)] | None — zonas adicionales a evitar;
                    cada entrada es (posición, distancia_mínima)

    Retorna
    -------
    (x, y) : tuple[int, int] — posición en el borde del mapa
    """
    def _candidate():
        edge = random.randint(0, 3)
        if edge == 0:    # borde superior
            return (random.randint(0, map_width - 1), 0)
        elif edge == 1:  # borde inferior
            return (random.randint(0, map_width - 1), map_height - 1)
        elif edge == 2:  # borde izquierdo
            return (0, random.randint(0, map_height - 1))
        else:            # borde derecho
            return (map_width - 1, random.randint(0, map_height - 1))

    def _valid(pos):
        # Restricción: distancia a la base
        if base_pos is not None and min_dist > 0:
            if abs(pos[0] - base_pos[0]) + abs(pos[1] - base_pos[1]) < min_dist:
                return False
        # Restricciones adicionales (recolectores, guardias, etc.)
        if exclude_zones:
            for zone_pos, zone_dist in exclude_zones:
                if abs(pos[0] - zone_pos[0]) + abs(pos[1] - zone_pos[1]) < zone_dist:
                    return False
        return True

    # Intentar hasta 200 veces encontrar una posición válida
    for _ in range(200):
        pos = _candidate()
        if _valid(pos):
            return pos

    # Fallback: relajar solo las zonas de aliados (mantener distancia a base si es posible)
    for _ in range(200):
        pos = _candidate()
        if base_pos is None or min_dist <= 0:
            return pos
        if abs(pos[0] - base_pos[0]) + abs(pos[1] - base_pos[1]) >= min_dist:
            return pos

    return _candidate()
