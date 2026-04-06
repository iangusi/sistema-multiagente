import random
from collections import defaultdict


class QLearning:
    """
    Implementación genérica de Q-learning tabular.

    Una misma instancia es compartida por todos los agentes del mismo tipo
    (todos los recolectores comparten una instancia, todos los guardias otra).

    La Q_table es un dict de dicts: {state: {action: float}}.
    Los estados y acciones no vistos se inicializan en 0.0 automáticamente.
    """

    def __init__(self, actions, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
        """
        Parámetros
        ----------
        actions       : list[str]  — acciones posibles para este tipo de agente
        alpha         : float      — tasa de aprendizaje (QL_ALPHA)
        gamma         : float      — factor de descuento (QL_GAMMA)
        epsilon       : float      — probabilidad de exploración inicial (QL_EPSILON)
        epsilon_decay : float      — factor de decaimiento por tick (QL_EPSILON_DECAY)
        epsilon_min   : float      — piso mínimo de epsilon (QL_EPSILON_MIN)
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q_table[state][action] = float, inicializado a 0.0 si no existe
        self.Q_table = defaultdict(lambda: defaultdict(float))

    # ------------------------------------------------------------------
    # Consulta de valores Q
    # ------------------------------------------------------------------

    def _q(self, state, action):
        """Devuelve Q[state][action], 0.0 si no existe."""
        return self.Q_table[state][action]

    def _argmax_action(self, state, heuristic_biases):
        """
        Devuelve la acción con mayor (Q[state][a] + bias).
        En caso de empate elige aleatoriamente entre los empatados.
        """
        biases = heuristic_biases or {}
        scores = {a: self._q(state, a) + biases.get(a, 0.0) for a in self.actions}
        max_score = max(scores.values())
        best = [a for a, s in scores.items() if s == max_score]
        return random.choice(best)

    # ------------------------------------------------------------------
    # Selección de acción
    # ------------------------------------------------------------------

    def get_action(self, state, heuristic_biases=None):
        """
        Selección epsilon-greedy sobre (Q[state][action] + bias).

        Con probabilidad epsilon elige una acción aleatoria (exploración).
        Con probabilidad 1-epsilon elige argmax(Q + bias) (explotación).

        Parámetros
        ----------
        state             : tuple hasheable que representa el estado discreto
        heuristic_biases  : dict {action: float} o None

        Retorna
        -------
        str : acción seleccionada
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return self._argmax_action(state, heuristic_biases)

    def get_best_action(self, state, heuristic_biases=None):
        """
        Como get_action pero sin exploración (epsilon=0).
        Útil para evaluación o para forzar explotación.

        Retorna
        -------
        str : acción con mayor (Q[state][a] + bias)
        """
        return self._argmax_action(state, heuristic_biases)

    # ------------------------------------------------------------------
    # Actualización
    # ------------------------------------------------------------------

    def update(self, state, action, reward, next_state):
        """
        Actualización estándar de Q-learning:

            Q[s][a] += alpha * (reward + gamma * max_a'(Q[s'][a']) - Q[s][a])

        Parámetros
        ----------
        state      : tuple — estado en el que se tomó la acción
        action     : str   — acción ejecutada
        reward     : float — recompensa recibida
        next_state : tuple — estado resultante
        """
        current_q = self._q(state, action)
        max_next_q = max(self._q(next_state, a) for a in self.actions)
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        self.Q_table[state][action] = current_q + self.alpha * td_error

    # ------------------------------------------------------------------
    # Decaimiento de epsilon
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        """
        Reduce epsilon multiplicativamente, respetando el piso mínimo:

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        Debe llamarse una vez por tick desde el environment.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, filepath):
        """Guardar Q-table a disco."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.Q_table), f)

    def load(self, filepath):
        """Cargar Q-table desde disco. Retorna True si el archivo existe."""
        import pickle
        import os
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                loaded = pickle.load(f)
                self.Q_table = defaultdict(lambda: defaultdict(float), loaded)
            print(f"Q-table cargada: {len(self.Q_table)} estados desde {filepath}")
            return True
        return False
