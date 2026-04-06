"""
Motor de entrenamiento headless.
Corre episodios sin pygame a máxima velocidad.
Guarda Q-tables entre fases del currículo.
"""
import os
import pickle
import time

from environment import Environment
from rl.q_learning import QLearning
from utils.constants import (
    QL_ALPHA, QL_GAMMA, QL_EPSILON, QL_EPSILON_DECAY, QL_EPSILON_MIN,
    TRAINING_MAX_TICKS_PER_EPISODE,
    TRAINING_EPSILON_RESET_FACTOR,
    QTABLE_SAVE_PATH,
)

SAVE_EVERY = 50   # episodios entre auto-guardados de checkpoint


class Trainer:
    """Entrena los agentes de Equipo A mediante un currículo progresivo."""

    def __init__(self):
        self.collector_ql = QLearning(
            actions=['EXPLORE', 'GO_TO_RESOURCE', 'RETURN_TO_BASE', 'FLEE', 'BUILD_TOWER'],
            alpha=QL_ALPHA, gamma=QL_GAMMA,
            epsilon=QL_EPSILON, epsilon_decay=QL_EPSILON_DECAY,
            epsilon_min=QL_EPSILON_MIN,
        )
        self.guard_ql = QLearning(
            actions=['PATROL', 'ESCORT', 'ATTACK', 'INTERCEPT',
                     'DEFEND_ZONE', 'INVESTIGATE', 'SCOUT'],
            alpha=QL_ALPHA, gamma=QL_GAMMA,
            epsilon=QL_EPSILON, epsilon_decay=QL_EPSILON_DECAY,
            epsilon_min=QL_EPSILON_MIN,
        )
        self.metrics_history = []
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Carga Q-tables previas si existen, para continuar entrenamiento."""
        c_path = f"{QTABLE_SAVE_PATH}collector_qtable.pkl"
        g_path = f"{QTABLE_SAVE_PATH}guard_qtable.pkl"
        if os.path.exists(c_path) and os.path.exists(g_path):
            print("Checkpoint encontrado — cargando Q-tables previas...", flush=True)
            self.collector_ql.load(c_path)
            self.guard_ql.load(g_path)
            # Mantener epsilon mínimo para no re-explorar desde cero
            self.collector_ql.epsilon = max(self.collector_ql.epsilon, QL_EPSILON_MIN)
            self.guard_ql.epsilon     = max(self.guard_ql.epsilon,     QL_EPSILON_MIN)
            print(f"  ε actual: {self.collector_ql.epsilon:.3f}", flush=True)
        else:
            print("No se encontró checkpoint — entrenando desde cero.", flush=True)

    def _save_latest(self):
        """Guarda solo los archivos 'latest' (checkpoint rápido)."""
        os.makedirs(QTABLE_SAVE_PATH, exist_ok=True)
        with open(f"{QTABLE_SAVE_PATH}collector_qtable.pkl", 'wb') as f:
            pickle.dump(dict(self.collector_ql.Q_table), f)
        with open(f"{QTABLE_SAVE_PATH}guard_qtable.pkl", 'wb') as f:
            pickle.dump(dict(self.guard_ql.Q_table), f)

    def train_curriculum(self, curriculum):
        """Ejecuta todas las fases del currículo en orden."""
        try:
            for phase_config in curriculum:
                print(f"\n{'='*60}", flush=True)
                print(f"FASE {phase_config['phase']}: {phase_config['name']}", flush=True)
                print(f"Cazadores: {phase_config['num_hunters']}", flush=True)
                print(f"Episodios:  {phase_config['episodes']}", flush=True)
                print(f"{'='*60}", flush=True)

                # Al iniciar fase >1: subir epsilon parcialmente para re-explorar
                if phase_config['phase'] > 1:
                    new_eps = max(
                        self.collector_ql.epsilon,
                        QL_EPSILON * TRAINING_EPSILON_RESET_FACTOR,
                    )
                    self.collector_ql.epsilon = new_eps
                    self.guard_ql.epsilon     = new_eps
                    print(f"  Epsilon reiniciado a {new_eps:.3f}", flush=True)

                phase_metrics = self._train_phase(phase_config)
                self.metrics_history.append(phase_metrics)

                self._save_qtables(phase_config['phase'])

                print(f"\nFase {phase_config['phase']} completada.", flush=True)
                print(f"  Win rate:             {phase_metrics['win_rate']:.2%}", flush=True)
                print(f"  Avg resources:        {phase_metrics['avg_resources']:.1f}%", flush=True)
                print(f"  Avg collectors alive: {phase_metrics['avg_collectors_alive']:.2f}", flush=True)
                print(f"  Q-table sizes: C={len(self.collector_ql.Q_table)}"
                      f"  G={len(self.guard_ql.Q_table)}", flush=True)

        except KeyboardInterrupt:
            print("\n\n[!] Entrenamiento interrumpido — guardando checkpoint...", flush=True)
            self._save_latest()
            print("[✓] Q-tables guardadas. Puedes retomar el entrenamiento.", flush=True)

    def _train_phase(self, phase_config):
        """Ejecuta los episodios de una fase. Retorna métricas."""
        wins                    = 0
        total_resources_pct     = 0.0
        total_collectors_alive  = 0
        episodes                = phase_config['episodes']
        num_hunters             = phase_config['num_hunters']
        phase_start             = time.time()
        LOG_EVERY               = 10

        for ep in range(episodes):
            env = Environment(
                num_hunters_override=num_hunters,
                collector_ql=self.collector_ql,
                guard_ql=self.guard_ql,
                headless=True,
            )

            for _ in range(TRAINING_MAX_TICKS_PER_EPISODE):
                env.tick()
                if env.game_over:
                    break

            if env.winner == 'A':
                wins += 1

            # Evitar división por cero si win_target es 0
            if env.win_target > 0:
                total_resources_pct += env.base_resources / env.win_target
            else:
                total_resources_pct += 1.0

            total_collectors_alive += sum(1 for c in env.collectors if c.is_alive)

            # Decay epsilon una vez por episodio
            self.collector_ql.decay_epsilon()
            self.guard_ql.decay_epsilon()

            if (ep + 1) % LOG_EVERY == 0:
                elapsed     = time.time() - phase_start
                ep_sec      = elapsed / (ep + 1)
                remaining   = ep_sec * (episodes - ep - 1)
                recent_wr   = wins / (ep + 1)
                print(f"  Ep {ep+1:4d}/{episodes} | "
                      f"WR={recent_wr:.2%} | "
                      f"ε={self.collector_ql.epsilon:.3f} | "
                      f"Q: C={len(self.collector_ql.Q_table)} G={len(self.guard_ql.Q_table)} | "
                      f"{ep_sec:.2f}s/ep | ETA {remaining/60:.1f}min",
                      flush=True)

            # Auto-guardado periódico para no perder progreso
            if (ep + 1) % SAVE_EVERY == 0:
                self._save_latest()
                print(f"  [checkpoint guardado en ep {ep+1}]", flush=True)

        return {
            'phase':               phase_config['phase'],
            'win_rate':            wins / episodes,
            'avg_resources':       (total_resources_pct / episodes) * 100,
            'avg_collectors_alive': total_collectors_alive / episodes,
        }

    def _save_qtables(self, phase):
        """Guarda las Q-tables al disco, por fase y como 'latest'."""
        os.makedirs(QTABLE_SAVE_PATH, exist_ok=True)

        # Por fase
        with open(f"{QTABLE_SAVE_PATH}collector_qtable_phase{phase}.pkl", 'wb') as f:
            pickle.dump(dict(self.collector_ql.Q_table), f)
        with open(f"{QTABLE_SAVE_PATH}guard_qtable_phase{phase}.pkl", 'wb') as f:
            pickle.dump(dict(self.guard_ql.Q_table), f)

        # Latest (usada por main.py y como checkpoint)
        self._save_latest()

        print(f"  Q-tables guardadas en '{QTABLE_SAVE_PATH}' (fase {phase})", flush=True)
