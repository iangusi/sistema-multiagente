"""
Punto de entrada para entrenar los agentes del Equipo A.

Uso:
    python train.py

Corre ~1800 episodios en 4 fases con dificultad creciente y guarda
las Q-tables en data/. Después ejecuta main.py para jugar con los
agentes entrenados.
"""
import sys
import os

# Asegurarse de que el directorio raíz del proyecto esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.trainer import Trainer
from training.curriculum import CURRICULUM


if __name__ == '__main__':
    print("=" * 60)
    print("ENTRENAMIENTO POR CURRÍCULO — EQUIPO A")
    print("=" * 60)
    print(f"Fases: {len(CURRICULUM)}")
    total_eps = sum(p['episodes'] for p in CURRICULUM)
    print(f"Total de episodios: {total_eps}")
    print()

    trainer = Trainer()
    trainer.train_curriculum(CURRICULUM)

    print("\n" + "=" * 60)
    print("Entrenamiento completado.")
    print("Q-tables guardadas en data/")
    print("Ejecuta  python main.py  para jugar con los agentes entrenados.")
    print("=" * 60)
