# DOCUMENTACIÓN DEL PROYECTO — ÍNDICE

Referencia técnica rápida para sesiones futuras. Leer este índice primero.

## Archivos de documentación

| Archivo | Qué cubre |
|---|---|
| [01_PROJECT_OVERVIEW.md](01_PROJECT_OVERVIEW.md) | Objetivo del juego, dos equipos, flujo general, cómo ejecutar |
| [02_FILE_STRUCTURE.md](02_FILE_STRUCTURE.md) | Árbol de archivos completo con rol de cada uno |
| [03_ENVIRONMENT.md](03_ENVIRONMENT.md) | Motor del juego: grid, tick, combate, riesgo, build kits |
| [04_AGENTS_COLLECTOR.md](04_AGENTS_COLLECTOR.md) | Recolector: estado RL 10-vars, heurísticas, acciones, recompensas |
| [05_AGENTS_GUARD.md](05_AGENTS_GUARD.md) | Guardia: estado RL 12-vars, acción SCOUT, heurísticas autónomas |
| [06_AGENTS_HUNTER_TOWER.md](06_AGENTS_HUNTER_TOWER.md) | Cazador (FSM + genes) y Torre (autónoma) — Equipo B |
| [07_RL_SYSTEM.md](07_RL_SYSTEM.md) | Q-learning tabular, estados, biases, actualización, save/load |
| [08_TRAINING_CURRICULUM.md](08_TRAINING_CURRICULUM.md) | Sistema de entrenamiento headless por currículo |
| [09_CONSTANTS_REFERENCE.md](09_CONSTANTS_REFERENCE.md) | Todas las constantes agrupadas por propósito |
| [10_INTEGRATION_MAP.md](10_INTEGRATION_MAP.md) | Quién llama a quién, flujo de datos entre módulos |

## Guía de modificaciones frecuentes

| Tarea | Archivos a tocar |
|---|---|
| Cambiar recompensas RL | `utils/constants.py` → sección RECOMPENSAS |
| Cambiar heurísticas de un agente | `agents/collector.py:calculate_heuristic_biases()` o `agents/guard.py:calculate_heuristic_biases()` |
| Agregar nueva acción a un agente | `agents/guard.py` (ACTIONS list + execute_X() + decide()) + `environment.py:_GUARD_ACTIONS` |
| Cambiar número de cazadores | `utils/constants.py:NUM_HUNTERS` o pasar `num_hunters_override` al Environment |
| Cambiar parámetros de entrenamiento | `utils/constants.py` → sección ENTRENAMIENTO POR CURRÍCULO |
| Agregar nueva variable al estado RL | `agents/collector.py:build_state()` + `calculate_heuristic_biases()` (desempaquetar nueva var) |
| Cambiar reglas de build kits | `environment.py:_check_build_kits()` |
| Cambiar velocidad/HP de agentes | `utils/constants.py` → sección del agente correspondiente |
