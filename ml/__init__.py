"""
Пакет машинного обучения (ML) для AI Self-Learning System.
Отвечает за работу с моделями, молекулярным анализом (.mol) и предсказаниями.
"""

import sys
import logging

# Настройка логирования для пакета ml
logger = logging.getLogger("ai_system.ml")
logger.setLevel(logging.INFO)

# Проверяем доступность ключевых библиотек машинного обучения (например, RDKit)
try:
    import rdkit
    RDKIT_AVAILABLE = True
    logger.info("RDKit успешно обнаружен, молекулярный ML активен.")
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("Внимание: RDKit не установлен. Функции молекулярного ML могут быть ограничены.")

# Экспортируем основные флаги и компоненты для удобного импорта в проекте
__all__ = ["RDKIT_AVAILABLE"]

# Здесь можно автоматически подтягивать модули обучения и предсказаний, если они созданы:
# from .train_mol import TrainMolModel
# from .predict_mol import PredictMolModel
