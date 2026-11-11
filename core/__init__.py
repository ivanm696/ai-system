"""
Core package for AI Self-Learning System.
Содержит базовую конфигурацию, реестр задач и системные компоненты.
"""

from .config import BASE_DIR, DOWNLOAD_DIR, DIRS, FILE_ROUTING
from .task_registry import BaseTask, TaskRegistry

__all__ = [
    "BASE_DIR",
    "DOWNLOAD_DIR",
    "DIRS",
    "FILE_ROUTING",
    "BaseTask",
    "TaskRegistry",
]
