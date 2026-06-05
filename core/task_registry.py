"""
Plugin Task Registry
Регистрируй новые задачи через декоратор @TaskRegistry.register("name")
"""
from abc import ABC, abstractmethod
from typing import Dict, Type, List
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """Базовый класс. Наследуй и реализуй метод run()."""

    @abstractmethod
    def run(self, input_data: dict) -> dict:
        pass

    @property
    def description(self) -> str:
        return self.__doc__ or "No description"


class TaskRegistry:
    _tasks: Dict[str, Type[BaseTask]] = {}

    @classmethod
    def register(cls, name: str):
        """Декоратор для регистрации новой задачи."""
        def decorator(task_cls: Type[BaseTask]):
            cls._tasks[name] = task_cls
            logger.info(f"Task registered: '{name}'")
            return task_cls
        return decorator

    @classmethod
    def run_task(cls, name: str, input_data: dict) -> dict:
        if name not in cls._tasks:
            available = list(cls._tasks.keys())
            raise ValueError(f"Task '{name}' not found. Available: {available}")
        logger.info(f"Running task: '{name}'")
        try:
            result = cls._tasks[name]().run(input_data)
            logger.info(f"Task '{name}' completed successfully")
            return result
        except Exception as e:
            logger.error(f"Task '{name}' failed: {e}")
            raise

    @classmethod
    def list_tasks(cls) -> List[str]:
        return list(cls._tasks.keys())

    @classmethod
    def task_info(cls) -> dict:
        return {name: cls._tasks[name].__doc__ or "—" for name in cls._tasks}
