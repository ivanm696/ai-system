# AI Self-Learning System

Самообучающаяся система с расширяемой архитектурой плагинов.

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый старт

```bash
python main.py          # демо всех задач
pytest tests/ -v        # тесты работоспособности
```

## Задачи (Tasks)

| Задача | Описание |
|---|---|
| `generate_pdf` | PDF-документы с разделами и штампом |
| `generate_website` | HTML-сайты по структуре |
| `organize_files` | Распределение файлов по папкам |
| `train_text_model` | Обучение текстового классификатора |
| `predict_text` | Предсказание класса текста |
| `train_mol_model` | Молекулярный ML (RDKit/SMILES) |
| `predict_mol` | Предсказание свойств молекул |

## Добавить новую задачу

```python
# tasks/my_task.py
from core.task_registry import TaskRegistry, BaseTask

@TaskRegistry.register("my_task")
class MyTask(BaseTask):
    """Описание задачи."""
    def run(self, input_data: dict) -> dict:
        # логика
        return {"status": "ok", "result": ...}
```

Импортируй в `main.py` — и задача доступна:
```python
import tasks.my_task
TaskRegistry.run_task("my_task", {...})
```

## Структура папок

```
download/
├── docs/     # PDF, HTML, DOCX
├── code/     # Python, JS, ...
├── data/     # CSV, JSON, ...
├── models/   # Сохранённые ML-модели
└── other/    # Остальное
```
