"""
AI Self-Learning System — демонстрационный запуск всех задач.

Запуск:
    python main.py
"""
import sys

import tasks.doc_generator
import tasks.file_organizer
import tasks.ml_tasks
import tasks.web_builder

from core.task_registry import TaskRegistry


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def demo_pdf() -> None:
    banner("generate_pdf — генерация PDF-документа")
    result = TaskRegistry.run_task("generate_pdf", {
        "filename": "demo_report",
        "title": "Демонстрационный отчёт",
        "sections": [
            {
                "heading": "Введение",
                "lines": [
                    "Этот PDF создан автоматически задачей generate_pdf.",
                    "Поддерживаются разделы, заголовки и штамп.",
                ],
            },
            {
                "heading": "Итоги",
                "lines": ["Система работает корректно."],
            },
        ],
        "stamp": "DEMO",
        "stamp_color": "red",
    })
    print(result)


def demo_website() -> None:
    banner("generate_website — генерация HTML-сайта")
    result = TaskRegistry.run_task("generate_website", {
        "name": "Demo Site",
        "filename": "demo_site",
        "primary_color": "#4A3FAA",
        "sections": [
            {"id": "home", "title": "Главная", "content": "Добро пожаловать!"},
            {"id": "about", "title": "О системе", "content": "Автоматически сгенерированный сайт."},
        ],
    })
    print(result)


def demo_file_organizer() -> None:
    banner("organize_files — распределение файлов по папкам")
    result = TaskRegistry.run_task("organize_files", {"source_dir": "download"})
    print(result)


def demo_text_ml() -> None:
    banner("train_text_model / predict_text — текстовый классификатор")
    texts = [
        "Machine learning is a subset of AI",
        "Neural networks process data in layers",
        "Python is great for data science",
        "Scikit-learn provides ML tools",
        "The cat sat on the mat",
        "Today is a beautiful sunny day",
        "I enjoy reading books in the evening",
        "Nature is beautiful in every season",
        "Deep learning uses convolutional networks",
        "Random forests are ensemble methods",
    ]
    labels = ["ml", "ml", "ml", "ml", "other", "other", "other", "other", "ml", "ml"]

    train_result = TaskRegistry.run_task("train_text_model", {"texts": texts, "labels": labels})
    print("train:", train_result)

    predict_result = TaskRegistry.run_task("predict_text", {
        "texts": ["Deep learning is amazing", "I love sunny weather"],
    })
    print("predict:", predict_result)


def demo_mol_ml() -> None:
    banner("train_mol_model / predict_mol — молекулярная модель (RDKit)")
    smiles = [
        "CCO", "CC(=O)O", "c1ccccc1", "CCCC", "CC(C)O",
        "CN1CCC[C@H]1c2cccnc2", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "c1ccc2ccccc2c1", "CC(=O)Oc1ccccc1C(=O)O",
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    ]
    targets = [1.2, 0.8, 2.1, 0.5, 0.9, 3.2, 2.8, 2.5, 1.5, 4.1]

    train_result = TaskRegistry.run_task("train_mol_model", {"smiles": smiles, "targets": targets})
    print("train:", train_result)

    predict_result = TaskRegistry.run_task("predict_mol", {
        "smiles": ["CCO", "c1ccccc1", "CCCC"],
    })
    print("predict:", predict_result)


def main() -> int:
    banner("Зарегистрированные задачи")
    for name, doc in TaskRegistry.task_info().items():
        print(f"- {name}: {doc.strip().splitlines()[0] if doc else '—'}")

    demo_pdf()
    demo_website()
    demo_file_organizer()
    demo_text_ml()
    demo_mol_ml()

    banner("Готово. Все задачи выполнены успешно.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
