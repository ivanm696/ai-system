"""
AI Self-Learning System — точка входа
Запуск: python main.py
"""
import sys
import json
import importlib
import os

# Добавляем корень в путь
sys.path.insert(0, os.path.dirname(__file__))

# Регистрируем все задачи (импорт = авторегистрация)
import tasks.doc_generator
import tasks.file_organizer
import tasks.ml_tasks
import tasks.web_builder

from core.task_registry import TaskRegistry


def run_demo():
    print("\n" + "="*55)
    print("  AI Self-Learning System — Demo")
    print("="*55)

    print("\n📋 Зарегистрированные задачи:")
    for name, doc in TaskRegistry.task_info().items():
        print(f"  • {name}: {doc.strip().splitlines()[0]}")

    # 1. PDF
    print("\n[1/4] Генерация PDF-документа (ISO 9001)...")
    result = TaskRegistry.run_task("generate_pdf", {
        "filename": "ISO_9001_demo",
        "title": "ISO 9001:2015 — Quality Management System",
        "stamp": "УТВЕРЖДЕНО",
        "stamp_color": "blue",
        "sections": [
            {"heading": "1. Область применения",
             "lines": ["Настоящий стандарт применяется ко всем процессам организации.",
                       "Соответствует требованиям ISO 9001:2015."]},
            {"heading": "2. Нормативные ссылки",
             "lines": ["ISO 9000:2015 — Основные положения и словарь.",
                       "ISO 9004:2018 — Управление качеством."]},
            {"heading": "3. Ответственность руководства",
             "lines": ["Руководство обязано обеспечить ресурсы для системы.",
                       "Проводить анализ результативности не реже 1 раза в год."]},
        ]
    })
    print(f"  ✓ PDF создан: {result['path']}")

    # 2. Website
    print("\n[2/4] Генерация HTML-сайта...")
    result = TaskRegistry.run_task("generate_website", {
        "name": "AI Research Portal",
        "filename": "portal_demo",
        "primary_color": "#4A3FAA",
        "sections": [
            {"id": "about", "title": "О проекте",
             "content": "Самообучающаяся нейросеть на продвинутых ML-технологиях."},
            {"id": "features", "title": "Возможности",
             "content": "Генерация документов, анализ молекул, веб-сайты, организация файлов."},
            {"id": "contact", "title": "Контакты",
             "content": "Репозиторий: github.com/ivanm696/solid-giggle"},
        ]
    })
    print(f"  ✓ Сайт создан: {result['path']}")

    # 3. Text ML
    print("\n[3/4] Обучение текстового классификатора...")
    result = TaskRegistry.run_task("train_text_model", {
        "texts": [
            "Машинное обучение — раздел искусственного интеллекта",
            "Нейронные сети используются для распознавания образов",
            "Python — популярный язык для data science",
            "Scikit-learn предоставляет инструменты для ML",
            "Кот сидел на подоконнике и смотрел в окно",
            "Сегодня хорошая погода для прогулки",
            "Я люблю читать книги вечером",
            "Природа красива в любое время года",
        ],
        "labels": ["ml", "ml", "ml", "ml", "other", "other", "other", "other"],
    })
    print(f"  ✓ Обучено. Точность: {result['metrics']['accuracy']*100:.1f}%")

    preds = TaskRegistry.run_task("predict_text", {
        "texts": ["Deep learning — подраздел машинного обучения", "Завтра пойду гулять в парк"]
    })
    print(f"  ✓ Предсказания: {preds['predictions']}")

    # 4. Mol ML
    print("\n[4/4] Обучение молекулярной модели (RDKit)...")
    smiles_train = [
        "CCO", "CC(=O)O", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
        "CN1CCC[C@H]1c2cccnc2", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
        "c1ccc2ccccc2c1", "CCCC", "CC(C)O",
    ]
    targets = [1.2, 0.8, 2.1, 1.5, 3.2, 2.8, 4.1, 2.5, 0.5, 0.9]
    result = TaskRegistry.run_task("train_mol_model", {
        "smiles": smiles_train, "targets": targets
    })
    print(f"  ✓ Обучено. RMSE: {result['metrics']['rmse']}")

    preds = TaskRegistry.run_task("predict_mol", {
        "smiles": ["CCO", "c1ccccc1"]
    })
    print(f"  ✓ Предсказания свойств: {[round(p,3) for p in preds['predictions']]}")

    print("\n" + "="*55)
    print("  ✅ Все задачи выполнены успешно!")
    print("="*55 + "\n")


if __name__ == "__main__":
    run_demo()
