"""
Тесты работоспособности всех модулей системы.
Запуск: python -m pytest tests/ -v
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tasks.doc_generator
import tasks.file_organizer
import tasks.ml_tasks
import tasks.web_builder

from core.task_registry import TaskRegistry
from core.config import DIRS


# ────────────────────────────────────────────────────────────
# Task Registry
# ────────────────────────────────────────────────────────────

class TestTaskRegistry:

    def test_tasks_registered(self):
        expected = {"generate_pdf", "organize_files", "train_text_model",
                    "predict_text", "train_mol_model", "predict_mol",
                    "generate_website"}
        registered = set(TaskRegistry.list_tasks())
        assert expected.issubset(registered), f"Missing: {expected - registered}"

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="not found"):
            TaskRegistry.run_task("nonexistent_task", {})

    def test_task_info_returns_docs(self):
        info = TaskRegistry.task_info()
        assert isinstance(info, dict)
        assert len(info) >= 7


# ────────────────────────────────────────────────────────────
# PDF Generator
# ────────────────────────────────────────────────────────────

class TestDocGenerator:

    def test_basic_pdf(self):
        result = TaskRegistry.run_task("generate_pdf", {
            "filename": "test_basic",
            "title": "Test Document",
            "sections": [
                {"heading": "Section 1", "lines": ["Line one.", "Line two."]}
            ]
        })
        assert result["status"] == "ok"
        assert os.path.exists(result["path"])
        assert result["path"].endswith(".pdf")

    def test_pdf_with_stamp(self):
        result = TaskRegistry.run_task("generate_pdf", {
            "filename": "test_stamp",
            "title": "Stamped Document",
            "stamp": "CONFIDENTIAL",
            "stamp_color": "red",
            "sections": []
        })
        assert result["status"] == "ok"
        assert os.path.getsize(result["path"]) > 1000  # PDF не пустой

    def test_pdf_in_docs_folder(self):
        result = TaskRegistry.run_task("generate_pdf", {
            "filename": "test_location",
            "title": "Location Test",
        })
        assert result["path"].startswith(DIRS["docs"])

    def test_pdf_multi_section(self):
        sections = [
            {"heading": f"Section {i}", "lines": [f"Content {i}"]}
            for i in range(5)
        ]
        result = TaskRegistry.run_task("generate_pdf", {
            "filename": "test_multi",
            "title": "Multi Section",
            "sections": sections
        })
        assert result["status"] == "ok"


# ────────────────────────────────────────────────────────────
# Website Generator
# ────────────────────────────────────────────────────────────

class TestWebsiteGenerator:

    def test_basic_website(self):
        result = TaskRegistry.run_task("generate_website", {
            "name": "Test Site",
            "filename": "test_site",
            "sections": [
                {"id": "home", "title": "Home", "content": "Welcome!"}
            ]
        })
        assert result["status"] == "ok"
        assert os.path.exists(result["path"])
        assert result["path"].endswith(".html")

    def test_html_content(self):
        result = TaskRegistry.run_task("generate_website", {
            "name": "Content Test",
            "filename": "test_content",
            "sections": [
                {"id": "about", "title": "About", "content": "This is about page."}
            ]
        })
        with open(result["path"], encoding="utf-8") as f:
            html = f.read()
        assert "Content Test" in html
        assert "About" in html
        assert "This is about page." in html
        assert "<!DOCTYPE html>" in html

    def test_custom_color(self):
        result = TaskRegistry.run_task("generate_website", {
            "name": "Colored Site",
            "filename": "test_color",
            "primary_color": "#FF5500",
            "sections": []
        })
        with open(result["path"], encoding="utf-8") as f:
            html = f.read()
        assert "#FF5500" in html


# ────────────────────────────────────────────────────────────
# File Organizer
# ────────────────────────────────────────────────────────────

class TestFileOrganizer:

    def test_invalid_dir(self):
        result = TaskRegistry.run_task("organize_files", {
            "source_dir": "/nonexistent/path"
        })
        assert result["status"] == "error"

    def test_organize_files(self, tmp_path):
        import uuid
        uid = uuid.uuid4().hex[:8]
        # Создаём тестовые файлы с уникальными именами
        (tmp_path / f"report_{uid}.pdf").write_text("pdf")
        (tmp_path / f"script_{uid}.py").write_text("py")
        (tmp_path / f"data_{uid}.csv").write_text("csv")
        (tmp_path / f"notes_{uid}.txt").write_text("txt")

        result = TaskRegistry.run_task("organize_files", {
            "source_dir": str(tmp_path)
        })
        assert result["status"] == "ok"
        assert result["total_moved"] == 4

        categories = {m["category"] for m in result["moved"]}
        assert "docs" in categories
        assert "code" in categories
        assert "data" in categories


# ────────────────────────────────────────────────────────────
# Text ML
# ────────────────────────────────────────────────────────────

class TestTextML:

    TEXTS = [
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
    LABELS = ["ml","ml","ml","ml","other","other","other","other","ml","ml"]

    def test_train_and_predict(self):
        result = TaskRegistry.run_task("train_text_model", {
            "texts": self.TEXTS,
            "labels": self.LABELS
        })
        assert result["status"] == "ok"
        assert "accuracy" in result["metrics"]
        assert 0.0 <= result["metrics"]["accuracy"] <= 1.0

    def test_predict_returns_list(self):
        TaskRegistry.run_task("train_text_model", {
            "texts": self.TEXTS, "labels": self.LABELS
        })
        result = TaskRegistry.run_task("predict_text", {
            "texts": ["Deep learning is amazing", "I love sunny weather"]
        })
        assert result["status"] == "ok"
        assert isinstance(result["predictions"], list)
        assert len(result["predictions"]) == 2

    def test_model_saved(self):
        result = TaskRegistry.run_task("train_text_model", {
            "texts": self.TEXTS, "labels": self.LABELS
        })
        assert os.path.exists(result["model_saved"])


# ────────────────────────────────────────────────────────────
# Mol ML (RDKit)
# ────────────────────────────────────────────────────────────

class TestMolML:

    SMILES = [
        "CCO", "CC(=O)O", "c1ccccc1", "CCCC", "CC(C)O",
        "CN1CCC[C@H]1c2cccnc2", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "c1ccc2ccccc2c1", "CC(=O)Oc1ccccc1C(=O)O", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    ]
    TARGETS = [1.2, 0.8, 2.1, 0.5, 0.9, 3.2, 2.8, 2.5, 1.5, 4.1]

    def test_train_mol_model(self):
        result = TaskRegistry.run_task("train_mol_model", {
            "smiles": self.SMILES, "targets": self.TARGETS
        })
        assert result["status"] == "ok"
        assert "rmse" in result["metrics"]
        assert result["metrics"]["rmse"] >= 0

    def test_predict_mol(self):
        TaskRegistry.run_task("train_mol_model", {
            "smiles": self.SMILES, "targets": self.TARGETS
        })
        result = TaskRegistry.run_task("predict_mol", {
            "smiles": ["CCO", "c1ccccc1", "CCCC"]
        })
        assert result["status"] == "ok"
        assert len(result["predictions"]) == 3
        for p in result["predictions"]:
            assert isinstance(p, float)

    def test_invalid_smiles_handled(self):
        TaskRegistry.run_task("train_mol_model", {
            "smiles": self.SMILES, "targets": self.TARGETS
        })
        # Невалидная SMILES не должна крашить систему
        result = TaskRegistry.run_task("predict_mol", {
            "smiles": ["INVALID_SMILES", "CCO"]
        })
        assert result["status"] == "ok"
        assert len(result["predictions"]) == 2
