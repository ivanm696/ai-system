"""Конфигурация проекта."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "download")

DIRS = {
    "docs": os.path.join(DOWNLOAD_DIR, "docs"),
    "data": os.path.join(DOWNLOAD_DIR, "data"),
    "code": os.path.join(DOWNLOAD_DIR, "code"),
    "other": os.path.join(DOWNLOAD_DIR, "other"),
    "models": os.path.join(DOWNLOAD_DIR, "models"),
}

FILE_ROUTING = {
    ".pdf": "docs", ".docx": "docs", ".doc": "docs", ".txt": "docs",
    ".py": "code", ".js": "code", ".ts": "code", ".java": "code",
    ".csv": "data", ".json": "data", ".xml": "data", ".xlsx": "data",
}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)
