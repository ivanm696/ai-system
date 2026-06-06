"""Task: organize_files.

Автоматическое распределение файлов по папкам по расширению.
"""

import os
import shutil
from core.config import DIRS, FILE_ROUTING
from core.task_registry import BaseTask, TaskRegistry


@TaskRegistry.register("organize_files")
class FileOrganizerTask(BaseTask):
    """Распределение файлов по папкам: docs / code / data / other."""

    def run(self, input_data: dict) -> dict:
        source_dir = input_data.get("source_dir", "download")
        moved = []
        skipped = []

        if not source_dir or not isinstance(source_dir, str):
            return {"status": "error", "message": "Invalid source directory"}

        if not os.path.isdir(source_dir):
            return {"status": "error", "message": f"Directory not found: {source_dir}"}

        for fname in os.listdir(source_dir):
            src = os.path.join(source_dir, fname)
            if not os.path.isfile(src):
                continue

            ext = os.path.splitext(fname)[1].lower()
            category = FILE_ROUTING.get(ext, "other")
            dest_dir = DIRS[category]
            dest = os.path.join(dest_dir, fname)

            if os.path.exists(dest):
                skipped.append(fname)
                continue

            shutil.move(src, dest)
            moved.append({"file": fname, "category": category})

        return {
            "status": "ok",
            "moved": moved,
            "skipped": skipped,
            "total_moved": len(moved),
        }
