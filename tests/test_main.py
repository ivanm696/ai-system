"""
Tests for main.py module loading logic.

The PR introduced:
- A sys.modules guard (skip if already loaded)
- A try/except ImportError handler around the importlib loading block
"""
import importlib.util
import os
import sys
import types
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAIN_PY_PATH = os.path.join(os.path.dirname(__file__), "..", "main.py")


def _exec_main(extra_globals: Optional[dict] = None) -> dict:
    """Execute main.py in an isolated namespace and return that namespace."""
    with open(_MAIN_PY_PATH) as fh:
        source = fh.read()

    # Provide the minimal globals that main.py needs
    ns: dict = {
        "__name__": "__main__",
        "sys": sys,
        "importlib": importlib,
        "tasks": MagicMock(),  # tasks namespace object
    }
    if extra_globals:
        ns.update(extra_globals)

    exec(compile(source, _MAIN_PY_PATH, "exec"), ns)  # noqa: S102
    return ns


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMainModuleLoadingGuard:
    """Tests for the sys.modules cache guard added in this PR."""

    def test_skips_loading_when_filepath_key_in_sys_modules(self):
        """If 'tasks/doc_generator.py' is already in sys.modules, importlib must
        NOT be called."""
        fake_mod = types.ModuleType("tasks.doc_generator")
        with patch.dict(sys.modules, {"tasks/doc_generator.py": fake_mod}):
            with patch("importlib.util.spec_from_file_location") as mock_spec:
                _exec_main()
                mock_spec.assert_not_called()

    def test_skips_loading_when_module_key_in_sys_modules(self):
        """If 'tasks/doc_generator' is already in sys.modules, importlib must
        NOT be called."""
        fake_mod = types.ModuleType("tasks.doc_generator")
        with patch.dict(sys.modules, {"tasks/doc_generator": fake_mod}):
            with patch("importlib.util.spec_from_file_location") as mock_spec:
                _exec_main()
                mock_spec.assert_not_called()

    def test_loads_module_when_not_in_sys_modules(self):
        """When neither key is present, the importlib loading path executes."""
        # Ensure neither guard key is present
        for key in ("tasks/doc_generator.py", "tasks/doc_generator"):
            sys.modules.pop(key, None)

        mock_spec = MagicMock()
        mock_module = MagicMock()

        with patch("importlib.util.spec_from_file_location", return_value=mock_spec) as mock_sfl, \
             patch("importlib.util.module_from_spec", return_value=mock_module):
            _exec_main()
            mock_sfl.assert_called_once_with("tasks.doc_generator", "tasks/doc_generator.py")

    def test_module_from_spec_called_with_spec(self):
        """module_from_spec must receive the spec returned by spec_from_file_location."""
        for key in ("tasks/doc_generator.py", "tasks/doc_generator"):
            sys.modules.pop(key, None)

        mock_spec = MagicMock()
        mock_module = MagicMock()

        with patch("importlib.util.spec_from_file_location", return_value=mock_spec), \
             patch("importlib.util.module_from_spec", return_value=mock_module) as mock_mfs:
            _exec_main()
            mock_mfs.assert_called_once_with(mock_spec)

    def test_exec_module_called_with_module(self):
        """spec.loader.exec_module must be called with the newly created module."""
        for key in ("tasks/doc_generator.py", "tasks/doc_generator"):
            sys.modules.pop(key, None)

        mock_spec = MagicMock()
        mock_module = MagicMock()

        with patch("importlib.util.spec_from_file_location", return_value=mock_spec), \
             patch("importlib.util.module_from_spec", return_value=mock_module):
            _exec_main()
            mock_spec.loader.exec_module.assert_called_once_with(mock_module)

    def test_tasks_doc_generator_attribute_set(self):
        """After loading, tasks.doc_generator should be set to the new module."""
        for key in ("tasks/doc_generator.py", "tasks/doc_generator"):
            sys.modules.pop(key, None)

        mock_spec = MagicMock()
        mock_module = MagicMock()
        tasks_ns = MagicMock()

        with patch("importlib.util.spec_from_file_location", return_value=mock_spec), \
             patch("importlib.util.module_from_spec", return_value=mock_module):
            _exec_main(extra_globals={"tasks": tasks_ns})

        assert tasks_ns.doc_generator is mock_module


class TestMainImportErrorHandling:
    """Tests for the ImportError handler added in this PR."""

    def _make_bad_spec(self):
        """Return a spec whose loader.exec_module raises ImportError."""
        spec = MagicMock()
        spec.loader.exec_module.side_effect = ImportError("missing dependency")
        return spec

    def test_import_error_is_caught(self):
        """An ImportError during exec_module must not propagate."""
        for key in ("tasks/doc_generator.py", "tasks/doc_generator"):
            sys.modules.pop(key, None)

        bad_spec = self._make_bad_spec()
        mock_module = MagicMock()

        with patch("importlib.util.spec_from_file_location", return_value=bad_spec), \
             patch("importlib.util.module_from_spec", return_value=mock_module), \
             patch("builtins.print"):
            # Should not raise
            _exec_main()

    def test_import_error_message_printed(self):
        """When ImportError occurs the error text must be printed."""
        for key in ("tasks/doc_generator.py", "tasks/doc_generator"):
            sys.modules.pop(key, None)

        bad_spec = self._make_bad_spec()
        mock_module = MagicMock()

        with patch("importlib.util.spec_from_file_location", return_value=bad_spec), \
             patch("importlib.util.module_from_spec", return_value=mock_module), \
             patch("builtins.print") as mock_print:
            _exec_main()

        # The print call must include the error string
        printed_args = " ".join(str(a) for a in mock_print.call_args[0])
        assert "missing dependency" in printed_args

    def test_import_error_message_format(self):
        """The printed message must match 'Error importing module: <err>'."""
        for key in ("tasks/doc_generator.py", "tasks/doc_generator"):
            sys.modules.pop(key, None)

        spec = MagicMock()
        spec.loader.exec_module.side_effect = ImportError("no module named foo")
        mock_module = MagicMock()

        with patch("importlib.util.spec_from_file_location", return_value=spec), \
             patch("importlib.util.module_from_spec", return_value=mock_module), \
             patch("builtins.print") as mock_print:
            _exec_main()

        printed = mock_print.call_args[0][0]
        assert "Error importing module:" in printed
        assert "no module named foo" in printed

    def test_non_import_error_propagates(self):
        """Non-ImportError exceptions (e.g. FileNotFoundError) must still propagate."""
        for key in ("tasks/doc_generator.py", "tasks/doc_generator"):
            sys.modules.pop(key, None)

        spec = MagicMock()
        spec.loader.exec_module.side_effect = FileNotFoundError("file gone")
        mock_module = MagicMock()

        with patch("importlib.util.spec_from_file_location", return_value=spec), \
             patch("importlib.util.module_from_spec", return_value=mock_module):
            with pytest.raises(FileNotFoundError):
                _exec_main()


class TestMainRegressionCases:
    """Regression and boundary tests."""

    def test_both_keys_absent_triggers_load(self):
        """Regression: confirm the guard is an OR condition, not AND.
        If both keys are absent the load must happen."""
        sys.modules.pop("tasks/doc_generator.py", None)
        sys.modules.pop("tasks/doc_generator", None)

        mock_spec = MagicMock()
        mock_module = MagicMock()

        with patch("importlib.util.spec_from_file_location", return_value=mock_spec) as mock_sfl, \
             patch("importlib.util.module_from_spec", return_value=mock_module):
            _exec_main()

        mock_sfl.assert_called_once()

    def test_filepath_key_takes_priority_over_load(self):
        """Boundary: even if 'tasks/doc_generator' is absent, the filepath key
        alone is sufficient to skip the load."""
        fake_mod = types.ModuleType("tasks.doc_generator")
        # Only the .py key is present
        with patch.dict(sys.modules,
                        {"tasks/doc_generator.py": fake_mod},
                        clear=False):
            sys.modules.pop("tasks/doc_generator", None)
            with patch("importlib.util.spec_from_file_location") as mock_sfl:
                _exec_main()
            mock_sfl.assert_not_called()

    def test_module_key_takes_priority_over_load(self):
        """Boundary: the dotted-name key alone is sufficient to skip loading."""
        fake_mod = types.ModuleType("tasks.doc_generator")
        sys.modules.pop("tasks/doc_generator.py", None)
        with patch.dict(sys.modules,
                        {"tasks/doc_generator": fake_mod},
                        clear=False):
            with patch("importlib.util.spec_from_file_location") as mock_sfl:
                _exec_main()
            mock_sfl.assert_not_called()

    def test_spec_from_file_location_receives_correct_module_name(self):
        """The module name argument must be 'tasks.doc_generator' (dotted, not slash)."""
        for key in ("tasks/doc_generator.py", "tasks/doc_generator"):
            sys.modules.pop(key, None)

        mock_spec = MagicMock()
        mock_module = MagicMock()

        with patch("importlib.util.spec_from_file_location", return_value=mock_spec) as mock_sfl, \
             patch("importlib.util.module_from_spec", return_value=mock_module):
            _exec_main()

        args, _ = mock_sfl.call_args
        assert args[0] == "tasks.doc_generator"

    def test_spec_from_file_location_receives_correct_path(self):
        """The file path argument must be 'tasks/doc_generator.py'."""
        for key in ("tasks/doc_generator.py", "tasks/doc_generator"):
            sys.modules.pop(key, None)

        mock_spec = MagicMock()
        mock_module = MagicMock()

        with patch("importlib.util.spec_from_file_location", return_value=mock_spec) as mock_sfl, \
             patch("importlib.util.module_from_spec", return_value=mock_module):
            _exec_main()

        args, _ = mock_sfl.call_args
        assert args[1] == "tasks/doc_generator.py"
