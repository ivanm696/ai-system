"""Regression tests for the ML pipeline and its safe persistence contract.

Run with: python -m pytest tests/test_pipeline.py -v
Imports are deferred so a broken pipeline produces test failures instead of
preventing collection of the entire file.
"""

import importlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest


TEXTS = ["нейронная сеть обучение"] * 5 + ["солнечный день прогулка"] * 5
LABELS = ["наука"] * 5 + ["природа"] * 5
SMILES = ["CCO", "CC", "CCC", "CCCC", "CO", "CCO", "CC", "CCC", "CCCC", "CO"]
TARGETS = [1.0, 2.0, 3.0, 4.0, 5.0] * 2


def make_model(class_name):
    return getattr(importlib.import_module("ml.pipeline"), class_name)()


def trained_text_model():
    model = make_model("TextMLModel")
    model.train(TEXTS, LABELS)
    return model


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    from core.config import DIRS

    monkeypatch.setitem(DIRS, "models", str(tmp_path))
    return tmp_path


class TestPipelineModels:
    @pytest.mark.parametrize("class_name", ["TextMLModel", "MolMLPipeline"])
    def test_models_start_untrained(self, class_name):
        assert make_model(class_name).trained is False

    @pytest.mark.parametrize(
        "class_name, inputs",
        [("TextMLModel", ["machine learning"]), ("MolMLPipeline", ["CCO"])],
    )
    def test_predict_before_training_raises(self, class_name, inputs):
        with pytest.raises(RuntimeError, match="not trained"):
            make_model(class_name).predict(inputs)

    def test_text_training_metrics_and_predictions(self):
        model = make_model("TextMLModel")
        metrics = model.train(TEXTS, LABELS)

        assert model.trained is True
        assert metrics["samples"] == len(TEXTS)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert model.predict([TEXTS[0], TEXTS[-1], TEXTS[0]]) == [
            LABELS[0], LABELS[-1], LABELS[0]
        ]

    def test_text_prediction_with_unseen_vocabulary(self):
        model = trained_text_model()
        predictions = model.predict(["unknown vocabulary", ""])

        assert isinstance(predictions, list)
        assert len(predictions) == 2
        assert set(predictions).issubset(set(LABELS))

    def test_molecular_training_metrics_and_predictions(self):
        model = make_model("MolMLPipeline")
        metrics = model.train(SMILES, TARGETS)
        predictions = model.predict(["CCO", "INVALID_SMILES", "CCO"])

        assert model.trained is True
        assert metrics["samples"] == len(SMILES)
        assert np.isfinite(metrics["rmse"])
        assert metrics["rmse"] >= 0.0
        assert isinstance(predictions, list)
        assert len(predictions) == 3
        assert all(isinstance(value, float) and np.isfinite(value) for value in predictions)
        assert predictions[0] == predictions[2]

    @pytest.mark.parametrize(
        "class_name, inputs, targets",
        [
            ("TextMLModel", TEXTS, LABELS[:-1]),
            ("MolMLPipeline", SMILES, TARGETS[:-1]),
            ("TextMLModel", [], []),
            ("MolMLPipeline", [], []),
        ],
        ids=["text-mismatched-labels", "mol-mismatched-targets", "text-empty", "mol-empty"],
    )
    def test_invalid_training_data_does_not_mark_model_trained(self, class_name, inputs, targets):
        model = make_model(class_name)

        with pytest.raises(ValueError):
            model.train(inputs, targets)

        assert model.trained is False

    def test_molecular_features_preserve_valid_and_invalid_row_positions(self):
        model = make_model("MolMLPipeline")
        features = model._featurize(["CCO", "INVALID_SMILES", "CCO"])

        assert features.shape == (3, 133)
        assert np.issubdtype(features.dtype, np.floating)
        assert np.isfinite(features).all()
        assert np.any(features[0])
        np.testing.assert_array_equal(features[1], np.zeros(133))
        np.testing.assert_array_equal(features[0], features[2])

    def test_molecular_features_for_all_invalid_smiles(self):
        model = make_model("MolMLPipeline")
        np.testing.assert_array_equal(
            model._featurize(["INVALID_SMILES", "also-invalid"]),
            np.zeros((2, 133)),
        )


class TestPipelinePersistence:
    @pytest.mark.parametrize("class_name", ["TextMLModel", "MolMLPipeline"])
    @pytest.mark.parametrize("name", [None, "custom_model"])
    def test_trained_model_save_writes_json_in_models_directory(self, models_dir, class_name, name):
        model = make_model(class_name)
        if class_name == "TextMLModel":
            model.train(TEXTS, LABELS)
            inputs = [TEXTS[0], TEXTS[-1]]
        else:
            model.train(SMILES, TARGETS)
            inputs = ["CCO", "CC"]
        before = model.predict(inputs)

        path = Path(model.save() if name is None else model.save(name))

        assert path.parent.resolve() == models_dir.resolve()
        assert path.is_file()
        # Reading as JSON also catches a regression to binary pickle output.
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert data
        assert model.predict(inputs) == before

    @pytest.mark.parametrize("name", [None, "unicode_classifier"])
    def test_text_model_round_trip_restores_predictions_and_trained_state(self, models_dir, name):
        original = trained_text_model()
        inputs = [TEXTS[0], TEXTS[-1], "unknown words", ""]
        expected = original.predict(inputs)
        if name is None:
            original.save()
        else:
            original.save(name)
        restored = make_model("TextMLModel")

        result = restored.load() if name is None else restored.load(name)

        assert result is restored
        assert restored.trained is True
        assert restored.predict(inputs) == expected

    def test_named_text_models_do_not_overwrite_each_other(self, models_dir):
        first = trained_text_model()
        second = make_model("TextMLModel")
        second.train(TEXTS, list(reversed(LABELS)))
        first_path = Path(first.save("first"))
        first_contents = first_path.read_bytes()
        second_path = Path(second.save("second"))

        assert first_path != second_path
        assert first_path.read_bytes() == first_contents
        for name, original in [("first", first), ("second", second)]:
            restored = make_model("TextMLModel").load(name)
            assert restored.predict([TEXTS[0], TEXTS[-1]]) == original.predict(
                [TEXTS[0], TEXTS[-1]]
            )

    def test_loading_missing_text_model_raises_without_marking_it_trained(self, models_dir):
        model = make_model("TextMLModel")

        with pytest.raises(FileNotFoundError):
            model.load("missing")

        assert model.trained is False
        assert list(models_dir.iterdir()) == []

    @pytest.mark.parametrize("contents", ["", '{"model":', "not JSON"])
    def test_malformed_json_is_rejected_without_marking_model_trained(self, models_dir, contents):
        path = Path(trained_text_model().save("corrupt"))
        path.write_text(contents, encoding="utf-8")
        model = make_model("TextMLModel")

        with pytest.raises(ValueError):
            model.load("corrupt")

        assert model.trained is False
        assert path.read_text(encoding="utf-8") == contents

    def test_pickle_payload_is_rejected_without_execution(self, models_dir):
        marker = models_dir / "payload-executed"

        class Payload:
            def __reduce__(self):
                # If an unsafe loader executes this, its only effect is a local
                # marker in pytest's temporary directory.
                return Path.write_text, (marker, "unsafe deserialization")

        path = Path(trained_text_model().save("untrusted"))
        path.write_bytes(pickle.dumps(Payload(), protocol=4))
        model = make_model("TextMLModel")

        try:
            with pytest.raises(ValueError):
                model.load("untrusted")
        finally:
            assert not marker.exists(), "Loading model data executed a pickle payload"

        assert model.trained is False
