"""
Tasks: train_text_model / predict_text / train_mol_model / predict_mol
ML обучение и инференс через Task Registry.
"""
from core.task_registry import TaskRegistry, BaseTask
from ml.pipeline import TextMLModel, MolMLPipeline

_text_model = TextMLModel()
_mol_model = MolMLPipeline()


@TaskRegistry.register("train_text_model")
class TrainTextTask(BaseTask):
    """Обучение текстового классификатора (TF-IDF + RandomForest)."""

    def run(self, input_data: dict) -> dict:
        texts = input_data["texts"]
        labels = input_data["labels"]
        metrics = _text_model.train(texts, labels)
        path = _text_model.save()
        return {"status": "ok", "metrics": metrics, "model_saved": path}


@TaskRegistry.register("predict_text")
class PredictTextTask(BaseTask):
    """Классификация текстов обученной моделью."""

    def run(self, input_data: dict) -> dict:
        texts = input_data["texts"]
        predictions = _text_model.predict(texts)
        return {"status": "ok", "predictions": predictions}


@TaskRegistry.register("train_mol_model")
class TrainMolTask(BaseTask):
    """Обучение молекулярной модели (SMILES → числовое свойство)."""

    def run(self, input_data: dict) -> dict:
        smiles = input_data["smiles"]
        targets = input_data["targets"]
        metrics = _mol_model.train(smiles, targets)
        path = _mol_model.save()
        return {"status": "ok", "metrics": metrics, "model_saved": path}


@TaskRegistry.register("predict_mol")
class PredictMolTask(BaseTask):
    """Предсказание свойств молекул по SMILES."""

    def run(self, input_data: dict) -> dict:
        smiles = input_data["smiles"]
        preds = _mol_model.predict(smiles)
        return {"status": "ok", "predictions": preds}
