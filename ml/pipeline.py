"""Machine-learning pipelines used by the registered ML tasks."""
from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

from core.config import DIRS


class TextMLModel:
    """TF-IDF text classifier backed by a random forest."""

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.model: RandomForestClassifier | None = None

    def train(self, texts: list[str], labels: list[Any]) -> dict[str, float]:
        if not texts or len(texts) != len(labels):
            raise ValueError("texts and labels must be non-empty and have the same length")

        self.vectorizer = TfidfVectorizer()
        features = self.vectorizer.fit_transform(texts)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(features, labels)
        return {"accuracy": float(self.model.score(features, labels))}

    def predict(self, texts: list[str]) -> list[Any]:
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("The text model has not been trained")
        if not texts:
            return []
        return self.model.predict(self.vectorizer.transform(texts)).tolist()

    def save(self, path: str | None = None) -> str:
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("Cannot save an untrained text model")
        if path is None:
            path = os.path.join(DIRS["models"], "text_model.joblib")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model}, path)
        return path

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.vectorizer = data["vectorizer"]
        self.model = data["model"]


class MolMLPipeline:
    """Random-forest regression pipeline using RDKit molecular descriptors."""

    _DESCRIPTORS = (
        Descriptors.MolWt,
        Descriptors.MolLogP,
        Descriptors.NumHDonors,
        Descriptors.NumHAcceptors,
        Descriptors.TPSA,
        Descriptors.NumRotatableBonds,
        Descriptors.RingCount,
        Descriptors.HeavyAtomCount,
    )

    def __init__(self) -> None:
        self.model: RandomForestRegressor | None = None

    @classmethod
    def _features(cls, smiles: list[str]) -> np.ndarray:
        rows = []
        for value in smiles:
            molecule = Chem.MolFromSmiles(value) if isinstance(value, str) else None
            if molecule is None:
                rows.append([0.0] * len(cls._DESCRIPTORS))
                continue
            rows.append([float(descriptor(molecule)) for descriptor in cls._DESCRIPTORS])
        return np.asarray(rows, dtype=float)

    def train(self, smiles: list[str], targets: list[float]) -> dict[str, float]:
        if not smiles or len(smiles) != len(targets):
            raise ValueError("smiles and targets must be non-empty and have the same length")

        features = self._features(smiles)
        values = np.asarray(targets, dtype=float)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(features, values)
        predictions = self.model.predict(features)
        rmse = float(np.sqrt(mean_squared_error(values, predictions)))
        return {"rmse": rmse}

    def predict(self, smiles: list[str]) -> list[float]:
        if self.model is None:
            raise RuntimeError("The molecular model has not been trained")
        if not smiles:
            return []
        return [float(value) for value in self.model.predict(self._features(smiles))]

    def save(self, path: str | None = None) -> str:
        if self.model is None:
            raise RuntimeError("Cannot save an untrained molecular model")
        if path is None:
            path = os.path.join(DIRS["models"], "mol_model.joblib")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"model": self.model}, path)
        return path

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
