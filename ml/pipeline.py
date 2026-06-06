"""ML Module — Self-Learning Pipeline.

Поддерживает: обучение на текстах, молекулярных данных (mol/RDKit), CSV-датасетах.
"""

import os
import json
import pickle  # nosec
import numpy as np
from typing import Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from core.config import DIRS


class TextMLModel:
    """Текстовый классификатор (TF-IDF + RandomForest)."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.trained = False

    def train(self, texts: list, labels: list) -> dict:
        X = self.vectorizer.fit_transform(texts).toarray()
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        self.trained = True
        acc = accuracy_score(y_test, self.model.predict(X_test))
        return {"accuracy": round(acc, 4), "samples": len(texts)}

    def predict(self, texts: list) -> list:
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        X = self.vectorizer.transform(texts).toarray()
        return self.model.predict(X).tolist()

    def save(self, name: str = "text_model"):
        path = os.path.join(DIRS["models"], f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "model": self.model}, f)  # nosec
        return path

    def load(self, name: str = "text_model"):
        path = os.path.join(DIRS["models"], f"{name}.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)  # nosec
        self.vectorizer = data["vectorizer"]
        self.model = data["model"]
        self.trained = True
        return self


class MolMLPipeline:
    """Молекулярный ML — интеграция с mol (RDKit).

    Предсказывает свойства молекул по SMILES-строкам.
    """

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.trained = False

    def _featurize(self, smiles_list: list) -> np.ndarray:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors
        except ImportError:
            raise ImportError("RDKit required: pip install rdkit")

        generator = AllChem.GetMorganGenerator(radius=2, fpSize=128)
        features = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                features.append([0.0] * 133)
                continue
            fp = list(generator.GetFingerprintAsNumPy(mol))
            descs = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
            ]
            features.append(fp + descs)
        return np.array(features, dtype=float)

    def train(self, smiles_list: list, targets: list) -> dict:
        X = self._featurize(smiles_list)
        y = np.array(targets, dtype=float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        self.trained = True
        rmse = float(np.sqrt(mean_squared_error(y_test, self.model.predict(X_test))))
        return {"rmse": round(rmse, 4), "samples": len(smiles_list)}

    def predict(self, smiles_list: list) -> list:
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict(self._featurize(smiles_list)).tolist()

    def save(self, name: str = "mol_model"):
        path = os.path.join(DIRS["models"], f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.model, f)  # nosec
        return path
