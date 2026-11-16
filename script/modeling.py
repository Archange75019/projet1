"""Modélisation : préprocesseur UNIQUE, imputeur d'étages et fabrique de modèles.

Une seule définition du prétraitement sert partout (comparaison de modèles,
ablation, GridSearch) : les résultats sont donc comparables entre eux.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import HIGH_RISE_FLOORS, OHE_MAX_CATEGORIES

NUMERIC_FEATURES = [
    "LogGFABuilding", "NumberofFloors", "NumberofBuildings", "GFAPerFloor",
    "BuildingAge", "NumUseTypes", "LargestUseShare", "SecondUseShare",
    "ParkingShare", "DistDowntownKm", "HasParking", "IsHighRise",
]
CATEGORICAL_FEATURES = ["BuildingType", "PrimaryPropertyType", "Neighborhood", "LargestUseFamily"]


class FloorsImputer(BaseEstimator, TransformerMixin):
    """NaN d'étages -> médiane de son PrimaryPropertyType (apprise sur le train),
    puis recalcul de GFAPerFloor et IsHighRise."""

    def fit(self, X, y=None):
        self.global_median_ = X["NumberofFloors"].median()
        self.median_by_type_ = X.groupby("PrimaryPropertyType")["NumberofFloors"].median()
        return self

    def transform(self, X):
        X = X.copy()
        fill = X["PrimaryPropertyType"].map(self.median_by_type_).fillna(self.global_median_)
        X["NumberofFloors"] = X["NumberofFloors"].fillna(fill).round().clip(lower=1)
        X["GFAPerFloor"] = np.expm1(X["LogGFABuilding"]) / X["NumberofFloors"]
        X["IsHighRise"] = (X["NumberofFloors"] >= HIGH_RISE_FLOORS).astype(int)
        return X


def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    """LE préprocesseur du projet. Les colonnes non listées sont écartées."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 max_categories=OHE_MAX_CATEGORIES, sparse_output=False)),
    ])
    transformers = [("num", numeric, list(num_cols))]
    if cat_cols:                       # permet l'ablation de TOUTES les catégorielles
        transformers.append(("cat", categorical, list(cat_cols)))
    return ColumnTransformer(transformers)


def make_model(estimator, num_cols=NUMERIC_FEATURES, cat_cols=CATEGORICAL_FEATURES):
    """FloorsImputer -> préprocesseur -> modèle, cible modélisée en log1p."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("floors", FloorsImputer()),
                            ("prep", build_preprocessor(num_cols, cat_cols)),
                            ("model", estimator)]),
        func=np.log1p, inverse_func=np.expm1,
    )