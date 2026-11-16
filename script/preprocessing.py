"""Préparation pour la modélisation.

Tout ce qui est APPRIS sur les données (médianes d'imputation, moyenne/écart-type,
catégories d'encodage) est enveloppé dans un Pipeline sklearn : il doit être
`fit` sur le train UNIQUEMENT, puis seulement `transform` sur le test.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .clean_columns import drop_leakage_columns
from .config import NON_FEATURE_COLS, OHE_MAX_CATEGORIES, SECOND_TARGET, TARGET


def split_xy(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Sépare X / y (y = log de la cible) puis train / test."""
    y = df["LogTarget"]
    X = drop_leakage_columns(df).drop(columns=[TARGET, "LogTarget"] + NON_FEATURE_COLS,
                                      errors="ignore")
    X = X.drop(columns=[c for c in SECOND_TARGET if c in X.columns])  # 2e cible, jamais une feature
    X["CouncilDistrictCode"] = X["CouncilDistrictCode"].astype(str)  # code, pas une quantité
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 max_categories=OHE_MAX_CATEGORIES, sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])