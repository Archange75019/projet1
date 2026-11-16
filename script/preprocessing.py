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
from .cleaning_log import CleaningLog
from .config import OHE_MAX_CATEGORIES, TARGET
from .modeling import CATEGORICAL_FEATURES, NUMERIC_FEATURES

EXTREME_COLS = ["NumberofBuildings", "GFAPerFloor", "NumUseTypes"]

def drop_feature_extremes(df: pd.DataFrame, log: CleaningLog | None = None,
                          cols=EXTREME_COLS, quantile=0.995) -> pd.DataFrame:
    """Retire le 0,5 % supérieur de quelques variables très asymétriques.
    Seuils calculés sur tout le jeu (avant split) : limite à mentionner."""
    mask = pd.Series(False, index=df.index)
    pct = f"{quantile * 100:.1f}".replace(".", ",")
    for c in cols:
        seuil = df[c].quantile(quantile)
        n_col = (df[c] > seuil).sum()
        print(f"{c} : seuil {pct} % = {seuil:,.1f} -> {n_col} bâtiments au-dessus")
        mask |= df[c] > seuil
    out = df[~mask]
    if log is not None:
        log.step("Suppression des valeurs extrêmes de features (quantile 99,5 %)", len(df), len(out))
    return out


def split_xy(df: pd.DataFrame, test_size=0.3, random_state=42, target=TARGET):
    """X = variables retenues (modeling.py), y = cible choisie (log appliqué dans le modèle)."""
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Variables absentes (appeler build_features avant) : {missing}")
    if target not in df.columns:
        raise KeyError(f"Cible absente : {target}")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    print(f"Numériques ({len(num_cols)}) : {num_cols}")
    print(f"Catégorielles ({len(cat_cols)}) : {cat_cols}")

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