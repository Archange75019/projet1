"""Feature engineering.

Toutes ces fonctions travaillent LIGNE PAR LIGNE : elles n'apprennent rien sur
l'ensemble du jeu de données, elles peuvent donc être appliquées avant le split
train/test sans fuite. Aucune n'utilise de relevé de consommation.
"""
import numpy as np
import pandas as pd

from .config import (AGE_BINS, AGE_LABELS, HIGH_RISE_FLOORS, REF_YEAR,
                     USE_FAMILY_RULES)

NEW_FEATURES = [
    "BuildingAge", "AgeClass", "NumUseTypes", "HasMultipleUses",
    "LargestUseShare", "SecondUseShare", "LargestUseFamily", "ParkingShare",
    "HasParking", "GFAPerBuilding", "GFAPerFloor", "IsHighRise",
    "LogGFABuilding", "LogGFAParking",
]


def use_family(use) -> str:
    """Regroupe un type d'usage détaillé (52 valeurs) en famille métier (≈ 12 valeurs)."""
    if pd.isna(use):
        return "Other"
    s = str(use).lower()
    for family, keywords in USE_FAMILY_RULES:
        if any(k in s for k in keywords):
            return family
    return "Other"


def add_age_features(df):
    df["BuildingAge"] = REF_YEAR - df["YearBuilt"]
    df["AgeClass"] = pd.cut(df["BuildingAge"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    return df


def add_use_features(df):
    df["NumUseTypes"] = df["ListOfAllPropertyUseTypes"].fillna("").apply(
        lambda s: len([u for u in s.split(",") if u.strip()])
    )
    df["HasMultipleUses"] = (df["NumUseTypes"] > 1).astype(int)
    # Dénominateur = max(surface totale, somme des surfaces d'usage) : les parts restent
    # dans [0, 1] sans écrêtage artificiel quand les surfaces déclarées sont incohérentes.
    uses_sum = (df["LargestPropertyUseTypeGFA"].fillna(0)
                + df["SecondLargestPropertyUseTypeGFA"].fillna(0)
                + df["ThirdLargestPropertyUseTypeGFA"].fillna(0))
    denom = np.maximum(df["PropertyGFATotal"], uses_sum)
    df["LargestUseShare"] = df["LargestPropertyUseTypeGFA"] / denom
    df["SecondUseShare"] = df["SecondLargestPropertyUseTypeGFA"] / denom
    df["LargestUseFamily"] = df["LargestPropertyUseType"].map(use_family)
    return df


def add_surface_features(df):
    df["ParkingShare"] = df["PropertyGFAParking"] / df["PropertyGFATotal"]
    df["HasParking"] = (df["PropertyGFAParking"] > 0).astype(int)
    df["GFAPerBuilding"] = (
        df["PropertyGFABuilding(s)"] / df["NumberofBuildings"]
    ).replace([np.inf, -np.inf], np.nan)
    df["GFAPerFloor"] = (df["PropertyGFABuilding(s)"] / df["NumberofFloors"]).replace([np.inf, -np.inf], np.nan)

    # NaN >= 10 vaut False : on garde NaN pour ne pas classer à tort en "pas high-rise"
    df["IsHighRise"] = (df["NumberofFloors"] >= HIGH_RISE_FLOORS).astype(float)
    df.loc[df["NumberofFloors"].isna(), "IsHighRise"] = np.nan

    df["LogGFABuilding"] = np.log1p(df["PropertyGFABuilding(s)"])
    df["LogGFAParking"] = np.log1p(df["PropertyGFAParking"])
    return df


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Applique les transformations structurelles, indépendamment de la cible."""
    df = clean.copy()
    df = add_age_features(df)
    df = add_use_features(df)
    df = add_surface_features(df)
    return df