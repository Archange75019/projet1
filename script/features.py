"""Feature engineering.

Toutes ces fonctions travaillent LIGNE PAR LIGNE : elles n'apprennent rien sur
l'ensemble du jeu de données, elles peuvent donc être appliquées avant le split
train/test sans fuite. Aucune n'utilise de relevé de consommation.
"""
import numpy as np
import pandas as pd

from .config import (AGE_BINS, AGE_LABELS, HIGH_RISE_FLOORS, REF_YEAR, SEATTLE_CENTER,
                     TARGET, USE_FAMILY_RULES)

NEW_FEATURES = [
    "BuildingAge", "NumUseTypes", "LargestUseShare", "SecondUseShare",
    "ParkingShare", "HasParking", "GFAPerBuilding", "GFAPerFloor",
    "IsHighRise", "LogGFABuilding", "LogGFAParking", "DistDowntownKm",
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
    df["GFAPerBuilding"] = df["PropertyGFABuilding(s)"] / df["NumberofBuildings"]
    df["GFAPerFloor"] = (df["PropertyGFABuilding(s)"] / df["NumberofFloors"]).replace([np.inf, -np.inf], np.nan)

    # NaN >= 10 vaut False : on garde NaN pour ne pas classer à tort en "pas high-rise"
    df["IsHighRise"] = (df["NumberofFloors"] >= HIGH_RISE_FLOORS).astype(float)
    df.loc[df["NumberofFloors"].isna(), "IsHighRise"] = np.nan

    df["LogGFABuilding"] = np.log1p(df["PropertyGFABuilding(s)"])
    df["LogGFAParking"] = np.log1p(df["PropertyGFAParking"])
    return df


def add_location_features(df):
    lat0, lon0 = SEATTLE_CENTER
    dlat = (df["Latitude"] - lat0) * 111.2
    dlon = (df["Longitude"] - lon0) * 111.2 * np.cos(np.radians(lat0))
    df["DistDowntownKm"] = np.sqrt(dlat ** 2 + dlon ** 2)
    return df


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Applique tout le feature engineering sur le jeu nettoyé et ajoute LogTarget."""
    df = clean.copy()
    df = add_age_features(df)
    df = add_use_features(df)
    df = add_surface_features(df)
    df = add_location_features(df)
    df["LogTarget"] = np.log10(df[TARGET])
    return df