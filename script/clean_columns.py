"""Nettoyage des COLONNES (valeurs manquantes, colonnes inutiles, harmonisation).

Aucune de ces fonctions ne supprime de ligne.
"""
import numpy as np
import pandas as pd

from .config import (FILL_NONE_COLS, FILL_ZERO_COLS, GFA_SUM_TOL, ID_COLS, LEAKAGE_COLS,
                     MAX_FLOORS, NEIGHBORHOOD_ALIASES, SECOND_TARGET)


def flag_invalid_floors(df, max_floors=MAX_FLOORS, log=None):
    """0 ou > max_floors = valeur manquante déguisée : NaN + indicateur FloorsMissing.
    L'imputation se fera plus tard, sur le jeu d'entraînement uniquement."""
    df = df.copy()
    bad = (df["NumberofFloors"] == 0) | (df["NumberofFloors"] > max_floors)
    df["FloorsMissing"] = bad.astype(int)
    df.loc[bad, "NumberofFloors"] = np.nan
    if log is not None:
        log.touch("NumberofFloors invalide -> NaN + FloorsMissing", bad.sum())
    else:
        print(f"NumberofFloors invalides passés en NaN : {bad.sum()} (lignes conservées : {len(df)})")
    return df


def fill_missing_uses(df):
    """NaN dans le 2e / 3e usage = 'aucun', pas 'inconnu'."""
    df = df.copy()
    for col in FILL_NONE_COLS:
        df[col] = df[col].fillna("None")
    for col in FILL_ZERO_COLS:
        df[col] = df[col].fillna(0.0)
    return df


def fix_largest_use(df, log=None):
    """Impute le plus grand usage quand il manque (à appeler APRÈS fill_missing_uses).

    Usage : usage unique de ListOfAllPropertyUseTypes, sinon le PrimaryPropertyType s'il
    figure dans la liste, sinon "Other". Surface : surface totale moins 2e et 3e usages
    (surface totale si le résultat est <= 0)."""
    df = df.copy()
    miss = df["LargestPropertyUseType"].isna()

    def pick(row):
        uses = [u.strip() for u in str(row["ListOfAllPropertyUseTypes"]).split(",") if u.strip()]
        if len(uses) == 1:
            return uses[0]
        if row["PrimaryPropertyType"] in uses:
            return row["PrimaryPropertyType"]
        return "Other"

    if miss.any():
        df.loc[miss, "LargestPropertyUseType"] = df[miss].apply(pick, axis=1)

    miss_gfa = df["LargestPropertyUseTypeGFA"].isna()
    if miss_gfa.any():
        rest = (df["PropertyGFATotal"] - df["SecondLargestPropertyUseTypeGFA"]
                - df["ThirdLargestPropertyUseTypeGFA"])
        df.loc[miss_gfa, "LargestPropertyUseTypeGFA"] = rest.where(rest > 0, df["PropertyGFATotal"])[miss_gfa]
    if log is not None:
        log.touch("Plus grand usage imputé (usage + surface)", (miss | miss_gfa).sum())
    return df


def flag_gfa_inconsistency(df, tol=GFA_SUM_TOL, log=None):
    """GFAInconsistent = 1 si la somme des surfaces des 3 usages dépasse tol x surface totale.
    Les lignes sont conservées : la feature 'part d'usage' est corrigée dans features.py."""
    df = df.copy()
    uses_sum = (df["LargestPropertyUseTypeGFA"].fillna(0)
                + df["SecondLargestPropertyUseTypeGFA"] + df["ThirdLargestPropertyUseTypeGFA"])
    df["GFAInconsistent"] = (uses_sum > tol * df["PropertyGFATotal"]).astype(int)
    if log is not None:
        log.touch("Surfaces d'usage incohérentes -> GFAInconsistent", df["GFAInconsistent"].sum())
    return df


def drop_constant_columns(df):
    """Colonnes 100 % vides, constantes, ou devenues constantes après filtrage."""
    useless = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    print("Colonnes constantes / vides supprimées :", useless)
    return df.drop(columns=useless)


def drop_identifier_columns(df):
    return df.drop(columns=[c for c in ID_COLS if c in df.columns])


def harmonize_neighborhood(df):
    df = df.copy()
    df["Neighborhood"] = (
        df["Neighborhood"].str.strip().str.upper().replace(NEIGHBORHOOD_ALIASES)
    )
    return df


def drop_leakage_columns(df, keep=None):
    """Retire les colonnes liées à la consommation. `keep` : colonnes à conserver
    malgré tout (ex. SECOND_TARGET pour l'extension double cible)."""
    keep = set(keep or [])
    return df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns and c not in keep])