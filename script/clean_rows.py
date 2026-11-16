"""Nettoyage des LIGNES (suppression de bâtiments).

Chaque fonction prend un DataFrame et un CleaningLog, et renvoie le DataFrame filtré.
"""
import numpy as np
import pandas as pd

from .cleaning_log import CleaningLog
from .config import (EUI_IQR_K, EUI_MIN_GROUP, KEEP_COMPLIANCE, TARGET, TUKEY_COEF)


def _apply(df: pd.DataFrame, keep_mask: pd.Series, label: str, log: CleaningLog) -> pd.DataFrame:
    before = len(df)
    out = df[keep_mask]
    log.step(label, before, len(out))
    return out


def drop_multifamily_by_building_type(df, log):
    mask = df["BuildingType"].str.contains("Multifamily", case=False, na=False)
    return _apply(df, ~mask, "Exclusion Multifamily (BuildingType)", log)


def drop_multifamily_by_use(df, log):
    mask = (
            df["ListOfAllPropertyUseTypes"].str.contains("multifamily", case=False, na=False)
            | df["LargestPropertyUseType"].str.contains("multifamily", case=False, na=False)
    )
    return _apply(df, ~mask, "Exclusion Multifamily (colonnes d'usage)", log)


def keep_compliant_only(df, log):
    """Élimine aussi les Outlier signalés par la ville et les DefaultData."""
    return _apply(df, df["ComplianceStatus"] == KEEP_COMPLIANCE,
                  f"Conservation des seuls {KEEP_COMPLIANCE}", log)


def drop_invalid_target(df, log, target=TARGET):
    mask = df[target].notna() & (df[target] > 0)
    return _apply(df, mask, "Suppression cible manquante ou <= 0", log)


def drop_zero_buildings(df, log):
    return _apply(df, df["NumberofBuildings"] > 0, "Suppression NumberofBuildings = 0", log)


def drop_target_outliers(df, log, target=TARGET, coef=TUKEY_COEF):
    """Méthode de Tukey (IQR) appliquée au log10 de la cible."""
    log_t = np.log10(df[target])
    q1, q3 = log_t.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - coef * iqr, q3 + coef * iqr
    mask_out = (log_t < low) | (log_t > high)
    print(f"Seuils : {10**low:,.0f} kBtu / {10**high:,.0f} kBtu -> {mask_out.sum()} outliers")
    return _apply(df, ~mask_out, "Suppression des outliers (Tukey sur log10 de la cible)", log)


def eui_outlier_mask(df, target=TARGET, k=EUI_IQR_K, min_group=EUI_MIN_GROUP,
                     group_col="PrimaryPropertyType") -> pd.Series:
    """True pour les bâtiments dont l'intensité (cible / surface totale) est aberrante
    PAR RAPPORT À LEUR TYPE D'USAGE : |log10(EUI) - médiane du type| > k x IQR du type."""
    log_eui = np.log10(df[target] / df["PropertyGFATotal"])
    sizes = df[group_col].map(df[group_col].value_counts())
    small = sizes < min_group

    def robust_dist(x, ref):
        iqr = ref.quantile(0.75) - ref.quantile(0.25)
        return (x - ref.median()).abs() / iqr

    out = pd.Series(False, index=df.index)
    for _, idx in df[~small].groupby(group_col).groups.items():
        out.loc[idx] = robust_dist(log_eui.loc[idx], log_eui.loc[idx]) > k
    if small.any():  # petits groupes : on se compare à l'ensemble du jeu
        out.loc[small] = robust_dist(log_eui[small], log_eui) > k
    return out


def drop_eui_outliers(df, log, target=TARGET):
    """Écarte les consommations incohérentes avec la taille du bâtiment (ex. 36 000 sf
    pour 30 M kBtu). Complète Tukey, qui ne détecte que les extrêmes en valeur absolue."""
    mask = eui_outlier_mask(df, target)
    if mask.any():
        cols = ["OSEBuildingID", "PrimaryPropertyType", "PropertyGFATotal", target]
        show = df.loc[mask, [c for c in cols if c in df.columns]].copy()
        show["EUI (kBtu/sf)"] = (show[target] / show["PropertyGFATotal"]).round(1)
        print(show.sort_values("EUI (kBtu/sf)").to_string(index=False))
    return _apply(df, ~mask, "Suppression des outliers d'intensité (par type d'usage)", log)


def audit_rows(df, target=TARGET, max_floors=80, year_min=1800, year_max=2016) -> pd.DataFrame:
    """Diagnostic des incohérences (ne modifie rien)."""
    uses_sum = (df["LargestPropertyUseTypeGFA"].fillna(0)
                + df["SecondLargestPropertyUseTypeGFA"].fillna(0)
                + df["ThirdLargestPropertyUseTypeGFA"].fillna(0))
    checks = {
        "Cible manquante ou <= 0": (df[target].isna() | (df[target] <= 0)).sum(),
        "Surface totale <= 0": (df["PropertyGFATotal"] <= 0).sum(),
        "GFA total != Parking + Bâtiment": (
                df["PropertyGFATotal"] != df["PropertyGFAParking"] + df["PropertyGFABuilding(s)"]
        ).sum(),
        "NumberofFloors == 0": (df["NumberofFloors"] == 0).sum(),
        f"NumberofFloors > {max_floors}": (df["NumberofFloors"] > max_floors).sum(),
        "NumberofBuildings == 0": (df["NumberofBuildings"] == 0).sum(),
        f"YearBuilt hors [{year_min}, {year_max}]": (~df["YearBuilt"].between(year_min, year_max)).sum(),
        "Surface du plus grand usage > 1,2 x surface totale": (
                df["LargestPropertyUseTypeGFA"] > 1.2 * df["PropertyGFATotal"]
        ).sum(),
        "Somme des 3 usages > 1,05 x surface totale": (uses_sum > 1.05 * df["PropertyGFATotal"]).sum(),
        "Plus grand usage manquant": df["LargestPropertyUseType"].isna().sum(),
        "Doublons OSEBuildingID": df["OSEBuildingID"].duplicated().sum(),
        "Doublons probables (lat, lon, surface) : à vérifier": df.duplicated(
            ["Latitude", "Longitude", "PropertyGFATotal"]).sum(),
    }
    return pd.Series(checks, name="nb de lignes").to_frame()