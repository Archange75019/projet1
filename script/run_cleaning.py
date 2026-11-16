"""Orchestration du nettoyage complet.

Usage notebook :
    from script.run_cleaning import clean_dataset, save_datasets
    df, log = clean_dataset(raw_df)
    save_datasets(df, "../data")

Usage terminal (depuis la racine du projet) :
    python -m script.run_cleaning data/2016-building-energy-benchmarking.csv data
"""
import os
import sys

import pandas as pd

from . import clean_columns as cc
from . import clean_rows as cr
from .cleaning_log import CleaningLog
from .config import CLEAN_FILENAME, MODEL_FILENAME, SECOND_TARGET


def clean_dataset(raw: pd.DataFrame, verbose: bool = True):
    log = CleaningLog(verbose=verbose)
    df = raw.copy()
    # 3) Colonnes
    df = cc.fill_missing_uses(df)
    df = cc.fix_largest_use(df, log=log)
    df = cc.flag_gfa_inconsistency(df, log=log)
    df = cc.drop_constant_columns(df)
    df = cc.drop_identifier_columns(df)
    df = cc.harmonize_neighborhood(df)

    #1) Périmètre : non-résidentiel, relevés fiables
    df = cr.drop_multifamily_by_building_type(df, log)
    df = cr.drop_multifamily_by_use(df, log)
    df = cr.keep_compliant_only(df, log)
    df = cr.drop_invalid_target(df, log)

    # 2) Incohérences physiques
    df = cc.flag_invalid_buildings(df, log)
    df = cc.flag_invalid_floors(df, log=log)



    # 4) Outliers : valeur absolue de la cible, puis intensité par type d'usage
    #df = cr.drop_target_outliers(df, log)
    #df = cr.drop_eui_outliers(df, log)

    return df, log


def save_datasets(df: pd.DataFrame, out_dir: str):
    """Écrit deux fichiers :
    - clean : jeu nettoyé complet (avec colonnes de consommation, utile pour l'EDA)
    - model : sans colonnes de fuite (garde TotalGHGEmissions comme 2e cible)"""
    os.makedirs(out_dir, exist_ok=True)
    clean_path = os.path.join(out_dir, CLEAN_FILENAME)
    model_path = os.path.join(out_dir, MODEL_FILENAME)

    df.to_csv(clean_path, index=False)
    model = cc.drop_leakage_columns(df, keep=SECOND_TARGET)
    model.to_csv(model_path, index=False)

    for path, frame in [(clean_path, df), (model_path, model)]:
        check = pd.read_csv(path)
        assert check.shape == frame.shape, f"Relecture incohérente : {path}"
        print(f"Écrit : {path} {frame.shape}")
    return clean_path, model_path


if __name__ == "__main__":
    src, out_dir = sys.argv[1], sys.argv[2]
    clean, log = clean_dataset(pd.read_csv(src))
    print(log.to_frame().to_string(index=False))
    print(log.touched_frame().to_string(index=False))
    print(log.summary())
    save_datasets(clean, out_dir)