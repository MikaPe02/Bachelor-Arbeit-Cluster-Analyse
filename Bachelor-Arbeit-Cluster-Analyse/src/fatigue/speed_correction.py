# src/fatigue/speed_correction.py
#
# ZWECK: Speed-Bereinigung der Dual-Axis Parameter mittels Residual-Methode.
#
# HINTERGRUND:
#   Laufer laufen auf einem Laufband mit konstanter Geschwindigkeit.
#   DF und SF_norm korrelieren mit der Laufgeschwindigkeit (van Oeveren, 2021;
#   analog zu Patoz et al., 2020). Um Laufstile unabhaengig von der Speed
#   zu vergleichen, werden Speed-Residuen berechnet:
#
#     Residual = beobachteter Wert − durch Speed vorhergesagter Wert
#
#   Die Residuen beschreiben den Laufstil bereinigt um den Speed-Effekt.
#
# VERWENDUNG:
#   df_km1, models = compute_speed_residuals_km1(df_all)
#   save_models(models, config.SPEED_MODELS_PKL)

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"Subject", "km", "speed_ms", "DF", "SF_norm"}


def compute_speed_residuals_km1(
    df_all: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Berechnet speed-bereinigte Residuen bei km 1.0 fuer das Clustering.

    Speed ist KONSTANT pro Laufer (Laufband). Die Funktion fittet zwei
    lineare Regressionen (DF ~ speed_ms, SF_norm ~ speed_ms) auf alle
    Laufer bei km 1.0 und gibt die Residuen zurueck.

    Parameters
    ----------
    df_all : pd.DataFrame
        Langer Datensatz mit Spalten: Subject, km, speed_ms, DF, SF_norm.
        Speed muss fuer alle km eines Probanden identisch sein.

    Returns
    -------
    df_km1 : pd.DataFrame
        Nur km == 1.0, mit zusaetzlichen Spalten:
        - DF_predicted, DF_residual
        - SF_predicted, SF_residual
    models : dict mit Schluesseln:
        'DF', 'SF'            – LinearRegression Objekte
        'R2_DF', 'R2_SF'      – Bestimmtheitsmasse
        'beta0_DF', 'beta1_DF' – Intercept und Slope fuer DF
        'beta0_SF', 'beta1_SF' – Intercept und Slope fuer SF_norm
    """
    # ── Validierung ───────────────────────────────────────────────────────────
    missing = _REQUIRED_COLS - set(df_all.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in df_all: {missing}")

    # Speed sollte pro Proband konstant sein (Laufband)
    speed_std = df_all.groupby("Subject")["speed_ms"].std()
    variable = speed_std[speed_std > 0.01]
    if not variable.empty:
        logger.warning(
            "Speed variiert fuer %d Probanden ueber km-Marken "
            "(erwartet: konstant, Laufband): %s",
            len(variable), variable.index.tolist(),
        )

    # ── Filter auf km 1.0 ────────────────────────────────────────────────────
    df_km1 = df_all[np.abs(df_all["km"] - 1.0) <= 1e-6].copy()

    n = len(df_km1)
    if n < 10:
        logger.warning("Nur %d Laufer bei km 1.0 – Regression wenig aussagekraeftig.", n)

    X    = df_km1[["speed_ms"]].to_numpy(dtype=float)
    y_df = df_km1["DF"].to_numpy(dtype=float)
    y_sf = df_km1["SF_norm"].to_numpy(dtype=float)

    # ── Regression DF ~ speed_ms ──────────────────────────────────────────────
    model_df = LinearRegression()
    model_df.fit(X, y_df)
    df_km1["DF_predicted"] = model_df.predict(X)
    df_km1["DF_residual"]  = df_km1["DF"] - df_km1["DF_predicted"]
    R2_df    = float(model_df.score(X, y_df))
    beta0_df = float(model_df.intercept_)
    beta1_df = float(model_df.coef_[0])

    # ── Regression SF_norm ~ speed_ms ─────────────────────────────────────────
    model_sf = LinearRegression()
    model_sf.fit(X, y_sf)
    df_km1["SF_predicted"] = model_sf.predict(X)
    df_km1["SF_residual"]  = df_km1["SF_norm"] - df_km1["SF_predicted"]
    R2_sf    = float(model_sf.score(X, y_sf))
    beta0_sf = float(model_sf.intercept_)
    beta1_sf = float(model_sf.coef_[0])

    # ── Validierung: Residuen-Mittelwert muss 0 sein ──────────────────────────
    mean_resid_df = float(df_km1["DF_residual"].mean())
    mean_resid_sf = float(df_km1["SF_residual"].mean())
    assert abs(mean_resid_df) < 1e-9, f"DF_residual Mean != 0: {mean_resid_df}"
    assert abs(mean_resid_sf) < 1e-9, f"SF_residual Mean != 0: {mean_resid_sf}"

    # ── Terminal-Output ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SPEED-BEREINIGUNG (km 1.0 Modelle)")
    print(f"{'='*65}")
    print(f"Anzahl Laufer:  {n}")
    print(f"Speed-Bereich:  {df_km1['speed_ms'].min():.2f} - "
          f"{df_km1['speed_ms'].max():.2f} m/s")
    print()
    print(f"DF-Modell:      DF     = {beta0_df:.4f} + {beta1_df:.4f} x speed_ms")
    print(f"                R²     = {R2_df:.3f}")
    print()
    print(f"SF-Modell:      SF_norm = {beta0_sf:.4f} + {beta1_sf:.4f} x speed_ms")
    print(f"                R²      = {R2_sf:.3f}")
    print()
    print("Residuen-Statistik:")
    print(f"  DF_residual:  Mean = {mean_resid_df:.2e},  "
          f"Std = {df_km1['DF_residual'].std():.4f}")
    print(f"  SF_residual:  Mean = {mean_resid_sf:.2e},  "
          f"Std = {df_km1['SF_residual'].std():.4f}")
    print(f"{'='*65}\n")

    models = {
        "DF":       model_df,
        "SF":       model_sf,
        "R2_DF":    R2_df,
        "R2_SF":    R2_sf,
        "beta0_DF": beta0_df,
        "beta1_DF": beta1_df,
        "beta0_SF": beta0_sf,
        "beta1_SF": beta1_sf,
    }

    return df_km1, models


def save_models(models: Dict, output_path) -> None:
    """
    Speichert die Regressionsmodelle als Pickle-Datei.

    Parameters
    ----------
    models      : dict – Ausgabe von compute_speed_residuals_km1()
    output_path : str oder Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(models, f)

    print(f"Modelle gespeichert: {output_path}")
    logger.info("Speed-Modelle gespeichert: %s", output_path)


def load_models(model_path) -> Dict:
    """
    Laedt gespeicherte Regressionsmodelle.

    Parameters
    ----------
    model_path : str oder Path

    Returns
    -------
    dict – identisch mit Ausgabe von compute_speed_residuals_km1()
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Modell-Datei nicht gefunden: {model_path}")

    with open(model_path, "rb") as f:
        models = pickle.load(f)

    logger.info("Speed-Modelle geladen: %s", model_path)
    return models
