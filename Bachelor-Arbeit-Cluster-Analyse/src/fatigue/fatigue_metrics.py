import numpy as np
import pandas as pd


def compute_delta(series):
    """
    Difference between last and first value.
    """

    return float(series.iloc[-1] - series.iloc[0])


def compute_slope(x, y):
    """
    Linear regression slope.

    x = km
    y = DF or SF_norm
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    slope, intercept = np.polyfit(x, y, 1)

    return float(slope)

def compute_subject_fatigue(df_subject):
    """
    Compute fatigue metrics for one subject.
    """

    df_subject = df_subject.sort_values("km")

    DF_start = float(df_subject["DF"].iloc[0])
    DF_end = float(df_subject["DF"].iloc[-1])

    SF_start = float(df_subject["SF_norm"].iloc[0])
    SF_end = float(df_subject["SF_norm"].iloc[-1])

    return dict(

        DF_start=DF_start,
        DF_end=DF_end,
        Delta_DF=compute_delta(df_subject["DF"]),
        Slope_DF=compute_slope(df_subject["km"], df_subject["DF"]),

        SF_start=SF_start,
        SF_end=SF_end,
        Delta_SF=compute_delta(df_subject["SF_norm"]),
        Slope_SF=compute_slope(df_subject["km"], df_subject["SF_norm"])

    )


def build_fatigue_feature_table(df_dual_axis: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per subject with fatigue features.

    Input columns required:
    - Subject, km, DF, SF_norm

    Output:
    - DF_start, DF_end, Delta_DF, Slope_DF
    - SF_start, SF_end, Delta_SF, Slope_SF
    """
    subjects = df_dual_axis["Subject"].unique()
    rows = []

    for s in subjects:
        df_s = df_dual_axis[df_dual_axis["Subject"] == s]
        metrics = compute_subject_fatigue(df_s)
        row = {"Subject": s}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)