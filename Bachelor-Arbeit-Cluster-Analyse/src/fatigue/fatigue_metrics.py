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