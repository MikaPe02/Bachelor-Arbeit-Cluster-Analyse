import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def compute_pca(
    df_features_z: pd.DataFrame,
    n_components: int = 2,
    feature_cols: list[str] | None = None,
):
    """
    Compute PCA on selected z-transformed features.

    Parameters
    ----------
    df_features_z : DataFrame
        Must contain 'Subject' + numeric feature columns.
    n_components : int
        Number of principal components.
    feature_cols : list[str] or None
        If provided, PCA is computed only on these columns.

    Returns
    -------
    df_pca : DataFrame
        Columns: Subject, PC1..PCn
    explained_variance_ratio : np.ndarray
    loadings : DataFrame
        Feature loadings for interpretation.
    """
    subjects = df_features_z["Subject"].astype(str).values

    if feature_cols is None:
        feature_cols = [c for c in df_features_z.columns if c != "Subject"]

    # Safety check: all requested cols exist
    missing = [c for c in feature_cols if c not in df_features_z.columns]
    if missing:
        raise ValueError(f"Missing columns in df_features_z: {missing}")

    X = df_features_z[feature_cols].to_numpy(dtype=float)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=cols)
    df_pca.insert(0, "Subject", subjects)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=cols
    )

    return df_pca, pca.explained_variance_ratio_, loadings

