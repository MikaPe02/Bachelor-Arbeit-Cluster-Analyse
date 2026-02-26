import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def compute_pca(df_features_z: pd.DataFrame, n_components: int = 2):
    """
    Compute PCA on z-transformed features.

    Parameters
    ----------
    df_features_z : DataFrame
        Must contain 'Subject' + numeric feature columns.
    n_components : int
        Number of principal components to compute.

    Returns
    -------
    df_pca : DataFrame
        Columns: Subject, PC1..PCn
    explained_variance_ratio : np.ndarray
        Fraction of variance explained by each PC.
    loadings : DataFrame
        PCA loadings (feature contributions) for interpretation.
    """
    subjects = df_features_z["Subject"].astype(str).values
    X = df_features_z.drop(columns=["Subject"]).to_numpy(dtype=float)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    # PC score table
    cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=cols)
    df_pca.insert(0, "Subject", subjects)

    # Loadings: how features contribute to PCs
    feature_names = df_features_z.drop(columns=["Subject"]).columns.tolist()
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=cols
    )

    return df_pca, pca.explained_variance_ratio_, loadings