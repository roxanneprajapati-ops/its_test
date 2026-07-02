"""
clustering.py
--------------
Optional K-Means clustering (rubric item C, "optional"). Group traffic
record into K traffic "state" using unsupervised learning, useful for
exploratory analysis - e.g. discover a natural "free-flow / build-up /
congested" state grouping without using the hand-made label.
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel
from src.utils import config

CLUSTER_FEATURES = ["Vehicle_Count", "Avg_Speed", "Queue_Length", "Occupancy_Pct"]


class TrafficStateClusterer(BaseModel):
    """Unsupervised grouping of traffic record into K cluster."""

    def __init__(self, n_clusters: int = config.KMEANS_N_CLUSTERS):
        super().__init__("traffic_state_kmeans")
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=config.ML_RANDOM_STATE, n_init=10)

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        X = self.scaler.fit_transform(df[CLUSTER_FEATURES])
        labels = self.model.fit_predict(X)
        self.metrics_ = {
            "n_clusters": self.n_clusters,
            "silhouette_score": float(silhouette_score(X, labels)),
            "cluster_centers": self.model.cluster_centers_.tolist(),
            "cluster_sizes": pd.Series(labels).value_counts().to_dict(),
        }
        return pd.Series(labels, index=df.index, name="Traffic_State_Cluster")

    def save(self, directory=config.MODELS_DIR):
        import joblib
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, directory / f"{self.model_name}.joblib")

    def load(self, directory=config.MODELS_DIR):
        import joblib
        bundle = joblib.load(directory / f"{self.model_name}.joblib")
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        return self
