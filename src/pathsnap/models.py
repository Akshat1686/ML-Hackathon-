"""Model zoo: every candidate scorer we compare (all binary: is this candidate the true arc?).

Each factory returns an sklearn estimator with predict_proba. Configurations here are the
*untuned* baselines recorded in the comparison table.
"""
from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import SEED

# family, short config description, factory, max training rows (None = all; KNN is subsampled for speed)
MODEL_SPECS = {
    "LogReg": ("Linear / probabilistic", "StandardScaler + L2 logistic regression, C=1.0, class_weight=None",
               lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)), None),
    "GaussianNB": ("Probabilistic", "QuantileTransformer(normal) + Gaussian NB (var_smoothing=1e-9)",
                   lambda: make_pipeline(QuantileTransformer(output_distribution="normal", random_state=SEED), GaussianNB()), None),
    "KNN": ("Instance-based", "StandardScaler + distance-weighted KNN, k=25 (fit on 120k-row subsample)",
            lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)), 120_000),
    "DecisionTree": ("Tree-based", "max_depth=12, min_samples_leaf=20",
                     lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED), None),
    "RandomForest": ("Ensemble (bagging)", "100 trees, max_depth=None, min_samples_leaf=5, max_samples=0.3",
                     lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_samples=0.3,
                                                    n_jobs=-1, random_state=SEED), None),
    "ExtraTrees": ("Ensemble (randomised trees)", "150 trees, min_samples_leaf=5, max_samples=0.3, bootstrap",
                   lambda: ExtraTreesClassifier(n_estimators=150, min_samples_leaf=5, bootstrap=True, max_samples=0.3,
                                                n_jobs=-1, random_state=SEED), None),
    "HistGB": ("Boosting", "HistGradientBoosting lr=0.1, max_iter=200, max_leaf_nodes=31, early stopping",
               lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, max_leaf_nodes=31,
                                                      early_stopping=True, random_state=SEED), None),
    "MLP": ("Neural network", "StandardScaler + MLP(64,32) relu, adam, alpha=1e-4, early stopping",
            lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                                                                  max_iter=60, early_stopping=True, random_state=SEED)), None),
}


def make_hgb(params: dict):
    """Tunable HistGradientBoosting used for Bayesian optimisation and as final model."""
    return HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"], max_iter=int(params["max_iter"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]), min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=params["l2_regularization"], max_features=params.get("max_features", 1.0),
        early_stopping=True, n_iter_no_change=15, validation_fraction=0.1, random_state=SEED)
