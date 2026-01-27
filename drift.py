import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from model_manager import ModelManager

# NOTE: PSI is a simple, interpretable drift score. Values > 0.2 usually indicate moderate drift,
# while > 0.3 is strong drift and should trigger retraining.

def _safe_probs(counts: np.ndarray) -> np.ndarray:
    """Normalize counts and avoid zeros to keep PSI finite."""
    probs = counts / counts.sum()
    probs[probs == 0] = 1e-6
    return probs


def population_stability_index(ref: pd.Series, new: pd.Series, buckets: int = 10) -> float:
    """Compute PSI for a numeric feature using reference quantile bins."""
    ref = ref.dropna()
    new = new.dropna()
    if ref.empty or new.empty:
        return np.nan

    quantiles = np.linspace(0, 1, buckets + 1)
    ref_bins = np.unique(np.quantile(ref, quantiles))
    if len(ref_bins) < 2:  # Constant feature edge case
        return 0.0
    ref_bins[0], ref_bins[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=ref_bins)
    new_counts, _ = np.histogram(new, bins=ref_bins)

    ref_probs = _safe_probs(ref_counts.astype(float))
    new_probs = _safe_probs(new_counts.astype(float))

    return float(np.sum((new_probs - ref_probs) * np.log(new_probs / ref_probs)))


def categorical_psi(ref: pd.Series, new: pd.Series) -> float:
    """PSI-style score for categorical features (frequency shift)."""
    ref = ref.dropna().astype(str)
    new = new.dropna().astype(str)
    if ref.empty or new.empty:
        return np.nan

    ref_freq = ref.value_counts(normalize=True)
    new_freq = new.value_counts(normalize=True)
    categories = set(ref_freq.index) | set(new_freq.index)

    psi = 0.0
    for cat in categories:
        ref_p = ref_freq.get(cat, 0.0)
        new_p = new_freq.get(cat, 0.0)
        if ref_p == 0:  # avoid log(0)
            ref_p = 1e-6
        if new_p == 0:
            new_p = 1e-6
        psi += (new_p - ref_p) * np.log(new_p / ref_p)
    return float(psi)


NUM_FEATURES = [
    'Rooms', 'Price', 'Distance', 'Bedroom2', 'Bathroom',
    'Car', 'Landsize', 'BuildingArea', 'YearBuilt',
    'Lattitude', 'Longtitude', 'Propertycount'
]

CATEGORICAL_FEATURES = ['Type', 'Regionname', 'Method']


def evaluate_drift(
    reference_df: pd.DataFrame,
    new_df: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    psi_threshold: float = 0.2,
    psi_high_threshold: float = 0.4,
    max_drift_feature_ratio: float = 0.3,
) -> Dict:
    """Return drift metrics and a retrain flag based on PSI thresholds."""
    per_feature_psi = {}

    for col in numerical_features:
        if col in reference_df and col in new_df:
            per_feature_psi[col] = population_stability_index(reference_df[col], new_df[col])

    for col in categorical_features:
        if col in reference_df and col in new_df:
            per_feature_psi[col] = categorical_psi(reference_df[col], new_df[col])

    drifted = [f for f, psi in per_feature_psi.items() if psi is not None and not np.isnan(psi) and psi >= psi_threshold]
    drift_ratio = len(drifted) / max(len(per_feature_psi), 1)
    single_feature_spike = any(psi is not None and not np.isnan(psi) and psi >= psi_high_threshold for psi in per_feature_psi.values())
    should_retrain = drift_ratio >= max_drift_feature_ratio or single_feature_spike

    return {
        "psi": per_feature_psi,
        "drifted_features": drifted,
        "drift_ratio": drift_ratio,
        "should_retrain": should_retrain,
    }


def simulate_drifted_houses(reference_df: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """Generate synthetic houses that are intentionally out-of-distribution."""
    rng = np.random.default_rng(42)
    base = reference_df.sample(min(len(reference_df), n), replace=len(reference_df) < n, random_state=42).copy()

    base['Price'] = base['Price'] * rng.uniform(1.8, 2.5, size=len(base))
    base['Distance'] = base['Distance'] + rng.uniform(15, 35, size=len(base))
    base['Rooms'] = np.clip(base['Rooms'] + rng.integers(2, 5, size=len(base)), 1, None)
    base['Bathroom'] = np.clip(base['Bathroom'] + rng.integers(1, 3, size=len(base)), 1, None)
    base['Type'] = rng.choice(['t', 'u'], size=len(base))
    base['Regionname'] = "Outer Ring"
    base['Method'] = rng.choice(['S', 'PI', 'VB'], size=len(base))

    return base


# --- Model comparison utilities ---


def _load_artifact(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)


def load_latest_two_models() -> Tuple[Dict, Dict, Dict, Dict]:
    """Load current and previous model artifacts from registry."""
    manager = ModelManager()
    registry = manager._load_registry()
    history = registry.get("history", [])
    if len(history) < 2:
        raise ValueError("Need at least two models saved to compare.")

    # history is append-only, latest is last
    current_meta = history[-1]
    prev_meta = history[-2]

    current_path = os.path.join(manager.models_dir, current_meta["filename"])
    prev_path = os.path.join(manager.models_dir, prev_meta["filename"])

    current_artifacts = _load_artifact(current_path)
    prev_artifacts = _load_artifact(prev_path)
    return prev_artifacts, current_artifacts, prev_meta, current_meta


def compare_latest_models(
    psi_threshold: float = 0.2,
    psi_high_threshold: float = 0.4,
    max_drift_feature_ratio: float = 0.3,
) -> Dict:
    prev_artifacts, curr_artifacts, prev_meta, curr_meta = load_latest_two_models()

    prev_df = prev_artifacts.get("reference_data")
    if prev_df is None:
        prev_df = prev_artifacts.get("database")

    curr_df = curr_artifacts.get("reference_data")
    if curr_df is None:
        curr_df = curr_artifacts.get("database")
    if prev_df is None or curr_df is None:
        raise KeyError("reference_data/database missing in artifacts")

    # Use canonical feature lists; intersect with available columns for safety
    num_feats = [c for c in NUM_FEATURES if c in prev_df.columns and c in curr_df.columns]
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in prev_df.columns and c in curr_df.columns]

    drift = evaluate_drift(
        reference_df=prev_df,
        new_df=curr_df,
        numerical_features=num_feats,
        categorical_features=cat_feats,
        psi_threshold=psi_threshold,
        psi_high_threshold=psi_high_threshold,
        max_drift_feature_ratio=max_drift_feature_ratio,
    )

    # Drift focusing only on the newly added rows (delta between current and previous)
    batch_rows = max(len(curr_df) - len(prev_df), 0)
    batch_drift = None
    if batch_rows > 0:
        delta_df = curr_df.tail(batch_rows)
        batch_drift = evaluate_drift(
            reference_df=prev_df,
            new_df=delta_df,
            numerical_features=num_feats,
            categorical_features=cat_feats,
            psi_threshold=psi_threshold,
            psi_high_threshold=psi_high_threshold,
            max_drift_feature_ratio=max_drift_feature_ratio,
        )

    return {
        "previous_model": prev_meta,
        "current_model": curr_meta,
        "drift": drift,
        "batch_drift": batch_drift,
        "reference_rows": len(prev_df),
        "current_rows": len(curr_df),
        "batch_rows": batch_rows,
    }
