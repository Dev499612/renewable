"""
train_pipeline.py
------------------
Reproduces Steps 2, 4 and 5 of the project notebooks:
    Step 2 -> Data Cleaning
    Step 4 -> Feature Engineering + StandardScaler
    Step 5 -> K-Means Clustering (K chosen by silhouette score)

Why this file exists
---------------------
The notebooks trained a StandardScaler and a KMeans model in memory, but
only the scaler was ever saved to disk. The Streamlit dashboard needs the
FITTED scaler and FITTED KMeans model (not retrained rules), so this script
re-runs the same pipeline once and saves every artifact the dashboard needs
to `artifacts/`.

Run this ONCE (or whenever the raw dataset changes):
    python train_pipeline.py

It expects the raw dataset at ./renewable_energy_dataset.csv
(same file used in Step 1 of the notebooks). Update RAW_DATA_PATH below if
your file is named/located differently.
"""

import glob
import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# STEP 2 (reproduced) - DATA CLEANING
# --------------------------------------------------------------------------
def clean_data(df_original: pd.DataFrame) -> pd.DataFrame:
    df_clean = df_original.copy()

    # Remove exact duplicate rows
    df_clean = df_clean.drop_duplicates(keep="first").reset_index(drop=True)

    # Standardize hidden missing values (blank/placeholder strings -> NaN)
    object_cols = df_clean.select_dtypes(include="object").columns.tolist()
    for col in object_cols:
        blank_mask = df_clean[col].astype(str).str.strip() == ""
        blank_mask = blank_mask | df_clean[col].isin(
            [" ", "  ", "NA", "N/A", "na", "n/a", "-", "--", "?"]
        )
        df_clean.loc[blank_mask, col] = np.nan

    # Drop the near-empty artifact column, if present
    if "Unnamed: 22" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Unnamed: 22"])

    # 'wind' was stored as text in the raw file -> convert to numeric
    if "wind" in df_clean.columns:
        df_clean["wind"] = pd.to_numeric(df_clean["wind"], errors="coerce")

    numeric_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()

    # Numeric imputation: median (<5% missing) or state/month group median (>=5%)
    numeric_missing_pct = df_clean[numeric_cols].isnull().sum() / len(df_clean) * 100
    for col in numeric_cols:
        pct = numeric_missing_pct[col]
        if pct == 0:
            continue
        elif pct < 5:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            group_median = df_clean.groupby(["Name of State/UT", "MONTH"])[col].transform("median")
            df_clean[col] = df_clean[col].fillna(group_median).fillna(df_clean[col].median())

    # Categorical imputation: mode
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode(dropna=True)[0])

    # Type corrections
    df_clean["YEAR"] = df_clean["YEAR"].astype(int)
    df_clean["MONTH"] = pd.Categorical(
        df_clean["MONTH"],
        categories=["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"],
        ordered=True,
    )
    df_clean["Name of State/UT"] = df_clean["Name of State/UT"].astype("category")

    # Punjab column-shift correction (data-entry error found during EDA)
    punjab_mask = df_clean["Name of State/UT"] == "Punjab"
    suspect = df_clean.loc[punjab_mask, "air_temp"].between(1900, 2100).sum() if "air_temp" in df_clean.columns else 0
    if suspect > 0:
        shift_cols = ["air_temp", "albedo", "clearsky_dhi", "clearsky_dni", "clearsky_gti",
                      "cloud_opacity", "dni", "ghi", "gti", "precipitation_rate",
                      "relative_humidity", "surface_pressure", "wind_speed_100m"]
        shifted_values = df_clean.loc[punjab_mask, shift_cols].to_numpy(dtype=float)
        df_clean.loc[punjab_mask, shift_cols[:-1]] = shifted_values[:, 1:]
        df_clean.loc[punjab_mask, shift_cols[-1]] = np.nan
        df_clean["wind_speed_100m"] = df_clean["wind_speed_100m"].fillna(df_clean["wind_speed_100m"].median())

    # Domain-range validation (physically impossible values corrected)
    def fix_negative(col):
        neg_mask = df_clean[col] < 0
        if neg_mask.any():
            df_clean.loc[neg_mask, col] = df_clean.loc[neg_mask, col].abs()

    for col in ["precipitation_rate", "dni", "ghi", "gti", "clearsky_dhi", "clearsky_dni",
                "clearsky_gti", "wind", "wind_speed_100m"]:
        if col in df_clean.columns:
            fix_negative(col)

    rh_invalid = ~df_clean["relative_humidity"].between(0, 100)
    if rh_invalid.any():
        df_clean.loc[rh_invalid, "relative_humidity"] = df_clean.loc[~rh_invalid, "relative_humidity"].median()

    temp_invalid = ~df_clean["air_temp"].between(-25, 55)
    if temp_invalid.any():
        df_clean.loc[temp_invalid, "air_temp"] = df_clean.loc[~temp_invalid, "air_temp"].median()

    albedo_invalid = ~df_clean["albedo"].between(0, 1)
    if albedo_invalid.any():
        df_clean.loc[albedo_invalid, "albedo"] = df_clean.loc[~albedo_invalid, "albedo"].median()

    # Rename columns to snake_case
    rename_map = {
        "Name of State/UT": "state_ut",
        "YEAR": "year",
        "MONTH": "month",
        "Latitude": "latitude",
        "Longitude": "longitude",
    }
    df_clean = df_clean.rename(columns=rename_map)

    return df_clean


# --------------------------------------------------------------------------
# STEP 4 (reproduced) - FEATURE ENGINEERING + SCALING
# --------------------------------------------------------------------------
def build_features(df_clean: pd.DataFrame):
    id_cols = ["state_ut", "year", "month", "latitude", "longitude"]
    clustering_features = [c for c in df_clean.columns if c not in id_cols]

    X = df_clean[clustering_features].copy()
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=clustering_features, index=X.index)

    return X_scaled, clustering_features, scaler


# --------------------------------------------------------------------------
# STEP 5 (reproduced) - K-MEANS CLUSTERING (K chosen by silhouette score)
# --------------------------------------------------------------------------
def fit_kmeans(X_scaled: pd.DataFrame, k_range=range(2, 11)):
    results = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": silhouette_score(X_scaled, labels),
        })
    results_df = pd.DataFrame(results)
    best_k = int(results_df.loc[results_df["silhouette_score"].idxmax(), "k"])

    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels_final = kmeans_final.fit_predict(X_scaled)

    return kmeans_final, labels_final, best_k, results_df


def resolve_raw_data_path():
    if RAW_DATA_PATH and os.path.exists(RAW_DATA_PATH):
        return RAW_DATA_PATH

    explicit_candidates = [
        os.path.join(ROOT_DIR, "renewable_energy_dataset.csv"),
        os.path.join(ROOT_DIR, "STATEWISE_CLIMATE_RENEWABLEENERGY_DATA", "Comprehensive_data.xlsx"),
        os.path.join(ROOT_DIR, "STATEWISE_CLIMATE_RENEWABLEENERGY_DATA", "Comprehensive_data.csv"),
    ]
    for candidate in explicit_candidates:
        if os.path.exists(candidate):
            return candidate

    search_patterns = [
        os.path.join(ROOT_DIR, "**", "*.csv"),
        os.path.join(ROOT_DIR, "**", "*.xlsx"),
        os.path.join(ROOT_DIR, "**", "*.xls"),
        os.path.join(ROOT_DIR, "**", "*.xlsm"),
    ]
    for pattern in search_patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "Could not find a training dataset. Place renewable_energy_dataset.csv or a workbook under STATEWISE_CLIMATE_RENEWABLEENERGY_DATA/."
    )


def load_raw_data(path: str) -> pd.DataFrame:
    lower_path = path.lower()
    if lower_path.endswith(".csv"):
        return pd.read_csv(path)
    if lower_path.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(path, sheet_name=0)
    raise ValueError(f"Unsupported data format: {path}")


def main():
    raw_data_path = resolve_raw_data_path()
    print(f"Loading raw dataset from '{raw_data_path}' ...")
    df_original = load_raw_data(raw_data_path)

    print("Cleaning data (Step 2 logic) ...")
    df_clean = clean_data(df_original)

    print("Building features + scaling (Step 4 logic) ...")
    X_scaled, feature_cols, scaler = build_features(df_clean)

    print("Selecting K and training final K-Means model (Step 5 logic) ...")
    kmeans_final, cluster_labels, best_k, k_search_results = fit_kmeans(X_scaled)
    print(f"Best K selected: {best_k}")

    # Attach cluster labels to both scaled and original-unit data
    X_scaled["cluster"] = cluster_labels
    df_clean["cluster"] = cluster_labels

    # Fit a 2D PCA purely for visualization (not used for clustering itself,
    # exactly as in the notebook)
    pca_viz = PCA(n_components=2)
    pca_viz.fit(X_scaled[feature_cols])

    # Cluster profiles: average of each feature, in real (original) units
    cluster_profiles = df_clean.groupby("cluster")[feature_cols].mean()

    # Cluster centroids in STANDARDIZED space. Because features were scaled
    # with the global mean/std, each centroid value here IS the z-score of
    # that cluster's average relative to the overall dataset average.
    # The dashboard uses these z-scores (not manual rules) to describe and
    # name clusters automatically.
    centroids_scaled = pd.DataFrame(kmeans_final.cluster_centers_, columns=feature_cols)
    centroids_scaled.index.name = "cluster"

    # --------------------------------------------------------------------
    # SAVE ARTIFACTS
    # --------------------------------------------------------------------
    joblib.dump(scaler, f"{ARTIFACTS_DIR}/standard_scaler.joblib")
    joblib.dump(kmeans_final, f"{ARTIFACTS_DIR}/kmeans_model.joblib")
    joblib.dump(pca_viz, f"{ARTIFACTS_DIR}/pca_viz.joblib")

    with open(f"{ARTIFACTS_DIR}/feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    cluster_profiles.to_csv(f"{ARTIFACTS_DIR}/cluster_profiles.csv")
    centroids_scaled.to_csv(f"{ARTIFACTS_DIR}/cluster_centroids_scaled.csv")
    df_clean.to_csv(f"{ARTIFACTS_DIR}/clustered_dataset.csv", index=False)
    k_search_results.to_csv(f"{ARTIFACTS_DIR}/k_selection_results.csv", index=False)

    with open(f"{ARTIFACTS_DIR}/meta.json", "w") as f:
        json.dump({
            "best_k": best_k,
            "n_records": int(len(df_clean)),
            "n_features": len(feature_cols),
        }, f, indent=2)

    print(f"\nAll artifacts saved to '{ARTIFACTS_DIR}/'. You can now run the dashboard:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()
