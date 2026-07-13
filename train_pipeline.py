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
# --------------------------------------------------------------------------
# STEP 4 - MULTI-MODEL FEATURE SELECTION, SCALING, PCA, AND TRAINING
# --------------------------------------------------------------------------

# Suggested features for each model as requested by the user
RESOURCE_FEATURES = {
    "solar": [
        "Solar", "DNI", "GHI", "GTI", 
        "ClearSky_DNI", "ClearSky_GHI", "Cloud_Opacity", 
        "Relative_Humidity", "Air_Temperature", "Precipitation_Rate"
    ],
    "wind": [
        "Wind", "Wind_Speed_100m", "Surface_Pressure", "Air_Temperature"
    ],
    "hydro": [
        "Hydro", "Precipitation_Rate", "Relative_Humidity", "Surface_Pressure"
    ],
    "biomass": [
        "Biomass", "Relative_Humidity", "Air_Temperature", "Precipitation_Rate"
    ]
}

def map_features(suggested_features: list, actual_cols: list) -> list:
    """
    Automatically maps suggested features to the closest column names in the dataset.
    Handles case mismatches and missing columns using fuzzy string matching.
    """
    import difflib
    mapped = []
    for feat in suggested_features:
        # 1. Look for case-insensitive exact match
        found = False
        for col in actual_cols:
            if col.lower() == feat.lower():
                mapped.append(col)
                found = True
                break
        if found:
            continue
        
        # 2. Fallback to fuzzy match if no exact match is found
        matches = difflib.get_close_matches(feat, actual_cols, n=1, cutoff=0.3)
        if matches:
            mapped.append(matches[0])
            
    # Remove any potential duplicates while preserving original order
    seen = set()
    return [x for x in mapped if not (x in seen or seen.add(x))]


def fit_kmeans_optimal_k(X_transformed: pd.DataFrame, k_range=range(2, 11)):
    """
    Calculates K-Means clustering across a range of K.
    Tracks both inertia (Elbow Method) and Silhouette Scores.
    Automatically chooses the optimal K based on the maximum Silhouette Score.
    """
    results = []
    # Safeguard against dataset containing fewer rows than maximum K
    max_k = min(len(X_transformed) - 1, 10)
    actual_range = range(2, max_k + 1)
    
    for k in actual_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_transformed)
        sil = silhouette_score(X_transformed, labels)
        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": sil,
        })
        
    results_df = pd.DataFrame(results)
    best_k = int(results_df.loc[results_df["silhouette_score"].idxmax(), "k"])
    return best_k, results_df


def train_resource_model(df_clean: pd.DataFrame, resource: str, suggested_features: list):
    """
    Trains a completely separate machine learning pipeline for a single renewable resource.
    - Separate feature selection & mapping
    - Separate StandardScaler
    - Separate PCA (only if it reduces dimensionality while retaining >=95% variance)
    - Separate optimal K-Means clustering
    - Separate cluster profiles and standardized centroids
    - Separate 2D visualization PCA for the dashboard plotting
    """
    print(f"\n==================================================")
    print(f" TRAINING MODEL FOR RESOURCE: {resource.upper()} ")
    print(f"==================================================")
    
    # 1. Feature selection & mapping
    actual_cols = df_clean.columns.tolist()
    # Exclude ID columns to avoid clustering on state names, month, or location coordinates
    id_cols = ["state_ut", "year", "month", "latitude", "longitude"]
    candidate_cols = [c for c in actual_cols if c not in id_cols]
    
    feature_cols = map_features(suggested_features, candidate_cols)
    print(f"Suggested Features: {suggested_features}")
    print(f"Mapped Features in Dataset: {feature_cols}")
    
    # Extract features subset
    X = df_clean[feature_cols].copy()
    
    # 2. Separate StandardScaler
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=feature_cols, index=X.index)
    
    # 3. Separate PCA Variance Check
    # Check whether PCA is actually useful. If PCA preserves >=95% variance while
    # reducing dimensionality, use it. Otherwise, skip it.
    D = len(feature_cols)
    pca = None
    use_pca = False
    X_for_clustering = X_scaled.copy()
    
    if D > 1:
        pca_check = PCA(random_state=42)
        pca_check.fit(X_scaled)
        cumulative_variance = np.cumsum(pca_check.explained_variance_ratio_)
        # Find minimum components needed to capture >= 95% variance
        n_comp_needed = int(np.argmax(cumulative_variance >= 0.95) + 1)
        
        if n_comp_needed < D:
            # PCA is useful as it reduces dimensionality
            print(f"PCA status: USE. Dimensionality reduced from {D} to {n_comp_needed} (captures 95% variance)")
            pca = PCA(n_components=n_comp_needed, random_state=42)
            X_transformed = pca.fit_transform(X_scaled)
            X_for_clustering = pd.DataFrame(
                X_transformed,
                columns=[f"pc{i+1}" for i in range(n_comp_needed)],
                index=X.index
            )
            use_pca = True
        else:
            print(f"PCA status: SKIP. Dimensionality reduction not possible: {n_comp_needed} of {D} components needed to capture 95% variance.")
    else:
        print("PCA status: SKIP. Model contains only 1 feature, cannot reduce dimensionality.")
        
    # 4. Determine K and Train KMeans
    best_k, k_search_results = fit_kmeans_optimal_k(X_for_clustering)
    print(f"Selected Optimal K: {best_k}")
    
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_for_clustering)
    
    # 5. Fit a 2D PCA purely for visualization (not used for clustering)
    pca_viz = None
    if D >= 2:
        pca_viz = PCA(n_components=2, random_state=42)
        pca_viz.fit(X_scaled)
        
    # 6. Cluster profiles: average of each feature in original units
    df_with_labels = df_clean.copy()
    df_with_labels["cluster"] = cluster_labels
    cluster_profiles = df_with_labels.groupby("cluster")[feature_cols].mean()
    
    # 7. Cluster centroids in standardized space (mean of standardized features)
    X_scaled_with_labels = X_scaled.copy()
    X_scaled_with_labels["cluster"] = cluster_labels
    centroids_scaled = X_scaled_with_labels.groupby("cluster").mean()
    
    # 8. Save artifacts to models/<resource>/
    resource_dir = os.path.join(ROOT_DIR, "models", resource)
    os.makedirs(resource_dir, exist_ok=True)
    
    joblib.dump(scaler, os.path.join(resource_dir, "scaler.joblib"))
    joblib.dump(kmeans_final, os.path.join(resource_dir, "kmeans.joblib"))
    if pca is not None:
        joblib.dump(pca, os.path.join(resource_dir, "pca.joblib"))
    if pca_viz is not None:
        joblib.dump(pca_viz, os.path.join(resource_dir, "pca_viz.joblib"))
        
    cluster_profiles.to_csv(os.path.join(resource_dir, "cluster_profile.csv"))
    centroids_scaled.to_csv(os.path.join(resource_dir, "cluster_centroids_scaled.csv"))
    k_search_results.to_csv(os.path.join(resource_dir, "k_selection_results.csv"), index=False)
    
    # Write metadata json
    meta = {
        "resource": resource,
        "feature_cols": feature_cols,
        "use_pca": use_pca,
        "pca_components": int(pca.n_components_) if pca is not None else 0,
        "best_k": best_k,
    }
    with open(os.path.join(resource_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"Successfully trained and saved all {resource.upper()} artifacts to '{resource_dir}'.")
    return cluster_labels


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

    # Dictionary to collect predictions from each model
    resource_predictions = {}

    # Train each model independently
    for resource, suggested_features in RESOURCE_FEATURES.items():
        cluster_labels = train_resource_model(df_clean, resource, suggested_features)
        resource_predictions[f"{resource}_cluster"] = cluster_labels

    # Add all resource clustering labels to the cleaned dataset
    df_clustered = df_clean.copy()
    for col_name, labels in resource_predictions.items():
        df_clustered[col_name] = labels

    # Save the global clustered dataset in the models/ folder
    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    df_clustered.to_csv(os.path.join(models_dir, "clustered_dataset.csv"), index=False)
    
    # Save a generic meta file in models/ for high-level record counting
    with open(os.path.join(models_dir, "meta.json"), "w") as f:
        json.dump({
            "n_records": int(len(df_clean)),
            "resources": list(RESOURCE_FEATURES.keys())
        }, f, indent=2)

    print(f"\nAll 4 resource models successfully trained! Outputs organized under '{models_dir}/'.")
    print("Run the dashboard using: streamlit run app.py")


if __name__ == "__main__":
    main()

