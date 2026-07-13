# Renewable Energy Site Suitability Analyzer — Dashboard

## Files

| File | Purpose |
|---|---|
| `train_pipeline.py` | Reproduces Steps 2, 4 & 5 of your notebooks (cleaning → scaling → K-Means) and **saves the trained model artifacts** to `artifacts/`. Run once. |
| `cluster_insights.py` | Generates cluster names, explanations and recommendations **automatically** from the trained cluster centroids — no manual rules or hard-coded thresholds. |
| `app.py` | The Streamlit dashboard itself. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
pip install -r requirements.txt
```

Place your raw dataset (the same file used in Step 1 of the notebooks) in this
folder as `renewable_energy_dataset.csv`. If it's named differently, edit
`RAW_DATA_PATH` at the top of `train_pipeline.py`.

## Step 1 — Train and save the model artifacts (run once)

```bash
python train_pipeline.py
```

This reproduces your cleaning, feature engineering, scaling and K-Means steps
exactly, then saves everything the dashboard needs into `artifacts/`:

- `standard_scaler.joblib` — the fitted StandardScaler
- `kmeans_model.joblib` — the fitted K-Means model
- `pca_viz.joblib` — a 2D PCA fit used only for the visualization tab
- `cluster_profiles.csv` — average feature values per cluster (real units)
- `cluster_centroids_scaled.csv` — cluster centroids in standardized space
  (these values ARE the z-scores the dashboard uses to name/explain clusters)
- `clustered_dataset.csv` — full dataset with cluster labels attached
- `meta.json` — record/feature/cluster counts

Re-run this script any time your raw dataset changes.

## Step 2 — Launch the dashboard

```bash
streamlit run app.py
```

## How "no hard-coding" is enforced

- **Cluster names** come from comparing the four resource-potential columns
  (`solar`, `wind`, `hydro`, `biomass`) already in your dataset, using each
  cluster's own centroid z-scores — the highest one wins. No cluster is
  manually labeled.
- **Explanations** rank meteorological features by their z-score within the
  predicted cluster; a small lookup table only says *which* known variables
  are relevant to *which* resource (e.g., DNI relates to solar) — it never
  decides names, scores, or which cluster a location falls into.
- **Recommendations** are sentence templates filled in with the cluster's own
  dominant resource and a generic statistical magnitude band (z > 1.0 =
  "notably high", etc. — a standard statistics convention, not a
  domain-specific numeric cutoff).
- **Predictions** for a new location always go through the same
  `scaler.transform()` → `kmeans.predict()` pipeline used during training.

See the comments at the top of `cluster_insights.py` for the full reasoning.
