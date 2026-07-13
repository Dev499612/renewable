"""
app.py
------
Renewable Energy Site Suitability Analyzer - Dashboard
 
Everything shown here is derived from the TRAINED artifacts saved by
train_pipeline.py (StandardScaler, KMeans model, cluster profiles, cluster
centroids). No cluster names, suitability scores, or recommendations are
hard-coded in this file -- see cluster_insights.py for how they are
generated automatically from the model's own output.
 
Layout is organized into tabs (Overview / India Map / Explainability /
Cluster Comparison / State Rankings / State Comparison / Data) so the
dashboard feels like an app rather than one long scroll. The prediction
pipeline itself (scaler.transform -> kmeans.predict) is unchanged from
the original version.
 
Run:
    streamlit run app.py
"""
 
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
 
from cluster_insights import (
    build_all_cluster_insights, RESOURCE_COLUMNS, resource_specific_note, z_to_suitability_score,
)
 
ARTIFACTS_DIR = "artifacts"
WEATHER_MAP = {
    "Temperature (°C)": ("air_temp", "🌡️"),
    "Humidity (%)": ("relative_humidity", "💧"),
    "Rainfall (mm/hr)": ("precipitation_rate", "🌧️"),
    "Wind Speed (100m, m/s)": ("wind_speed_100m", "💨"),
    "Cloud Opacity (%)": ("cloud_opacity", "☁️"),
    "Solar Radiation - GHI": ("ghi", "☀️"),
}
# One color per cluster, reused everywhere so a given cluster always looks
# the same across every chart and the map.
CLUSTER_COLORS = ["#1F8A54", "#6FCB9F", "#0E2419", "#E3B23C", "#2F5D46", "#4A90D9", "#C4756B",
                   "#8E5FB0", "#D46A8C", "#3E8E8E"]
 
st.set_page_config(
    page_title="Renewable Energy Site Suitability Analyzer",
    page_icon="🌍",
    layout="wide",
)
 
# Small CSS fix: Streamlit's built-in st.metric() truncates long values with
# an ellipsis and gives no way to wrap them. Cluster names and state names
# can be long, so KPI numbers here use this custom card instead -- same
# values as before, just readable instead of cut off.
st.markdown("""
<style>
.kpi-card{
    background:rgba(127,127,127,0.08); border:1px solid rgba(127,127,127,0.25);
    border-radius:14px; padding:14px 16px; height:100%;
}
.kpi-card .kpi-label{ font-size:0.8rem; opacity:0.7; margin-bottom:4px; }
.kpi-card .kpi-value{ font-size:1.4rem; font-weight:700; line-height:1.25; white-space:normal; }
.kpi-card .kpi-sub{ font-size:0.85rem; opacity:0.65; margin-top:2px; }
.ranking-table{ width:100%; border-collapse:collapse; }
.ranking-table th{ text-align:left; padding:8px 12px; font-weight:600; border-bottom:2px solid rgba(127,127,127,0.3); }
.ranking-table td{ padding:8px 12px; border-bottom:1px solid rgba(127,127,127,0.1); }
.rank-badge{ display:inline-block; width:28px; height:28px; line-height:28px; text-align:center; border-radius:50%; font-weight:700; font-size:0.85rem; }
.rank-1{ background:#FFD700; color:#1a1a1a; }
.rank-2{ background:#C0C0C0; color:#1a1a1a; }
.rank-3{ background:#CD7F32; color:#fff; }
</style>
""", unsafe_allow_html=True)
 
 
def kpi_card(label, value, sub=None):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)
 
 
 
# --------------------------------------------------------------------------
# LOAD ARTIFACTS (cached so the model/data only loads once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(f"{ARTIFACTS_DIR}/standard_scaler.joblib")
    kmeans = joblib.load(f"{ARTIFACTS_DIR}/kmeans_model.joblib")
    pca_viz = joblib.load(f"{ARTIFACTS_DIR}/pca_viz.joblib")
 
    with open(f"{ARTIFACTS_DIR}/feature_columns.json") as f:
        feature_cols = json.load(f)
    with open(f"{ARTIFACTS_DIR}/meta.json") as f:
        meta = json.load(f)
 
    cluster_profiles = pd.read_csv(f"{ARTIFACTS_DIR}/cluster_profiles.csv", index_col="cluster")
    centroids_scaled = pd.read_csv(f"{ARTIFACTS_DIR}/cluster_centroids_scaled.csv", index_col="cluster")
    clustered_data = pd.read_csv(f"{ARTIFACTS_DIR}/clustered_dataset.csv")
 
    insights = build_all_cluster_insights(centroids_scaled)
 
    # Pre-compute a 2D PCA projection of every record, for the scatter plot
    X_scaled_all = scaler.transform(clustered_data[feature_cols])
    pca_coords = pca_viz.transform(X_scaled_all)
    clustered_data = clustered_data.copy()
    clustered_data["pc1"] = pca_coords[:, 0]
    clustered_data["pc2"] = pca_coords[:, 1]
 
    return {
        "scaler": scaler,
        "kmeans": kmeans,
        "pca_viz": pca_viz,
        "feature_cols": feature_cols,
        "meta": meta,
        "cluster_profiles": cluster_profiles,
        "centroids_scaled": centroids_scaled,
        "clustered_data": clustered_data,
        "insights": insights,
    }
 
 
@st.cache_data
def build_state_map_data(clustered_data: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """One row per state -- averaged across all months/years -- used for the
    geographic map. Cached since it only depends on the (static) trained
    dataset, not on any sidebar selection."""
    agg = {**{c: "mean" for c in feature_cols}, "latitude": "mean", "longitude": "mean",
           "cluster": lambda x: x.mode().iloc[0]}
    state_avg = clustered_data.groupby("state_ut").agg(agg).reset_index()
    return state_avg
 
 
def predict_cluster(scaler, kmeans, feature_cols, feature_values):
    """Runs the TRAINED scaler + TRAINED KMeans model on a feature vector.
    Kept as a single function so this exact pipeline step is easy to spot
    and never gets duplicated or drift from itself elsewhere in the file."""
    features_df = pd.DataFrame([feature_values.values], columns=feature_cols)
    scaled = scaler.transform(features_df)
    cluster_id = int(kmeans.predict(scaled)[0])
    return cluster_id, scaled
 
 
@st.cache_data
def build_state_rankings(_scaler, feature_cols, clustered_data):
    """Compute a fully data-driven ranking of every state by each renewable
    resource potential. Uses the trained scaler's standardization (already fit
    on the entire dataset) to compute z-scores, then converts to 0-100
    suitability scores via the standard normal CDF (z_to_suitability_score).
    
    No hardcoded weights, no manual thresholds -- every number is derived
    from the model's own trained statistics. If the model is retrained on
    new data, these rankings update automatically."""
    # Average all months/years per state to get the overall profile
    feature_df = clustered_data.groupby("state_ut")[feature_cols].mean().reset_index()
    
    # Standardize using the trained scaler so scores reflect standing
    # relative to the full dataset distribution (the same distribution
    # the KMeans model was trained on).
    scaled = _scaler.transform(feature_df[feature_cols])
    
    rankings = {"state_ut": feature_df["state_ut"].values}
    for resource_name, resource_col in RESOURCE_COLUMNS.items():
        feat_idx = feature_cols.index(resource_col)
        z_scores = scaled[:, feat_idx]
        scores = [z_to_suitability_score(z) for z in z_scores]
        rankings[resource_name] = scores
        rankings[f"{resource_name} z"] = [round(z, 3) for z in z_scores]
    
    ranking_df = pd.DataFrame(rankings)
    
    # Add overall score (average across all four resources)
    score_cols = list(RESOURCE_COLUMNS.keys())
    ranking_df["Overall"] = ranking_df[score_cols].mean(axis=1).round(1)
    
    return ranking_df


@st.cache_data
def compute_cluster_confidence(_scaler, _kmeans, feature_cols, clustered_data):
    """For each state's year-round average profile, compute:
    - assigned_cluster: the K-Means label
    - confidence: a 0-1 score where 1 = very certain, 0 = borderline
      (based on the ratio of distance to nearest other centroid vs.
       distance to assigned centroid)
    - second_cluster: the next-best cluster
    - second_cluster_name: human-readable name of that cluster
    
    States with confidence < 0.7 are considered 'bi-cluster' states
    that sit on a boundary between two climate zones."""
    centroids = _kmeans.cluster_centers_
    n_clusters = centroids.shape[0]
    
    feature_df = clustered_data.groupby("state_ut")[feature_cols].mean().reset_index()
    X = _scaler.transform(feature_df[feature_cols])
    
    results_rows = []
    for i, row_vec in enumerate(X):
        # Euclidean distance to every centroid
        dists = np.sqrt(((centroids - row_vec) ** 2).sum(axis=1))
        closest_idx = np.argmin(dists)
        closest_dist = dists[closest_idx]
        
        # Remove the assigned centroid, find the next closest
        dists_no_assigned = dists.copy()
        dists_no_assigned[closest_idx] = np.inf
        second_idx = np.argmin(dists_no_assigned)
        second_dist = dists_no_assigned[second_idx]
        
        # Confidence: ratio of (distance to assigned) / (distance to second-best)
        # If assigned is much closer → confidence near 1
        # If they are equal → confidence near 0.5
        # Range is 0.5 to 1.0, so we normalize: 2 * (1 - (closest / second))
        # capped at 0 for the lower bound.
        if second_dist == 0:
            conf = 1.0  # exact match, perfect confidence
        else:
            raw_ratio = closest_dist / second_dist
            # Scale: ratio=0.5 -> 0.67, ratio=0.8 -> 0.44, ratio=1.0 -> 0.0
            conf = max(0.0, 1.0 - raw_ratio)
        
        results_rows.append({
            "state_ut": feature_df["state_ut"].iloc[i],
            "assigned_cluster": int(closest_idx),
            "confidence": round(conf, 3),
            "second_cluster": int(second_idx),
        })
    
    return pd.DataFrame(results_rows)
 
 
try:
    A = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found in the `artifacts/` folder.\n\n"
        "Run `python train_pipeline.py` first (with your raw dataset in "
        "this folder) to generate the trained scaler, K-Means model and "
        "cluster profiles that this dashboard depends on."
    )
    st.stop()
 
CLUSTER_NAME_LOOKUP = A["insights"]["cluster_name"].to_dict()
 
# Pre-compute state rankings once (cached) -- same data used in the
# State Rankings tab below. No need to recompute on every interaction.
STATE_RANKINGS = build_state_rankings(A["scaler"], A["feature_cols"], A["clustered_data"])

# Pre-compute cluster confidence for every state
CLUSTER_CONFIDENCE = compute_cluster_confidence(
    A["scaler"], A["kmeans"], A["feature_cols"], A["clustered_data"]
)
# Add readable cluster names
CLUSTER_CONFIDENCE["assigned_name"] = CLUSTER_CONFIDENCE["assigned_cluster"].map(CLUSTER_NAME_LOOKUP)
CLUSTER_CONFIDENCE["second_name"] = CLUSTER_CONFIDENCE["second_cluster"].map(CLUSTER_NAME_LOOKUP)
# Flag bi-cluster states (confidence < 0.7)
CLUSTER_CONFIDENCE["bi_cluster"] = CLUSTER_CONFIDENCE["confidence"] < 0.7
 
 
# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
st.sidebar.title("🔧 Analysis Controls")
 
resource_choice = st.sidebar.selectbox(
    "Renewable Resource of Interest",
    options=list(RESOURCE_COLUMNS.keys()),
    help=(
        "Highlights this resource in the charts/map and adds a note about it in "
        "Overall Analysis. It does NOT change the predicted cluster -- that's based "
        "on the location's full climate profile, not on which resource you're browsing for."
    ),
)
 
states = sorted(A["clustered_data"]["state_ut"].unique())
state_choice = st.sidebar.selectbox("State / UT", options=states)
 
analyze_clicked = st.sidebar.button("🔍 Analyze Site", use_container_width=True)

# A plain st.button() only returns True on the exact rerun where it was
# clicked -- changing ANY other sidebar widget afterwards triggers a new
# rerun where the button "wasn't clicked", which would otherwise wipe the
# whole results view. Session state makes "has been analyzed" persist
# across those reruns.
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if analyze_clicked:
    st.session_state.analyzed = True
 
 
# --------------------------------------------------------------------------
# PAGE HEADER
# --------------------------------------------------------------------------
st.title("🌍 Renewable Energy Site Suitability Analyzer")
st.markdown(
    "This dashboard uses a **trained K-Means clustering model** to group locations "
    "by their meteorological and renewable-resource characteristics. Select a state "
    "in the sidebar, then click **Analyze Site** -- all insights below are "
    "derived automatically from the trained model, not manual rules."
)
st.divider()
 
if not st.session_state.analyzed:
    st.info("👈 Choose a state in the sidebar, then click **Analyze Site** to begin.")
    st.stop()
 
# --------------------------------------------------------------------------
# RUN THE TRAINED PIPELINE FOR THE SELECTED STATE
# --------------------------------------------------------------------------
feature_cols = A["feature_cols"]
# Average across all months/years for the selected state to get an overall picture
state_data = A["clustered_data"][
    A["clustered_data"]["state_ut"] == state_choice
]
if state_data.empty:
    st.warning("No data found for that state. Try a different selection.")
    st.stop()
 
location_features = state_data[feature_cols].mean()  # average across all months and years
predicted_cluster, X_new_scaled = predict_cluster(A["scaler"], A["kmeans"], feature_cols, location_features)
pc_coords_new = A["pca_viz"].transform(X_new_scaled)[0]
 
cluster_name = A["insights"].loc[predicted_cluster, "cluster_name"]
explanation = A["insights"].loc[predicted_cluster, "explanation"]
recommendation = A["insights"].loc[predicted_cluster, "recommendation"]
cluster_profile_row = A["cluster_profiles"].loc[predicted_cluster]
cluster_color = CLUSTER_COLORS[predicted_cluster % len(CLUSTER_COLORS)]

# This location's own precise z-score for the selected resource (not the
# cluster-average version) -> converted into a 0-100 suitability score via
# the standard normal CDF. See z_to_suitability_score() in cluster_insights.py
# for why this involves no hardcoded weights or thresholds.
resource_col = RESOURCE_COLUMNS[resource_choice]
resource_feature_idx = feature_cols.index(resource_col)
location_resource_z = X_new_scaled[0][resource_feature_idx]
suitability_score = z_to_suitability_score(location_resource_z)

# Look up cluster confidence for the selected state
state_conf_row = CLUSTER_CONFIDENCE[CLUSTER_CONFIDENCE["state_ut"] == state_choice].iloc[0]
conf_score = state_conf_row["confidence"]
second_cluster_id = state_conf_row["second_cluster"]
second_cluster_name = state_conf_row["second_name"]
is_bi_cluster = state_conf_row["bi_cluster"]
 
 
# ============================================================================
# TABS -- keeps the dashboard interactive/navigable instead of one long scroll
# ============================================================================
tab_overview, tab_map, tab_explain, tab_compare, tab_rankings, tab_scenario, tab_data = st.tabs(
    ["🏠 Overview", "🗺️ India Map", "🔎 Explainability", "📈 Cluster Comparison",
     "🏆 State Rankings", "🆚 State Comparison", "📁 Data & Download"]
)
 
# ------------------------------------------------------------------
# TAB 1 — OVERVIEW
# ------------------------------------------------------------------
with tab_overview:
    st.subheader("📊 Overall Analysis")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Predicted Cluster", cluster_name, sub=f"Cluster {predicted_cluster}")
    with c2:
        kpi_card("Location", state_choice, sub="overall average")
    with c3:
        kpi_card("Resource of Interest", resource_choice, sub="selected in sidebar")
    with c4:
        kpi_card(f"{resource_choice} Suitability Score", f"{suitability_score}/100",
                  sub=f"this location's own percentile, z={location_resource_z:+.2f}")
    
    # Cluster confidence indicator
    conf_pct = round(conf_score * 100)
    conf_label = f"{conf_pct}%"
    if conf_score >= 0.85:
        conf_icon = "🟢 Strong Fit"
    elif conf_score >= 0.7:
        conf_icon = "🟡 Moderate Fit"
    else:
        conf_icon = "🟠 Boundary State"
    kpi_card("Cluster Confidence", conf_label, sub=conf_icon)
    
    st.success(f"**Recommendation:** {recommendation}")

    resource_note = resource_specific_note(A["centroids_scaled"].loc[predicted_cluster], resource_choice)
    st.info(f"**About {resource_choice} here:** {resource_note}")
    
    # Bi-cluster alert if applicable
    if is_bi_cluster:
        st.warning(
            f"⚠️ **{state_choice} sits on a cluster boundary.** "
            f"The model assigned it to **{cluster_name}** (Cluster {predicted_cluster}), "
            f"but it is also very close to **{second_cluster_name}** (Cluster {second_cluster_id}). "
            "This means the state shares characteristics of both zones — consider evaluating "
            "site suitability with both profiles in mind."
        )
    elif conf_score >= 0.85:
        st.info(
            f"✅ **{state_choice} is a strong fit for {cluster_name}.** "
            f"The model is {conf_pct}% confident in this assignment, with no close "
            "alternative cluster nearby."
        )
 
    st.subheader("🌤️ Weather Summary (Overall Year-Round Averages)")
    weather_cols = st.columns(len(WEATHER_MAP))
    for col_widget, (label, (feat, icon)) in zip(weather_cols, WEATHER_MAP.items()):
        if feat in location_features.index:
            col_widget.metric(f"{icon} {label}", f"{location_features[feat]:.2f}")
 
    st.subheader("⚡ Renewable Resource Comparison")
    resource_rows = [{"Resource": name, "Potential": location_features[col]} for name, col in RESOURCE_COLUMNS.items()]
    resource_df = pd.DataFrame(resource_rows)
    colors = ["#F5A623" if r == resource_choice else "#4A90D9" for r in resource_df["Resource"]]
    fig_resource = go.Figure(go.Bar(
        x=resource_df["Resource"], y=resource_df["Potential"],
        marker_color=colors, text=resource_df["Potential"].round(2), textposition="outside",
    ))
    fig_resource.update_layout(
        title=f"Overall Resource Potential at {state_choice} — {resource_choice} highlighted",
        yaxis_title="Potential (dataset units)",
    )
    fig_resource.update_layout(height=400)
    st.plotly_chart(fig_resource, use_container_width=True)
    st.caption("Want to see how all states compare? Check the 🏆 **State Rankings** tab.")
 
    st.subheader("🧬 Cluster Profile (Cluster Averages)")
    st.markdown(
        f"Cluster **{predicted_cluster}** ({cluster_name}) — average feature values "
        "learned by the model across every location assigned to this cluster:"
    )
    st.dataframe(
        cluster_profile_row.rename("Average Value").to_frame().style.format("{:.2f}"),
        use_container_width=True,
    )
 
# ------------------------------------------------------------------
# TAB 2 — INDIA MAP (geographic view of every state's climate cluster)
# ------------------------------------------------------------------
with tab_map:
    st.subheader("🗺️ India Renewable Climate Map")
    st.caption(
        "Every state's dominant climate cluster, plotted geographically. Marker size reflects "
        f"**{resource_choice}** potential; color shows the cluster the model assigned. "
        "Your analyzed location is marked with a black star."
    )
 
    state_map_df = build_state_map_data(A["clustered_data"], feature_cols)
    state_map_df["cluster_str"] = state_map_df["cluster"].astype(str)
    state_map_df["cluster_name"] = state_map_df["cluster"].map(CLUSTER_NAME_LOOKUP)
    resource_col = RESOURCE_COLUMNS[resource_choice]
 
    color_map = {str(cid): CLUSTER_COLORS[cid % len(CLUSTER_COLORS)] for cid in CLUSTER_NAME_LOOKUP}
 
    fig_map = px.scatter_map(
        state_map_df, lat="latitude", lon="longitude",
        color="cluster_str", size=resource_col,
        size_max=32, zoom=3.5, height=560,
        hover_name="state_ut",
        hover_data={"cluster_name": True, resource_col: ":.1f", "latitude": False, "longitude": False, "cluster_str": False},
        color_discrete_map=color_map,
        labels={"cluster_str": "Cluster", resource_col: resource_choice},
    )
    fig_map.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
 
    # Highlight the analyzed location with a star, using its real lat/long
    loc_row = state_data.iloc[0]
    fig_map.add_trace(go.Scattermap(
        lat=[loc_row["latitude"]], lon=[loc_row["longitude"]],
        mode="markers+text", marker=dict(size=20, color="black", symbol="star"),
        text=[f"{state_choice}"], textposition="top center",
        name="Selected site", showlegend=True,
    ))
    st.plotly_chart(fig_map, use_container_width=True)
 
    with st.expander("What does this map show?"):
        st.write(
            "Each dot is a state, colored by the climate cluster the trained K-Means model "
            "assigned it to (based on its year-round average weather and resource-potential "
            "profile), and sized by how much of your selected resource it has. States with "
            "the same color share a similar climate profile according to the model -- this "
            "is a quick way to spot regional patterns in renewable potential across India."
        )
    
    # ---- Bi-cluster states section ----
    st.subheader("🔄 Boundary States (Bi-Cluster Zones)")
    st.caption(
        "States highlighted below sit on a boundary between two climate clusters. "
        "The model is less confident about their assignment, meaning they share "
        "characteristics of two zones. These states may benefit from a hybrid "
        "renewable energy strategy."
    )
    
    bi_cluster_df = CLUSTER_CONFIDENCE[CLUSTER_CONFIDENCE["bi_cluster"]].copy()
    if len(bi_cluster_df) > 0:
        bi_cluster_df = bi_cluster_df.sort_values("confidence").reset_index(drop=True)
        bi_cluster_df["Confidence"] = (bi_cluster_df["confidence"] * 100).round(0).astype(int).astype(str) + "%"
        bi_cluster_df["Assigned Cluster"] = bi_cluster_df["assigned_name"]
        bi_cluster_df["Also Close To"] = bi_cluster_df["second_name"]
        
        st.dataframe(
            bi_cluster_df[["state_ut", "Assigned Cluster", "Also Close To", "Confidence"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No boundary states detected — all states have a clear dominant cluster.")
 
    st.subheader("PCA Projection (all individual records)")
    scatter_df = A["clustered_data"][["pc1", "pc2", "cluster", "state_ut"]].copy()
    scatter_df["cluster"] = scatter_df["cluster"].astype(str)
    fig_pca = px.scatter(
        scatter_df, x="pc1", y="pc2", color="cluster",
        hover_data=["state_ut"], opacity=0.5,
        color_discrete_map=color_map,
        title="All Locations Projected to 2D (PCA) — Colored by Cluster",
    )
    fig_pca.add_trace(go.Scatter(
        x=[pc_coords_new[0]], y=[pc_coords_new[1]],
        mode="markers", marker=dict(size=18, color="black", symbol="star"),
        name=f"{state_choice}",
    ))
    fig_pca.update_layout(height=480)
    st.plotly_chart(fig_pca, use_container_width=True)
    st.caption(
        "PCA is used only for visualization, not for clustering itself. These 2 components "
        f"explain {sum(A['pca_viz'].explained_variance_ratio_) * 100:.1f}% of total variance."
    )
 
# ------------------------------------------------------------------
# TAB 3 — EXPLAINABILITY
# ------------------------------------------------------------------
with tab_explain:
    st.subheader("🔎 Why This Cluster?")
    st.write(explanation)
    
    # Cluster confidence breakdown
    st.subheader("📊 Cluster Assignment Confidence")
    if is_bi_cluster:
        st.warning(
            f"**{state_choice}** is close to **both** {cluster_name} and {second_cluster_name}. "
            f"Confidence: **{conf_pct}%** — consider evaluating both profiles."
        )
    else:
        st.success(
            f"**{state_choice}** is clearly assigned to **{cluster_name}**. "
            f"Confidence: **{conf_pct}%**."
        )
    
    # Visual bar showing confidence vs threshold
    conf_color = "#2E7D32" if conf_score >= 0.85 else "#F5A623" if conf_score >= 0.7 else "#C62828"
    fig_conf = go.Figure(go.Bar(
        x=[conf_score * 100],
        y=["Confidence"],
        orientation="h",
        marker_color=conf_color,
        text=f"{conf_pct}%",
        textposition="outside",
    ))
    fig_conf.add_vline(x=70, line_dash="dash", line_color="orange", annotation_text="Boundary threshold (70%)")
    fig_conf.update_layout(
        title="Cluster Assignment Confidence (higher = more distinct from other clusters)",
        xaxis=dict(range=[0, 100], title="Confidence (%)"),
        height=200,
    )
    st.plotly_chart(fig_conf, use_container_width=True)
 
    centroid_row = A["centroids_scaled"].loc[predicted_cluster].sort_values(key=abs, ascending=False)
    top_features = centroid_row.head(8)
    fig_z = go.Figure(go.Bar(
        x=top_features.values, y=top_features.index, orientation="h",
        marker_color=["#2E7D32" if v > 0 else "#C62828" for v in top_features.values],
    ))
    fig_z.update_layout(
        title="Top Features Driving This Cluster (standardized distance from dataset average)",
        xaxis_title="Z-score (0 = dataset average)", height=420,
    )
    st.plotly_chart(fig_z, use_container_width=True)
 
    st.subheader("✅ Recommendation")
    st.info(recommendation)
 
# ------------------------------------------------------------------
# TAB 4 — CLUSTER COMPARISON
# ------------------------------------------------------------------
with tab_compare:
    st.subheader("📈 Cluster Comparison")
    compare_cols = list(RESOURCE_COLUMNS.values())
    compare_df = A["cluster_profiles"][compare_cols].reset_index().melt(
        id_vars="cluster", var_name="Resource", value_name="Average Potential"
    )
    compare_df["Resource"] = compare_df["Resource"].map({v: k for k, v in RESOURCE_COLUMNS.items()})
    fig_compare = px.bar(
        compare_df, x="cluster", y="Average Potential", color="Resource", barmode="group",
        title="Resource Potential by Cluster (your predicted cluster is outlined)",
    )
    for cl in sorted(compare_df["cluster"].unique()):
        if cl == predicted_cluster:
            fig_compare.add_vrect(x0=cl - 0.5, x1=cl + 0.5, fillcolor="yellow", opacity=0.15, line_width=0)
    fig_compare.update_layout(height=460)
    st.plotly_chart(fig_compare, use_container_width=True)
 
    st.subheader("Cluster Directory")
    directory_df = A["insights"][["cluster_name", "recommendation"]].reset_index()
    directory_df.columns = ["Cluster", "Name", "Recommendation"]
    st.dataframe(directory_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------
# TAB 5 — STATE RANKINGS (fully data-driven ranking of every state
# by each renewable resource, using the trained scaler + standard
# normal CDF — no hardcoded weights or thresholds)
# ------------------------------------------------------------------
with tab_rankings:
    st.subheader("🏆 State Rankings by Renewable Potential")
    st.caption(
        "Every state is scored 0–100 for each renewable resource using the same trained "
        "model pipeline. Scores reflect each state's standing relative to the dataset's "
        "distribution (via the StandardScaler that the KMeans model was trained on). "
        "No manual weights, rules, or thresholds — rankings update automatically if the "
        "model is retrained on new data."
    )

    rank_resource = st.selectbox(
        "Rank states by:",
        options=["Overall"] + list(RESOURCE_COLUMNS.keys()),
        index=(["Overall"] + list(RESOURCE_COLUMNS.keys())).index(resource_choice) + 1
        if resource_choice in list(RESOURCE_COLUMNS.keys()) else 0,
    )

    # Sort by the selected metric (descending)
    sorted_df = STATE_RANKINGS.sort_values(by=rank_resource, ascending=False).reset_index(drop=True)
    sorted_df["Rank"] = range(1, len(sorted_df) + 1)

    # Highlight the currently selected state
    sorted_df["Selected"] = sorted_df["state_ut"].apply(
        lambda s: "⭐" if s == state_choice else ""
    )

    # ---- Bar chart of top states ----
    top_n = min(15, len(sorted_df))
    chart_df = sorted_df.head(top_n).copy()

    bar_colors = ["#F5A623" if s == state_choice else "#4A90D9" for s in chart_df["state_ut"]]

    fig_rank = go.Figure(go.Bar(
        x=chart_df[rank_resource],
        y=chart_df["state_ut"],
        orientation="h",
        marker_color=bar_colors,
        text=chart_df[rank_resource].round(1),
        textposition="outside",
    ))
    fig_rank.update_layout(
        title=f"Top {top_n} States by {rank_resource} Potential (selected state highlighted)",
        xaxis_title="Suitability Score (0–100)",
        yaxis=dict(autorange="reversed"),
        height=480,
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    # ---- Full ranking table ----
    st.subheader(f"📋 Complete Ranking — {rank_resource}")
    display_cols = ["Rank", "state_ut", rank_resource, "Selected"]
    if rank_resource == "Overall":
        display_cols = ["Rank", "state_ut", "Overall", "Selected"] + list(RESOURCE_COLUMNS.keys())
    else:
        z_label = f"{rank_resource} z"
        display_cols = ["Rank", "state_ut", rank_resource, z_label, "Selected"]

    st.dataframe(
        sorted_df[display_cols].style.format({
            col: "{:.1f}" for col in sorted_df[display_cols].select_dtypes(include="float64").columns
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ---- State's own ranking summary ----
    st.subheader(f"📍 {state_choice} — Rank Summary")
    rank_cols = st.columns(len(RESOURCE_COLUMNS) + 1)
    metrics = ["Overall"] + list(RESOURCE_COLUMNS.keys())
    for col_widget, metric in zip(rank_cols, metrics):
        metric_sorted = STATE_RANKINGS.sort_values(by=metric, ascending=False).reset_index(drop=True)
        rank_val = int(metric_sorted[metric_sorted["state_ut"] == state_choice].index[0]) + 1
        score = metric_sorted[metric_sorted["state_ut"] == state_choice][metric].values[0]
        with col_widget:
            st.metric(
                label=f"{metric}",
                value=f"#{rank_val}",
                delta=f"{score:.1f}/100",
            )

# ------------------------------------------------------------------
# TAB 6 — STATE COMPARISON (compare two states side by side using
# overall year-round averages: cluster, weather, and resource potential)
# ------------------------------------------------------------------
with tab_scenario:
    st.subheader("🆚 Compare Two States")
    st.caption(
        "Pick any two states to compare their predicted clusters, "
        "weather, and resource potential side by side -- based on overall "
        "year-round averages, using the same trained pipeline as the main analysis above."
    )

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**State A**")
        scen_a_state = st.selectbox("State / UT ", options=states, index=states.index(state_choice), key="scen_a_state")
    with sc2:
        st.markdown("**State B**")
        default_b = next((s for s in states if s != scen_a_state), states[0])
        scen_b_state = st.selectbox("State / UT  ", options=states, index=states.index(default_b), key="scen_b_state")

    def _run_scenario(state):
        m = A["clustered_data"][A["clustered_data"]["state_ut"] == state]
        feats = m[feature_cols].mean()
        cid, _ = predict_cluster(A["scaler"], A["kmeans"], feature_cols, feats)
        return {
            "state": state, "features": feats,
            "cluster": cid, "cluster_name": A["insights"].loc[cid, "cluster_name"],
            "recommendation": A["insights"].loc[cid, "recommendation"],
        }

    scen_a = _run_scenario(scen_a_state)
    scen_b = _run_scenario(scen_b_state)

    r1, r2 = st.columns(2)
    for col_widget, scen in zip((r1, r2), (scen_a, scen_b)):
        with col_widget:
            kpi_card(f"{scen['state']}", scen["cluster_name"], sub=f"Cluster {scen['cluster']}")

    st.markdown("")
    r1, r2 = st.columns(2)
    for col_widget, scen in zip((r1, r2), (scen_a, scen_b)):
        with col_widget:
            st.info(scen["recommendation"])

    st.subheader("Weather Side by Side (Overall Year-Round Averages)")
    weather_rows = []
    for label, (feat, icon) in WEATHER_MAP.items():
        weather_rows.append({
            "Metric": f"{icon} {label}",
            f"{scen_a['state']}": round(scen_a["features"][feat], 2),
            f"{scen_b['state']}": round(scen_b["features"][feat], 2),
        })
    st.dataframe(pd.DataFrame(weather_rows), use_container_width=True, hide_index=True)

    st.subheader("Resource Potential Side by Side")
    resource_compare_rows = []
    for name, col in RESOURCE_COLUMNS.items():
        resource_compare_rows.append({"Resource": name, "State": f"{scen_a['state']}", "Potential": scen_a["features"][col]})
        resource_compare_rows.append({"Resource": name, "State": f"{scen_b['state']}", "Potential": scen_b["features"][col]})
    fig_scenario = px.bar(
        pd.DataFrame(resource_compare_rows), x="Resource", y="Potential", color="State", barmode="group",
        title=f"Resource Potential — {resource_choice} is your resource of interest",
    )
    fig_scenario.update_layout(height=420)
    st.plotly_chart(fig_scenario, use_container_width=True)

    a_val = scen_a["features"][RESOURCE_COLUMNS[resource_choice]]
    b_val = scen_b["features"][RESOURCE_COLUMNS[resource_choice]]
    if a_val == b_val:
        verdict = f"Both states have identical {resource_choice.lower()} potential ({a_val:.2f})."
    else:
        winner, win_val = (scen_a, a_val) if a_val > b_val else (scen_b, b_val)
        loser_val = b_val if winner is scen_a else a_val
        if abs(loser_val) < 1.0:
            verdict = (
                f"**{winner['state']}** has substantially higher "
                f"{resource_choice.lower()} potential ({win_val:.2f} vs {loser_val:.2f})."
            )
        else:
            pct_diff = abs(win_val - loser_val) / abs(loser_val) * 100
            verdict = (
                f"**{winner['state']}** has higher {resource_choice.lower()} potential "
                f"({win_val:.2f} vs {loser_val:.2f}, about {pct_diff:.0f}% more)."
            )
    st.success(f"**Verdict for {resource_choice}:** {verdict}")
 
# ------------------------------------------------------------------
# TAB 7 — DATA & DOWNLOAD
# ------------------------------------------------------------------
with tab_data:
    st.subheader("📁 Dataset Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", A["meta"]["n_records"])
    m2.metric("Number of Features", A["meta"]["n_features"])
    m3.metric("Number of Clusters", A["meta"]["best_k"])
 
    st.subheader("⬇️ Download Clustered Dataset")
    csv_bytes = A["clustered_data"].drop(columns=["pc1", "pc2"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Full Clustered Dataset (CSV)",
        data=csv_bytes,
        file_name="clustered_renewable_energy_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )