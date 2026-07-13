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
Cluster Comparison / Data) so the dashboard feels like an app rather than
one long scroll. The prediction pipeline itself (scaler.transform ->
kmeans.predict) is unchanged from the original version.
 
Run:
    streamlit run app.py
"""
 
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
 
from cluster_insights import (
    build_all_cluster_insights, RESOURCE_COLUMNS, resource_specific_note, z_to_suitability_score,
)
 
ARTIFACTS_DIR = "artifacts"
MONTH_ORDER = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
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
 
available_months = [m for m in MONTH_ORDER if m in A["clustered_data"].loc[
    A["clustered_data"]["state_ut"] == state_choice, "month"].unique()]
month_choice = st.sidebar.selectbox("Month", options=available_months)
 
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
    "and month in the sidebar, then click **Analyze Site** -- all insights below are "
    "derived automatically from the trained model, not manual rules."
)
st.divider()
 
if not st.session_state.analyzed:
    st.info("👈 Choose a state and month in the sidebar, then click **Analyze Site** to begin.")
    st.stop()
 
# --------------------------------------------------------------------------
# RUN THE TRAINED PIPELINE FOR THE SELECTED LOCATION
# --------------------------------------------------------------------------
feature_cols = A["feature_cols"]
match = A["clustered_data"][
    (A["clustered_data"]["state_ut"] == state_choice) &
    (A["clustered_data"]["month"] == month_choice)
]
if match.empty:
    st.warning("No data found for that combination. Try a different month.")
    st.stop()
 
location_features = match[feature_cols].mean()  # average across available years
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
 
 
# ============================================================================
# TABS -- keeps the dashboard interactive/navigable instead of one long scroll
# ============================================================================
tab_overview, tab_map, tab_explain, tab_compare, tab_scenario, tab_data = st.tabs(
    ["🏠 Overview", "🗺️ India Map", "🔎 Explainability", "📈 Cluster Comparison",
     "🆚 Scenario Comparison", "📁 Data & Download"]
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
        kpi_card("Location", state_choice, sub=month_choice)
    with c3:
        kpi_card("Resource of Interest", resource_choice, sub="selected in sidebar")
    with c4:
        kpi_card(f"{resource_choice} Suitability Score", f"{suitability_score}/100",
                  sub=f"this location's own percentile, z={location_resource_z:+.2f}")
    st.success(f"**Recommendation:** {recommendation}")

    resource_note = resource_specific_note(A["centroids_scaled"].loc[predicted_cluster], resource_choice)
    st.info(f"**About {resource_choice} here:** {resource_note}")
 
    st.subheader("🌤️ Weather Summary")
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
        title=f"Resource Potential at {state_choice} ({month_choice}) — {resource_choice} highlighted",
        yaxis_title="Potential (dataset units)",
    )
    fig_resource.update_layout(height=400)
    st.plotly_chart(fig_resource, use_container_width=True)
    st.caption("Want to compare this against a different state or month? See the 🆚 **Scenario Comparison** tab.")
 
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
    loc_row = match.iloc[0]
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
 
    st.subheader("PCA Projection (all individual records)")
    scatter_df = A["clustered_data"][["pc1", "pc2", "cluster", "state_ut", "month"]].copy()
    scatter_df["cluster"] = scatter_df["cluster"].astype(str)
    fig_pca = px.scatter(
        scatter_df, x="pc1", y="pc2", color="cluster",
        hover_data=["state_ut", "month"], opacity=0.5,
        color_discrete_map=color_map,
        title="All Locations Projected to 2D (PCA) — Colored by Cluster",
    )
    fig_pca.add_trace(go.Scatter(
        x=[pc_coords_new[0]], y=[pc_coords_new[1]],
        mode="markers", marker=dict(size=18, color="black", symbol="star"),
        name=f"{state_choice} ({month_choice})",
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
# TAB 5 — SCENARIO COMPARISON (compare two full state+month scenarios
# side by side: cluster, weather, and resource potential for each)
# ------------------------------------------------------------------
with tab_scenario:
    st.subheader("🆚 Compare Two Scenarios")
    st.caption(
        "Pick any two state+month combinations to compare their predicted clusters, "
        "weather, and resource potential side by side -- each runs through the same "
        "trained pipeline as the main analysis above."
    )

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**Scenario A**")
        scen_a_state = st.selectbox("State / UT ", options=states, index=states.index(state_choice), key="scen_a_state")
        scen_a_months = [m for m in MONTH_ORDER if m in A["clustered_data"].loc[
            A["clustered_data"]["state_ut"] == scen_a_state, "month"].unique()]
        scen_a_month = st.selectbox("Month ", options=scen_a_months,
                                     index=scen_a_months.index(month_choice) if month_choice in scen_a_months else 0,
                                     key="scen_a_month")
    with sc2:
        st.markdown("**Scenario B**")
        default_b = next((s for s in states if s != scen_a_state), states[0])
        scen_b_state = st.selectbox("State / UT  ", options=states, index=states.index(default_b), key="scen_b_state")
        scen_b_months = [m for m in MONTH_ORDER if m in A["clustered_data"].loc[
            A["clustered_data"]["state_ut"] == scen_b_state, "month"].unique()]
        scen_b_month = st.selectbox("Month  ", options=scen_b_months, index=0, key="scen_b_month")

    def _run_scenario(state, month):
        m = A["clustered_data"][(A["clustered_data"]["state_ut"] == state) & (A["clustered_data"]["month"] == month)]
        feats = m[feature_cols].mean()
        cid, _ = predict_cluster(A["scaler"], A["kmeans"], feature_cols, feats)
        return {
            "state": state, "month": month, "features": feats,
            "cluster": cid, "cluster_name": A["insights"].loc[cid, "cluster_name"],
            "recommendation": A["insights"].loc[cid, "recommendation"],
        }

    scen_a = _run_scenario(scen_a_state, scen_a_month)
    scen_b = _run_scenario(scen_b_state, scen_b_month)

    r1, r2 = st.columns(2)
    for col_widget, scen in zip((r1, r2), (scen_a, scen_b)):
        with col_widget:
            kpi_card(f"{scen['state']} ({scen['month']})", scen["cluster_name"], sub=f"Cluster {scen['cluster']}")

    st.markdown("")
    r1, r2 = st.columns(2)
    for col_widget, scen in zip((r1, r2), (scen_a, scen_b)):
        with col_widget:
            st.info(scen["recommendation"])

    st.subheader("Weather Side by Side")
    weather_rows = []
    for label, (feat, icon) in WEATHER_MAP.items():
        weather_rows.append({
            "Metric": f"{icon} {label}",
            f"{scen_a['state']} ({scen_a['month']})": round(scen_a["features"][feat], 2),
            f"{scen_b['state']} ({scen_b['month']})": round(scen_b["features"][feat], 2),
        })
    st.dataframe(pd.DataFrame(weather_rows), use_container_width=True, hide_index=True)

    st.subheader("Resource Potential Side by Side")
    resource_compare_rows = []
    for name, col in RESOURCE_COLUMNS.items():
        resource_compare_rows.append({"Resource": name, "Scenario": f"{scen_a['state']} ({scen_a['month']})", "Potential": scen_a["features"][col]})
        resource_compare_rows.append({"Resource": name, "Scenario": f"{scen_b['state']} ({scen_b['month']})", "Potential": scen_b["features"][col]})
    fig_scenario = px.bar(
        pd.DataFrame(resource_compare_rows), x="Resource", y="Potential", color="Scenario", barmode="group",
        title=f"Resource Potential — {resource_choice} is your resource of interest",
    )
    fig_scenario.update_layout(height=420)
    st.plotly_chart(fig_scenario, use_container_width=True)

    # A short, fully data-derived verdict for whichever resource is
    # selected in the sidebar -- no fixed thresholds, just a direct
    # comparison of the two scenarios' own trained-feature values.
    a_val = scen_a["features"][RESOURCE_COLUMNS[resource_choice]]
    b_val = scen_b["features"][RESOURCE_COLUMNS[resource_choice]]
    if a_val == b_val:
        verdict = f"Both scenarios have identical {resource_choice.lower()} potential ({a_val:.2f})."
    else:
        winner, win_val = (scen_a, a_val) if a_val > b_val else (scen_b, b_val)
        loser_val = b_val if winner is scen_a else a_val
        if abs(loser_val) < 1.0:
            # Baseline too close to zero for a meaningful percentage --
            # show the plain values instead of a distorted/huge percentage.
            verdict = (
                f"**{winner['state']} ({winner['month']})** has substantially higher "
                f"{resource_choice.lower()} potential ({win_val:.2f} vs {loser_val:.2f})."
            )
        else:
            pct_diff = abs(win_val - loser_val) / abs(loser_val) * 100
            verdict = (
                f"**{winner['state']} ({winner['month']})** has higher {resource_choice.lower()} potential "
                f"({win_val:.2f} vs {loser_val:.2f}, about {pct_diff:.0f}% more)."
            )
    st.success(f"**Verdict for {resource_choice}:** {verdict}")
 
# ------------------------------------------------------------------
# TAB 6 — DATA & DOWNLOAD
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
 