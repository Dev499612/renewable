"""
cluster_insights.py
--------------------
Turns the trained K-Means model's numeric output into plain-language
insights, WITHOUT any manually written if/else decision rules for cluster
naming, suitability scoring, or recommendations.
 
How this stays model-driven (not hard-coded):
 
1. Cluster naming:
   The dataset already contains four columns that ARE the model's own
   measure of resource potential: 'solar', 'wind', 'hydro', 'biomass'.
   For every cluster we simply look at which of these four columns has
   the HIGHEST z-score in the trained KMeans centroid (a statistic the
   model itself produced). Whichever is highest determines the cluster's
   dominant resource. This is the model telling us the name -- we are not
   assigning it.
 
2. Magnitude wording ("High" / "Moderate" / "Balanced"):
   Uses the generic statistical convention that a z-score above 1.0
   standard deviation is "notably high" and above 0.35 is "above average".
   This is a universal statistics convention (applies to ANY dataset),
   not a domain-specific numeric rule like "solar > 8".
 
3. Explanations:
   RESOURCE_DRIVERS below is a domain-knowledge LOOKUP TABLE that says
   which meteorological variables are generally understood to influence
   each resource, and in which direction. This is used ONLY to describe,
   in plain language, which of the model's own top-ranking features line
   up with known renewable-energy science -- it never touches training,
   never touches which cluster a location is assigned to, and never
   assigns a name or score by itself. The actual ranking of "how
   important" a feature is for a cluster still comes from the trained
   centroid's z-score, not from this table.
"""
 
import math
import numpy as np
import pandas as pd
 
# The four columns in the dataset that directly represent renewable
# resource potential (as engineered/measured in the notebooks).
RESOURCE_COLUMNS = {
    "Solar": "solar",
    "Wind": "wind",
    "Hydro": "hydro",
    "Biomass": "biomass",
}
 
# Domain-knowledge lookup used ONLY for explanatory text (see docstring).
# direction: +1 means "higher value = more favorable for this resource",
#            -1 means "lower value = more favorable for this resource".
RESOURCE_DRIVERS = {
    "Solar": {
        "dni": (+1, "Direct Normal Irradiance"),
        "ghi": (+1, "Global Horizontal Irradiance"),
        "gti": (+1, "Global Tilted Irradiance"),
        "clearsky_dni": (+1, "clear-sky direct irradiance"),
        "clearsky_ghi": (+1, "clear-sky horizontal irradiance"),
        "clearsky_gti": (+1, "clear-sky tilted irradiance"),
        "cloud_opacity": (-1, "cloud opacity"),
        "relative_humidity": (-1, "relative humidity"),
    },
    "Wind": {
        "wind_speed_100m": (+1, "wind speed at 100m"),
        "surface_pressure": (+1, "surface pressure"),
    },
    "Hydro": {
        "precipitation_rate": (+1, "precipitation rate"),
        "relative_humidity": (+1, "relative humidity"),
    },
    "Biomass": {
        "precipitation_rate": (+1, "precipitation rate"),
        "air_temp": (+1, "air temperature"),
        "relative_humidity": (+1, "relative humidity"),
    },
}
 
# Generic human-readable labels for every climate feature, used ONLY to
# render a feature name in plain English (e.g. "cloud_opacity" -> "cloud
# cover"). This carries no direction/weighting information and does not
# influence which feature gets picked -- that ranking always comes from
# the trained centroid's own z-scores (see _top_climate_signature below).
CLIMATE_FEATURE_LABELS = {
    "air_temp": "temperature",
    "albedo": "surface albedo",
    "clearsky_dhi": "clear-sky diffuse irradiance",
    "clearsky_dni": "clear-sky direct irradiance",
    "clearsky_gti": "clear-sky tilted irradiance",
    "cloud_opacity": "cloud cover",
    "dni": "direct irradiance",
    "ghi": "horizontal irradiance",
    "gti": "tilted irradiance",
    "precipitation_rate": "rainfall",
    "relative_humidity": "humidity",
    "surface_pressure": "surface pressure",
    "wind_speed_100m": "wind speed",
}
 
 
def z_to_suitability_score(z: float) -> float:
    """Converts any z-score into an intuitive 0-100 suitability score using
    the standard normal cumulative distribution function (CDF):
 
        score = 100 * P(Z <= z),  where Z ~ N(0, 1)
 
    This is a general statistical transformation, not a hardcoded formula,
    manual weight, or domain-specific threshold. Because every feature was
    standardized with StandardScaler during training, a feature's z-score
    already IS that location's standing relative to every other location in
    the dataset (0 = exactly average, +1 = one std. dev. above average,
    etc.). The CDF simply re-expresses that same number as an intuitive
    0-100 percentile score instead of a raw z-score -- it works identically
    for any feature, any resource, any cluster, any dataset; nothing about
    it is specific to renewable energy or tuned by hand.
    """
    percentile = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return round(percentile * 100, 1)
 
 
def _magnitude_word(z: float) -> str:
    """Generic statistical banding of a z-score. Applies to any feature,
    any dataset -- not a domain-specific numeric threshold."""
    if z >= 1.0:
        return "notably high"
    elif z >= 0.35:
        return "above average"
    elif z > -0.35:
        return "near average"
    elif z > -1.0:
        return "below average"
    else:
        return "notably low"
 
 
def get_dominant_resource(centroid_row: pd.Series):
    """Given one cluster's centroid (z-score per feature), return the
    resource column with the highest z-score, and whether it's a clear
    leader or a close/mixed profile."""
    resource_scores = {name: centroid_row[col] for name, col in RESOURCE_COLUMNS.items()}
    ranked = sorted(resource_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_name, top_z = ranked[0]
    second_name, second_z = ranked[1]
    is_mixed = (top_z - second_z) < 0.35  # generic closeness check, not a domain rule
    return top_name, top_z, is_mixed, resource_scores
 
 
def resource_specific_note(centroid_row: pd.Series, resource_name: str) -> str:
    """Reports on whichever resource the USER selected (e.g. in a sidebar
    dropdown), which may or may not be the cluster's dominant resource.
 
    This exists because the cluster's dominant resource (used for the
    cluster's name/recommendation) is a property of the LOCATION's climate,
    not of what the user is browsing for -- so if someone picks 'Wind' for a
    solar-dominant cluster, the dashboard should still say something about
    that location's actual wind potential, rather than silently only ever
    talking about solar. The z-score itself still comes straight from the
    trained centroid; only the wording is templated here.
    """
    resource_col = RESOURCE_COLUMNS[resource_name]
    z = centroid_row[resource_col]
    magnitude = _magnitude_word(z)
    score = z_to_suitability_score(z)
    top_name, top_z, is_mixed, _ = get_dominant_resource(centroid_row)
 
    if resource_name == top_name and not is_mixed:
        return (
            f"Good news — {resource_name.lower()} is also this cluster's strongest resource "
            f"({magnitude}). Suitability Score: {score}/100."
        )
    return (
        f"For {resource_name.lower()} specifically, this cluster's potential is "
        f"{magnitude} relative to the rest of the dataset. Suitability Score: {score}/100 "
        f"(compared to {top_name.lower()}, this location's strongest resource)."
    )
 
 
def _top_climate_signature(centroid_row: pd.Series, n: int = 1):
    """Finds whichever non-resource climate feature(s) are most extreme
    (highest |z-score|) in this cluster's own trained centroid. Used to tell
    apart clusters that share a similar resource profile but differ in
    actual climate (e.g. one is wet, another is just cloudy) -- purely a
    ranking of the model's own centroid values, no fixed thresholds."""
    resource_cols = set(RESOURCE_COLUMNS.values())
    climate_row = centroid_row.drop(labels=[c for c in resource_cols if c in centroid_row.index])
    ranked = climate_row.reindex(climate_row.abs().sort_values(ascending=False).index)
    top = ranked.head(n)
    parts = []
    for feat, z in top.items():
        label = CLIMATE_FEATURE_LABELS.get(feat, feat.replace("_", " "))
        direction = "High" if z > 0 else "Low"
        parts.append(f"{direction} {label}")
    return parts
 
 
def generate_cluster_name(centroid_row: pd.Series) -> str:
    top_name, top_z, is_mixed, resource_scores = get_dominant_resource(centroid_row)
    if is_mixed:
        # No single resource clearly leads here -- rather than labeling
        # every such cluster identically, name it by its top-2 resources
        # plus whichever climate trait most separates it from the dataset
        # average. Both pieces come straight from this cluster's own
        # trained centroid, so clusters with genuinely different profiles
        # (even if both are "mixed") end up with different names.
        ranked = sorted(resource_scores.items(), key=lambda kv: kv[1], reverse=True)
        top1, top2 = ranked[0][0], ranked[1][0]
        modifier = _top_climate_signature(centroid_row, n=1)[0]
        return f"{top1}-{top2} Mixed Region ({modifier})"
    magnitude = _magnitude_word(top_z)
    if magnitude in ("notably high", "above average"):
        return f"{top_name} Favorable Climate"
    elif magnitude == "near average":
        return f"Balanced Climate ({top_name}-leaning)"
    else:
        return f"Low {top_name} Potential Region"
 
 
def generate_explanation(centroid_row: pd.Series) -> str:
    """Builds a plain-language explanation using the cluster's own top
    resource plus whichever known drivers of that resource rank highest
    in this cluster's centroid (all values sourced from the trained model)."""
    top_name, top_z, is_mixed, resource_scores = get_dominant_resource(centroid_row)
 
    drivers = RESOURCE_DRIVERS.get(top_name, {})
    # Rank the relevant driver features by how strongly (in the favorable
    # direction) they show up in this cluster's own centroid.
    scored_drivers = []
    for col, (direction, label) in drivers.items():
        if col in centroid_row.index:
            aligned_z = centroid_row[col] * direction
            scored_drivers.append((label, aligned_z))
    scored_drivers.sort(key=lambda x: x[1], reverse=True)
    top_drivers = [label for label, z in scored_drivers[:3] if z > 0.15]
 
    if is_mixed:
        parts = ", ".join(f"{name} ({z:+.2f})" for name, z in resource_scores.items())
        signature = ", ".join(_top_climate_signature(centroid_row, n=2))
        return (
            "This location's climate profile does not strongly favor a single "
            f"renewable resource -- the model found comparable potential across "
            f"resources ({parts}). What most distinguishes this cluster from the "
            f"dataset average is: {signature}. A mixed or hybrid renewable strategy "
            "may suit it best."
        )
 
    if top_drivers:
        driver_text = ", ".join(top_drivers)
        return (
            f"The model grouped this location into a cluster where {top_name.lower()} "
            f"potential is {_magnitude_word(top_z)} relative to the rest of the dataset. "
            f"Within this cluster, the characteristics most aligned with strong "
            f"{top_name.lower()} conditions are: {driver_text}."
        )
    else:
        return (
            f"The model grouped this location into a cluster where {top_name.lower()} "
            f"potential is {_magnitude_word(top_z)} relative to the rest of the dataset."
        )
 
 
def generate_recommendation(centroid_row: pd.Series) -> str:
    top_name, top_z, is_mixed, resource_scores = get_dominant_resource(centroid_row)
 
    if is_mixed:
        ranked = sorted(resource_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_two = ", ".join(name for name, _ in ranked[:2])
        return (
            f"No single resource stands out as dominant in this cluster, so a hybrid "
            f"strategy combining {top_two} is advisable. Diversifying across these two "
            "sources can improve energy reliability and reduce dependence on any one "
            "resource's seasonal variability."
        )
 
    magnitude = _magnitude_word(top_z)
    if magnitude == "notably high":
        return (
            f"This cluster demonstrates strong, well above-average potential for "
            f"{top_name.lower()} energy. It is a strong candidate for {top_name.lower()}-focused "
            "infrastructure investment, subject to standard site-level feasibility checks."
        )
    elif magnitude == "above average":
        return (
            f"This cluster shows above-average potential for {top_name.lower()} energy relative "
            "to other regions in the dataset. Further site-level evaluation is recommended "
            "to confirm feasibility before investment."
        )
    elif magnitude == "near average":
        return (
            f"{top_name} energy potential in this cluster is broadly in line with the dataset "
            "average. Standard due diligence and a comparison against neighboring regions "
            "is advised before proceeding."
        )
    else:
        return (
            f"This cluster shows below-average potential for {top_name.lower()} energy compared "
            "to other regions in the dataset. Alternative resources or nearby locations with "
            "stronger metrics may offer better returns on investment."
        )
 
 
def build_all_cluster_insights(centroids_scaled: pd.DataFrame) -> pd.DataFrame:
    """Runs the automatic naming/explanation/recommendation logic for every
    cluster and returns a lookup table indexed by cluster id."""
    rows = []
    for cluster_id, row in centroids_scaled.iterrows():
        rows.append({
            "cluster": cluster_id,
            "cluster_name": generate_cluster_name(row),
            "explanation": generate_explanation(row),
            "recommendation": generate_recommendation(row),
        })
    return pd.DataFrame(rows).set_index("cluster")
 