import math
import numpy as np
import pandas as pd

# The columns in the dataset that directly represent renewable resource potential
RESOURCE_COLUMNS = {
    "Solar": "solar",
    "Wind": "wind",
    "Hydro": "hydro",
    "Biomass": "biomass",
}

# Meteorological drivers and directions for each resource
RESOURCE_DRIVERS = {
    "Solar": {
        "dni": (+1, "Direct Normal Irradiance"),
        "ghi": (+1, "Global Horizontal Irradiance"),
        "gti": (+1, "Global Tilted Irradiance"),
        "clearsky_dni": (+1, "clear-sky direct irradiance"),
        "clearsky_gti": (+1, "clear-sky tilted irradiance"),
        "cloud_opacity": (-1, "cloud opacity"),
        "relative_humidity": (-1, "relative humidity"),
        "air_temp": (+1, "air temperature"),
        "precipitation_rate": (-1, "precipitation rate"),
    },
    "Wind": {
        "wind": (+1, "wind power potential"),
        "wind_speed_100m": (+1, "wind speed at 100m"),
        "surface_pressure": (+1, "surface pressure"),
        "air_temp": (+1, "air temperature"),
    },
    "Hydro": {
        "hydro": (+1, "hydroelectric potential"),
        "precipitation_rate": (+1, "precipitation rate"),
        "relative_humidity": (+1, "relative humidity"),
        "surface_pressure": (+1, "surface pressure"),
    },
    "Biomass": {
        "biomass": (+1, "biomass potential"),
        "relative_humidity": (+1, "relative humidity"),
        "air_temp": (+1, "air temperature"),
        "precipitation_rate": (+1, "precipitation rate"),
    },
}

# User-friendly descriptions for climate variables
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
    "solar": "solar potential",
    "wind": "wind potential",
    "hydro": "hydro potential",
    "biomass": "biomass potential",
}

# Mappings of climate features to distinct descriptors
CLIMATE_MODIFIERS = {
    "cloud_opacity": {True: "Cloudy", False: "Clear-sky"},
    "relative_humidity": {True: "Humid", False: "Dry"},
    "precipitation_rate": {True: "Rainy", False: "Dry"},
    "air_temp": {True: "Warm", False: "Cool"},
    "surface_pressure": {True: "High-Pressure", False: "Low-Pressure"},
    "wind_speed_100m": {True: "Windy", False: "Calm"},
}


def z_to_suitability_score(z: float) -> float:
    """
    Converts a z-score into a 0-100 percentile suitability score using the 
    Standard Normal Cumulative Distribution Function (CDF).
    """
    percentile = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return round(percentile * 100, 1)


def _magnitude_word(z: float) -> str:
    """
    Generic statistical banding of a z-score to define relative standing.
    """
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


def generate_cluster_name(centroid_row: pd.Series, resource_choice: str, feature_cols: list) -> str:
    """
    Dynamically generates descriptive cluster names based on the resource potential standing
    and the most extreme climate feature z-score (climate modifier).
    """
    target_col = RESOURCE_COLUMNS[resource_choice]
    z_resource = centroid_row.get(target_col, 0.0)
    
    # Determine suitability descriptor
    if z_resource >= 1.0:
        suitability = f"High {resource_choice} Potential"
    elif z_resource >= 0.35:
        suitability = f"Above Average {resource_choice}"
    elif z_resource > -0.35:
        suitability = f"Balanced {resource_choice}"
    elif z_resource > -1.0:
        suitability = f"Below Average {resource_choice}"
    else:
        suitability = f"Low {resource_choice} Potential"
        
    # Find most extreme climate feature (excluding the resource column itself)
    climate_feats = [c for c in feature_cols if c != target_col]
    if not climate_feats:
        return f"{suitability} Region"
        
    # Find the feature that deviates most from the baseline average
    extreme_feat = max(climate_feats, key=lambda c: abs(centroid_row.get(c, 0.0)))
    extreme_z = centroid_row.get(extreme_feat, 0.0)
    
    # Map to descriptive modifier
    modifier = ""
    if extreme_feat in CLIMATE_MODIFIERS:
        is_positive = extreme_z > 0
        modifier = CLIMATE_MODIFIERS[extreme_feat][is_positive]
    else:
        label = CLIMATE_FEATURE_LABELS.get(extreme_feat, extreme_feat.replace("_", " "))
        direction = "Rich" if extreme_z > 0 else "Low"
        modifier = f"{label.title()} {direction}"
        
    return f"{suitability} Climate ({modifier})"


def generate_explanation(centroid_row: pd.Series, resource_choice: str, feature_cols: list) -> str:
    """
    Builds a plain-language explanation of the cluster using the cluster's own target potential
    and the primary driver features aligned with renewable energy principles.
    """
    target_col = RESOURCE_COLUMNS[resource_choice]
    z = centroid_row.get(target_col, 0.0)
    magnitude = _magnitude_word(z)
    score = z_to_suitability_score(z)
    
    drivers = RESOURCE_DRIVERS.get(resource_choice, {})
    scored_drivers = []
    for col, (direction, label) in drivers.items():
        if col in centroid_row.index and col != target_col:
            aligned_z = centroid_row[col] * direction
            scored_drivers.append((label, aligned_z))
            
    scored_drivers.sort(key=lambda x: x[1], reverse=True)
    
    # Extract top positive and negative drivers relative to what is expected for high suitability
    top_positive = [label for label, val in scored_drivers if val > 0.2]
    top_negative = [label for label, val in scored_drivers[::-1] if val < -0.2]
    
    explanation = (
        f"The K-Means model automatically grouped this location into a climate zone where "
        f"{resource_choice.lower()} potential is {magnitude} (Suitability Score: {score}/100) "
        f"relative to the overall dataset."
    )
    
    driver_desc = []
    if top_positive:
        driver_desc.append(f"favorable factors such as higher {', '.join(top_positive[:3])}")
    if top_negative:
        driver_desc.append(f"constraining factors like unfavorable {', '.join(top_negative[:2])}")
        
    if driver_desc:
        explanation += " This assignment is driven by " + " and ".join(driver_desc) + "."
    else:
        explanation += " The weather attributes in this cluster are generally close to the average baseline."
        
    return explanation


def generate_recommendation(centroid_row: pd.Series, resource_choice: str) -> str:
    """
    Generates recommendations based on the cluster's resource suitability level.
    """
    target_col = RESOURCE_COLUMNS[resource_choice]
    z = centroid_row.get(target_col, 0.0)
    
    if z >= 1.0:
        return (
            f"This region is highly recommended for {resource_choice.lower()} development. "
            f"It demonstrates exceptionally high potential, making it a primary candidate "
            f"for commercial-scale infrastructure investment and detailed site assessments."
        )
    elif z >= 0.35:
        return (
            f"This region is moderately recommended for {resource_choice.lower()} development. "
            f"It shows above-average potential. Localized environmental studies and energy yield "
            f"simulations should be conducted to verify site viability."
        )
    elif z > -0.35:
        return (
            f"This region has balanced or average potential for {resource_choice.lower()} energy. "
            f"Consider a hybrid energy configuration (e.g. co-locating with other resources) "
            f"or executing a feasibility study comparing it to neighboring higher-potential sites."
        )
    elif z > -1.0:
        return (
            f"This region has below-average potential for {resource_choice.lower()} energy. "
            f"Standard standalone commercial installations are unlikely to be optimal. "
            f"We recommend checking alternative resources or looking at adjacent regions."
        )
    else:
        return (
            f"This region is not recommended for standalone {resource_choice.lower()} projects due to "
            f"significantly below-average resource profiles. Investment should focus on other, "
            f"more abundant renewable resources in the area."
        )


def resource_specific_note(centroid_row: pd.Series, resource_choice: str) -> str:
    """
    Returns a detailed resource suitability note derived from the cluster's trained centroid z-score.
    """
    target_col = RESOURCE_COLUMNS[resource_choice]
    z = centroid_row.get(target_col, 0.0)
    magnitude = _magnitude_word(z)
    score = z_to_suitability_score(z)
    
    return (
        f"The cluster centroid indicates that {resource_choice.lower()} potential is {magnitude} "
        f"with a standardized z-score of {z:+.2f}. This translates to a Suitability Score "
        f"of {score}/100 relative to the overall dataset distribution."
    )


def build_all_cluster_insights(centroids_scaled: pd.DataFrame, resource_choice: str, feature_cols: list) -> pd.DataFrame:
    """
    Runs the automatic naming/explanation/recommendation logic for every cluster
    in the selected resource model and returns a lookup table indexed by cluster id.
    """
    rows = []
    for cluster_id, row in centroids_scaled.iterrows():
        rows.append({
            "cluster": cluster_id,
            "cluster_name": generate_cluster_name(row, resource_choice, feature_cols),
            "explanation": generate_explanation(row, resource_choice, feature_cols),
            "recommendation": generate_recommendation(row, resource_choice),
        })
    return pd.DataFrame(rows).set_index("cluster")

 