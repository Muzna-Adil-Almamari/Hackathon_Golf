
import streamlit as st
import folium
from folium.plugins import HeatMap
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import random
import os
from streamlit_folium import st_folium
import io
from fpdf import FPDF
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timezone
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import networkx as nx
import requests

# =============================
# Load Oman boundary polygon
# =============================
oman_boundary = gpd.read_file(
    r"C:\Users\bbuser\Desktop\mapproject\Hackathon_Golf\main\geoBoundaries-OMN-ADM1.geojson" 
)
oman_boundary = oman_boundary.to_crs(epsg=4326)

# =============================
# Config
# =============================
st.set_page_config(page_title="5G Tower Planner -- Advanced Algorithms", layout="wide")

OMAN_POP_FILE_CSV = "locatin.csv"
MIN_LAT, MAX_LAT = 16.6, 26.4
MIN_LON, MAX_LON = 51.9, 59.9

# =============================
# Location and Algorithm Selection
# =============================
def get_coordinates_from_place(place_name):
    """Get coordinates from place name using OpenStreetMap Nominatim API"""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place_name + ", Oman", "format": "json", "limit": 1}
    headers = {"User-Agent": "5GTowerPlanner/1.0"}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return lat, lon, True, None
        else:
            return None, None, False, "Place not found"
    except requests.exceptions.RequestException as e:
        return None, None, False, f"Error occurred: {str(e)}"

def determine_best_algorithm(place_name, population_df):
    """Always use Hotspot Detection algorithm for optimal population coverage"""
    return "Hotspot Detection", "Using Enhanced Hotspot Detection algorithm - guarantees placement of exact number of requested towers while prioritizing highest population density areas"

# =============================
# Helpers
# =============================
def generate_mock_population_oman(num_points=2000):
    random.seed(42)
    data = []
    for _ in range(num_points):
        lat = random.uniform(MIN_LAT, MAX_LAT)
        lon = random.uniform(MIN_LON, MAX_LON)
        density = random.randint(5, 500)
        data.append([lat, lon, density])
    df = pd.DataFrame(data, columns=["lat", "lon", "density"])
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    )
    gdf_within = gpd.sjoin(gdf, oman_boundary, predicate="within")
    return gdf_within.drop(columns=["geometry", "index_right"]).reset_index(drop=True)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2.0)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2.0)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def population_covered(towers, population_df, radius_m):
    if len(population_df) == 0 or len(towers) == 0:
        return 0.0
    pop_gdf = gpd.GeoDataFrame(
        population_df,
        geometry=gpd.points_from_xy(population_df.lon, population_df.lat),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    total_covered = 0.0
    covered_points = set()
    
    for t in towers:
        lon, lat = t[1], t[0]
        tower_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)
        buffer = tower_geom.buffer(radius_m).iloc[0]
        covered = pop_gdf[pop_gdf.geometry.within(buffer)]
        
        # Avoid double counting
        for idx in covered.index:
            if idx not in covered_points:
                covered_points.add(idx)
                total_covered += float(covered.loc[idx, "density"])
    
    return total_covered


# =============================
# ALGORITHM 1: Enhanced Greedy (Original improved)
# =============================
def greedy_tower_placement(pop_df, num_towers, radius_m):
    if len(pop_df) == 0:
        return []
    remaining = pop_df.copy().reset_index(drop=True)
    towers = []
    radius_deg = radius_m / 111320.0
    
    for _ in range(num_towers):
        best_idx = None
        best_score = -1
        
        for idx, row in remaining.iterrows():
            lat0, lon0 = row.lat, row.lon
            candidates = remaining[
                (remaining.lat >= lat0 - radius_deg) & (remaining.lat <= lat0 + radius_deg) &
                (remaining.lon >= lon0 - radius_deg) & (remaining.lon <= lon0 + radius_deg)
            ]
            if candidates.empty:
                continue
            
            dists = candidates.apply(lambda r: haversine(lat0, lon0, r.lat, r.lon), axis=1)
            covered_pop = candidates.loc[dists <= radius_m, "density"].sum()
            
            if covered_pop > best_score:
                best_score = covered_pop
                best_idx = idx
        
        if best_idx is None:
            break
            
        chosen = remaining.loc[best_idx]
        tower_point = (float(chosen.lat), float(chosen.lon))
        towers.append(tower_point)
        
        # Remove covered population
        dists_all = remaining.apply(lambda r: haversine(tower_point[0], tower_point[1], r.lat, r.lon), axis=1)
        remaining = remaining.loc[dists_all > radius_m].reset_index(drop=True)
        
        if remaining.empty:
            break
    
    return towers


# =============================
# ALGORITHM 2: Population-Weighted K-Means Clustering
# =============================
def kmeans_tower_placement(pop_df, num_towers, radius_m):
    if len(pop_df) == 0 or num_towers == 0:
        return []
    
    # Create weighted points based on population density
    weighted_points = []
    weights = []
    
    for _, row in pop_df.iterrows():
        # Add multiple copies of high-density points
        weight = max(1, int(row.density / 50))  # Scale factor
        for _ in range(min(weight, 10)):  # Cap to avoid memory issues
            weighted_points.append([row.lat, row.lon])
            weights.append(row.density)
    
    if len(weighted_points) < num_towers:
        # Fallback to original points
        weighted_points = pop_df[['lat', 'lon']].values.tolist()
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_towers, random_state=42, n_init=10)
    cluster_centers = kmeans.fit(weighted_points).cluster_centers_
    
    towers = [(float(center[0]), float(center[1])) for center in cluster_centers]
    return towers


# =============================
# ALGORITHM 3: Population Density Hotspot Detection (Fixed to place exact number of towers)
# =============================
def hotspot_tower_placement(pop_df, num_towers, radius_m):
    if len(pop_df) == 0 or num_towers == 0:
        return []
    
    towers = []
    remaining_pop = pop_df.copy()
    original_pop = pop_df.copy()
    
    for tower_num in range(num_towers):
        if remaining_pop.empty:
            # If no uncovered population left, find best locations from original data
            # that are not too close to existing towers
            best_location = None
            best_score = -1
            min_distance_to_existing = radius_m * 0.8  # Minimum 80% of radius between towers
            
            for _, candidate in original_pop.iterrows():
                candidate_point = (candidate.lat, candidate.lon)
                
                # Check distance to existing towers
                too_close = False
                for existing_tower in towers:
                    distance = haversine(candidate_point[0], candidate_point[1], 
                                       existing_tower[0], existing_tower[1])
                    if distance < min_distance_to_existing:
                        too_close = True
                        break
                
                if not too_close:
                    # Calculate potential coverage for this location
                    potential_coverage = 0
                    for _, pop_point in original_pop.iterrows():
                        dist = haversine(candidate.lat, candidate.lon, pop_point.lat, pop_point.lon)
                        if dist <= radius_m:
                            potential_coverage += pop_point.density
                    
                    if potential_coverage > best_score:
                        best_score = potential_coverage
                        best_location = candidate_point
            
            # If we found a good location, use it
            if best_location:
                towers.append(best_location)
            else:
                # As last resort, place tower at a random location from original data
                # that maintains minimum distance from existing towers
                attempts = 0
                while attempts < 100:  # Limit attempts to avoid infinite loop
                    candidate = original_pop.sample(n=1).iloc[0]
                    candidate_point = (candidate.lat, candidate.lon)
                    
                    # Check minimum distance to existing towers
                    valid_location = True
                    for existing_tower in towers:
                        distance = haversine(candidate_point[0], candidate_point[1], 
                                           existing_tower[0], existing_tower[1])
                        if distance < min_distance_to_existing:
                            valid_location = False
                            break
                    
                    if valid_location:
                        towers.append(candidate_point)
                        break
                    attempts += 1
                
                # If still no valid location found, place with reduced minimum distance
                if len(towers) < tower_num + 1:
                    candidate = original_pop.sample(n=1).iloc[0]
                    towers.append((candidate.lat, candidate.lon))
        else:
            # Standard hotspot detection: find highest density uncovered area
            max_density_idx = remaining_pop.density.idxmax()
            hotspot = remaining_pop.loc[max_density_idx]
            
            tower_point = (float(hotspot.lat), float(hotspot.lon))
            towers.append(tower_point)
            
            # Remove points within coverage radius from remaining population
            distances = remaining_pop.apply(
                lambda row: haversine(hotspot.lat, hotspot.lon, row.lat, row.lon), axis=1
            )
            remaining_pop = remaining_pop[distances > radius_m].reset_index(drop=True)
    
    return towers


# =============================
# ALGORITHM 4: Genetic Algorithm for Tower Placement
# =============================
def genetic_algorithm_tower_placement(pop_df, num_towers, radius_m, generations=50, population_size=20):
    if len(pop_df) == 0 or num_towers == 0:
        return []
    
    def fitness_function(towers):
        return population_covered(towers, pop_df, radius_m)
    
    def create_individual():
        return random.sample([(row.lat, row.lon) for _, row in pop_df.iterrows()], 
                           min(num_towers, len(pop_df)))
    
    def mutate(individual, mutation_rate=0.1):
        if random.random() < mutation_rate:
            # Replace random tower with random location
            if len(individual) > 0:
                idx = random.randint(0, len(individual) - 1)
                new_point = random.choice([(row.lat, row.lon) for _, row in pop_df.iterrows()])
                individual[idx] = new_point
        return individual
    
    def crossover(parent1, parent2):
        # Single point crossover
        if len(parent1) <= 1:
            return parent1[:]
        point = random.randint(1, len(parent1) - 1)
        child = parent1[:point] + parent2[point:]
        return child[:num_towers]
    
    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    
    # Evolution
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [(individual, fitness_function(individual)) for individual in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep best half
        population = [ind for ind, _ in fitness_scores[:population_size // 2]]
        
        # Generate new individuals
        while len(population) < population_size:
            if len(population) >= 2:
                parent1 = random.choice(population[:5])  # Select from best
                parent2 = random.choice(population[:5])
                child = crossover(parent1, parent2)
                child = mutate(child)
                population.append(child)
            else:
                population.append(create_individual())
    
    # Return best solution
    final_fitness = [(individual, fitness_function(individual)) for individual in population]
    best_solution = max(final_fitness, key=lambda x: x[1])[0]
    
    return best_solution


# =============================
# ALGORITHM 5: Simulated Annealing
# =============================
def simulated_annealing_tower_placement(pop_df, num_towers, radius_m, max_iterations=1000):
    if len(pop_df) == 0 or num_towers == 0:
        return []
    
    # Initial solution (random)
    current_solution = random.sample([(row.lat, row.lon) for _, row in pop_df.iterrows()], 
                                   min(num_towers, len(pop_df)))
    current_fitness = population_covered(current_solution, pop_df, radius_m)
    
    best_solution = current_solution[:]
    best_fitness = current_fitness
    
    initial_temp = 10000
    final_temp = 1
    alpha = 0.95
    
    temperature = initial_temp
    
    for iteration in range(max_iterations):
        # Generate neighbor solution
        neighbor = current_solution[:]
        if len(neighbor) > 0:
            # Randomly modify one tower location
            idx = random.randint(0, len(neighbor) - 1)
            new_point = random.choice([(row.lat, row.lon) for _, row in pop_df.iterrows()])
            neighbor[idx] = new_point
        
        neighbor_fitness = population_covered(neighbor, pop_df, radius_m)
        
        # Accept or reject
        delta = neighbor_fitness - current_fitness
        if delta > 0 or (temperature > 0 and random.random() < np.exp(delta / temperature)):
            current_solution = neighbor
            current_fitness = neighbor_fitness
            
            if current_fitness > best_fitness:
                best_solution = current_solution[:]
                best_fitness = current_fitness
        
        # Cool down
        temperature *= alpha
        if temperature < final_temp:
            temperature = final_temp
    
    return best_solution


# ==============
# Make PDF Report
# ================
def make_pdf_report(towers, radius_m, total_pop, covered_pop, algorithm_name):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "5G Tower Placement Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", ln=True)
    pdf.cell(0, 8, f"Algorithm Used: {algorithm_name}", ln=True)
    pdf.ln(4)
    pdf.cell(0, 8, f"Total population in selected dataset: {int(total_pop):,}", ln=True)
    pdf.cell(0, 8, f"Total population covered by proposed towers: {int(covered_pop):,}", ln=True)
    pct = (covered_pop / total_pop * 100) if total_pop > 0 else 0.0
    pdf.cell(0, 8, f"Coverage percentage: {pct:.2f}%", ln=True)
    pdf.ln(8)
    for i, t in enumerate(towers, start=1):
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"Tower {i}", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 6, f"  Latitude : {t[0]:.6f}", ln=True)
        pdf.cell(0, 6, f"  Longitude: {t[1]:.6f}", ln=True)
        pdf.cell(0, 6, f"  Coverage radius (m): {radius_m}", ln=True)
        pdf.ln(4)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_buf = io.BytesIO(pdf_bytes)
    pdf_buf.seek(0)
    return pdf_buf


def load_population_csv_or_mock():
    if os.path.exists(OMAN_POP_FILE_CSV):
        try:
            df = pd.read_csv(OMAN_POP_FILE_CSV)
            if "lat" in df.columns and "lon" in df.columns:
                pop_col = "population" if "population" in df.columns else ("density" if "density" in df.columns else None)
                if pop_col is None:
                    st.error("CSV found but no 'density' or 'population' column. Using mock data.")
                    return generate_mock_population_oman()
                df = df[["lat", "lon", pop_col]].copy()
                df.columns = ["lat", "lon", "density"]
            else:
                st.warning("CSV missing 'lat'/'lon'. Using mock data.")
                df = generate_mock_population_oman()
        except Exception as e:
            st.warning(f"Failed to read CSV ({e}). Using mock data.")
            df = generate_mock_population_oman()
    else:
        df = generate_mock_population_oman()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    gdf_within = gpd.sjoin(gdf, oman_boundary, predicate="within")
    return gdf_within.drop(columns=["geometry", "index_right"]).reset_index(drop=True)


# =============================
# UI -- Title
# =============================
# =============================
# UI -- Title
# =============================
st.title("📡 5G Tower Planner -- Smart Location-Based Placement")

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("📍 Location Selection")
    
    # Location input methods
    location_method = st.radio(
        "Choose location method:",
        ["Search by place name", "Manual coordinates", "Bounding box"]
    )
    
    target_lat, target_lon, location_found, location_error = None, None, False, None
    
    if location_method == "Search by place name":
        place_name = st.text_input("Enter place name in Oman:", value="Salalah")
        
        if st.button("🔍 Find Location"):
            if place_name.strip():
                with st.spinner("Searching for location..."):
                    target_lat, target_lon, location_found, location_error = get_coordinates_from_place(place_name.strip())
                
                if location_found:
                    st.success(f"✅ Found {place_name}: {target_lat:.4f}, {target_lon:.4f}")
                    st.session_state["target_location"] = (target_lat, target_lon, place_name)
                else:
                    st.error(f"❌ {location_error}")
            else:
                st.warning("Please enter a place name")
    
    elif location_method == "Manual coordinates":
        target_lat = st.number_input("Target Latitude", value=17.0150, format="%.6f")
        target_lon = st.number_input("Target Longitude", value=54.0924, format="%.6f")
        place_name = st.text_input("Location name (optional):", value="Custom Location")
        
        if st.button("📍 Set Location"):
            st.session_state["target_location"] = (target_lat, target_lon, place_name)
            st.success(f"✅ Location set: {target_lat:.4f}, {target_lon:.4f}")
    
    # Show current target location
    if "target_location" in st.session_state:
        loc_lat, loc_lon, loc_name = st.session_state["target_location"]
        st.info(f"📌 Current target: **{loc_name}** ({loc_lat:.4f}, {loc_lon:.4f})")
    
    st.header("Radio & Constraints")
    num_towers = st.slider("# Towers (K)", 1, 30, 5)
    freq_mhz = st.select_slider("Frequency (MHz)", options=[700, 1800, 2100, 2600, 3500], value=3500)
    tx_power_dbm = st.slider("Tx Power (dBm EIRP)", 30, 60, 46)
    min_rsrp_dbm = st.slider("Coverage Threshold (dBm)", -120, -70, -100)
    overlap_penalty = st.slider("Overlap penalty λ", 0.0, 1.0, 0.5, 0.1)

    profile = st.selectbox("Propagation Profile", ["Urban", "Rural"])
    pathloss_exp = 3.2 if profile == "Urban" else 2.6

    cell_size_m = st.slider("Cell size (meters)", 50, 500, 100, 10)

    st.header("Population Grid Overlay")
    show_pop = st.checkbox("Show Population Grid Layer", True)
    pop_alpha = st.slider("Population layer opacity (0–255)", 0, 255, 150)
    pop_step = st.slider("Population sampling step (cells)", 1, 10, 1)

    st.markdown("**Analysis Region (around target location)**")
    if location_method == "Bounding box":
        lat_min = st.number_input("Min latitude", value=float(MIN_LAT))
        lat_max = st.number_input("Max latitude", value=float(MAX_LAT))
        lon_min = st.number_input("Min longitude", value=float(MIN_LON))
        lon_max = st.number_input("Max longitude", value=float(MAX_LON))
    else:
        # Auto-calculate region around target location
        region_radius_km = st.slider("Analysis radius (km)", 5, 100, 25)
        
        if "target_location" in st.session_state:
            loc_lat, loc_lon, _ = st.session_state["target_location"]
            # Convert km to approximate degrees
            lat_offset = region_radius_km / 111.0
            lon_offset = region_radius_km / (111.0 * cos(radians(loc_lat)))
            
            lat_min = max(MIN_LAT, loc_lat - lat_offset)
            lat_max = min(MAX_LAT, loc_lat + lat_offset)
            lon_min = max(MIN_LON, loc_lon - lon_offset)
            lon_max = min(MAX_LON, loc_lon + lon_offset)
            
            st.write(f"Region: {lat_min:.3f} to {lat_max:.3f} lat, {lon_min:.3f} to {lon_max:.3f} lon")
        else:
            lat_min, lat_max = MIN_LAT, MAX_LAT
            lon_min, lon_max = MIN_LON, MAX_LON
    use_mock = st.checkbox("Use mock (Oman-wide) population", value=True)
    points_count = st.number_input("Number of mock points", value=2000, min_value=200, max_value=20000, step=100)

# =============================
# Load population (always, to show map)
# =============================
if use_mock:
    base_population_df = generate_mock_population_oman(num_points=int(points_count))
else:
    base_population_df = load_population_csv_or_mock()

# Filter by bounding box
base_population_df = base_population_df[
    (base_population_df.lat >= lat_min) & (base_population_df.lat <= lat_max) &
    (base_population_df.lon >= lon_min) & (base_population_df.lon <= lon_max)
].reset_index(drop=True)

total_population_current = float(base_population_df.density.sum())
map_center = ((lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0)

# Parameter signature to detect changes since last run
current_params_key = (
    float(lat_min), float(lat_max), float(lon_min), float(lon_max),
    int(num_towers), float(cell_size_m),
    bool(use_mock), int(points_count) if use_mock else -1,
    str(st.session_state.get("target_location", ""))
)

# Notify if the UI parameters changed since the last optimization
if "params_key" in st.session_state and st.session_state.get("params_key") != current_params_key:
    st.info("Parameters changed. Click 'Run Optimization' to update tower placement and metrics.")

# Algorithm description
if "target_location" in st.session_state:
    loc_name = st.session_state["target_location"][2]
    st.success(f"🎯 **Target Location**: {loc_name}")
    st.info("🔥 **Using Enhanced Hotspot Detection Algorithm** - This algorithm guarantees placement of the exact number of towers you specify while prioritizing the highest population density areas for maximum coverage efficiency.")

st.markdown("**Algorithm Guarantee**: The system will place exactly the number of towers you specify, even if it means placing additional towers in lower-density areas to meet your requirements.")
st.markdown("When ready, click the button below to compute tower placement using **Enhanced Hotspot Detection**.")
run_clicked = st.button("🚀 Run Enhanced Hotspot Detection")

# =============================
# Run optimization on demand
# =============================
if run_clicked and "target_location" in st.session_state:
    loc_name = st.session_state["target_location"][2]
    
    # Determine the best algorithm based on location and population data
    algorithm, algorithm_reason = determine_best_algorithm(loc_name, base_population_df)
    
    st.info(f"🧠 **Selected Algorithm**: {algorithm}")
    st.write(f"**Reason**: {algorithm_reason}")
    
    with st.spinner(f"Running Hotspot Detection algorithm for {loc_name}..."):
        # Always use Hotspot Detection algorithm
        towers = hotspot_tower_placement(base_population_df, num_towers=int(num_towers), radius_m=float(cell_size_m))
        
        covered_population = population_covered(towers, base_population_df, radius_m=float(cell_size_m))

        # Persist results and context
        st.session_state["towers"] = towers
        st.session_state["covered_population"] = float(covered_population)
        st.session_state["total_population"] = float(total_population_current)
        st.session_state["coverage_radius"] = float(cell_size_m)
        st.session_state["algorithm_used"] = algorithm
        st.session_state["algorithm_reason"] = algorithm_reason
        st.session_state["params_key"] = current_params_key
        
        st.success(f"✅ Enhanced Hotspot Detection complete! Successfully placed all {len(towers)} towers in {loc_name} (exactly {num_towers} as requested).")
        
        # Verification message
        if len(towers) == num_towers:
            st.info(f"🎯 **Tower Count Verification**: ✅ Placed {len(towers)} out of {num_towers} requested towers (100% success)")
        else:
            st.warning(f"⚠️ **Tower Count**: Placed {len(towers)} out of {num_towers} requested towers")

elif run_clicked and "target_location" not in st.session_state:
    st.error("❌ Please select a target location first!")

# =============================
# Always show the map (heatmap + boundary). Add towers only if results match current params.
# =============================
# Determine map center
if "target_location" in st.session_state:
    map_center_lat, map_center_lon, _ = st.session_state["target_location"]
    zoom_level = 10
else:
    map_center_lat, map_center_lon = map_center[0], map_center[1]
    zoom_level = 6

m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=zoom_level)
heat_data = [[row.lat, row.lon, row.density] for _, row in base_population_df.iterrows()]
if len(heat_data) > 0:
    HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)
folium.GeoJson(
    oman_boundary,
    name="Oman Boundary",
    style_function=lambda x: {"fillColor": "none", "color": "black", "weight": 2}
).add_to(m)

# Add target location marker
if "target_location" in st.session_state:
    target_lat, target_lon, target_name = st.session_state["target_location"]
    folium.Marker(
        location=[target_lat, target_lon],
        icon=folium.Icon(icon="star", prefix="fa", color="blue"),
        popup=f"🎯 Target: {target_name}"
    ).add_to(m)

# Decide if we should overlay towers (only when params match last run)
show_stored = (
    "towers" in st.session_state and
    st.session_state.get("params_key") == current_params_key
)

if show_stored:
    towers = st.session_state["towers"]
    for i, t in enumerate(towers, 1):
        folium.Marker(
            location=t,
            icon=folium.Icon(icon="signal", prefix="fa", color="red"),
            popup=f"Tower {i} @ {t[0]:.4f}, {t[1]:.4f}<br>Algorithm: {st.session_state.get('algorithm_used', 'Unknown')}"
        ).add_to(m)
        
        # Add coverage circles
        folium.Circle(
            location=t,
            radius=float(st.session_state["coverage_radius"]),
            color="red",
            fillColor="red",
            fillOpacity=0.1,
            popup=f"Coverage area for Tower {i}"
        ).add_to(m)

st_folium(m, width=900, height=600)

# =============================
# Report (only when we have current matching results)
# =============================
if show_stored:
    total_population = st.session_state["total_population"]
    covered_population = st.session_state["covered_population"]
    coverage_radius_val = st.session_state["coverage_radius"]
    algorithm_used = st.session_state.get("algorithm_used", "Unknown")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Population", f"{int(total_population):,}")
    with col2:
        st.metric("Population Covered", f"{int(covered_population):,}")
    with col3:
        coverage_pct = (covered_population / total_population * 100) if total_population > 0 else 0.0
        st.metric("Coverage Percentage", f"{coverage_pct:.2f}%")
    
    st.markdown(f"**🎯 Target Location:** {st.session_state.get('target_location', ['', '', 'Unknown'])[2]}")
    st.markdown(f"**🧠 Algorithm Used:** {algorithm_used}")
    st.markdown(f"**🔥 Why Hotspot Detection:** Identifies and prioritizes the highest population density areas first, ensuring maximum coverage of populated regions")
    st.markdown(f"**📡 Number of Towers Placed:** {len(st.session_state['towers'])}")
    st.markdown(f"**📶 Coverage Radius:** {coverage_radius_val} meters")

    pdf_buf = make_pdf_report(
        st.session_state["towers"], 
        coverage_radius_val, 
        total_population, 
        covered_population, 
        algorithm_used
    )
    st.download_button("📄 Download PDF Report", pdf_buf, file_name="5G_tower_report.pdf", mime="application/pdf")
    
    # Show detailed tower information
    with st.expander("📍 Tower Locations Details"):
        towers_df = pd.DataFrame(st.session_state["towers"], columns=["Latitude", "Longitude"])
        towers_df.index += 1
        towers_df.index.name = "Tower #"
        st.dataframe(towers_df)
else:
    st.info("Map is shown without tower placement. Set your parameters and click 'Run Optimization' to compute and display towers and coverage metrics.")