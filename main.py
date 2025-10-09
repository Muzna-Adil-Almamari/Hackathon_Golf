# Quick run instructions:
# 1) Open terminal and cd to the project folder:
#      cd "c:\Users\admin\Desktop\Hackathon_Golf1"
# 2) Create & activate a virtual environment:
#      Windows: python -m venv .venv && .venv\Scripts\activate
#      mac/linux: python3 -m venv .venv && source .venv/bin/activate
# 3) Install dependencies:
#      pip install --upgrade pip
#      pip install streamlit folium geopandas shapely fpdf scikit-learn scipy networkx requests haversine streamlit-folium pandas numpy
#    (If geopandas fails to install, consider using conda: conda install -c conda-forge geopandas)
# 4) Run the app:
#      streamlit run "c:\Users\admin\Desktop\Hackathon_Golf1\main.py"
#    Optional: specify port: streamlit run main.py --server.port 8501
# 5) Open the printed URL (usually http://localhost:8501) in your browser.

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
import math
from haversine import haversine, Unit


from app_logger import get_logger


# Initialize logger
logger = get_logger("5G_Tower_Planner")
# =============================
# Load Oman boundary polygon
# =============================
try:
    oman_boundary = gpd.read_file("geoBoundaries-OMN-ADM2.geojson")
    logger.info("Oman boundary loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Oman boundary: {e}")
    st.error("Failed to load Oman boundary.")
    st.error(f"Could not load Oman boundary file: {str(e)}")
    # Provide minimal fallback boundary
    oman_boundary = gpd.GeoDataFrame({
        'geometry': [None],
        'country': ['Oman']
    })

oman_boundary = oman_boundary.to_crs(epsg=4326)

# =============================
# Config
# =============================
st.set_page_config(page_title="5G Tower Planner -- Advanced Algorithms", layout="wide")

OMAN_POP_FILE_CSV = "oman-cities-by-population-2025.csv"
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
    logger.info(f"Requesting coordinates for place: {place_name}")

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return lat, lon, True, None
        else:
            logger.warning(f"Place not found: {place_name}")
            return None, None, False, "Place not found"
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching coordinates for {place_name}: {e}")
        return None, None, False, f"Error occurred: {str(e)}"

def determine_best_algorithm(place_name, population_df):
    """Automatically determine the best algorithm based on location characteristics with logging"""

    urban_areas = ["muscat", "salalah", "sohar", "nizwa", "sur", "rustaq", "bahla", "ibri"]

    logger.info(f"Determining best algorithm for place: {place_name}")

    if len(population_df) > 0:
        avg_density = population_df['density'].mean()
        max_density = population_df['density'].max()
        density_variance = population_df['density'].var()

        logger.info(f"Population metrics - Avg: {avg_density}, Max: {max_density}, Variance: {density_variance}")

        is_urban = any(city in place_name.lower() for city in urban_areas)
        logger.info(f"Is urban area: {is_urban}")

        if is_urban and max_density > 300:
            logger.info("Selected Hotspot Detection algorithm for high-density urban area")
            return ("Hotspot Detection",
                    "High-density urban area detected - using Hotspot Detection for optimal coverage of dense population centers, "
                    "Using high-band (mmWave) 5G (24–40 GHz) with coverage of 50–300 m, offering very high speeds")
        elif avg_density > 150 and density_variance > 10000:
            logger.info("Selected Genetic Algorithm for complex population distribution")
            return ("Genetic Algorithm",
                    "Complex population distribution detected - using Genetic Algorithm for global optimization, "
                    "Using high-band (mmWave) 5G (24–40 GHz) with coverage of 50–300 m, offering very high speeds")
        elif len(population_df) > 1000 and avg_density < 100:
            logger.info("Selected Population-Weighted K-Means for large sparse area")
            return ("Population-Weighted K-Means",
                    "Large area with distributed population - using K-Means clustering for balanced coverage, "
                    "Using high-band (mmWave) 5G (24–40 GHz) with coverage of 50–300 m, offering very high speeds")
        elif avg_density > 200:
            logger.info("Selected Simulated Annealing for moderate to high-density area")
            return ("Simulated Annealing",
                    "Moderate to high density area - using Simulated Annealing for optimized placement, "
                    "Using high-band (mmWave) 5G (24–40 GHz) with coverage of 50–300 m, offering very high speeds")
        else:
            logger.info("Selected Enhanced Greedy as default for standard population distribution")
            return ("Enhanced Greedy",
                    "Standard population distribution - using Enhanced Greedy algorithm for efficient placement, "
                    "Using high-band (mmWave) 5G (24–40 GHz) with coverage of 50–300 m, offering very high speeds")
    else:
        logger.warning("No population data available - using Enhanced Greedy as default")
        return ("Enhanced Greedy",
                "No population data available - using Enhanced Greedy as default, "
                "Using high-band (mmWave) 5G (24–40 GHz) with coverage of 50–300 m, offering very high speeds")
# =============================
# Helpers
# =============================
def fix_tower_overlap(towers, area_type, max_iterations=100):
    """
    Adjusts tower positions to avoid overlap based on given area_type radius.
    
    Parameters:
        towers (list of tuples): (lat, lon)
        area_type (str): "urban" or "rural"
        max_iterations (int): Maximum adjustment iterations.
    
    Returns:
        list of tuples: Adjusted tower positions (lat, lon)
    """
    radius_map = {
        "urban": 700,
        "rural": 5000
    }
    r = radius_map.get(area_type.lower(), 1000)
    logger.info(f"Starting fix_tower_overlap for {len(towers)} towers in {area_type} area with radius {r}m")

    def move_tower(lat, lon, bearing, distance_m):
        """Returns new lat/lon moved from original by distance and bearing."""
        R = 6371000  # Earth radius in meters
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing)

        new_lat = math.asin(math.sin(lat_rad) * math.cos(distance_m / R) +
                            math.cos(lat_rad) * math.sin(distance_m / R) * math.cos(bearing_rad))
        new_lon = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_m / R) * math.cos(lat_rad),
                                       math.cos(distance_m / R) - math.sin(lat_rad) * math.sin(new_lat))

        return math.degrees(new_lat), math.degrees(new_lon)

    towers = list(towers)  # Copy list

    for iteration in range(1, max_iterations + 1):
        overlap_found = False
        logger.debug(f"Iteration {iteration} checking for overlaps...")

        for i in range(len(towers)):
            lat1, lon1 = towers[i]

            for j in range(i + 1, len(towers)):
                lat2, lon2 = towers[j]

                dist = haversine_meters(lat1, lon1, lat2, lon2)
                min_dist = 2 * r  # since same radius for all

                if dist < min_dist:  # Overlap detected
                    overlap_found = True
                    logger.info(f"Overlap detected between tower {i} and tower {j} (distance: {dist:.2f}m, min required: {min_dist}m)")

                    # Bearing from tower1 to tower2
                    bearing = math.degrees(math.atan2(
                        math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2)),
                        math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
                        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                        math.cos(math.radians(lon2 - lon1))
                    ))

                    shift_amount = (min_dist - dist) / 2 + 1  # Add 1m buffer
                    logger.debug(f"Shifting tower {i} by {shift_amount:.2f}m opposite bearing, tower {j} by same towards bearing")

                    # Move both towers in opposite directions
                    towers[i] = move_tower(lat1, lon1, bearing + 180, shift_amount)
                    towers[j] = move_tower(lat2, lon2, bearing, shift_amount)

        if not overlap_found:
            logger.info(f"No overlaps detected in iteration {iteration}. Adjustment complete.")
            break
    else:
        logger.warning(f"Max iterations ({max_iterations}) reached. Some overlaps may remain.")

    return towers


# def generate_mock_population_oman(num_points=2000):
#     random.seed(42)
#     data = []
#     for _ in range(num_points):
#         lat = random.uniform(MIN_LAT, MAX_LAT)
#         lon = random.uniform(MIN_LON, MAX_LON)
#         density = random.randint(5, 500)
#         city = f"Point_{_}"
#         data.append([city, lat, lon, density])
#     df = pd.DataFrame(data, columns=["city", "lat", "lon", "density"])
#     gdf = gpd.GeoDataFrame(
#         df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
#     )
#     gdf_within = gpd.sjoin(gdf, oman_boundary, predicate="within")
#     return gdf_within.drop(columns=["geometry", "index_right"]).reset_index(drop=True)

def load_population_from_csv(csv_file="mock_population_oman.csv"):
    """Load population data from CSV file."""
    try:
        df = pd.read_csv(csv_file)
        required_columns = {"city", "lat", "lon", "density"}
        if required_columns.issubset(df.columns):
            logger.info(f"Population CSV loaded successfully: {csv_file}, {len(df)} records")
            return df
        else:
            logger.error(f"CSV file missing required columns: {required_columns}")
            return pd.DataFrame(columns=["city", "lat", "lon", "density"])
    except Exception as e:
        logger.exception(f"Failed to load population CSV: {e}")
        return pd.DataFrame(columns=["city", "lat", "lon", "density"])


def haversine(lat1, lon1, lat2, lon2):
    """Compute Haversine distance in meters between two lat/lon points."""
    R = 6371000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2.0)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0)**2
    distance = 2 * R * atan2(sqrt(a), sqrt(1 - a))
    logger.debug(f"Haversine distance between ({lat1},{lon1}) and ({lat2},{lon2}): {distance:.2f} m")
    return distance


def population_covered(towers, population_df, radius_m):
    """Calculate total population covered by towers given a coverage radius."""
    if len(population_df) == 0 or len(towers) == 0:
        logger.warning("No population data or no towers provided for coverage calculation.")
        return 0.0

    pop_gdf = gpd.GeoDataFrame(
        population_df,
        geometry=gpd.points_from_xy(population_df.lon, population_df.lat),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    total_covered = 0.0
    covered_points = set()
    
    for idx, t in enumerate(towers):
        lon, lat = t[1], t[0]
        tower_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)
        buffer = tower_geom.buffer(radius_m).iloc[0]
        covered = pop_gdf[pop_gdf.geometry.within(buffer)]

        logger.info(f"Tower {idx} at ({lat},{lon}) covers {len(covered)} population points")
        
        for idx_covered in covered.index:
            if idx_covered not in covered_points:
                covered_points.add(idx_covered)
                total_covered += float(covered.loc[idx_covered, "density"])
    
    logger.info(f"Total population covered by {len(towers)} towers: {total_covered}")
    return total_covered

# =============================
# ALGORITHM 1: Enhanced Greedy (Original improved)
# =============================
def greedy_tower_placement(pop_df, num_towers, radius_m):
    if len(pop_df) == 0:
        logger.warning("Greedy placement: population dataframe is empty.")
        return []

    remaining = pop_df.copy().reset_index(drop=True)
    towers = []
    radius_deg = radius_m / 111320.0
    logger.info(f"Starting Greedy tower placement for {num_towers} towers with radius {radius_m} m.")

    for tower_idx in range(num_towers):
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
            logger.info(f"No suitable tower location found for tower {tower_idx}. Stopping early.")
            break

        chosen = remaining.loc[best_idx]
        tower_point = (float(chosen.lat), float(chosen.lon))
        towers.append(tower_point)
        logger.info(f"Placed tower {tower_idx} at {tower_point}, covering population: {best_score:.2f}")

        # Remove covered population
        dists_all = remaining.apply(lambda r: haversine(tower_point[0], tower_point[1], r.lat, r.lon), axis=1)
        remaining = remaining.loc[dists_all > radius_m].reset_index(drop=True)

        if remaining.empty:
            logger.info("All population covered, ending greedy placement early.")
            break

    logger.info(f"Greedy tower placement completed: {len(towers)} towers placed.")
    return towers


# =============================
# ALGORITHM 2: Population-Weighted K-Means Clustering
# =============================
def kmeans_tower_placement(pop_df, num_towers, radius_m):
    if len(pop_df) == 0 or num_towers == 0:
        logger.warning("K-Means placement: population dataframe is empty or number of towers is zero.")
        return []

    logger.info(f"Starting K-Means tower placement for {num_towers} towers with radius {radius_m} m.")

    # Create weighted points based on population density
    weighted_points = []
    weights = []

    for _, row in pop_df.iterrows():
        weight = max(1, int(row.density / 50))  # Scale factor
        for _ in range(min(weight, 10)):  # Cap to avoid memory issues
            weighted_points.append([row.lat, row.lon])
            weights.append(row.density)

    if len(weighted_points) < num_towers:
        logger.info("Not enough weighted points for K-Means, using original population points.")
        weighted_points = pop_df[['lat', 'lon']].values.tolist()

    kmeans = KMeans(n_clusters=num_towers, random_state=42, n_init=10)
    cluster_centers = kmeans.fit(weighted_points).cluster_centers_

    towers = [(float(center[0]), float(center[1])) for center in cluster_centers]
    logger.info(f"K-Means tower placement completed: {len(towers)} towers placed.")
    return towers

# =============================
# ALGORITHM 3: Population Density Hotspot Detection
# =============================
def haversine_meters(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points in meters."""
    R = 6371000  # Earth's radius in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def hotspot_tower_placement(pop_df, num_towers, radius_m):
    if len(pop_df) == 0:
        logger.warning("Hotspot placement: population dataframe is empty.")
        return []

    towers = []
    remaining_pop = pop_df.copy()
    logger.info(f"Starting Hotspot tower placement for {num_towers} towers with radius {radius_m} m.")

    for tower_idx in range(num_towers):
        if remaining_pop.empty:
            logger.info("No remaining population to cover. Ending hotspot placement early.")
            break

        # Find the point with highest population density
        max_density_idx = remaining_pop.density.idxmax()
        hotspot = remaining_pop.loc[max_density_idx]
        tower_point = (float(hotspot.lat), float(hotspot.lon))
        towers.append(tower_point)

        logger.info(f"Placed tower {tower_idx} at {tower_point}, covering population density: {hotspot.density}")

        # Remove points within coverage radius
        distances = remaining_pop.apply(
            lambda row: haversine(hotspot.lat, hotspot.lon, row.lat, row.lon), axis=1
        )
        remaining_pop = remaining_pop[distances > radius_m].reset_index(drop=True)

    logger.info(f"Hotspot tower placement completed: {len(towers)} towers placed.")
    return towers
# =============================
# ALGORITHM 4: Genetic Algorithm for Tower Placement
# =============================
def genetic_algorithm_tower_placement(pop_df, num_towers, radius_m, generations=50, population_size=20):
    if len(pop_df) == 0 or num_towers == 0:
        logger.warning("Genetic Algorithm: Population dataframe is empty or number of towers is 0.")
        return []

    logger.info(f"Starting Genetic Algorithm for {num_towers} towers, {generations} generations, population size {population_size}.")

    def fitness_function(towers):
        score = population_covered(towers, pop_df, radius_m)
        return score

    def create_individual():
        individual = random.sample(
            [(row.lat, row.lon) for _, row in pop_df.iterrows()],
            min(num_towers, len(pop_df))
        )
        return individual

    def mutate(individual, mutation_rate=0.1):
        if random.random() < mutation_rate:
            if len(individual) > 0:
                idx = random.randint(0, len(individual) - 1)
                new_point = random.choice([(row.lat, row.lon) for _, row in pop_df.iterrows()])
                logger.debug(f"Mutating tower at index {idx} to new location {new_point}.")
                individual[idx] = new_point
        return individual

    def crossover(parent1, parent2):
        if len(parent1) <= 1:
            return parent1[:]
        point = random.randint(1, len(parent1) - 1)
        child = parent1[:point] + parent2[point:]
        return child[:num_towers]

    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    logger.info("Initial population created.")

    # Evolution
    for generation in range(generations):
        fitness_scores = [(individual, fitness_function(individual)) for individual in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        best_score = fitness_scores[0][1]
        logger.info(f"Generation {generation}: Best fitness score = {best_score}")

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
    logger.info(f"Genetic Algorithm completed. Best solution covers population score: {fitness_function(best_solution)}")
    return best_solution
# =============================
# ALGORITHM 5: Simulated Annealing
# =============================
def simulated_annealing_tower_placement(pop_df, num_towers, radius_m, max_iterations=1000):
    if len(pop_df) == 0 or num_towers == 0:
        logger.warning("Simulated Annealing: Population dataframe is empty or number of towers is 0.")
        return []

    logger.info(f"Starting Simulated Annealing for {num_towers} towers with {max_iterations} iterations.")

    # Initial solution (random)
    current_solution = random.sample(
        [(row.lat, row.lon) for _, row in pop_df.iterrows()],
        min(num_towers, len(pop_df))
    )
    current_fitness = population_covered(current_solution, pop_df, radius_m)
    
    best_solution = current_solution[:]
    best_fitness = current_fitness

    logger.info(f"Initial fitness: {current_fitness}")

    initial_temp = 10000
    final_temp = 1
    alpha = 0.95
    temperature = initial_temp

    for iteration in range(max_iterations):
        neighbor = current_solution[:]
        if len(neighbor) > 0:
            idx = random.randint(0, len(neighbor) - 1)
            new_point = random.choice([(row.lat, row.lon) for _, row in pop_df.iterrows()])
            logger.debug(f"Iteration {iteration}: Modifying tower index {idx} to new point {new_point}.")
            neighbor[idx] = new_point

        neighbor_fitness = population_covered(neighbor, pop_df, radius_m)
        delta = neighbor_fitness - current_fitness

        if delta > 0 or (temperature > 0 and random.random() < np.exp(delta / temperature)):
            current_solution = neighbor
            current_fitness = neighbor_fitness
            logger.debug(f"Iteration {iteration}: Accepted new solution with fitness {current_fitness}.")

            if current_fitness > best_fitness:
                best_solution = current_solution[:]
                best_fitness = current_fitness
                logger.info(f"Iteration {iteration}: New best fitness found: {best_fitness}")

        # Cool down
        temperature *= alpha
        if temperature < final_temp:
            temperature = final_temp

    logger.info(f"Simulated Annealing completed. Best fitness: {best_fitness}")
    return best_solution

# ==============
# Make PDF Report
# ==============
def make_pdf_report(towers, radius_m, total_pop, covered_pop, algorithm_name):
    try:
        logger.info(f"Generating PDF report using {algorithm_name} algorithm.")
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

        logger.info("PDF report generated successfully.")
        return pdf_buf

    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        return None


def load_population_csv_or_mock(spread_points_per_city=500):
    def generate_points_in_polygon(polygon, num_points):
        minx, miny, maxx, maxy = polygon.bounds
        points = []
        while len(points) < num_points:
            p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if polygon.contains(p):
                points.append(p)
        return points

    try:
        if os.path.exists(OMAN_POP_FILE_CSV):
            logger.info(f"Loading population data from CSV: {OMAN_POP_FILE_CSV}")
            df = pd.read_csv(OMAN_POP_FILE_CSV)
            if "city" in df.columns and "lat" in df.columns and "lon" in df.columns:
                pop_col = next((col for col in ["population", "density"] if col in df.columns), None)
                if pop_col:
                    df = df[["city", "lat", "lon", pop_col]].copy()
                    df.columns = ["city", "lat", "lon", "density"]

                    expanded_rows = []
                    for _, row in df.iterrows():
                        city_name = str(row["city"]).strip().lower()
                        total_pop = float(row["density"])

                        # Try to match city name with ADM2 shapeName
                        match = oman_boundary[
                            oman_boundary.apply(lambda r: str(r.get("shapeName", "")).strip().lower() == city_name, axis=1)
                        ]

                        if not match.empty and match.geometry.iloc[0] is not None:
                            poly = match.geometry.iloc[0]
                            points = generate_points_in_polygon(poly, spread_points_per_city)
                            per_point_pop = total_pop / spread_points_per_city
                            for p in points:
                                expanded_rows.append({
                                    "city": row["city"],
                                    "lat": p.y,
                                    "lon": p.x,
                                    "density": per_point_pop
                                })
                            logger.debug(f"Generated {len(points)} points for city {city_name}.")
                        else:
                            expanded_rows.append(row.to_dict())
                            logger.warning(f"No polygon found for city '{city_name}', using single point.")

                    return pd.DataFrame(expanded_rows)

        logger.warning("Population CSV not found or invalid, using mock population data.")
        return load_population_from_csv()

    except Exception as e:
        logger.error(f"Error loading population data ({str(e)}), using mock data.")
        return load_population_from_csv()
# =============================
# UI -- Title
# =============================
st.title("📡 5G Tower Planner -- Smart Location-Based Placement")
logger.info("Streamlit app started.")

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("📍 Location Selection")
    
    # Location input methods
    location_method = st.radio(
        "Choose location method:",
        ["Search by place name", "Manual coordinates"]
    )
    logger.info(f"User selected location input method: {location_method}")
    
    # New file uploader for population CSV
    uploaded_file = st.file_uploader("Upload Population CSV File", type=["csv"])
    st.caption("Note: Your .csv file should include the following columns: city, lat, lon, and population/density.")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "lat" in df.columns and "lon" in df.columns:
                pop_col = "population" if "population" in df.columns else ("density" if "density" in df.columns else None)
                if pop_col is None:
                    st.error("Uploaded CSV does not contain 'density' or 'population' columns. Please check the file.")
                    logger.warning("Uploaded CSV missing 'density' or 'population' columns.")
                else:
                    st.success("Population CSV uploaded successfully!")
                    logger.info(f"Population CSV uploaded successfully with {len(df)} rows.")
            else:
                st.warning("CSV missing 'lat'/'lon' columns. Please check the file.")
                logger.warning("Uploaded CSV missing 'lat'/'lon' columns.")
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")
            logger.error(f"Error reading uploaded CSV: {e}")
    
    target_lat, target_lon, location_found, location_error = None, None, False, None
    
    if location_method == "Search by place name":
        place_name = st.text_input("Enter place name in Oman:", value="Muscat")
        
        if st.button("🔍 Find Location"):
            logger.info(f"User requested location search for: {place_name}")
            if place_name.strip():
                with st.spinner("Searching for location..."):
                    target_lat, target_lon, location_found, location_error = get_coordinates_from_place(place_name.strip())
                
                if location_found:
                    st.success(f"✅ Found {place_name}: {target_lat:.4f}, {target_lon:.4f}")
                    st.session_state["target_location"] = (target_lat, target_lon, place_name)
                    logger.info(f"Location found: {place_name} ({target_lat}, {target_lon})")
                else:
                    st.error(f"❌ {location_error}")
                    logger.warning(f"Location search failed for {place_name}: {location_error}")
            else:
                st.warning("Please enter a place name")
                logger.warning("User pressed 'Find Location' without entering a place name.")
    
    elif location_method == "Manual coordinates":
        target_lat = st.number_input("Target Latitude", value=17.0150, format="%.6f")
        target_lon = st.number_input("Target Longitude", value=54.0924, format="%.6f")
        place_name = st.text_input("Location name (optional):", value="Custom Location")
        
        if st.button("📍 Set Location"):
            st.session_state["target_location"] = (target_lat, target_lon, place_name)
            st.success(f"✅ Location set: {target_lat:.4f}, {target_lon:.4f}")
            logger.info(f"Manual coordinates set by user: {place_name} ({target_lat}, {target_lon})")
    
    # Show current target location
    if "target_location" in st.session_state:
        loc_lat, loc_lon, loc_name = st.session_state["target_location"]
        st.info(f"📌 Current target: **{loc_name}** ({loc_lat:.4f}, {loc_lon:.4f})")
        logger.debug(f"Current target location in session: {loc_name} ({loc_lat}, {loc_lon})")
    
    st.header("Radio & Constraints")
    num_towers = st.slider("# Number Of Towers", 1, 300, 5)
    logger.info(f"Number of towers selected: {num_towers}")

    profile = st.selectbox("Propagation Profile", ["Urban", "Rural"])
    logger.info(f"Propagation profile selected: {profile}")
    cell_size_m = 700 if profile == "Urban" else 5000

    st.markdown("**Analysis Region (around target location)**")
    if location_method == "Bounding box":
        lat_min = st.number_input("Min latitude", value=float(MIN_LAT))
        lat_max = st.number_input("Max latitude", value=float(MAX_LAT))
        lon_min = st.number_input("Min longitude", value=float(MIN_LON))
        lon_max = st.number_input("Max longitude", value=float(MAX_LON))
        logger.info("Bounding box manually set by user.")
    else:
        region_radius_km = st.slider("Analysis radius (km)", 5, 100, 25)
        logger.info(f"Region radius selected: {region_radius_km} km")
        
        if "target_location" in st.session_state:
            loc_lat, loc_lon, _ = st.session_state["target_location"]
            lat_offset = region_radius_km / 111.0
            lon_offset = region_radius_km / (111.0 * cos(radians(loc_lat)))
            
            lat_min = max(MIN_LAT, loc_lat - lat_offset)
            lat_max = min(MAX_LAT, loc_lat + lat_offset)
            lon_min = max(MIN_LON, loc_lon - lon_offset)
            lon_max = min(MAX_LON, loc_lon + lon_offset)
            
            st.write(f"Region: {lat_min:.3f} to {lat_max:.3f} lat, {lon_min:.3f} to {lon_max:.3f} lon")
            logger.info(f"Auto-calculated region around target location: lat({lat_min}-{lat_max}), lon({lon_min}-{lon_max})")
        else:
            lat_min, lat_max = MIN_LAT, MAX_LAT
            lon_min, lon_max = MIN_LON, MAX_LON
            logger.warning("No target location set; using full default region.")
    
    use_mock = st.checkbox("Use mock (Oman-wide) population", value=True)
    points_count = st.number_input("Number of mock points", value=100000, min_value=200, max_value=100000, step=100)
    logger.info(f"Use mock population: {use_mock}, Number of points: {points_count}")
# =============================
# Load population (always, to show map)
# =============================
if use_mock:
    base_population_df = load_population_from_csv(csv_file="mock_population_oman.csv")
    logger.info(f"Loaded mock population CSV with {len(base_population_df)} points.")
else:
    base_population_df = load_population_csv_or_mock()
    logger.info(f"Loaded real/mock population CSV with {len(base_population_df)} points.")

if "target_location" in st.session_state:
    loc_lat, loc_lon, loc_name = st.session_state["target_location"]
    logger.info(f"Using target location: {loc_name} ({loc_lat}, {loc_lon})")
    
    # Try to find polygon for the city
    match_poly = oman_boundary[
        oman_boundary.apply(lambda r: str(r.get("shapeName", "")).strip().lower() == loc_name.strip().lower(), axis=1)
    ]
    
    if not match_poly.empty:
        city_poly = match_poly.geometry.iloc[0]
        base_population_df = base_population_df[
            base_population_df.apply(lambda r: Point(r.lon, r.lat).within(city_poly), axis=1)
        ].reset_index(drop=True)
        st.info(f"📍 Using full polygon for {loc_name} ({len(base_population_df)} population points).")
        logger.info(f"Filtered population points using polygon: {len(base_population_df)} points remain.")
    else:
        # Fallback to bounding box
        base_population_df = base_population_df[
            (base_population_df.lat >= lat_min) & (base_population_df.lat <= lat_max) &
            (base_population_df.lon >= lon_min) & (base_population_df.lon <= lon_max)
        ].reset_index(drop=True)
        st.info(f"📍 Using bounding box for {loc_name} ({len(base_population_df)} population points).")
        logger.warning(f"No polygon found for {loc_name}. Using bounding box ({len(base_population_df)} points).")
else:
    st.warning("No target location found. Please search or select a location.")
    logger.warning("Population filtering skipped: no target location set.")

total_population_current = float(base_population_df.density.sum())
logger.info(f"Total population in current analysis region: {total_population_current}")

map_center = ((lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0)
logger.debug(f"Map center set at: {map_center}")

# Parameter signature to detect changes since last run
current_params_key = (
    float(lat_min), float(lat_max), float(lon_min), float(lon_max),
    int(num_towers), float(cell_size_m),
    bool(use_mock), int(points_count) if use_mock else -1,
    str(st.session_state.get("target_location", ""))
)
logger.debug(f"Current UI parameters key: {current_params_key}")

# Notify if the UI parameters changed since the last optimization
if "params_key" in st.session_state and st.session_state.get("params_key") != current_params_key:
    st.info("Parameters changed. Click 'Run Optimization' to update tower placement and metrics.")
    logger.info("UI parameters changed since last optimization.")

# Algorithm descriptions
if "target_location" in st.session_state:
    loc_name = st.session_state["target_location"][2]
    st.success(f"🎯 **Target Location**: {loc_name}")
    st.info("🤖 The system will automatically select the best placement algorithm based on the location's population characteristics.")
    logger.info(f"Ready to run optimization for target location: {loc_name}")
else:
    st.warning("📍 Please select a target location first using the sidebar options.")
    logger.warning("Run optimization disabled: no target location set.")

st.markdown(
    "When ready, click the button below to compute tower placement. "
    "The system will analyze the population distribution and choose the optimal algorithm automatically."
)
run_clicked = st.button("🚀 Run Smart Optimization")
if run_clicked:
    logger.info("User clicked 'Run Smart Optimization'.")
# =============================
# Run optimization on demand
# =============================
if run_clicked and "target_location" in st.session_state:
    loc_name = st.session_state["target_location"][2]
    logger.info(f"Starting tower optimization for target location: {loc_name}")
    
    # Determine the best algorithm based on location and population data
    algorithm, algorithm_reason = determine_best_algorithm(loc_name, base_population_df)
    logger.info(f"Selected algorithm: {algorithm} ({algorithm_reason})")
    
    st.info(f"🧠 **Selected Algorithm**: {algorithm}")
    st.write(f"**Reason**: {algorithm_reason}")
    
    with st.spinner(f"Running {algorithm} algorithm for {loc_name}..."):
        # Algorithm-specific parameters (set automatically)
        ga_generations = 50
        ga_population = 20
        sa_iterations = 1000
        logger.debug(f"GA params: generations={ga_generations}, population={ga_population}")
        logger.debug(f"SA params: max_iterations={sa_iterations}")
        
        # Select and run the chosen algorithm
        towers = []
        if algorithm == "Enhanced Greedy":
            towers = greedy_tower_placement(base_population_df, num_towers=int(num_towers), radius_m=float(cell_size_m))
            logger.info(f"Enhanced Greedy produced {len(towers)} towers before overlap fix.")
        elif algorithm == "Population-Weighted K-Means":
            towers = kmeans_tower_placement(base_population_df, num_towers=int(num_towers), radius_m=float(cell_size_m))
            logger.info(f"K-Means produced {len(towers)} towers before overlap fix.")
        elif algorithm == "Hotspot Detection":
            towers = hotspot_tower_placement(base_population_df, num_towers=int(num_towers), radius_m=float(cell_size_m))
            logger.info(f"Hotspot Detection produced {len(towers)} towers before overlap fix.")
        elif algorithm == "Genetic Algorithm":
            towers = genetic_algorithm_tower_placement(
                base_population_df,
                num_towers=int(num_towers),
                radius_m=float(cell_size_m),
                generations=ga_generations,
                population_size=ga_population
            )
            logger.info(f"Genetic Algorithm produced {len(towers)} towers before overlap fix.")
        elif algorithm == "Simulated Annealing":
            towers = simulated_annealing_tower_placement(
                base_population_df,
                num_towers=int(num_towers),
                radius_m=float(cell_size_m),
                max_iterations=sa_iterations
            )
            logger.info(f"Simulated Annealing produced {len(towers)} towers before overlap fix.")
        
        # Apply overlap correction
        towers = fix_tower_overlap(towers=towers, area_type=profile)
        logger.info(f"Towers after overlap fix: {len(towers)}")

        # Calculate coverage
        covered_population = population_covered(towers, base_population_df, radius_m=float(cell_size_m))
        logger.info(f"Covered population: {covered_population} / {total_population_current} ({covered_population/total_population_current*100:.2f}%)")
        
        # Persist results and context
        st.session_state.update({
            "towers": towers,
            "covered_population": float(covered_population),
            "total_population": float(total_population_current),
            "coverage_radius": float(cell_size_m),
            "algorithm_used": algorithm,
            "algorithm_reason": algorithm_reason,
            "params_key": current_params_key
        })
        logger.info(f"Optimization results saved to session state.")
        
        st.success(f"✅ Optimization complete! Placed {len(towers)} towers in {loc_name} using {algorithm}.")

elif run_clicked and "target_location" not in st.session_state:
    st.error("❌ Please select a target location first!")
    logger.warning("Optimization attempted without a target location.")
# =============================
# Always show the map (heatmap + boundary). Add towers only if results match current params.
# =============================

# Determine map center
if "target_location" in st.session_state:
    map_center_lat, map_center_lon, _ = st.session_state["target_location"]
    zoom_level = 10
    logger.info(f"Map centered on target location: {map_center_lat}, {map_center_lon}")
else:
    map_center_lat, map_center_lon = map_center[0], map_center[1]
    zoom_level = 6
    logger.info(f"Map centered on default center: {map_center_lat}, {map_center_lon}")

# Create folium map
m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=zoom_level)
heat_data = [[row.lat, row.lon, row.density] for _, row in base_population_df.iterrows()]
if len(heat_data) > 0:
    HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)
    logger.info(f"Added heatmap with {len(heat_data)} population points.")

# Add Oman boundary
folium.GeoJson(
    oman_boundary,
    name="Oman Boundary",
    style_function=lambda x: {"fillColor": "none", "color": "black", "weight": 2}
).add_to(m)
logger.info("Oman boundary overlay added to map.")

# Add target location marker
if "target_location" in st.session_state:
    target_lat, target_lon, target_name = st.session_state["target_location"]
    folium.Marker(
        location=[target_lat, target_lon],
        icon=folium.Icon(icon="star", prefix="fa", color="blue"),
        popup=f"🎯 Target: {target_name}"
    ).add_to(m)
    logger.info(f"Target location marker added: {target_name} ({target_lat}, {target_lon})")

# Decide if we should overlay towers (only when params match last run)
show_stored = (
    "towers" in st.session_state and
    st.session_state.get("params_key") == current_params_key
)

if show_stored:
    towers = st.session_state["towers"]
    logger.info(f"Overlaying {len(towers)} towers on the map.")
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
    logger.info("Tower markers and coverage circles added.")

st_folium(m, width=900, height=600)
logger.info("Map displayed in Streamlit.")

# =============================
# Report (only when we have current matching results)
# =============================
if show_stored:
    total_population = st.session_state["total_population"]
    covered_population = st.session_state["covered_population"]
    coverage_radius_val = st.session_state["coverage_radius"]
    algorithm_used = st.session_state.get("algorithm_used", "Unknown")

    logger.info(f"Displaying report: Total pop={total_population}, Covered pop={covered_population}, Algorithm={algorithm_used}")

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
    st.markdown(f"**💡 Selection Reason:** {st.session_state.get('algorithm_reason', 'N/A')}")
    st.markdown(f"**📡 Number of Towers Placed:** {len(st.session_state['towers'])}")
    st.markdown(f"**📶 Coverage Radius:** {coverage_radius_val} meters")

    pdf_buf = make_pdf_report(
        st.session_state["towers"], 
        coverage_radius_val, 
        total_population, 
        covered_population, 
        algorithm_used
    )
    logger.info("PDF report generated.")
    
    st.download_button("📄 Download PDF Report", pdf_buf, file_name="5G_tower_report.pdf", mime="application/pdf")
    
    # Show detailed tower information
    with st.expander("📍 Tower Locations Details"):
        towers_df = pd.DataFrame(st.session_state["towers"], columns=["Latitude", "Longitude"])
        towers_df.index += 1
        towers_df.index.name = "Tower #"
        st.dataframe(towers_df)
        logger.info("Displayed detailed tower locations in the UI.")
else:
    st.info("Map is shown without tower placement. Set your parameters and click 'Run Optimization' to compute and display towers and coverage metrics.")
    logger.info("Map displayed without towers (no matching results or parameters changed).")

if __name__ == "__main__":
    # If someone runs the file directly with python, provide a short hint.
    print("This file is a Streamlit app. Start it with the Streamlit CLI, for example:")
    print(r'  streamlit run "c:\Users\admin\Desktop\Hackathon_Golf1\main.py"')
    print("Or see the top-of-file comments for detailed instructions.")
