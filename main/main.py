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

# =============================
# Load Oman boundary polygon
# =============================
oman_boundary = gpd.read_file(r"geoBoundaries-OMN-ADM1.geojson")

oman_boundary = oman_boundary.to_crs(epsg=4326)

# =============================
# Config
# =============================
st.set_page_config(page_title="5G Tower Planner", layout="wide")

OMAN_POP_FILE_CSV = "omn_pd_2007_1km_ASCII_XYZ.csv"
MIN_LAT, MAX_LAT = 16.6, 26.4
MIN_LON, MAX_LON = 51.9, 59.9

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
    for t in towers:
        lon, lat = t[1], t[0]
        tower_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)
        buffer = tower_geom.buffer(radius_m).iloc[0]
        covered = pop_gdf[pop_gdf.geometry.within(buffer)]
        total_covered += float(covered["density"].sum())
    return total_covered


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
        dists_all = remaining.apply(lambda r: haversine(tower_point[0], tower_point[1], r.lat, r.lon), axis=1)
        remaining = remaining.loc[dists_all > radius_m].reset_index(drop=True)
        if remaining.empty:
            break
    return towers

# ==============
# Make PDF Report
# ================
def make_pdf_report(towers, radius_m, total_pop, covered_pop):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "5G Tower Placement Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", ln=True)
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
st.title("📡 5G Tower Planner -- Region-aware (Greedy suggestions + PDF)")

# =============================
# Sidebar
# =============================
with st.sidebar:
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

    st.markdown("**Region bounding box (filter dataset)**")
    lat_min = st.number_input("Min latitude", value=float(MIN_LAT))
    lat_max = st.number_input("Max latitude", value=float(MAX_LAT))
    lon_min = st.number_input("Min longitude", value=float(MIN_LON))
    lon_max = st.number_input("Max longitude", value=float(MAX_LON))
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
    bool(use_mock), int(points_count) if use_mock else -1
)

# Notify if the UI parameters changed since the last optimization
if "params_key" in st.session_state and st.session_state.get("params_key") != current_params_key:
    st.info("Parameters changed. Click (run optimization) to update tower placement and metrics.")

st.markdown("When ready, click the button below to compute tower placement. The map below is shown without towers until you run optimization.")
run_clicked = st.button("(run optimization)")

# =============================
# Run optimization on demand
# =============================
if run_clicked:
    towers = greedy_tower_placement(
        base_population_df, num_towers=int(num_towers), radius_m=float(cell_size_m)
    )
    covered_population = population_covered(
        towers, base_population_df, radius_m=float(cell_size_m)
    )

    # Persist results and context
    st.session_state["towers"] = towers
    st.session_state["covered_population"] = float(covered_population)
    st.session_state["total_population"] = float(total_population_current)
    st.session_state["coverage_radius"] = float(cell_size_m)
    st.session_state["params_key"] = current_params_key

# =============================
# Always show the map (heatmap + boundary). Add towers only if results match current params.
# =============================
m = folium.Map(location=[map_center[0], map_center[1]], zoom_start=6)
heat_data = [[row.lat, row.lon, row.density] for _, row in base_population_df.iterrows()]
if len(heat_data) > 0:
    HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)
folium.GeoJson(
    oman_boundary,
    name="Oman Boundary",
    style_function=lambda x: {"fillColor": "none", "color": "black", "weight": 2}
).add_to(m)

# Decide if we should overlay towers (only when params match last run)
show_stored = (
    "towers" in st.session_state and
    st.session_state.get("params_key") == current_params_key
)

if show_stored:
    towers = st.session_state["towers"]
    for t in towers:
        folium.Marker(
            location=t,
            icon=folium.Icon(icon="signal", prefix="fa", color="red"),
            popup=f"Tower @ {t[0]:.4f}, {t[1]:.4f}"
        ).add_to(m)

st_folium(m, width=900, height=600)

# =============================
# Report (only when we have current matching results)
# =============================
if show_stored:
    total_population = st.session_state["total_population"]
    covered_population = st.session_state["covered_population"]
    coverage_radius_val = st.session_state["coverage_radius"]

    st.markdown(f"**Total population in region:** {int(total_population):,}")
    st.markdown(f"**Population covered by towers:** {int(covered_population):,}")
    coverage_pct = (covered_population / total_population * 100) if total_population > 0 else 0.0
    st.markdown(f"**Coverage percentage:** {coverage_pct:.2f}%")

    pdf_buf = make_pdf_report(st.session_state["towers"], coverage_radius_val, total_population, covered_population)
    st.download_button("📄 Download PDF report", pdf_buf, file_name="5G_tower_report.pdf", mime="application/pdf")
else:
    st.info("Map is shown without tower placement. Set your parameters and click (run optimization) to compute and display towers and coverage metrics.")