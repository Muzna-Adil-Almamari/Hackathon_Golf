import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
from dataclasses import dataclass

st.set_page_config(page_title="5G Tower Planner — Mini-Simulator", layout="wide")

# =============================
# Synthetic data generators
# =============================
def make_synthetic_city(h=50, w=50, seed=42):
    """Return (population_grid, obstacle_grid) as floats with shape (h, w)."""
    rng = np.random.default_rng(seed)

    # Base population (Poisson) + downtown blob
    pop = rng.poisson(lam=5, size=(h, w)).astype(float)
    yy, xx = np.mgrid[0:h, 0:w]
    center = np.array([h * 0.55, w * 0.45])
    dist = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    downtown = np.exp(-(dist ** 2) / (2 * (h * 0.12) ** 2))
    pop = pop + (downtown * 20.0)
    pop[pop < 0] = 0.0

    # Obstacles: light random clutter + two strong “walls”
    obstacles = rng.uniform(0.0, 0.2, size=(h, w))
    obstacles[:, 20:22] += 0.5          # vertical wall
    obstacles[35:37, :] += 0.4          # horizontal wall
    obstacles = np.clip(obstacles, 0.0, 1.0)

    return pop, obstacles


# =============================
# Geo helpers (grid -> lat/lon)
# =============================
@dataclass
class GeoFrame:
    lat0: float = 23.5859   # arbitrary reference (Muscat-ish)
    lon0: float = 58.4059
    cell_size_m: float = 100.0

def grid_to_latlon(h, w, gf: GeoFrame):
    """Return (lat[h,w], lon[h,w]) for grid cell centers."""
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(gf.lat0))
    yy, xx = np.mgrid[0:h, 0:w]
    lat = gf.lat0 + (yy * gf.cell_size_m) / m_per_deg_lat
    lon = gf.lon0 + (xx * gf.cell_size_m) / m_per_deg_lon
    return lat, lon


# =============================
# Radio propagation
# =============================
def log_distance_pathloss(d_m, freq_mhz, n):
    """Log-distance path loss (Friis referenced at 1 m)."""
    d = np.maximum(d_m, 1.0)
    # Friis @ 1m: 32.45 + 20log10(f_MHz) + 20log10(0.001 km) = 32.45 + 20log10(f) - 60
    fspl_1m = 32.45 + 20 * np.log10(freq_mhz) - 60.0
    return fspl_1m + 10 * n * np.log10(d)

def compute_received_power(tx_power_dbm, freq_mhz, cell_size_m, towers, obstacles, pathloss_exp):
    """
    Return prx[h,w] where each cell is the max received power from all towers.
    Obstacles are modeled as an extra 0..20 dB loss per cell.
    """
    H, W = obstacles.shape
    yy, xx = np.mgrid[0:H, 0:W]
    y_m = yy * cell_size_m + cell_size_m / 2.0
    x_m = xx * cell_size_m + cell_size_m / 2.0
    prx = np.full((H, W), -200.0, dtype=float)

    for (ty, tx) in towers:
        ty_m = ty * cell_size_m + cell_size_m / 2.0
        tx_m = tx * cell_size_m + cell_size_m / 2.0
        d = np.hypot(y_m - ty_m, x_m - tx_m)
        pl = log_distance_pathloss(d, freq_mhz, n=pathloss_exp)
        obs_loss = obstacles * 20.0  # up to +20 dB extra loss
        prx_candidate = tx_power_dbm - (pl + obs_loss)
        prx = np.maximum(prx, prx_candidate)

    return prx


# =============================
# Greedy optimizer (baseline)
# =============================
def greedy_place_towers(pop, obstacles, K, tx_power_dbm, freq_mhz, cell_size_m,
                        min_rsrp_dbm, overlap_penalty, pathloss_exp):
    """Iteratively place K towers maximizing demand-weighted coverage with overlap penalty."""
    H, W = pop.shape
    covered = np.zeros((H, W), dtype=bool)
    towers = []
    history = []
    candidates = [(y, x) for y in range(H) for x in range(W)]

    best_prx = None
    for k in range(K):
        best_gain = -1.0
        best_site = None
        best_prx_k = None

        for (y, x) in candidates:
            trial = towers + [(y, x)]
            prx = compute_received_power(tx_power_dbm, freq_mhz, cell_size_m, trial, obstacles, pathloss_exp)
            newly_covered = prx >= min_rsrp_dbm
            new_cells = np.logical_and(newly_covered, ~covered)
            old_cells = np.logical_and(newly_covered, covered)

            gain = (pop[new_cells].sum()) + overlap_penalty * (pop[old_cells].sum())
            if gain > best_gain:
                best_gain, best_site, best_prx_k = gain, (y, x), prx

        towers.append(best_site)
        best_prx = best_prx_k
        covered = best_prx >= min_rsrp_dbm

        history.append({
            "step": k + 1,
            "row": int(best_site[0]),
            "col": int(best_site[1]),
            "marginal_gain": float(best_gain),
            "covered_pop": float(pop[covered].sum()),
        })

    final_prx = compute_received_power(tx_power_dbm, freq_mhz, cell_size_m, towers, obstacles, pathloss_exp)
    final_cov = final_prx >= min_rsrp_dbm
    return towers, final_prx, final_cov, history


# =============================
# UI — Sidebar controls
# =============================
st.title("5G Tower Planner — Mini-Simulator (Interactive)")

with st.sidebar:
    st.header("Scenario Data")
    data_choice = st.selectbox("Map data", ["Synthetic (default)", "Upload CSV grids"])
    h = st.number_input("Grid height (cells)", 10, 500, 50)
    w = st.number_input("Grid width (cells)", 10, 500, 50)

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

    st.header("Data upload (optional)")
    pop_file = obs_file = None
    if data_choice == "Upload CSV grids":
        pop_file = st.file_uploader("Population grid CSV", type=["csv"])
        obs_file = st.file_uploader("Obstacle grid CSV", type=["csv"])

run = st.button("Run Optimization", type="primary")


# =============================
# Load / validate data
# =============================
if data_choice == "Synthetic (default)":
    pop, obstacles = make_synthetic_city(h=int(h), w=int(w))
else:
    if not pop_file or not obs_file:
        st.info("Upload both Population and Obstacle CSVs with the same shape.")
        st.stop()
    pop = pd.read_csv(pop_file, header=None).astype(float).values
    obstacles = pd.read_csv(obs_file, header=None).astype(float).values
    if pop.shape != obstacles.shape:
        st.error(f"Shape mismatch: population {pop.shape} vs obstacles {obstacles.shape}")
        st.stop()
    h, w = pop.shape

gf = GeoFrame(cell_size_m=float(cell_size_m))
lat, lon = grid_to_latlon(h, w, gf)


# =============================
# Cached solver
# =============================
@st.cache_data(show_spinner=False)
def run_solver(pop, obstacles, K, tx_power_dbm, freq_mhz, cell_size_m, min_rsrp_dbm, overlap_penalty, pathloss_exp):
    towers, prx, cov, history = greedy_place_towers(
        pop, obstacles, K, tx_power_dbm, freq_mhz, cell_size_m, min_rsrp_dbm, overlap_penalty, pathloss_exp
    )
    return towers, prx, cov, history


# =============================
# Run + Visualize
# =============================
if run:
    towers, prx, cov, history = run_solver(
        pop, obstacles, num_towers, tx_power_dbm, freq_mhz, cell_size_m, min_rsrp_dbm, overlap_penalty, pathloss_exp
    )

    # KPIs
    covered_pop = float(pop[cov].sum())
    total_pop = float(pop.sum()) if pop.sum() > 0 else 1.0
    coverage_pct = 100.0 * covered_pop / total_pop

    c1, c2, c3 = st.columns(3)
    c1.metric("Coverage % (demand-weighted)", f"{coverage_pct:0.1f}%")
    c2.metric("Covered Demand", f"{covered_pop:,.0f}")
    c3.metric("# Towers", f"{len(towers)}")

    # Convert grid to points for heat
    # Use a step to keep the map responsive on large grids
    step = max(1, int(max(h, w) // 60))
    pts = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            pts.append({
                "lat": float(lat[y, x]),
                "lon": float(lon[y, x]),
                "rsrp": float(prx[y, x]),
                "cov": 1 if cov[y, x] else 0,
                "pop": float(pop[y, x]),
            })
    df_points = pd.DataFrame(pts)

    # Towers as points
    df_towers = pd.DataFrame(
        [{"lat": float(lat[ty, tx]), "lon": float(lon[ty, tx])} for (ty, tx) in towers]
    )

    # Population cells for GridCellLayer (overlay on top)
    df_cells = None
    if show_pop:
        cells = []
        pmin = float(pop.min())
        pmax = float(pop.max()) if float(pop.max()) > 0 else 1.0
        for y in range(0, h, pop_step):
            for x in range(0, w, pop_step):
                val = float(pop[y, x])
                norm = 0.0 if pmax == pmin else (val - pmin) / (pmax - pmin)
                # color ramp: low -> green, high -> red (RGBA, alpha from slider)
                color = [int(255 * norm), int(255 * (1 - norm)), 0, int(pop_alpha)]
                cells.append({
                    "lat": float(lat[y, x]),
                    "lon": float(lon[y, x]),
                    "pop": val,
                    "color": color,
                })
        df_cells = pd.DataFrame(cells)

    # Map view
    center_view = pdk.ViewState(
        latitude=float(lat[h // 2, w // 2]),
        longitude=float(lon[h // 2, w // 2]),
        zoom=12,
        pitch=0,
    )

    # Select base map mode
    mode = st.radio("Map mode", ["Received Power (dBm)", "Coverage (0/1)"], horizontal=True)
    weight_col = "rsrp" if mode.startswith("Received") else "cov"

    # Layers
    heat = pdk.Layer(
        "HeatmapLayer",
        df_points,
        get_position='[lon, lat]',
        get_weight=weight_col,
        radiusPixels=40,
        aggregation='"MEAN"',
    )

    # Population grid overlay (drawn ABOVE the heat layer)
    population_layer = None
    if show_pop and df_cells is not None and len(df_cells) > 0:
        population_layer = pdk.Layer(
            "GridCellLayer",
            df_cells,
            get_position='[lon, lat]',
            cellSize=float(cell_size_m),   # meters; aligns with your grid
            get_fill_color='color',
            pickable=True,
            extruded=False,
        )

    towers_layer = pdk.Layer(
        "ScatterplotLayer",
        df_towers,
        get_position='[lon, lat]',
        get_radius=30,
        pickable=True,
    )

    # Order: heat (base) -> population grid (overlay on top) -> towers (kept visible on top)
    layers = [heat]
    if population_layer is not None:
        layers.append(population_layer)
    layers.append(towers_layer)

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=center_view,
            layers=layers,
            tooltip={"text": "Pop: {pop}\n(RSRP if heatmap mode)"}
        )
    )

    st.subheader("Placement steps (marginal gains)")
    st.dataframe(pd.DataFrame(history))

    with st.expander("Scenario & Model Parameters"):
        st.json({
            "grid_shape": [int(h), int(w)],
            "cell_size_m": float(cell_size_m),
            "num_towers": int(num_towers),
            "freq_mhz": int(freq_mhz),
            "tx_power_dbm": float(tx_power_dbm),
            "min_rsrp_dbm": float(min_rsrp_dbm),
            "overlap_penalty": float(overlap_penalty),
            "profile": profile,
            "pathloss_exponent": float(pathloss_exp),
        })

else:
    st.info("Set your parameters on the left, then click **Run Optimization**.")
