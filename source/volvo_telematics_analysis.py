"""
Exploratory analysis of raw Volvo telematics pings (data/volvo_telematics).

Each monthly CSV is one row per GPS ping:
  VEHICLE_ID, LATITUDE, LONGITUDE, LOCATION_DATE_TIME, STATE, EVENT_CD,
  BATTERY_LEVEL, GEOFENCE_ID

EVENT_CD is dominated by periodic SCHED pings, with STOP/START marking the
start/end of a stationary period (a "stop") and IGNON/IGNOFF/CON/DIS as
sparser ignition and charger connect/disconnect events. There is no speed
column, so speed is estimated from consecutive pings per vehicle.

This is a first look before building any GeoJSON layers: load the raw pings,
restrict them to the GTA (the fleet is Quebec-based with trips into Ontario,
but the Quebec/Montreal-area activity isn't of interest here), print summary
stats, and plot the geographic coverage plus heatmaps of speed, stop density,
and stop duration over an Ontario basemap.

It also flags stop locations with "regular" vehicle visits, as a first-pass
signal for BEV charging-infrastructure siting: only a subset of trucks will
initially transition to BEV, so a location is more useful the more distinct
vehicles independently visit it often enough (see MIN_VISITS_PER_WEEK) to be
individually viable BEV candidates -- not the location with the single most
visits, or the one "dominated" by one truck.

Run as a script (python source/volvo_telematics_analysis.py) or import the
loader/plotting functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "volvo_telematics"
ONTARIO_BOUNDARY_PATH = PROJECT_ROOT / "archived geojsons" / "Ontario_Provincial_Boundary.geojson"
GTA_BOUNDARY_PATH = PROJECT_ROOT / "active geojsons" / "GTA_Boundary.geojson"
PLOTS_DIR = PROJECT_ROOT / "plots"

# GPS jumps implying a speed above this are treated as bad fixes, not travel.
MAX_PLAUSIBLE_SPEED_KMH = 150.0
# Gaps longer than this between pings are too coarse to estimate a speed from.
MAX_PING_GAP_HOURS = 2.0
EARTH_RADIUS_KM = 6371.0088
KM_PER_LAT_DEG = 111.32
# Gaussian smoothing bandwidth for the stop-density KDE surface, and the grid
# resolution it's evaluated on -- mirrors the leaflet.heat layer used for the
# CIMA+ data in the interactive map (radius/blur), but computed with numpy
# since scipy isn't available in this environment.
STOP_DENSITY_BANDWIDTH_KM = 1.5
STOP_DENSITY_GRID_BINS = 300
# Stops beyond this (multi-day, e.g. a vehicle parked out of service) swamp the
# color scale and aren't representative of a normal charging/rest stop.
MAX_STOP_DURATION_FOR_PLOT_MIN = 24 * 60
# Stops shorter than this are treated as noise (e.g. a brief pause at a light
# or in traffic) rather than a real stop.
MIN_STOP_DURATION_MIN = 30
# Grid cell size used to cluster stops into approximate physical locations
# (groups pings within this distance as "the same stop"). A simple grid snap
# stands in for a proper radius-based clustering (e.g. DBSCAN), since neither
# scipy nor sklearn is available in this environment. The regularity counts
# were checked as fairly stable from 0.15-2.0km (see conversation), so this is
# set larger than the smallest plausible "same site" radius mainly so the
# plotted cells are visible at GTA scale, not because finer misses locations.
STOP_LOCATION_GRID_KM = 1.0
# A vehicle "regularly" visits a location if its own visit rate there is at
# least this many times per week -- see MIN_VISITS_PER_WEEK docstring on
# find_regular_stop_locations for why this is per-vehicle, not a share of a
# location's total visits.
MIN_VISITS_PER_WEEK = 3.0
# Swept over in print_regularity_thresholds to see how many locations/vehicles
# still qualify as the bar is raised above MIN_VISITS_PER_WEEK.
VISITS_PER_WEEK_THRESHOLDS = [3, 4, 5, 6, 8, 10]
# For the all-pings version of the regularity analysis (corridor points a
# vehicle repeatedly drives through, not just places it stops): consecutive
# pings from the same vehicle at the same clustered location within this many
# hours of each other count as one visit, not one per 3-minute SCHED ping.
VISIT_SESSION_GAP_HOURS = 1.0


# ---------- Loading ----------

def find_csv_files(data_dir: Path = DATA_DIR) -> List[Path]:
    """All monthly ping CSVs (skips the .csv.gz duplicates and hidden files)."""
    files = sorted(data_dir.glob("*/*/part-*.csv"))
    if not files:
        raise FileNotFoundError(f"No part-*.csv files found under {data_dir}")
    return files


def load_telematics_data(
    files: Optional[List[Path]] = None,
    sample_frac: Optional[float] = None,
) -> pd.DataFrame:
    """Load and concatenate all monthly ping CSVs into one DataFrame.

    sample_frac, if given, randomly subsamples each file's rows (useful to
    speed up exploration before committing to a full run).
    """
    files = files or find_csv_files()
    dtypes = {
        "VEHICLE_ID": str,
        "STATE": str,
        "EVENT_CD": str,
        "BATTERY_LEVEL": float,
        "GEOFENCE_ID": str,
    }
    frames = []
    for csv_path in files:
        df = pd.read_csv(csv_path, dtype=dtypes)
        if sample_frac is not None:
            df = df.sample(frac=sample_frac, random_state=0)
        frames.append(df)
        print(f"[Load] {len(df)} rows from {csv_path.relative_to(PROJECT_ROOT)}")

    df = pd.concat(frames, ignore_index=True)
    df["LOCATION_DATE_TIME"] = pd.to_datetime(
        df["LOCATION_DATE_TIME"], format="%Y%m%d:%H:%M:%S"
    )
    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
    print(f"[Load] {len(df)} total pings, {df['VEHICLE_ID'].nunique()} vehicles, "
          f"{df['LOCATION_DATE_TIME'].min()} to {df['LOCATION_DATE_TIME'].max()}")
    return df


def load_ontario_boundary(path: Path = ONTARIO_BOUNDARY_PATH) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def load_gta_boundary(path: Path = GTA_BOUNDARY_PATH) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def filter_to_gta(df: pd.DataFrame, gta_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Keep only pings that fall within the GTA boundary (its municipalities, unioned)."""
    gta_polygon = shapely.union_all(gta_gdf.geometry.to_numpy())
    shapely.prepare(gta_polygon)

    points = shapely.points(df["LONGITUDE"].to_numpy(), df["LATITUDE"].to_numpy())
    in_gta = shapely.contains(gta_polygon, points)

    filtered = df[in_gta].copy()
    print(f"[GTA Filter] Kept {len(filtered)}/{len(df)} pings within the GTA boundary")
    return filtered


def compute_vehicle_span_weeks(df: pd.DataFrame) -> pd.Series:
    """Weeks between each vehicle's first and last ping in the full (unfiltered)
    dataset -- the denominator for a fair per-vehicle visit rate, so a vehicle
    that's only briefly been in the fleet isn't penalized relative to one
    observed for the whole two years. Clipped to a minimum of one week so a
    vehicle seen only once or twice doesn't produce a divide-by-zero/absurd rate.
    """
    span = df.groupby("VEHICLE_ID")["LOCATION_DATE_TIME"].agg(["min", "max"])
    weeks = (span["max"] - span["min"]).dt.total_seconds() / (7 * 24 * 3600)
    return weeks.clip(lower=1.0)


# ---------- Derived metrics ----------

def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def calculate_speed_kmh(lat, lon, time, prev_lat, prev_lon, prev_time):
    """Speed (km/h) implied by the great-circle distance and elapsed time
    between a point and the previous row's point.

    Accepts either scalars or equal-length arrays/Series. Returns NaN where
    the elapsed time is non-positive or exceeds MAX_PING_GAP_HOURS, or where
    the implied speed exceeds MAX_PLAUSIBLE_SPEED_KMH (a bad GPS fix rather
    than real travel) -- this includes the case where prev_* is NaT/NaN
    (e.g. a vehicle's first ping), since that yields a NaN distance/time.
    """
    dist_km = _haversine_km(prev_lat, prev_lon, lat, lon)
    dt_hours = (time - prev_time) / np.timedelta64(1, "h")
    speed = dist_km / dt_hours

    implausible = (dt_hours <= 0) | (dt_hours > MAX_PING_GAP_HOURS) | (speed > MAX_PLAUSIBLE_SPEED_KMH)
    if np.ndim(speed) == 0:
        return float("nan") if bool(implausible) else float(speed)
    return np.where(implausible, np.nan, speed)


def add_speed_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate km/h between each ping and the previous ping from the same vehicle."""
    df = df.sort_values(["VEHICLE_ID", "LOCATION_DATE_TIME"]).copy()
    grp = df.groupby("VEHICLE_ID")
    prev_lat = grp["LATITUDE"].shift()
    prev_lon = grp["LONGITUDE"].shift()
    prev_time = grp["LOCATION_DATE_TIME"].shift()

    df["SPEED_KMH"] = calculate_speed_kmh(
        df["LATITUDE"], df["LONGITUDE"], df["LOCATION_DATE_TIME"],
        prev_lat, prev_lon, prev_time,
    )

    n_valid = df["SPEED_KMH"].notna().sum()
    print(f"[Speed] Estimated speed for {n_valid}/{len(df)} pings "
          f"(mean {df['SPEED_KMH'].mean():.1f} km/h)")
    return df


def extract_stops(
    df: pd.DataFrame,
    min_duration_min: float = MIN_STOP_DURATION_MIN,
) -> pd.DataFrame:
    """Stop location + duration for each STOP event, ended by that vehicle's next START.

    STOP/START events mark the boundaries of stationary periods; a STOP not
    followed by a START (e.g. end of the data window) has no known duration
    and is dropped. Stops shorter than min_duration_min are also dropped, as
    noise rather than a real stop (e.g. a light or momentary traffic pause).
    """
    events = df[df["EVENT_CD"].isin(["STOP", "START"])].sort_values(
        ["VEHICLE_ID", "LOCATION_DATE_TIME"]
    ).copy()
    grp = events.groupby("VEHICLE_ID")
    next_event = grp["EVENT_CD"].shift(-1)
    next_time = grp["LOCATION_DATE_TIME"].shift(-1)

    stops = events[(events["EVENT_CD"] == "STOP") & (next_event == "START")].copy()
    stops["STOP_DURATION_MIN"] = (
        next_time[stops.index] - stops["LOCATION_DATE_TIME"]
    ).dt.total_seconds() / 60.0

    too_short = (stops["STOP_DURATION_MIN"] < min_duration_min).sum()
    stops = stops[stops["STOP_DURATION_MIN"] >= min_duration_min].copy()
    print(f"[Stops] {len(stops)} completed stops of at least {min_duration_min:.0f} min "
          f"(dropped {too_short} shorter stops; median duration "
          f"{stops['STOP_DURATION_MIN'].median():.1f} min)")
    return stops


# ---------- Repeat-visit analysis ----------

def assign_location_clusters(
    stops: pd.DataFrame,
    grid_km: float = STOP_LOCATION_GRID_KM,
) -> pd.DataFrame:
    """Snap each stop to a coarse lat/lon grid cell, as a stand-in for a proper
    radius-based clustering (e.g. DBSCAN) -- groups stops a few hundred meters
    apart (GPS jitter, slightly different parking spots at the same site) as
    the same physical location. No scipy/sklearn available in this environment.
    """
    stops = stops.copy()
    lat_mid = stops["LATITUDE"].mean()
    km_per_lon_deg = KM_PER_LAT_DEG * np.cos(np.radians(lat_mid))
    lat_cell_deg = grid_km / KM_PER_LAT_DEG
    lon_cell_deg = grid_km / km_per_lon_deg

    lat_idx = (stops["LATITUDE"] / lat_cell_deg).round().astype(int)
    lon_idx = (stops["LONGITUDE"] / lon_cell_deg).round().astype(int)
    stops["LOCATION_CLUSTER"] = lat_idx.astype(str) + "_" + lon_idx.astype(str)
    return stops


def extract_location_visits(
    pings: pd.DataFrame,
    grid_km: float = STOP_LOCATION_GRID_KM,
    session_gap_hours: float = VISIT_SESSION_GAP_HOURS,
) -> pd.DataFrame:
    """One row per (vehicle, location-cluster, visit) from ALL pings, not just
    STOP events -- so a corridor point a vehicle repeatedly drives through
    (without ever stopping there) counts as a visited location too.

    Consecutive pings from the same vehicle at the same clustered location are
    collapsed into a single visit whenever they're within session_gap_hours of
    each other; otherwise a vehicle idling or driving slowly through one grid
    cell would rack up one "visit" per 3-minute SCHED ping.
    """
    pings = assign_location_clusters(pings, grid_km=grid_km)
    pings = pings.sort_values(["VEHICLE_ID", "LOCATION_CLUSTER", "LOCATION_DATE_TIME"])

    grp = pings.groupby(["VEHICLE_ID", "LOCATION_CLUSTER"])
    gap_hours = grp["LOCATION_DATE_TIME"].diff() / np.timedelta64(1, "h")
    new_visit = gap_hours.isna() | (gap_hours > session_gap_hours)
    pings["_VISIT_NUM"] = new_visit.groupby([pings["VEHICLE_ID"], pings["LOCATION_CLUSTER"]]).cumsum()

    visits = (
        pings.groupby(["VEHICLE_ID", "LOCATION_CLUSTER", "_VISIT_NUM"])
        .agg(LATITUDE=("LATITUDE", "mean"), LONGITUDE=("LONGITUDE", "mean"))
        .reset_index()
    )
    print(f"[Visits] {len(visits)} visit sessions from {len(pings)} pings "
          f"(gap > {session_gap_hours:.1f}h starts a new visit)")
    return visits


def find_regular_locations(
    visits: pd.DataFrame,
    vehicle_span_weeks: pd.Series,
    min_visits_per_week: float = MIN_VISITS_PER_WEEK,
    label: str = "",
) -> pd.DataFrame:
    """For each clustered location, how many distinct vehicles visit it
    "regularly" -- i.e. that vehicle's own visit rate there is at least
    min_visits_per_week -- rather than how concentrated visits are among a
    single dominant vehicle.

    That distinction matters for BEV siting: a location with 10 vehicles each
    independently visiting 3x/week is a better charging-infrastructure
    candidate than one vehicle visiting 6x/week alone, because it could serve
    more vehicles' transitions -- but only once each vehicle's own cadence
    clears whatever regularity bar makes BEV operationally viable for it.

    visits is one row per (vehicle, location, visit) -- either stops (via
    assign_location_clusters) or all-ping visit sessions (via
    extract_location_visits).
    """
    counts = (
        visits.groupby(["LOCATION_CLUSTER", "VEHICLE_ID"])
        .agg(VISITS=("VEHICLE_ID", "size"),
             LATITUDE=("LATITUDE", "mean"),
             LONGITUDE=("LONGITUDE", "mean"))
        .reset_index()
    )
    counts["VISITS_PER_WEEK"] = counts["VISITS"] / counts["VEHICLE_ID"].map(vehicle_span_weeks)
    counts["IS_REGULAR"] = counts["VISITS_PER_WEEK"] >= min_visits_per_week

    locations = counts.groupby("LOCATION_CLUSTER").agg(
        LATITUDE=("LATITUDE", "mean"),
        LONGITUDE=("LONGITUDE", "mean"),
        TOTAL_VISITS=("VISITS", "sum"),
        TOTAL_VEHICLES=("VEHICLE_ID", "nunique"),
        REGULAR_VEHICLES=("IS_REGULAR", "sum"),
    ).reset_index()
    locations = locations.sort_values(
        ["REGULAR_VEHICLES", "TOTAL_VISITS"], ascending=False
    ).reset_index(drop=True)

    n_qualifying = (locations["REGULAR_VEHICLES"] >= 1).sum()
    tag = f"[{label}] " if label else ""
    print(f"[Regularity] {tag}{n_qualifying}/{len(locations)} locations have >= 1 vehicle "
          f"visiting at >= {min_visits_per_week:.0f}/week")
    return locations


def print_regularity_thresholds(
    visits: pd.DataFrame,
    vehicle_span_weeks: pd.Series,
    thresholds=VISITS_PER_WEEK_THRESHOLDS,
    label: str = "",
) -> None:
    """How the count of qualifying (vehicle, location) pairs and locations
    changes as the visits/week bar is raised -- to see how far the "at least
    N regular vehicles per location" idea holds up before it gets too strict.
    """
    counts = (
        visits.groupby(["LOCATION_CLUSTER", "VEHICLE_ID"]).size()
        .rename("VISITS").reset_index()
    )
    counts["VISITS_PER_WEEK"] = counts["VISITS"] / counts["VEHICLE_ID"].map(vehicle_span_weeks)

    tag = f" ({label})" if label else ""
    print(f"\n[Regularity]{tag} Qualifying locations by visits/week threshold:")
    print(f"  {'threshold':>10}  {'pairs':>6}  {'>=1 vehicle':>12}  {'>=2 vehicles':>13}")
    for t in thresholds:
        regular = counts[counts["VISITS_PER_WEEK"] >= t]
        per_location = regular.groupby("LOCATION_CLUSTER")["VEHICLE_ID"].nunique()
        n_any = (per_location >= 1).sum()
        n_shared = (per_location >= 2).sum()
        print(f"  {t:>8.1f}/wk  {len(regular):>6d}  {n_any:>12d}  {n_shared:>13d}")


# ---------- Plotting ----------

def _padded_extent(lon, lat, pad_frac: float = 0.1):
    """Axis limits covering the given points, padded by a fraction of their range."""
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    lon_pad = (lon_max - lon_min) * pad_frac or 0.5
    lat_pad = (lat_max - lat_min) * pad_frac or 0.5
    return (lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad)


def _new_basemap_axes(basemap_gdf: gpd.GeoDataFrame, title: str, extent=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    basemap_gdf.plot(ax=ax, color="#F0F0F0", edgecolor="#333333", linewidth=1, alpha=0.5, zorder=0)
    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return fig, ax


def plot_coverage_map(
    df: pd.DataFrame,
    basemap_gdf: gpd.GeoDataFrame,
    output_path: Path = PLOTS_DIR / "volvo_coverage_map.png",
) -> Path:
    """Ping density per grid cell (binned counts, no smoothing/interpolation)."""
    extent = _padded_extent(df["LONGITUDE"], df["LATITUDE"])
    fig, ax = _new_basemap_axes(basemap_gdf, "Volvo Telematics Ping Coverage", extent)
    hb = ax.hexbin(
        df["LONGITUDE"], df["LATITUDE"],
        gridsize=150, cmap="inferno", bins="log", mincnt=1, zorder=1,
    )
    fig.colorbar(hb, ax=ax, label="Ping count (log scale)", shrink=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved {output_path}")
    return output_path


def plot_speed_heatmap(
    df: pd.DataFrame,
    basemap_gdf: gpd.GeoDataFrame,
    output_path: Path = PLOTS_DIR / "volvo_speed_heatmap.png",
) -> Path:
    """Mean estimated speed per grid cell."""
    valid = df.dropna(subset=["SPEED_KMH"])
    extent = _padded_extent(valid["LONGITUDE"], valid["LATITUDE"])
    fig, ax = _new_basemap_axes(basemap_gdf, "Average Speed (km/h)", extent)
    hb = ax.hexbin(
        valid["LONGITUDE"], valid["LATITUDE"], C=valid["SPEED_KMH"],
        reduce_C_function=np.mean, gridsize=150, cmap="viridis", mincnt=3, zorder=1,
    )
    fig.colorbar(hb, ax=ax, label="Average speed (km/h)", shrink=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved {output_path}")
    return output_path


def _gaussian_kernel_1d(sigma_px: float) -> np.ndarray:
    radius = max(1, int(round(4 * sigma_px)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma_px) ** 2)
    return kernel / kernel.sum()


def _smoothed_density_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    extent,
    bins: int = STOP_DENSITY_GRID_BINS,
    bandwidth_km: float = STOP_DENSITY_BANDWIDTH_KM,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Point counts (optionally weighted) binned to a grid, then Gaussian-blurred
    into a continuous density surface -- a numpy-only stand-in for a proper KDE
    (no scipy here), in the same spirit as the leaflet.heat layer used for the
    CIMA+ data. weights lets e.g. a location with more qualifying vehicles
    contribute more to the surface than one point per location would.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    hist, _, _ = np.histogram2d(
        lon, lat, bins=bins, range=[[lon_min, lon_max], [lat_min, lat_max]],
        weights=weights,
    )
    grid = hist.T  # rows = lat, cols = lon, to match imshow's (row, col) convention

    lat_mid = (lat_min + lat_max) / 2
    km_per_lon_deg = KM_PER_LAT_DEG * np.cos(np.radians(lat_mid))
    sigma_x_px = bandwidth_km / ((lon_max - lon_min) / bins * km_per_lon_deg)
    sigma_y_px = bandwidth_km / ((lat_max - lat_min) / bins * KM_PER_LAT_DEG)

    kernel_x = _gaussian_kernel_1d(sigma_x_px)
    kernel_y = _gaussian_kernel_1d(sigma_y_px)
    grid = np.apply_along_axis(lambda row: np.convolve(row, kernel_x, mode="same"), 1, grid)
    grid = np.apply_along_axis(lambda col: np.convolve(col, kernel_y, mode="same"), 0, grid)
    return grid


def plot_stop_density_heatmap(
    stops: pd.DataFrame,
    basemap_gdf: gpd.GeoDataFrame,
    output_path: Path = PLOTS_DIR / "volvo_stop_density_heatmap.png",
    bandwidth_km: float = STOP_DENSITY_BANDWIDTH_KM,
    gamma: float = 0.45,
) -> Path:
    """Smoothed stop-density surface (Gaussian-blurred + bilinearly interpolated),
    the same style of continuous heatmap used for the CIMA+ stop data.

    Rendered as an RGBA image whose alpha channel fades smoothly with density
    (gamma-adjusted so mid/low density areas are still visible) rather than a
    log color scale -- a log norm would exaggerate the Gaussian kernel's tiny
    tail values and its hard-masked cutoff at exact zero would read as blocky
    edges instead of the soft glow this is meant to look like.
    """
    extent = _padded_extent(stops["LONGITUDE"], stops["LATITUDE"])
    grid = _smoothed_density_grid(
        stops["LONGITUDE"].to_numpy(), stops["LATITUDE"].to_numpy(), extent,
        bandwidth_km=bandwidth_km,
    )
    intensity = np.clip(grid / grid.max(), 0, 1) ** gamma

    cmap = plt.get_cmap("inferno")
    rgba = cmap(intensity)
    rgba[..., 3] = intensity

    fig, ax = _new_basemap_axes(basemap_gdf, "Stop Density (smoothed)", extent)
    ax.imshow(rgba, extent=extent, origin="lower", interpolation="bilinear", zorder=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, ax=ax, label="Stop density, smoothed (relative)", shrink=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved {output_path}")
    return output_path


def plot_stop_duration_heatmap(
    stops: pd.DataFrame,
    basemap_gdf: gpd.GeoDataFrame,
    output_path: Path = PLOTS_DIR / "volvo_stop_duration_heatmap.png",
    max_duration_min: float = MAX_STOP_DURATION_FOR_PLOT_MIN,
) -> Path:
    """Median stop duration per grid cell.

    Stops longer than max_duration_min are excluded from the color scale (but
    still counted in the stop-density plot) so a handful of multi-day parked
    vehicles don't wash out the normal range.
    """
    plotted = stops[stops["STOP_DURATION_MIN"] <= max_duration_min]
    excluded = len(stops) - len(plotted)
    if excluded:
        print(f"[Plot] Excluding {excluded} stops longer than "
              f"{max_duration_min:.0f} min from the duration heatmap")

    extent = _padded_extent(plotted["LONGITUDE"], plotted["LATITUDE"])
    fig, ax = _new_basemap_axes(basemap_gdf, "Average Stop Duration (minutes)", extent)
    hb = ax.hexbin(
        plotted["LONGITUDE"], plotted["LATITUDE"], C=plotted["STOP_DURATION_MIN"],
        reduce_C_function=np.median, gridsize=150, cmap="plasma", mincnt=3, zorder=1,
    )
    fig.colorbar(hb, ax=ax, label="Median stop duration (min)", shrink=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved {output_path}")
    return output_path


def plot_regular_locations(
    locations: pd.DataFrame,
    basemap_gdf: gpd.GeoDataFrame,
    output_path: Path,
    min_visits_per_week: float = MIN_VISITS_PER_WEEK,
    title: str = "Locations with Regular Visitors",
    bandwidth_km: float = STOP_DENSITY_BANDWIDTH_KM,
    gamma: float = 0.45,
) -> Optional[Path]:
    """Smoothed heatmap of locations with at least one regularly-visiting
    vehicle, in the same Gaussian-blurred glow style as the stop-density
    heatmap -- but weighted by REGULAR_VEHICLES per location (via
    _smoothed_density_grid's weights), rather than one unweighted point per
    location, so a location with more qualifying vehicles stands out more and
    an isolated single-vehicle location is still visible as a soft blob
    instead of a near-invisible point at GTA scale.

    Unlike stop density, REGULAR_VEHICLES is a hard threshold result tied to a
    specific grid cell, not an inherently continuous quantity -- the blur here
    is a legibility aid, not a claim that the surrounding area also qualifies.
    """
    qualifying = locations[locations["REGULAR_VEHICLES"] >= 1]
    if qualifying.empty:
        print(f"[Plot] No locations with >= 1 vehicle at >= {min_visits_per_week:.0f}/week; "
              f"skipping {output_path.name}")
        return None

    extent = _padded_extent(qualifying["LONGITUDE"], qualifying["LATITUDE"])
    grid = _smoothed_density_grid(
        qualifying["LONGITUDE"].to_numpy(), qualifying["LATITUDE"].to_numpy(), extent,
        bandwidth_km=bandwidth_km, weights=qualifying["REGULAR_VEHICLES"].to_numpy(),
    )
    intensity = np.clip(grid / grid.max(), 0, 1) ** gamma

    cmap = plt.get_cmap("viridis")
    rgba = cmap(intensity)
    rgba[..., 3] = intensity

    fig, ax = _new_basemap_axes(
        basemap_gdf, f"{title} (>= {min_visits_per_week:.0f} visits/week)", extent,
    )
    ax.imshow(rgba, extent=extent, origin="lower", interpolation="bilinear", zorder=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, ax=ax, label="Regular vehicles, smoothed (relative)", shrink=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved {output_path}")
    return output_path


# ---------- Summary ----------

def print_summary(df: pd.DataFrame) -> None:
    print("\n[Summary] Rows by STATE:")
    print(df["STATE"].value_counts().to_string())
    print("\n[Summary] Rows by EVENT_CD:")
    print(df["EVENT_CD"].value_counts().to_string())
    print("\n[Summary] Pings per vehicle:")
    print(df.groupby("VEHICLE_ID").size().describe().to_string())


# ---------- Orchestration ----------

def main(sample_frac: Optional[float] = None) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    basemap_gdf = load_ontario_boundary()
    gta_gdf = load_gta_boundary()
    df = load_telematics_data(sample_frac=sample_frac)
    vehicle_span_weeks = compute_vehicle_span_weeks(df)
    df = filter_to_gta(df, gta_gdf)
    print_summary(df)

    df = add_speed_estimates(df)
    stops = extract_stops(df)

    plot_coverage_map(df, basemap_gdf)
    plot_speed_heatmap(df, basemap_gdf)
    plot_stop_density_heatmap(stops, basemap_gdf)
    plot_stop_duration_heatmap(stops, basemap_gdf)

    stop_visits = assign_location_clusters(stops)
    print_regularity_thresholds(stop_visits, vehicle_span_weeks, label="stops")
    stop_locations = find_regular_locations(stop_visits, vehicle_span_weeks, label="stops")
    plot_regular_locations(
        stop_locations, basemap_gdf, PLOTS_DIR / "volvo_regular_stop_locations.png",
        title="Stop Locations with Regular Visitors",
    )

    ping_visits = extract_location_visits(df)
    print_regularity_thresholds(ping_visits, vehicle_span_weeks, label="all pings")
    ping_locations = find_regular_locations(ping_visits, vehicle_span_weeks, label="all pings")
    plot_regular_locations(
        ping_locations, basemap_gdf, PLOTS_DIR / "volvo_regular_ping_locations.png",
        title="All-Ping Locations with Regular Visitors",
    )


if __name__ == "__main__":
    main()
