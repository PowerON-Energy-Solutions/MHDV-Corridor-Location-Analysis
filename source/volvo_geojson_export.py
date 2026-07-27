"""
Convert Volvo telematics analysis outputs (source/volvo_telematics_analysis.py)
into GeoJSON layers for the interactive map (viewer/geojson_viewer.html).

Three rendering styles, matching what's already used for the CIMA+ data (see
source/interactive_map.py's LAYER_CONFIG for how each file below is wired in):

  - "Choropleth" grid layer (stop duration): each feature is a flat-colored
    square grid cell with no interpolation between cells, so the map reads the
    same way the source hexbin plot does -- a density/value color map, not a
    smoothed surface.
  - "Heatmap" point layers (stop density, regular-visitor locations): one
    Point per stop/location with a numeric weight property. The smoothing
    happens client-side via the same leaflet.heat layer (radius/blur, default
    gradient) already used for cima_stops_heat.geojson, so these render with
    the same "2D interpolated" look, not a pre-baked raster image.
  - "Line edge" layer (ping density, average speed): consecutive-in-time pings
    from the same vehicle are snapped to grid cells and connected into edges
    between distinct cells; ping-transitions between the same two cells
    (either direction) are merged into one undirected edge. Ping density and
    average speed share this same edge geometry/width (PingCount), just
    colored differently -- see build_ping_edges.

Run as a script (python source/volvo_geojson_export.py) or import the
builders.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from volvo_telematics_analysis import (
    PROJECT_ROOT,
    KM_PER_LAT_DEG,
    MIN_VISITS_PER_WEEK,
    MAX_STOP_DURATION_FOR_PLOT_MIN,
    load_telematics_data,
    load_gta_boundary,
    filter_to_gta,
    compute_vehicle_span_weeks,
    add_speed_estimates,
    extract_stops,
    extract_location_visits,
    find_regular_locations,
)

GEOJSON_DIR = PROJECT_ROOT / "active geojsons"
COORD_DECIMALS = 6

# Grid cell size for the stop-duration choropleth -- finer than the
# STOP_LOCATION_GRID_KM clustering grid, since it's meant to reproduce the
# original hexbin plot's resolution, not to group distinct sites together.
CHOROPLETH_GRID_KM = 0.5
# Grid cell size for snapping ping-edge endpoints (build_ping_edges). Most
# consecutive-ping hops turn out to be very short (median ~1m -- the fleet is
# mostly idling, not driving, between SCHED pings), so at the finer
# CHOROPLETH_GRID_KM resolution most surviving (non-same-cell) edges are just
# GPS jitter drifting across a cell boundary while stationary, each snapping
# to a different pair of neighbors -- not real shared road usage. A coarser
# grid here is a deliberate "proximity" requirement: two trips only merge
# into one edge if they cover roughly the same stretch of road, not the same
# few meters.
EDGE_GRID_KM = 1.5


def _round(value: float) -> float:
    return round(float(value), COORD_DECIMALS)


def _grid_indices(lat: pd.Series, lon: pd.Series, grid_km: float):
    lat_mid = lat.mean()
    km_per_lon_deg = KM_PER_LAT_DEG * np.cos(np.radians(lat_mid))
    lat_cell_deg = grid_km / KM_PER_LAT_DEG
    lon_cell_deg = grid_km / km_per_lon_deg
    lat_idx = (lat / lat_cell_deg).round().astype(int)
    lon_idx = (lon / lon_cell_deg).round().astype(int)
    return lat_idx, lon_idx, lat_cell_deg, lon_cell_deg


def _grid_cell_polygon(lat_idx: int, lon_idx: int, lat_cell_deg: float, lon_cell_deg: float) -> dict:
    lon_min = _round((lon_idx - 0.5) * lon_cell_deg)
    lon_max = _round((lon_idx + 0.5) * lon_cell_deg)
    lat_min = _round((lat_idx - 0.5) * lat_cell_deg)
    lat_max = _round((lat_idx + 0.5) * lat_cell_deg)
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon_min, lat_min], [lon_max, lat_min],
            [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min],
        ]],
    }


def _write_geojson(collection: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(collection, f, separators=(",", ":"))
    print(f"[Write] {len(collection['features'])} features -> {out_path} "
          f"({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


# ---------- Line edge layer (ping density, average speed) ----------

def build_ping_edges(df: pd.DataFrame, grid_km: float = EDGE_GRID_KM):
    """One row per undirected grid-cell-to-cell edge, connecting consecutive-
    in-time pings from the same vehicle.

    df must already have SPEED_KMH (from add_speed_estimates): that column's
    own NaN-ing of implausible jumps/large gaps (see calculate_speed_kmh) is
    reused here to decide which consecutive-ping pairs become an edge at all,
    so a segment spanning a multi-hour gap or an impossible speed is dropped
    rather than drawn as a straight line across it.

    Consecutive pings that snap to the same cell (e.g. a vehicle idling) are
    dropped -- that's dwelling, not travel, and already covered by the stop
    layers. A->B and B->A are merged into one edge (min/max cell ordering),
    weighted by PingCount (how many ping-transitions map to that edge) with
    AvgSpeedKmh averaged over those same transitions.
    """
    df = df.sort_values(["VEHICLE_ID", "LOCATION_DATE_TIME"]).copy()
    lat_idx, lon_idx, lat_cell_deg, lon_cell_deg = _grid_indices(
        df["LATITUDE"], df["LONGITUDE"], grid_km
    )
    df["LAT_IDX"] = lat_idx
    df["LON_IDX"] = lon_idx

    grp = df.groupby("VEHICLE_ID")
    prev_lat_idx = grp["LAT_IDX"].shift()
    prev_lon_idx = grp["LON_IDX"].shift()

    valid = df["SPEED_KMH"].notna() & prev_lat_idx.notna()
    same_cell = (df["LAT_IDX"] == prev_lat_idx) & (df["LON_IDX"] == prev_lon_idx)
    edges = df[valid & ~same_cell].copy()
    edges["PREV_LAT_IDX"] = prev_lat_idx[valid & ~same_cell].astype(int)
    edges["PREV_LON_IDX"] = prev_lon_idx[valid & ~same_cell].astype(int)

    # Canonical (from, to) ordering -- lexicographically smaller endpoint
    # first -- so a trip A->B and another trip B->A land on the same edge.
    a_lat, a_lon = edges["LAT_IDX"].to_numpy(), edges["LON_IDX"].to_numpy()
    b_lat, b_lon = edges["PREV_LAT_IDX"].to_numpy(), edges["PREV_LON_IDX"].to_numpy()
    swap = (a_lat > b_lat) | ((a_lat == b_lat) & (a_lon > b_lon))
    edges["FROM_LAT_IDX"] = np.where(swap, b_lat, a_lat)
    edges["FROM_LON_IDX"] = np.where(swap, b_lon, a_lon)
    edges["TO_LAT_IDX"] = np.where(swap, a_lat, b_lat)
    edges["TO_LON_IDX"] = np.where(swap, a_lon, b_lon)

    grouped = (
        edges.groupby(["FROM_LAT_IDX", "FROM_LON_IDX", "TO_LAT_IDX", "TO_LON_IDX"])["SPEED_KMH"]
        .agg(PingCount="size", AvgSpeedKmh="mean")
        .reset_index()
    )
    print(f"[Edges] {len(grouped)} unique cell-to-cell edges from {len(edges)} "
          f"valid ping-transitions ({grid_km:.1f}km grid)")
    return grouped, lat_cell_deg, lon_cell_deg


def export_ping_edges(
    edges: pd.DataFrame,
    lat_cell_deg: float,
    lon_cell_deg: float,
    out_path: Path = GEOJSON_DIR / "volvo_ping_edges.geojson",
) -> Path:
    """LineString per edge from build_ping_edges, with PingCount (width, for
    both layers) and AvgSpeedKmh (color, for the average-speed layer) properties.
    """

    def cell_center(lat_idx, lon_idx):
        return [_round(lon_idx * lon_cell_deg), _round(lat_idx * lat_cell_deg)]

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    cell_center(row.FROM_LAT_IDX, row.FROM_LON_IDX),
                    cell_center(row.TO_LAT_IDX, row.TO_LON_IDX),
                ],
            },
            "properties": {
                "PingCount": int(row.PingCount),
                "AvgSpeedKmh": round(float(row.AvgSpeedKmh), 1),
            },
        }
        for row in edges.itertuples()
    ]
    collection = {
        "type": "FeatureCollection",
        "scaleMax": int(edges["PingCount"].max()),
        "speedMin": round(float(edges["AvgSpeedKmh"].min()), 1),
        "speedMax": round(float(edges["AvgSpeedKmh"].max()), 1),
        "features": features,
    }
    return _write_geojson(collection, out_path)


# ---------- Heatmap point layers (stop density, regular visitors) ----------

def export_stop_density_heat(
    stops: pd.DataFrame,
    out_path: Path = GEOJSON_DIR / "volvo_stop_density_heat.geojson",
) -> Path:
    """One point per stop, weight 1 each -- leaflet.heat does the KDE-style
    smoothing client-side, the same mechanism as cima_stops_heat.geojson.
    """
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [_round(lon), _round(lat)]},
            "properties": {"StopCount": 1},
        }
        for lon, lat in zip(stops["LONGITUDE"], stops["LATITUDE"])
    ]
    collection = {"type": "FeatureCollection", "scaleMax": 1, "features": features}
    return _write_geojson(collection, out_path)


def export_stop_duration_choropleth(
    stops: pd.DataFrame,
    out_path: Path = GEOJSON_DIR / "volvo_stop_duration.geojson",
    grid_km: float = CHOROPLETH_GRID_KM,
    max_duration_min: float = MAX_STOP_DURATION_FOR_PLOT_MIN,
) -> Path:
    """Median stop duration per grid cell -- ping/grid-based like coverage and
    speed (flat-colored cells, no smoothing), matching plot_stop_duration_heatmap's
    hexbin exactly: median per cell, multi-day outliers excluded from the scale.
    """
    plotted = stops[stops["STOP_DURATION_MIN"] <= max_duration_min]
    excluded = len(stops) - len(plotted)
    if excluded:
        print(f"[Stop Duration] Excluding {excluded} stops longer than "
              f"{max_duration_min:.0f} min")

    lat_idx, lon_idx, lat_cell_deg, lon_cell_deg = _grid_indices(
        plotted["LATITUDE"], plotted["LONGITUDE"], grid_km
    )
    grouped = (
        plotted.assign(LAT_IDX=lat_idx, LON_IDX=lon_idx)
        .groupby(["LAT_IDX", "LON_IDX"])["STOP_DURATION_MIN"]
        .agg(MedianStopDurationMin="median", StopCount="size")
        .reset_index()
    )
    features = [
        {
            "type": "Feature",
            "geometry": _grid_cell_polygon(row.LAT_IDX, row.LON_IDX, lat_cell_deg, lon_cell_deg),
            "properties": {
                "MedianStopDurationMin": round(float(row.MedianStopDurationMin), 1),
                "StopCount": int(row.StopCount),
            },
        }
        for row in grouped.itertuples()
    ]
    collection = {
        "type": "FeatureCollection",
        "scaleMax": round(float(grouped["MedianStopDurationMin"].max()), 1),
        "features": features,
    }
    return _write_geojson(collection, out_path)


def export_regular_locations_heat(
    locations: pd.DataFrame,
    out_path: Path = GEOJSON_DIR / "volvo_regular_ping_locations_heat.geojson",
    min_visits_per_week: float = MIN_VISITS_PER_WEEK,
) -> Optional[Path]:
    """One point per location with >= 1 regularly-visiting vehicle, weighted by
    REGULAR_VEHICLES kept as an absolute count -- NOT normalized to a 0-1
    relative scale -- so the map layer reflects the real number of vehicles.
    """
    qualifying = locations[locations["REGULAR_VEHICLES"] >= 1]
    if qualifying.empty:
        print(f"[Regular Locations] No locations with >= 1 vehicle at >= "
              f"{min_visits_per_week:.0f}/week; skipping {out_path.name}")
        return None
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [_round(row.LONGITUDE), _round(row.LATITUDE)]},
            "properties": {"RegularVehicles": int(row.REGULAR_VEHICLES)},
        }
        for row in qualifying.itertuples()
    ]
    collection = {
        "type": "FeatureCollection",
        "scaleMax": int(qualifying["REGULAR_VEHICLES"].max()),
        "features": features,
    }
    return _write_geojson(collection, out_path)


# ---------- Orchestration ----------

def main() -> None:
    gta_gdf = load_gta_boundary()
    df = load_telematics_data()
    vehicle_span_weeks = compute_vehicle_span_weeks(df)
    df = filter_to_gta(df, gta_gdf)
    df = add_speed_estimates(df)

    edges, lat_cell_deg, lon_cell_deg = build_ping_edges(df)
    export_ping_edges(edges, lat_cell_deg, lon_cell_deg)

    stops = extract_stops(df)
    export_stop_density_heat(stops)
    export_stop_duration_choropleth(stops)

    ping_visits = extract_location_visits(df)
    ping_locations = find_regular_locations(ping_visits, vehicle_span_weeks, label="all pings")
    export_regular_locations_heat(ping_locations)


if __name__ == "__main__":
    main()
