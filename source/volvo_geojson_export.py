"""
Convert Volvo telematics analysis outputs (source/volvo_telematics_analysis.py)
into GeoJSON layers for the interactive map (viewer/geojson_viewer.html).

Three rendering styles, matching what's already used for the CIMA+ data (see
source/interactive_map.py's LAYER_CONFIG for how each file below is wired in):

  - "Choropleth" grid layers (stop density, stop duration): each feature is a
    flat-colored square grid cell with no interpolation between cells, so the
    map reads the same way the source hexbin plots do -- a density/value
    color map, not a smoothed surface.
  - "Heatmap" point layer (regular-visitor locations): one Point per location
    with a numeric weight property. The smoothing happens client-side via the
    same leaflet.heat layer (radius/blur, default gradient) already used for
    cima_stops_heat.geojson, so it renders with the same "2D interpolated"
    look, not a pre-baked raster image.
  - "Line edge" layer (ping density, average speed): consecutive-in-time pings
    from the same vehicle are each snapped either to the nearest vertex of the
    existing freight-corridor/highway network (see load_road_vertices) if
    within ROAD_SNAP_THRESHOLD_KM, or otherwise to a coarse grid cell, and
    connected into edges between distinct locations; ping-transitions between
    the same two locations (either direction) are merged into one undirected
    edge. This is a lightweight stand-in for full GPS map-matching -- no new
    road data or routing engine, just a nearest-point lookup against a road
    network already in this repo -- so trips actually on a mapped highway
    consolidate onto that highway's own geometry instead of a straight chord
    between two arbitrary grid cells; local streets (not in that network)
    still fall back to the grid. Ping density and average speed share this
    same edge geometry/width (PingCount), just colored differently -- see
    build_ping_edges.

Run as a script (python source/volvo_geojson_export.py) or import the
builders.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

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
# Fallback grid cell size for ping-edge endpoints that aren't near a mapped
# road (see ROAD_NETWORK_PATH/ROAD_SNAP_THRESHOLD_KM below). Most
# consecutive-ping hops turn out to be very short (median ~1m -- the fleet is
# mostly idling, not driving, between SCHED pings), so at a finer resolution
# most surviving (non-same-location) edges would just be GPS jitter drifting
# across a cell boundary while stationary, each snapping to a different pair
# of neighbors -- not real shared road usage. A coarser grid here is a
# deliberate "proximity" requirement: two trips only merge into one edge if
# they cover roughly the same stretch of ground, not the same few meters.
EDGE_GRID_KM = 1.5
# Freight-corridor/highway network already in the repo (also used for the
# "OpenData: Primary Freight Corridors" map layer), reused here as a
# lightweight road reference to snap ping-edge endpoints onto.
ROAD_NETWORK_PATH = GEOJSON_DIR / "Value_of_Goods_2016_-3795296180828259022.geojson"
# A ping within this distance of the nearest road-network vertex is snapped
# onto it; farther away (e.g. a local street not in this network) falls back
# to the EDGE_GRID_KM grid.
ROAD_SNAP_THRESHOLD_KM = 0.2


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

def load_road_vertices(
    gta_gdf: gpd.GeoDataFrame,
    path: Path = ROAD_NETWORK_PATH,
    buffer_km: float = 2.0,
    simplify_km: float = EDGE_GRID_KM,
) -> tuple:
    """Distinct vertices from the freight-corridor/highway network (already
    used elsewhere as the "Primary Freight Corridors" map layer), within a
    small buffer of the GTA -- the point cloud build_ping_edges snaps nearby
    pings onto, as a lightweight stand-in for full GPS map-matching.

    The raw geometry's vertices are very unevenly spaced (a sample of segments
    had a median gap of ~320m but a 25th percentile of only ~75m, i.e. lots of
    vertices clustered tightly at bends/intersections), which is too fine to
    use directly as merge keys -- two pings a block apart on the same highway
    would often snap to two different nearby-but-distinct vertices instead of
    consolidating. Simplifying first (Douglas-Peucker, tolerance simplify_km)
    reduces each road to vertices spaced roughly that far apart, so nearby
    pings are far more likely to share the same nearest vertex.
    """
    roads = gpd.read_file(path)
    gta_polygon = shapely.union_all(gta_gdf.geometry.to_numpy())
    buffer_deg = buffer_km / KM_PER_LAT_DEG
    roads = roads[roads.intersects(gta_polygon.buffer(buffer_deg))]
    roads = roads.assign(geometry=roads.geometry.simplify(simplify_km / KM_PER_LAT_DEG))

    lons, lats = [], []
    for geom in roads.geometry:
        if geom is None:
            continue
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            for x, y in part.coords:
                lons.append(x)
                lats.append(y)
    road_lon, road_lat = np.array(lons), np.array(lats)
    print(f"[Roads] {len(road_lon)} road vertices from {len(roads)} segments "
          f"within {buffer_km:.0f}km of the GTA")
    return road_lon, road_lat


def build_ping_edges(
    df: pd.DataFrame,
    grid_km: float = EDGE_GRID_KM,
    road_lon: Optional[np.ndarray] = None,
    road_lat: Optional[np.ndarray] = None,
    road_snap_km: float = ROAD_SNAP_THRESHOLD_KM,
) -> pd.DataFrame:
    """One row per undirected location-to-location edge, connecting
    consecutive-in-time pings from the same vehicle.

    df must already have SPEED_KMH (from add_speed_estimates): that column's
    own NaN-ing of implausible jumps/large gaps (see calculate_speed_kmh) is
    reused here to decide which consecutive-ping pairs become an edge at all,
    so a segment spanning a multi-hour gap or an impossible speed is dropped
    rather than drawn as a straight line across it.

    Every ping is located by its grid_km grid cell -- that's what actually
    drives consolidation (merging into the same edge), same as before. When
    road_lon/road_lat are given, any grid cell touched by at least one ping
    within road_snap_km of a mapped road vertex uses that vertex's own
    coordinates as its representative point instead of the cell's centroid,
    so an edge through that cell is drawn hugging the real road rather than
    an arbitrary grid-square center. (An earlier version keyed directly on
    the nearest road vertex instead of the grid cell -- real road vertices
    turn out to be far more finely and unevenly spaced than useful for
    merging, so two trips a block apart on the same highway often landed on
    different vertices and never consolidated; grid-cell keying doesn't have
    that problem, and still gets the visual benefit via the anchor swap.)

    Consecutive pings that resolve to the same location (e.g. a vehicle
    idling, or GPS jitter within one grid cell) are dropped -- that's
    dwelling, not travel, and already covered by the stop layers. A->B and
    B->A merge into one edge, weighted by PingCount (how many ping-transitions
    map to it) with AvgSpeedKmh averaged over those same transitions.
    """
    df = df.sort_values(["VEHICLE_ID", "LOCATION_DATE_TIME"]).reset_index(drop=True).copy()
    lat_idx, lon_idx, lat_cell_deg, lon_cell_deg = _grid_indices(
        df["LATITUDE"], df["LONGITUDE"], grid_km
    )
    lat_idx, lon_idx = lat_idx.to_numpy(), lon_idx.to_numpy()
    location_key = np.char.add(np.char.add("g", lat_idx.astype(str)), np.char.add("_", lon_idx.astype(str)))
    anchor_lon = lon_idx * lon_cell_deg
    anchor_lat = lat_idx * lat_cell_deg

    if road_lon is not None and len(road_lon):
        lat_mid = df["LATITUDE"].mean()
        km_per_lon_deg = KM_PER_LAT_DEG * np.cos(np.radians(lat_mid))
        tree = shapely.STRtree(shapely.points(road_lon, road_lat))
        nearest = tree.nearest(shapely.points(df["LONGITUDE"].to_numpy(), df["LATITUDE"].to_numpy()))
        dist_km = np.sqrt(
            ((df["LATITUDE"].to_numpy() - road_lat[nearest]) * KM_PER_LAT_DEG) ** 2
            + ((df["LONGITUDE"].to_numpy() - road_lon[nearest]) * km_per_lon_deg) ** 2
        )
        snapped = dist_km <= road_snap_km
        print(f"[Edges] {int(snapped.sum())}/{len(df)} pings within "
              f"{road_snap_km * 1000:.0f}m of a road vertex")

        # First snapped ping per grid cell sets that cell's anchor point;
        # cells with no nearby road keep their centroid (set above).
        anchor_lookup = pd.DataFrame(
            {"lon": road_lon[nearest[snapped]], "lat": road_lat[nearest[snapped]]},
            index=location_key[snapped],
        )
        anchor_lookup = anchor_lookup[~anchor_lookup.index.duplicated(keep="first")]
        key_series = pd.Series(location_key)
        anchor_lon = key_series.map(anchor_lookup["lon"]).fillna(pd.Series(anchor_lon)).to_numpy()
        anchor_lat = key_series.map(anchor_lookup["lat"]).fillna(pd.Series(anchor_lat)).to_numpy()
        n_cells = pd.Series(location_key).nunique()
        n_anchored = len(anchor_lookup)
        print(f"[Edges] {n_anchored}/{n_cells} grid cells anchored to a road vertex")

    df["LOCATION_KEY"] = location_key
    df["LOC_LON"] = anchor_lon
    df["LOC_LAT"] = anchor_lat

    grp = df.groupby("VEHICLE_ID")
    prev_key = grp["LOCATION_KEY"].shift()
    prev_lon = grp["LOC_LON"].shift()
    prev_lat = grp["LOC_LAT"].shift()

    valid = df["SPEED_KMH"].notna() & prev_key.notna()
    same_location = df["LOCATION_KEY"] == prev_key
    edges = df[valid & ~same_location].copy()
    edges["PREV_KEY"] = prev_key[valid & ~same_location]
    edges["PREV_LON"] = prev_lon[valid & ~same_location]
    edges["PREV_LAT"] = prev_lat[valid & ~same_location]

    # Canonical (from, to) ordering -- lexicographically smaller key first --
    # so a trip A->B and another trip B->A land on the same edge.
    key_a, key_b = edges["LOCATION_KEY"].to_numpy(), edges["PREV_KEY"].to_numpy()
    swap = key_a > key_b
    edges["FROM_LON"] = np.where(swap, edges["PREV_LON"], edges["LOC_LON"])
    edges["FROM_LAT"] = np.where(swap, edges["PREV_LAT"], edges["LOC_LAT"])
    edges["TO_LON"] = np.where(swap, edges["LOC_LON"], edges["PREV_LON"])
    edges["TO_LAT"] = np.where(swap, edges["LOC_LAT"], edges["PREV_LAT"])
    edges["FROM_KEY"] = np.where(swap, key_b, key_a)
    edges["TO_KEY"] = np.where(swap, key_a, key_b)

    grouped = (
        edges.groupby(["FROM_KEY", "TO_KEY"])
        .agg(PingCount=("SPEED_KMH", "size"), AvgSpeedKmh=("SPEED_KMH", "mean"),
             FROM_LON=("FROM_LON", "first"), FROM_LAT=("FROM_LAT", "first"),
             TO_LON=("TO_LON", "first"), TO_LAT=("TO_LAT", "first"))
        .reset_index()
    )
    print(f"[Edges] {len(grouped)} unique edges from {len(edges)} valid "
          f"ping-transitions ({grid_km:.1f}km grid fallback)")
    return grouped


def export_ping_edges(
    edges: pd.DataFrame,
    out_path: Path = GEOJSON_DIR / "volvo_ping_edges.geojson",
) -> Path:
    """LineString per edge from build_ping_edges, with PingCount (width, for
    both layers) and AvgSpeedKmh (color, for the average-speed layer) properties.
    """
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [_round(row.FROM_LON), _round(row.FROM_LAT)],
                    [_round(row.TO_LON), _round(row.TO_LAT)],
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


# ---------- Choropleth grid layers (stop density, stop duration) ----------

def export_stop_density_choropleth(
    stops: pd.DataFrame,
    out_path: Path = GEOJSON_DIR / "volvo_stop_density.geojson",
    grid_km: float = CHOROPLETH_GRID_KM,
) -> Path:
    """Stop count per grid cell -- ping/grid-based like stop duration and the
    original hexbin plot (flat-colored cells, no smoothing), rather than a
    smoothed leaflet.heat point layer.
    """
    lat_idx, lon_idx, lat_cell_deg, lon_cell_deg = _grid_indices(
        stops["LATITUDE"], stops["LONGITUDE"], grid_km
    )
    counts = (
        pd.DataFrame({"lat_idx": lat_idx, "lon_idx": lon_idx})
        .value_counts().rename("StopCount").reset_index()
    )
    features = [
        {
            "type": "Feature",
            "geometry": _grid_cell_polygon(row.lat_idx, row.lon_idx, lat_cell_deg, lon_cell_deg),
            "properties": {"StopCount": int(row.StopCount)},
        }
        for row in counts.itertuples()
    ]
    collection = {
        "type": "FeatureCollection",
        "scaleMax": int(counts["StopCount"].max()),
        "features": features,
    }
    return _write_geojson(collection, out_path)


def export_stop_duration_choropleth(
    stops: pd.DataFrame,
    out_path: Path = GEOJSON_DIR / "volvo_stop_duration.geojson",
    grid_km: float = CHOROPLETH_GRID_KM,
    max_duration_min: float = MAX_STOP_DURATION_FOR_PLOT_MIN,
) -> Path:
    """Median stop duration per grid cell -- flat-colored cells, no smoothing,
    matching plot_stop_duration_heatmap's hexbin exactly: median per cell,
    multi-day outliers excluded from the scale.
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


# ---------- Heatmap point layer (regular visitors) ----------

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

    road_lon, road_lat = load_road_vertices(gta_gdf)
    edges = build_ping_edges(df, road_lon=road_lon, road_lat=road_lat)
    export_ping_edges(edges)

    stops = extract_stops(df)
    export_stop_density_choropleth(stops)
    export_stop_duration_choropleth(stops)

    ping_visits = extract_location_visits(df)
    ping_locations = find_regular_locations(ping_visits, vehicle_span_weeks, label="all pings")
    export_regular_locations_heat(ping_locations)


if __name__ == "__main__":
    main()
