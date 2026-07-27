"""
Convert Volvo telematics analysis outputs (source/volvo_telematics_analysis.py)
into GeoJSON layers for the interactive map (viewer/geojson_viewer.html).

Two rendering styles, matching what's already used for the CIMA+ data (see
source/interactive_map.py's LAYER_CONFIG for how each file below is wired in):

  - "Choropleth" grid layers (data coverage, average speed): each feature is a
    flat-colored square grid cell with no interpolation between cells, so the
    map reads the same way the hexbin PNGs do -- a density/value color map,
    not a smoothed surface.
  - "Heatmap" point layers (stop density, stop duration, regular-visitor
    locations): one Point per stop/location with a numeric weight property.
    The smoothing happens client-side via the same leaflet.heat layer
    (radius/blur, default gradient) already used for cima_stops_heat.geojson,
    so these render with the same "2D interpolated" look, not a pre-baked
    raster image.

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

# Grid cell size for the choropleth (non-smoothed) layers -- finer than the
# STOP_LOCATION_GRID_KM clustering grid, since these are meant to reproduce
# the original hexbin plots' resolution, not to group distinct sites together.
CHOROPLETH_GRID_KM = 0.5
MIN_SPEED_SAMPLES_PER_CELL = 3


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


# ---------- Choropleth grid layers (data coverage, average speed) ----------

def export_coverage_choropleth(
    df: pd.DataFrame,
    out_path: Path = GEOJSON_DIR / "volvo_coverage.geojson",
    grid_km: float = CHOROPLETH_GRID_KM,
) -> Path:
    """Ping count per grid cell -- same population as plot_coverage_map."""
    lat_idx, lon_idx, lat_cell_deg, lon_cell_deg = _grid_indices(
        df["LATITUDE"], df["LONGITUDE"], grid_km
    )
    counts = (
        pd.DataFrame({"lat_idx": lat_idx, "lon_idx": lon_idx})
        .value_counts().rename("PingCount").reset_index()
    )
    features = [
        {
            "type": "Feature",
            "geometry": _grid_cell_polygon(row.lat_idx, row.lon_idx, lat_cell_deg, lon_cell_deg),
            "properties": {"PingCount": int(row.PingCount)},
        }
        for row in counts.itertuples()
    ]
    collection = {
        "type": "FeatureCollection",
        "scaleMax": int(counts["PingCount"].max()),
        "features": features,
    }
    return _write_geojson(collection, out_path)


def export_speed_choropleth(
    df: pd.DataFrame,
    out_path: Path = GEOJSON_DIR / "volvo_speed.geojson",
    grid_km: float = CHOROPLETH_GRID_KM,
    min_samples: int = MIN_SPEED_SAMPLES_PER_CELL,
) -> Path:
    """Mean estimated speed per grid cell -- same population as plot_speed_heatmap."""
    valid = df.dropna(subset=["SPEED_KMH"])
    lat_idx, lon_idx, lat_cell_deg, lon_cell_deg = _grid_indices(
        valid["LATITUDE"], valid["LONGITUDE"], grid_km
    )
    grouped = (
        valid.assign(LAT_IDX=lat_idx, LON_IDX=lon_idx)
        .groupby(["LAT_IDX", "LON_IDX"])["SPEED_KMH"]
        .agg(AvgSpeedKmh="mean", SampleCount="size")
        .reset_index()
    )
    grouped = grouped[grouped["SampleCount"] >= min_samples]
    features = [
        {
            "type": "Feature",
            "geometry": _grid_cell_polygon(row.LAT_IDX, row.LON_IDX, lat_cell_deg, lon_cell_deg),
            "properties": {
                "AvgSpeedKmh": round(float(row.AvgSpeedKmh), 1),
                "SampleCount": int(row.SampleCount),
            },
        }
        for row in grouped.itertuples()
    ]
    collection = {
        "type": "FeatureCollection",
        "scaleMax": round(float(grouped["AvgSpeedKmh"].max()), 1),
        "features": features,
    }
    return _write_geojson(collection, out_path)


# ---------- Heatmap point layers (stop density, duration, regular visitors) ----------

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


def export_stop_duration_heat(
    stops: pd.DataFrame,
    out_path: Path = GEOJSON_DIR / "volvo_stop_duration_heat.geojson",
    max_duration_min: float = MAX_STOP_DURATION_FOR_PLOT_MIN,
) -> Path:
    """One point per stop, weighted by its own duration -- multi-day outliers
    excluded, the same treatment plot_stop_duration_heatmap gives its color scale.
    """
    plotted = stops[stops["STOP_DURATION_MIN"] <= max_duration_min]
    excluded = len(stops) - len(plotted)
    if excluded:
        print(f"[Stop Duration] Excluding {excluded} stops longer than "
              f"{max_duration_min:.0f} min")
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [_round(row.LONGITUDE), _round(row.LATITUDE)]},
            "properties": {"StopDurationMin": round(float(row.STOP_DURATION_MIN), 1)},
        }
        for row in plotted.itertuples()
    ]
    collection = {
        "type": "FeatureCollection",
        "scaleMax": round(float(plotted["STOP_DURATION_MIN"].max()), 1),
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

    export_coverage_choropleth(df)

    df = add_speed_estimates(df)
    export_speed_choropleth(df)

    stops = extract_stops(df)
    export_stop_density_heat(stops)
    export_stop_duration_heat(stops)

    ping_visits = extract_location_visits(df)
    ping_locations = find_regular_locations(ping_visits, vehicle_span_weeks, label="all pings")
    export_regular_locations_heat(ping_locations)


if __name__ == "__main__":
    main()
