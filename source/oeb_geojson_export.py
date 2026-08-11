"""
Convert OEB (Ontario Energy Board) Centralized Capacity Information Map data
into GeoJSON layers for the interactive map (viewer/geojson_viewer.html).

Inputs (raw exports, downloaded via /Users/danikae/OEB/download_oeb_files.py,
not committed to this repo -- see that script's docstring for the source):
  - OEB_Available_Load_Capacity.geojson  163,201 feeder-level polygons
    province-wide, each with an available load capacity (MW) value -- the
    grid-capacity signal relevant to siting charging infrastructure.
  - Electric_LDC_Boundaries.geojson      63 electricity distributor (LDC)
    service-area polygons province-wide, for context (which utility to
    contact for a given area).

Both are filtered down to the GTA (reusing this repo's GTA_Boundary.geojson)
and geometry-simplified before export, since the raw province-wide capacity
file is ~880MB -- 163,201 feeder polygons is far more detail than useful at
GTA scale, and most of it is outside this project's area of interest anyway.

Run as a script (python source/oeb_geojson_export.py) or import the builders.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import shapely

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEOJSON_DIR = PROJECT_ROOT / "active geojsons"

# Raw OEB exports live outside the repo (see module docstring).
OEB_RAW_DIR = Path.home() / "OEB"
DEFAULT_CAPACITY_GEOJSON = OEB_RAW_DIR / "OEB_Available_Load_Capacity.geojson"
DEFAULT_LDC_GEOJSON = OEB_RAW_DIR / "Electric_LDC_Boundaries.geojson"
GTA_BOUNDARY_PATH = PROJECT_ROOT / "active geojsons" / "GTA_Boundary.geojson"

COORD_DECIMALS = 6
# Geometry simplification tolerance, in degrees (~5.6m) -- cuts total vertex
# count roughly 3.5x on the feeder-capacity layer with no invalid/empty
# geometries at this tolerance (checked directly against the GTA subset).
SIMPLIFY_DEG = 0.00005

# Properties kept from the raw feeder-capacity export; drops bookkeeping
# fields (objectid, idldc, last_update epoch, Shape__Area/Length) not useful
# on the map.
CAPACITY_PROPERTIES = [
    "ldc_name", "feeder_ltl_voltage_3ph", "feeder_ltn_voltage_1ph",
    "configuration", "capacityrange", "capacity",
]
LDC_PROPERTIES = ["LDC_Name_12", "LDC_Web_12", "LDC_Type"]


def _load_gta_polygon():
    gta = gpd.read_file(GTA_BOUNDARY_PATH)
    return gta, shapely.union_all(gta.geometry.to_numpy())


def _simplify_and_round(geometry, simplify_deg: float):
    valid = geometry.apply(lambda g: g if g.is_valid else shapely.make_valid(g))
    simplified = valid.simplify(simplify_deg, preserve_topology=True)
    return simplified.apply(lambda g: shapely.set_precision(g, grid_size=10 ** -COORD_DECIMALS))


def export_feeder_capacity(
    capacity_geojson: Path = DEFAULT_CAPACITY_GEOJSON,
    out_path: Path = GEOJSON_DIR / "oeb_feeder_capacity.geojson",
    simplify_deg: float = SIMPLIFY_DEG,
) -> Path:
    """Feeder-level available load capacity (MW), filtered to the GTA and
    geometry-simplified. capacity can be negative (already over-committed,
    no available capacity) or null (not reported) -- both are kept as-is
    rather than dropped, so the map/tooltip can distinguish "no capacity"
    from "no data".
    """
    if not capacity_geojson.exists():
        raise FileNotFoundError(
            f"Raw OEB capacity export not found: {capacity_geojson}. See "
            f"download_oeb_files.py in that directory to fetch it."
        )
    gta, gta_polygon = _load_gta_polygon()
    bounds = gta.total_bounds

    print(f"[Capacity] Reading {capacity_geojson} within the GTA bounding box...")
    gdf = gpd.read_file(capacity_geojson, bbox=tuple(bounds))
    print(f"[Capacity] {len(gdf)} features in bbox; filtering to the GTA polygon")
    gdf = gdf[gdf.intersects(gta_polygon)].copy()
    print(f"[Capacity] {len(gdf)} features intersect the GTA")

    gdf["geometry"] = _simplify_and_round(gdf.geometry, simplify_deg)

    features = []
    for row in gdf.itertuples():
        props = {name: getattr(row, name, None) for name in CAPACITY_PROPERTIES}
        if props["capacity"] is not None:
            props["capacity"] = round(float(props["capacity"]), 1)
        features.append({
            "type": "Feature",
            "geometry": json.loads(shapely.to_geojson(row.geometry)),
            "properties": props,
        })

    capacities = [f["properties"]["capacity"] for f in features if f["properties"]["capacity"] is not None]
    collection = {
        "type": "FeatureCollection",
        "scaleMin": round(min(capacities), 1) if capacities else 0,
        "scaleMax": round(max(capacities), 1) if capacities else 1,
        "features": features,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(collection, f, separators=(",", ":"))
    print(f"[Write] {len(features)} features -> {out_path} "
          f"({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return out_path


def export_ldc_boundaries(
    ldc_geojson: Path = DEFAULT_LDC_GEOJSON,
    out_path: Path = GEOJSON_DIR / "oeb_ldc_boundaries.geojson",
    simplify_deg: float = SIMPLIFY_DEG,
) -> Optional[Path]:
    """Electricity distributor (LDC) service-area boundaries, filtered to
    those intersecting the GTA -- a context/reference layer, not a capacity
    signal.
    """
    if not ldc_geojson.exists():
        raise FileNotFoundError(
            f"Raw OEB LDC boundary export not found: {ldc_geojson}. See "
            f"download_oeb_files.py in that directory to fetch it."
        )
    _, gta_polygon = _load_gta_polygon()

    gdf = gpd.read_file(ldc_geojson)
    gdf = gdf[gdf.intersects(gta_polygon)].copy()
    if gdf.empty:
        print(f"[LDC] No LDC boundaries intersect the GTA; skipping {out_path.name}")
        return None
    print(f"[LDC] {len(gdf)} LDC boundaries intersect the GTA: "
          f"{', '.join(sorted(gdf['LDC_Name_12']))}")

    gdf["geometry"] = _simplify_and_round(gdf.geometry, simplify_deg)

    features = [
        {
            "type": "Feature",
            "geometry": json.loads(shapely.to_geojson(row.geometry)),
            "properties": {name: getattr(row, name, None) for name in LDC_PROPERTIES},
        }
        for row in gdf.itertuples()
    ]
    collection = {"type": "FeatureCollection", "features": features}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(collection, f, separators=(",", ":"))
    print(f"[Write] {len(features)} features -> {out_path} "
          f"({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


def main() -> None:
    export_feeder_capacity()
    export_ldc_boundaries()


if __name__ == "__main__":
    main()
