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

Both are filtered down to this project's area of interest -- the GTA plus
Hamilton and Cambridge, which sit outside the strict GTA definition (this is
closer to "GTHA + Cambridge") but are already part of the broader city list
elsewhere in this project (see public_charging_analysis.py) -- and geometry-
simplified before export, since the raw province-wide capacity file is
~880MB: 163,201 feeder polygons is far more detail than useful at this
scale, and most of it is outside this project's area of interest anyway.

Run as a script (python source/oeb_geojson_export.py) or import the builders.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import shapely

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEOJSON_DIR = PROJECT_ROOT / "active geojsons"

# Raw OEB exports live outside the repo (see module docstring).
OEB_RAW_DIR = Path.home() / "OEB"
DEFAULT_CAPACITY_GEOJSON = OEB_RAW_DIR / "OEB_Available_Load_Capacity.geojson"
DEFAULT_LDC_GEOJSON = OEB_RAW_DIR / "Electric_LDC_Boundaries.geojson"
GTA_BOUNDARY_PATH = PROJECT_ROOT / "active geojsons" / "GTA_Boundary.geojson"
MUNICIPAL_BOUNDARY_PATH = PROJECT_ROOT / "data" / "Municipal_Boundary_-_Lower_and_Single_Tier.geojson"
# Municipalities outside the strict GTA boundary to also include (matched by
# MUNICIPAL_NAME in MUNICIPAL_BOUNDARY_PATH).
EXTRA_MUNICIPALITIES = ["CITY OF HAMILTON", "CITY OF CAMBRIDGE"]

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


def _load_service_area():
    """GTA boundary plus EXTRA_MUNICIPALITIES, unioned into one polygon, with
    its combined bounding box. Water-extent rows are excluded (e.g. Hamilton
    has a separate "Water" polygon for Hamilton Harbour) the same way
    public_charging_analysis.py already does for this same boundary file.
    """
    gta = gpd.read_file(GTA_BOUNDARY_PATH)[["geometry"]]

    municipal = gpd.read_file(MUNICIPAL_BOUNDARY_PATH)
    if "MUNICIPAL_AREA_EXTENT_TYPE" in municipal.columns:
        municipal = municipal[municipal["MUNICIPAL_AREA_EXTENT_TYPE"] != "Water"]
    extra = municipal[municipal["MUNICIPAL_NAME"].isin(EXTRA_MUNICIPALITIES)][["geometry"]]
    print(f"[Service Area] GTA boundary + {len(extra)} extra municipality "
          f"polygons ({', '.join(EXTRA_MUNICIPALITIES)})")

    combined = pd.concat([gta, extra], ignore_index=True)
    return shapely.union_all(combined.geometry.to_numpy()), combined.total_bounds


def _simplify_and_round(geometry, simplify_deg: float):
    valid = geometry.apply(lambda g: g if g.is_valid else shapely.make_valid(g))
    simplified = valid.simplify(simplify_deg, preserve_topology=True)
    return simplified.apply(lambda g: shapely.set_precision(g, grid_size=10 ** -COORD_DECIMALS))


def export_feeder_capacity(
    capacity_geojson: Path = DEFAULT_CAPACITY_GEOJSON,
    out_path: Path = GEOJSON_DIR / "oeb_feeder_capacity.geojson",
    simplify_deg: float = SIMPLIFY_DEG,
) -> Path:
    """Feeder-level available load capacity (MW), filtered to the service
    area (see _load_service_area) and geometry-simplified. capacity can be
    negative (already over-committed, no available capacity) or null (not
    reported) -- both are kept as-is rather than dropped, so the map/tooltip
    can distinguish "no capacity" from "no data".
    """
    if not capacity_geojson.exists():
        raise FileNotFoundError(
            f"Raw OEB capacity export not found: {capacity_geojson}. See "
            f"download_oeb_files.py in that directory to fetch it."
        )
    service_area, bounds = _load_service_area()

    print(f"[Capacity] Reading {capacity_geojson} within the service-area bounding box...")
    gdf = gpd.read_file(capacity_geojson, bbox=tuple(bounds))
    print(f"[Capacity] {len(gdf)} features in bbox; filtering to the service area")
    gdf = gdf[gdf.intersects(service_area)].copy()
    print(f"[Capacity] {len(gdf)} features intersect the service area")

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
    those intersecting the service area (see _load_service_area) -- a
    context/reference layer, not a capacity signal.
    """
    if not ldc_geojson.exists():
        raise FileNotFoundError(
            f"Raw OEB LDC boundary export not found: {ldc_geojson}. See "
            f"download_oeb_files.py in that directory to fetch it."
        )
    service_area, _ = _load_service_area()

    gdf = gpd.read_file(ldc_geojson)
    gdf = gdf[gdf.intersects(service_area)].copy()
    if gdf.empty:
        print(f"[LDC] No LDC boundaries intersect the service area; skipping {out_path.name}")
        return None
    print(f"[LDC] {len(gdf)} LDC boundaries intersect the service area: "
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
