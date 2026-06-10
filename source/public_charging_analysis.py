"""
Public charging analysis helpers for GTA overlays.

Functions here:
1) Extract numbered highway segments within the GTA boundary and save to GeoJSON.
2) Convert public charging cities (CSV) to point and boundary GeoJSONs.
3) Filter FSA boundaries for the public charging FSAs list and save to GeoJSON.
4) Convert refueling stops (CSV) to a point GeoJSON.

Outputs are written alongside the input data (by default into the data/ folder).
"""

from pathlib import Path
from typing import Iterable, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from source.Highways import normalize_highway_name
from postal_code_analysis import load_fsa_shapefile


# ---------- Paths ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# ---------- Utility helpers ----------

def _ensure_crs(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    """Reproject to the target CRS if needed."""
    if gdf.crs != target_crs:
        return gdf.to_crs(target_crs)
    return gdf


def _save_geojson(gdf: gpd.GeoDataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    return output_path


# ---------- 1) GTA numbered highways ----------

def extract_gta_numbered_highways(
    highways_geojson: Path = DATA_DIR / "Value_of_Goods_2016_-3795296180828259022.geojson",
    gta_boundary_geojson: Path = DATA_DIR / "GTA_Boundary.geojson",
    output_path: Path = DATA_DIR / "gta_numbered_highways.geojson",
) -> gpd.GeoDataFrame:
    """
    Load highway network and GTA boundary, keep numbered highway segments that
    intersect the GTA, and save them to GeoJSON.
    """

    if not highways_geojson.exists():
        raise FileNotFoundError(f"Highway GeoJSON not found: {highways_geojson}")
    if not gta_boundary_geojson.exists():
        raise FileNotFoundError(f"GTA boundary GeoJSON not found: {gta_boundary_geojson}")

    gdf_roads = gpd.read_file(highways_geojson)
    gdf_gta = gpd.read_file(gta_boundary_geojson)

    # Align CRS with explicit logging
    print(f"\n[GTA Highway Extraction] CRS Alignment:")
    print(f"  Roads CRS: {gdf_roads.crs}")
    print(f"  GTA Boundary CRS: {gdf_gta.crs}")
    
    if gdf_roads.crs != gdf_gta.crs:
        print(f"  → Reprojecting GTA boundary to {gdf_roads.crs}")
        gdf_gta = gdf_gta.to_crs(gdf_roads.crs)
    else:
        print(f"  ✓ CRS already aligned")

    # Clip to GTA extent
    gdf_gta_union = gdf_gta.unary_union
    gdf_gta_roads = gdf_roads[gdf_roads.intersects(gdf_gta_union)].copy()

    # Keep only numbered highways (digits only)
    def _get_numbered_highway(row) -> str | None:
        for key in ["RDNAME", "HWY_NUM", "HWYNUM", "HIGHWAY", "NAME"]:
            if key in row and pd.notna(row[key]):
                candidate = normalize_highway_name(str(row[key]))
                if candidate and candidate.isdigit():
                    return candidate
        return None

    gdf_gta_roads["highway_id"] = gdf_gta_roads.apply(_get_numbered_highway, axis=1)
    gdf_gta_roads = gdf_gta_roads[gdf_gta_roads["highway_id"].notna()].copy()

    _save_geojson(gdf_gta_roads, output_path)
    return gdf_gta_roads


# ---------- 2) Public charging cities ----------

def cities_to_geojson(
    cities_csv: Path = DATA_DIR / "public_charging_cities.csv",
    municipal_boundary_geojson: Path = DATA_DIR / "Municipal_Boundary_-_Lower_and_Single_Tier.geojson",
    points_output_path: Path = DATA_DIR / "public_charging_cities_points.geojson",
    boundaries_output_path: Path = DATA_DIR / "public_charging_cities_boundaries.geojson",
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Convert public charging cities (CSV) to point and boundary GeoJSONs."""

    if not cities_csv.exists():
        raise FileNotFoundError(f"Cities CSV not found: {cities_csv}")
    if not municipal_boundary_geojson.exists():
        raise FileNotFoundError(f"Municipal boundary GeoJSON not found: {municipal_boundary_geojson}")

    # Mapping from city names in CSV to MUNICIPAL_NAME in boundary file
    city_to_municipal_name = {
        "Cambridge": "CITY OF CAMBRIDGE",
        "Windsor": "CITY OF WINDSOR",
        "Hamilton": "CITY OF HAMILTON",
        "Mississauga": "CITY OF MISSISSAUGA",
        "Oshawa": "CITY OF OSHAWA",
        "Barrie": "CITY OF BARRIE",
    }

    df = pd.read_csv(cities_csv)
    required_cols = {"City", "Latitude", "Longitude"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Cities CSV missing columns: {sorted(missing)}")

    # --- Points output (from CSV) ---
    df_clean = df.dropna(subset=["Latitude", "Longitude"]).copy()
    df_clean["Longitude"] = pd.to_numeric(df_clean["Longitude"], errors="coerce")
    df_clean["Latitude"] = pd.to_numeric(df_clean["Latitude"], errors="coerce")
    df_clean = df_clean.dropna(subset=["Latitude", "Longitude"])

    geometry = [Point(lon, lat) for lon, lat in zip(df_clean["Longitude"], df_clean["Latitude"])]
    gdf_points = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4326")
    print(f"\n[Cities to GeoJSON] Created {len(gdf_points)} city points with CRS: {gdf_points.crs}")

    _save_geojson(gdf_points, points_output_path)

    # --- Boundary output (from municipal boundary file) ---
    cities_in_csv = df["City"].dropna().unique()
    target_municipal_names = [
        city_to_municipal_name[city] for city in cities_in_csv
        if city in city_to_municipal_name
    ]

    if not target_municipal_names:
        raise ValueError(f"No matching municipalities found. Cities in CSV: {cities_in_csv}")

    gdf_boundaries = gpd.read_file(municipal_boundary_geojson)
    print(f"\n[Cities to GeoJSON] Loaded {len(gdf_boundaries)} municipalities from boundary file")
    print(f"  Boundary CRS: {gdf_boundaries.crs}")

    gdf_filtered = gdf_boundaries[gdf_boundaries["MUNICIPAL_NAME"].isin(target_municipal_names)].copy()
    if "MUNICIPAL_AREA_EXTENT_TYPE" in gdf_filtered.columns:
        gdf_filtered = gdf_filtered[gdf_filtered["MUNICIPAL_AREA_EXTENT_TYPE"] != "Water"].copy()
    else:
        print("  ⚠ MUNICIPAL_AREA_EXTENT_TYPE column not found; skipping Water exclusion")

    print(f"  Filtered to {len(gdf_filtered)} matching city boundaries")

    _save_geojson(gdf_filtered, boundaries_output_path)
    return gdf_points, gdf_filtered


# ---------- 3) Public charging FSAs ----------

def fsas_to_geojson(
    fsas_csv: Path = DATA_DIR / "public_charging_fsas.csv",
    fsa_shapefile: Path = DATA_DIR / "lfsa000b21a_e" / "lfsa000b21a_e.shp",
    output_path: Path = DATA_DIR / "public_charging_fsas.geojson",
) -> gpd.GeoDataFrame:
    """
    Filter FSA boundaries to the list in the CSV and save to GeoJSON.
    """

    if not fsas_csv.exists():
        raise FileNotFoundError(f"FSA CSV not found: {fsas_csv}")
    if not fsa_shapefile.exists():
        raise FileNotFoundError(f"FSA shapefile not found: {fsa_shapefile}")

    df = pd.read_csv(fsas_csv)
    if "FSA" not in df.columns:
        raise ValueError("FSA CSV must contain an 'FSA' column")
    target_fsas = df["FSA"].dropna().astype(str).str.upper().str[:3].unique()

    gdf_fsa = load_fsa_shapefile(fsa_shapefile)
    print(f"\n[FSAs to GeoJSON] FSA Shapefile CRS: {gdf_fsa.crs}")
    gdf_filtered = gdf_fsa[gdf_fsa["CFSAUID"].str[:3].str.upper().isin(target_fsas)].copy()
    print(f"  Filtered to {len(gdf_filtered)} FSAs with CRS: {gdf_filtered.crs}")

    _save_geojson(gdf_filtered, output_path)
    return gdf_filtered


# ---------- 4) Refueling stops ----------

def refueling_stops_to_geojson(
    refueling_csv: Path = DATA_DIR / "refueling_stops.csv",
    output_path: Path = DATA_DIR / "refueling_stops.geojson",
) -> gpd.GeoDataFrame:
    """Convert refueling stop locations (CSV) to a GeoJSON of points."""

    if not refueling_csv.exists():
        raise FileNotFoundError(f"Refueling stops CSV not found: {refueling_csv}")

    df = pd.read_csv(refueling_csv)
    required_cols = {"City", "Latitude", "Longitude"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Refueling stops CSV missing columns: {sorted(missing)}")

    df_clean = df.dropna(subset=["Latitude", "Longitude"]).copy()
    df_clean["Longitude"] = pd.to_numeric(df_clean["Longitude"], errors="coerce")
    df_clean["Latitude"] = pd.to_numeric(df_clean["Latitude"], errors="coerce")
    df_clean["Count"] = pd.to_numeric(df_clean.get("Count"), errors="coerce")
    df_clean = df_clean.dropna(subset=["Latitude", "Longitude"])

    geometry = [Point(lon, lat) for lon, lat in zip(df_clean["Longitude"], df_clean["Latitude"])]
    gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4326")
    print(f"\n[Refueling Stops to GeoJSON] Created {len(gdf)} stop points with CRS: {gdf.crs}")

    _save_geojson(gdf, output_path)
    return gdf


# ---------- Orchestration ----------

def run_all(
    highways_geojson: Path = DATA_DIR / "Value_of_Goods_2016_-3795296180828259022.geojson",
    gta_boundary_geojson: Path = DATA_DIR / "GTA_Boundary.geojson",
    cities_csv: Path = DATA_DIR / "public_charging_cities.csv",
    fsas_csv: Path = DATA_DIR / "public_charging_fsas.csv",
    fsa_shapefile: Path = DATA_DIR / "lfsa000b21a_e" / "lfsa000b21a_e.shp",
    refueling_csv: Path = DATA_DIR / "refueling_stops.csv",
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Run all extraction tasks and return GeoDataFrames."""

    gdf_highways = extract_gta_numbered_highways(highways_geojson, gta_boundary_geojson)
    gdf_city_points, gdf_city_boundaries = cities_to_geojson(cities_csv)
    gdf_fsas = fsas_to_geojson(fsas_csv, fsa_shapefile)
    # Also export refueling stops for completeness; return signature remains unchanged
    refueling_stops_to_geojson(refueling_csv)
    return gdf_highways, gdf_city_points, gdf_city_boundaries, gdf_fsas


if __name__ == "__main__":
    gdf_highways, gdf_city_points, gdf_city_boundaries, gdf_fsas = run_all()
    print(
        f"Saved outputs to: {DATA_DIR / 'gta_numbered_highways.geojson'}, "
        f"{DATA_DIR / 'public_charging_cities_points.geojson'}, "
        f"{DATA_DIR / 'public_charging_cities_boundaries.geojson'}, "
        f"{DATA_DIR / 'public_charging_fsas.geojson'}, "
        f"{DATA_DIR / 'refueling_stops.geojson'}"
    )