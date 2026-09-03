"""
Convert Hydro One's "List of Station Capacity" (data/honi_lsc.pdf) into a
point GeoJSON of generator connection capacity for the interactive map.

The PDF (source: https://www.hydroone.com/business-services/generators/
station-capacity-calculator) lists, per Hydro One station/bus, the estimated
Thermal Capacity (MW) available for new generation connections. It has no
coordinates -- station names are informal ("ABBEY DS", "AGINCOURT TS DESN1")
rather than addresses, and none of the public GIS sources checked (Ontario's
"Utility Site" open-data layer, OpenStreetMap substations, Hydro One's own
Generation Capacity Map service) give a reliable name-matched point dataset
(see conversation notes / commit history for what was tried and why each
didn't work).

This instead geocodes each station's cleaned base place name (stripping
DS/TS/PDS/#N/DESN#/kV suffixes) via OpenStreetMap's Nominatim, since most
station names are named after the town/neighbourhood they're in. That's
place-centroid accuracy, not exact substation coordinates, and some names
won't resolve confidently (rejected rather than guessed -- see
_is_locality_result). Results are cached locally (not committed) since
Nominatim's usage policy caps at ~1 request/second and this is ~1,260 names.

Run as a script (python source/honi_geojson_export.py) or import the builders.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "honi_lsc.pdf"
GEOJSON_DIR = PROJECT_ROOT / "active geojsons"
GEOCODE_CACHE_PATH = PROJECT_ROOT / "data" / "honi_geocode_cache.json"
OUT_PATH = GEOJSON_DIR / "honi_generation_capacity.geojson"

# Same GTA+Hamilton+Cambridge service area as oeb_geojson_export.py, padded
# generously since this is town/neighbourhood-centroid geocoding, not exact
# coordinates -- a station just outside the strict boundary shouldn't be
# dropped for that reason alone.
SERVICE_AREA_BOUNDS_PAD_DEG = 0.5
GTA_BOUNDARY_PATH = PROJECT_ROOT / "active geojsons" / "GTA_Boundary.geojson"
MUNICIPAL_BOUNDARY_PATH = PROJECT_ROOT / "data" / "Municipal_Boundary_-_Lower_and_Single_Tier.geojson"
EXTRA_MUNICIPALITIES = ["CITY OF HAMILTON", "CITY OF CAMBRIDGE"]

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "charging-corridor-analysis (one-off station geocoding)"
NOMINATIM_DELAY_SEC = 1.1

COORD_DECIMALS = 6
# Marker diameter (as radius, px) scales linearly with capacity -- a literal
# proportionality, not a min/max-normalized fraction like the line-width
# layers elsewhere use -- with a small floor so near-zero values are still a
# visible dot rather than disappearing.
MIN_RADIUS_PX = 3.0
MAX_RADIUS_PX = 26.0


# ---------- PDF parsing ----------

def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def parse_station_rows(pdf_path: Path = PDF_PATH) -> list:
    """Raw table rows from every page, minus the repeated header row."""
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                rows.extend(table)
    header = rows[0]
    return [r for r in rows if r != header and r and r[0]]


def _station_capacity(rows_for_station: list) -> Optional[float]:
    """Thermal Capacity (MW) for one station, from its Total bus row if
    present (summing the other rows when that row just says "Sum of Buses"
    instead of a number), else summing whatever rows exist.
    """
    total_rows = [r for r in rows_for_station if r[1] == "Total"]
    if total_rows:
        value = total_rows[0][6]
        if value == "Sum of Buses":
            parts = [float(r[6]) for r in rows_for_station if r[1] != "Total" and _is_number(r[6])]
            return sum(parts) if parts else None
        return float(value) if _is_number(value) else None
    parts = [float(r[6]) for r in rows_for_station if _is_number(r[6])]
    return sum(parts) if parts else None


def _base_place_name(station_name: str) -> str:
    """Strip suffixes that vary within one physical site (voltage, #N,
    DESNn, station type) to get the place name to geocode, e.g.
    "NOBLETON DS 27.6 kV" -> "NOBLETON", "BRAMALEA TS DESN1" -> "BRAMALEA".
    """
    name = station_name
    name = re.sub(r"\s*-?\s*\d+(\.\d+)?\s*k[Vv]\s*$", "", name)
    name = re.sub(r"\s*#\d+\s*$", "", name)
    name = re.sub(r"\s+DESN\d+\s*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+(TS|DS|PDS)\s*$", "", name, flags=re.IGNORECASE)
    return name.strip()


def parse_station_capacities(pdf_path: Path = PDF_PATH) -> list:
    """One record per station: {station_name, base_place_name, capacity_mw,
    voltage_kv, upstream_ts}. capacity_mw is None if not quantifiable (e.g.
    marked "TC" -- transmission-constrained, no number given -- or "N/A").
    """
    rows = parse_station_rows(pdf_path)
    by_station: dict = {}
    for r in rows:
        by_station.setdefault(r[0], []).append(r)

    records = []
    for station_name, station_rows in by_station.items():
        rep = next((r for r in station_rows if r[1] == "Total"), station_rows[0])
        records.append({
            "station_name": station_name,
            "base_place_name": _base_place_name(station_name),
            "capacity_mw": _station_capacity(station_rows),
            "voltage_kv": float(rep[3]) if _is_number(rep[3]) else None,
            "upstream_ts": rep[7] or None,
        })
    print(f"[Parse] {len(records)} stations from {pdf_path.name} "
          f"({sum(1 for r in records if r['capacity_mw'] is not None)} with a numeric capacity)")
    return records


# ---------- Geocoding ----------

# Nominatim result classes/types accepted as "this is a real place", not a
# POI that happens to share the name (e.g. "ABBEY" matching a school called
# "Loretto Abbey" rather than any place actually named Abbey).
_LOCALITY_CLASSES = {"place", "boundary"}


def _is_locality_result(result: dict) -> bool:
    return result.get("class") in _LOCALITY_CLASSES


def _load_geocode_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_geocode_cache(cache: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1)


def _geocode_one(place_name: str) -> Optional[dict]:
    query = urllib.parse.quote(f"{place_name}, Ontario, Canada")
    url = f"{NOMINATIM_URL}?q={query}&format=json&limit=3"
    req = urllib.request.Request(url, headers={"User-Agent": NOMINATIM_USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        results = json.loads(resp.read())
    for result in results:
        if _is_locality_result(result):
            return {
                "lat": float(result["lat"]),
                "lon": float(result["lon"]),
                "class": result["class"],
                "type": result.get("type"),
                "display_name": result.get("display_name"),
            }
    return None


def geocode_place_names(
    place_names: list,
    cache_path: Path = GEOCODE_CACHE_PATH,
) -> dict:
    """place_name -> geocode result dict (or None if unresolved), cached to
    cache_path so re-runs don't re-query Nominatim for names already tried.
    """
    cache = _load_geocode_cache(cache_path)
    to_fetch = [n for n in place_names if n not in cache]
    print(f"[Geocode] {len(place_names)} unique place names, "
          f"{len(place_names) - len(to_fetch)} already cached, {len(to_fetch)} to fetch")

    for i, name in enumerate(to_fetch, 1):
        try:
            cache[name] = _geocode_one(name)
        except Exception as e:
            print(f"[Geocode] Error on {name!r}: {e}")
            cache[name] = None
        if i % 25 == 0 or i == len(to_fetch):
            print(f"[Geocode] [{i}/{len(to_fetch)}]")
            _save_geocode_cache(cache, cache_path)
        time.sleep(NOMINATIM_DELAY_SEC)

    _save_geocode_cache(cache, cache_path)
    n_resolved = sum(1 for v in cache.values() if v is not None)
    print(f"[Geocode] {n_resolved}/{len(cache)} place names resolved to a locality")
    return cache


# ---------- Service-area bounds ----------

def _service_area_bounds(pad_deg: float = SERVICE_AREA_BOUNDS_PAD_DEG):
    import geopandas as gpd
    import pandas as pd

    gta = gpd.read_file(GTA_BOUNDARY_PATH)[["geometry"]]
    municipal = gpd.read_file(MUNICIPAL_BOUNDARY_PATH)
    if "MUNICIPAL_AREA_EXTENT_TYPE" in municipal.columns:
        municipal = municipal[municipal["MUNICIPAL_AREA_EXTENT_TYPE"] != "Water"]
    extra = municipal[municipal["MUNICIPAL_NAME"].isin(EXTRA_MUNICIPALITIES)][["geometry"]]
    bounds = pd.concat([gta, extra], ignore_index=True).total_bounds
    return (bounds[0] - pad_deg, bounds[1] - pad_deg, bounds[2] + pad_deg, bounds[3] + pad_deg)


# ---------- Export ----------

def export_generation_capacity(
    pdf_path: Path = PDF_PATH,
    out_path: Path = OUT_PATH,
    cache_path: Path = GEOCODE_CACHE_PATH,
) -> Path:
    records = parse_station_capacities(pdf_path)
    unique_place_names = sorted({r["base_place_name"] for r in records})
    geocoded = geocode_place_names(unique_place_names, cache_path)

    lon_min, lat_min, lon_max, lat_max = _service_area_bounds()

    features = []
    n_unresolved = 0
    n_out_of_area = 0
    for r in records:
        result = geocoded.get(r["base_place_name"])
        if result is None:
            n_unresolved += 1
            continue
        lat, lon = result["lat"], result["lon"]
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            n_out_of_area += 1
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lon, COORD_DECIMALS), round(lat, COORD_DECIMALS)]},
            "properties": {
                "StationName": r["station_name"],
                "CapacityMW": round(r["capacity_mw"], 1) if r["capacity_mw"] is not None else None,
                "VoltageKV": r["voltage_kv"],
                "UpstreamTS": r["upstream_ts"],
                "GeocodedPlace": result["display_name"],
            },
        })
    print(f"[Export] {len(features)} stations in the service area "
          f"({n_unresolved} unresolved, {n_out_of_area} resolved outside the area)")

    capacities = [f["properties"]["CapacityMW"] for f in features if f["properties"]["CapacityMW"] is not None]
    collection = {
        "type": "FeatureCollection",
        "scaleMax": round(max(capacities), 1) if capacities else 1,
        "features": features,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(collection, f, separators=(",", ":"))
    print(f"[Write] {len(features)} features -> {out_path} "
          f"({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


def main() -> None:
    export_generation_capacity()


if __name__ == "__main__":
    main()
