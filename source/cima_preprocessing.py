"""
Preprocess CIMA+/Geotab zone analytics (Toronto, 2024-09-01 to 2024-09-08)
into tiered GeoJSON layers for the interactive map.

Inputs (defaults point at the raw exports in Downloads):
  - Stop analytics CSV: one row per road-segment zone with stop metrics.
    (Its Geography column is empty in this export, so segment geometry is
    joined from the zone CSVs by ZoneId.)
  - Origin-destination CSVs: journey metrics between zone pairs, no geometry.
  - Zone CSVs: ZoneId -> segment LineString geometry and bounding box, used
    for stop-segment shapes and OD endpoints.

Outputs (rank tiers, descending by volume, tier 1 = busiest):
  - active geojsons/cima_stops_<i>.geojson  All ~64k segments split into
    tiers of 10,000 by StopCount.
  - active geojsons/cima_od_<i>.geojson     Top 50,000 zone pairs by JourneyCount in
    tiers of 10,000, drawn as centroid-to-centroid LineStrings; self-pairs
    excluded. (The full matrix has millions of pairs and cannot be embedded.)

Each file carries a root-level "scaleMax" (the dataset-wide metric maximum)
so the map can scale line widths consistently across tiers.

Run as a script (python source/cima_preprocessing.py) or import the builders.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

DOWNLOADS = Path.home() / "Downloads"
DEFAULT_SA_CSV = DOWNLOADS / "OneDrive_2026-06-10" / "2024-01-09 to 2024-08-09" / "sa_result.csv"
DEFAULT_OD_DIR = DOWNLOADS / "OneDrive_2026-06-10" / "2024-01-09 to 2024-08-09" / "od_results"
DEFAULT_ZONES_DIR = DOWNLOADS / "Toronto-zones"
GEOJSON_DIR = Path(__file__).parent.parent / "active geojsons"

COORD_DECIMALS = 6
STOP_TIER_SIZE = 10_000
OD_TOP_N = 50_000
OD_TIER_SIZE = 10_000

# ZoneIds are 19-digit integers; always read them as strings so they survive
# pandas dtype inference (float64 promotion would corrupt join keys).
SA_USECOLS = [
    "ZoneId", "ZoneDescription", "ZoneSubType", "StopCount", "VehicleCount",
    "DailyStopCountAvg", "StopDurationAvg", "StopDurationMed",
    "IdleDurationAvg",
]
OD_USECOLS = [
    "ZonePairId", "OriginZoneId", "OriginZoneDescription",
    "DestinationZoneId", "DestinationZoneDescription",
    "JourneyCount", "DailyJourneyCountAvg",
    "JourneyDistanceAvg", "JourneyDurationAvg", "TravelSpeedAvg",
]
OD_ID_DTYPES = {"ZonePairId": str, "OriginZoneId": str, "DestinationZoneId": str}


def _round_coords(coords):
    if isinstance(coords, (int, float)):
        return round(coords, COORD_DECIMALS)
    return [_round_coords(c) for c in coords]


def _num(value, digits=2):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _write_tiers(features: List[dict], out_dir: Path, stem: str,
                 tier_size: int, scale_max: float) -> List[Path]:
    """Split ranked features into tier files <stem>_1.geojson, _2, ...

    Stale higher-numbered tiers from earlier runs are removed so the layer
    set always matches the current data.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(0, len(features), tier_size):
        tier = i // tier_size + 1
        out_path = out_dir / f"{stem}_{tier}.geojson"
        collection = {
            "type": "FeatureCollection",
            "scaleMax": scale_max,
            "features": features[i:i + tier_size],
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(collection, f, separators=(",", ":"))
        print(f"[Write] {len(collection['features'])} features -> {out_path} "
              f"({out_path.stat().st_size / 1024:.0f} KB)")
        paths.append(out_path)
    for stale in sorted(out_dir.glob(f"{stem}_*.geojson")):
        if stale not in paths:
            stale.unlink()
            print(f"[Write] Removed stale tier {stale}")
    return paths


def build_zone_geometry_lookup(
    zones_dir: Path = DEFAULT_ZONES_DIR,
) -> Dict[str, dict]:
    """ZoneId -> GeoJSON geometry dict parsed from the zone exports."""
    lookup: Dict[str, dict] = {}
    skipped = 0
    for csv_path in sorted(zones_dir.glob("zones_*.csv")):
        df = pd.read_csv(
            csv_path, usecols=["ZoneId", "Geography"],
            dtype={"ZoneId": str}, engine="c",
        )
        for zid, geography in df.itertuples(index=False):
            try:
                geometry = json.loads(geography.strip())
            except (AttributeError, ValueError, TypeError):
                skipped += 1
                continue
            geometry["coordinates"] = _round_coords(geometry["coordinates"])
            lookup[zid] = geometry
    if skipped:
        print(f"[Zones] Skipped {skipped} zones with unparseable Geography")
    print(f"[Zones] Loaded {len(lookup)} zone geometries from {zones_dir}")
    return lookup


def build_stops_geojsons(
    sa_csv: Path = DEFAULT_SA_CSV,
    zones_dir: Path = DEFAULT_ZONES_DIR,
    out_dir: Path = GEOJSON_DIR,
    tier_size: int = STOP_TIER_SIZE,
) -> List[Path]:
    """All stop-analytics segments, ranked by StopCount, in tier files.

    The stop-analytics export ships with an empty Geography column, so each
    segment's LineString comes from the zone CSVs, joined on ZoneId.
    """
    geometries = build_zone_geometry_lookup(zones_dir)
    print(f"[Stops] Reading {sa_csv}")
    df = pd.read_csv(sa_csv, usecols=SA_USECOLS, dtype={"ZoneId": str}, engine="c")
    df = df.sort_values("StopCount", ascending=False)

    features = []
    skipped = 0
    for row in df.itertuples(index=False):
        geometry = geometries.get(row.ZoneId)
        if geometry is None:
            skipped += 1
            continue
        desc = row.ZoneDescription
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "Road": desc if isinstance(desc, str) and desc else "Unnamed segment",
                "Road Type": row.ZoneSubType if isinstance(row.ZoneSubType, str) else "",
                "StopCount": int(row.StopCount),
                "VehicleCount": int(row.VehicleCount),
                "DailyStopCountAvg": _num(row.DailyStopCountAvg, 1),
                "StopDurationAvg (min)": _num(row.StopDurationAvg),
                "StopDurationMed (min)": _num(row.StopDurationMed),
                "IdleDurationAvg (min)": _num(row.IdleDurationAvg),
            },
        })
    if skipped:
        print(f"[Stops] Skipped {skipped} rows with no zone geometry")
    print(f"[Stops] {len(features)} segments, StopCount "
          f"{df.StopCount.min():.0f} - {df.StopCount.max():.0f}")
    return _write_tiers(features, out_dir, "cima_stops", tier_size,
                        scale_max=float(df.StopCount.max()))


def build_zone_centroid_lookup(
    zones_dir: Path = DEFAULT_ZONES_DIR,
) -> Dict[str, Tuple[float, float]]:
    """ZoneId -> (lon, lat) bounding-box centre from the zone exports."""
    lookup: Dict[str, Tuple[float, float]] = {}
    for csv_path in sorted(zones_dir.glob("zones_*.csv")):
        df = pd.read_csv(
            csv_path,
            usecols=["ZoneId", "MinLongitude", "MinLatitude",
                     "MaxLongitude", "MaxLatitude"],
            dtype={"ZoneId": str}, engine="c",
        )
        for zid, lon0, lat0, lon1, lat1 in df.itertuples(index=False):
            lookup[zid] = (
                round((lon0 + lon1) / 2, COORD_DECIMALS),
                round((lat0 + lat1) / 2, COORD_DECIMALS),
            )
    print(f"[Zones] Loaded {len(lookup)} zone centroids from {zones_dir}")
    return lookup


def build_od_geojsons(
    od_dir: Path = DEFAULT_OD_DIR,
    zones_dir: Path = DEFAULT_ZONES_DIR,
    out_dir: Path = GEOJSON_DIR,
    top_n: int = OD_TOP_N,
    tier_size: int = OD_TIER_SIZE,
    overfetch: float = 1.2,
) -> List[Path]:
    """Top-N origin-destination pairs by JourneyCount as centroid lines.

    Streams the od_results files keeping a running top keep_n so memory stays
    bounded; keep_n over-fetches so pairs lost to missing centroids do not
    leave the layers short.
    """
    centroids = build_zone_centroid_lookup(zones_dir)
    keep_n = int(top_n * overfetch)

    od_files = sorted(od_dir.glob("origin_destination_*.csv"))
    if not od_files:
        raise FileNotFoundError(f"No origin_destination_*.csv files in {od_dir}")
    print(f"[OD] Scanning {len(od_files)} files in {od_dir}")

    running = None
    self_pairs = 0
    for i, csv_path in enumerate(od_files, 1):
        df = pd.read_csv(csv_path, usecols=OD_USECOLS, dtype=OD_ID_DTYPES, engine="c")
        before = len(df)
        df = df[df.OriginZoneId != df.DestinationZoneId]
        self_pairs += before - len(df)
        df = df.nlargest(keep_n, "JourneyCount")
        running = df if running is None else (
            pd.concat([running, df]).nlargest(keep_n, "JourneyCount"))
        if i % 50 == 0 or i == len(od_files):
            print(f"[OD] [{i}/{len(od_files)}] running min JourneyCount = "
                  f"{running.JourneyCount.min():.0f}")

    running = running.drop_duplicates("ZonePairId", keep="first")
    print(f"[OD] Excluded {self_pairs} self-pairs; "
          f"{len(running)} candidate pairs before centroid join")

    features = []
    missing = 0
    for row in running.itertuples(index=False):
        origin = centroids.get(row.OriginZoneId)
        dest = centroids.get(row.DestinationZoneId)
        if origin is None or dest is None:
            missing += 1
            continue
        odesc = row.OriginZoneDescription
        ddesc = row.DestinationZoneDescription
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [list(origin), list(dest)],
            },
            "properties": {
                "Origin": odesc if isinstance(odesc, str) and odesc else row.OriginZoneId,
                "Destination": ddesc if isinstance(ddesc, str) and ddesc else row.DestinationZoneId,
                "JourneyCount": int(row.JourneyCount),
                "DailyJourneyCountAvg": _num(row.DailyJourneyCountAvg, 1),
                "JourneyDistanceAvg (km)": _num(row.JourneyDistanceAvg),
                "JourneyDurationAvg (min)": _num(row.JourneyDurationAvg),
                "TravelSpeedAvg (km/h)": _num(row.TravelSpeedAvg),
            },
        })
        if len(features) >= top_n:
            break
    if missing:
        print(f"[OD] Dropped {missing} pairs with no zone centroid")
    print(f"[OD] {len(features)} pairs kept, JourneyCount "
          f"{features[-1]['properties']['JourneyCount']} - "
          f"{features[0]['properties']['JourneyCount']}")
    return _write_tiers(features, out_dir, "cima_od", tier_size,
                        scale_max=float(features[0]["properties"]["JourneyCount"]))


def main():
    build_stops_geojsons()
    build_od_geojsons()


if __name__ == "__main__":
    main()
