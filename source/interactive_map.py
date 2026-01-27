"""
Interactive GeoJSON viewer using Folium/Leaflet.
Loads all known GeoJSON layers from the geojsons directory and renders
an interactive map with layer toggles and a styled legend (QGIS-like stack).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import folium
import geopandas as gpd

# Layer configuration: ordered by zorder for proper stacking.
LAYER_CONFIG = [
    {
        "name": "Ontario Boundary",
        "filename": "Ontario_Provincial_Boundary.geojson",
        "style_key": "ontario_boundary",
        "zorder": 0,
    },
    {
        "name": "GTA Boundary",
        "filename": "GTA_Boundary.geojson",
        "style_key": "gta_boundary",
        "zorder": 1,
    },
    {
        "name": "Public Charging FSAs",
        "filename": "public_charging_fsas.geojson",
        "style_key": "public_charging_fsas",
        "zorder": 2,
    },
    {
        "name": "Target FSAs",
        "filename": "target_fsas.geojson",
        "style_key": "target_fsas",
        "zorder": 3,
    },
    {
        "name": "Pembina FSAs",
        "filename": "pembina_fsas.geojson",
        "style_key": "pembina_fsas",
        "zorder": 4,
    },
    {
        "name": "Primary Freight Corridors",
        "filename": "Value_of_Goods_2016_-3795296180828259022.geojson",
        "style_key": "freight_corridors",
        "zorder": 5,
    },
    {
        "name": "Target Highways",
        "filename": "target_highways.geojson",
        "style_key": "target_highways",
        "zorder": 6,
    },
    {
        "name": "Numbered Highways",
        "filename": "gta_numbered_highways.geojson",
        "style_key": "numbered_highways",
        "zorder": 7,
    },
    {
        "name": "Refueling Stops",
        "filename": "refueling_stops.geojson",
        "style_key": "refueling_stops",
        "zorder": 8,
    },
    {
        "name": "Public Charging Cities",
        "filename": "public_charging_cities.geojson",
        "style_key": "public_charging_cities",
        "zorder": 9,
    },
]

# Style palette keyed by style_key in LAYER_CONFIG.
STYLE_MAP: Dict[str, Dict[str, Any]] = {
    "ontario_boundary": {
        "color": "#9a9a9a",
        "fillColor": "#d9d9d7",
        "fillOpacity": 0.35,
        "weight": 1.5,
    },
    "gta_boundary": {
        "color": "#4f5d73",
        "fillColor": "#c7d3f0",
        "fillOpacity": 0.25,
        "weight": 1.4,
    },
    "freight_corridors": {
        "color": "#8c6bb1",
        "fillColor": "#bcbddc",
        "fillOpacity": 0.25,
        "weight": 1.2,
        "dashArray": "3 2",
    },
    "numbered_highways": {
        "color": "#ff0000",
        "weight": 2.0,
    },
    "target_highways": {
        "color": "#ffaa33",
        "weight": 2.0,
        "dashArray": "6 3",
    },
    "target_fsas": {
        "color": "#d4af00",
        "fillColor": "#ffaa33",
        "fillOpacity": 0.55,
        "weight": 0.8,
    },
    "pembina_fsas": {
        "color": "#1f7a45",
        "fillColor": "#3c9d5e",
        "fillOpacity": 0.55,
        "weight": 0.8,
    },
    "public_charging_fsas": {
        "color": "#cc0000",
        "fillColor": "#ff6666",
        "fillOpacity": 0.5,
        "weight": 0.8,
    },
    "public_charging_cities": {
        "color": "#cc0000",
        "fillColor": "#ff0000",
        "fillOpacity": 0.9,
        "radius": 6,
        "weight": 0.8,
    },
    "refueling_stops": {
        "color": "#ffaa33",
        "fillColor": "#ffaa33",
        "fillOpacity": 0.9,
        "radius": 6,
        "weight": 0.8,
    },
    "freight_corridors": {
        "color": "#000000",
        "fillColor": "#333333",
        "fillOpacity": 0.25,
        "weight": 1.2,
        "dashArray": "3 2",
    },
    "gta_boundary": {
        "color": "#9000ff",
        "fillColor": "#ba66ff",
        "fillOpacity": 0.15,
        "weight": 0,
    },
}

# Legend items organized by section.
LEGEND_ITEMS: List[Dict[str, Any]] = [
    {
        "group": "Of Interest for Public Charging",
        "items": [
            {"label": "Cities", "color": STYLE_MAP["public_charging_cities"]["fillColor"], "shape": "circle", "layer_name": "Public Charging Cities"},
            {"label": "FSAs", "color": STYLE_MAP["public_charging_fsas"]["fillColor"], "shape": "square", "layer_name": "Public Charging FSAs"},
            {"label": "Highways", "color": STYLE_MAP["numbered_highways"]["color"], "shape": "line", "layer_name": "Numbered Highways"},
        ]
    },
    {
        "group": "Highly Used by Consortium Members",
        "items": [
            {"label": "Endpoints", "color": STYLE_MAP["target_fsas"]["fillColor"], "shape": "square", "layer_name": "Target FSAs"},
            {"label": "Highways", "color": STYLE_MAP["target_highways"]["color"], "shape": "line", "layer_name": "Target Highways"},
            {"label": "Refueling Stops", "color": STYLE_MAP["refueling_stops"]["fillColor"], "shape": "circle", "layer_name": "Refueling Stops"},
        ]
    },
    {
        "group": "Third-party Results",
        "items": [
            {"label": "GTHA Priority Zones from Pembina", "color": STYLE_MAP["pembina_fsas"]["fillColor"], "shape": "square", "layer_name": "Pembina FSAs"},
        ]
    },
    {
        "group": "Context",
        "items": [
            {"label": "Ontario Boundary", "color": STYLE_MAP["ontario_boundary"]["color"], "shape": "square", "layer_name": "Ontario Boundary"},
            {"label": "GTA Boundary", "color": STYLE_MAP["gta_boundary"]["color"], "shape": "square", "layer_name": "GTA Boundary"},
            {"label": "Ontario Freight Corridors", "color": STYLE_MAP["freight_corridors"]["color"], "shape": "line", "layer_name": "Primary Freight Corridors"},
        ]
    },
]


def _style_function(style_key: str):
    style = STYLE_MAP.get(style_key, {})

    def fn(_feature):
        return {
            "color": style.get("color", "#444"),
            "fillColor": style.get("fillColor", style.get("color", "#888")),
            "fillOpacity": style.get("fillOpacity", 0.0),
            "weight": style.get("weight", 1.0),
            "dashArray": style.get("dashArray"),
        }

    return fn


def _highway_style_function(style_key: str):
    """Style function for highway features that scales weight by AADTT16 (truck traffic volume)."""
    style = STYLE_MAP.get(style_key, {})
    base_weight = style.get("weight", 1.0)
    
    def fn(feature):
        props = feature.get("properties", {})
        aadtt16 = props.get("AADTT16")
        
        # Scale weight based on AADTT16 (annual average daily truck traffic)
        # Normalize to a reasonable range for visual distinction
        if aadtt16:
            try:
                volume = float(aadtt16)
                # Scale from 0-5000 volume to 0.5-2.5 weight multiplier
                scaled_weight = base_weight * (0.5 + (min(volume, 5000) / 5000) * 2.0)
            except (ValueError, TypeError):
                scaled_weight = base_weight
        else:
            scaled_weight = base_weight
        
        return {
            "color": style.get("color", "#444"),
            "fillColor": style.get("fillColor", style.get("color", "#888")),
            "fillOpacity": style.get("fillOpacity", 0.0),
            "weight": scaled_weight,
            "dashArray": style.get("dashArray"),
        }

    return fn


def _load_geojson(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Check if the GeoJSON has a CRS that needs transformation
    crs = data.get("crs", {})
    crs_name = crs.get("properties", {}).get("name", "")
    
    # If it's in EPSG:3347 (Ontario projection), transform to WGS84
    if "3347" in crs_name:
        gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:3347")
        gdf = gdf.to_crs("EPSG:4326")
        # Convert back to GeoJSON format
        data = json.loads(gdf.to_json())
    
    return data


def _add_geojson_layer(map_obj: folium.Map, data: Dict[str, Any], layer_cfg: Dict[str, str], layer_id: str):
    name = layer_cfg["name"]
    style_key = layer_cfg["style_key"]
    zorder = layer_cfg.get("zorder", 0)
    
    # Use AADTT16-based styling for highway/line layers
    is_highway = style_key in {"target_highways", "numbered_highways", "freight_corridors"}
    style_fn = _highway_style_function(style_key) if is_highway else _style_function(style_key)

    features = data.get("features", [])
    is_point_geom = all(
        feat.get("geometry", {}).get("type") in {"Point", "MultiPoint"}
        for feat in features
    )

    if is_point_geom:
        fg = folium.FeatureGroup(name=name, show=True)
        fg.name = layer_id
        fg._layer_id = layer_id
        fg.options['zindex'] = zorder
        style = STYLE_MAP.get(style_key, {})
        for feat in features:
            geom = feat.get("geometry", {})
            props = feat.get("properties", {})
            gtype = geom.get("type")
            coords = geom.get("coordinates", [])

            def add_point(coord_pair):
                lat, lon = coord_pair[1], coord_pair[0]
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=style.get("radius", 6),
                    color="#000000",
                    fill=True,
                    fill_color=style.get("fillColor", style.get("color", "#555")),
                    fill_opacity=style.get("fillOpacity", 0.9),
                    weight=1.5,
                    tooltip=folium.Tooltip(
                        "<br>".join([f"{k}: {v}" for k, v in props.items()]) or name,
                        sticky=False,
                    ),
                ).add_to(fg)

            if gtype == "Point" and len(coords) >= 2:
                add_point(coords)
            elif gtype == "MultiPoint":
                for pt in coords:
                    if len(pt) >= 2:
                        add_point(pt)

        fg.add_to(map_obj)
    else:
        # Use GeoJson with polygon/line styling and a tooltip listing properties.
        prop_fields = list(features[0].get("properties", {}).keys()) if features else []
        highlight_fn = lambda f: {
            "weight": style_fn(f).get("weight", 1.0) + 1,
        }
        gj = folium.GeoJson(
            data,
            name=name,
            style_function=style_fn,
            highlight_function=highlight_fn,
            tooltip=folium.features.GeoJsonTooltip(
                fields=prop_fields,
                aliases=None,
                sticky=False,
            ) if prop_fields else None,
        )
        gj.name = layer_id
        gj.options['zindex'] = zorder
        gj.add_to(map_obj)
    
    return


def _add_legend(map_obj: folium.Map):
    # Build a grouped legend with section headers.
    legend_html_sections = []
    
    for group_data in LEGEND_ITEMS:
        group_name = group_data["group"]
        items = group_data["items"]
        
        # Add section header
        section_html = f"<div style='font-weight:600;margin-top:10px;margin-bottom:4px;color:#111;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;border-bottom:1px solid #ddd;padding-bottom:3px;'>{group_name}</div>"
        legend_html_sections.append(section_html)
        
        # Add items in section
        for item in items:
            shape = item["shape"]
            color = item["color"]
            label = item["label"]
            layer_name = item.get("layer_name", label)
            
            if shape == "circle":
                marker_html = f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:{color};border:1px solid #000;margin-right:6px;'></span>"
            elif shape == "line":
                marker_html = f"<span style='display:inline-block;width:18px;height:3px;background:{color};margin-right:6px;'></span>"
            else:  # square / polygon
                marker_html = f"<span style='display:inline-block;width:12px;height:12px;background:{color};margin-right:6px;border:1px solid #555;'></span>"
            
            legend_html_sections.append(
                f"<div class='legend-entry' data-layer-name='{layer_name}' "
                f"style='margin:2px 0;display:flex;align-items:center;padding:3px 0;' >"
                f"{marker_html}<span style='font-size:11px;color:#333;user-select:none;'>{label}</span></div>"
            )
    
    sections_str = "".join(legend_html_sections)
    
    # Build legend HTML with proper structure
    legend_div = (
        "<div id='map-legend' style='position:fixed;bottom:20px;left:20px;z-index:9999;"
        "background:rgba(255,255,255,0.95);padding:10px 12px;border:1px solid #ccc;border-radius:6px;"
        "box-shadow:0 2px 6px rgba(0,0,0,0.3);max-width:280px;font-family:Arial,sans-serif;'>"
        f"{sections_str}"
        "</div>"
    )
    
    map_obj.get_root().html.add_child(folium.Element(legend_div))


def build_interactive_map(geojson_dir: Path | None = None, output_path: Path | None = None) -> Path:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    geojson_dir = geojson_dir or project_root / "geojsons"
    output_path = output_path or (project_root / "plots" / "geojson_viewer.html")

    if not geojson_dir.exists():
        raise FileNotFoundError(f"GeoJSON directory not found: {geojson_dir}")

    geojson_data = {}
    for cfg in LAYER_CONFIG:
        path = geojson_dir / cfg["filename"]
        if not path.exists():
            print(f"[Skip] Missing layer {cfg['name']}: {path}")
            continue
        geojson_data[cfg["filename"]] = _load_geojson(path)
        print(f"[Load] {cfg['name']} from {path}")

    # Map initialization centered on Ontario boundary if available.
    boundary_path = geojson_dir / "Ontario_Provincial_Boundary.geojson"
    if boundary_path.exists():
        gdf_boundary = gpd.read_file(boundary_path)
        bounds = gdf_boundary.total_bounds  # minx, miny, maxx, maxy
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        map_obj = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")
        map_obj.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    else:
        map_obj = folium.Map(location=[50.0, -85.0], zoom_start=5, tiles="CartoDB positron")

    # Add layers
    for i, cfg in enumerate(LAYER_CONFIG):
        filename = cfg["filename"]
        data = geojson_data.get(filename)
        if not data:
            continue
        layer_id = f"layer_{i}"
        _add_geojson_layer(map_obj, data, cfg, layer_id)

    _add_legend(map_obj)
    
    # Add Folium's built-in layer control (will work out of the box with proper styling)
    folium.LayerControl(collapsed=False, position='topright').add_to(map_obj)

    # Add script to sync legend with layer control checkboxes
    legend_sync_script = (
        "<script>"
        "function syncLegendWithLayerControl() { "
        "  var layerControlInputs = document.querySelectorAll('.leaflet-control-layers input[type=\"checkbox\"]'); "
        "  var legendEntries = document.querySelectorAll('.legend-entry'); "
        "  legendEntries.forEach(function(entry) { "
        "    var targetLayerName = entry.getAttribute('data-layer-name'); "
        "    var isChecked = false; "
        "    layerControlInputs.forEach(function(input) { "
        "      var label = input.nextElementSibling; "
        "      if (label && label.textContent.trim() === targetLayerName) { "
        "        isChecked = input.checked; "
        "      } "
        "    }); "
        "    entry.style.opacity = isChecked ? '1' : '0.5'; "
        "  }); "
        "} "
        "function initDynamicLegend() { "
        "  var controlPanel = document.querySelector('.leaflet-control-layers'); "
        "  if (!controlPanel) { "
        "    setTimeout(initDynamicLegend, 300); "
        "    return; "
        "  } "
        "  syncLegendWithLayerControl(); "
        "  var inputs = document.querySelectorAll('.leaflet-control-layers input[type=\"checkbox\"]'); "
        "  inputs.forEach(function(input) { "
        "    input.addEventListener('change', syncLegendWithLayerControl); "
        "  }); "
        "} "
        "setTimeout(initDynamicLegend, 800); "
        "</script>"
    )
    map_obj.get_root().html.add_child(folium.Element(legend_sync_script))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(output_path)
    print(f"[Save] Interactive map written to {output_path}")
    return output_path
    print(f"[Save] Interactive map written to {output_path}")
    return output_path


def main():
    build_interactive_map()


if __name__ == "__main__":
    main()
