"""
Interactive GeoJSON viewer using Folium/Leaflet.
Loads all known GeoJSON layers from the "active geojsons" directory and renders
an interactive map with layer toggles and a styled legend (QGIS-like stack).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List

import folium
import geopandas as gpd

# Layer configuration: ordered by zorder for proper stacking.
LAYER_CONFIG = [
    {
        "name": "OpenData: GTA Boundary",
        "filename": "GTA_Boundary.geojson",
        "style_key": "gta_boundary",
        "zorder": 1,
    },
    {
        "name": "Working Group Survey: FSAs",
        "filename": "working_group_survey_fsas.geojson",
        "style_key": "working_group_survey_fsas",
        "zorder": 3,
    },
    {
        "name": "OpenData: Primary Freight Corridors",
        "filename": "Value_of_Goods_2016_-3795296180828259022.geojson",
        "style_key": "freight_corridors",
        "zorder": 5,
    },
    {
        "name": "Working Group Survey: Highways",
        "filename": "working_group_survey_highways.geojson",
        "style_key": "working_group_survey_highways",
        "zorder": 6,
    },
    {
        "name": "Working Group Survey: Current Refueling Stops",
        "filename": "refueling_stops.geojson",
        "style_key": "refueling_stops",
        "zorder": 8,
    },
    {
        "name": "OpenData: City Boundaries",
        "filename": "public_charging_cities_boundaries.geojson",
        "style_key": "city_boundaries",
        "zorder": 99,  # Draw on top
    },
]

# CIMA+/Geotab layers come in rank tiers (tier 1 = highest volume, produced
# by cima_preprocessing.py). Only tier 1 of each dataset is shown initially.
CIMA_STOP_TIERS = 7
CIMA_OD_TIERS = 5
for _tier in range(1, CIMA_STOP_TIERS + 1):
    LAYER_CONFIG.append({
        "name": f"CIMA+ stop data ({_tier})",
        "filename": f"cima_stops_{_tier}.geojson",
        "style_key": "cima_stops",
        "zorder": 10,
        "show": _tier == 1,
    })
for _tier in range(1, CIMA_OD_TIERS + 1):
    LAYER_CONFIG.append({
        "name": f"CIMA+ Origin Destination data ({_tier})",
        "filename": f"cima_od_{_tier}.geojson",
        "style_key": "cima_od",
        "zorder": 11,
        "show": _tier == 1,
    })

# Style palette keyed by style_key in LAYER_CONFIG.
STYLE_MAP: Dict[str, Dict[str, Any]] = {
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
    "working_group_survey_highways": {
        "color": "#0b3d91",
        "weight": 2.0,
        "dashArray": "6 3",
    },
    "working_group_survey_fsas": {
        "color": "#d4af00",
        "fillColor": "#ffaa33",
        "fillOpacity": 1.0,
        "weight": 0.8,
    },
    "refueling_stops": {
        "color": "#ffaa33",
        "fillColor": "#ffaa33",
        "fillOpacity": 0.9,
        "radius": 6,
        "weight": 0.8,
    },
    "city_boundaries": {
        "color": "#b80303",
        "weight": 2.5,
        "fillOpacity": 0.0,
        "opacity": 1.0,
        "fill": False,
        "dashArray": "6, 6",
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
    "cima_stops": {
        "color": "#008b8b",
        "weight": 3.0,
        "opacity": 0.9,
    },
    "cima_od": {
        "color": "#e91e8c",
        "weight": 1.5,
        "opacity": 0.6,
    },
}

# Legend items organized by section.
LEGEND_ITEMS: List[Dict[str, Any]] = [
    {
        "group": "Highly Used by Consortium Members",
        "items": [
            {"label": "Working Group Survey: FSAs", "color": STYLE_MAP["working_group_survey_fsas"]["fillColor"], "shape": "square", "layer_name": "Working Group Survey: FSAs"},
            {"label": "Working Group Survey: Highways", "color": STYLE_MAP["working_group_survey_highways"]["color"], "shape": "line", "layer_name": "Working Group Survey: Highways"},
            {"label": "Working Group Survey: Current Refueling Stops", "color": STYLE_MAP["refueling_stops"]["fillColor"], "shape": "circle", "layer_name": "Working Group Survey: Current Refueling Stops"},
        ]
    },
    {
        "group": "CIMA+ / Geotab Telematics (Toronto, 2024-09-01 to 09-08)",
        "items": [
            {"label": "Stop segments, tiers 1-7 by stop count (width = stops)", "color": STYLE_MAP["cima_stops"]["color"], "shape": "line", "layer_name": "CIMA+ stop data (1)"},
            {"label": "Origin-Destination flows, tiers 1-5 (width = journeys)", "color": STYLE_MAP["cima_od"]["color"], "shape": "line", "layer_name": "CIMA+ Origin Destination data (1)"},
        ]
    },
    {
        "group": "Context",
        "items": [
            {"label": "OpenData: City Boundaries", "color": "#b80303", "shape": "line", "layer_name": "OpenData: City Boundaries", "dash": True},
            {"label": "OpenData: GTA Boundary", "color": STYLE_MAP["gta_boundary"]["color"], "shape": "square", "layer_name": "OpenData: GTA Boundary"},
            {"label": "OpenData: Primary Freight Corridors", "color": STYLE_MAP["freight_corridors"]["color"], "shape": "line", "layer_name": "OpenData: Primary Freight Corridors"},
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
            "opacity": style.get("opacity", 1.0),
            "fill": True,
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
            "opacity": style.get("opacity", 1.0),
            "weight": scaled_weight,
            "dashArray": style.get("dashArray"),
        }

    return fn


def _cima_scaled_style_function(style_key: str, data: Dict[str, Any], metric: str,
                                min_weight: float = 1.0, max_weight: float = 8.0):
    """Style function that scales line weight/opacity by a numeric property.

    Normalizes against the dataset-wide "scaleMax" written into each tier file
    by cima_preprocessing.py, so widths are comparable across tiers; falls
    back to the layer's own maximum."""
    style = STYLE_MAP.get(style_key, {})
    base_opacity = style.get("opacity", 0.9)
    max_value = data.get("scaleMax")
    if not max_value:
        values = [
            feat.get("properties", {}).get(metric) or 0
            for feat in data.get("features", [])
        ]
        max_value = max(values) if values else 1

    def fn(feature):
        value = feature.get("properties", {}).get(metric) or 0
        frac = float(value) / max_value if max_value else 0.0
        return {
            "color": style.get("color", "#444"),
            "fillColor": style.get("fillColor", style.get("color", "#888")),
            "fillOpacity": 0.0,
            "opacity": base_opacity * (0.4 + 0.6 * frac),
            "weight": min_weight + (max_weight - min_weight) * frac,
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
    show = layer_cfg.get("show", True)
    
    # Use AADTT16-based styling for highway/line layers
    is_highway = style_key in {"working_group_survey_highways", "freight_corridors"}
    if style_key == "cima_stops":
        style_fn = _cima_scaled_style_function(style_key, data, "StopCount", 1.0, 6.0)
    elif style_key == "cima_od":
        style_fn = _cima_scaled_style_function(style_key, data, "JourneyCount", 0.8, 7.0)
    elif is_highway:
        style_fn = _highway_style_function(style_key)
    else:
        style_fn = _style_function(style_key)

    features = data.get("features", [])
    is_point_geom = all(
        feat.get("geometry", {}).get("type") in {"Point", "MultiPoint"}
        for feat in features
    )

    if is_point_geom:
        fg = folium.FeatureGroup(name=name, show=show)
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
            show=show,
            style_function=style_fn,
            highlight_function=highlight_fn,
            tooltip=folium.features.GeoJsonTooltip(
                fields=prop_fields,
                aliases=None,
                sticky=False,
            ) if prop_fields else None,
        )
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
                dash = item.get("dash", False)
                if dash:
                    marker_html = f"<span style='display:inline-block;width:18px;height:0;border-bottom:3px dashed {color};margin-right:6px;'></span>"
                else:
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
    geojson_dir = geojson_dir or project_root / "active geojsons"
    output_path = output_path or (project_root / "viewer" / "geojson_viewer.html")

    # Ensure the destination directory exists so the viewer can be written reliably.
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Map initialization centered on the GTA boundary if available.
    boundary_path = geojson_dir / "GTA_Boundary.geojson"
    if boundary_path.exists():
        gdf_boundary = gpd.read_file(boundary_path)
        bounds = gdf_boundary.total_bounds  # minx, miny, maxx, maxy
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        map_obj = folium.Map(location=center, zoom_start=6, tiles="Esri WorldTopoMap", prefer_canvas=True)
        map_obj.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    else:
        map_obj = folium.Map(location=[50.0, -85.0], zoom_start=5, tiles="Esri WorldTopoMap", prefer_canvas=True)

    # Build legend-ordered layer sequence with group metadata
    legend_order = []
    layer_to_label = {}  # Map layer name to legend label
    layer_to_group = {}  # Map layer name to group
    
    for group_data in LEGEND_ITEMS:
        group_name = group_data["group"]
        for item in group_data["items"]:
            layer_name = item["layer_name"]
            legend_order.append(layer_name)
            layer_to_label[layer_name] = item["label"]
            layer_to_group[layer_name] = group_name

    # Tiered layers ("Name (1)", "Name (2)", ...) share one legend entry for
    # tier 1; slot the remaining tiers directly after it so they stay
    # contiguous in the layer control.
    expanded_order = []
    for layer_name in legend_order:
        expanded_order.append(layer_name)
        tier_match = re.match(r"^(.*) \(1\)$", layer_name)
        if tier_match:
            base = re.escape(tier_match.group(1))
            siblings = [
                cfg["name"] for cfg in LAYER_CONFIG
                if cfg["name"] != layer_name
                and re.match(rf"^{base} \(\d+\)$", cfg["name"])
            ]
            expanded_order.extend(
                sorted(siblings, key=lambda n: int(n.rsplit("(", 1)[1].rstrip(")")))
            )
    legend_order = expanded_order

    # Create a mapping from layer name to config
    layer_map = {cfg["name"]: cfg for cfg in LAYER_CONFIG}
    
    # Add layers in z-order (for proper visual stacking)
    for cfg in sorted(LAYER_CONFIG, key=lambda x: x["zorder"]):
        filename = cfg["filename"]
        data = geojson_data.get(filename)
        if not data:
            continue
        _add_geojson_layer(map_obj, data, cfg, f"layer_{cfg['zorder']}")

    _add_legend(map_obj)
    
    # Add Folium's built-in layer control (will work out of the box with proper styling)
    folium.LayerControl(collapsed=False, position='topright').add_to(map_obj)

    # Build layer name mapping for JS
    layer_label_map = {}
    layer_group_map = {}
    for layer_name in legend_order:
        layer_label_map[layer_name] = layer_to_label.get(layer_name, layer_name)
        layer_group_map[layer_name] = layer_to_group.get(layer_name, "")
    
    layer_label_json = json.dumps(layer_label_map)
    layer_group_json = json.dumps(layer_group_map)
    legend_order_json = json.dumps(legend_order)
    
    # Add script to reorder checkboxes and sync legend
    legend_sync_script = (
        "<script>"
        f"var layerLabelMap = {layer_label_json}; "
        f"var layerGroupMap = {layer_group_json}; "
        f"var legendOrder = {legend_order_json}; "
        "function reorderLayerControl() { "
        "  var controlPanel = document.querySelector('.leaflet-control-layers-overlays'); "
        "  if (!controlPanel) return false; "
        "  controlPanel.style.display = 'flex'; "
        "  controlPanel.style.flexDirection = 'column'; "
        "  var labels = controlPanel.querySelectorAll('label'); "
        "  labels.forEach(function(label) { "
        "    var span = label.querySelector('span'); "
        "    if (!span) return; "
        "    var layerName = span.textContent.trim(); "
        "    var order = legendOrder.indexOf(layerName); "
        "    label.style.order = order >= 0 ? order : 1000; "
        "  }); "
        "  return true; "
        "} "
        "function syncLegendWithLayerControl() { "
        "  var checkedLayers = new Set(); "
        "  document.querySelectorAll('.leaflet-control-layers input[type=\"checkbox\"]').forEach(function(input) { "
        "    if (input.checked) { "
        "      var span = input.nextElementSibling; "
        "      if (span && span.textContent) { "
        "        checkedLayers.add(span.textContent.trim()); "
        "      } "
        "    } "
        "  }); "
        "  document.querySelectorAll('.legend-entry').forEach(function(entry) { "
        "    entry.style.opacity = checkedLayers.has(entry.getAttribute('data-layer-name')) ? '1' : '0.5'; "
        "  }); "
        "} "
        "function initLegendSync() { "
        "  if (!document.querySelector('.leaflet-control-layers')) { "
        "    setTimeout(initLegendSync, 200); "
        "    return; "
        "  } "
        "  reorderLayerControl(); "
        "  syncLegendWithLayerControl(); "
        "  document.querySelectorAll('.leaflet-control-layers input[type=\"checkbox\"]').forEach(function(input) { "
        "    input.addEventListener('change', syncLegendWithLayerControl); "
        "  }); "
        "} "
        "if (document.readyState === 'loading') { "
        "  document.addEventListener('DOMContentLoaded', function() { setTimeout(initLegendSync, 500); }); "
        "} else { "
        "  setTimeout(initLegendSync, 500); "
        "} "
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
