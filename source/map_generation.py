"""
Map generation utilities for Ontario highway visualization.
Contains functions for processing highway data and creating PDF/PNG maps.
"""

import json
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import re
import numpy as np


def load_ontario_boundary(boundary_path):
    """Load Ontario boundary from GeoJSON"""
    with open(boundary_path, 'r') as f:
        geojson_data = json.load(f)
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
    return gdf


def normalize_highway_name(name):
    """Normalize highway name to extract just the number"""
    if not name:
        return None
    normalized = re.sub(r'^(highway|hwy|hw)\s*', '', str(name).strip(), flags=re.IGNORECASE)
    if 'QEW' in normalized.upper():
        return 'QEW'
    match = re.match(r'^(\d+|QEW)', normalized, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return normalized.upper()


def load_data(geojson_path, highways_csv_path):
    """Load GeoJSON and highways CSV"""
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    highways_df = pd.read_csv(highways_csv_path)
    return geojson_data, highways_df


def geojson_to_gdf(geojson_data, highways_df=None, filter_targets=False):
    """Convert GeoJSON to GeoDataFrame"""
    features = []
    
    for feature in geojson_data['features']:
        props = feature.get('properties', {})
        geometry = feature.get('geometry', {})
        
        if geometry.get('type') == 'LineString':
            props['geometry'] = geometry
            features.append(props)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features([{'properties': f, 'geometry': f.pop('geometry')} for f in features])
    
    # Add matched_highway if filtering for targets
    if filter_targets and highways_df is not None:
        target_highways_set = set(highways_df['highway'].astype(str))
        
        # QEW pattern matching - MUST be checked before standard matching
        # because QEW segments have HWY_NUM=1 which could match other highways
        qew_patterns = ["QEW", "Queen Elizabeth", "Elizabeth Way"]
        qew_regex = re.compile("|".join(qew_patterns), re.IGNORECASE)
        
        gdf['matched_highway'] = None
        
        for idx, row in gdf.iterrows():
            road_name = row.get('RDNAME', '')
            hwy_num = row.get('HWY_NUM')
            
            # ALWAYS check for QEW patterns in RDNAME first, regardless of target set
            # If found AND QEW is in our targets, assign it as QEW
            if road_name and qew_regex.search(str(road_name)):
                if 'QEW' in target_highways_set:
                    gdf.at[idx, 'matched_highway'] = 'QEW'
                    continue
                # If QEW pattern found but QEW not in targets, skip to avoid mismatching
                continue
            
            # Then check standard highway matching for non-QEW roads
            normalized_name = normalize_highway_name(road_name)
            normalized_hwy = normalize_highway_name(str(hwy_num)) if hwy_num else None
            
            if normalized_name in target_highways_set:
                gdf.at[idx, 'matched_highway'] = normalized_name
            elif normalized_hwy in target_highways_set:
                gdf.at[idx, 'matched_highway'] = normalized_hwy
        
        # Filter to only target highways
        gdf = gdf[gdf['matched_highway'].notna()]
    
    return gdf


def get_highway_color(count):
    """Get color for highway based on its count value"""
    if count == 1:
        return '#FF0000'  # Red
    elif count == 2:
        return '#FFA500'  # Orange
    elif count == 3:
        return '#00AA00'  # Green
    else:
        return '#0000FF'  # Blue for higher counts


def get_distinct_colors(n):
    """Generate n distinct colors for individual highways"""
    # Use a colormap to generate distinct colors
    cmap = plt.colormaps.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(n)]
    # Convert to hex
    return ['#%02x%02x%02x' % tuple([int(c*255) for c in color[:3]]) for color in colors]


def create_pdf_map(gdf_all, gdf_targets, gdf_ontario, highways_df, filename='highway_network_map.pdf', output_dir='plots'):
    """Create a static PDF map with geopandas and matplotlib"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Set background color (white)
    ax.set_facecolor('white')
    
    # Add Ontario boundary
    gdf_ontario.plot(ax=ax, color='#F0F0F0', edgecolor='#333333', linewidth=2, alpha=0.5, zorder=0)
    
    # Plot base network (all roads in black)
    gdf_all.plot(ax=ax, color='black', linewidth=0.5, alpha=0.4, zorder=1)
    
    # Create color mapping for target highways (convert to string for matching)
    highway_info = highways_df.copy()
    highway_info['highway'] = highway_info['highway'].astype(str)
    highway_info = highway_info.set_index('highway').to_dict('index')
    
    # Collect all colored line segments
    line_segments = []
    line_colors = []
    
    # Add target highways with colors based on count
    for hwy in sorted(gdf_targets['matched_highway'].unique()):
        if hwy in highway_info:
            count = highway_info[hwy]['count']
            color = get_highway_color(count)
            
            # Filter this highway
            hwy_gdf = gdf_targets[gdf_targets['matched_highway'] == hwy]
            
            # Collect line segments for this highway
            for idx, row in hwy_gdf.iterrows():
                geom = row.geometry
                if geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                    line_segments.append(coords)
                    line_colors.append(color)
    
    # Create LineCollection for all colored segments
    if line_segments:
        lc = LineCollection(line_segments, colors=line_colors, linewidths=2, alpha=0.9, zorder=3)
        ax.add_collection(lc)
        ax.autoscale()
    
    # Create simplified legend (just colors and counts)
    legend_elements = []
    legend_elements.append(mpatches.Patch(color='black', alpha=0.4, label='Road Network'))
    
    # Group by count and create legend
    counts_dict = {}
    for idx, row in highways_df.iterrows():
        count = row['count']
        if count not in counts_dict:
            counts_dict[count] = True
    
    # Add legend items for each count value
    for count in sorted(counts_dict.keys(), reverse=True):
        color = get_highway_color(count)
        legend_elements.append(mpatches.Patch(color=color, label=f'Count: {count}'))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
    
    # Set title and labels
    ax.set_title('Ontario Highway Network with Target Highways Highlighted', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save as both PDF and PNG
    output_pdf = f"{output_dir}/{filename}"
    output_png = f"{output_dir}/{filename.replace('.pdf', '.png')}"
    
    plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')
    
    return fig, ax


def create_weighted_pdf_map(gdf_all, gdf_targets, gdf_ontario, highways_df, weight_by, filename='highway_network_map_weighted.pdf', output_dir='plots'):
    """Create a static PDF map with line width weighted by attribute"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Set background color (white)
    ax.set_facecolor('white')
    
    # Add Ontario boundary
    gdf_ontario.plot(ax=ax, color='#F0F0F0', edgecolor='#333333', linewidth=2, alpha=0.5, zorder=0)
    
    # Plot base network (all roads in black)
    gdf_all.plot(ax=ax, color='black', linewidth=0.5, alpha=0.4, zorder=1)
    
    # Get min/max for weight attribute
    valid_values = gdf_targets[weight_by].dropna()
    if len(valid_values) == 0:
        min_val = 1
        max_val = 1
    else:
        min_val = valid_values.min()
        max_val = valid_values.max()
        
        if max_val == min_val:
            min_val = max_val - 1
    
    # Create color mapping for target highways (convert to string for matching)
    highway_info = highways_df.copy()
    highway_info['highway'] = highway_info['highway'].astype(str)
    highway_info = highway_info.set_index('highway').to_dict('index')
    
    # Collect all colored line segments with their widths
    line_segments = []
    line_colors = []
    line_widths = []
    
    # Add target highways with colors based on count and width based on attribute
    for hwy in sorted(gdf_targets['matched_highway'].unique()):
        if hwy in highway_info:
            count = highway_info[hwy]['count']
            color = get_highway_color(count)
            
            # Filter this highway
            hwy_gdf = gdf_targets[gdf_targets['matched_highway'] == hwy]
            
            # Collect line segments with their widths
            for idx, row in hwy_gdf.iterrows():
                if pd.notna(row[weight_by]) and max_val > min_val:
                    # Calculate width based on attribute value
                    normalized = (row[weight_by] - min_val) / (max_val - min_val)
                    linewidth = 0.5 + normalized * 4  # Width from 0.5 to 4.5
                else:
                    linewidth = 2.5
                
                geom = row.geometry
                if geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                    line_segments.append(coords)
                    line_colors.append(color)
                    line_widths.append(linewidth)
    
    # Create LineCollection for all colored segments with varying widths
    if line_segments:
        lc = LineCollection(line_segments, colors=line_colors, linewidths=line_widths, alpha=0.9, zorder=3)
        ax.add_collection(lc)
        ax.autoscale()
    
    # Create simplified legend (just colors and counts)
    legend_elements = []
    legend_elements.append(mpatches.Patch(color='black', alpha=0.4, label='Road Network'))
    
    # Group by count and create legend
    counts_dict = {}
    for idx, row in highways_df.iterrows():
        count = row['count']
        if count not in counts_dict:
            counts_dict[count] = True
    
    # Add legend items for each count value
    for count in sorted(counts_dict.keys(), reverse=True):
        color = get_highway_color(count)
        legend_elements.append(mpatches.Patch(color=color, label=f'Count: {count}'))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
    
    # Set title and labels
    ax.set_title(f'Ontario Highway Network - Line Width Weighted by {weight_by}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save as both PDF and PNG
    output_pdf = f"{output_dir}/{filename}"
    output_png = f"{output_dir}/{filename.replace('.pdf', '.png')}"
    
    plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')
    
    return fig, ax
