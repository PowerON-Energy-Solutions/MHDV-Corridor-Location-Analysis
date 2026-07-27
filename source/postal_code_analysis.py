"""
Postal Code and Forward Sortation Area (FSA) Analysis
This module analyzes survey postal codes and visualizes their geographic distribution
across Ontario's Forward Sortation Areas.
"""

import json
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import hsv_to_rgb
from pathlib import Path

__all__ = [
    'load_fsa_shapefile',
    'load_postal_codes',
    'load_pembina_fsas',
    'load_ontario_boundary',
    'filter_fsa_by_postal_codes',
    'save_target_fsa_geojson',
    'save_pembina_fsa_geojson',
    'attach_region_to_fsas',
    'visualize_fsas'
]


def load_fsa_shapefile(shapefile_path):
    """
    Load Forward Sortation Area shapefile and filter for Ontario.
    
    Args:
        shapefile_path: Path to the FSA shapefile
        
    Returns:
        GeoDataFrame filtered for Ontario
    """
    gdf = gpd.read_file(shapefile_path)
    print(f"Loaded shapefile with {len(gdf)} total FSA records")
    
    # Filter for Ontario
    gdf_ontario = gdf[gdf['PRNAME'] == 'Ontario'].copy()
    print(f"Filtered to {len(gdf_ontario)} Ontario FSA records")
    
    return gdf_ontario


def load_postal_codes(csv_path):
    """
    Load target postal codes from CSV.
    
    Args:
        csv_path: Path to the postal_codes.csv file
        
    Returns:
        DataFrame with postal codes
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from postal codes CSV")
    print(f"Columns: {list(df.columns)}")
    
    # Use CFSAUID column which contains Forward Sortation Areas
    df['FSA'] = df['Postal Code'].str.upper().str[:3]
    
    print(f"Extracted {df['FSA'].nunique()} unique FSAs")
    print(f"FSAs: {sorted(df['FSA'].unique())}")
    
    return df


def load_pembina_fsas(csv_path):
    """
    Load Pembina corridor FSAs and normalize casing.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from Pembina FSAs CSV")
    expected_cols = {'Region', 'FSA'}
    missing_cols = expected_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Pembina FSA CSV missing columns: {missing_cols}")

    df['FSA'] = df['FSA'].astype(str).str.upper().str.strip().str[:3]
    df['Region'] = df['Region'].astype(str).str.strip()

    print(f"Unique FSAs: {df['FSA'].nunique()} | Regions: {df['Region'].nunique()}")
    print(f"FSAs: {sorted(df['FSA'].unique())}")

    return df


def load_ontario_boundary(geojson_path):
    """
    Load Ontario provincial boundary from GeoJSON.
    
    Args:
        geojson_path: Path to the Ontario boundary GeoJSON
        
    Returns:
        GeoDataFrame with boundary
    """
    gdf = gpd.read_file(geojson_path)
    print(f"Loaded Ontario boundary")
    return gdf


def filter_fsa_by_postal_codes(gdf_fsa, df_postal, gdf_boundary):
    """
    Filter FSA shapefile to only include FSAs from postal codes.
    Aligns coordinate systems between datasets.
    
    Args:
        gdf_fsa: GeoDataFrame with FSA data
        df_postal: DataFrame with postal codes
        gdf_boundary: GeoDataFrame with Ontario boundary (reference CRS)
        
    Returns:
        Filtered and reprojected GeoDataFrame
    """
    # Check and align coordinate systems
    print(f"\n[FSA Filter] Coordinate Systems:")
    print(f"  FSA shapefile CRS: {gdf_fsa.crs}")
    print(f"  Ontario boundary CRS: {gdf_boundary.crs}")
    
    # Reproject FSA to match boundary CRS if they differ
    if gdf_fsa.crs != gdf_boundary.crs:
        print(f"  → Reprojecting FSA shapefile to {gdf_boundary.crs}")
        gdf_fsa = gdf_fsa.to_crs(gdf_boundary.crs)
    else:
        print(f"  ✓ CRS already aligned")
    
    target_fsas = df_postal['FSA'].unique()
    
    # Filter FSA data
    gdf_filtered = gdf_fsa[gdf_fsa['CFSAUID'].str[:3].str.upper().isin(target_fsas)].copy()
    
    print(f"Filtered to {len(gdf_filtered)} FSA features matching target postal codes")
    
    return gdf_filtered


def save_target_fsa_geojson(gdf_fsa_target, output_path: Path) -> Path:
    """Persist the filtered target FSAs to GeoJSON.

    Args:
        gdf_fsa_target: GeoDataFrame containing target FSAs.
        output_path: Destination path for the GeoJSON.

    Returns:
        The path that was written to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_fsa_target.to_file(output_path, driver="GeoJSON")
    print(f"[FSA Export] Saved {len(gdf_fsa_target)} target FSAs to {output_path}")
    return output_path


def attach_region_to_fsas(gdf_fsa, df_regions):
    """Attach region metadata to FSA geometries using the 3-character FSA code."""
    gdf_with_region = gdf_fsa.copy()
    fsa_prefix = gdf_with_region['CFSAUID'].str[:3].str.upper()
    region_lookup = df_regions.set_index('FSA')['Region']
    gdf_with_region['Region'] = fsa_prefix.map(region_lookup)
    gdf_with_region['FSA'] = fsa_prefix

    missing_regions = gdf_with_region['Region'].isna().sum()
    if missing_regions:
        print(f"[Region Join] Warning: {missing_regions} FSA geometries missing region mapping")
    else:
        print("[Region Join] All FSA geometries mapped to regions")

    return gdf_with_region


def save_pembina_fsa_geojson(gdf_fsa_pembina, output_path: Path) -> Path:
    """Persist Pembina corridor FSAs (with regions) to GeoJSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_fsa_pembina.to_file(output_path, driver="GeoJSON")
    print(f"[Pembina Export] Saved {len(gdf_fsa_pembina)} FSAs to {output_path}")
    return output_path


def visualize_fsas(gdf_fsa, gdf_boundary, output_path=None, label_directions=None):
    """
    Create a visualization of Forward Sortation Areas with Ontario boundary.
    
    Args:
        gdf_fsa: GeoDataFrame with target FSA features
        gdf_boundary: GeoDataFrame with Ontario boundary
        output_path: Path to save the output figure. If None, saves to default location.
        label_directions: Dict mapping FSA codes to directions ('left', 'right', 'up', 'down').
                         If None, defaults to 'up' for all small features.
    """
    if output_path is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        plots_dir = project_root / 'plots'
        plots_dir.mkdir(exist_ok=True)
        output_path = plots_dir / 'fsa_visualization.png'
    
    # Default label directions (customize here)
    if label_directions is None:
        label_directions = {
            'L4N': 'left',
            'K7G': 'right',
            'K6V': 'lower right',
            'K7A': 'down',
            'K6J': 'down',
            'L6Y': 'lower left',
            'L5T': 'down',
            'L6T': 'left'
        }
    
    fig, ax = plt.subplots(figsize=(14, 10))

    # Build a color palette with high contrast
    fsa_codes = gdf_fsa['CFSAUID'].str[:3].unique()
    n_colors = len(fsa_codes)
    
    # Use a curated palette of visually distinct colors
    distinct_colors = [
        '#E41A1C',  # Red
        '#377EB8',  # Blue
        '#4DAF4A',  # Green
        '#984EA3',  # Purple
        '#FF7F00',  # Orange
        '#A65628',  # Brown
        '#F781BF',  # Pink
        '#999999',  # Gray
        '#B3DE69',  # Light green
        '#BEBADA',  # Light purple
        '#FB8072',  # Light red
        '#80B1D3',  # Light blue
        '#FDB462',  # Light orange
        '#8DD3C7',  # Teal
        '#FFFFB3',  # Yellow
        '#FCCDE5',  # Light pink
        '#B3CDE3',  # Light cyan
        '#DECBE4',  # Lavender
        '#FED9A6',  # Peach
        '#FFFFCC',  # Pale yellow
        '#E5D8BD',  # Tan
        '#FDDAEC',  # Rose
        '#F2F2F2',  # Off-white
        '#CCCCCC',  # Silver
    ]
    
    # Shuffle palette and assign colors in order for even distribution
    np.random.seed(42)
    shuffled_palette = distinct_colors.copy()
    np.random.shuffle(shuffled_palette)
    
    # Create color map by assigning shuffled palette colors sequentially
    color_map = dict(zip(fsa_codes, shuffled_palette[:n_colors]))
    
    gdf_fsa = gdf_fsa.assign(_fsa_color=gdf_fsa['CFSAUID'].str[:3].map(color_map))
    
    # Plot Ontario boundary
    gdf_boundary.plot(ax=ax, color='#F0F0F0', edgecolor='#333333',
                      linewidth=2, alpha=0.5, zorder=0)
    
    # Plot target FSAs
    gdf_fsa.plot(ax=ax, color=list(gdf_fsa['_fsa_color']), edgecolor='#2c3e50',
                 linewidth=1.5, alpha=0.7, zorder=2)
    
    # Add FSA labels
    for idx, row in gdf_fsa.iterrows():
        centroid = row.geometry.centroid
        fsa_code = row['CFSAUID'][:3]
        minx, miny, maxx, maxy = row.geometry.bounds
        max_span = max(maxx - minx, maxy - miny)
        small_feature = max_span < 0.4
        label_x, label_y = centroid.x, centroid.y
        if small_feature:
            offset = 0.12 + max(max_span * 0.5, 0.08)  # Reduced multiplier and increased floor
            direction = label_directions.get(fsa_code, 'up')  # Default to 'up'
            
            if direction == 'up':
                label_y += offset
            elif direction == 'down':
                label_y -= offset
            elif direction == 'right':
                label_x += offset
            elif direction == 'left':
                label_x -= offset
            elif direction == 'upper right':
                label_x += offset
                label_y += offset
            elif direction == 'upper left':
                label_x -= offset
                label_y += offset
            elif direction == 'lower right':
                label_x += offset
                label_y -= offset
            elif direction == 'lower left':
                label_x -= offset
                label_y -= offset
        ax.annotate(
            fsa_code,
            xy=(centroid.x, centroid.y),
            xytext=(label_x, label_y),
            fontsize=10,
            ha='center',
            va='center',
            fontweight='bold',
            color='black',
            zorder=3,
            arrowprops=dict(arrowstyle='-', color='#444444', lw=0.8, alpha=0.8) if small_feature else None
        )
    # Zoom to extent of target FSAs with a small padding
    minx, miny, maxx, maxy = gdf_fsa.total_bounds
    x_pad = (maxx - minx) * 0.05
    y_pad = (maxy - miny) * 0.05
    ax.set_xlim(minx - x_pad, maxx + x_pad)
    ax.set_ylim(miny - y_pad, maxy + y_pad)
    
    # Add major cities for context
    major_cities = {
        'Toronto': (-79.3832, 43.6532),
        'Ottawa': (-75.6972, 45.4215),
        'Kingston': (-76.4860, 44.2312),
        'Belleville': (-77.3832, 44.1628),
        'Peterborough': (-78.3197, 44.3091),
        'London': (-81.2497, 42.9849),
        'Windsor': (-83.0368, 42.3149),
        'Sudbury': (-80.9930, 46.4917),
    }
    
    for city, (lon, lat) in major_cities.items():
        # Only plot cities within the current view
        if minx - x_pad <= lon <= maxx + x_pad and miny - y_pad <= lat <= maxy + y_pad:
            ax.plot(lon, lat, marker='*', color='black', markersize=8, zorder=4)
            ax.text(lon, lat - 0.08, city, fontsize=9, ha='center', 
                   style='italic', color='#333333', zorder=4,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='none', alpha=0.7))
    
    ax.set_title('Forward Sortation Areas - Survey Target Regions', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    #plt.show()


def main():
    """Main execution function."""
    # Define paths relative to this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / 'data'
    geojson_dir = project_root / 'active geojsons'
    
    fsa_shapefile = data_dir / 'lfsa000b21a_e' / 'lfsa000b21a_e.shp'
    postal_codes_csv = data_dir / 'postal_codes.csv'
    pembina_fsas_csv = data_dir / 'pembina_fsas.csv'
    ontario_boundary_geojson = data_dir / 'Ontario_Provincial_Boundary.geojson'
    
    # Verify files exist
    print(f"Looking for files in: {data_dir}")
    for file_path, file_name in [
        (fsa_shapefile, 'FSA Shapefile'),
        (postal_codes_csv, 'Postal Codes CSV'),
        (pembina_fsas_csv, 'Pembina FSAs CSV'),
        (ontario_boundary_geojson, 'Ontario Boundary GeoJSON')
    ]:
        exists_status = "✓" if file_path.exists() else "✗"
        print(f"  {exists_status} {file_name}: {file_path}")
    
    # Check if files exist before proceeding
    if not fsa_shapefile.exists():
        raise FileNotFoundError(f"FSA Shapefile not found at: {fsa_shapefile}")
    if not postal_codes_csv.exists():
        raise FileNotFoundError(f"Postal codes CSV not found at: {postal_codes_csv}")
    if not pembina_fsas_csv.exists():
        raise FileNotFoundError(f"Pembina FSAs CSV not found at: {pembina_fsas_csv}")
    if not ontario_boundary_geojson.exists():
        raise FileNotFoundError(f"Ontario boundary GeoJSON not found at: {ontario_boundary_geojson}")
    
    print("=" * 70)
    print("POSTAL CODE AND FORWARD SORTATION AREA ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading Forward Sortation Area shapefile...")
    gdf_fsa_all = load_fsa_shapefile(fsa_shapefile)
    
    print("\n2. Loading postal codes...")
    df_postal = load_postal_codes(postal_codes_csv)
    
    print("\n3. Loading Ontario boundary...")
    gdf_boundary = load_ontario_boundary(ontario_boundary_geojson)
    
    # Filter FSAs
    print("\n4. Filtering FSAs to match target postal codes...")
    gdf_fsa_target = filter_fsa_by_postal_codes(gdf_fsa_all, df_postal, gdf_boundary)

    # Persist filtered FSAs for downstream use (write to geojsons/ for consistency)
    target_fsa_geojson = geojson_dir / 'working_group_survey_fsas.geojson'
    save_target_fsa_geojson(gdf_fsa_target, target_fsa_geojson)

    print("\n5. Loading Pembina FSAs...")
    df_pembina = load_pembina_fsas(pembina_fsas_csv)

    print("\n6. Filtering Pembina FSAs and attaching regions...")
    gdf_fsa_pembina = filter_fsa_by_postal_codes(gdf_fsa_all, df_pembina, gdf_boundary)
    gdf_fsa_pembina = attach_region_to_fsas(gdf_fsa_pembina, df_pembina)

    pembina_fsa_geojson = geojson_dir / 'pembina_fsas.geojson'
    save_pembina_fsa_geojson(gdf_fsa_pembina, pembina_fsa_geojson)
    
    # Display statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total postal codes loaded: {len(df_postal)}")
    print(f"Unique FSAs in postal codes: {df_postal['FSA'].nunique()}")
    print(f"FSA features in shapefile: {len(gdf_fsa_target)}")
    print(f"Total area covered: {gdf_fsa_target.geometry.area.sum() / 1e6:.2f} km²")
    print(f"Pembina CSV rows: {len(df_pembina)} | Unique Pembina FSAs: {df_pembina['FSA'].nunique()}")
    print(f"Pembina FSA geometries exported: {len(gdf_fsa_pembina)}")
    
    # Visualize
    print("\n7. Creating visualization...")
    visualize_fsas(gdf_fsa_target, gdf_boundary)
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    
    return gdf_fsa_target, df_postal, gdf_boundary


if __name__ == '__main__':
    gdf_fsa, df_postal, gdf_boundary = main()
