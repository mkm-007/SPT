"""
SPT geospatial starter: produce two map examples for internship applications.

Uses Natural Earth low-res countries via GeoPandas sample / remote URL.
No ArcGIS license required. Document CRS and layers in the printed summary.
"""

from __future__ import annotations

from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = ROOT / "data" / "ne_110m_admin_0_countries.zip"
OUT.mkdir(parents=True, exist_ok=True)

# Fallback URL if local zip missing (Natural Earth 110m admin-0, public domain)
NE_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/"
    "ne_110m_admin_0_countries.zip"
)


def load_countries() -> gpd.GeoDataFrame:
    source = DATA if DATA.exists() else NE_URL
    print(f"Loading countries from {source}")
    gdf = gpd.read_file(source)
    print(f"Features: {len(gdf)} | CRS: {gdf.crs}")
    return gdf


def static_world_map(gdf: gpd.GeoDataFrame) -> Path:
    """Choropleth-style overview: continent fill as a simple categorical map."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    gdf.plot(
        column="CONTINENT",
        ax=ax,
        legend=True,
        legend_kwds={"loc": "lower left", "fontsize": 7},
        edgecolor="white",
        linewidth=0.2,
    )
    ax.set_title("World continents (Natural Earth 110m) — SPT practice map")
    ax.set_axis_off()
    path = OUT / "map_static_continents.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def static_region_map(gdf: gpd.GeoDataFrame) -> Path:
    """Zoomed regional map: East Africa focus (humanitarian ops relevance)."""
    region = gdf[gdf["CONTINENT"] == "Africa"].copy()
    # Bounding box roughly East Africa / Horn for a readable regional product
    region = region.cx[20:55, -15:20]
    fig, ax = plt.subplots(1, 1, figsize=(8, 9))
    region.plot(ax=ax, color="#c6dbef", edgecolor="#08306b", linewidth=0.6)
    for _, row in region.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        pt = row.geometry.representative_point()
        name = row.get("NAME") or row.get("ADMIN") or ""
        if name:
            ax.annotate(
                name,
                xy=(pt.x, pt.y),
                fontsize=6,
                ha="center",
                color="#08306b",
            )
    ax.set_title("East Africa / Horn region countries — SPT practice map")
    ax.set_axis_off()
    path = OUT / "map_static_east_africa.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def interactive_map(gdf: gpd.GeoDataFrame) -> Path:
    africa = gdf[gdf["CONTINENT"] == "Africa"]
    m = folium.Map(location=[0, 20], zoom_start=3, tiles="CartoDB positron")
    folium.GeoJson(
        africa.__geo_interface__,
        name="Africa countries",
        style_function=lambda _f: {
            "fillColor": "#3182bd",
            "color": "#08519c",
            "weight": 0.5,
            "fillOpacity": 0.35,
        },
        tooltip=folium.GeoJsonTooltip(fields=["NAME", "CONTINENT"]),
    ).add_to(m)
    folium.LayerControl().add_to(m)
    path = OUT / "map_interactive_africa.html"
    m.save(str(path))
    print(f"Wrote {path}")
    return path


def write_notes(gdf: gpd.GeoDataFrame) -> Path:
    path = OUT / "MAP_NOTES.md"
    path.write_text(
        f"""# Map notes (SPT)

## Data
- Source: Natural Earth 110m Admin 0 countries (public domain)
- Features loaded: {len(gdf)}
- Native CRS: {gdf.crs}

## Products
1. `map_static_continents.png` — world overview by continent
2. `map_static_east_africa.png` — regional country outlines + labels
3. `map_interactive_africa.html` — Folium interactive layer

## What I practiced
- Reading a spatial dataset and inspecting CRS
- Filtering / clipping for a regional product
- Static cartography (Matplotlib) and interactive web map (Folium)
- Writing short notes for a non-technical handoff

## Not claimed
- ArcGIS Enterprise, QGIS production workflows, or remote-sensing indices (next SPT steps)
""",
        encoding="utf-8",
    )
    print(f"Wrote {path}")
    return path


def main() -> None:
    gdf = load_countries()
    static_world_map(gdf)
    static_region_map(gdf)
    interactive_map(gdf)
    write_notes(gdf)
    print("Done. Attach PNG/HTML examples with the Mercy Corps application.")


if __name__ == "__main__":
    main()
