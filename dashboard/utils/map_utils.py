import folium
from folium.plugins import MiniMap, MousePosition


def create_base_map(map_bounds):
    """
    Create a Folium map centred on the AOI with a dark basemap.

    map_bounds: [[south, west], [north, east]]
    """
    south, west = map_bounds[0]
    north, east = map_bounds[1]
    centre      = [(south + north) / 2, (west + east) / 2]

    m = folium.Map(
        location=centre,
        zoom_start=11,
        tiles='CartoDB dark_matter',
        prefer_canvas=True,
    )

    # Add additional tile layers user can switch to
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
    ).add_to(m)

    folium.TileLayer('OpenStreetMap', name='Street Map').add_to(m)

    # AOI boundary box
    folium.Rectangle(
        bounds=map_bounds,
        color='#ffffff',
        weight=1.5,
        fill=False,
        tooltip='Area of Interest',
    ).add_to(m)

    MiniMap(toggle_display=True, tile_layer='CartoDB dark_matter').add_to(m)
    MousePosition(
        position='bottomleft',
        separator=' | Lon: ',
        prefix='Lat: ',
    ).add_to(m)

    return m


def add_water_mask_layer(m, year, png_b64, map_bounds, layer_name=None):
    """Add a water mask PNG overlay to the map."""
    name = layer_name or f'Water Mask {year}'
    folium.raster_layers.ImageOverlay(
        image=png_b64,
        bounds=map_bounds,
        opacity=0.7,
        name=name,
        interactive=False,
        cross_origin=False,
    ).add_to(m)
    return m


def add_change_layer(m, year_a, year_b, png_b64, map_bounds):
    """Add a change detection overlay to the map."""
    folium.raster_layers.ImageOverlay(
        image=png_b64,
        bounds=map_bounds,
        opacity=0.75,
        name=f'Change {year_a} → {year_b}',
        interactive=False,
        cross_origin=False,
    ).add_to(m)
    return m


def add_water_mask_legend(m, year):
    """No-op — legend is now rendered below the map in Streamlit."""
    return m


def finalise_map(m):
    """Add layer control and return the map."""
    folium.LayerControl(collapsed=False).add_to(m)
    return m
