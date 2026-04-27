import os
import sys
import streamlit as st
from streamlit_folium import st_folium

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import YEARS
from utils.data_loader import (
    load_summary, load_water_mask,
    get_map_bounds, mask_to_overlay_png,
    change_map_to_overlay_png, compute_stats_for_year,
)
from utils.charts import (
    area_timeseries_chart, yoy_change_chart,
    quality_timeseries_chart, change_summary_chart,
)
from utils.map_utils import (
    create_base_map, add_water_mask_layer,
    add_change_layer, add_water_mask_legend, finalise_map,
)


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Water Body Monitor',
    page_icon='🌊',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0e1117; }
  [data-testid="stSidebar"]          { background: #1a1a2e; }
  .metric-card {
    background: #1a1a2e;
    border: 1px solid #2d2d2d;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
  }
  .metric-label  { font-size: 12px; color: #aaa; margin-bottom: 4px; }
  .metric-value  { font-size: 24px; font-weight: 700; color: white; }
  .metric-delta  { font-size: 12px; margin-top: 4px; }
  .delta-pos     { color: #2dc653; }
  .delta-neg     { color: #ef233c; }
  .section-title { font-size: 13px; color: #aaa; text-transform: uppercase;
                   letter-spacing: 1px; margin: 16px 0 8px 0; }
  div[data-testid="stTabs"] button { color: #aaa; }
  div[data-testid="stTabs"] button[aria-selected="true"] { color: white; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def metric_card(label, value, delta=None, unit=''):
    if value is None:
        value_str = 'N/A'
    elif isinstance(value, float):
        value_str = f'{value:.3f}{unit}'
    else:
        value_str = f'{value}{unit}'

    delta_html = ''
    if delta is not None and not (delta != delta):   # not NaN
        sign      = '+' if delta >= 0 else ''
        css_class = 'delta-pos' if delta >= 0 else 'delta-neg'
        arrow     = '▲' if delta >= 0 else '▼'
        delta_html = (f'<div class="metric-delta {css_class}">'
                      f'{arrow} {sign}{delta:.3f} vs prev year</div>')

    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value_str}</div>
      {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner('Loading data...'):
    df         = load_summary()
    map_bounds = get_map_bounds()
    available_years = [y for y in YEARS if y in df.index]


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## Water Body Monitor')
    st.markdown('---')

    section('Display Mode')
    mode = st.radio(
        label='mode',
        options=['Water Mask', 'Change Detection'],
        label_visibility='collapsed',
    )

    st.markdown('---')
    section('Map Year' if mode == 'Water Mask' else 'Change Detection Years')

    if mode == 'Water Mask':
        selected_year = st.select_slider(
            'Select year for map',
            options=available_years,
            value=available_years[-1],
            label_visibility='collapsed',
        )
        compare_year = None
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption('From')
            year_a = st.selectbox('From', available_years[:-1],
                                  index=0, label_visibility='collapsed')
        with col_b:
            st.caption('To')
            year_b = st.selectbox('To', available_years[1:],
                                  index=len(available_years) - 2,
                                  label_visibility='collapsed')
        selected_year = year_b

    st.markdown('---')
    section('Chart Year Range')
    year_range = st.slider(
        'Year range',
        min_value=available_years[0],
        max_value=available_years[-1],
        value=(available_years[0], available_years[-1]),
        label_visibility='collapsed',
    )

    st.markdown('---')
    section('Quality Metrics')
    quality_options = {
        'ndti'    : 'Turbidity (NDTI)',
        'ndci'    : 'Chlorophyll-a (NDCI)',
        'clarity' : 'Water Clarity',
        'algae'   : 'Algae Index',
        'sediment': 'Sediment',
    }
    selected_quality = [
        col for col, label in quality_options.items()
        if st.checkbox(label, value=(col in ['ndti', 'ndci', 'clarity']))
    ]


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('## 🌊 Water Body Monitor')
st.caption(f'Time series analysis | {available_years[0]}–{available_years[-1]} | March–June composite')
st.markdown('---')


# ── ROW 1: MAP + STATS ────────────────────────────────────────────────────────
map_col, stats_col = st.columns([6, 4], gap='large')

with map_col:
    section(f'Map — {"Water Mask " + str(selected_year) if mode == "Water Mask" else f"Change {year_a} → {year_b}"}')

    with st.spinner('Rendering map...'):
        m = create_base_map(map_bounds)

        if mode == 'Water Mask':
            png, overlay_bounds = mask_to_overlay_png(selected_year)
            m = add_water_mask_layer(m, selected_year, png, overlay_bounds)
            m = add_water_mask_legend(m, selected_year)
        else:
            png, overlay_bounds = change_map_to_overlay_png(year_a, year_b)
            m = add_change_layer(m, year_a, year_b, png, overlay_bounds)

        m = finalise_map(m)
        st_folium(m, height=460, width=None, returned_objects=[])

    # Legend rendered below the map (outside the iframe)
    if mode == 'Water Mask':
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:20px;padding:8px 4px 2px;font-size:13px;color:#ccc;">
          <strong style="color:white;">Water Mask &mdash; {selected_year}</strong>
          <span>
            <span style="background:#0077b6;display:inline-block;width:16px;height:12px;
                         border-radius:3px;vertical-align:middle;margin-right:5px;"></span>
            Water
          </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:20px;padding:8px 4px 2px;font-size:13px;color:#ccc;">
          <strong style="color:white;">Water and Change {year_a} &rarr; {year_b}</strong>
          <span>
            <span style="background:#00b4d8;display:inline-block;width:16px;height:12px;
                         border-radius:3px;vertical-align:middle;margin-right:5px;"></span>
            Stable Water
          </span>
          <span>
            <span style="background:#2dc653;display:inline-block;width:16px;height:12px;
                         border-radius:3px;vertical-align:middle;margin-right:5px;"></span>
            Water Gain
          </span>
          <span>
            <span style="background:#ef233c;display:inline-block;width:16px;height:12px;
                         border-radius:3px;vertical-align:middle;margin-right:5px;"></span>
            Water Loss
          </span>
        </div>
        """, unsafe_allow_html=True)


with stats_col:
    section(f'Statistics — {selected_year}')
    stats     = compute_stats_for_year(df, selected_year)

    metric_card('Water Area',
                stats.get('water_area_km2'),
                delta=stats.get('area_change_km2'),
                unit=' km²')

    pct = stats.get('area_change_pct')
    pct_str = f'{pct:+.1f}%' if pct is not None and pct == pct else 'N/A'
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">% Change vs Previous Year</div>
      <div class="metric-value" style="color: {'#2dc653' if (pct or 0) >= 0 else '#ef233c'}">
        {pct_str}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    section('Water Quality')

    quality_labels = {
        'ndti'    : ('Turbidity',     'Higher = more turbid'),
        'ndci'    : ('Chlorophyll-a', 'Higher = more algae'),
        'clarity' : ('Water Clarity', 'Higher = clearer'),
        'sediment': ('Sediment',      'Higher = more sediment'),
    }

    # Compute deltas from previous year
    prev_stats = compute_stats_for_year(df, selected_year - 1) if (selected_year - 1) in df.index else {}

    for col, (label, note) in quality_labels.items():
        val       = stats.get(col)
        prev_val  = prev_stats.get(col)
        delta     = (val - prev_val) if (val is not None and prev_val is not None) else None
        metric_card(f'{label}', val, delta=delta)


# ── ROW 2: CHART TABS ─────────────────────────────────────────────────────────
st.markdown('---')
tab_area, tab_quality, tab_change = st.tabs([
    '📈 Water Area', '🔬 Water Quality', '🗺️ Change Summary'
])

with tab_area:
    col1, col2 = st.columns([6, 4], gap='large')
    with col1:
        st.plotly_chart(
            area_timeseries_chart(df, year_range),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            yoy_change_chart(df, year_range),
            use_container_width=True,
        )

    # Summary table
    st.markdown('---')
    section('Data Table')
    display_cols = ['water_area_km2', 'area_change_km2', 'area_change_pct']
    display_df   = df.loc[year_range[0]:year_range[1], display_cols].copy()
    display_df.columns = ['Area (km²)', 'Change (km²)', 'Change (%)']
    st.dataframe(
        display_df.style.format({
            'Area (km²)' : '{:.3f}',
            'Change (km²)': '{:+.3f}',
            'Change (%)' : '{:+.1f}',
        }).background_gradient(subset=['Area (km²)'], cmap='Blues'),
        use_container_width=True,
    )

with tab_quality:
    if not selected_quality:
        st.info('Select at least one quality metric from the sidebar.')
    else:
        st.plotly_chart(
            quality_timeseries_chart(df, year_range, selected_quality),
            use_container_width=True,
        )

        # Quality table
        st.markdown('---')
        section('Quality Data Table')
        quality_df = df.loc[year_range[0]:year_range[1],
                            [c for c in selected_quality if c in df.columns]]
        quality_df.columns = [quality_options.get(c, c) for c in quality_df.columns]
        st.dataframe(
            quality_df.style.format('{:.4f}'),
            use_container_width=True,
        )

with tab_change:
    st.plotly_chart(
        change_summary_chart(df, year_range),
        use_container_width=True,
    )

    # Overall change summary
    st.markdown('---')
    section('Overall Change Summary')
    first_y = year_range[0]
    last_y  = year_range[1]
    if first_y in df.index and last_y in df.index:
        area_first  = df.loc[first_y, 'water_area_km2']
        area_last   = df.loc[last_y,  'water_area_km2']
        total_change = area_last - area_first
        total_pct    = (total_change / area_first * 100) if area_first > 0 else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card(f'Area in {first_y}', area_first, unit=' km²')
        with c2:
            metric_card(f'Area in {last_y}',  area_last,  unit=' km²')
        with c3:
            metric_card('Total Change',  total_change,
                        delta=None, unit=' km²')
            sign  = '+' if total_pct >= 0 else ''
            color = '#2dc653' if total_pct >= 0 else '#ef233c'
            st.markdown(f'<div style="color:{color};font-size:18px;font-weight:700;">'
                        f'{sign}{total_pct:.1f}% overall</div>',
                        unsafe_allow_html=True)
