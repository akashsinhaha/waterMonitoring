import os
import sys
import logging
import tempfile

# Suppress the spurious torch.classes RuntimeError logged by Streamlit's file watcher
logging.getLogger('streamlit.watcher.local_sources_watcher').setLevel(logging.ERROR)

import streamlit as st
from streamlit_folium import st_folium

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
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
    add_change_layer, add_water_mask_legend,
    add_draw_control, finalise_map,
)
from utils import gee_loader, inference


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
def quality_status(metric_key, value):
    """Return (status_label, color) for a quality metric value."""
    if value is None or value != value:
        return None
    # For these metrics, lower is better
    worse_when_higher = {
        'ndti'    : (0.0,  0.10),
        'ndci'    : (0.0,  0.05),
        'sediment': (0.05, 0.15),
    }
    # For clarity, higher is better
    better_when_higher = {
        'clarity': (0.1, 0.05),
    }
    if metric_key in worse_when_higher:
        good_threshold, fair_threshold = worse_when_higher[metric_key]
        if value <= good_threshold:
            return ('Good', '#2dc653')
        elif value <= fair_threshold:
            return ('Fair', '#f9c74f')
        else:
            return ('Poor', '#ef233c')
    elif metric_key in better_when_higher:
        good_threshold, fair_threshold = better_when_higher[metric_key]
        if value >= good_threshold:
            return ('Good', '#2dc653')
        elif value >= fair_threshold:
            return ('Fair', '#f9c74f')
        else:
            return ('Poor', '#ef233c')
    return None


def metric_card(label, value, delta=None, unit='', description='', status=None):
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

    description_html = (f'<div style="font-size:11px;color:#888;margin-bottom:6px;">'
                        f'{description}</div>') if description else ''

    status_html = ''
    if status is not None:
        status_label, status_color = status
        status_html = (f'<span style="float:right;font-size:11px;font-weight:600;'
                       f'color:{status_color};background:rgba(255,255,255,0.06);'
                       f'padding:2px 8px;border-radius:12px;">'
                       f'{status_label}</span>')

    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{status_html}{label}</div>
      {description_html}
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
        options=['Water Mask', 'Change Detection', 'Custom AOI'],
        label_visibility='collapsed',
    )

    if mode != 'Custom AOI':
        st.markdown('---')
        section('Map Year' if mode == 'Water Mask' else 'Change Detection Years')

        if mode == 'Water Mask':
            selected_year = st.select_slider(
                'Select year for map',
                options=available_years,
                value=available_years[-1],
                label_visibility='collapsed',
            )
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
            'ndti'    : 'Water Muddiness',
            'ndci'    : 'Algae Level',
            'clarity' : 'Water Clearness',
            'algae'   : 'Algae Bloom Risk',
            'sediment': 'Soil & Sand',
        }
        selected_quality = [
            col for col, label in quality_options.items()
            if st.checkbox(label, value=(col in ['ndti', 'ndci', 'clarity']))
        ]
    else:
        # Defaults used by chart tabs (not shown in Custom AOI mode)
        selected_year = available_years[-1]
        year_range    = (available_years[0], available_years[-1])
        quality_options  = {}
        selected_quality = []

    st.markdown('---')
    section('GEE Status')
    try:
        import ee as _ee
        _gee_available = True
    except ImportError:
        _gee_available = False

    if not _gee_available:
        st.markdown('<span style="color:#ef233c;">❌ earthengine-api not installed</span>', unsafe_allow_html=True)
        st.caption('Run: pip install earthengine-api geemap')
    else:
        st.markdown('<span style="color:#2dc653;">✅ earthengine-api ready</span>', unsafe_allow_html=True)
        st.caption('Authenticate once:\nearthengine authenticate')


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('## 🌊 Water Body Monitor')
st.caption(f'Time series analysis | {available_years[0]}–{available_years[-1]} | March–June composite')
st.markdown('---')


# ── ROW 1: MAP + STATS  (or Custom AOI panel) ────────────────────────────────
if mode == 'Custom AOI':
    # ── CUSTOM AOI LAYOUT ─────────────────────────────────────────────────────
    _cached_custom = st.session_state.get('custom_results', {})
    _is_multi_year = (
        bool(_cached_custom)
        and _cached_custom.get('start_year') != _cached_custom.get('end_year')
        and _cached_custom.get('change_png') is not None
    )

    aoi_map_col, aoi_controls_col = st.columns([6, 4], gap='large')

    with aoi_map_col:
        section('Draw Your Area of Interest')
        st.caption('Step 1 — Click the ▭ rectangle tool on the map and draw a box over your area.')

        custom_map = create_base_map(map_bounds)
        custom_map = add_draw_control(custom_map)

        if _cached_custom:
            _rc_start = _cached_custom.get('start_year')
            _rc_end   = _cached_custom.get('end_year')
            if _is_multi_year:
                custom_map = add_change_layer(
                    custom_map, _rc_start, _rc_end,
                    _cached_custom['change_png'],
                    _cached_custom['change_bounds'],
                )
                for _yr, _yd in _cached_custom.get('by_year', {}).items():
                    custom_map = add_water_mask_layer(
                        custom_map, _yr, _yd['overlay_png'], _yd['overlay_bounds'],
                        layer_name=f'Water Mask {_yr}',
                    )
            else:
                _yr_list = list(_cached_custom.get('by_year', {}).values())
                if _yr_list:
                    custom_map = add_water_mask_layer(
                        custom_map, _rc_end,
                        _yr_list[0]['overlay_png'], _yr_list[0]['overlay_bounds'],
                        layer_name=f'Water Mask {_rc_end}',
                    )

        # Re-centre viewport on the AOI so overlays are visible
        _aoi_viewport_bounds = None
        if _cached_custom and _cached_custom.get('by_year'):
            _aoi_viewport_bounds = next(iter(_cached_custom['by_year'].values())).get('overlay_bounds')
        elif st.session_state.get('custom_aoi_geojson'):
            try:
                _g = st.session_state['custom_aoi_geojson'].get('geometry', st.session_state['custom_aoi_geojson'])
                _c = _g['coordinates'][0]
                _aoi_viewport_bounds = [[min(p[1] for p in _c), min(p[0] for p in _c)],
                                        [max(p[1] for p in _c), max(p[0] for p in _c)]]
            except Exception:
                pass
        if _aoi_viewport_bounds:
            custom_map.fit_bounds(_aoi_viewport_bounds)

        custom_map = finalise_map(custom_map)
        _map_draw_output = st_folium(
            custom_map, height=460, width=None,
            returned_objects=['last_active_drawing', 'all_drawings'],
            key='custom_aoi_map',
        )

        # Capture drawn shape — check last_active_drawing first, fall back to all_drawings
        if _map_draw_output:
            _new_shape = _map_draw_output.get('last_active_drawing')
            if _new_shape is None:
                _all = _map_draw_output.get('all_drawings') or []
                _new_shape = _all[-1] if _all else None
            if _new_shape:
                st.session_state.custom_aoi_geojson = _new_shape

        # Legend below map
        if _cached_custom:
            if _is_multi_year:
                _ls, _le = _cached_custom['start_year'], _cached_custom['end_year']
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:16px;padding:8px 4px 2px;font-size:13px;color:#ccc;">
                  <strong style="color:white;">Change {_ls} &rarr; {_le}</strong>
                  <span><span style="background:#00b4d8;display:inline-block;width:14px;height:10px;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Stable Water</span>
                  <span><span style="background:#2dc653;display:inline-block;width:14px;height:10px;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Water Gain</span>
                  <span><span style="background:#ef233c;display:inline-block;width:14px;height:10px;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Water Loss</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="display:flex;align-items:center;gap:16px;padding:8px 4px 2px;font-size:13px;color:#ccc;">
                  <strong style="color:white;">Water Mask — Custom AOI</strong>
                  <span><span style="background:#0077b6;display:inline-block;width:14px;height:10px;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Water</span>
                </div>""", unsafe_allow_html=True)

    with aoi_controls_col:
        section('Analysis Settings')

        _drawn_aoi = st.session_state.get('custom_aoi_geojson')

        # Also check if the map returned a new drawing this render cycle
        # (handles versions where session_state isn't set yet)
        if _map_draw_output:
            _live = _map_draw_output.get('last_active_drawing')
            if _live is None:
                _all_live = _map_draw_output.get('all_drawings') or []
                _live = _all_live[-1] if _all_live else None
            if _live:
                _drawn_aoi = _live

        _step1_ok = _drawn_aoi is not None

        if _step1_ok:
            st.success('✅ Step 1 — Area selected on map')
        else:
            st.info('⬜ Step 1 — Draw a rectangle on the map first\n\nClick the ▭ tool (top-left of map), drag to draw, then release.')

        _yr_col_a, _yr_col_b = st.columns(2)
        with _yr_col_a:
            _custom_start_year = st.number_input(
                'Start Year', min_value=2000, max_value=2030,
                value=2022, step=1, key='custom_start_year',
            )
        with _yr_col_b:
            _custom_end_year = st.number_input(
                'End Year', min_value=2000, max_value=2030,
                value=2024, step=1, key='custom_end_year',
            )

        _years_valid = _custom_start_year <= _custom_end_year
        if not _years_valid:
            st.error('Start year must be ≤ end year.')
        else:
            _n_years = int(_custom_end_year) - int(_custom_start_year) + 1
            st.caption(f'{"✅" if _years_valid else "⬜"} Step 2 — {_n_years} year(s) selected: {int(_custom_start_year)}–{int(_custom_end_year)}')

        _gee_project_input = st.text_input(
            'GEE Project ID',
            value=config.GEE_PROJECT,
            placeholder='e.g. my-gee-project-123',
            help='Your Google Earth Engine project ID.',
            key='gee_project_input',
        )
        _project_ok = bool(_gee_project_input.strip())
        st.caption(f'{"✅" if _project_ok else "⬜"} Step 3 — GEE Project ID entered')

        if _drawn_aoi:
            try:
                _aoi_km2 = gee_loader.check_aoi_size_km2(_drawn_aoi)
                st.caption(f'Selected area: ~{_aoi_km2:.0f} km²')
                if _aoi_km2 > config.AOI_MAX_KM2:
                    st.warning(f'Large area (~{_aoi_km2:.0f} km²). Downloads may take several minutes.')
            except Exception:
                pass

        _analyze_disabled = not (_step1_ok and _years_valid and _project_ok)

        if st.button('🔍 Analyze', disabled=_analyze_disabled,
                     use_container_width=True, key='analyze_button',
                     type='primary'):
            _analysis_error = None
            try:
                with st.spinner('Connecting to Google Earth Engine...'):
                    gee_loader.init_gee(_gee_project_input.strip())

                _tmp_dir    = tempfile.mkdtemp(prefix='rfwater_gee_')
                _unet_model = inference.load_unet_model(config.UNET_MODEL_PATH)
                _all_yr_res = {}
                _last_crs = _last_tfm = None
                _years_to_run = list(range(int(_custom_start_year), int(_custom_end_year) + 1))

                for _idx, _yr in enumerate(_years_to_run):
                    with st.spinner(f'Downloading data — {_yr} ({_idx+1}/{len(_years_to_run)})...'):
                        _s2_p, _s1a_p, _s1d_p = gee_loader.download_sentinel_composite(
                            _drawn_aoi, _yr, _tmp_dir
                        )
                    with st.spinner(f'Running model — {_yr}...'):
                        _s2f, _s2u, _s1a, _s1d, _crs, _tfm = \
                            inference.load_rasters_from_paths(_s2_p, _s1a_p, _s1d_p)
                        _wmask, _ = inference.run_unet_inference(_unet_model, _s2u, _s1a, _s1d)
                        _stats    = inference.compute_quality_stats(_s2f, _wmask)
                        _ov_png, _ov_bnd = inference.mask_array_to_overlay_png(_wmask, _crs, _tfm)
                        _all_yr_res[_yr] = {
                            'water_mask': _wmask, 'stats': _stats,
                            'overlay_png': _ov_png, 'overlay_bounds': _ov_bnd,
                        }
                        _last_crs, _last_tfm = _crs, _tfm

                _chg_png = _chg_bnd = None
                if len(_years_to_run) > 1:
                    _chg_png, _chg_bnd = inference.build_change_overlay_png(
                        _all_yr_res[int(_custom_start_year)]['water_mask'],
                        _all_yr_res[int(_custom_end_year)]['water_mask'],
                        _last_crs, _last_tfm,
                    )

                st.session_state.custom_results = {
                    'by_year': {
                        _yr: {k: v for k, v in _r.items() if k != 'water_mask'}
                        for _yr, _r in _all_yr_res.items()
                    },
                    'change_png': _chg_png, 'change_bounds': _chg_bnd,
                    'start_year': int(_custom_start_year),
                    'end_year'  : int(_custom_end_year),
                }
                st.rerun()

            except RuntimeError as _e:
                _analysis_error = str(_e)
            except Exception as _e:
                _analysis_error = f'Analysis failed: {_e}'

            if _analysis_error:
                st.error(_analysis_error)

        # ── Results ───────────────────────────────────────────────────────────
        if _cached_custom:
            _rc_start  = _cached_custom['start_year']
            _rc_end    = _cached_custom['end_year']
            _by_yr     = _cached_custom.get('by_year', {})

            st.markdown('---')

            if len(_by_yr) > 1:
                import pandas as _pd
                section(f'Water Area {_rc_start}–{_rc_end}')
                _tbl = [
                    {'Year': yr, 'Water Area (km²)': f"{_by_yr[yr]['stats'].get('water_area_km2', 0):.2f}"}
                    for yr in sorted(_by_yr.keys())
                ]
                st.dataframe(_pd.DataFrame(_tbl).set_index('Year'), use_container_width=True)

                _a0 = _by_yr[_rc_start]['stats'].get('water_area_km2', 0)
                _a1 = _by_yr[_rc_end]['stats'].get('water_area_km2', 0)
                _dk = _a1 - _a0
                _dp = (_dk / _a0 * 100) if _a0 > 0 else 0
                _cc = '#2dc653' if _dk >= 0 else '#ef233c'
                _cs = '+' if _dk >= 0 else ''
                st.markdown(
                    f'<div style="background:#1a2a3a;border-left:3px solid {_cc};'
                    f'padding:10px 14px;border-radius:6px;font-size:13px;color:#ccc;margin-bottom:8px;">'
                    f'Change {_rc_start}→{_rc_end}: '
                    f'<strong style="color:{_cc};">{_cs}{_dk:.2f} km² ({_cs}{_dp:.1f}%)</strong></div>',
                    unsafe_allow_html=True,
                )

            section(f'Water Health — {_rc_end}')
            _es = _by_yr.get(_rc_end, {}).get('stats', {})
            metric_card('Water Area', _es.get('water_area_km2'), unit=' km²',
                        description='Total surface area covered by water')

            for _qk, (_ql, _qd) in {
                'ndti'    : ('Water Muddiness',     'How cloudy or murky the water looks'),
                'ndci'    : ('Algae Level',          'Amount of algae growing in the water'),
                'clarity' : ('Water Clearness',      'How clear and see-through the water is'),
                'sediment': ('Soil & Sand in Water', 'How much dirt is floating in the water'),
            }.items():
                _qv = _es.get(_qk)
                metric_card(_ql, _qv, description=_qd, status=quality_status(_qk, _qv))

            if st.button('🗑️ Clear Results', key='clear_custom_results', use_container_width=True):
                st.session_state.pop('custom_results', None)
                st.session_state.pop('custom_aoi_geojson', None)
                st.rerun()

else:
    # ── STANDARD WATER MASK / CHANGE DETECTION LAYOUT ─────────────────────────
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

        if mode == 'Water Mask':
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:20px;padding:8px 4px 2px;font-size:13px;color:#ccc;">
              <strong style="color:white;">Water Mask &mdash; {selected_year}</strong>
              <span><span style="background:#0077b6;display:inline-block;width:16px;height:12px;
                           border-radius:3px;vertical-align:middle;margin-right:5px;"></span>Water</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:20px;padding:8px 4px 2px;font-size:13px;color:#ccc;">
              <strong style="color:white;">Water and Change {year_a} &rarr; {year_b}</strong>
              <span><span style="background:#00b4d8;display:inline-block;width:16px;height:12px;
                           border-radius:3px;vertical-align:middle;margin-right:5px;"></span>Stable Water</span>
              <span><span style="background:#2dc653;display:inline-block;width:16px;height:12px;
                           border-radius:3px;vertical-align:middle;margin-right:5px;"></span>Water Gain</span>
              <span><span style="background:#ef233c;display:inline-block;width:16px;height:12px;
                           border-radius:3px;vertical-align:middle;margin-right:5px;"></span>Water Loss</span>
            </div>""", unsafe_allow_html=True)


    with stats_col:
        section(f'Statistics — {selected_year}')
        stats = compute_stats_for_year(df, selected_year)

        water_area = stats.get('water_area_km2')
        area_pct   = stats.get('area_change_pct')
        if water_area is not None:
            if area_pct is not None and area_pct == area_pct:
                direction_word = 'more' if area_pct >= 0 else 'less'
                pct_abs        = abs(area_pct)
                summary_text   = (f'The water body covered <strong>{water_area:.2f} km²</strong> '
                                  f'in {selected_year}, which is <strong>{pct_abs:.1f}% '
                                  f'{direction_word}</strong> than the previous year.')
            else:
                summary_text = (f'The water body covered <strong>{water_area:.2f} km²</strong> '
                                f'in {selected_year}.')
            st.markdown(f'<div style="background:#1a2a3a;border-left:3px solid #0077b6;'
                        f'padding:10px 14px;border-radius:6px;font-size:13px;color:#ccc;'
                        f'margin-bottom:12px;">{summary_text}</div>', unsafe_allow_html=True)

        metric_card('Water Area',
                    stats.get('water_area_km2'),
                    delta=stats.get('area_change_km2'),
                    unit=' km²',
                    description='Total surface area covered by water')

        pct     = stats.get('area_change_pct')
        pct_str = f'{pct:+.1f}%' if pct is not None and pct == pct else 'N/A'
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Change vs Previous Year</div>
          <div style="font-size:11px;color:#888;margin-bottom:6px;">Did the water body grow or shrink?</div>
          <div class="metric-value" style="color: {'#2dc653' if (pct or 0) >= 0 else '#ef233c'}">
            {pct_str}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        section('Water Health')

        quality_labels = {
            'ndti'    : ('Water Muddiness',      'How cloudy or murky the water looks'),
            'ndci'    : ('Algae Level',           'Amount of algae growing in the water'),
            'clarity' : ('Water Clearness',       'How clear and see-through the water is'),
            'sediment': ('Soil & Sand in Water',  'How much dirt is floating in the water'),
        }
        prev_stats = compute_stats_for_year(df, selected_year - 1) if (selected_year - 1) in df.index else {}

        for col, (plain_label, description_text) in quality_labels.items():
            val      = stats.get(col)
            prev_val = prev_stats.get(col)
            delta    = (val - prev_val) if (val is not None and prev_val is not None) else None
            metric_card(plain_label, val, delta=delta,
                        description=description_text, status=quality_status(col, val))


# ── ROW 2: CHART TABS (hidden in Custom AOI mode) ────────────────────────────
if mode != 'Custom AOI':
    st.markdown('---')
    tab_area, tab_quality, tab_change = st.tabs([
        '📈 Water Area', '🔬 Water Health', '🗺️ Change Summary'
    ])

    with tab_area:
        col1, col2 = st.columns([6, 4], gap='large')
        with col1:
            st.plotly_chart(area_timeseries_chart(df, year_range), use_container_width=True)
        with col2:
            st.plotly_chart(yoy_change_chart(df, year_range), use_container_width=True)

        st.markdown('---')
        section('Data Table')
        display_cols = ['water_area_km2', 'area_change_km2', 'area_change_pct']
        display_df   = df.loc[year_range[0]:year_range[1], display_cols].copy()
        display_df.columns = ['Area (km²)', 'Change (km²)', 'Change (%)']
        st.dataframe(
            display_df.style.format({
                'Area (km²)'  : '{:.3f}',
                'Change (km²)': '{:+.3f}',
                'Change (%)'  : '{:+.1f}',
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
            st.markdown('---')
            section('Quality Data Table')
            quality_df = df.loc[year_range[0]:year_range[1],
                                [c for c in selected_quality if c in df.columns]]
            quality_df.columns = [quality_options.get(c, c) for c in quality_df.columns]
            st.dataframe(quality_df.style.format('{:.4f}'), use_container_width=True)

    with tab_change:
        st.plotly_chart(change_summary_chart(df, year_range), use_container_width=True)

        st.markdown('---')
        section('Overall Change Summary')
        first_y = year_range[0]
        last_y  = year_range[1]
        if first_y in df.index and last_y in df.index:
            area_first   = df.loc[first_y, 'water_area_km2']
            area_last    = df.loc[last_y,  'water_area_km2']
            total_change = area_last - area_first
            total_pct    = (total_change / area_first * 100) if area_first > 0 else 0

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card(f'Area in {first_y}', area_first, unit=' km²')
            with c2:
                metric_card(f'Area in {last_y}', area_last, unit=' km²')
            with c3:
                metric_card('Total Change', total_change, unit=' km²')
                sign  = '+' if total_pct >= 0 else ''
                color = '#2dc653' if total_pct >= 0 else '#ef233c'
                st.markdown(
                    f'<div style="color:{color};font-size:18px;font-weight:700;">'
                    f'{sign}{total_pct:.1f}% overall</div>',
                    unsafe_allow_html=True,
                )
