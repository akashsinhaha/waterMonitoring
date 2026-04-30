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
        # Custom AOI sidebar controls
        selected_year = available_years[-1]   # unused default

        _c_cached = st.session_state.get('custom_results', {})
        _c_yrs    = sorted(_c_cached.get('by_year', {}).keys()) if _c_cached else []

        if len(_c_yrs) > 1:
            st.markdown('---')
            section('Change Detection Years')
            _col_ca, _col_cb = st.columns(2)
            with _col_ca:
                st.caption('From')
                custom_cd_year_a = st.selectbox(
                    'From', _c_yrs[:-1], index=0,
                    key='custom_cd_year_a', label_visibility='collapsed',
                )
            with _col_cb:
                st.caption('To')
                custom_cd_year_b = st.selectbox(
                    'To', _c_yrs[1:], index=len(_c_yrs) - 2,
                    key='custom_cd_year_b', label_visibility='collapsed',
                )
            st.markdown('---')
            section('Chart Year Range')
            year_range = st.slider(
                'Year range',
                min_value=_c_yrs[0], max_value=_c_yrs[-1],
                value=(_c_yrs[0], _c_yrs[-1]),
                label_visibility='collapsed', key='custom_year_range',
            )
        elif _c_yrs:
            custom_cd_year_a = _c_yrs[0]
            custom_cd_year_b = _c_yrs[0]
            year_range       = (_c_yrs[0], _c_yrs[0])
        else:
            custom_cd_year_a = None
            custom_cd_year_b = None
            year_range       = (available_years[0], available_years[-1])

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
            if st.checkbox(label, value=(col in ['ndti', 'ndci', 'clarity']),
                           key=f'cq_{col}')
        ]

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
    _c_by_yr       = _cached_custom.get('by_year', {})

    # Use sidebar-selected view years if available, else fall back to analysis range
    _view_year_a = custom_cd_year_a if (custom_cd_year_a and custom_cd_year_a in _c_by_yr) else _cached_custom.get('start_year')
    _view_year_b = custom_cd_year_b if (custom_cd_year_b and custom_cd_year_b in _c_by_yr) else _cached_custom.get('end_year')
    _is_multi_year = (
        bool(_cached_custom)
        and _view_year_a is not None
        and _view_year_b is not None
        and _view_year_a != _view_year_b
    )

    aoi_map_col, aoi_controls_col = st.columns([6, 4], gap='large')

    with aoi_map_col:
        section('Draw Your Area of Interest')
        st.caption('Step 1 — Click the ▭ rectangle tool on the map and draw a box over your area.')

        custom_map = create_base_map(map_bounds)
        custom_map = add_draw_control(custom_map)

        if _cached_custom and _c_by_yr:
            if _is_multi_year:
                _ma         = _c_by_yr[_view_year_a].get('water_mask')
                _mb         = _c_by_yr[_view_year_b].get('water_mask')
                _stored_crs = _cached_custom.get('raster_crs')
                _stored_tfm = _cached_custom.get('raster_transform')
                if _ma is not None and _mb is not None and _stored_crs and _stored_tfm:
                    _dyn_chg_png, _dyn_chg_bnd = inference.build_change_overlay_png(
                        _ma, _mb, _stored_crs, _stored_tfm
                    )
                    custom_map = add_change_layer(
                        custom_map, _view_year_a, _view_year_b,
                        _dyn_chg_png, _dyn_chg_bnd,
                    )
            else:
                _single_data = _c_by_yr.get(_view_year_b) or next(iter(_c_by_yr.values()))
                custom_map = add_water_mask_layer(
                    custom_map, _view_year_b,
                    _single_data['overlay_png'], _single_data['overlay_bounds'],
                    layer_name=f'Water Mask {_view_year_b}',
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
        if _cached_custom and _c_by_yr:
            if _is_multi_year:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:16px;padding:8px 4px 2px;font-size:13px;color:#ccc;">
                  <strong style="color:white;">Change {_view_year_a} &rarr; {_view_year_b}</strong>
                  <span><span style="background:#00b4d8;display:inline-block;width:14px;height:10px;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Stable Water</span>
                  <span><span style="background:#2dc653;display:inline-block;width:14px;height:10px;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Water Gain</span>
                  <span><span style="background:#ef233c;display:inline-block;width:14px;height:10px;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Water Loss</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:16px;padding:8px 4px 2px;font-size:13px;color:#ccc;">
                  <strong style="color:white;">Water Mask — {_view_year_b}</strong>
                  <span><span style="background:#0077b6;display:inline-block;width:14px;height:10px;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Water</span>
                </div>""", unsafe_allow_html=True)

    with aoi_controls_col:
        # ── Controls ──────────────────────────────────────────────────────────
        section('Analysis Settings')

        _drawn_aoi = st.session_state.get('custom_aoi_geojson')
        if _map_draw_output:
            _live = _map_draw_output.get('last_active_drawing')
            if _live is None:
                _all_live = _map_draw_output.get('all_drawings') or []
                _live = _all_live[-1] if _all_live else None
            if _live:
                _drawn_aoi = _live

        _step1_ok = _drawn_aoi is not None
        if _step1_ok:
            st.success('✅ Area selected on map')
        else:
            st.info('⬜ Draw a rectangle on the map first\n\nClick the ▭ tool (top-left of map), drag to draw, then release.')

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

        _gee_project_input = st.text_input(
            'GEE Project ID',
            value=config.GEE_PROJECT,
            placeholder='e.g. my-gee-project-123',
            help='Your Google Earth Engine project ID.',
            key='gee_project_input',
        )
        _project_ok = bool(_gee_project_input.strip())

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
                     use_container_width=True, key='analyze_button', type='primary'):
            _analysis_error = None
            try:
                with st.spinner('Connecting to Google Earth Engine...'):
                    gee_loader.init_gee(_gee_project_input.strip())

                _tmp_dir      = tempfile.mkdtemp(prefix='rfwater_gee_')
                _unet_model   = inference.load_unet_model(config.UNET_MODEL_PATH)
                _all_yr_res   = {}
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
                        _yr: _r   # keep water_mask for on-the-fly change overlays
                        for _yr, _r in _all_yr_res.items()
                    },
                    'change_png'      : _chg_png,
                    'change_bounds'   : _chg_bnd,
                    'start_year'      : int(_custom_start_year),
                    'end_year'        : int(_custom_end_year),
                    'raster_crs'      : _last_crs,
                    'raster_transform': _last_tfm,
                }
                st.rerun()

            except RuntimeError as _e:
                _analysis_error = str(_e)
            except Exception as _e:
                _analysis_error = f'Analysis failed: {_e}'

            if _analysis_error:
                st.error(_analysis_error)

        # ── Statistics (identical layout to main stats column) ────────────────
        if _cached_custom and _c_by_yr:
            _by_yr = _c_by_yr
            _es    = _by_yr.get(_view_year_b, {}).get('stats', {})
            _ss    = _by_yr.get(_view_year_a, {}).get('stats', {}) if _is_multi_year else {}

            _cst_water_area  = _es.get('water_area_km2')
            _cst_area_delta  = (_cst_water_area - _ss.get('water_area_km2', 0)) if (_is_multi_year and _cst_water_area is not None) else None
            _cst_area_pct    = ((_cst_area_delta / _ss['water_area_km2'] * 100)
                                if (_cst_area_delta is not None and _ss.get('water_area_km2', 0) > 0)
                                else None)

            st.markdown('---')
            section(f'Statistics — {_view_year_b}')

            # Summary sentence
            if _cst_water_area is not None:
                if _cst_area_pct is not None:
                    _dir  = 'more' if _cst_area_pct >= 0 else 'less'
                    _ref  = f'{_view_year_a}' if _is_multi_year else 'the previous year'
                    _summ = (f'The water body covered <strong>{_cst_water_area:.2f} km²</strong> '
                             f'in {_view_year_b}, which is <strong>{abs(_cst_area_pct):.1f}% '
                             f'{_dir}</strong> than {_ref}.')
                else:
                    _summ = (f'The water body covered <strong>{_cst_water_area:.2f} km²</strong> '
                             f'in {_view_year_b}.')
                st.markdown(f'<div style="background:#1a2a3a;border-left:3px solid #0077b6;'
                            f'padding:10px 14px;border-radius:6px;font-size:13px;color:#ccc;'
                            f'margin-bottom:12px;">{_summ}</div>', unsafe_allow_html=True)

            metric_card('Water Area', _cst_water_area,
                        delta=_cst_area_delta, unit=' km²',
                        description='Total surface area covered by water')

            _pct_str  = f'{_cst_area_pct:+.1f}%' if _cst_area_pct is not None else 'N/A'
            _pct_col  = '#2dc653' if (_cst_area_pct or 0) >= 0 else '#ef233c'
            _pct_lbl  = f'Change vs {_view_year_a}' if _is_multi_year else 'Change vs Previous Year'
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{_pct_lbl}</div>
              <div style="font-size:11px;color:#888;margin-bottom:6px;">Did the water body grow or shrink?</div>
              <div class="metric-value" style="color:{_pct_col}">{_pct_str}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)
            section('Water Health')

            for _qk, (_ql, _qd) in {
                'ndti'    : ('Water Muddiness',     'How cloudy or murky the water looks'),
                'ndci'    : ('Algae Level',          'Amount of algae growing in the water'),
                'clarity' : ('Water Clearness',      'How clear and see-through the water is'),
                'sediment': ('Soil & Sand in Water', 'How much dirt is floating in the water'),
            }.items():
                _qv = _es.get(_qk)
                _sv = _ss.get(_qk) if _ss else None
                _qd_val = (_qv - _sv) if (_qv is not None and _sv is not None) else None
                metric_card(_ql, _qv, delta=_qd_val,
                            description=_qd, status=quality_status(_qk, _qv))

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


# ── ROW 2: CHART TABS ────────────────────────────────────────────────────────
if mode == 'Custom AOI' and _cached_custom and len(_c_by_yr) >= 1:
    import pandas as _pd
    # Build a DataFrame from custom results so we can reuse the same chart functions
    _cst_rows = {
        _yr: {
            'water_area_km2': _c_by_yr[_yr]['stats'].get('water_area_km2', 0),
            'ndti'          : _c_by_yr[_yr]['stats'].get('ndti'),
            'ndci'          : _c_by_yr[_yr]['stats'].get('ndci'),
            'clarity'       : _c_by_yr[_yr]['stats'].get('clarity'),
            'algae'         : _c_by_yr[_yr]['stats'].get('algae'),
            'sediment'      : _c_by_yr[_yr]['stats'].get('sediment'),
        }
        for _yr in sorted(_c_by_yr.keys())
    }
    _cst_df = _pd.DataFrame.from_dict(_cst_rows, orient='index')
    _cst_df.index.name = 'year'
    _cst_df['area_change_km2'] = _cst_df['water_area_km2'].diff()
    _cst_df['area_change_pct'] = _cst_df['water_area_km2'].pct_change() * 100
    _cst_yr_range = (year_range[0], year_range[1]) if year_range else (sorted(_c_by_yr.keys())[0], sorted(_c_by_yr.keys())[-1])

    st.markdown('---')
    _ctab_area, _ctab_quality, _ctab_change = st.tabs([
        '📈 Water Area', '🔬 Water Health', '🗺️ Change Summary'
    ])

    with _ctab_area:
        _cc1, _cc2 = st.columns([6, 4], gap='large')
        with _cc1:
            st.plotly_chart(area_timeseries_chart(_cst_df, _cst_yr_range), use_container_width=True)
        with _cc2:
            st.plotly_chart(yoy_change_chart(_cst_df, _cst_yr_range), use_container_width=True)
        st.markdown('---')
        section('Data Table')
        _cst_disp = _cst_df.loc[_cst_yr_range[0]:_cst_yr_range[1],
                                 ['water_area_km2', 'area_change_km2', 'area_change_pct']].copy()
        _cst_disp.columns = ['Area (km²)', 'Change (km²)', 'Change (%)']
        st.dataframe(
            _cst_disp.style.format({
                'Area (km²)'  : '{:.3f}',
                'Change (km²)': '{:+.3f}',
                'Change (%)'  : '{:+.1f}',
            }).background_gradient(subset=['Area (km²)'], cmap='Blues'),
            use_container_width=True,
        )

    with _ctab_quality:
        if not selected_quality:
            st.info('Select at least one quality metric from the sidebar.')
        else:
            st.plotly_chart(
                quality_timeseries_chart(_cst_df, _cst_yr_range, selected_quality),
                use_container_width=True,
            )
            st.markdown('---')
            section('Quality Data Table')
            _cst_q_df = _cst_df.loc[_cst_yr_range[0]:_cst_yr_range[1],
                                     [c for c in selected_quality if c in _cst_df.columns]]
            _cst_q_df.columns = [quality_options.get(c, c) for c in _cst_q_df.columns]
            st.dataframe(_cst_q_df.style.format('{:.4f}'), use_container_width=True)

    with _ctab_change:
        st.plotly_chart(change_summary_chart(_cst_df, _cst_yr_range), use_container_width=True)
        st.markdown('---')
        section('Overall Change Summary')
        _cst_yrs_sorted = sorted(_c_by_yr.keys())
        _cst_fy, _cst_ly = _cst_yrs_sorted[0], _cst_yrs_sorted[-1]
        _cst_af = _cst_df.loc[_cst_fy, 'water_area_km2']
        _cst_al = _cst_df.loc[_cst_ly, 'water_area_km2']
        _cst_tc = _cst_al - _cst_af
        _cst_tp = (_cst_tc / _cst_af * 100) if _cst_af > 0 else 0
        _cc1, _cc2, _cc3 = st.columns(3)
        with _cc1:
            metric_card(f'Area in {_cst_fy}', _cst_af, unit=' km²')
        with _cc2:
            metric_card(f'Area in {_cst_ly}', _cst_al, unit=' km²')
        with _cc3:
            metric_card('Total Change', _cst_tc, unit=' km²')
            _tc_sign  = '+' if _cst_tp >= 0 else ''
            _tc_color = '#2dc653' if _cst_tp >= 0 else '#ef233c'
            st.markdown(f'<div style="color:{_tc_color};font-size:18px;font-weight:700;">'
                        f'{_tc_sign}{_cst_tp:.1f}% overall</div>', unsafe_allow_html=True)

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
