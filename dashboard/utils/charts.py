import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


BLUE   = '#0077b6'
GREEN  = '#2dc653'
RED    = '#ef233c'
ORANGE = '#e07b39'
PURPLE = '#9b5de5'
TEAL   = '#00b4d8'
YELLOW = '#f77f00'

DARK_BG    = '#0e1117'
CARD_BG    = '#1a1a2e'
GRID_COLOR = '#2d2d2d'


def _base_layout(title='', height=400):
    return dict(
        title=dict(text=title, font=dict(size=14, color='white')),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color='white', size=11),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
        margin=dict(l=50, r=20, t=50, b=40),
        height=height,
        hovermode='x unified',
    )


def area_timeseries_chart(df, year_range):
    """Line chart of water area (km²) over selected year range."""
    filtered = df.loc[year_range[0]:year_range[1]]
    years    = filtered.index.tolist()
    areas    = filtered['water_area_km2'].tolist()

    fig = go.Figure()

    # Filled area
    fig.add_trace(go.Scatter(
        x=years, y=areas,
        mode='lines+markers',
        name='Water Area',
        line=dict(color=BLUE, width=2.5),
        marker=dict(size=8, color=BLUE, symbol='circle',
                    line=dict(color='white', width=2)),
        fill='tozeroy',
        fillcolor='rgba(0, 119, 182, 0.15)',
        hovertemplate='%{x}: %{y:.3f} km²<extra></extra>',
    ))

    # Trend line
    if len(years) > 2:
        z      = np.polyfit(years, areas, 1)
        trend  = np.poly1d(z)
        trend_direction = 'Increasing' if z[0] > 0 else 'Decreasing'
        fig.add_trace(go.Scatter(
            x=years, y=trend(years),
            mode='lines', name=f'Trend ({trend_direction})',
            line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dot'),
            hoverinfo='skip',
        ))

    fig.update_layout(
        **_base_layout('Water Area Over Time (km²)', height=380),
        yaxis_title='Area (km²)',
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
    )
    fig.update_xaxes(tickmode='array', tickvals=years, gridcolor=GRID_COLOR, showgrid=True)
    return fig


def yoy_change_chart(df, year_range):
    """Bar chart of year-over-year area change."""
    filtered = df.loc[year_range[0]:year_range[1]].copy()
    filtered = filtered.dropna(subset=['area_change_km2'])
    years    = filtered.index.tolist()
    changes  = filtered['area_change_km2'].tolist()
    colors   = [GREEN if c >= 0 else RED for c in changes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years, y=changes,
        marker_color=colors,
        name='Area Change',
        hovertemplate='%{x}: %{y:+.3f} km²<extra></extra>',
        text=[f'{c:+.2f}' for c in changes],
        textposition='outside',
        textfont=dict(size=9),
    ))
    fig.add_hline(y=0, line_color='white', line_width=0.8)

    fig.update_layout(
        **_base_layout('Year-over-Year Water Area Change (km²)', height=320),
        yaxis_title='Change (km²)',
        showlegend=False,
    )
    fig.update_xaxes(tickmode='array', tickvals=years, gridcolor=GRID_COLOR, showgrid=True)
    return fig


def quality_timeseries_chart(df, year_range, selected_metrics):
    """
    Multi-line chart for selected water quality indices.
    selected_metrics: list of column names to plot.
    """
    metric_config = {
        'ndti'    : ('Turbidity (NDTI)',     ORANGE),
        'ndci'    : ('Chlorophyll-a (NDCI)', GREEN),
        'clarity' : ('Water Clarity',        TEAL),
        'algae'   : ('Algae Index',          PURPLE),
        'sediment': ('Sediment (B4)',         '#ffd166'),
    }

    filtered = df.loc[year_range[0]:year_range[1]]
    years    = filtered.index.tolist()
    fig      = go.Figure()

    for metric in selected_metrics:
        if metric not in filtered.columns:
            continue
        label, color = metric_config.get(metric, (metric, BLUE))
        values = filtered[metric].tolist()

        fig.add_trace(go.Scatter(
            x=years, y=values,
            mode='lines+markers',
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color,
                        line=dict(color='white', width=1.5)),
            hovertemplate=f'{label}: %{{y:.4f}}<extra></extra>',
        ))

        # Trend
        valid = [(y, v) for y, v in zip(years, values) if not np.isnan(v)]
        if len(valid) > 2:
            vx, vy = zip(*valid)
            z      = np.polyfit(vx, vy, 1)
            trend  = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=list(vx), y=list(trend(vx)),
                mode='lines', name=f'{label} trend',
                line=dict(color=color, width=1, dash='dot'),
                opacity=0.4, hoverinfo='skip', showlegend=False,
            ))

    fig.update_layout(
        **_base_layout('Water Quality Indices Over Time', height=400),
        yaxis_title='Index Value',
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10),
                    orientation='h', y=-0.2),
    )
    fig.update_xaxes(tickmode='array', tickvals=years, gridcolor=GRID_COLOR, showgrid=True)
    return fig


def change_summary_chart(df, year_range):
    """Stacked overview: area line + change bars in subplots."""
    filtered  = df.loc[year_range[0]:year_range[1]].copy()
    years     = filtered.index.tolist()
    areas     = filtered['water_area_km2'].tolist()
    changes   = filtered['area_change_km2'].fillna(0).tolist()
    colors    = [GREEN if c >= 0 else RED for c in changes]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.06,
        subplot_titles=('Water Area (km²)', 'Year-over-Year Change (km²)'),
    )

    fig.add_trace(go.Scatter(
        x=years, y=areas, mode='lines+markers',
        line=dict(color=BLUE, width=2.5),
        marker=dict(size=7, color=BLUE, line=dict(color='white', width=1.5)),
        fill='tozeroy', fillcolor='rgba(0,119,182,0.15)',
        name='Water Area',
        hovertemplate='%{x}: %{y:.3f} km²<extra></extra>',
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=years, y=changes,
        marker_color=colors, name='Change',
        hovertemplate='%{x}: %{y:+.3f} km²<extra></extra>',
    ), row=2, col=1)

    fig.add_hline(y=0, line_color='white', line_width=0.8, row=2, col=1)

    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
        font=dict(color='white', size=11),
        height=480, showlegend=False,
        margin=dict(l=50, r=20, t=50, b=40),
        hovermode='x unified',
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, tickmode='array', tickvals=years)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig
