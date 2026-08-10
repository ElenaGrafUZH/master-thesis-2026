import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_descriptive_stats(
        df,
        col_df,
        df_descriptive_stats,
        col_mean,
        col_median,
        n_patients,
        feature,
        title,
        colors,
        tickformat = None,
        bins = None,
        bins_stats = None,
        ):
    
    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{"colspan": 2}, None],   # Row 1: spans both columns
            [{"colspan": 2}, None],   # Row 2: spans both columns
            [{"colspan": 2}, None],   # Row 3: spans both columns
            [{}, {}]                  # Row 4: two separate columns
        ],
        subplot_titles=(
            f"{feature} per Study ID",
            f"",
            f"Distribution of Daily {feature}",
            f"Distribution of Mean {feature}, n = {n_patients}",
            f"Distribution of Median {feature}, n = {n_patients}",
        )
    )
    study_ids = sorted(df['study_id'].unique())
    mid = len(study_ids) // 2
    df1 = df[df['study_id'].isin(study_ids[:mid])].copy()
    df2 = df[df['study_id'].isin(study_ids[mid:])].copy()

    desc1 = df_descriptive_stats[df_descriptive_stats['study_id'].isin(study_ids[:mid])].copy()
    desc2 = df_descriptive_stats[df_descriptive_stats['study_id'].isin(study_ids[mid:])].copy()
    # --- Row 1: Box plot + mean scatter ---
    for study in df1['study_id'].unique():
        subset = df1[df1['study_id'] == study]
        fig.add_trace(
            go.Box(
                y=subset[col_df],
                x=subset['study_id'],
                name=study,
                boxpoints='all',
                marker_color=colors.get("blue1"),
                showlegend=False
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=desc1['study_id'].astype(str),
            y=desc1[col_mean],
            mode='markers',
            name=f'Mean {feature}',
            marker=dict(color=colors.get("red1")),
            
        ),
        row=1, col=1
    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
 
    ))

    # --- Row 2: Box plot + mean scatter ---
    for study in df2['study_id'].unique():
        subset = df2[df2['study_id'] == study]
        fig.add_trace(
            go.Box(
                y=subset[col_df],
                x=subset['study_id'],
                name=study,
                boxpoints='all',
                marker_color=colors.get("blue1"),
                showlegend=False
            ),
            row=2, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=desc2['study_id'].astype(str),
            y=desc2[col_mean],
            mode='markers',
            name=f'Mean {feature}',
            marker=dict(color=colors.get("red1")),
            showlegend=False
            
        ),
        row=2, col=1
    )

    # --- Row 3: Histogram of daily sleep duration ---
    bw = bins if bins is not None else dict(size=auto_bin_width(df[col_df]))
    bw_stats = bins_stats if bins_stats is not None else dict(size=auto_bin_width(df_descriptive_stats[col_mean]))
    fig.add_trace(
        go.Histogram(
            x=df[col_df],
            name=f'Daily {feature}',
            xbins=bw,
            marker_line_width=1,
            marker_line_color='black',
            marker_color=colors.get("blue1"),
            showlegend=False
        ),
        row=3, col=1
    )

    # --- Row 4: Mean sleep duration histogram ---
    fig.add_trace(
        go.Histogram(
            x=df_descriptive_stats[col_mean],
            name=f'Mean {feature}',
            xbins=bw_stats,
            marker_line_width=1,
            marker_line_color='black',
            marker_color=colors.get("blue1"),
            showlegend=False
        ),
        row=4, col=1
    )

    # --- Row 4: Median sleep duration histogram ---
    fig.add_trace(
        go.Histogram(
            x=df_descriptive_stats[col_median],
            name=f'Median {feature}',
            xbins=bw_stats,
            marker_line_width=1,
            marker_line_color='black',
            marker_color=colors.get("blue1"),
            showlegend=False
        ),
        row=4, col=2
    )

    # --- Axis labels ---
    fig.update_xaxes(title_text='Study ID', row=1, col=1, showgrid=False, showline=True, linecolor="black")
    fig.update_yaxes(title_text=f'{title}', tickformat=f"{tickformat}", row=1, col=1, showgrid=False, showline=True, linecolor="black")

    fig.update_xaxes(title_text='Study ID', row=2, col=1, showgrid=False, showline=True, linecolor="black")
    fig.update_yaxes(title_text=f'{title}', tickformat=f"{tickformat}", row=2, col=1, showgrid=False, showline=True, linecolor="black")

    fig.update_xaxes(title_text=f'{title}', tickformat=f"{tickformat}", row=3, col=1, showgrid=False, showline=True, linecolor="black")
    fig.update_yaxes(title_text='Count', row=3, col=1, showgrid=True, gridcolor="lightgray", showline=True, linecolor="black")

    fig.update_xaxes(title_text=f'Mean {title}', tickformat=f"{tickformat}", row=4, col=1, showgrid=False, showline=True, linecolor="black")
    fig.update_yaxes(title_text='Count', row=4, col=1, showgrid=True, gridcolor="lightgray", showline=True, linecolor="black")

    fig.update_xaxes(title_text=f'Median {title}', tickformat=f"{tickformat}", row=4, col=2, showgrid=False, showline=True, linecolor="black")
    fig.update_yaxes(title_text='Count', row=4, col=2, showgrid=True, gridcolor="lightgray", showline=True, linecolor="black")
    fig.update_xaxes(tickformat=tickformat)


    # --- Layout ---
    fig.update_layout(
        title_text=f"{feature} Summary",
        height=1600,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig

def auto_bin_width(series):
    q75, q25 = np.percentile(series, [75 ,25])
    iqr = q75 - q25
    n = len(series)
    bw = 2 * iqr / (n ** (1/3))
    return round(bw, 2)

