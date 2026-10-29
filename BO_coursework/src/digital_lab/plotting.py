"""Shared plotting/formatting helpers, used by the student notebook and the benchmark."""

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


def fmt_mean_std(mean: float | None, std: float | None, precision: int = 4) -> str:
    """Render a (mean, std) pair as a string.

    Parameters
    ----------
    mean : float or None
        Mean value; None renders as "N/A".
    std : float or None
        Standard deviation; None omits the "± std" suffix.
    precision : int, optional
        Number of decimal places, by default 4.

    Returns
    -------
    str
        "mean ± std", "mean" (if std is None), or "N/A" (if mean is None).
    """
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color to a CSS rgba() string.

    Parameters
    ----------
    hex_color : str
        Color as "#rrggbb".
    alpha : float
        Opacity in [0, 1].

    Returns
    -------
    str
        Color as "rgba(r,g,b,a)".
    """
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def add_band_trace(  # pylint: disable=too-many-arguments
    fig: go.Figure,
    x: list,
    mean: list,
    std: list,
    color: str,
    name: str,
    *,
    dash: str = "solid",
    row: int | None = None,
    col: int | None = None,
    showlegend: bool = True,
) -> None:
    """Add a mean line with a shaded +/-1 std band to fig.

    Draws three traces: an invisible upper bound, a lower bound filled up to
    it, then the mean line on top.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to add the traces to, in place.
    x : list
        Shared x-values for all three traces.
    mean : list
        Mean y-values.
    std : list
        Standard deviation at each x, used to compute the band.
    color : str
        Line/fill color, as "#rrggbb".
    name : str
        Legend label for the mean line.
    dash : str, optional
        Plotly dash style for the mean line, by default "solid".
    row, col : int or None, optional
        Subplot to draw in, for figures made with `make_subplots`.
    showlegend : bool, optional
        Whether the mean line gets a legend entry, by default True.

    Returns
    -------
    None
    """
    upper = [m + s for m, s in zip(mean, std)]
    lower = [max(m - s, 0.0) for m, s in zip(mean, std)]  # titers can't be negative
    where = {"row": row, "col": col} if row is not None else {}
    fig.add_trace(
        go.Scatter(x=x, y=upper, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"), **where
    )
    fig.add_trace(go.Scatter(
        x=x, y=lower, mode="lines", line={"width": 0}, fill="tonexty",
        fillcolor=hex_to_rgba(color, 0.15), showlegend=False, hoverinfo="skip",
    ), **where)
    fig.add_trace(go.Scatter(
        x=x, y=mean, mode="lines", name=name, legendgroup=name, showlegend=showlegend,
        line={"color": color, "width": 2, "dash": dash, "shape": "hv"},
        hovertemplate=f"{name}<br>%{{x:,.0f}}: %{{y:,.0f}}<extra></extra>",
    ), **where)


SCALES = ("MTP", "MBR", "Pilot")
X_AXES = {"spent": "Money spent [$]", "n_experiments": "Number of experiments"}


def best_so_far_on_grid(histories: list, scale: str, x_col: str, grid: np.ndarray) -> tuple:
    """Mean and std over seeds of the best titer measured at `scale`, on a common x-grid.

    Each seed's best-so-far is a step function of `x_col`, held at its last value after the
    run ends, and 0 before its first measurement at `scale` (nothing found there yet). Every
    seed is therefore defined everywhere, so the mean of these non-decreasing curves can never
    go down.

    Parameters
    ----------
    histories : list[polars.DataFrame]
        One history per seed, with "reactor", "titer" and `x_col` columns.
    scale : str
        Reactor whose titers count.
    x_col : str
        "spent" or "n_experiments".
    grid : numpy.ndarray
        Increasing x-values to evaluate on.

    Returns
    -------
    mean, std : numpy.ndarray
        Arrays like `grid`; NaN everywhere if no seed ever measured at `scale`.
    """
    curves = []
    for history in histories:
        part = history.filter(pl.col("reactor") == scale).sort(x_col)
        values = np.zeros(len(grid))
        if part.height:
            xs = part[x_col].to_numpy()
            best = np.maximum.accumulate(part["titer"].to_numpy())
            idx = np.searchsorted(xs, grid, side="right") - 1
            values[idx >= 0] = best[idx[idx >= 0]]
        curves.append(values)
    stack = np.array(curves)
    if not curves or not stack.any():
        return np.full(len(grid), np.nan), np.full(len(grid), np.nan)
    return stack.mean(axis=0), stack.std(axis=0)


def convergence_figure(
    runs: dict, x_col: str, colors: dict, dashes: dict | None = None, checkpoints: list | None = None
) -> go.Figure:
    """Best titer measured so far at each scale (MTP, MBR, Pilot side by side) vs `x_col`.

    Parameters
    ----------
    runs : dict[str, list[polars.DataFrame]]
        Algorithm name -> its per-seed histories.
    x_col : str
        "spent" or "n_experiments".
    colors : dict[str, str]
        Algorithm name -> "#rrggbb".
    dashes : dict[str, str] or None, optional
        Algorithm name -> plotly dash style, by default solid for all.
    checkpoints : list[float] or None, optional
        Money spent at which runs are scored; drawn as dotted lines on the Pilot panel.

    Returns
    -------
    plotly.graph_objects.Figure
        One row, three panels, each with its own y-axis (titers differ a lot between scales).
    """
    dashes = dashes or {}
    x_max = max((h[x_col].max() for hs in runs.values() for h in hs if h.height), default=1)
    grid = np.arange(1, x_max + 1) if x_col == "n_experiments" else np.linspace(0, x_max, 400)
    fig = make_subplots(
        rows=1, cols=len(SCALES), subplot_titles=[f"best at {s}" for s in SCALES], horizontal_spacing=0.07
    )
    for col, scale in enumerate(SCALES, start=1):
        for name, histories in runs.items():
            mean, std = best_so_far_on_grid(histories, scale, x_col, grid)
            ok = ~np.isnan(mean)
            if not ok.any():
                continue
            add_band_trace(
                fig, grid[ok].tolist(), mean[ok].tolist(), std[ok].tolist(), colors[name], name,
                dash=dashes.get(name, "solid"), row=1, col=col, showlegend=col == 1,
            )
        fig.update_xaxes(title_text=X_AXES[x_col], row=1, col=col)
    for checkpoint in checkpoints or []:
        fig.add_vline(
            x=checkpoint, line={"color": "#8a8984", "dash": "dot", "width": 1}, row=1, col=len(SCALES),
            annotation_text=f"${checkpoint:,.0f}", annotation_position="top left", annotation_font_size=10,
        )
    fig.update_yaxes(title_text="Best titer so far (mean ± std)", row=1, col=1)
    fig.update_layout(
        title=f"Best titer measured so far at each scale vs {X_AXES[x_col].lower()}",
        template="plotly_white", height=420,
        legend={"orientation": "h", "yref": "container", "y": 0, "yanchor": "bottom", "x": 0.5, "xanchor": "center"},
        margin={"b": 90},
    )
    return fig
