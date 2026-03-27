"""Draft Streamlit dashboard scaffold for deal collateral views."""

from __future__ import annotations

import hashlib
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta


def _det_u01(seed_str: str, i: int) -> float:
    """Deterministic uniform [0, 1); identical across Python/NumPy versions (SHA-256 based)."""
    h = hashlib.sha256(f"{seed_str}|{i}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(2**64)


def _det_normal(seed_str: str, i: int, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Deterministic ~N(mu, sigma^2) via Box–Muller; stateless per index i."""
    u1 = max(1e-12, _det_u01(seed_str, i * 2))
    u2 = _det_u01(seed_str, i * 2 + 1)
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mu + sigma * z


def _plotly_int_list(x) -> list[int]:
    """Plain ints for Plotly x (avoids numpy dtype JSON quirks in some hosts)."""
    arr = np.asarray(x)
    return [int(v) for v in arr.ravel().tolist()]


def _plotly_float_list(y) -> list[float]:
    """Plain floats for Plotly y (preserves NaNs as JSON null-friendly floats)."""
    return np.asarray(y, dtype=float).ravel().tolist()


# Consistent grade colors for stacked bars (explicit map for Snowflake/Plotly stacks).
GRADE_COLOR_MAP = {"A": "#1f77b4", "B": "#aec7e8", "C": "#ff7f0e", "D": "#ffbb78"}
GEO_COLOR_MAP = {"CA": "#1f77b4", "TX": "#aec7e8", "FL": "#d62728", "NY": "#e377c2", "IL": "#17becf"}
# Plotly: positive tickangle = counter-clockwise (anti-clockwise) from the horizontal.
DATE_X_TICKANGLE = 45

_DATA_END_DATE = datetime.today()

st.set_page_config(page_title="Deal Collateral Dashboard (Draft)")
st.title("Deal Collateral Dashboard (Draft)")

st.markdown("""---""")
deal_select_col, _ = st.columns([1, 3])
with deal_select_col:
    selected_deal = st.selectbox("Select a deal", ["Deal 1", "Deal 2", "Deal 3"])

# Mock data generator for demonstration
def get_volume_df(grades, months_back, freq="ME", grade_vol_multiplier=1):
    """
    Monthly *total* origination volume hovers around ~$10M (no artificial linear ramp).
    Split across grades using the same grade-mix proportions as elsewhere.
    """
    dates = pd.DatetimeIndex(pd.date_range(end=_DATA_END_DATE, periods=months_back, freq=freq)).normalize()
    mix_df = get_grade_mix_df(grades, months_back, freq=freq)
    mix_df["date"] = pd.to_datetime(mix_df["date"]).dt.normalize()
    # Mild mean-reverting noise around $10M (not a time trend). Deterministic noise (no np.random).
    base_usd = 10_000_000.0 * float(grade_vol_multiplier)
    ar = 0.35
    noise_sd = 450_000.0 * float(grade_vol_multiplier)
    shocks = np.array(
        [_det_normal("volume:monthly_total", j, 0.0, noise_sd) for j in range(months_back)],
        dtype=float,
    )
    total_monthly = np.zeros(months_back, dtype=float)
    total_monthly[0] = base_usd + shocks[0]
    for i in range(1, months_back):
        total_monthly[i] = base_usd + ar * (total_monthly[i - 1] - base_usd) + shocks[i]
    total_monthly = np.clip(total_monthly, 7_500_000.0 * grade_vol_multiplier, 12_500_000.0 * grade_vol_multiplier)

    # Wide lookup avoids Period/categorical equality quirks in hosted runtimes (empty filters -> wrong stacks).
    mix_wide = mix_df.pivot_table(index="date", columns="grade", values="proportion", aggfunc="first")
    mix_wide = mix_wide.reindex(columns=list(grades)).fillna(0.0)
    mix_wide = mix_wide.div(mix_wide.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.25)

    data: list[dict] = []
    for i, d in enumerate(dates):
        ts = pd.Timestamp(d)
        tot = float(total_monthly[i])
        row = mix_wide.loc[ts]
        for grade in grades:
            prop = float(row[grade])
            vol = int(round(tot * prop))
            data.append({"date": ts, "grade": str(grade), "volume": vol})
    out = pd.DataFrame(data)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["grade"] = out["grade"].astype(str)
    return out.sort_values(["date", "grade"])

def get_coupon_df(grades, months_back, freq="ME"):
    """
    Simulate coupons: ~6% for A, ~7.5% for B, ~9% for C, ~12.5% for D, noise < +/-0.4 over a year.
    Chosen weights in grade mix (see below) will give weighted avg ≈ 8.5%.
    """
    dates = pd.DatetimeIndex(pd.date_range(end=_DATA_END_DATE, periods=months_back, freq=freq)).normalize()
    data = []
    # These values chosen so that with mix weights in get_grade_mix_df, the overall WAC ≈ 8.5%
    base_coupons = {
        'A': 6.0,
        'B': 7.5,
        'C': 9.0,
        'D': 12.5
    }
    # Each grade walks slightly, but limited
    for grade in grades:
        seed = f"coupon:{grade}"
        noise = np.array([_det_normal(seed, j, 0.0, 0.035) for j in range(months_back)], dtype=float)
        walk = np.cumsum(noise)
        walk = np.clip(walk, -0.4, 0.4)
        coupons = (base_coupons[grade] + walk).round(2)
        for d, c in zip(dates, coupons):
            data.append({"date": d, "grade": grade, "coupon": c})
    out = pd.DataFrame(data)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["grade"] = pd.Categorical(out["grade"], categories=grades, ordered=True)
    return out.sort_values(["date", "grade"])

def get_grade_mix_df(grades, months_back, freq="ME"):
    """
    Simulate slow-moving grade mix: each handle starts at base weight and only drifts slightly over time.
    """
    dates = pd.DatetimeIndex(pd.date_range(end=_DATA_END_DATE, periods=months_back, freq=freq)).normalize()
    data = []
    # Chosen so WAC ≈ 8.5%, and to be realistic (A is most common, D rare)
    base_props = {
        'A': 0.35,
        'B': 0.30,
        'C': 0.25,
        'D': 0.10
    }
    # We keep each grade's initial proportion and add small NORMAL noise, then re-normalize to sum 1.
    prop_series = {grade: [] for grade in grades}
    last_props = [base_props[g] for g in grades]
    for i, d in enumerate(dates):
        # Small normal drift (deterministic per month/grade index)
        drift = np.array(
            [_det_normal("grade_mix:global", i * len(grades) + idx, 0.0, 0.01) for idx in range(len(grades))],
            dtype=float,
        )
        new_props = np.array(last_props) + drift
        new_props = np.clip(new_props, 0.04, 0.6)  # not less than 4%, not more than 60%
        new_props = new_props / new_props.sum()  # re-normalize
        for idx, grade in enumerate(grades):
            prop_series[grade].append(new_props[idx])
        last_props = new_props
    # Reshape to DataFrame
    for i, d in enumerate(dates):
        for grade in grades:
            data.append({"date": d, "grade": grade, "proportion": prop_series[grade][i]})
    out = pd.DataFrame(data)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["grade"] = out["grade"].astype(str)
    return out.sort_values(["date", "grade"])

grades = ['A', 'B', 'C', 'D']

def render_volume_tab(df: pd.DataFrame, y_label="Volume ($)", key_suffix: str = ""):
    wide = df.pivot_table(index="date", columns="grade", values="volume", aggfunc="sum")
    wide = wide.reindex(columns=grades).fillna(0.0)
    x_dates = pd.to_datetime(wide.index).tolist()
    fig = go.Figure()
    for g in grades:
        fig.add_trace(
            go.Bar(
                name=g,
                x=x_dates,
                y=_plotly_float_list(wide[g].values),
                marker_color=GRADE_COLOR_MAP.get(g),
            )
        )
    fig.update_layout(
        title="Volume by Origination Vintage (Stacked by Grade)",
        barmode="stack",
        xaxis_title="Origination Month",
        yaxis_title=y_label,
        xaxis_tickformat="%b\n%Y",
        xaxis_tickangle=DATE_X_TICKANGLE,
        yaxis=dict(tickprefix="$", tickformat=",.0f", title="Volume ($)"),
        legend_title="Loan Grade",
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig, width='stretch', key=f"volume_{key_suffix}")

def render_coupon_tab(df: pd.DataFrame, key_suffix: str = ""):
    # Merge grade-level coupons into a wide form for easier WAC calc
    pivot = df.pivot(index="date", columns="grade", values="coupon")
    dates = sorted(df['date'].unique())

    # For weighted average, simulate volumes per grade and date using the grade mix from get_grade_mix_df
    # So WAC displayed will match what would result from the coupon and grade mix dummies
    mix_df = get_grade_mix_df(grades, len(dates), freq='ME')
    # Assign volume per period
    total_volume = 50000  # arbitrary
    mock_vols = {g: np.zeros(len(dates)) for g in grades}
    for i, d in enumerate(dates):
        month_period = pd.Timestamp(d).to_period("M")
        for grade in grades:
            # Match by month (not exact timestamp) so independently generated demo dates still align.
            filt = (
                (mix_df['date'].dt.to_period("M") == month_period)
                & (mix_df['grade'] == grade)
            )
            prop_vals = mix_df.loc[filt, 'proportion'].values
            prop = prop_vals[0] if prop_vals.size > 0 else 0.0
            mock_vols[grade][i] = total_volume * prop

    wa_coupons = np.zeros(len(dates))
    for i, d in enumerate(dates):
        vols = np.array([mock_vols[g][i] for g in grades])
        coups = np.array([pivot.loc[d, g] if not pd.isnull(pivot.loc[d, g]) else 0 for g in grades])
        total_vol = vols.sum()
        if total_vol > 0:
            wa_coupons[i] = (vols * coups).sum() / total_vol
        else:
            wa_coupons[i] = np.nan

    # Plot: one line per grade, AND black dotted line for weighted average
    fig = go.Figure()
    for grade in grades:
        subset = df[df['grade'] == grade]
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(subset["date"]).tolist(),
            y=_plotly_float_list(subset["coupon"].values),
            mode='lines+markers',
            name=f"Grade {grade}",
            marker=dict(size=4),
        ))
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(dates).tolist(),
        y=_plotly_float_list(wa_coupons),
        mode='lines',
        name="Weighted Avg Coupon",
        line=dict(color='black', dash='dot', width=2),
        hovertemplate="Weighted Avg Coupon: %{y:.2f}<br>Date: %{x|%b %Y}<extra></extra>",
    ))

    fig.update_layout(
        title="Weighted Average Coupon by Grade (%)",
        xaxis_title="Origination Month",
        yaxis_title="Coupon (%)",
        xaxis_tickformat="%b\n%Y",
        xaxis_tickangle=DATE_X_TICKANGLE,
        legend_title="Loan Grade",
    )
    st.plotly_chart(fig, width='stretch', key=f"coupon_{key_suffix}")

def render_grade_mix_tab(df: pd.DataFrame, key_suffix: str = ""):
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"]).dt.normalize()
    plot_df["grade"] = plot_df["grade"].astype(str)
    wide = plot_df.pivot_table(index="date", columns="grade", values="proportion", aggfunc="sum")
    wide = wide.reindex(columns=grades).fillna(0.0)
    wide = wide.div(wide.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.25)
    x_dates = pd.to_datetime(wide.index).tolist()
    fig = go.Figure()
    for g in grades:
        fig.add_trace(
            go.Bar(
                name=g,
                x=x_dates,
                y=_plotly_float_list(wide[g].values),
                marker_color=GRADE_COLOR_MAP.get(g),
            )
        )
    fig.update_layout(
        title="Grade Mix by Origination Vintage (%)",
        barmode="stack",
        xaxis_title="Origination Month",
        xaxis_tickformat="%b\n%Y",
        xaxis_tickangle=DATE_X_TICKANGLE,
        yaxis=dict(tickformat=".0%", range=[0, 1], title="Percent of Total Loans (per Vintage)"),
        legend_title="Loan Grade",
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig, width='stretch', key=f"grade_mix_{key_suffix}")

def _make_borrower_chars_df(level: str, months_back: int) -> pd.DataFrame:
    """
    Deterministic mock borrower characteristics by origination vintage month.
    """
    dates = _get_monthly_dates(months_back)
    t = np.arange(months_back, dtype=float)

    level_adj = 1.0 if level == "Platform" else 0.98
    bc = f"{level}:borrower_chars"

    # Weighted-average FICO (demo): compute from grade mix proportions.
    mix_df = get_grade_mix_df(grades, months_back, freq="ME")
    # Representative FICO by grade bucket (higher grade -> higher FICO).
    fico_by_grade = {"A": 740, "B": 710, "C": 680, "D": 655}
    wtd_fico = []
    for d in dates:
        mp = pd.Timestamp(d).to_period("M")
        w = 0.0
        s = 0.0
        for g in grades:
            filt = (mix_df["date"].dt.to_period("M") == mp) & (mix_df["grade"] == g)
            prop_vals = mix_df.loc[filt, "proportion"].values
            prop = float(prop_vals[0]) if prop_vals.size > 0 else 0.0
            w += prop
            s += prop * fico_by_grade[g]
        base = (s / w) if w > 0 else 700.0
        wtd_fico.append(base)
    fico_wtd = np.array(wtd_fico, dtype=float)
    # Add a very gentle improvement trend + small noise, and a level adjustment.
    fico_wtd = fico_wtd + 4.5 * (1 - np.exp(-0.06 * t)) + np.array(
        [_det_normal(f"{bc}:fico", j, 0.0, 1.4) for j in range(months_back)], dtype=float
    )
    fico_wtd = np.clip(fico_wtd * level_adj, 650, 780)

    dti_mean = 36.5 - 1.6 * (1 - np.exp(-0.05 * t)) + np.array(
        [_det_normal(f"{bc}:dti", j, 0.0, 0.45) for j in range(months_back)], dtype=float
    )
    dti_mean = np.clip(dti_mean / level_adj, 28, 45)

    ltv_mean = 78.5 - 2.2 * (1 - np.exp(-0.05 * t)) + np.array(
        [_det_normal(f"{bc}:ltv", j, 0.0, 0.6) for j in range(months_back)], dtype=float
    )
    ltv_mean = np.clip(ltv_mean / level_adj, 60, 90)

    loan_amt_mean = 14500 + 900 * (1 - np.exp(-0.04 * t)) + np.array(
        [_det_normal(f"{bc}:loan", j, 0.0, 180) for j in range(months_back)], dtype=float
    )
    loan_amt_mean = np.clip(loan_amt_mean * (1.02 if level == "Platform" else 1.0), 9000, 22000)

    income_mean = 74000 + 3800 * (1 - np.exp(-0.04 * t)) + np.array(
        [_det_normal(f"{bc}:income", j, 0.0, 800) for j in range(months_back)], dtype=float
    )
    income_mean = np.clip(income_mean * level_adj, 45000, 140000)

    util_mean = 42.0 - 2.0 * (1 - np.exp(-0.05 * t)) + np.array(
        [_det_normal(f"{bc}:util", j, 0.0, 0.7) for j in range(months_back)], dtype=float
    )
    util_mean = np.clip(util_mean / level_adj, 20, 70)

    out = pd.DataFrame(
        {
            "date": dates,
            "fico_wtd": fico_wtd,
            "dti_mean": dti_mean,
            "ltv_mean": ltv_mean,
            "loan_amt_mean": loan_amt_mean,
            "income_mean": income_mean,
            "util_mean": util_mean,
        }
    )
    out["date"] = pd.to_datetime(out["date"])
    return out


def _make_geo_mix_df(level: str, months_back: int) -> pd.DataFrame:
    """
    Deterministic mock geography mix (top states) by vintage month.
    First month uses the base mix exactly (avoids a spurious "all zero" first slice in stacked areas).
    """
    dates = _get_monthly_dates(months_back)
    states = ["CA", "TX", "FL", "NY", "IL"]

    base = np.array([0.22, 0.18, 0.14, 0.12, 0.10])
    base = base / base.sum()

    rows: list[dict] = []
    last = base
    for i, d in enumerate(dates):
        if i == 0:
            cur = base.astype(float)
        else:
            # One deterministic stream for all months: constant seed, advancing index (not seed per month).
            geo_seed = f"{level}:geo_mix"
            stream_i = (i - 1) * len(states)
            drift = np.array(
                [_det_normal(geo_seed, stream_i + k, 0.0, 0.01) for k in range(len(states))],
                dtype=float,
            )
            cur = np.clip(last + drift, 0.05, 0.45)
            cur = cur / cur.sum()
        last = cur
        for s, p in zip(states, cur):
            rows.append({"date": d, "state": s, "proportion": float(p)})
    geo = pd.DataFrame(rows)
    geo["date"] = pd.to_datetime(geo["date"]).dt.normalize()
    geo["state"] = geo["state"].astype(str)
    return geo.sort_values(["date", "state"])


def render_investor_tabs(level: str, section_title: str):
    months_back = _get_months_back(level)
    chars_df = _make_borrower_chars_df(level, months_back)
    geo_df = _make_geo_mix_df(level, months_back)
    key_base = f"{section_title}_{level}".replace(" ", "_").lower()

    tabs = st.tabs(["Credit Profile", "Affordability", "Geography"])

    with tabs[0]:
        # go.Bar is more reliable than px.bar in some Snowflake Streamlit + Plotly builds.
        fico_y = _plotly_float_list(chars_df["fico_wtd"].fillna(700.0).astype(float).values)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=pd.to_datetime(chars_df["date"]).tolist(),
                    y=fico_y,
                    marker_color="rgb(12, 64, 118)",
                    hovertemplate="Weighted Avg FICO: %{y:.1f}<br>%{x|%b %Y}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="Credit Profile by Origination Vintage",
            xaxis_title="Origination Month",
            yaxis_title="Weighted Avg FICO",
            xaxis_tickformat="%b\n%Y",
            xaxis_tickangle=DATE_X_TICKANGLE,
            yaxis=dict(range=[500, 800], fixedrange=True, autorange=False),
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch", key=f"invest_credit_{key_base}")

    with tabs[1]:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(chars_df["date"]).tolist(),
                y=_plotly_float_list(chars_df["dti_mean"].values),
                mode="lines+markers",
                name="Avg DTI (%)",
                line=dict(color="rgb(12, 64, 118)", width=3),
                marker=dict(size=4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(chars_df["date"]).tolist(),
                y=_plotly_float_list(chars_df["income_mean"].values),
                mode="lines",
                name="Avg Income ($)",
                line=dict(color="rgb(140, 96, 0)", width=2, dash="dot"),
                yaxis="y2",
            )
        )
        fig.update_layout(
            title="Affordability by Origination Vintage",
            xaxis_title="Origination Month",
            yaxis_title="DTI (%)",
            xaxis_tickformat="%b\n%Y",
            xaxis_tickangle=DATE_X_TICKANGLE,
            template="plotly_white",
            hovermode="x unified",
            legend_title="Metric",
            yaxis2=dict(title="Income ($)", overlaying="y", side="right", tickprefix="$"),
        )
        st.plotly_chart(fig, width="stretch", key=f"invest_afford_{key_base}")

    with tabs[2]:
        states_order = ["CA", "TX", "FL", "NY", "IL"]
        g = geo_df.copy()
        g["date"] = pd.to_datetime(g["date"]).dt.normalize()
        g["state"] = g["state"].astype(str)
        wide_geo = g.pivot_table(index="date", columns="state", values="proportion", aggfunc="first")
        wide_geo = wide_geo.reindex(columns=states_order).fillna(0.0)
        wide_geo = wide_geo.div(wide_geo.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0 / len(states_order))
        # Categorical x (month labels) so every stacked bar has the same width; date+range squeezes edge bars.
        x_cats = [pd.Timestamp(ts).strftime("%b\n%Y") for ts in wide_geo.index]
        fig = go.Figure()
        for s in states_order:
            fig.add_trace(
                go.Bar(
                    name=s,
                    x=x_cats,
                    y=_plotly_float_list(wide_geo[s].values),
                    marker_color=GEO_COLOR_MAP.get(s),
                    hovertemplate="%{x}<br>" + s + ": %{y:.1%}<extra></extra>",
                )
            )
        fig.update_layout(
            title="Top-State Geography Mix by Origination Vintage",
            barmode="stack",
            xaxis_title="Origination Month",
            xaxis=dict(type="category", categoryorder="array", categoryarray=x_cats, tickangle=DATE_X_TICKANGLE),
            yaxis=dict(tickformat=".0%", range=[0, 1], title="Share"),
            template="plotly_white",
            hovermode="x unified",
            legend_title="State",
        )
        st.plotly_chart(fig, width="stretch", key=f"invest_geo_{key_base}")

def _get_monthly_dates(total_months: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.date_range(end=_DATA_END_DATE, periods=total_months, freq="ME")).normalize()


def _get_months_back(level: str) -> int:
    # Halved vs prior scaffold:
    # - Platform: 48 -> 24
    # - Deal Collateral: 18 -> 9
    return 24 if level == "Platform" else 9


def _rgb_lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Linear interpolate between two RGB colors."""
    tt = float(np.clip(t, 0.0, 1.0))
    return (
        int(round(a[0] + (b[0] - a[0]) * tt)),
        int(round(a[1] + (b[1] - a[1]) * tt)),
        int(round(a[2] + (b[2] - a[2]) * tt)),
    )


def _vintage_fade_color(vintage_frac: float) -> str:
    # Early vintages: blue. Late vintages: deep yellow.
    # Use darker endpoints so the traces read clearly even at moderate opacity.
    blue = (12, 64, 118)  # deep blue
    deep_yellow = (140, 96, 0)  # dark deep yellow
    r, g, b = _rgb_lerp(blue, deep_yellow, vintage_frac)
    return f"rgb({r},{g},{b})"

def _cut_date_fade_color(cut_frac: float) -> str:
    # Cut dates fade blue -> orange over time.
    blue = (12, 64, 118)  # deep blue
    orange = (255, 127, 14)  # plotly orange
    r, g, b = _rgb_lerp(blue, orange, cut_frac)
    return f"rgb({r},{g},{b})"


def _month_periods(start: pd.Period, end: pd.Period) -> list[pd.Period]:
    periods: list[pd.Period] = []
    cur = start
    while cur <= end:
        periods.append(cur)
        cur = cur + 1
    return periods


def _simulate_model_point(metric: str, improvement: float, vintage_frac: float, cut_frac: float) -> float:
    """
    Demo model output for *end-of-life (maturity) outcome* at a given cut date.

    `improvement`/`cut_frac` increase over time (later cut dates -> "better" performance).
    `vintage_frac` is 0..1 over vintages (used for slight cross-vintage spread).
    """
    imp = float(np.clip(improvement, 0.0, 1.0))
    v = float(np.clip(vintage_frac, 0.0, 1.0))

    # Smooth progression that changes gradually across cut dates.
    eased = float(np.power(imp, 1.15))

    if metric == "IRR":
        # %; improves upward over time (e.g., 8.00 -> 9.20)
        start, end = 8.00, 9.20
        base = start + (end - start) * eased
        # Small dispersion across vintages + tiny wiggle across cut dates (kept tight).
        base += (v - 0.5) * 0.06 + (cut_frac - 0.5) * 0.02
        return float(base)

    if metric == "CGL":
        # %; improves downward over time (e.g., 5.00 -> 3.00)
        # Tighten range so dots move ~0.8% total.
        start, end = 5.00, 4.20
        base = start + (end - start) * eased
        base += (v - 0.5) * 0.10 + (cut_frac - 0.5) * 0.03
        return float(max(0.0, base))

    if metric == "MOIC":
        # multiple; improves upward over time (e.g., 1.16 -> 1.20)
        start, end = 1.16, 1.20
        base = start + (end - start) * eased
        base += (v - 0.5) * 0.003 + (cut_frac - 0.5) * 0.001
        return float(base)

    raise ValueError(f"Unknown metric: {metric}")


def render_model_projection_tabs():
    tab_labels = ["CGL", "MOIC", "IRR"]
    tabs = st.tabs(tab_labels)

    # Use the dashboard's Deal Collateral vintage set for this section.
    months_back = _get_months_back("Deal Collateral")
    vintage_dates = _get_monthly_dates(months_back)
    vintage_periods = [pd.Timestamp(d).to_period("M") for d in vintage_dates]

    max_term_months = 36
    as_of_period = pd.Timestamp(_DATA_END_DATE).to_period("M")
    first_vintage = min(vintage_periods)
    cut_periods = _month_periods(first_vintage, as_of_period)

    def render_metric(metric: str, y_axis_title: str, hover_fmt: str):
        fig = go.Figure()

        # One dot per (vintage, cut date). X axis is vintage, so dots stack vertically per vintage.
        # Fade color over cut date time.
        last_idx = max(len(cut_periods) - 1, 0)
        all_y: list[float] = []
        for cut_idx, cut_p in enumerate(cut_periods):
            cut_frac = cut_idx / max(last_idx, 1)
            improvement = cut_frac
            is_most_recent = cut_idx == last_idx
            color = _cut_date_fade_color(cut_frac)
            size = 7 if is_most_recent else 4
            line_w = 1.5 if is_most_recent else 0.0

            xs: list[str] = []
            ys: list[float] = []
            texts: list[str] = []

            for v_idx, v_p in enumerate(vintage_periods):
                if v_p > cut_p:
                    continue  # vintage not originated yet at this cut date
                vintage_frac = v_idx / max(len(vintage_periods) - 1, 1)
                y = _simulate_model_point(metric, improvement=improvement, vintage_frac=vintage_frac, cut_frac=cut_frac)
                xs.append(v_p.strftime("%b %Y"))
                ys.append(y)
                texts.append(f"Vintage {v_p.strftime('%b %Y')}<br>Cut {cut_p.strftime('%b %Y')}")

            if not xs:
                continue
            all_y.extend(ys)

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name=cut_p.strftime("%b %Y"),
                    cliponaxis=False,
                    marker=dict(
                        color=color,
                        size=size,
                        opacity=0.95 if is_most_recent else 0.65,
                        line=dict(color="rgba(0,0,0,0.35)", width=line_w),
                    ),
                    hovertemplate=f"{metric}: {hover_fmt}<br>%{{text}}<extra></extra>",
                    text=texts,
                    showlegend=True,
                )
            )

        if all_y:
            y_min = float(np.min(all_y))
            y_max = float(np.max(all_y))
            # Extra headroom to avoid marker clipping.
            # When points are tightly clustered (e.g., MOIC), padding based on (y_max - y_min)
            # can be too small, so base it primarily on y_max.
            headroom_ratio = {"MOIC": 0.18, "IRR": 0.12, "CGL": 0.10}.get(metric, 0.10)
            abs_min_pad = {"MOIC": 0.08, "IRR": 0.8, "CGL": 0.5}.get(metric, 0.5)
            pad = max(y_max * headroom_ratio, abs_min_pad)
            # Always start at 0, but ensure enough headroom so the largest dots aren't clipped.
            fig.update_yaxes(range=[0, y_max + pad], rangemode="tozero")
        else:
            fig.update_yaxes(rangemode="tozero")

        fig.update_layout(
            title=f"{metric} Model Output by Vintage (Dots by Cut Date)",
            xaxis_title="Vintage",
            yaxis_title=y_axis_title,
            xaxis_tickangle=DATE_X_TICKANGLE,
            template="plotly_white",
            hovermode="closest",
            legend_title="Cut Date",
        )

        st.plotly_chart(fig, width="stretch", key=f"modelproj_{metric.lower()}")

    with tabs[0]:
        render_metric("CGL", "CGL (%)", "%{y:.2f}%")
    with tabs[1]:
        render_metric("MOIC", "MOIC (x)", "%{y:.2f}X")
    with tabs[2]:
        render_metric("IRR", "IRR (%)", "%{y:.2f}%")


def _dq_projection_params(dq_bucket: str) -> dict:
    # Values are for demo/scaffold purposes; shape is what matters.
    return {
        # These params control the curvature only. Both real and projected curves are forced
        # to start at 0 and remain non-decreasing.
        "DQ0+": {"end": 9.0, "k": 0.06, "curv": 1.15, "real_end_ratio": 0.88, "real_k_ratio": 0.95, "jitter": 0.005},
        "DQ30+": {"end": 5.5, "k": 0.05, "curv": 1.10, "real_end_ratio": 0.90, "real_k_ratio": 0.98, "jitter": 0.004},
        "DQ60+": {"end": 2.6, "k": 0.045, "curv": 1.08, "real_end_ratio": 0.92, "real_k_ratio": 0.99, "jitter": 0.003},
        "DQ90+": {"end": 1.3, "k": 0.04, "curv": 1.06, "real_end_ratio": 0.94, "real_k_ratio": 1.00, "jitter": 0.002},
    }[dq_bucket]


def _make_dq_curves(level: str, dq_bucket: str, total_months: int, real_months: int):
    params = _dq_projection_params(dq_bucket)
    dates = _get_monthly_dates(total_months)
    t = np.arange(total_months)

    # Make projection depend slightly on platform vs deal.
    level_adjust = 1.0 if level == "Platform" else 0.97

    # Monotonic smooth curve: end * (1-exp(-k t))^curv. Starts at 0 at t=0.
    projected = (params["end"] * level_adjust) * np.power(1 - np.exp(-params["k"] * t), params["curv"])

    # Real is another monotonic smooth curve, with a tiny non-negative "jitter"
    # that cannot introduce any downward movement.
    base_real = (params["end"] * params["real_end_ratio"]) * np.power(
        1 - np.exp(-(params["k"] * params["real_k_ratio"]) * t),
        params["curv"],
    )
    # Create only upward increments so the series stays non-decreasing.
    jscale = params["jitter"] * params["end"]
    up_seed = f"{level}:{dq_bucket}:real:up"
    upward_steps = np.array(
        [max(0.0, _det_normal(up_seed, j, 0.0, jscale)) for j in range(total_months)],
        dtype=float,
    )
    upward_steps = np.cumsum(upward_steps)
    if upward_steps[-1] > 0:
        upward_steps = upward_steps / upward_steps[-1] * (params["jitter"] * params["end"] * 6)
    real = base_real + upward_steps

    real_y = real.copy()
    if real_months < total_months:
        real_y[real_months:] = np.nan  # break the line after we stop having "real" history

    # Safety clamps: enforce start at 0 and monotonicity.
    projected[0] = 0.0
    real[0] = 0.0
    projected = np.maximum.accumulate(projected)
    real = np.maximum.accumulate(real)
    return dates, real, projected


def _make_cgl_curves(level: str, total_months: int, real_months: int):
    dates = _get_monthly_dates(total_months)
    t = np.arange(total_months)
    level_adjust = 1.0 if level == "Platform" else 0.96

    # Monotonic smooth curve, forced to start at 0.
    projected_end = 7.0 * level_adjust
    projected = projected_end * np.power(1 - np.exp(-0.045 * t), 1.08)

    base_real_end = 6.6 * (1.0 if level == "Platform" else 0.98)
    base_real = base_real_end * np.power(1 - np.exp(-0.040 * t), 1.07)

    # Tiny upward-only jitter for the "real" line.
    js = 0.003 * projected_end
    upward_steps = np.array(
        [max(0.0, _det_normal(f"{level}:cgl:real:up", j, 0.0, js)) for j in range(total_months)],
        dtype=float,
    )
    upward_steps = np.cumsum(upward_steps)
    if upward_steps[-1] > 0:
        upward_steps = upward_steps / upward_steps[-1] * (0.01 * projected_end)
    real = base_real + upward_steps

    real_y = real.copy()
    if real_months < total_months:
        real_y[real_months:] = np.nan

    projected[0] = 0.0
    real[0] = 0.0
    projected = np.maximum.accumulate(projected)
    real = np.maximum.accumulate(real)
    return dates, real, projected


def _make_moic_curves(level: str, total_months: int, real_months: int):
    dates = _get_monthly_dates(total_months)
    t = np.arange(total_months)

    # Projected: slope up to ~1.0, then plateau at 1.17X after about 36 months.
    plateau_month = 36
    # Make the early climb slower so the curve "feels" like DQ/CGL.
    t_to_one = 18
    moic_one = 1.0
    moic_plateau = 1.17

    def eased_frac(s: float, k: float) -> float:
        """
        Monotonic easing from 0->1 over s in [0,1], with adjustable curvature.
        Normalized so eased_frac(1)=1 exactly.
        """
        if s <= 0:
            return 0.0
        if s >= 1:
            return 1.0
        denom = 1.0 - np.exp(-k)
        if denom == 0:
            return s
        return (1.0 - np.exp(-k * s)) / denom

    projected = np.empty(total_months, dtype=float)
    for i in range(total_months):
        if i <= t_to_one:
            s = 0.0 if t_to_one == 0 else (i / t_to_one)
            # Start slower, then gradually steepen.
            projected[i] = moic_one * eased_frac(s, k=2.2)
        elif i <= plateau_month:
            denom = plateau_month - t_to_one
            s = 0.0 if denom == 0 else ((i - t_to_one) / denom)
            # Approach plateau smoothly by ~36 months.
            projected[i] = moic_one + (moic_plateau - moic_one) * eased_frac(s, k=3.2)
        else:
            projected[i] = moic_plateau

    projected[0] = 0.0
    projected = np.maximum.accumulate(projected)

    # Real: very slightly below projected but still monotonic and starting at 0.
    real_plateau = moic_plateau * (0.99 if level == "Platform" else 0.975)
    real_one = moic_one * (1.0 if level == "Platform" else 0.99)
    upward_steps = np.array(
        [max(0.0, _det_normal(f"{level}:moic:real:up", j, 0.0, 0.002)) for j in range(total_months)],
        dtype=float,
    )
    upward_steps = np.cumsum(upward_steps)
    if upward_steps[-1] > 0:
        upward_steps = upward_steps / upward_steps[-1] * 0.01

    real = np.empty(total_months, dtype=float)
    for i in range(total_months):
        if i <= t_to_one:
            s = 0.0 if t_to_one == 0 else (i / t_to_one)
            real[i] = real_one * eased_frac(s, k=2.0)
        elif i <= plateau_month:
            denom = plateau_month - t_to_one
            s = 0.0 if denom == 0 else ((i - t_to_one) / denom)
            real[i] = real_one + (real_plateau - real_one) * eased_frac(s, k=3.0)
        else:
            real[i] = real_plateau
    real = real + upward_steps
    real[0] = 0.0
    real = np.maximum.accumulate(real)
    return dates, real, projected


def render_dq_tab(level: str):
    dq_bucket = st.selectbox(
        "Select DQ bucket",
        ["DQ0+", "DQ30+", "DQ60+", "DQ90+"],
        index=0,
        key=f"dq_bucket_{level}",
    )
    total_months = _get_months_back(level)
    max_term_months = 36
    mob_x = np.arange(0, max_term_months + 1, dtype=int)
    vintage_dates = _get_monthly_dates(total_months)  # origination months shown (consistent w/ Overview)
    _, real_age, projected_age = _make_dq_curves(level, dq_bucket, max_term_months + 1, max_term_months + 1)

    fig = go.Figure()
    # Base case projected line (one line for the whole pool).
    fig.add_trace(
        go.Scatter(
            x=_plotly_int_list(mob_x),
            y=_plotly_float_list(projected_age),
            mode="lines",
            name="Base Case",
            line=dict(color="rgb(255, 127, 14)", width=3, dash="dot"),
            hovertemplate=f"Projected {dq_bucket}: %{{y:.2f}}%<br>Calendar MOB: %{{x}}<extra></extra>",
        )
    )

    # Real curves: one solid line per origination vintage (start month).
    as_of_period = pd.Timestamp(_DATA_END_DATE).to_period("M")
    for i in range(total_months):
        vintage_period = pd.Timestamp(vintage_dates[i]).to_period("M")
        real_max_mob = int((as_of_period - vintage_period).n)
        real_max_mob = int(np.clip(real_max_mob, 0, max_term_months))
        vintage_start = vintage_dates[i].strftime("%b %Y")
        # Smoothly vary magnitude across vintages so each line is distinct, but still monotonic.
        vintage_factor = 0.8 + 0.4 * (i / max(total_months - 1, 1))  # wider spread
        vintage_frac = i / max(total_months - 1, 1)
        # Plot on calendar axis so ALL curves start at Calendar MOB 0.
        y = np.maximum.accumulate(real_age * vintage_factor)
        y[0] = 0.0  # enforce exact start at 0
        # Stop the real trace once we run out of real months for this vintage.
        if real_max_mob < max_term_months:
            y[real_max_mob + 1 :] = np.nan
        fig.add_trace(
            go.Scatter(
                x=_plotly_int_list(mob_x),
                y=_plotly_float_list(y),
                mode="lines",
                name=vintage_start,
                showlegend=True,
                line=dict(color=_vintage_fade_color(vintage_frac), width=2, dash="solid"),
                opacity=0.6,
                connectgaps=False,
                hovertemplate=f"Real {dq_bucket}: %{{y:.2f}}%<br>Vintage start: {vintage_start}<br>Calendar MOB: %{{x}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{dq_bucket} Curves (Real vs Base Case Projection)",
        xaxis_title="Calendar MOB",
        yaxis_title=f"{dq_bucket} (%)",
        xaxis=dict(range=[0, max_term_months], dtick=6),
        legend_title="Series",
        template="plotly_white",
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch", key=f"perf_{level}_dq")


def render_cgl_tab(level: str):
    total_months = _get_months_back(level)
    max_term_months = 36
    mob_x = np.arange(0, max_term_months + 1, dtype=int)
    vintage_dates = _get_monthly_dates(total_months)
    _, real_age, projected_age = _make_cgl_curves(level, max_term_months + 1, max_term_months + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_plotly_int_list(mob_x),
            y=_plotly_float_list(projected_age),
            mode="lines",
            name="Base Case",
            line=dict(color="rgb(255, 127, 14)", width=3, dash="dot"),
            hovertemplate="Projected CGL: %{y:.2f}%<br>Calendar MOB: %{x}<extra></extra>",
        )
    )

    as_of_period = pd.Timestamp(_DATA_END_DATE).to_period("M")
    for i in range(total_months):
        vintage_period = pd.Timestamp(vintage_dates[i]).to_period("M")
        real_max_mob = int((as_of_period - vintage_period).n)
        real_max_mob = int(np.clip(real_max_mob, 0, max_term_months))
        vintage_start = vintage_dates[i].strftime("%b %Y")
        vintage_factor = 0.85 + 0.3 * (i / max(total_months - 1, 1))  # wider spread
        vintage_frac = i / max(total_months - 1, 1)
        # Plot on calendar axis so ALL curves start at Calendar MOB 0.
        y = np.maximum.accumulate(real_age * vintage_factor)
        y[0] = 0.0
        if real_max_mob < max_term_months:
            y[real_max_mob + 1 :] = np.nan
        fig.add_trace(
            go.Scatter(
                x=_plotly_int_list(mob_x),
                y=_plotly_float_list(y),
                mode="lines",
                name=vintage_start,
                showlegend=True,
                line=dict(color=_vintage_fade_color(vintage_frac), width=2, dash="solid"),
                opacity=0.6,
                connectgaps=False,
                hovertemplate=f"Real CGL: %{{y:.2f}}%<br>Vintage start: {vintage_start}<br>Calendar MOB: %{{x}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="CGL Curves (Real vs Base Case Projection)",
        xaxis_title="Calendar MOB",
        yaxis_title="CGL (%)",
        xaxis=dict(range=[0, max_term_months], dtick=6),
        legend_title="Series",
        template="plotly_white",
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch", key=f"perf_{level}_cgl")


def render_moic_tab(level: str):
    total_months = _get_months_back(level)
    max_term_months = 36
    mob_x = np.arange(0, max_term_months + 1, dtype=int)
    vintage_dates = _get_monthly_dates(total_months)
    _, real_age, projected_age = _make_moic_curves(level, max_term_months + 1, max_term_months + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_plotly_int_list(mob_x),
            y=_plotly_float_list(projected_age),
            mode="lines",
            name="Base Case",
            line=dict(color="rgb(255, 127, 14)", width=3, dash="dot"),
            hovertemplate="Projected MOIC: %{y:.2f}X<br>Calendar MOB: %{x}<extra></extra>",
        )
    )

    as_of_period = pd.Timestamp(_DATA_END_DATE).to_period("M")
    for i in range(total_months):
        vintage_period = pd.Timestamp(vintage_dates[i]).to_period("M")
        real_max_mob = int((as_of_period - vintage_period).n)
        real_max_mob = int(np.clip(real_max_mob, 0, max_term_months))
        vintage_start = vintage_dates[i].strftime("%b %Y")
        vintage_factor = 0.92 + 0.16 * (i / max(total_months - 1, 1))  # wider spread
        vintage_frac = i / max(total_months - 1, 1)
        # Plot on calendar axis so ALL curves start at Calendar MOB 0.
        y = np.maximum.accumulate(real_age * vintage_factor)
        y[0] = 0.0
        if real_max_mob < max_term_months:
            y[real_max_mob + 1 :] = np.nan
        fig.add_trace(
            go.Scatter(
                x=_plotly_int_list(mob_x),
                y=_plotly_float_list(y),
                mode="lines",
                name=vintage_start,
                showlegend=True,
                line=dict(color=_vintage_fade_color(vintage_frac), width=2, dash="solid"),
                opacity=0.6,
                connectgaps=False,
                hovertemplate=f"Real MOIC: %{{y:.2f}}X<br>Vintage start: {vintage_start}<br>Calendar MOB: %{{x}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="MOIC (Real vs Base Case Projection)",
        xaxis_title="Calendar MOB",
        yaxis_title="MOIC (x)",
        xaxis=dict(range=[0, max_term_months], dtick=6),
        legend_title="Series",
        template="plotly_white",
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch", key=f"perf_{level}_moic")


def render_performance_tabs(level: str):
    tab_labels = ["DQ Curves", "CGL", "MOIC"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_dq_tab(level)
    with tabs[1]:
        render_cgl_tab(level)
    with tabs[2]:
        render_moic_tab(level)

def render_tabs(level: str, section_title: str):
    tab_labels = ["Volume", "Coupon", "Grade Mix"]
    tabs = st.tabs(tab_labels)

    months_back = _get_months_back(level)  # keep performance + overview aligned
    grade_vol_multiplier = 2 if level == "Platform" else 1
    volume_df = get_volume_df(grades, months_back, grade_vol_multiplier=grade_vol_multiplier)
    coupon_df = get_coupon_df(grades, months_back)
    grademix_df = get_grade_mix_df(grades, months_back)

    with tabs[0]:
        render_volume_tab(volume_df, key_suffix=f"{section_title}_{level}")
    with tabs[1]:
        render_coupon_tab(coupon_df, key_suffix=f"{section_title}_{level}")
    with tabs[2]:
        render_grade_mix_tab(grademix_df, key_suffix=f"{section_title}_{level}")

def render_section(title: str) -> None:
    st.header(title)
    if title == "Model Projection":
        with st.expander("Deal Collateral", expanded=False):
            render_model_projection_tabs()
        return

    with st.expander("Platform", expanded=False):
        if title == "Performance":
            render_performance_tabs("Platform")
        elif title == "Other Trends":
            render_investor_tabs("Platform", section_title=title)
        else:
            render_tabs("Platform", section_title=title)
    with st.expander("Deal Collateral", expanded=False):
        if title == "Performance":
            render_performance_tabs("Deal Collateral")
        elif title == "Other Trends":
            render_investor_tabs("Deal Collateral", section_title=title)
        else:
            render_tabs("Deal Collateral", section_title=title)

render_section("Overview")
render_section("Performance")
render_section("Model Projection")
render_section("Other Trends")
