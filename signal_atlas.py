"""Signal Atlas — a small decision cockpit for a neighborhood coffee network.

Run with:
    pip install -r requirements.txt
    streamlit run signal_atlas.py

The app intentionally keeps the data local and reproducible. It demonstrates:
* NumPy for signal generation and Monte Carlo scenarios
* pandas DataFrames and Series for cleaning, grouping, indexing, and KPIs
* Matplotlib for the narrative charts
* Seaborn for pattern discovery and statistical relationships
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(
    page_title="Signal Atlas",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", context="notebook")

PALETTE = {
    "ink": "#17202A",
    "muted": "#637083",
    "teal": "#0F8B8D",
    "coral": "#E76F51",
    "gold": "#E9C46A",
    "paper": "#F8FAFC",
}


@st.cache_data
def make_dataset(seed: int = 17, days: int = 365) -> pd.DataFrame:
    """Create a deterministic, believable operating dataset with NumPy."""
    rng = np.random.default_rng(seed)
    calendar = pd.date_range(
        end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1),
        periods=days,
        freq="D",
    )

    locations = pd.DataFrame(
        {
            "location": ["Riverside", "Market Square", "University"],
            "base_demand": [154, 182, 136],
            "price_index": [1.00, 1.08, 0.96],
            "weather_bias": [0.98, 1.03, 1.00],
        }
    )
    rows: list[pd.DataFrame] = []
    for location in locations.itertuples(index=False):
        day_number = np.arange(days)
        dates = pd.Series(calendar, name="date")
        weekday = dates.dt.dayofweek.to_numpy()
        seasonal = 1 + 0.10 * np.sin((day_number / 365) * 2 * np.pi - 0.5)
        weekend = np.where(weekday >= 5, 0.82, 1.0)
        trend = 1 + day_number * 0.00055
        temperature = 19 + 10 * np.sin((day_number / 365) * 2 * np.pi) + rng.normal(0, 3.2, days)
        rain_probability = np.clip(0.28 - 0.07 * np.sin(day_number / 365 * 2 * np.pi), 0.06, 0.55)
        rainy = rng.random(days) < rain_probability
        event = ((weekday == 4) & (rng.random(days) > 0.55)) | (
            (weekday == 5) & (rng.random(days) > 0.76)
        )
        weather_multiplier = np.where(rainy, 0.79, 1.03 + np.clip(temperature - 20, -6, 8) * 0.012)
        event_multiplier = np.where(event, 1.18, 1.0)
        expected = (
            location.base_demand
            * seasonal
            * weekend
            * trend
            * weather_multiplier
            * event_multiplier
            * location.weather_bias
        )
        demand = np.maximum(35, rng.normal(expected, np.sqrt(expected) * 2.7))
        staff = np.clip(np.round(demand / 39 + rng.normal(0, 0.55, days)), 3, 10).astype(int)
        ticket = location.price_index * (6.1 + 0.17 * temperature + rng.normal(0, 0.35, days))
        revenue = demand * ticket
        waste_rate = np.clip(0.045 + np.maximum(staff * 6 - demand, 0) / 1500 + rng.normal(0, 0.008, days), 0.015, 0.14)

        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "location": location.location,
                    "demand": demand.round(1),
                    "revenue": revenue.round(2),
                    "staff": staff,
                    "temperature": temperature.round(1),
                    "rainy": rainy,
                    "event_day": event,
                    "waste_rate": waste_rate,
                    "day_name": dates.dt.day_name(),
                    "weekday": weekday,
                }
            )
        )

    dataset = pd.concat(rows, ignore_index=True)
    dataset["revenue_per_staff"] = (dataset["revenue"] / dataset["staff"]).round(2)
    dataset["demand_index"] = (
        dataset.groupby("location")["demand"]
        .transform(lambda series: (series / series.mean() * 100).round(1))
    )
    return dataset


def money(value: float) -> str:
    return f"₹{value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1f}%"


def apply_filters(dataset: pd.DataFrame, location: str, start: date, end: date) -> pd.DataFrame:
    """Return a clean filtered DataFrame; date inputs are intentionally explicit."""
    mask = (
        dataset["location"].eq(location)
        & dataset["date"].dt.date.ge(start)
        & dataset["date"].dt.date.le(end)
    )
    return dataset.loc[mask].copy()


def simulate_week(
    selected: pd.DataFrame,
    staffing_delta: int,
    temperature_shift: float,
    simulations: int = 5000,
) -> pd.Series:
    """Run a demand/revenue scenario as a pandas Series."""
    if selected.empty:
        return pd.Series(dtype=float)
    baseline = selected["revenue"].tail(28).mean()
    volatility = selected["revenue"].tail(90).std()
    staffing_effect = 1 + np.clip(staffing_delta * 0.018, -0.12, 0.14)
    weather_effect = 1 + np.clip(temperature_shift * 0.012, -0.10, 0.10)
    rng = np.random.default_rng(2026)
    samples = rng.normal(
        loc=baseline * staffing_effect * weather_effect * 7,
        scale=max(volatility * np.sqrt(7), baseline * 0.18),
        size=simulations,
    )
    return pd.Series(np.maximum(samples, 0), name="simulated_weekly_revenue")


def draw_trend(selected: pd.DataFrame) -> plt.Figure:
    daily = (
        selected.set_index("date")["revenue"]
        .resample("W-SUN")
        .sum()
        .rename("weekly_revenue")
        .to_frame()
    )
    daily["rolling"] = daily["weekly_revenue"].rolling(4, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(11, 3.7))
    ax.plot(daily.index, daily["weekly_revenue"], color=PALETTE["gold"], alpha=0.45, linewidth=1.5)
    ax.plot(daily.index, daily["rolling"], color=PALETTE["teal"], linewidth=2.8, label="4-week signal")
    ax.fill_between(daily.index, daily["weekly_revenue"], daily["rolling"], color=PALETTE["teal"], alpha=0.06)
    ax.set_title("Revenue signal, with short-term noise separated from direction", loc="left", weight="bold")
    ax.set_ylabel("Weekly revenue (₹)")
    ax.set_xlabel("")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def draw_heatmap(selected: pd.DataFrame) -> plt.Figure:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = (
        selected.assign(day_name=pd.Categorical(selected["day_name"], categories=order, ordered=True))
        .pivot_table(index="day_name", columns="rainy", values="demand", aggfunc="mean", observed=False)
        .rename(columns={False: "Clear", True: "Rain"})
        .reindex(order)
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.7))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".0f",
        cmap=sns.blend_palette(["#E8F1F2", PALETTE["teal"]], as_cmap=True),
        linewidths=1,
        linecolor="white",
        cbar=False,
        ax=ax,
    )
    ax.set_title("Demand rhythm: weekday × weather", loc="left", weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def draw_relationship(selected: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.7))
    sns.regplot(
        data=selected.sample(min(len(selected), 450), random_state=4),
        x="temperature",
        y="revenue",
        scatter_kws={"s": 22, "alpha": 0.26, "color": PALETTE["coral"]},
        line_kws={"color": PALETTE["ink"], "linewidth": 2},
        ax=ax,
    )
    ax.set_title("Relationship to test: temperature and revenue", loc="left", weight="bold")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Daily revenue (₹)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


dataset = make_dataset()

with st.sidebar:
    st.markdown("## ◈ Signal Atlas")
    st.caption("A decision cockpit for finding the signal inside everyday operating data.")
    st.divider()
    location = st.selectbox("Location", sorted(dataset["location"].unique()))
    min_date = dataset["date"].min().date()
    max_date = dataset["date"].max().date()
    date_range = st.date_input("Observation window", (max(min_date, max_date - timedelta(days=180)), max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
    st.divider()
    st.markdown("**Scenario controls**")
    staffing_delta = st.slider("Staffing change", -2, 2, 0, help="Compare next-week outcomes against the recent baseline.")
    temperature_shift = st.slider("Temperature shift (°C)", -5, 5, 0)
    st.caption("The sample data is deterministic, so the story stays reproducible while you explore.")

selected = apply_filters(dataset, location, start_date, end_date)
if selected.empty:
    st.warning("No observations match this window. Widen the date range to continue.")
    st.stop()

total_revenue = selected["revenue"].sum()
avg_daily = selected["revenue"].mean()
avg_demand = selected["demand"].mean()
waste = selected["waste_rate"].mean() * 100

st.markdown(
    f"""
    <div style="padding: 1.4rem 0 0.6rem 0;">
      <div style="color:{PALETTE['teal']};font-weight:700;letter-spacing:.13em;font-size:.78rem;">OPERATING SIGNALS / {location.upper()}</div>
      <h1 style="margin:.25rem 0 .35rem 0;font-size:2.5rem;letter-spacing:-.04em;">What changed, and what should we do next?</h1>
      <p style="color:{PALETTE['muted']};font-size:1.05rem;">A transparent sandbox for turning messy daily observations into a decision worth making.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Revenue in window", money(total_revenue))
k2.metric("Average daily revenue", money(avg_daily))
k3.metric("Average daily demand", f"{avg_demand:,.0f}", f"{selected['demand_index'].mean() - 100:+.1f}% vs location norm")
k4.metric("Estimated waste", pct(waste), "lower is better")

st.pyplot(draw_trend(selected), use_container_width=True)

left, right = st.columns(2)
with left:
    st.pyplot(draw_heatmap(selected), use_container_width=True)
with right:
    st.pyplot(draw_relationship(selected), use_container_width=True)

st.markdown("### Next-week scenario")
st.caption("This is a range, not a promise: 5,000 NumPy trials use recent volatility to show how a staffing or weather change could move weekly revenue.")
simulation = simulate_week(selected, staffing_delta, temperature_shift)
if simulation.empty:
    st.info("Not enough data for a simulation.")
else:
    s1, s2, s3 = st.columns(3)
    s1.metric("Expected weekly revenue", money(float(simulation.mean())))
    s2.metric("Likely range", f"{money(float(simulation.quantile(.10)))} – {money(float(simulation.quantile(.90)))}")
    s3.metric("Chance above baseline", pct(float((simulation > selected["revenue"].tail(28).mean() * 7).mean() * 100)))
    fig, ax = plt.subplots(figsize=(12, 2.6))
    sns.histplot(simulation, bins=45, color=PALETTE["teal"], alpha=0.85, ax=ax)
    ax.axvline(simulation.mean(), color=PALETTE["coral"], linewidth=2, label="Expected")
    ax.set_title("Monte Carlo distribution of next-week revenue", loc="left", weight="bold")
    ax.set_xlabel("Simulated weekly revenue (₹)")
    ax.set_ylabel("Trials")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

with st.expander("Inspect the DataFrame and Series behind the story"):
    st.write("Filtered pandas DataFrame")
    st.dataframe(
        selected.sort_values("date", ascending=False).head(25),
        use_container_width=True,
        hide_index=True,
    )
    st.write("Revenue by weekday — pandas Series")
    weekday_series = selected.groupby("day_name", sort=False)["revenue"].mean().sort_values(ascending=False)
    st.bar_chart(weekday_series)

st.caption("Signal Atlas uses synthetic operating data. Replace make_dataset() with a CSV or database query when you have a real source.")