# Signal Atlas

Signal Atlas is an interactive “what changed?” explorer for a small neighborhood coffee network. It is a compact, real data-product-style example of the Python visualization stack:

- **NumPy** creates reproducible environmental signals and runs a 5,000-trial weekly revenue scenario.
- **pandas DataFrames** handle the generated observations, filtering, grouping, resampling, and derived metrics.
- **pandas Series** power the weekday summary and the simulation output.
- **Seaborn** reveals weekday/weather demand patterns and the temperature/revenue relationship.
- **Matplotlib** turns the analysis into a clean narrative: trend, heatmap, relationship, and scenario distribution.
- **Streamlit** makes the analysis interactive without hiding the Python underneath.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run signal_atlas.py
```

The app uses deterministic synthetic data, so it works immediately without an API key or database. The final caption in the interface points to the one function to replace when you have real data.

## What to try

1. Compare **Market Square** and **University** over the same date window.
2. Set staffing to `+2` and temperature to `-3°C` to see how the scenario distribution shifts.
3. Open the DataFrame/Series inspector to see the exact rows and grouped values behind the charts.