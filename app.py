"""Supply Chain Risk Monitor — Streamlit dashboard."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from risk_engine.data_loader import (
    load_disruption_events,
    load_industry_exposure,
    load_port_congestion,
    load_shipping_rates,
)
from risk_engine.features import SEVERITY_LABELS, build_disruption_features, encode_prediction_input
from risk_engine.model import load_models, save_models, train_severity_classifier, train_freight_regressor, train_recovery_regressor

_MODEL_DIR = Path(__file__).parent / "models"


@st.cache_resource(show_spinner="Training models on first run…")
def ensure_models() -> None:
    """Train and save models if they haven't been built yet."""
    if (_MODEL_DIR / "severity_classifier.pkl").exists():
        return
    df = load_disruption_events()
    X, y_sev, y_freight, y_recovery = build_disruption_features(df)
    sev_clf, _ = train_severity_classifier(X, y_sev)
    freight_reg, _ = train_freight_regressor(X, y_freight)
    recovery_reg, _ = train_recovery_regressor(X, y_recovery)
    save_models(sev_clf, freight_reg, recovery_reg, list(X.columns))


ensure_models()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Risk Monitor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEVERITY_COLORS = {
    "medium": "#FFC107",
    "high": "#FF7F00",
    "extreme": "#D32F2F",
    "Medium": "#FFC107",
    "High": "#FF7F00",
    "Extreme": "#D32F2F",
}

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def get_events() -> pd.DataFrame:
    return load_disruption_events()


@st.cache_data
def get_industry() -> pd.DataFrame:
    return load_industry_exposure()


@st.cache_data
def get_ports() -> pd.DataFrame:
    return load_port_congestion()


@st.cache_data
def get_rates() -> pd.DataFrame:
    return load_shipping_rates()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🚢 SC Risk Monitor")
page = st.sidebar.radio(
    "Navigate",
    [
        "Risk Predictor",
        "Disruption History",
        "Industry Vulnerability",
        "Shipping Markets",
        "Port Congestion",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Data coverage: 2000–2025 · 58 disruption events · 10 industries · 14 major ports")


# ══════════════════════════════════════════════════════════════════════════════
# RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "Risk Predictor":
    st.title("🔮 Disruption Risk Predictor")
    st.markdown(
        "Describe a supply chain disruption scenario and get ML predictions for "
        "**severity**, **freight rate shock**, and **recovery timeline**. "
        "Models are trained on 58 historical events (2001–2025)."
    )

    events = get_events()
    disruption_types = sorted(events["disruption_type"].unique().tolist())
    regions = sorted(events["region_affected"].unique().tolist())

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            disruption_type = st.selectbox("Disruption Type", disruption_types)
            region = st.selectbox("Region Affected", regions)
            duration_days = st.slider("Expected Duration (days)", 1, 730, 60)
            year = st.number_input("Year", min_value=2000, max_value=2030, value=2025, step=1)

        with col2:
            st.markdown("**Risk Characteristics**")
            is_pandemic = st.checkbox("Pandemic-related")
            is_geopolitical = st.checkbox("Geopolitical conflict")
            is_natural = st.checkbox("Natural disaster")
            is_financial = st.checkbox("Financial crisis")
            straits_affected = st.checkbox("Major straits affected (Suez, Panama, Malacca)")
            port_closure = st.checkbox("Port closure expected")

        submitted = st.form_submit_button("Predict Risk", type="primary", use_container_width=True)

    if submitted:
        try:
            severity_clf, freight_reg, recovery_reg, feature_columns = load_models()

            X_input = encode_prediction_input(
                disruption_type, region, int(duration_days), int(year),
                is_pandemic, is_geopolitical, is_natural, is_financial,
                straits_affected, port_closure, feature_columns,
            )

            sev_pred = int(severity_clf.predict(X_input)[0])
            sev_proba = severity_clf.predict_proba(X_input)[0]
            freight_pred = float(freight_reg.predict(X_input)[0])
            recovery_pred = float(recovery_reg.predict(X_input)[0])

            severity_name = SEVERITY_LABELS[sev_pred]
            severity_icon = {0: "🟡", 1: "🟠", 2: "🔴"}[sev_pred]

            st.markdown("---")
            st.subheader("Predictions")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Severity", f"{severity_icon} {severity_name}")
            c2.metric(
                "Freight Rate Shock",
                f"{freight_pred:+.1f}%",
                help="Expected % change in spot container freight rates",
            )
            c3.metric(
                "Recovery Time",
                f"{recovery_pred:.0f} months",
                help="Estimated months until rates/flows normalize",
            )

            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Severity Probabilities")
                prob_df = pd.DataFrame(
                    {"Severity": ["Medium", "High", "Extreme"], "Probability": sev_proba}
                )
                fig = px.bar(
                    prob_df, x="Severity", y="Probability",
                    color="Severity",
                    color_discrete_map={
                        "Medium": "#FFC107", "High": "#FF7F00", "Extreme": "#D32F2F"
                    },
                    text_auto=".1%",
                )
                fig.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                st.subheader("Similar Historical Events")
                sev_label = {0: "medium", 1: "high", 2: "extreme"}[sev_pred]
                similar = (
                    events[events["severity"] == sev_label]
                    .sort_values("freight_rate_shock_pct", key=abs, ascending=False)
                    .head(6)[["year", "event_name", "disruption_type", "freight_rate_shock_pct", "recovery_months"]]
                )
                similar.columns = ["Year", "Event", "Type", "Freight Shock %", "Recovery (mo)"]
                st.dataframe(similar, use_container_width=True, hide_index=True)

            # Feature importance
            st.subheader("Top Predictors (Feature Importance)")
            fi = pd.Series(severity_clf.feature_importances_, index=feature_columns)
            top_fi = fi.nlargest(12).sort_values()
            fig_fi = px.bar(
                x=top_fi.values, y=top_fi.index,
                orientation="h",
                labels={"x": "Importance", "y": "Feature"},
                color=top_fi.values,
                color_continuous_scale="Blues",
            )
            fig_fi.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)

        except FileNotFoundError:
            st.error(
                "⚠️ Models not found. Run `python train.py` from the project directory first, "
                "then refresh this page."
            )


# ══════════════════════════════════════════════════════════════════════════════
# DISRUPTION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Disruption History":
    st.title("📋 Disruption History (2001–2025)")
    events = get_events()

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_type = st.selectbox("Disruption Type", ["All"] + sorted(events["disruption_type"].unique()))
    with col2:
        sel_region = st.selectbox("Region", ["All"] + sorted(events["region_affected"].unique()))
    with col3:
        sel_sev = st.selectbox("Severity", ["All", "medium", "high", "extreme"])

    year_range = st.slider(
        "Year Range",
        int(events["year"].min()), int(events["year"].max()),
        (int(events["year"].min()), int(events["year"].max())),
    )

    df = events.copy()
    df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
    if sel_type != "All":
        df = df[df["disruption_type"] == sel_type]
    if sel_region != "All":
        df = df[df["region_affected"] == sel_region]
    if sel_sev != "All":
        df = df[df["severity"] == sel_sev]

    st.caption(f"{len(df)} events match current filters")

    # Main scatter
    fig_scatter = px.scatter(
        df,
        x="year",
        y="freight_rate_shock_pct",
        size=df["duration_days"].clip(lower=5),
        color="severity",
        color_discrete_map=SEVERITY_COLORS,
        hover_name="event_name",
        hover_data={
            "disruption_type": True,
            "region_affected": True,
            "recovery_months": True,
            "gdp_impact_pct": True,
        },
        labels={"freight_rate_shock_pct": "Freight Rate Shock (%)", "year": "Year"},
        title="Disruption Events — Freight Rate Shock by Year (bubble size = duration)",
        category_orders={"severity": ["medium", "high", "extreme"]},
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    st.plotly_chart(fig_scatter, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        top = df.reindex(df["freight_rate_shock_pct"].abs().sort_values(ascending=False).index).head(12)
        fig_top = px.bar(
            top,
            x="freight_rate_shock_pct",
            y="event_name",
            color="severity",
            color_discrete_map=SEVERITY_COLORS,
            orientation="h",
            title="Top Events by Freight Shock Magnitude",
            labels={"freight_rate_shock_pct": "Freight Shock (%)", "event_name": ""},
            category_orders={"severity": ["medium", "high", "extreme"]},
        )
        fig_top.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig_top, use_container_width=True)

    with col_b:
        type_counts = df.groupby(["disruption_type", "severity"]).size().reset_index(name="count")
        fig_types = px.bar(
            type_counts,
            x="disruption_type",
            y="count",
            color="severity",
            color_discrete_map=SEVERITY_COLORS,
            title="Event Count by Type and Severity",
            labels={"disruption_type": "Type", "count": "Events"},
            category_orders={"severity": ["medium", "high", "extreme"]},
        )
        fig_types.update_xaxes(tickangle=30)
        st.plotly_chart(fig_types, use_container_width=True)

    # Recovery by disruption type
    avg_rec = df.groupby("disruption_type")["recovery_months"].mean().sort_values(ascending=True).reset_index()
    fig_rec = px.bar(
        avg_rec,
        x="recovery_months",
        y="disruption_type",
        orientation="h",
        title="Average Recovery Time by Disruption Type (months)",
        labels={"recovery_months": "Avg Recovery (months)", "disruption_type": ""},
        color="recovery_months",
        color_continuous_scale="Reds",
    )
    fig_rec.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_rec, use_container_width=True)

    # Data table
    st.subheader("Event Details")
    display_cols = {
        "date": "Date", "event_name": "Event", "disruption_type": "Type",
        "region_affected": "Region", "severity": "Severity",
        "duration_days": "Duration (d)", "bdi_shock_pct": "BDI Shock %",
        "freight_rate_shock_pct": "Freight Shock %",
        "gdp_impact_pct": "GDP Impact %", "recovery_months": "Recovery (mo)",
    }
    st.dataframe(
        df[list(display_cols)].rename(columns=display_cols),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# INDUSTRY VULNERABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Industry Vulnerability":
    st.title("🏭 Industry Vulnerability Analysis")
    industry = get_industry()

    exposure_cols = [
        "pandemic_exposure", "geopolitical_exposure", "natural_disaster_exposure",
        "tariff_exposure", "logistics_exposure", "energy_exposure",
        "labor_exposure", "cyber_exposure",
    ]
    col_labels = ["Pandemic", "Geopolitical", "Natural Disaster", "Tariff",
                  "Logistics", "Energy", "Labor", "Cyber"]

    # Year slider for heatmap
    sel_year = st.slider(
        "Select Year for Exposure Heatmap",
        int(industry["year"].min()), int(industry["year"].max()),
        int(industry["year"].max()),
    )

    year_df = (
        industry[industry["year"] == sel_year]
        .set_index("industry")[exposure_cols]
        .rename(columns=dict(zip(exposure_cols, col_labels)))
    )

    fig_heat = px.imshow(
        year_df,
        color_continuous_scale="RdYlGn_r",
        title=f"Industry Exposure Scores — {sel_year}  (scale 0–10)",
        labels={"color": "Score"},
        aspect="auto",
        text_auto=".1f",
    )
    fig_heat.update_coloraxes(cmin=0, cmax=10)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Overall vulnerability trend
    st.subheader("Overall Vulnerability Over Time")
    all_industries = sorted(industry["industry"].unique())
    sel_industries = st.multiselect(
        "Select industries", all_industries, default=all_industries[:5]
    )

    if sel_industries:
        trend_df = industry[industry["industry"].isin(sel_industries)]
        fig_trend = px.line(
            trend_df,
            x="year",
            y="overall_vulnerability",
            color="industry",
            markers=True,
            title="Overall Vulnerability Score by Industry (2000–2024)",
            labels={"overall_vulnerability": "Vulnerability Score", "year": "Year"},
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        latest_year = int(industry["year"].max())
        latest = industry[industry["year"] == latest_year]
        fig_jit = px.scatter(
            latest,
            x="just_in_time_dependency",
            y="inventory_days",
            size="overall_vulnerability",
            color="industry",
            hover_data=["supplier_concentration_hhi", "overall_vulnerability"],
            title=f"JIT Dependency vs Inventory Buffer — {latest_year}",
            labels={
                "just_in_time_dependency": "JIT Dependency (0–1)",
                "inventory_days": "Inventory Buffer (days)",
            },
        )
        st.plotly_chart(fig_jit, use_container_width=True)

    with col_b:
        latest = industry[industry["year"] == latest_year].sort_values("nearshoring_score_2024")
        fig_near = px.bar(
            latest,
            x="nearshoring_score_2024",
            y="industry",
            orientation="h",
            title=f"Nearshoring Progress — {latest_year}",
            labels={"nearshoring_score_2024": "Nearshoring Score (0–1)", "industry": ""},
            color="nearshoring_score_2024",
            color_continuous_scale="Greens",
        )
        fig_near.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_near, use_container_width=True)

    # Concentration risk trend
    st.subheader("Supplier Concentration (HHI) Over Time")
    if sel_industries:
        fig_hhi = px.line(
            industry[industry["industry"].isin(sel_industries)],
            x="year",
            y="supplier_concentration_hhi",
            color="industry",
            markers=True,
            title="Supplier Concentration (HHI) — higher = more concentrated",
            labels={"supplier_concentration_hhi": "HHI", "year": "Year"},
        )
        fig_hhi.add_hline(
            y=0.25, line_dash="dash", line_color="red", opacity=0.5,
            annotation_text="High concentration threshold",
        )
        st.plotly_chart(fig_hhi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SHIPPING MARKETS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Shipping Markets":
    st.title("📈 Shipping Market Indicators (2000–2024)")
    rates = get_rates()

    # Baltic Dry Index
    fig_bdi = go.Figure()
    fig_bdi.add_trace(go.Scatter(
        x=rates["date"], y=rates["baltic_dry_index"],
        name="BDI", line=dict(color="#1f77b4", width=1.5), fill="tozeroy",
        fillcolor="rgba(31,119,180,0.08)",
    ))
    fig_bdi.update_layout(
        title="Baltic Dry Index (2000–2024)",
        xaxis_title="", yaxis_title="BDI Points",
        hovermode="x unified",
    )
    st.plotly_chart(fig_bdi, use_container_width=True)

    # Container spot rate
    fig_cont = go.Figure()
    fig_cont.add_trace(go.Scatter(
        x=rates["date"], y=rates["container_rate_usd_40ft"],
        name="Container Rate", line=dict(color="#ff7f0e", width=1.5), fill="tozeroy",
        fillcolor="rgba(255,127,14,0.08)",
    ))
    fig_cont.update_layout(
        title="Container Spot Rate — USD per 40ft (2000–2024)",
        xaxis_title="", yaxis_title="USD / 40ft",
        hovermode="x unified",
    )
    st.plotly_chart(fig_cont, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_sc = px.area(
            rates, x="date", y="supply_chain_pressure_index",
            title="Global Supply Chain Pressure Index",
            labels={"supply_chain_pressure_index": "Pressure Index", "date": ""},
            color_discrete_sequence=["#d62728"],
        )
        fig_sc.update_layout(showlegend=False)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_b:
        fig_otd = px.line(
            rates, x="date", y="on_time_delivery_pct",
            title="On-Time Delivery Rate",
            labels={"on_time_delivery_pct": "On-Time %", "date": ""},
            color_discrete_sequence=["#2ca02c"],
        )
        fig_otd.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_otd, use_container_width=True)

    # Air cargo
    fig_air = px.line(
        rates, x="date", y="air_cargo_rate_usd_kg",
        title="Air Cargo Rate (USD/kg)",
        labels={"air_cargo_rate_usd_kg": "USD / kg", "date": ""},
        color_discrete_sequence=["#9467bd"],
    )
    st.plotly_chart(fig_air, use_container_width=True)

    # Key stats
    st.subheader("Key Statistics")
    bdi_peak_idx = rates["baltic_dry_index"].idxmax()
    cont_peak_idx = rates["container_rate_usd_40ft"].idxmax()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Peak BDI",
        f"{rates.loc[bdi_peak_idx, 'baltic_dry_index']:,.0f}",
        rates.loc[bdi_peak_idx, "date"].strftime("%b %Y"),
    )
    c2.metric(
        "Peak Container Rate",
        f"${rates.loc[cont_peak_idx, 'container_rate_usd_40ft']:,.0f}",
        rates.loc[cont_peak_idx, "date"].strftime("%b %Y"),
    )
    c3.metric("Latest BDI", f"{rates['baltic_dry_index'].iloc[-1]:,.0f}")
    c4.metric("Latest Container Rate", f"${rates['container_rate_usd_40ft'].iloc[-1]:,.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# PORT CONGESTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Port Congestion":
    st.title("🚢 Port Congestion Monitor (2019–present)")
    ports = get_ports()

    all_ports = sorted(ports["port"].unique().tolist())
    sel_ports = st.multiselect("Select ports", all_ports, default=all_ports[:6])

    if not sel_ports:
        st.info("Select at least one port above.")
        st.stop()

    df_port = ports[ports["port"].isin(sel_ports)]

    # Congestion index
    fig_cong = px.line(
        df_port, x="week_start", y="congestion_index",
        color="port",
        title="Port Congestion Index Over Time",
        labels={"congestion_index": "Congestion Index", "week_start": ""},
    )
    fig_cong.add_hline(
        y=1.0, line_dash="dash", line_color="gray", opacity=0.5,
        annotation_text="Baseline (1.0)",
    )
    st.plotly_chart(fig_cong, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_wait = px.line(
            df_port, x="week_start", y="avg_wait_days",
            color="port",
            title="Average Vessel Wait Days",
            labels={"avg_wait_days": "Wait Days", "week_start": ""},
        )
        st.plotly_chart(fig_wait, use_container_width=True)

    with col_b:
        fig_anchor = px.line(
            df_port, x="week_start", y="vessels_at_anchor",
            color="port",
            title="Vessels at Anchor",
            labels={"vessels_at_anchor": "Vessels", "week_start": ""},
        )
        st.plotly_chart(fig_anchor, use_container_width=True)

    # Latest snapshot
    latest_week = ports["week_start"].max()
    latest_snap = ports[(ports["week_start"] == latest_week) & (ports["port"].isin(sel_ports))]

    fig_snap = px.bar(
        latest_snap.sort_values("congestion_index"),
        x="congestion_index",
        y="port",
        orientation="h",
        color="congestion_index",
        color_continuous_scale="RdYlGn_r",
        title=f"Congestion Snapshot — {latest_week.strftime('%Y-%m-%d')}",
        labels={"congestion_index": "Congestion Index", "port": ""},
        text_auto=".2f",
    )
    fig_snap.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_snap.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_snap, use_container_width=True)

    # Summary table
    st.subheader("Port Statistics Summary")
    stats = (
        ports[ports["port"].isin(sel_ports)]
        .groupby("port")
        .agg(
            avg_congestion=("congestion_index", "mean"),
            peak_congestion=("congestion_index", "max"),
            avg_wait_days=("avg_wait_days", "mean"),
            peak_vessels=("vessels_at_anchor", "max"),
            avg_utilization=("port_utilization_pct", "mean"),
        )
        .round(2)
        .reset_index()
    )
    stats.columns = [
        "Port", "Avg Congestion", "Peak Congestion",
        "Avg Wait Days", "Peak Vessels at Anchor", "Avg Utilization",
    ]
    st.dataframe(stats, use_container_width=True, hide_index=True)
