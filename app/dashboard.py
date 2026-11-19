import pandas as pd
import plotly.express as px
import streamlit as st
from pyspark.sql import SparkSession

st.set_page_config(page_title="NYC Taxi Lakehouse — Gold BI", layout="wide")
st.title("NYC Taxi Lakehouse — Restitution BI (Couche Gold)")
st.caption("Lecture exclusive des tables Delta pré-agrégées dans `/lakehouse/gold/`")

BASE_GOLD = "hdfs://namenode:9000/lakehouse/gold"
GOLD_PARETO_PATH = f"{BASE_GOLD}/nyc_taxi_gold_pareto"
GOLD_TIME_PATH = f"{BASE_GOLD}/nyc_taxi_gold_time"
GOLD_SEGMENTS_PATH = f"{BASE_GOLD}/nyc_taxi_gold_segments"


@st.cache_resource
def get_spark():
    return (
        SparkSession.builder
        .appName("Streamlit_Gold_Dashboard")
        .master("local[*]")
        .config("spark.driver.memory", "1536m")
        .config("spark.executor.memory", "1536m")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


@st.cache_data(ttl=300)
def load_delta_pdf(path: str) -> pd.DataFrame:
    spark = get_spark()
    return spark.read.format("delta").load(path).toPandas()


try:
    df_pareto = load_delta_pdf(GOLD_PARETO_PATH)
    df_time = load_delta_pdf(GOLD_TIME_PATH)
    df_seg = load_delta_pdf(GOLD_SEGMENTS_PATH)
except Exception as exc:
    st.error(f"Erreur de lecture des tables Gold sur HDFS : {exc}")
    st.stop()

# --- Filtres interactifs ---
st.sidebar.header("Filtres interactifs")

payment_options = sorted(df_pareto["payment_type_name"].dropna().unique().tolist())
selected_payments = st.sidebar.multiselect(
    "Modes de paiement (Pareto)",
    options=payment_options,
    default=payment_options,
)

segment_options = ["Tous"] + sorted(df_seg["segment"].dropna().unique().tolist())
selected_segment = st.sidebar.selectbox(
    "Segment K-Means",
    options=segment_options,
    index=0,
)

pareto_filtered = (
    df_pareto[df_pareto["payment_type_name"].isin(selected_payments)]
    if selected_payments
    else df_pareto
)
seg_filtered = (
    df_seg[df_seg["segment"] == selected_segment]
    if selected_segment != "Tous"
    else df_seg
)

# --- Visualisation 1 : Pareto ---
st.subheader("1. Classement Métier & Courbe de Pareto par mode de paiement")
fig_pareto = px.bar(
    pareto_filtered.sort_values("total_revenue", ascending=False),
    x="payment_type_name",
    y="total_revenue",
    text_auto=".2s",
    hover_data=["volume_trips", "median_revenue", "iqr_revenue", "cumulated_pareto_ratio"],
    title="Revenu total et indicateurs robustes par mode de paiement",
)
st.plotly_chart(fig_pareto, use_container_width=True)

with st.expander("Requête SQL & Justification statistique"):
    st.code(
        """
WITH payment_stats AS (
    SELECT
        payment_type_name,
        COUNT(*) AS volume_trips,
        SUM(fare_amount) AS total_revenue,
        PERCENTILE_APPROX(fare_amount, 0.50) AS median_revenue,
        (PERCENTILE_APPROX(fare_amount, 0.75) - PERCENTILE_APPROX(fare_amount, 0.25)) AS iqr_revenue
    FROM v_silver
    WHERE payment_type_name IS NOT NULL
    GROUP BY payment_type_name
)
SELECT *,
       DENSE_RANK() OVER (ORDER BY total_revenue DESC) AS rank_revenue,
       SUM(total_revenue) OVER (ORDER BY total_revenue DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
         / SUM(total_revenue) OVER () AS cumulated_pareto_ratio
FROM payment_stats;
        """,
        language="sql",
    )
    st.write(
        "La distribution des montants étant asymétrique à droite, la médiane (`P50`) et l'écart interquartile (`IQR`) "
        "résument la course courante et la dispersion centrale sans être tirés par les valeurs extrêmes, "
        "tandis que le cumul de Pareto identifie les catégories concentrant l'essentiel du revenu."
    )

# --- Visualisation 2 : Série temporelle & LAG ---
st.subheader("2. Dynamique temporelle & Évolution période-sur-période (`LAG`)")
fig_time = px.line(
    df_time.sort_values("trip_month"),
    x="trip_month",
    y="total_revenue",
    markers=True,
    hover_data=["total_trips", "avg_fare", "prev_month_revenue", "growth_rate_pct"],
    title="Évolution du revenu total par période",
)
st.plotly_chart(fig_time, use_container_width=True)

with st.expander("Requête SQL & Justification statistique"):
    st.code(
        """
WITH monthly_revenue AS (
    SELECT
        trip_month,
        COUNT(*) AS total_trips,
        SUM(fare_amount) AS total_revenue,
        AVG(fare_amount) AS avg_fare
    FROM v_silver
    WHERE trip_month IS NOT NULL
    GROUP BY trip_month
)
SELECT *,
       LAG(total_revenue, 1) OVER (ORDER BY trip_month) AS prev_month_revenue,
       ROUND(((total_revenue - LAG(total_revenue, 1) OVER (ORDER BY trip_month))
              / LAG(total_revenue, 1) OVER (ORDER BY trip_month)) * 100, 2) AS growth_rate_pct
FROM monthly_revenue
ORDER BY trip_month;
        """,
        language="sql",
    )
    st.write(
        "La fonction `LAG(total_revenue, 1)` calcule directement la variation par rapport à la période précédente "
        "sans jointure coûteuse, tout en croisant le volume (`total_trips`) pour contextualiser les variations."
    )

# --- Visualisation 3 : Segmentation K-Means ---
st.subheader("3. Typologie / Segmentation K-Means")
fig_seg = px.scatter(
    seg_filtered,
    x="trip_count",
    y="total_fare",
    color="segment",
    size="median_fare",
    hover_data=["trip_month", "avg_distance", "cluster_id"],
    title="Segmentation des entités selon le volume et le revenu",
)
st.plotly_chart(fig_seg, use_container_width=True)

with st.expander("Requête SQL & Justification statistique"):
    st.code(
        """
SELECT
    trip_month,
    COUNT(*) AS trip_count,
    SUM(fare_amount) AS total_fare,
    PERCENTILE_APPROX(fare_amount, 0.5) AS median_fare,
    AVG(trip_distance) AS avg_distance
FROM v_silver
WHERE trip_month IS NOT NULL
GROUP BY trip_month;
        """,
        language="sql",
    )
    st.write(
        "L'application de `np.log1p` puis d'un `StandardScaler` réduit l'asymétrie des variables positives "
        "et équilibre les amplitudes avant le clustering euclidien K-Means."
    )
