
import streamlit as st
import pandas as pd

from utils.data_loader import load_dataset
from utils.sql_queries import create_connection, get_suspicious_users
from utils.detection_rules import compute_risk_score
from utils.stats_tools import correlation_matrix, spam_ttest, confidence_interval
from utils.charts import corr_heatmap
from utils.helpers import to_native_pandas, Timer


st.set_page_config(page_title="YouTube Abuse Detection Dashboard", layout="wide")

# --------------------------------------
# LOAD DATA
# --------------------------------------
df = load_dataset("youtube_abuse_dataset.csv")
conn = create_connection(df)

# --------------------------------------
# SIDEBAR FILTERS
# --------------------------------------
st.sidebar.header("Filter Parameters")

min_uploads = st.sidebar.slider("Minimum uploads", 1, 50, 5)
min_spam_ratio = st.sidebar.slider("Minimum spam ratio", 0.0, 1.0, 0.4)
min_devices = st.sidebar.slider("Minimum devices", 1, 10, 3)
country_filter = st.sidebar.multiselect(
    "Filter by country",
    options=df["geo_location"].unique(),
    default=df["geo_location"].unique()
)

# --------------------------------------
# QUERY DATABASE
# --------------------------------------
with Timer() as t:
    filtered_users = get_suspicious_users(
        conn,
        min_uploads=min_uploads,
        min_spam_ratio=min_spam_ratio,
        min_devices=min_devices,
        country_filter=country_filter
    )

st.sidebar.write(f"⏱ SQL Query Time: {t.duration:.3f} seconds")

if hasattr(filtered_users, "to_native"):
    filtered_users = filtered_users.to_native()

filtered_users = compute_risk_score(filtered_users)

# --------------------------------------
# UI TABS
# --------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset Info", "Suspicious Users", "Pattern Analysis", "Statistical Analysis"
])

# --------------------------------------
# TAB 1: DATASET INFO
# --------------------------------------
with tab1:
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Unique Users", df["user_id"].nunique())
    col3.metric("Unique Devices", df["device_id"].nunique())
    col4.metric("Countries", df["geo_location"].nunique())

    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

# --------------------------------------
# TAB 2: SUSPICIOUS USERS
# --------------------------------------
with tab2:
    st.header("🚨 Suspicious Users")

    if len(filtered_users) == 0:
        st.warning("⚠️ No suspicious users found.")
    else:
        st.dataframe(filtered_users.head(20), use_container_width=True)

        st.download_button(
            "📥 Download Suspicious Users CSV",
            filtered_users.to_csv(index=False),
            "suspicious_users.csv",
            "text/csv"
        )

# --------------------------------------
# TAB 3: PATTERN ANALYSIS
# --------------------------------------
with tab3:
    st.header("📈 Behavioral Patterns")

    df_plot = df.copy()
    df_plot["date"] = df_plot["timestamp"].dt.date

    daily = df_plot.groupby("date")["upload_rate"].sum()
    st.line_chart(daily)

# --------------------------------------
# TAB 4: STATISTICAL ANALYSIS
# --------------------------------------
with tab4:
    st.header("📊 Statistical Analysis")

    corr = correlation_matrix(df)
    st.pyplot(corr_heatmap(corr))

    t_stat, p_val = spam_ttest(df, filtered_users)
    st.write(f"**T-statistic:** {t_stat:.3f}")
    st.write(f"**P-value:** {p_val:.6f}")

# get CI on raw abuse_score for suspicious users
    susp_raw = df[df["user_id"].isin(filtered_users["user_id"])]
    ci_low, ci_high = confidence_interval(susp_raw["abuse_score"])

    st.write(f"**95% CI for Abuse Score:** {ci_low:.3f} → {ci_high:.3f}")
