# import streamlit as st
# import pandas as pd
# import sqlite3
# import matplotlib.pyplot as plt
# import seaborn as sns
# import altair as alt

# st.set_page_config(layout="wide", page_title="YouTube Abuse Detection Dashboard")

# # IMPORTANT: Display upgrade message prominently
# st.sidebar.markdown("---")
# st.sidebar.markdown("### 🔧 System Info")
# # try:
# #     import altair as alt
# #     altair_version = alt.__version__
# #     major, minor = map(int, altair_version.split('.')[:2])
# #     if major < 5 or (major == 5 and minor < 5):
# #         st.sidebar.error(f"⚠️ Altair {altair_version} detected. Upgrade required!")
# #         st.sidebar.code("pip install --upgrade altair>=5.5.0", language="bash")
# #     else:
# #         st.sidebar.success(f"✅ Altair {altair_version}")
# # except:
# #     pass
# # st.sidebar.markdown("---")

# # Check Altair version and warn user if needed
# # try:
# #     altair_version = alt.__version__
# #     major, minor = map(int, altair_version.split('.')[:2])
# #     if major < 5 or (major == 5 and minor < 5):
# #         st.warning(f"⚠️ You're using Altair {altair_version}. Please upgrade to 5.5.0+ to fix chart rendering issues: `pip install --upgrade altair`")
# # except:
# #     pass

# # Helper function to ensure native pandas (workaround for Altair < 5.5.0)
# def to_native_pandas(data):
#     """Convert any dataframe to native pandas, avoiding narwhals issues"""
#     if hasattr(data, 'to_native'):
#         data = data.to_native()
#     # Force re-creation to ensure it's truly native
#     if isinstance(data, pd.DataFrame):
#         return pd.DataFrame(data.to_dict())
#     elif isinstance(data, pd.Series):
#         return pd.Series(data.to_dict())
#     return data

# # -------------------------
# # 1️⃣ Load dataset
# # -------------------------
# df = pd.read_csv("youtube_abuse_dataset.csv", parse_dates=["timestamp"])

# # CRITICAL: Convert to native pandas immediately to avoid narwhals issues
# if hasattr(df, 'to_native'):
#     df = df.to_native()

# # -------------------------
# # 2️⃣ SQLite in-memory
# # -------------------------
# conn = sqlite3.connect(":memory:")
# df.to_sql("events", conn, index=False, if_exists='replace')

# # -------------------------
# # 3️⃣ Sidebar filters
# # -------------------------
# st.sidebar.header("Filter Parameters")
# min_uploads = st.sidebar.slider("Minimum uploads", 1, 50, 5)
# min_spam_ratio = st.sidebar.slider("Minimum spam ratio", 0.0, 1.0, 0.4)
# min_devices = st.sidebar.slider("Minimum devices", 1, 10, 3)
# country_filter = st.sidebar.multiselect(
#     "Filter by country",
#     options=df['geo_location'].unique(),
#     default=df['geo_location'].unique()
# )

# # -------------------------
# # 4️⃣ SQL query: suspicious users
# # -------------------------
# query = f"""
# SELECT user_id,
#        COUNT(*) AS total_uploads,
#        SUM(spam_flag) AS total_spam_comments,
#        COUNT(DISTINCT device_id) AS num_devices,
#        SUM(geo_location != (SELECT geo_location
#                             FROM events e2
#                             WHERE e2.user_id = e1.user_id
#                             GROUP BY e2.user_id
#                             ORDER BY COUNT(*) DESC
#                             LIMIT 1)) AS num_geo_anomalies
# FROM events e1
# WHERE geo_location IN ({','.join(['?']*len(country_filter))})
# GROUP BY user_id
# HAVING total_uploads >= ? AND num_devices >= ? AND (total_spam_comments*1.0/total_uploads) >= ?;
# """

# params = country_filter + [min_uploads, min_devices, min_spam_ratio]
# filtered_users = pd.read_sql(query, conn, params=params)
# import time
# start = time.time()

# filtered_users = pd.read_sql(query, conn, params=params)

# duration = time.time() - start
# st.sidebar.write(f"⏱ SQL Query Time: {duration:.3f} seconds")


# # Robustness fix: Ensure filtered_users is a native DataFrame before manipulation
# if hasattr(filtered_users, 'to_native'):
#     filtered_users = filtered_users.to_native()

# # -------------------------
# # 5️⃣ Risk score (Rule-based detection)
# # -------------------------
# filtered_users['risk_score'] = (
#     filtered_users['total_uploads']*0.25 +
#     filtered_users['total_spam_comments']*0.35 +
#     filtered_users['num_devices']*0.2 +
#     filtered_users['num_geo_anomalies']*0.2
# )

# # Sort by risk score
# filtered_users = filtered_users.sort_values(by='risk_score', ascending=False)

# # -------------------------
# # 7️⃣ Tabbed Interface
# # -------------------------
# tab1, tab2, tab3,tab4 = st.tabs(["Dataset Info", "Suspicious Users", "Pattern Analysis","Statistical Analysis"])

# # ============================================
# # TAB 1: Dataset Info
# # ============================================
# with tab1:
#     st.header("Dataset Overview")
    
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Total Records", f"{len(df):,}")
#     with col2:
#         st.metric("Unique Users", f"{df['user_id'].nunique():,}")
#     with col3:
#         st.metric("Unique Devices", f"{df['device_id'].nunique():,}")
#     with col4:
#         st.metric("Countries", f"{df['geo_location'].nunique()}")
    
#     st.subheader("📋 Schema & Data Types")
#     schema_df = pd.DataFrame({
#         'Column': df.columns,
#         'Data Type': df.dtypes.values,
#         'Non-Null Count': df.count().values,
#         'Null Count': df.isnull().sum().values,
#         'Unique Values': [df[col].nunique() for col in df.columns]
#     })
#     st.dataframe(schema_df, use_container_width=True)
    
#     st.subheader("🔗 Data Relationships")
#     st.write("""
#     **Primary Entities:**
#     - `user_id` → Unique identifier for each user
#     - `device_id` → Device used for upload
#     - `geo_location` → Country/region of upload
    
#     **Relationships:**
#     - **One-to-Many**: user_id → events (one user can have multiple upload events)
#     - **Many-to-Many**: user_id ↔ device_id (users can use multiple devices)
#     - **Many-to-Many**: user_id ↔ geo_location (users can upload from multiple locations)
#     """)
    
#     st.subheader("📊 Sample Data")
#     st.dataframe(df.head(20), use_container_width=True)
    
#     st.subheader("📈 Quick Statistics")
#     stats_col1, stats_col2 = st.columns(2)
    
#     with stats_col1:
#         st.write("**Upload Rate Statistics**")
#         st.write(df['upload_rate'].describe())
        
#         st.write("**Spam Flag Distribution**")
#         spam_dist = df['spam_flag'].value_counts()
#         # Create native pandas DataFrame - force dict conversion for Altair compatibility
#         spam_data = {
#             'Spam': ['No', 'Yes'],
#             'Count': [int(spam_dist.get(0, 0)), int(spam_dist.get(1, 0))]
#         }
#         spam_chart_df = pd.DataFrame(spam_data).set_index('Spam')
#         spam_chart_df = to_native_pandas(spam_chart_df)
#         st.bar_chart(spam_chart_df)
    
#     with stats_col2:
#         st.write("**Top 10 Countries by Upload Activity**")
#         country_counts = df['geo_location'].value_counts().head(10)
#         # Convert to native pandas with dict for Altair compatibility
#         country_chart_df = pd.DataFrame({
#             'count': country_counts.values
#         }, index=country_counts.index)
#         country_chart_df = to_native_pandas(country_chart_df)
#         st.bar_chart(country_chart_df)
        
#         st.write("**User Activity Distribution**")
#         user_uploads = df.groupby('user_id').size()
#         fig, ax = plt.subplots(figsize=(8, 4))
#         ax.hist(user_uploads, bins=30, edgecolor='black', alpha=0.7)
#         ax.set_xlabel('Number of Uploads per User')
#         ax.set_ylabel('Frequency')
#         ax.set_title('User Upload Frequency Distribution')
#         st.pyplot(fig)

# # ============================================
# # TAB 2: Suspicious Users
# # ============================================
# with tab2:
#     st.header("🚨 Suspicious Users Analysis")
    
#     if len(filtered_users) == 0:
#         st.warning("⚠️ No users match your filter criteria. Try relaxing the filters in the sidebar.")
#     else:
#         # Metrics
#         metric_col1, metric_col2, metric_col3 = st.columns(3)
#         with metric_col1:
#             st.metric("Total Suspicious Users", len(filtered_users))
#         with metric_col2:
#             st.metric("High-Risk Users (Score > 50)", len(filtered_users[filtered_users['risk_score'] > 50]))
#         with metric_col3:
#             avg_spam = (filtered_users['total_spam_comments'].sum() / filtered_users['total_uploads'].sum() * 100)
#             st.metric("Avg Spam Rate", f"{avg_spam:.1f}%")
        
#         st.subheader("Top 20 Most Suspicious Users")
#         st.dataframe(filtered_users.head(20), use_container_width=True)
        
#         # Risk score distribution
#         st.subheader("Risk Score Distribution")
#         col1, col2 = st.columns([2, 1])
        
#         with col1:
#             fig, ax = plt.subplots(figsize=(10, 4))
#             ax.hist(filtered_users['risk_score'], bins=30, edgecolor='black', alpha=0.7, color='crimson')
#             ax.set_xlabel('Risk Score')
#             ax.set_ylabel('Number of Users')
#             ax.set_title('Distribution of Risk Scores')
#             st.pyplot(fig)
        
#         with col2:
#             st.write("**Risk Scoring Formula:**")
#             st.write("""
#             - Total Uploads × 0.25
#             - Spam Comments × 0.35
#             - Multiple Devices × 0.20
#             - Geo Anomalies × 0.20
            
#             **Interpretation:**
#             - 0-20: Low risk
#             - 20-50: Medium risk
#             - 50+: High risk
#             """)
#     st.download_button(
#     "📥 Download Suspicious Users CSV",
#     filtered_users.to_csv(index=False),
#     "suspicious_users.csv",
#     "text/csv"
# )

# # ============================================
# # TAB 3: Pattern Analysis
# # ============================================
# with tab3:
#     if len(filtered_users) == 0:
#         st.warning("⚠️ No users match your filter criteria. Try relaxing the filters in the sidebar.")
#     else:
#         st.header("📈 Behavioral Patterns")
        
#         # Upload trends over time
#         st.subheader("Upload Activity Over Time")
        
#         df_native = df.copy()
#         if hasattr(df_native, 'to_native'):
#             df_native = df_native.to_native()

#         df_native['date'] = pd.to_datetime(df_native['timestamp']).dt.date
#         daily_uploads = df_native.groupby('date')['upload_rate'].sum().reset_index()

#         daily_uploads_df = pd.DataFrame({
#             'date': pd.to_datetime(daily_uploads['date']), 
#             'upload_rate': daily_uploads['upload_rate'].values
#         }).set_index('date')
#         daily_uploads_df = to_native_pandas(daily_uploads_df)

#         st.line_chart(daily_uploads_df)
#         st.caption("Shows total upload activity across all users over time. Spikes may indicate coordinated abuse campaigns.")

#         # Spam patterns
#         st.subheader("Spam Comment Patterns")
#         col1, col2 = st.columns(2)
        
#         with col1:
#             spam_ratio_values = (filtered_users['total_spam_comments'] / filtered_users['total_uploads']).values
#             user_id_values = filtered_users['user_id'].values

#             spam_ratio_df = pd.DataFrame({
#                 'spam_ratio': spam_ratio_values
#             }, index=user_id_values)
#             spam_ratio_df = to_native_pandas(spam_ratio_df)

#             st.bar_chart(spam_ratio_df.head(20))
#             st.caption("Top 20 users by spam ratio")
        
#         with col2:
#             fig, ax = plt.subplots(figsize=(6, 4))
#             ax.scatter(filtered_users['total_uploads'], 
#                       filtered_users['total_spam_comments'], 
#                       alpha=0.6, color='crimson')
#             ax.set_xlabel('Total Uploads')
#             ax.set_ylabel('Total Spam Comments')
#             ax.set_title('Uploads vs Spam Comments')
#             st.pyplot(fig)
#             st.caption("Correlation between upload volume and spam activity")

#         # Device usage patterns
#         st.subheader("Multi-Device Usage Analysis")
#         col1, col2 = st.columns(2)
        
#         with col1:
#             fig2, ax2 = plt.subplots(figsize=(8,4))
#             sns.histplot(filtered_users['num_devices'], bins=10, ax=ax2, color='steelblue')
#             ax2.set_xlabel('Number of Devices Used')
#             ax2.set_ylabel('Number of Users')
#             ax2.set_title('Device Switching Behavior')
#             st.pyplot(fig2)
        
#         with col2:
#             st.write("**Why Multi-Device Matters:**")
#             st.write("""
#             Legitimate users typically use 1-2 devices (phone + computer).
            
#             Abusers often:
#             - Switch devices to evade bans
#             - Use device farms (automated abuse)
#             - Share accounts across multiple people
            
#             **Red flags:**
#             - 5+ devices = Highly suspicious
#             - 10+ devices = Almost certainly abuse
#             """)

#         # Geo-location patterns
#         st.subheader("Geographic Anomalies")
#         geo_anomaly_users = filtered_users[filtered_users['num_geo_anomalies'] > 0]
        
#         col1, col2 = st.columns([1, 1])
#         with col1:
#             if len(geo_anomaly_users) == 0:
#                 st.info("✅ No suspicious geo-location patterns detected.")
#             else:
#                 st.metric("Users with Geo Anomalies", len(geo_anomaly_users))
#                 st.write(f"**{len(geo_anomaly_users)/len(filtered_users)*100:.1f}%** of suspicious users upload from multiple countries")
        
#         with col2:
#             if len(geo_anomaly_users) > 0:
#                 fig3, ax3 = plt.subplots(figsize=(6, 4))
#                 ax3.hist(geo_anomaly_users['num_geo_anomalies'], bins=15, color='orange', edgecolor='black')
#                 ax3.set_xlabel('Number of Different Locations')
#                 ax3.set_ylabel('Number of Users')
#                 ax3.set_title('Location Switching Frequency')
#                 st.pyplot(fig3)
# #tab4
# with tab4:
#     st.header("📉 Statistical Analysis")

#     st.subheader("Correlation Matrix")

#     numeric_cols = [
#         "comment_length", 
#         "spam_flag", 
#         "upload_rate", 
#         "reports", 
#         "abuse_score", 
#         "is_abusive"
#     ]

#     corr = df[numeric_cols].corr()

#     fig, ax = plt.subplots(figsize=(6,4))
#     sns.heatmap(corr, annot=True, cmap="coolwarm")
#     st.pyplot(fig)
#     from scipy.stats import ttest_ind

#     normal_df = df[~df["user_id"].isin(filtered_users["user_id"])].copy()
#     normal_df["spam_ratio"] = normal_df["spam_flag"]

#     susp_df = df[df["user_id"].isin(filtered_users["user_id"])].copy()
#     susp_df["spam_ratio"] = susp_df["spam_flag"]

#     t_stat, p_val = ttest_ind(
#         normal_df["spam_ratio"],
#         susp_df["spam_ratio"],
#         equal_var=False
#     )

#     st.subheader("Spam Ratio Significance Test")
#     st.write(f"T-statistic: {t_stat:.3f}")
#     st.write(f"P-value: {p_val:.6f}")
#     if p_val < 0.05:
#         st.success("Significant difference! Suspicious users behave differently.")
#     else:
#         st.warning("No significant difference detected.")
#     from scipy.stats import sem, t

#     mean = susp_df["abuse_score"].mean()
#     standard_error = sem(susp_df["abuse_score"])
#     ci_low, ci_high = t.interval(
#         0.95, len(susp_df)-1, loc=mean, scale=standard_error
#     )

#     st.subheader("95% Confidence Interval: Abuse Score")
#     st.write(f"{ci_low:.3f} → {ci_high:.3f}")


# # -------------------------
# # Download CSV
# # -------------------------
# # st.sidebar.markdown("---")
# # st.download_button(
# #     "📥 Download Suspicious Users CSV",
# #     filtered_users.to_csv(index=False),
# #     "suspicious_users.csv",
# #     "text/csv"
# # )
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
