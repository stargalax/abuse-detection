import sqlite3
import pandas as pd

def create_connection(df):
    conn = sqlite3.connect(":memory:")
    df.to_sql("events", conn, index=False, if_exists="replace")
    return conn


def get_suspicious_users(conn, min_uploads, min_spam_ratio, min_devices, country_filter):
    query = f"""
    SELECT user_id,
           COUNT(*) AS total_uploads,
           SUM(spam_flag) AS total_spam_comments,
           COUNT(DISTINCT device_id) AS num_devices,
           SUM(geo_location != (
                SELECT geo_location
                FROM events e2
                WHERE e2.user_id = e1.user_id
                GROUP BY e2.user_id
                ORDER BY COUNT(*) DESC
                LIMIT 1
           )) AS num_geo_anomalies
    FROM events e1
    WHERE geo_location IN ({','.join(['?'] * len(country_filter))})
    GROUP BY user_id
    HAVING total_uploads >= ?
       AND num_devices >= ?
       AND (total_spam_comments * 1.0 / total_uploads) >= ?;
    """

    params = country_filter + [min_uploads, min_devices, min_spam_ratio]

    return pd.read_sql(query, conn, params=params)
