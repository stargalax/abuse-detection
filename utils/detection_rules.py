def compute_risk_score(df):
    df["risk_score"] = (
        df["total_uploads"] * 0.25 +
        df["total_spam_comments"] * 0.35 +
        df["num_devices"] * 0.20 +
        df["num_geo_anomalies"] * 0.20
    )
    return df.sort_values("risk_score", ascending=False)
