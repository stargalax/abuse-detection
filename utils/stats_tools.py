from scipy.stats import ttest_ind, sem, t

def correlation_matrix(df):
    numeric_cols = [
        "comment_length",
        "spam_flag",
        "upload_rate",
        "reports",
        "abuse_score",
        "is_abusive",
    ]
    return df[numeric_cols].corr()

def spam_ttest(df, susp_users):
    normal = df[~df["user_id"].isin(susp_users["user_id"])]
    susp = df[df["user_id"].isin(susp_users["user_id"])]
    return ttest_ind(normal["spam_flag"], susp["spam_flag"], equal_var=False)

def confidence_interval(series):
    mean = series.mean()
    standard_error = sem(series)
    ci_low, ci_high = t.interval(0.95, len(series)-1, loc=mean, scale=standard_error)
    return ci_low, ci_high
