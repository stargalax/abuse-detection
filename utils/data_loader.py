import pandas as pd
from .helpers import to_native_pandas

def load_dataset(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if hasattr(df, "to_native"):
        df = df.to_native()
    return df
