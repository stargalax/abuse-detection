import pandas as pd
import time

def to_native_pandas(data):
    """Fix narwhals/Altair issues by forcing pure pandas."""
    if hasattr(data, "to_native"):
        data = data.to_native()
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(data.to_dict())
    if isinstance(data, pd.Series):
        return pd.Series(data.to_dict())
    return data


class Timer:
    """Simple context manager for measuring execution time."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.duration = time.time() - self.start
