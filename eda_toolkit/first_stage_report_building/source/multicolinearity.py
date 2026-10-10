import pandas as pd
import numpy as np
import scipy

def test_multicollinearity(df: pd.DataFrame, numeric_cols: list, target_col: str, corr_threshold: float = 0.8) -> dict:
    df_numeric = df.drop(columns=[target_col])
    df_numeric = df_numeric[numeric_cols]

    df_numeric = df_numeric.dropna()

    corr_mat = str(df_numeric.corr())

    return corr_mat