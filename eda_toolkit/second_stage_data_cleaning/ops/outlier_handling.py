import pandas as pd
import numpy as np
from datetime import datetime

def drop_outliers(df: pd.DataFrame, col: str, log_dict: dict) -> tuple:
    lq = np.percentile(df[col], 25)
    uq = np.percentile(df[col], 75)
    iqr = uq-lq
    lt = lq - (1.5*iqr)
    ut = uq + (1.5*iqr)

    overall_mask = (df[col] >= lt) & (df[col] <= ut)
    log_dict[datetime.now()] = {'action': 'drop_outliers', 'col': col, 'rows_affected': len(df) - overall_mask.sum()}

    return df.loc[overall_mask], log_dict

def cap_outliers(df: pd.DataFrame, col: str, log_dict: dict) -> tuple:
    lq = np.percentile(df[col], 25)
    uq = np.percentile(df[col], 75)
    iqr = uq-lq
    lt = lq - (1.5*iqr)
    ut = uq + (1.5*iqr)
    overall_mask = (df[col] >= lt) & (df[col] <= ut)
    log_dict[datetime.now()] = {'action': 'cap_outliers', 'col': col, 'rows_affected': len(df) - overall_mask.sum()}

    df[col] = df[col].clip(lower = lt, upper = ut)

    return df, log_dict

def log_transform_col(df: pd.DataFrame, col: str, log_dict: dict) -> tuple:
    df[col] = np.log1p(df[col])
    log_dict[datetime.now()] = {'action': 'cap_outliers', 'col': col, 'rows_affected': len(df)}
    
    return df, log_dict