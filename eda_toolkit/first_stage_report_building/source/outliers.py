import pandas as pd
import numpy as np

def check_outliers_overall(df: pd.DataFrame, col: str) -> dict:
    lq = np.percentile(df[col], 25)
    uq = np.percentile(df[col], 75)
    iqr = uq-lq
    lt = lq - (1.5*iqr)
    ut = uq + (1.5*iqr)

    lower_mask = df[col] < lt
    upper_mask = df[col] > ut
    overall_mask = (df[col] < lt) | (df[col] > ut)

    perc_under = lower_mask.sum()/len(df)
    perc_over = upper_mask.sum()/len(df)
    perc_all = overall_mask.sum()/len(df)

    perc_dict = {'proportion_under_threshold': float(perc_under),
                 'proportion_over_threshold': float(perc_over),
                 'proportion_anomalous': float(perc_all)}

    return perc_dict

def check_outliers_in_group(df: pd.DataFrame, target_col: str, group_col: str) -> dict:
    grouping_items = list(df[group_col].unique())
    outlier_warnings = {}
    for item in grouping_items:
        df_check = df.loc[df[group_col] == item]

        perc_outliers = check_outliers_overall(df_check, target_col)
        if perc_outliers['percentage_anomalous'] > 0.5:
            outlier_warnings[(target_col, item)] = perc_outliers['percentage_anomalous']

    return outlier_warnings