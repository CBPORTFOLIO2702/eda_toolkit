import pandas as pd
import numpy as np
import scipy

def validate_date(df, date_col) -> tuple:
    df[date_col] = pd.datetime(df[date_col], errors = 'coerce')
    perc_error = float(df.isna().sum()/len(df))

    return df, perc_error

def get_missing_perc(df: pd.DataFrame) -> tuple:
    col_missing_perc = {}
    missing_alarm_list = []
    for col in list(df.columns):
        col_missing_perc[col] = float(df[col].isna().sum()/len(df))
        if df[col].isna().sum()/len(df) > 0.3:
            missing_alarm_list.append(col)

    return col_missing_perc, missing_alarm_list

def test_mar(df: pd.DataFrame, missing_col_list: list) -> tuple:
    mar_numerical = {}
    mar_categorical = {}
    for col_target in missing_col_list:
        df[f'is_{col_target}_null'] = df[col_target].isna().astype(int)
        for col_other in list(df.columns):
            if col_target == col_other:
                continue
            elif df[col_other].dtype in (int, float):
                p_val = scipy.stats.pointbiserialr(df[f'is_{col_target}_null'], df[col_other]).pvalue
                if p_val < 0.05:
                    mar_numerical[(col_target, col_other)] = p_val
            else:
                crosstab = pd.crosstab(df[col_other], df[f'is_{col_target}_null'])
                p_val = scipy.stats.chi2_contingency(crosstab)[1]
                if p_val < 0.05:
                    mar_categorical[(col_target, col_other)] = p_val

    return mar_numerical, mar_categorical

