import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def test_leakage_binary(df: pd.DataFrame, target: str) -> tuple:
    leak_numerical = {}
    leak_categorical = {}
    for col_other in list(df.columns):
        mask = df[col_other].notna() & df[target].notna()
        if mask.sum() < 2:
            continue
        if target == col_other:
            continue
        elif df[col_other].dtype in (int, float):
            df_masked = df.loc[mask]
            r, p_val = scipy.stats.pointbiserialr(df_masked[target], df_masked[col_other])
            if (abs(r) > 0.9) & (p_val < 0.05):
                leak_numerical[str((target, col_other))] = float(p_val)
        else:
            df_masked = df.loc[mask]
            crosstab = pd.crosstab(df_masked[col_other], df_masked[target])
            p_val = scipy.stats.chi2_contingency(crosstab)[1]
            if p_val < 0.05:
                leak_categorical[str((target, col_other))] = float(p_val)

    return leak_numerical, leak_categorical

def test_leakage_continuous(df:pd.DataFrame, target:str) -> dict:
    leakage = {}
    for col in list(df.columns):
        mask = (df[target].notnull()) & (df[col].notnull())
        if mask.sum() < 3:
            continue
        if col == target:
            continue
        elif df[col].dtype in (int, float):
            r = scipy.stats.pearsonr(df.loc[mask][target], df.loc[mask][col]).statistic
            if abs(r) > 0.9:
                leakage[str((target, col))] = float(r)

    return leakage

def quantify_leakage_continuous(df: pd.DataFrame, target: str, suspected_leaks: list):
    leakage = {}
    for col in suspected_leaks:
        mask = (df[target].notnull()) & (df[col].notnull())
        if mask.sum() < 3:
            continue
        if col == target:
            continue
        elif df[col].dtype in (int, float):
            simple_lr = LinearRegression()
            simple_lr.fit(X = df[col], y = df[target])
            yhat = simple_lr.predict(X = df[col])
            score = r2_score(df[col], yhat)

            if score > 0.9:
                leakage[str((target, col))] = float(score)

    return leakage