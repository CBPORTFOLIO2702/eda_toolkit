import pandas as pd
import numpy as np
from datetime import datetime

def impute_mean_overall(df: pd.DataFrame, col: str, log_dict: dict) -> tuple:
    mean_for_col = df[col].mean()
    null_count = int(df[col].isna().sum())
    df[col] = df[col].fillna(mean_for_col)
    log_dict[datetime.now()] = {'action': 'impute_mean', 'col': col, 'rows_affected': null_count}
    
    return df, log_dict

def impute_median_overall(df: pd.DataFrame, col: str, log_dict: dict) -> tuple:
    median_for_col = df[col].median()
    null_count = int(df[col].isna().sum())
    df[col] = df[col].fillna(median_for_col)
    log_dict[datetime.now()] = {'action': 'impute_median', 'col': col, 'rows_affected': null_count}

    return df, log_dict

def impute_mean_grouped(df: pd.DataFrame, target_col: str, group_cols: list, log_dict: dict) -> tuple:
    means_for_group = df.groupby(group_cols)[target_col].mean().reset_index()
    null_count = int(df[target_col].isna().sum())
    means_for_group = means_for_group.rename({target_col: f"{target_col}_mean"})
    df = df.merge(means_for_group, how = 'inner', on = group_cols)
    df[target_col] = df[target_col].fillna(df[f"{target_col}_mean"])

    log_dict[datetime.now()] = {'action': 'impute_mean', 'col': target_col, 'grouping': group_cols,  'rows_affected': null_count}

    df = df.drop(columns=[f"{target_col}_mean"])

    return df, log_dict

def impute_median_grouped(df: pd.DataFrame, target_col: str, group_cols: list, log_dict: dict) -> tuple:
    medians_for_group = df.groupby(group_cols)[target_col].median().reset_index()
    null_count = int(df[target_col].isna().sum())
    medians_for_group = medians_for_group.rename({target_col: f"{target_col}_median"})
    df = df.merge(medians_for_group, how = 'inner', on = group_cols)
    df[target_col] = df[target_col].fillna(df[f"{target_col}_median"])

    df = df.drop(columns=[f"{target_col}_median"])

    log_dict[datetime.now()] = {'action': 'impute_mean', 'col': target_col, 'grouping': group_cols,  'rows_affected': null_count}

    return df, log_dict

def impute_mode_overall(df: pd.DataFrame, col: str, log_dict: dict) -> tuple:
    mode_for_col = df[col].mode()
    null_count = int(df[col].isna().sum())
    df[col] = df[col].fillna(mode_for_col)
    log_dict[datetime.now()] = {'action': 'impute_mode', 'col': col, 'rows_affected': null_count}

    return df, log_dict

def impute_mode_grouped(df: pd.DataFrame, target_col: str, group_cols: list, log_dict: dict) -> tuple:
    modes_for_group = df.groupby(group_cols)[target_col].mode().reset_index()
    null_count = int(df[target_col].isna().sum())
    modes_for_group = modes_for_group.rename({target_col: f"{target_col}_mode"})
    df = df.merge(modes_for_group, how = 'inner', on = group_cols)
    df[target_col] = df[target_col].fillna(df[f"{target_col}_mode"])

    df = df.drop(columns=[f"{target_col}_mode"])
    log_dict[datetime.now()] = {'action': 'impute_mean', 'col': target_col, 'grouping': group_cols,  'rows_affected': null_count}

    return df, log_dict

def add_is_col_missing(df: pd.DataFrame, col: str, log_dict: dict) -> tuple:
    df[f'is_{col}_missing'] = np.where(df[col].isna(), 1, 0)
    null_count = int(df[col].isna().sum())
    log_dict[datetime.now()] = {'action': 'flag_missing', 'col': col, 'rows_affected': null_count}
    
    return df, log_dict