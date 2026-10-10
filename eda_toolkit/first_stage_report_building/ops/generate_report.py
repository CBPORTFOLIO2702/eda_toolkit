import pandas as pd
import numpy as np
import logging
import json

from eda_toolkit.first_stage_report_building.source.column_analysis import get_target_distribution, get_class_imbalance
from eda_toolkit.first_stage_report_building.source.leakage_detection import test_leakage_continuous, quantify_leakage_continuous, test_leakage_binary
from eda_toolkit.first_stage_report_building.source.missingness import test_mar, get_missing_perc, validate_date
from eda_toolkit.first_stage_report_building.source.outliers import check_outliers_in_group, check_outliers_overall
from eda_toolkit.first_stage_report_building.source.shape_and_dtypes import profile_dtypes, profile_data_shape, get_duplicates
from eda_toolkit.first_stage_report_building.source.multicolinearity import test_multicollinearity

def build_report_regression(df: pd.DataFrame, target_col:str, date_cols: list = []) -> None:
    report = {}

    data_profile = {}
    data_profile['shape_report'] = profile_data_shape(df)
    data_profile['dtype_report'] = profile_dtypes(df)

    numeric_cols = data_profile['dtype_report']['numeric_cols']
    categorical_cols = data_profile['dtype_report']['non_numeric_cols']

    report['data_profile'] = data_profile

    missingness = {}

    perc_problematic_dates = {}

    for col in date_cols:
        df, date_missing_perc = validate_date(df, col)
        perc_problematic_dates[col] = date_missing_perc

    perc_missing, alarming_cols = get_missing_perc(df)

    missingness['missing_percentage'] = perc_missing
    missingness['alarming_cols'] = alarming_cols

    missing_at_random_num, missing_at_random_cat = test_mar(df, alarming_cols)

    missingness['suspected_missing_at_random_numerical'] = missing_at_random_num
    missingness['suspected_missing_at_random_categorical'] = missing_at_random_cat

    missingness['perc_invalid_dates'] = perc_problematic_dates

    report['missingness'] = missingness

    leakage_detection = {}

    numerical_leak = test_leakage_continuous(df, target_col)

    report['target_leakage'] = leakage_detection

    leakage_detection['detected_numercial_leakages'] = numerical_leak

    outliers = {}

    for col in numeric_cols:
        outliers[col] = check_outliers_overall(df, col)

    report['outliers'] = outliers

    col_profiling = {}

    for col in numeric_cols:
        if col == target_col:
            continue
        else:
            col_profiling[col] = get_target_distribution(df, col)

    report['numeric_column_distributions'] = col_profiling

    target_profile = {}
    target_profile['target_distribution'] = get_target_distribution(df, target_col)

    report['target_profile'] = target_profile

    multicolinearity = test_multicollinearity(df, numeric_cols, target_col)
    report['multicolinearity'] = multicolinearity


    with open('eda_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("Saved successfully")


def build_report_binary_classification(df: pd.DataFrame, target_col:str, date_cols: list = []) -> None:
    report = {}

    data_profile = {}
    data_profile['shape_report'] = profile_data_shape(df)
    data_profile['dtype_report'] = profile_dtypes(df)

    numeric_cols = data_profile['dtype_report']['numeric_cols']
    categorical_cols = data_profile['dtype_report']['non_numeric_cols']

    report['data_profile'] = data_profile

    missingness = {}

    perc_problematic_dates = {}

    for col in date_cols:
        df, date_missing_perc = validate_date(df, col)
        perc_problematic_dates[col] = date_missing_perc

    perc_missing, alarming_cols = get_missing_perc(df)

    missingness['missing_percentage'] = perc_missing
    missingness['alarming_cols'] = alarming_cols

    missing_at_random_num, missing_at_random_cat = test_mar(df, alarming_cols)

    missingness['suspected_missing_at_random_numerical'] = missing_at_random_num
    missingness['suspected_missing_at_random_categorical'] = missing_at_random_cat

    missingness['perc_invalid_dates'] = perc_problematic_dates

    report['missingness'] = missingness

    leakage_detection = {}

    numerical_leak, cat_leak = test_leakage_binary(df, target_col)

    report['target_leakage'] = leakage_detection

    leakage_detection['detected_numercial_leakages'] = numerical_leak
    leakage_detection['detected_categorical_leakages'] = cat_leak

    outliers = {}

    for col in numeric_cols:
        outliers[col] = check_outliers_overall(df, col)

    report['outliers'] = outliers

    col_profiling = {}

    for col in numeric_cols:
        if col == target_col:
            continue
        else:
            col_profiling[col] = get_target_distribution(df, col)

    report['numeric_column_distributions'] = col_profiling

    target_profile = {}

    target_profile['class_imbalance'] = get_class_imbalance(df, target_col)

    report['target_profile'] = target_profile

    multicolinearity = test_multicollinearity(df, numeric_cols, target_col)
    report['multicolinearity'] = multicolinearity

    # Since you've fixed the numpy/tuple-key issues at the source,
    # this should now serialize cleanly without a custom encoder
    with open('eda_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("Saved successfully")