import pandas as pd
import numpy as np

def profile_data_shape(df: pd.DataFrame) -> dict:
    df_shape_profile = {'n_rows': len(df),
                        'n_cols': len(df.columns)
                        }

    return df_shape_profile

def profile_dtypes(df: pd.DataFrame) -> dict:
    dtypes = df.dtypes
    numerical_cols = [key for key in dict(dtypes) if dict(dtypes)[key] in ['int64', 'float64']]
    non_numeric_cols = [key for key in dict(dtypes) if dict(dtypes)[key] in ['object', 'str']]

    dtype_dict = {'numeric_cols': numerical_cols,
                  'non_numeric_cols': non_numeric_cols
                  }

    return dtype_dict

def get_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    dupes = df.duplicated(keep = 'first').astype(int)

    return dupes