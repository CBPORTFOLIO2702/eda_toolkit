import pandas as pd
import numpy as np

def get_class_imbalance(df: pd.DataFrame, target: str) -> dict:
    classes = df[target].unique()
    value_counts = df[target].value_counts()
    class_dict = {}
    for idx in range(len(classes)):
        class_dict[str(classes[idx])] = float((value_counts[idx]/len(df)))

    return class_dict

def get_target_distribution(df: pd.DataFrame, target: str) -> dict:
    target = df[target]
    min = target.min()
    lq = np.percentile(target, 25)
    median = target.median()
    uq = np.percentile(target, 75)
    max = target.max()
    mean = target.mean()
    std = target.std()
    skew = target.skew()
    kurtosis = target.kurtosis()

    target_dist = {'mean': float(mean),
                   'std': float(std),
                   'min': float(min),
                   'lower_quartile': float(lq),
                   'median': float(median),
                   'upper_quartile': float(uq),
                   'max': float(max),
                   'skew': float(skew),
                   'kurtosis': float(kurtosis)}

    return target_dist