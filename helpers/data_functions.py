# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_flavor as pf

def clean_returns(df, threshold=0.99):
    df_cleaned = df.replace(0.0, np.nan)  # replace 0.0 with NaN
    df_cleaned = df_cleaned.dropna(axis=0, how='all')  # drop rows with all NaN values
    df_cleaned = df_cleaned.dropna(axis=1, thresh=threshold * len(df_cleaned))
    return df_cleaned.fillna(0.0)

@pf.register_dataframe_method
def winsorize(df, percentile=0.01, axis=0):
    clip_axis = 1 if axis == 0 else 0
    lower_bound = df.quantile(percentile, axis=axis)
    upper_bound = df.quantile(1 - percentile, axis=axis)
    return df.clip(lower=lower_bound, upper=upper_bound, axis=clip_axis)

@pf.register_dataframe_method
def zstandardize(df, axis=0):
    stdz_axis = 1 if axis == 0 else 0
    mean = df.mean(axis=axis)
    std = df.std(axis=axis)
    return df.sub(mean, axis=stdz_axis).div(std, axis=stdz_axis)