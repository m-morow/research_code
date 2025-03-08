#!/usr/bin/env python

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime

def date_string_to_datetime(dataframe, header, src_col, target_col, sorted=True):
    """
    returns dataframe with converted datetime [YYYY-MM-DD] of YYYYMMDD string.
    compatible with matplotlib plotting x-axis 
    YYYYMMDD is MT's preferred shorthand, easier to code here...

    Parameters:
    -------
    dataframe: pandas dataframe or path to .csv file
    header: line of header, usually 0 or None
    src_col: column that holds YYYYMMDD string
    target_col: column to update with datetime
    sorted: default True, returns sorted dataframe by date

    Returns:
    --------
    sorted dataframe by date
    
    """
    if '.csv' in dataframe: 
        df = pd.read_csv(dataframe, header=header)
    else:
        df = dataframe
    for i, date in enumerate(np.array(df[src_col])):
        df.loc[[i], [target_col]] = datetime.strptime(str(date), "%Y%m%d")
    if sorted:
        return df.sort_values(by=target_col)
    else:
        return df