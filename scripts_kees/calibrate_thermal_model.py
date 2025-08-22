# Arctic-XBeach
# Script to run Arctic-XBeach simulations, calibrate and anlyse
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Functions
def compute_RMSE(run_id, df_val_data):


# Part 1 - load in Erikson data
df_erikson1 = pd.read_csv(r"database\raw_datasets\erikson2\Temp_arrays USGS-UCSC_Oberle\BI_T-1_processed.csv", parse_dates=['date_time'])

# Part 2 - load in model output data


# Part 3 - determine skill