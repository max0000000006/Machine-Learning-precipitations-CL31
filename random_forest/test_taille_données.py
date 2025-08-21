import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import xarray as xr
import os
from fastai.metrics import Metric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.signal import find_peaks
from fastai.tabular.all import *
import seaborn as sns
from scipy.signal import firwin, lfilter, freqz, filtfilt
from fastai.callback.tracker import EarlyStoppingCallback
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestRegressor
from fastai.tabular.model import TabularModel
from fastai.learner import Learner
from fastai.tabular.all import *
import numpy as np
from fastai.metrics import Metric
from netCDF4 import Dataset
import pandas as pd
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.signal import firwin, lfilter, freqz, filtfilt
import datetime
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.decomposition import PCA

base_path_CL = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\DATA_TRAINING2024\\2024_CL\\202404\\"
base_path_MRR = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\DATA_TRAINING2024\\2024_MRR\\Mk_processed202404\\"
nc_files_CL31 = [f for f in os.listdir(base_path_CL) if f.endswith('.nc')]
nc_files_MRR = [f for f in os.listdir(base_path_MRR) if f.endswith('.nc')]
#file_path_CL = os.path.join(base_path_CL, nc_files_CL31[0])  # Just to check the first file
#data = Dataset(file_path_CL, 'r')  # Open the first file to check its structure
#print(data.variables.keys())  # Print the variable names in the first file
for file in nc_files_CL31:
    file_path = os.path.join(base_path_CL, file)
    with Dataset(file_path, 'r') as ds:
        beta_raw = ds.variables['rcs_0'][:]
        if np.shape(beta_raw)!=(2880,770):
            print(f"File {file} has shape {np.shape(beta_raw)} instead of (2880, 770) for file : ")
print("CL ok")
print(len(nc_files_CL31))
"""
Z = np.zeros((58,31))
for file in nc_files_MRR:
    file_path = os.path.join(base_path_MRR, file)
    with Dataset(file_path, 'r') as ds:
        snowfall_rate = ds.variables['SnowfallRate'][:]
        #snowfall_rate = np.vstack([snowfall_rate, Z])
        if np.shape(snowfall_rate)!=(1440, 31):
            print(f"File {file} has shape {np.shape(snowfall_rate)} instead of (1440, 31) for file : ")
print("MRR ok")
print(len(nc_files_MRR))
"""
M=np.array(beta_raw[100,:])


"""q1, q3 = np.percentile(M, [25, 75], axis=0)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
M = np.clip(M, lower_bound, upper_bound)"""
"""
order = 50
cutoff = 0.000001  # Normalisé : 0.1 correspond à 0.1 * Nyquist = 0.05 * sampling_rate
fir_coeff = firwin(numtaps=order + 1, cutoff=cutoff, window='hamming')
fs = 10000  # Fréquence d’échantillonnage (arbitraire)
t = np.linspace(0, 1, fs, endpoint=False)

M_filtered = np.zeros_like(M)
M_filtered[:] = filtfilt(fir_coeff, [1.0], M[:])

M=M_filtered
"""


h = np.arange(0,7700,10)
plt.figure(figsize=(6,6))
plt.plot(M,h)
plt.ylabel("Altitude (m)")
plt.xlabel("Signal rétrodiffusé (sr-1.m-1)")
plt.show()