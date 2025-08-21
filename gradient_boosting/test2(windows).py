import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt
import os
import xarray as xr
from netCDF4 import Dataset
from scipy.signal import firwin, lfilter, freqz, filtfilt
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from netCDF4 import Dataset
from scipy.signal import firwin, lfilter, freqz, filtfilt
import os
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


base_path_CL = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\training_data_ML\\202212_CL31\\"
base_path_MRR = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\training_data_ML\\202212_MRR\\"
nc_files_CL31 = sorted([f for f in os.listdir(base_path_CL) if f.endswith('.nc')])
nc_files_MRR = sorted([f for f in os.listdir(base_path_MRR) if f.endswith('.nc')])

CEILO = None  # Initialize CEILO as None to concatenate later
#file_path_CL = os.path.join(base_path_CL, nc_files_CL31[0])  # Just to check the first file
#CEILO = Dataset(file_path_CL, 'r').variables['rcs_0'][:]

for k in range(len(nc_files_CL31)):
    file_path = os.path.join(base_path_CL, nc_files_CL31[k])
    with Dataset(file_path, 'r') as ds:
        beta_raw = ds.variables['rcs_0'][:]
        if CEILO is None:
            CEILO = beta_raw
        else:   
            CEILO = np.concatenate((CEILO, beta_raw), axis=0)
MRR = None
#file_path_MRR = os.path.join(base_path_MRR, nc_files_MRR[0])  # Just to check the first file
#MRR = Dataset(file_path_MRR, 'r').variables['SnowfallRate'][:]

for k in range(len(nc_files_MRR)):
    file_path = os.path.join(base_path_MRR, nc_files_MRR[k])
    with Dataset(file_path, 'r') as ds:
        snowfall_rate = ds.variables['SnowfallRate'][:]
        if MRR is None:
            MRR = snowfall_rate
        else:
            MRR = np.concatenate((MRR, snowfall_rate), axis=0)

#MRR = np.delete(MRR, np.s_[9:], axis=1)
MRR = np.delete(MRR, np.s_[0:2], axis=1) 
CEILO = np.delete(CEILO, np.s_[320:], axis=1)
CEILO = np.delete(CEILO, np.s_[0:30], axis=1)
CEILO_mean = np.zeros((np.shape(CEILO)[0]//2,np.shape(CEILO)[1]))
for i in range(np.shape(CEILO_mean)[0]):
    CEILO_mean[i,:] = (CEILO[2*i,:] + CEILO[2*i+1,:])/2
print(np.shape(CEILO_mean))  # Afficher la forme de CEILO
print(np.shape(MRR))  # Afficher la forme de MRR



def make_vertical_windows(X,Y):
    a,b = X.shape # données du CEILO
    c,d = Y.shape # données du MRR
    if a != c:
        raise ValueError("Les dimensions des données CEILO et MRR ne correspondent pas.")
    if b % 10 != 0:
        raise ValueError("Le nombre de colonnes dans les données CEILO doit être un multiple de 10 pour créer des fenêtres verticales de 10.")
    num_windows = b // 10
    X_windows = np.zeros((a, num_windows, 10))  # Initialiser    
    for i in range(num_windows):
        X_windows[:, i, :] = X[:, i*10:(i+1)*10]  # Extraire les fenêtres de 10 colonnes
    return X_windows

LABELS = np.zeros_like(MRR , dtype=float)  # Initialize LABELS with zeros
for k in range (np.shape(MRR)[0]):
    for j in range(np.shape(MRR)[1]):
        if (MRR[k,j])<0:
            LABELS[k,j] = 0
        else:
            LABELS[k,j] = 1

M = make_vertical_windows(CEILO_mean, MRR)
print(np.shape(M))
print(np.shape(MRR))
print(np.shape(CEILO))  # Afficher la forme de CEILO
print(np.shape(LABELS))



for i in range(np.shape(LABELS)[0]):
    compteur = 0
    altitude_debut = -1
    for j in range(np.shape(LABELS)[1]):
        if compteur == 0 and LABELS[i,j]==1:
            altitude_debut=j
        if LABELS[i,j]==1:
            compteur +=1
        if compteur == 10 and altitude_debut!=-1 and altitude_debut!=0:
            LABELS[i,j:] = np.nan
            break
        if compteur == 7 and altitude_debut==0:
            LABELS[i,j:] = np.nan
            break

M = M.reshape(-1,10)
LABELS = LABELS.flatten()


mask = ~np.isnan(LABELS)
M = M[mask]
LABELS = LABELS[mask]

print(np.shape(M))  # Print the shape of M to verify the concatenation
print(np.shape(LABELS))  # Print the shape of LABELS to verify the concaten

compteur = 0
compteur_2 = 0
for k in range(np.shape(LABELS)[0]):
    if LABELS[k] == 1:
        compteur += 1
prop = compteur / np.shape(LABELS)[0]
print(f"Proportion de précipitation : {prop:.2f}")

Seuil = np.shape(LABELS)[0] - 2*compteur
print(Seuil)
L= np.zeros((np.shape(M)[0]))
for k in range(np.shape(LABELS)[0]-1,-1,-1):
    if LABELS[k] == 0:
        L[compteur_2] = k
        compteur_2 += 1
        if compteur_2 > Seuil:
            break
M = np.delete(M, L.astype(int), axis=0)
LABELS = np.delete(LABELS, L.astype(int), axis=0)
print(np.shape(M))
print(np.shape(LABELS))
prop = compteur / np.shape(LABELS)[0]
print(f"Proportion de précipitation : {prop:.2f}")


q1, q3 = np.percentile(M, [25, 75], axis=0)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
M = np.clip(M, lower_bound, upper_bound)


fs = 1e6
order = 50
cutoff = 1e2

fir_coeff = firwin(numtaps=order + 1, cutoff=cutoff, window='hamming', fs = fs) 

M_filtered = np.zeros_like(M)
for i in range(M.shape[1]):
    if M.shape[0] > 3 * len(fir_coeff):  # Vérifie que longueur est suffisante
        M_filtered[:, i] = filtfilt(fir_coeff, [1.0], M[:, i])
    else:
        raise ValueError("Signal trop court pour filtrer avec filtfilt.")
M=M_filtered


FEATURES = np.zeros((M.shape[0], 7))  # Adjust for new features

for k in range(M.shape[0]):
    profile = M[k, :]
    
    # Existing features
    FEATURES[k, 0] = np.sqrt(np.sum(profile ** 2))  # Signal energy
    FEATURES[k, 1] = scipy.stats.skew(profile, nan_policy='omit')
    FEATURES[k, 2] = np.corrcoef(profile[:-1], profile[1:])[0, 1] if len(profile) > 1 else 0
    x = np.arange(len(profile))
    slope, _ = np.polyfit(x, profile, 1)
    FEATURES[k, 3] = slope
    FEATURES[k, 4] = np.min(profile)
    FEATURES[k, 5] = np.max(profile)
    
    # New features

    FEATURES[k, 6] = np.mean(np.diff(profile))  # Average vertical gradient
#M = np.concatenate([FEATURES, M], axis=1)

X_train, X_test, y_train, y_test = train_test_split(FEATURES, LABELS, test_size=0.3, random_state=42)

# Modèle
clf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=3)
clf.fit(X_train, y_train)

# Évaluation
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Rapport de classification :")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Matrice de confusion : précipitation vs. pas de précipitation")
plt.grid()
plt.show()