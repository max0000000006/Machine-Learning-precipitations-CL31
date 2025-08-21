from fastai.tabular.all import *
import numpy as np
from fastai.metrics import Metric
from netCDF4 import Dataset
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.signal import firwin, lfilter, freqz, filtfilt

base_path_CL = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\training_data_ML\\202212_CL31"
base_path_MRR = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\training_data_ML\\202212_MRR"
nc_files_CL = sorted([f for f in os.listdir(base_path_CL) if f.endswith('.nc')])
nc_files_MRR = sorted([f for f in os.listdir(base_path_MRR) if f.endswith('.nc')])

MRR = None
CEILO = None


for f in nc_files_CL:
    file_path = os.path.join(base_path_CL, f)
    with Dataset(file_path,'r') as ds:
        rcs = ds.variables['rcs_0'][:,40:50]
        if CEILO is None:
            CEILO = rcs
        else :
            CEILO = np.concatenate((CEILO, rcs), axis=0)

for f in nc_files_MRR:
    file_path = os.path.join(base_path_MRR,f)
    with Dataset(file_path,'r') as ds:
        sfr = ds.variables['SnowfallRate'][:,3:4]
        if MRR is None:
            MRR = sfr
        else:
            MRR = np.concatenate((MRR,sfr),axis = 0)


for i in range(np.shape(MRR)[0]):
    for j in range(np.shape(MRR)[1]):
        if MRR[i,j]<=0:
            MRR[i,j]=np.nan

for i in range(np.shape(CEILO)[0]):
    for j in range(np.shape(CEILO)[1]):
        if CEILO[i,j]==-999.0:
            CEILO[i,j]=np.nan


CEILO_mean = np.zeros((np.shape(CEILO)[0]//2,np.shape(CEILO)[1]))
for i in range(np.shape(CEILO_mean)[0]):
        CEILO_mean[i,:] = (CEILO[2*i,:] + CEILO[2*i+1,:])/2 



row_mask = ~np.isnan(CEILO_mean).any(axis=1) & ~np.isnan(MRR).any(axis=1)
# Appliquer le masque proprement
CEILO_mean = CEILO_mean[row_mask]
MRR = MRR[row_mask]


M = CEILO_mean  
order = 80
cutoff = 0.005  # Normalisé : 0.1 correspond à 0.1 * Nyquist = 0.05 * sampling_rate
fir_coeff = firwin(numtaps=order + 1, cutoff=cutoff, window='hamming')
fs = 10000  # Fréquence d’échantillonnage (arbitraire)
t = np.linspace(0, 1, fs, endpoint=False)
M_filtered = np.zeros_like(M)
for i in range(M.shape[1]):
    if M.shape[0] > 3 * len(fir_coeff):  # Vérifie que longueur est suffisante
        M_filtered[:, i] = filtfilt(fir_coeff, [1.0], M[:, i])
    else:
        raise ValueError("Signal trop court pour filtrer avec filtfilt.")
M=M_filtered

FEATURES = np.zeros((np.shape(M)[0],17))
for k in range(np.shape(M)[0]):
    FEATURES[k][0]=(np.mean(M[k,:]))
    FEATURES[k][1]=(np.var(M[k,:]))
    FEATURES[k][2]=(np.max(M[k,:]))
    FEATURES[k][3]=(np.min(M[k,:]))
    FEATURES[k][4]=(np.sum([value**2 for value in M[k,:]]))
    FEATURES[k][5]=(np.sum(M[k,:] > 0) / np.shape(M)[1])
    FEATURES[k][6]=(np.max(M[k,:]) - np.min(M[k,:]))
    FEATURES[k][7]=(np.sum([value for value in M[k,:]]))
    #FEATURES[k][8]= 300 + (k%7)*100
    FEATURES[k][8] = scipy.stats.skew(M[k,:], nan_policy='omit')
    FEATURES[k][9] = scipy.stats.kurtosis(M[k,:], nan_policy='omit')
    diffs = np.diff(M[k,:])
    FEATURES[k][10] = np.mean(np.abs(diffs))  # Mean absolute difference
    second_diffs = np.diff(M[k,:], n=2)
    FEATURES[k][11] = np.mean(np.abs(second_diffs))
    FEATURES[k][12] = np.corrcoef(M[k,:-1], M[k,1:])[0,1]
    hist, _ = np.histogram(M[k,:], bins=10, density=True)
    FEATURES[k][13] = -np.sum(hist * np.log(hist + 1e-10))
    fft_vals = np.abs(np.fft.fft(M[k,:]))
    FEATURES[k][14] = np.argmax(fft_vals[1:]) + 1  # Dominant frequency index (skip DC)
    FEATURES[k][15] = np.sum(fft_vals[1:]**2)
    x = np.arange(len(M[k,:]))
    slope, _ = np.polyfit(x, M[k,:], 1)
    FEATURES[k][16] = slope


#Importation du modèle:

row_mask = ~np.isnan(FEATURES).any(axis=1) & ~np.isnan(MRR).any(axis=1)
FEATURES= FEATURES[row_mask]
MRR = MRR[row_mask]

learn= load_learner('modele2024.plk')
y_true = MRR.flatten()
df_pred = pd.DataFrame(FEATURES, columns=[f'FEATURES[{i}]' for i in range(FEATURES.shape[1])])
df_pred['MRR'] = y_true
df_pred['log_MRR'] = np.log1p(df_pred['MRR'])

dl = learn.dls.test_dl(df_pred)

saved_metrics = learn.metrics
learn.metrics = []


preds_log, targs_log = learn.get_preds(dl = dl)
y_pred= np.expm1(preds_log)

df_feat = pd.DataFrame(FEATURES, columns=[f"f{i}" for i in range(FEATURES.shape[1])])
nan_rows = df_feat[df_feat.isnull().any(axis=1)]
print(f"🧪 Lignes contenant des NaN dans FEATURES : {len(nan_rows)}")
print(nan_rows.head())

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"📊 RMSE : {rmse:.4f}")
print(f"📊 MAE  : {mae:.4f}")
print(f"📊 R²   : {r2:.4f}")

plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([0, np.max(MRR)], [0, np.max(MRR)], '--', color='gray')  # Diagonale idéale
plt.xlabel("Valeur réelle (MRR)")
plt.ylabel("Prédiction du modèle")
plt.title("📈 Valeurs réelles vs. prédites")
plt.grid(True)
plt.show()




