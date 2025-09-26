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





# Initialize all the path to the NetCDF files
base_path_CL = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\2024\\2024_CL\\"
base_path_MRR = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\2024\\2024_MRR\\"
List_month_CL = ['202401','202402','202403','202404','202405','202406','202407','202408','202409','202410','202411','202412']
List_month_MRR = ['Mk_processed202401','Mk_processed202402','Mk_processed202403','Mk_processed202404','Mk_processed202405','Mk_processed202406','Mk_processed202407','Mk_processed202408','Mk_processed202409','Mk_processed202410','Mk_processed202411','Mk_processed202412']
incomplete_months = ['Mk_processed202404','Mk_processed202405','Mk_processed202406','Mk_processed202407','Mk_processed202408','Mk_processed202409','Mk_processed202410']


CEILO = None  # Initialize CEILO as None to concatenate later
k= 0
for month in List_month_CL:
    k+= 1
    month_dir = os.path.join(base_path_CL, month)
    if not os.path.exists(month_dir):
        print(f"⚠️ Dossier non trouvé : {month_dir}")
        continue
    nc_files = sorted([f for f in os.listdir(month_dir) if f.endswith('.nc')])
    for file in nc_files:
        file_path = os.path.join(month_dir, file)
        try:
            with Dataset(file_path, 'r') as ds:
                wt = ds.variables['window_transmission'][:]
                vv = ds.variables['vertical_visibility'][:]
                beta_raw = ds.variables['rcs_0'][:, 30:40]
                times_raw = ds.variables['time'][:]
                n = np.shape(beta_raw)[0]
                m = [k] * n
                if CEILO is None:
                    CEILO = beta_raw
                    all_times = times_raw
                    vertical_vis = vv
                    window_trans = wt
                    month_l = m
                else:
                    CEILO = np.concatenate((CEILO, beta_raw), axis=0)
                    all_times = np.concatenate((all_times, times_raw), axis=0)
                    vertical_vis = np.concatenate((vertical_vis,vv),axis = 0)
                    window_trans = np.concatenate((window_trans,wt),axis=0)
                    month_l = np.concatenate((month_l,m),axis=0)
        except Exception as e:
            print(f"❌ Erreur avec {file_path} : {e}")

#Initialize MRR

MRR = None
Z = np.zeros((58,1))
for month in List_month_MRR:
    month_dir = os.path.join(base_path_MRR, month)
    if not os.path.exists(month_dir):
        print(f"⚠️ Dossier non trouvé : {month_dir}")
        continue

    nc_files = sorted([f for f in os.listdir(month_dir) if f.endswith('.nc')])
    for file in nc_files:
        file_path = os.path.join(month_dir, file)
        try:
            with Dataset(file_path, 'r') as ds:
                snowfall_rate = ds.variables['SnowfallRate'][:, 2:3]  # (T, 1)
                # Ajout conditionnel du vecteur Z si le mois est incomplet
                if month in incomplete_months:
                    snowfall_rate = np.vstack([snowfall_rate, Z])

                if MRR is None:
                    MRR = snowfall_rate
                else:
                    MRR = np.concatenate((MRR, snowfall_rate), axis=0)
        except Exception as e:
            print(f"❌ Erreur avec {file_path} : {e}")

#Creating CEILO_mean to smooth the data of CEIL and get the same resolution as the MRR
CEILO_mean = np.zeros((np.shape(CEILO)[0]//2,np.shape(CEILO)[1]))
for i in range(np.shape(CEILO_mean)[0]):
    CEILO_mean[i,:] = (CEILO[2*i,:] + CEILO[2*i+1,:])/2

#PREP MRR
for i in range(np.shape(MRR)[0]):
    for j in range(np.shape(MRR)[1]):
        if MRR[i,j]<=0:
            MRR[i,j]=np.nan


# Remove all the filled data
row_mask = ~np.isnan(CEILO_mean).any(axis=1) & ~np.isnan(MRR).any(axis=1)
CEILO_mean = CEILO_mean[row_mask]
MRR = MRR[row_mask]


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

M = make_vertical_windows(CEILO_mean, MRR)
M = M.reshape(-1,10)

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


print(np.shape(M))
print(np.shape(MRR))

FEATURES = np.zeros((M.shape[0], 7))  # Adjust for new features

for k in range(M.shape[0]):
    profile = M[k, :]
    
    FEATURES[k, 0] = np.sqrt(np.sum(profile ** 2))  # Signal energy
    FEATURES[k, 1] = scipy.stats.skew(profile, nan_policy='omit')
    FEATURES[k, 2] = np.corrcoef(profile[:-1], profile[1:])[0, 1] if len(profile) > 1 else 0
    x = np.arange(len(profile))
    slope, _ = np.polyfit(x, profile, 1)
    FEATURES[k, 3] = slope
    FEATURES[k, 4] = np.min(profile)
    FEATURES[k, 5] = np.max(profile)
    FEATURES[k, 6] = np.mean(np.diff(profile))  # Average vertical gradient



class R2Score(Metric):
    def reset(self): 
        self.y_true, self.y_pred = [], []

    def accumulate(self, learn):
        self.y_true += learn.y.cpu().numpy().tolist()
        self.y_pred += learn.pred.cpu().numpy().tolist()

    @property
    def value(self): 
        return r2_score(self.y_true, self.y_pred)

    @property
    def name(self): 
        return "R2"

def weighted_mse(y_pred, y_true):
    weights = y_true / (y_true.mean()+ 1e-8)  
    return (weights * (y_pred - y_true)**2).mean()


def huber_loss(y_pred, y_true,delta = 0.1):
    error = y_true - y_pred
    is_small_error = torch.abs(error) < delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    return torch.where(is_small_error, squared_loss, linear_loss).mean()

X_full = FEATURES
y_full = MRR


feature_names = [f'FEATURES[{i}]' for i in range(X_full.shape[1])]
cont_vars = [f'FEATURES[{i}]' for i in range(np.shape(X_full)[1])]
dep_var = 'log_MRR'


df_full = pd.DataFrame(X_full, columns=[f'FEATURES[{i}]' for i in range(X_full.shape[1])]) # Features
df_full['MRR'] = y_full
df_full['log_MRR'] = np.log1p(df_full['MRR'])


#Entraînement du modèle avec fastAI
splits = RandomSplitter(seed=42)(df_full)
dls = TabularDataLoaders.from_df(df_full, y_names=dep_var, cont_names=cont_vars, splits=splits)
rf = tabular_learner(dls,layers=[200,100,50], y_range=(0, np.log1p(y_full).max() + 1),loss_func=lambda x,y : huber_loss(x,y,delta=0.2),  metrics=[rmse,R2Score()] )
rf.fit_one_cycle(500,slice(1e-5,1e-3),cbs=EarlyStoppingCallback(monitor='valid_loss', patience=50))
rf.show_results()
rf.recorder.plot_loss()
preds_val_log, targs_val_log = rf.get_preds(dl=dls.valid)
full_dl = dls.test_dl(df_full)
y_test_pred, y_test = rf.get_preds(dl=full_dl)
y_test_pred = np.expm1(y_test_pred)
y_test = np.expm1(y_test)



r2 = r2_score(y_test, y_test_pred)
rmse = root_mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)

#Exportation du modèle
rf.export("model.pkl")



print(f"📈 R² (test) : {r2:.4f}")
print(f"📉 RMSE (test) : {rmse:.4f}")
print(f"📉 MAE (test) : {mae:.4f}")

print("Train mean:", FEATURES.mean(axis=0))

print(np.isnan(y_full).sum())
print((y_full <= 0).sum())
print(np.unique(y_full[:20]))


# Résumé de performances
metrics_summary = {
    "Dataset": ["Train/Test Split", "202403"],
    "RMSE": [root_mean_squared_error(y_test, y_test_pred)],
    "R2": [r2_score(y_test, y_test_pred)],
    "MAE": [mean_absolute_error(y_test, y_test_pred)]
}

# Affichage des performances
print("\n📊 Résumé des performances :")
for i in range(1):
    print(f"{metrics_summary['Dataset'][i]} | R² = {metrics_summary['R2'][i]:.4f} | "
          f"RMSE = {metrics_summary['RMSE'][i]:.4f} | MAE = {metrics_summary['MAE'][i]:.4f}")

# Sauvegarde CSV
with open("summary_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", "R2", "RMSE", "MAE"])
    for i in range(1):
        writer.writerow([
            metrics_summary["Dataset"][i],
            round(metrics_summary["R2"][i], 4),
            round(metrics_summary["RMSE"][i], 4),
            round(metrics_summary["MAE"][i], 4)
        ])
print("✅ Résumé exporté sous summary_metrics.csv")



#Affichage des résultats
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([0, max(y_test.max(), y_test_pred.max())], [0, max(y_test.max(), y_test_pred.max())], '--', color='gray')
plt.xlabel("Valeur réelle (y_test)")
plt.ylabel("Prédiction (y_test_pred)")
plt.title("📈 Prédictions vs Réel (test set)")
plt.grid(True)
plt.tight_layout()
plt.show()