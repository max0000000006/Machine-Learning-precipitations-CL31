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
base_path_CL = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\DATA_TRAINING2024\\2024_CL\\"
base_path_MRR = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\DATA_TRAINING2024\\2024_MRR\\"
List_month_CL = ['202401','202402','202403','202404','202405','202406','202407','202408','202409','202410','202411','202412']
List_month_MRR = ['Mk_processed202401','Mk_processed202402','Mk_processed202403','Mk_processed202404','Mk_processed202405','Mk_processed202406','Mk_processed202407','Mk_processed202408','Mk_processed202409','Mk_processed202410','Mk_processed202411','Mk_processed202412']
incomplete_months = ['Mk_processed202404','Mk_processed202405','Mk_processed202406','Mk_processed202407','Mk_processed202408','Mk_processed202409','Mk_processed202410']
path_CLt = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\DATA_TRAINING2024\\2024_CL\\202406"
path_MRRt = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\DATA_TRAINING2024\\2024_MRR\\Mk_processed202406"
#path_CLt = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\training_data_ML\\202112_CL31\\"
#path_MRRt = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\training_data_ML\\202112_MRR\\"
# CEILO Pretreatment

CEILO = None  # Initialize CEILO as None to concatenate later
CEILOt = None
all_times = []
window_trans=[]
vertical_vis=[]
month_l = []
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
MRRt = None
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

nc_files_CL = sorted([f for f in os.listdir(path_CLt) if f.endswith('.nc')])
nc_files_MRR = sorted([f for f in os.listdir(path_MRRt) if f.endswith('.nc')])

all_timet=[]
window_transt=[]
vertical_vist=[]
montht_l = []
k=12
for f in nc_files_CL:
    file_path = os.path.join(path_CLt, f)
    with Dataset(file_path,'r') as ds:
        rcs = ds.variables['rcs_0'][:,40:50]
        times_raw = ds.variables['time'][:]
        n=np.shape(rcs)[0]
        m = [k] * n
        if CEILOt is None:
            CEILOt = rcs
            all_timet = times_raw
            montht_l = m
        else :
            CEILOt = np.concatenate((CEILOt, rcs), axis=0)
            all_timet = np.concatenate((all_timet, times_raw), axis=0)
            montht_l = np.concatenate((montht_l,m),axis=0)

for f in nc_files_MRR:
    file_path = os.path.join(path_MRRt, f)
    with Dataset(file_path,'r') as ds:
        sfr = ds.variables['SnowfallRate'][:,3:4]
        sfr = np.vstack([sfr, Z])
        if MRRt is None:
            MRRt = sfr
        else:
            MRRt = np.concatenate((MRRt,sfr),axis = 0)

print(np.shape(CEILOt))
print(np.shape(MRRt))
print(np.shape(CEILO))
print(np.shape(MRR))
max_MRR = 0
for i in range(np.shape(MRRt)[0]):
    for j in range(np.shape(MRRt)[1]):
        if MRRt[i,j]<=0:
            MRRt[i,j]=np.nan
        if max_MRR<MRR[i,j]:
            max_MRR = MRR[i,j]


"""for i in range(np.shape(CEILOt)[0]):
    for j in range(np.shape(CEILOt)[1]):
        if CEILOt[i,j]==-999.0:
            CEILOt[i,j]= np.nan"""


CEILOt_mean = np.zeros((np.shape(CEILOt)[0]//2,np.shape(CEILOt)[1]))
for i in range(np.shape(CEILOt_mean)[0]):
        CEILOt_mean[i,:] = (CEILOt[2*i,:] + CEILOt[2*i+1,:])/2 

timet_mean = np.zeros(len(all_timet)//2)
for i in range(len(timet_mean)):
    timet_mean[i] = (all_timet[2*i] + all_timet[2*i+1]) / 2
"""
vertical_vist_mean = np.zeros(len(vertical_vist)//2)
for i in range(len(vertical_vist_mean)):
    vertical_vist_mean[i] = (vertical_vist[2*i] + vertical_vist[2*i+1])/2

window_transt_mean = np.zeros(len(window_transt)//2)
for i in range(len(window_transt_mean)):
    window_transt_mean[i] = (window_transt[2*i] + window_transt[2*i+1])/2
"""
montht_l_mean = np.zeros(len(montht_l)//2)
for i in range(len(montht_l_mean)):
    montht_l_mean[i]=(montht_l[2*i]+montht_l[2*i +1])/2





row_mask = ~np.isnan(CEILOt_mean).any(axis=1) & ~np.isnan(MRRt).any(axis=1)
# Appliquer le masque proprement
CEILOt_mean = CEILOt_mean[row_mask]
MRRt = MRRt[row_mask]
timet_mean = timet_mean[row_mask]
montht_l_mean = montht_l_mean[row_mask]

"""
MRR = None  # Initialize MRR
Z = np.zeros((58,1))
for k in range(len(nc_files_MRR)):
    file_path = os.path.join(base_path_MRR, nc_files_MRR[k])
    print(file_path)
    with Dataset(file_path, 'r') as ds:
        snowfall_rate = ds.variables['SnowfallRate'][:,2:3]
        #snowfall_rate = np.vstack([snowfall_rate, Z])
        if MRR is None:
            MRR = snowfall_rate
        else:
            MRR = np.concatenate((MRR, snowfall_rate), axis=0)
"""

print(np.shape(MRR))
print(np.shape(CEILO))
print(np.shape(MRRt))
print(np.shape(CEILOt))



#PREP CEILO
"""for i in range(np.shape(CEILO)[0]):
    for j in range(np.shape(CEILO)[1]):
        if CEILO[i,j]==-999.0:
            CEILO[i,j]= np.nan"""
   

#Creating CEILO_mean to smooth the data of CEIL and get the same resolution as the MRR
CEILO_mean = np.zeros((np.shape(CEILO)[0]//2,np.shape(CEILO)[1]))
for i in range(np.shape(CEILO_mean)[0]):
    CEILO_mean[i,:] = (CEILO[2*i,:] + CEILO[2*i+1,:])/2

time_mean = np.zeros(len(all_times)//2)
for i in range(len(time_mean)):
    time_mean[i] = (all_times[2*i] + all_times[2*i+1]) / 2

vertical_vis_mean = np.zeros(len(vertical_vis)//2)
for i in range(len(vertical_vis_mean)):
    vertical_vis_mean[i] = (vertical_vis[2*i] + vertical_vis[2*i+1])/2

window_trans_mean = np.zeros(len(window_trans)//2)
for i in range(len(window_trans_mean)):
    window_trans_mean[i] = (window_trans[2*i] + window_trans[2*i+1])/2

month_l_mean = np.zeros(len(month_l)//2)
for i in range(len(month_l_mean)):
    month_l_mean[i]=(month_l[2*i]+month_l[2*i +1])/2

#PREP MRR
for i in range(np.shape(MRR)[0]):
    for j in range(np.shape(MRR)[1]):
        if MRR[i,j]<=0:
            MRR[i,j]=np.nan






print(np.shape(MRR))
print(np.shape(CEILO_mean))

# Remove all the filled data
row_mask = ~np.isnan(CEILO_mean).any(axis=1) & ~np.isnan(MRR).any(axis=1)
# Appliquer le masque proprement
CEILO_mean = CEILO_mean[row_mask]
MRR = MRR[row_mask]
time_mean = time_mean[row_mask]  # même masque que CEILO_mean
vertical_vis_mean = vertical_vis_mean[row_mask]
window_trans_mean = window_trans_mean[row_mask]
month_l_mean = month_l_mean[row_mask]


ref_time = datetime.datetime(1970,1,1)  # adapter selon ton fichier
datetimes = [ref_time + datetime.timedelta(seconds=float(t)) for t in time_mean]
hours = np.array([dt.hour + dt.minute/60 + dt.second/3600 for dt in datetimes])
hour_sin = np.sin(2 * np.pi * hours / 24)
hour_cos = np.cos(2 * np.pi * hours / 24)


print(np.shape(MRR))
print(np.shape(CEILO_mean))



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
        if (MRR[k,j])==0:
            LABELS[k,j] = 0
        else:
            LABELS[k,j] = 1



#M = make_vertical_windows(CEILO_mean, MRR)
#M = M.reshape(-1,10)
#MRR = MRR.flatten()
M = CEILO_mean

    
compteur = 0
compteur_2 = 0
for k in range(np.shape(LABELS)[0]):
    if LABELS[k] == 1:
        compteur += 1
prop = compteur / np.shape(LABELS)[0]
print(f"Proportion de précipitation : {prop:.2f}")
"""
Seuil = np.shape(LABELS)[0] - 1.1*compteur
print(Seuil)
L= np.zeros((np.shape(M)[0]))
for k in range(np.shape(LABELS)[0]-1,-1,-1):
    if LABELS[k] == 0:
        L[compteur_2] = k
        compteur_2 += 1
        if compteur_2 > Seuil:
            break


M = np.delete(M, L.astype(int), axis=0)
MRR = np.delete(MRR,L.astype(int),axis = 0)
LABELS = np.delete(LABELS, L.astype(int), axis=0)
"""

prop = compteur / np.shape(LABELS)[0]
print(f"Proportion de précipitation : {prop:.2f}")
print(np.shape(M))
print(np.shape(MRR))

LABELS = LABELS.flatten()
#MRR = MRR.flatten()



M = CEILO_mean
Mt = CEILOt_mean
    


q1, q3 = np.percentile(M, [25, 75], axis=0)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
M = np.clip(M, lower_bound, upper_bound)


q1, q3 = np.percentile(Mt, [25, 75], axis=0)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
Mt = np.clip(Mt, lower_bound, upper_bound)


fs = 1e6
order = 50
cutoff = 1e3

fir_coeff = firwin(numtaps=order + 1, cutoff=cutoff, window='hamming', fs = fs) 

M_filtered = np.zeros_like(M)
for i in range(M.shape[1]):
    if M.shape[0] > 3 * len(fir_coeff):  # Vérifie que longueur est suffisante
        M_filtered[:, i] = filtfilt(fir_coeff, [1.0], M[:, i])
    else:
        raise ValueError("Signal trop court pour filtrer avec filtfilt.")
M=M_filtered


Mt_filtered = np.zeros_like(Mt)
for i in range(Mt.shape[1]):
    if Mt.shape[0] > 3 * len(fir_coeff):  # Vérifie que longueur est suffisante
        Mt_filtered[:, i] = filtfilt(fir_coeff, [1.0], Mt[:, i])
    else:
        raise ValueError("Signal trop court pour filtrer avec filtfilt.")
Mt=Mt_filtered



row_std_t = np.nanstd(Mt, axis=1)
low_var_rows_t = np.where(row_std_t < 1e-6)[0]
print(f"{len(low_var_rows_t)} lignes avec une variance quasi nulle dans Mt")

row_std = np.nanstd(M, axis=1)
low_var_rows = np.where(row_std < 1e-6)[0]
print(f"{len(low_var_rows)} lignes avec une variance quasi nulle dans M")

"""
q1, q3 = np.percentile(M, [25, 75], axis=0)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
M = np.clip(M, lower_bound, upper_bound)


q1, q3 = np.percentile(Mt, [25, 75], axis=0)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
Mt = np.clip(Mt, lower_bound, upper_bound)"""

"""Mmin = np.min(M)
Mmax = np.max(M)
M = (M - Mmin)/(Mmax - Mmin)
Mt = (Mt - Mmin)/(Mmax - Mmin)"""


#scaler = QuantileTransformer(output_distribution='normal',random_state=42)
#scaler = RobustScaler()
#M = scaler.fit_transform(M)
#Mt = scaler.transform(Mt)

"""mean_train = np.mean(M, axis=0)
std_train = np.std(M, axis=0)

mean_test = np.mean(Mt, axis=0)
std_test = np.std(Mt, axis=0)

Mt = (Mt - mean_test) * (std_train / std_test) + mean_train"""


print("NaN dans M ?", np.isnan(M).any())
print("NaN dans Mt ?", np.isnan(Mt).any())
print("Inf dans M ?", np.isinf(M).any())
print("Inf dans Mt ?", np.isinf(Mt).any())

print("Min/Max de M :", np.nanmin(M), np.nanmax(M))
print("Min/Max de Mt :", np.nanmin(Mt), np.nanmax(Mt))

# Assuming M is your data array with shape (n_samples, n_heights)
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

# Add your cyclical time feature (hour_sin)
FEATURES = np.concatenate([FEATURES, hour_sin.reshape(-1, 1)], axis=1)


ref_time = datetime.datetime(1970,1,1)  # adapter selon ton fichier
datetimes = [ref_time + datetime.timedelta(seconds=float(t)) for t in timet_mean]
hours = np.array([dt.hour + dt.minute/60 + dt.second/3600 for dt in datetimes])
hour_sin = np.sin(2 * np.pi * hours / 24)
hour_cos = np.cos(2 * np.pi * hours / 24)

# Assuming M is your data array with shape (n_samples, n_heights)
FEATURESt = np.zeros((Mt.shape[0], 7))  # Adjust for new features

for k in range(Mt.shape[0]):
    profile = Mt[k, :]
    
    # Existing features
    FEATURESt[k, 0] = np.sqrt(np.sum(profile ** 2))  # Signal energy
    FEATURESt[k, 1] = scipy.stats.skew(profile, nan_policy='omit')
    FEATURESt[k, 2] = np.corrcoef(profile[:-1], profile[1:])[0, 1] if len(profile) > 1 else 0
    x = np.arange(len(profile))
    slope, _ = np.polyfit(x, profile, 1)
    FEATURESt[k, 3] = slope
    FEATURESt[k, 4] = np.min(profile)
    FEATURESt[k, 5] = np.max(profile)
    
    # New features

    FEATURESt[k, 6] = np.mean(np.diff(profile))  # Average vertical gradient

# Add your cyclical time feature (hour_sin)
FEATURESt = np.concatenate([FEATURESt, hour_sin.reshape(-1, 1)], axis=1)


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

print(np.shape(MRR))
GRP = np.concatenate([FEATURES, M], axis=1)

i = np.argmax(MRR)
i_ = np.argmax(FEATURES[:,0])

if i == i_:
    print("good")
    print(i)

print("Max MRR : ",np.max(MRR), np.argmax(MRR))
print("Min MRR : ", np.min(MRR), np.argmin(MRR))
print("Max MRRt : ", np.max(MRRt), np.argmax(MRRt))
print("Min MRRt : ", np.min(MRRt), np.argmin(MRRt))

print("Max Feature0 : " , np.max(FEATURES[:,0]), np.argmax(FEATURES[:,0]))
print("Max Feature1 : " , np.max(FEATURES[:,1]), np.argmax(FEATURES[:,1]))
print("Max Feature2 : " , np.max(FEATURES[:,2]), np.argmax(FEATURES[:,2]))
print("Max Feature3 : " , np.max(FEATURES[:,3]), np.argmax(FEATURES[:,3]))
print("Max Feature4 : " , np.max(FEATURES[:,4]), np.argmax(FEATURES[:,4]))
print("Max Feature5 : " , np.max(FEATURES[:,5]), np.argmax(FEATURES[:,5]))
print("Max Feature6 : " , np.max(FEATURES[:,6]), np.argmax(FEATURES[:,6]))


print("Max Featuret0 : " , np.max(FEATURESt[:,0]), np.argmax(FEATURESt[:,0]))
print("Max Featuret1 : " , np.max(FEATURESt[:,1]), np.argmax(FEATURESt[:,1]))
print("Max Featuret2 : " , np.max(FEATURESt[:,2]), np.argmax(FEATURESt[:,2]))
print("Max Featuret3 : " , np.max(FEATURESt[:,3]), np.argmax(FEATURESt[:,3]))
print("Max Featuret4 : " , np.max(FEATURESt[:,4]), np.argmax(FEATURESt[:,4]))
print("Max Featuret5 : " , np.max(FEATURESt[:,5]), np.argmax(FEATURESt[:,5]))
print("Max Featuret6 : " , np.max(FEATURESt[:,6]), np.argmax(FEATURESt[:,6]))










#scaler = QuantileTransformer(n_quantiles=1000,output_distribution='normal',random_state=42)
#scaler = StandardScaler()
#FEATURESt = scaler.fit_transform(FEATURESt)
#FEATURES = scaler.fit_transform(FEATURES)



#X_full = np.concatenate([FEATURES, M], axis=1)
X_full = FEATURES
#X_full = M
y_full = MRR

#X12 = np.concatenate([FEATURESt, Mt], axis=1)
X12 = FEATURESt
#X12 = Mt
y12 = MRRt

feature_names = [f'FEATURES[{i}]' for i in range(X_full.shape[1])]
cont_vars = [f'FEATURES[{i}]' for i in range(np.shape(X_full)[1])]
dep_var = 'log_MRR'


df_full = pd.DataFrame(X_full, columns=[f'FEATURES[{i}]' for i in range(X_full.shape[1])]) # Features
df_full['MRR'] = y_full
df_full['log_MRR'] = np.log1p(df_full['MRR'])


# Discrétisation de log_MRR en quantiles
"""df_full['bin'] = pd.qcut(df_full['log_MRR'], q=5, labels=False, duplicates='drop')  # 5 bins égaux

# Visualisation
df_full['bin'].value_counts().sort_index().plot(kind='bar')
plt.title("Répartition des échantillons par bin de log_MRR")
plt.xlabel("Bin index (quantile)")
plt.ylabel("Nombre d'échantillons")
plt.grid(True)
plt.tight_layout()
plt.show()


min_count = df_full['bin'].value_counts().min()
df_balanced = df_full.groupby('bin').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)


print("✅ Nouvelle répartition équilibrée des bins :")
print(df_balanced['bin'].value_counts())


X_full = df_balanced[cont_vars].values
y_full = df_balanced['MRR'].values.reshape(-1, 1)"""

"""df_train = pd.DataFrame(X_train, columns=[f'FEATURES[{i}]' for i in range(X_full.shape[1])]) # Features
df_train['MRR'] = y_train
df_train['log_MRR'] = np.log1p(df_train['MRR'])
print(np.shape(df_train))
"""

df_test = pd.DataFrame(X12, columns=[f'FEATURES[{i}]' for i in range(X12.shape[1])])
df_test['MRR'] = y12
df_test['log_MRR'] = np.log1p(df_test['MRR'])

"""
df_12 = pd.DataFrame(FEATURES, columns=[f'FEATURES[{i}]' for i in range(FEATURES.shape[1])])
df_12['MRR']=MRR
df_12['log_MRR']= np.log1p(df_12['MRR'])"""


X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

mod = RandomForestRegressor(n_estimators=300, random_state=42)
mod.fit(X_train, y_train.ravel())

"""
rf = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
rf.fit(X_train, y_train)
"""
MRR_mean = np.mean(MRRt)
delta_ = 0.1 * MRR_mean
splits = RandomSplitter(seed=42)(df_full)
dls = TabularDataLoaders.from_df(df_full, y_names=dep_var, cont_names=cont_vars, splits=splits)
rf = tabular_learner(dls,layers=[500,300,100], y_range=(0, np.log1p(y_full).max() + 1),loss_func=lambda x,y : huber_loss(x,y,delta=0.1),  metrics=[rmse,R2Score()] )
rf.fit_one_cycle(500,slice(1e-5,1e-3),cbs=EarlyStoppingCallback(monitor='valid_loss', patience=50))
rf.show_results()
rf.recorder.plot_loss()
preds_val_log, targs_val_log = rf.get_preds(dl=dls.valid)
test_dl = dls.test_dl(df_test)
full_dl = dls.test_dl(df_full)
y12_pred, y12 = rf.get_preds(dl=test_dl)
y_test_pred, y_test = rf.get_preds(dl=full_dl)
#y_test_pred = rf.predict(X_test)
y_test_pred = np.expm1(y_test_pred)
y_test = np.expm1(y_test)
y12_pred = np.expm1(y12_pred)
#y_test = np.expm1(targs_val_log)
#y_test_pred = np.expm1(y_test_pred)
#y12_pred = rf.predict(X12)
#y12_pred = np.expm1(y12_pred)
y12 = np.expm1(y12)
r2 = r2_score(y_test, y_test_pred)
rmse = root_mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)

def moving_average(x, window=3):
    return np.convolve(x, np.ones(window)/window, mode='same')
y12_pred = moving_average(np.ravel(y12_pred))


# supression des outliers
y12_pred = np.clip(y12_pred,0,max_MRR)
y_test_pred = np.clip(y_test_pred,0,max_MRR)


print(f"📈 R² (test) : {r2:.4f}")
print(f"📉 RMSE (test) : {rmse:.4f}")
print(f"📉 MAE (test) : {mae:.4f}")


r2 = r2_score(y12, y12_pred)
rmse = root_mean_squared_error(y12, y12_pred)
mae = mean_absolute_error(y12, y12_pred)


print(f"📈 R² (test) : {r2:.4f}")
print(f"📉 RMSE (test) : {rmse:.4f}")
print(f"📉 MAE (test) : {mae:.4f}")


importances = mod.feature_importances_
for name, importance in zip(feature_names , importances):
    print(f"{name}: {importance:.4f}")


print("Train mean:", FEATURES.mean(axis=0))
print("Test mean (202403):", FEATURESt.mean(axis=0))

print(np.isnan(y_full).sum())
print((y_full <= 0).sum())
print(np.unique(y_full[:20]))

delta = (FEATURESt.mean(axis=0) - FEATURES.mean(axis=0)) / FEATURES.mean(axis=0)
for i, d in enumerate(delta):
    print(f"FEATURE[{i}] : {d*100:.1f}%")

# Résumé de performances
metrics_summary = {
    "Dataset": ["Train/Test Split", "202403"],
    "R2": [r2_score(y_test, y_test_pred), r2_score(y12, y12_pred)],
    "RMSE": [root_mean_squared_error(y_test, y_test_pred), root_mean_squared_error(y12, y12_pred)],
    "MAE": [mean_absolute_error(y_test, y_test_pred), mean_absolute_error(y12, y12_pred)]
}

# Affichage
print("\n📊 Résumé des performances :")
for i in range(2):
    print(f"{metrics_summary['Dataset'][i]} | R² = {metrics_summary['R2'][i]:.4f} | "
          f"RMSE = {metrics_summary['RMSE'][i]:.4f} | MAE = {metrics_summary['MAE'][i]:.4f}")

# Sauvegarde CSV
with open("summary_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", "R2", "RMSE", "MAE"])
    for i in range(2):
        writer.writerow([
            metrics_summary["Dataset"][i],
            round(metrics_summary["R2"][i], 4),
            round(metrics_summary["RMSE"][i], 4),
            round(metrics_summary["MAE"][i], 4)
        ])
print("✅ Résumé exporté sous summary_metrics.csv")


"""plt.hist(y_full, bins=100, alpha=0.5, label='Train')
plt.hist(np.log1p(MRRt.flatten()), bins=100, alpha=0.5, label='202403')
plt.legend()"""


for i in range(FEATURES.shape[1]):
    plt.figure()
    plt.hist(FEATURES[:, i], bins=50, alpha=0.5, label="Train")
    plt.hist(FEATURESt[:, i], bins=50, alpha=0.5, label="202403")
    plt.title(f'Distribution de FEATURE[{i}]')
    plt.legend()
    plt.grid(True)
    plt.show()



X_pca = PCA(n_components=2).fit_transform(np.vstack([FEATURES, FEATURESt]))
labels = ['train'] * len(FEATURES) + ['test'] * len(FEATURESt)

sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels)
plt.title("PCA des features : train vs test")
plt.show()



plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([0, max(y_test.max(), y_test_pred.max())], [0, max(y_test.max(), y_test_pred.max())], '--', color='gray')
plt.xlabel("Valeur réelle (y_test)")
plt.ylabel("Prédiction (y_test_pred)")
plt.title("📈 Prédictions vs Réel (test set)")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,6))
plt.scatter(y12, y12_pred, alpha=0.5)
plt.plot([0, max(y12.max(), y12_pred.max())], [0, max(y12.max(), y12_pred.max())], '--', color='gray')
plt.xlabel("Valeur réelle (y_test)")
plt.ylabel("Prédiction (y_test_pred)")
plt.title("📈 Prédictions vs Réel (202212)")
plt.grid(True)
plt.tight_layout()
plt.show()