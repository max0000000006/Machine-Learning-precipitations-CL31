import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import xarray as xr
import os
from fastai.metrics import Metric
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fastai.tabular.all import *
import seaborn as sns
from scipy.signal import firwin, lfilter, freqz, filtfilt
from fastai.callback.tracker import EarlyStoppingCallback



# Initialize all the path to the NetCDF files
base_path_CL = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\DATA_TRAINING2024\\2024_CL\\"
base_path_MRR = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\DATA_TRAINING2024\\2024_MRR\\"
List_month_CL = ['202401','202402','202403','202405','202406','202408','202409','202410','202411','202412']
List_month_MRR = ['Mk_processed202401','Mk_processed202402','Mk_processed202403','Mk_processed202405','Mk_processed202406','Mk_processed202408','Mk_processed202409','Mk_processed202410','Mk_processed202411','Mk_processed202412']
incomplete_months = ['Mk_processed202404','Mk_processed202405','Mk_processed202406','Mk_processed202407','Mk_processed202408','Mk_processed202409','Mk_processed202410']

# CEILO Pretreatment

CEILO = None  # Initialize CEILO as None to concatenate later

for month in List_month_CL:
    month_dir = os.path.join(base_path_CL, month)
    if not os.path.exists(month_dir):
        print(f"⚠️ Dossier non trouvé : {month_dir}")
        continue

    nc_files = sorted([f for f in os.listdir(month_dir) if f.endswith('.nc')])
    for file in nc_files:
        print(file)
        file_path = os.path.join(month_dir, file)
        try:
            with Dataset(file_path, 'r') as ds:
                beta_raw = ds.variables['rcs_0'][:, 40:50]
                if CEILO is None:
                    CEILO = beta_raw
                else:
                    CEILO = np.concatenate((CEILO, beta_raw), axis=0)
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
        print(file)
        file_path = os.path.join(month_dir, file)
        try:
            with Dataset(file_path, 'r') as ds:
                snowfall_rate = ds.variables['SnowfallRate'][:, 3:4]  # (T, 1)

                # Ajout conditionnel du vecteur Z si le mois est incomplet
                if month in incomplete_months:
                    snowfall_rate = np.vstack([snowfall_rate, Z])

                if MRR is None:
                    MRR = snowfall_rate
                else:
                    MRR = np.concatenate((MRR, snowfall_rate), axis=0)
        except Exception as e:
            print(f"❌ Erreur avec {file_path} : {e}")


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



#PREP CEILO
for i in range(np.shape(CEILO)[0]):
    for j in range(np.shape(CEILO)[1]):
        if CEILO[i,j]==-999.0:
            CEILO[i,j]=0

#Creating CEILO_mean to smooth the data of CEIL and get the same resolution as the MRR
CEILO_mean = np.zeros((np.shape(CEILO)[0]//2,np.shape(CEILO)[1]))
for i in range(np.shape(CEILO_mean)[0]):
    CEILO_mean[i,:] = (CEILO[2*i,:] + CEILO[2*i+1,:])/2 


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


print(np.shape(MRR))
print(np.shape(CEILO_mean))

"""q1, q3 = np.percentile(CEILO_mean, [10, 90], axis=0)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
CEILO_mean = np.clip(CEILO_mean, lower_bound, upper_bound)"""


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



M = make_vertical_windows(CEILO_mean, MRR)
M = M.reshape(-1,10)
LABELS = LABELS.flatten()
#MRR = MRR.flatten()

    
compteur = 0
compteur_2 = 0
for k in range(np.shape(LABELS)[0]):
    if LABELS[k] == 1:
        compteur += 1
prop = compteur / np.shape(LABELS)[0]
print(f"Proportion de précipitation : {prop:.2f}")

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


prop = compteur / np.shape(LABELS)[0]
print(f"Proportion de précipitation : {prop:.2f}")
print(np.shape(M))
print(np.shape(MRR))

LABELS = LABELS.flatten()
#MRR = MRR.flatten()

    
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

print(FEATURES[1])
print(FEATURES[2])


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
X_full = FEATURES
y_full = MRR

seuil = 0.0
mask_train = (y_full >= seuil).flatten()
mask_test  = np.ones_like(y_full, dtype=bool).flatten()

X_train = X_full[mask_train]
y_train = y_full[mask_train]

X_test  = X_full[mask_test]
y_test  = y_full[mask_test]

df_full = pd.DataFrame(X_full, columns=[f'FEATURES[{i}]' for i in range(X_full.shape[1])]) # Features
df_full['MRR'] = y_full
df_full['log_MRR'] = np.log1p(df_full['MRR'])
print(np.shape(df_full))

df_train = pd.DataFrame(X_train, columns=[f'FEATURES[{i}]' for i in range(X_full.shape[1])]) # Features
df_train['MRR'] = y_train
df_train['log_MRR'] = np.log1p(df_train['MRR'])
print(np.shape(df_train))

df_test = pd.DataFrame(X_test, columns=[f'FEATURES[{i}]' for i in range(X_full.shape[1])])
df_test['MRR'] = y_test
df_test['log_MRR'] = np.log1p(df_test['MRR'])


df_12 = pd.DataFrame(FEATURES, columns=[f'FEATURES[{i}]' for i in range(FEATURES.shape[1])])
df_12['MRR']=MRR
df_12['log_MRR']= np.log1p(df_12['MRR'])
print(np.shape(df_12))




# TRAINING
cont_vars = [f'FEATURES[{i}]' for i in range(np.shape(FEATURES)[1])]
dep_var = 'log_MRR'
splits = RandomSplitter(seed=42)(df_full)
dls = TabularDataLoaders.from_df(df_full, y_names=dep_var, cont_names=cont_vars, splits=splits, procs = [FillMissing])
learn = tabular_learner(dls,layers=[750,500,250,100], y_range=(0, np.log1p(y_full).max() + 1),loss_func=lambda x,y : huber_loss(x,y,delta = 0.1),  metrics=[rmse,R2Score()] )
learn.fit_one_cycle(500,slice(1e-5,1e-3))#,cbs=EarlyStoppingCallback(monitor='valid_loss', patience=50))
learn.export('modele2024.plk')
learn.show_results()



#RESULTS 

train_loss, valid_loss = learn.recorder.values[-1][:2]
print(f"Train loss: {train_loss:.4f}")
print(f"Valid loss: {valid_loss:.4f}")


test_dl = dls.test_dl(df_test)
full_dl = dls.test_dl(df_full)
preds_log, targs_log = learn.get_preds(dl=test_dl)
full_preds,_ = learn.get_preds(dl=full_dl)

# Revenir à l’échelle normale
preds = np.expm1(preds_log.numpy().flatten())
targs = np.expm1(targs_log.numpy().flatten())
preds = np.clip(preds, a_min=None, a_max=10)  # Cap predictions at 10
true_vals = df_test['MRR'].values
full_vals = df_full['MRR'].values
full_preds = np.expm1(full_preds.numpy().flatten())


preds_clipped = np.clip(preds, 0, y_train.max() * 2)
print(f"R² (clipped, test): {r2_score(true_vals, preds_clipped):.4f}")
print(f"MAE (clipped, test): {mean_absolute_error(true_vals, preds_clipped):.4f}")

print(f"R² (test global) : {r2_score(full_vals, full_preds):.4f}")
print(f"MAE (test global) : {mean_absolute_error(full_vals, full_preds):.4f}")

print(f"R² (test) : {r2_score(preds, targs):.4f}")
print(f"MAE (test) : {mean_absolute_error(preds, targs):.4f}")



# Prédictions sur validation
preds_log, targs_log = learn.get_preds()

r2 = r2_score(targs, preds)
print(f"R² sur la validation : {r2:.4f}")

# Créer un DataFrame pour analyser
full_dl = dls.test_dl(df_full)
full_preds,_ = learn.get_preds(dl=full_dl)
full_vals = df_full['MRR'].values
full_preds = np.expm1(full_preds.numpy().flatten())
df_val = pd.DataFrame(X_full, columns=[f'FEATURES[{i}]' for i in range(X_full.shape[1])])
df_val['MRR_pred'] = full_preds
df_val['MRR_true'] = full_vals
df_val['erreur'] = df_val['MRR_pred'] - df_val['MRR_true']
df_val['erreur_abs'] = np.absolute(df_val['erreur'])

seuil = 0.2  # à ajuster selon ton contexte
erreurs_fortes = df_val[df_val['erreur_abs'] > seuil]
print(f"{len(erreurs_fortes)} points avec erreur > {seuil}")



#PLOT
plt.figure(figsize=(6,6))
sns.scatterplot(data=df_val, x='MRR_true', y='MRR_pred', alpha=0.5)
plt.plot([0, np.max(MRR)], [0, np.max(MRR)], '--', color='gray')  # Diagonale idéale
plt.xlabel("Valeur réelle (MRR)")
plt.ylabel("Prédiction du modèle")
plt.title("📈 Valeurs réelles vs. prédites")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,4))
sns.histplot(df_val['erreur'], bins=50, kde=True, color='tomato')
plt.axvline(0, color='black', linestyle='--')
plt.xlabel("Erreur (prédit - réel)")
plt.ylabel("Nombre d'occurrences")
plt.title("📉 Distribution des erreurs de prédiction")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,6))
sns.scatterplot(data=df_val, x='MRR_true', y='MRR_pred', alpha=0.3, label='Tout')
sns.scatterplot(data=erreurs_fortes, x='MRR_true', y='MRR_pred', color='red', label='Erreurs > 0.2')
plt.plot([0, df_val['MRR_true'].max()], [0, df_val['MRR_true'].max()], '--', color='gray')
plt.xlabel("Valeur réelle (MRR)")
plt.ylabel("Prédiction")
plt.title("Valeurs réelles vs. Prédictions avec erreurs fortes en rouge")
plt.grid(True)
plt.legend()
plt.show()
