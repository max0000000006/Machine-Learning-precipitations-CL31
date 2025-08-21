import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from netCDF4 import Dataset
import os
import seaborn as sns


base_path_CL = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\training_data_ML\\202212_CL31\\"
base_path_MRR = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\training_data_ML\\202212_MRR\\"
nc_files_CL31 = sorted([f for f in os.listdir(base_path_CL) if f.endswith('.nc')])
nc_files_MRR = sorted([f for f in os.listdir(base_path_MRR) if f.endswith('.nc')])

CEILO = None  
CBH = None
# Initialize CEILO as None to concatenate later
#file_path_CL = os.path.join(base_path_CL, nc_files_CL31[0])  # Just to check the first file
#CEILO = Dataset(file_path_CL, 'r').variables['rcs_0'][:]

for k in range(len(nc_files_CL31)):
    file_path = os.path.join(base_path_CL, nc_files_CL31[k])
    with Dataset(file_path, 'r') as ds:
        beta_raw = ds.variables['rcs_0'][:]
        cbh = ds.variables['cloud_base_height'][:]
        if CBH is None:
            CBH = cbh
        else:
            CBH = np.concatenate((CBH, cbh), axis=0)
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

CBH = CBH[::2,:]  # Prendre un échantillon tous les 2 pour réduire la taille
MRR = np.delete(MRR, np.s_[10:], axis=1)
MRR = np.delete(MRR, np.s_[0:3], axis=1) 
CEILO = np.delete(CEILO, np.s_[100:], axis=1)
CEILO = np.delete(CEILO, np.s_[0:30], axis=1)
CEILO = CEILO[::2,:]
print(np.shape(CBH))  # Afficher la forme de CBH
print(np.shape(CEILO))  # Afficher la forme de CEILO
print(np.shape(MRR))  # Afficher la forme de MRR

# Moyenne de MRR sur des blocs de 10 indices selon la deuxième dimension (axis=1)
block_size = 10
num_blocks = CEILO.shape[1] // block_size
CEILO_avg = np.array([np.mean(CEILO[:, i*block_size:(i+1)*block_size], axis=1) for i in range(num_blocks)]).T
print(np.shape(CEILO_avg))  # Afficher la forme de CEILO_avg
print(CEILO_avg[0,:])  # Afficher la première ligne de CEILO_avg
print(CEILO[0,:])

L=[]
for k in range(np.shape(MRR)[0]):
    for i in range(np.shape(MRR)[1]):
        if MRR[k,i] < 0:
            L.append(k)
        if CBH[k,0]>0 and CBH[k,0]<100: 
            L.append(k)

CEILO_avg = np.delete(CEILO_avg, L, axis=0)
MRR = np.delete(MRR, L, axis=0)

print(np.shape(CEILO_avg))  # Afficher la forme de CEILO_avg après suppression
print(CEILO_avg[0,:])
print(MRR[0,:])
X = CEILO_avg
Y = MRR

"""X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train.ravel())

Y_pred = model.predict(X_test)

sns.scatterplot(x=X.ravel(), y=Y.ravel(), alpha=0.3)
plt.xlabel("Moyenne CEILO (10 niveaux)")
plt.ylabel("MRR")
plt.title("Relation brute entre CEILO (moyenne) et MRR")
plt.grid(True)
plt.show()

print("Score R² :", r2_score(Y_test.ravel(), Y_pred))"""

C_MEAN = np.array([np.mean(CEILO_avg[i,:]) for i in range(np.shape(CEILO_avg)[0])])
MRR_MEAN = np.array([np.mean(MRR[i,:]) for i in range(np.shape(MRR)[0])])

plt.figure(figsize=(10, 6))

n_samples = 500  # nombre de courbes à tracer
idx = np.random.choice(range(CEILO.shape[0]), size=n_samples, replace=False)

"""for i in range(n_samples):
    plt.plot(CEILO_avg[i, :], MRR[i, : ], label=None, color='blue', alpha=0.1)"""



plt.scatter(C_MEAN,MRR_MEAN,color='red',marker='v')
plt.xlabel("Valeur CEILO")
plt.ylabel("Valeur MRR")
plt.title(f"{n_samples} courbes CEILO (x en fonction de l'altitude)")
plt.grid(True)
plt.show()

