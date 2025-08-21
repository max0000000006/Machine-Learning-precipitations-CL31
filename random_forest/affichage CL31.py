import os 
from netCDF4 import Dataset 
import numpy as np
import matplotlib.pyplot as plt
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


base_path = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\DataNc\\2020\\202001\\"
CL31 = Dataset(os.path.join(base_path, "CALVA_CL31_20200201.nc"))
#M = np.concatenate((M1, M2), axis=0)
M = CL31.variables["rcs_0"][:]

print(CL31.variables["rcs_0"])
"""VAL = np.zeros_like(M[::2,:])
print(np.shape(M))
print(np.shape(VAL))
for i in range(VAL.shape[0]):
    for j in range(M.shape[1]):
        if np.ma.is_masked(M[2*i, j]):
            VAL[i, j] = 0
        else:
            VAL[i,j] = (M[2*i,j]+M[2*i+1,j]) / 2


SFR = MRR.variables["SnowfallRate"][:]
print(np.shape(SFR))
print(np.shape(VAL))
SFR = np.delete(SFR, np.s_[10:], axis=1)
SFR = np.delete(SFR, np.s_[0:3], axis=1) 
VAL = np.delete(VAL, np.s_[100:], axis=1)
VAL = np.delete(VAL, np.s_[0:30], axis=1)
block_size = 10
num_blocks = VAL.shape[1] // block_size
CEILO_avg = np.array([np.mean(VAL[:, i*block_size:(i+1)*block_size], axis=1) for i in range(num_blocks)]).T
print(np.shape(CEILO_avg))  # Afficher la forme de CEILO_avg
print(CEILO_avg[0,:])  # Afficher la première ligne de CEILO_avg
print(VAL[0,:])

L=[]
for k in range(np.shape(SFR)[0]):
    for i in range(np.shape(SFR)[1]):
        if SFR[k,i] < 0:
            L.append(k)

CEILO_avg = np.delete(CEILO_avg, L, axis=0)
SFR= np.delete(SFR, L, axis=0)

print(np.shape(CEILO_avg))  # Afficher la forme de CEILO_avg après suppression
print(CEILO_avg[0,:])
print(SFR[0,:])
X = CEILO_avg
Y = SFR

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train.ravel())

Y_pred = model.predict(X_test)

sns.scatterplot(x=X.ravel(), y=Y.ravel(), alpha=0.3)
plt.xlabel("Moyenne CEILO (10 niveaux)")
plt.ylabel("MRR")
plt.title("Relation brute entre CEILO (moyenne) et MRR")
plt.grid(True)
plt.show()

print("Score R² :", r2_score(Y_test.ravel(), Y_pred))

plt.figure(figsize=(10, 6))"""
"""C_MEAN = np.array([np.mean(CEILO_avg[i,:]) for i in range(np.shape(CEILO_avg)[0])])
MRR_MEAN = np.array([np.mean(SFR[i,:]) for i in range(np.shape(CEILO_avg)[0])])
print(np.shape(C_MEAN))
print(np.shape(MRR_MEAN))

n_samples = 37  # nombre de courbes à tracer
idx = np.random.choice(range(VAL.shape[0]), size=n_samples, replace=False)
print(np.shape(CEILO_avg))
print(np.shape(SFR))



plt.scatter(C_MEAN, MRR_MEAN, label=None, color='red', marker="v")
plt.xlabel("Valeur CEILO")
plt.ylabel("Valeur MRR")
plt.title(f"{n_samples} courbes CEILO (x en fonction de l'altitude)")
plt.grid(True)
plt.show()


















nc_files_CL31 = sorted([f for f in os.listdir(base_path_CL) if f.endswith('.nc')])
file_path_CL = os.path.join(base_path_CL, nc_files_CL31[0])  # Just to check the first file
data = Dataset(file_path_CL, 'r')
CEILO = data.variables['rcs_0'][:].T
print(data.variables.keys())
BCKGRD = data.variables['bckgrd_rcs_0'][:]
print(np.shape(BCKGRD))
print(BCKGRD[0])
CEILO_corrected  = np.zeros_like(CEILO)
x_min, x_max = -20000, 20000
y_min, y_max = 1e-7, 1e-4


# Fonction d'interpolation linéaire
for k in range(np.shape(CEILO)[0]):
    for i in range(np.shape(CEILO)[1]):
        CEILO_corrected[k,i] = y_min + (y_max - y_min) * (CEILO[k,i] - x_min) / (x_max - x_min)
        a
plt.figure(figsize=(10, 6))
plt.imshow(CEILO_corrected, aspect='auto', cmap='viridis', origin='lower', alpha=1)  # Overlay CEILO data
plt.colorbar(label='rcs_0')
plt.title('rcs_0 from CL31 Data') 
plt.show()"""
