import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from netCDF4 import Dataset
import os

# 'SnowfallRate' et 'beta_smooth' sont les variables d'intérêt 

base_path = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\training_data_ML\\test\\"
CL31_1 = Dataset(os.path.join(base_path, "CALVA_CL31_202012270000.nc"))
CL31_2 = Dataset(os.path.join(base_path, "CALVA_CL31_202012271200.nc"))
CL31_fichier_nc = Dataset(os.path.join(base_path, "CALVA_CL31_20201227.nc"))
MRR_data  = Dataset(os.path.join(base_path, "20201227_60s_MK_Ze-corrected_S-included.nc"))


MRR = MRR_data.variables['SnowfallRate'][:]
CEILO = np.concatenate((CL31_1.variables['beta_raw'][:], CL31_2.variables['beta_raw'][:]), axis=0)

MRR = np.delete(MRR, np.s_[0:2], axis=1) 
CEILO = np.delete(CEILO, np.s_[320:770], axis=1)
CEILO = np.delete(CEILO, np.s_[0:30], axis=1)
CEILO = CEILO[::2,:]

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

LABELS = np.zeros_like(MRR , dtype=int)  # Initialize LABELS with zeros
for k in range (np.shape(MRR)[0]):
    for j in range(np.shape(MRR)[1]):
        if (MRR[k,j])<0:
            LABELS[k,j] = 0
        else:
            LABELS[k,j] = 1

M = make_vertical_windows(CEILO, MRR)
print(np.shape(CEILO))  # Afficher la forme de CEILO
print(np.shape(LABELS))
print(np.shape(M))  # Afficher la forme de M

M = M.reshape(-1,10)
LABELS = LABELS.reshape(-1)

print(np.shape(M))  # Afficher la forme de M après le reshape
print(np.shape(LABELS))  # Afficher la forme de LABELS après le reshape

compteur = 0
Seuil = 0.64 * np.shape(LABELS)[0]  
L= np.zeros((np.shape(M)[0]))
for k in range(np.shape(LABELS)[0]-1,-1,-1):
    if LABELS[k] == 0:
        L[compteur] = k
        compteur += 1
        if compteur > Seuil:
            break
M = np.delete(M, L.astype(int), axis=0)
LABELS = np.delete(LABELS, L.astype(int), axis=0)

compteur = 0
for k in range(np.shape(LABELS)[0]):
    if LABELS[k] == 1:
        compteur += 1
prop = compteur / np.shape(LABELS)[0]
print(f"Proportion de précipitation : {prop:.2f}")

FEATURES = np.zeros((np.shape(M)[0],6))
for k in range(np.shape(M)[0]):
    FEATURES[k][0]=(np.mean(M[k,:]))
    FEATURES[k][1]=(np.var(M[k,:]))
    FEATURES[k][2]=(np.max(M[k,:]))
    FEATURES[k][3]=(np.min(M[k,:]))
    FEATURES[k][4]=(np.sum([value**2 for value in M[k,:]]))
    FEATURES[k][5]=(np.sum(M[k,:] > 0) / np.shape(M)[1])

#M = np.concatenate([FEATURES, M], axis=1)


print(np.shape(M))  # Afficher la forme de M
print(np.shape(LABELS))  # Afficher la forme de LABELS

# Normalisation des données
X_train, X_test, y_train, y_test = train_test_split(M, LABELS, test_size=0.2, random_state=42)

clf = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)


print("\n📋 Rapport de classification :")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Matrice de confusion : précipitation vs. pas de précipitation")
plt.grid()
plt.show()