import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

a,b = np.shape(CEILO) # a = 2880 , b = 770
c,d = np.shape(MRR) # c = 1440 , d = 31 
MRR = np.delete(MRR, np.s_[0:2], axis=1) 
CEILO = np.delete(CEILO, np.s_[320:770], axis=1)
CEILO = np.delete(CEILO, np.s_[0:30], axis=1)
CEILO = CEILO[::2,:]

print(np.shape(CEILO))  # Afficher la forme de CEILO
print(np.shape(MRR))    # Afficher la forme de MRR

# Création des labels en fonction de MRR
LABELS = np.zeros((np.shape(CEILO)[0],), dtype=int)  # Initialize LABELS with zeros
for k in range (np.shape(MRR)[0]):
    for j in range(np.shape(MRR)[1]):
        if (MRR[k,j])<0:
            LABELS[k] = 0
        else:
            LABELS[k] = 1
            break
FEATURES = np.zeros((np.shape(CEILO)[0],6))
for k in range(np.shape(CEILO)[0]):
    FEATURES[k][0]=(np.mean(CEILO[k,:]))
    FEATURES[k][1]=(np.var(CEILO[k,:]))
    FEATURES[k][2]=(np.max(CEILO[k,:]))
    FEATURES[k][3]=(np.min(CEILO[k,:]))
    FEATURES[k][4]=(np.sum([value**2 for value in CEILO[k,:]]))
    FEATURES[k][5]=(np.sum(CEILO[k,:] > 0) / np.shape(CEILO)[1])

# Pour augmenter la précisin est tiré le plus partie des données du MRR qui donne les informations sur les précipitations
# avec une résolution de 100 m on va regrouper les données du CEILO par paquets de 10 (*10 m = 100 m ) pour exploiter les données au maximum.

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

CEILO = make_vertical_windows(CEILO, MRR)
print(CEILO[0,3,:])
print(np.shape(CEILO))  # Afficher la forme de CEILO après la création des fenêtres verticales

LABELS = np.zeros_like(MRR , dtype=int)  # Initialize LABELS with zeros
for k in range (np.shape(MRR)[0]):
    for j in range(np.shape(MRR)[1]):
        if (MRR[k,j])<0:
            LABELS[k,j] = 0
        else:
            LABELS[k,j] = 1
            break




X_train, X_test, y_train, y_test = train_test_split(CEILO, LABELS, test_size=0.2, random_state=42)

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