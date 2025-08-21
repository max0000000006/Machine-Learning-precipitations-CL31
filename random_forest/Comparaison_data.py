import numpy as np
import xarray as xr
from netCDF4 import Dataset
import os

base_path = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\training_data_ML\\"
CL31_1 = Dataset(os.path.join(base_path, "CALVA_CL31_202012270000.nc"))
CL31_2 = Dataset(os.path.join(base_path, "CALVA_CL31_202012271200.nc"))

M1 = np.concatenate((CL31_1.variables['beta_raw'][:], CL31_2.variables['beta_raw'][:]), axis=0)

CL31_fichier_nc = Dataset(os.path.join(base_path, "CALVA_CL31_20201227.nc"))

M2 = CL31_fichier_nc.variables['rcs_0'][:]

if M1.shape != M2.shape:
    print(f"Les matrices ont des formes différentes : M1={M1.shape}, M2={M2.shape}")


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

base_path = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\training_data_ML\\"
CEILO_data = Dataset(os.path.join(base_path, "CALVA_CL31_20201227.nc"))
MRR_data  = Dataset(os.path.join(base_path, "20201227_60s_MK_Ze-corrected_S-included.nc"))


CEILO = CEILO_data.variables['rcs_0'][:]
MRR = MRR_data.variables['SnowfallRate'][:]
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

#test à finir plus tard
def make_vertical_windows(X, Y , window_size=5, stride=1, threshold=0.5):
    T, H = X.shape
    segments = []
    labels = []
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if np.ma.is_masked(Y[i, j]):
                Y[i, j] = 0
            else:
                Y[i, j] = 1
    for t in range(T):  # pour chaque instant
        while i + window_size <= H:  # pour chaque fenêtre verticale
            segment = X[t, i : i + window_size]
            if sum(MRR[t//2,i*10 : i*10 + window_size*10]) < threshold:
                labels[t] = y[t]  # même label pour tous les sous-profils à t
            segments.append(segment)
            labels.append(label)
    return np.array(segments), np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(FEATURES, LABELS, test_size=0.2, random_state=42)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
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

