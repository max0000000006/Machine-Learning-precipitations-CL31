import numpy as np
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import os

base_path_CL = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\training_data_ML\\202112_CL31\\"
base_path_MRR = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\training_data_ML\\202112_MRR\\"
nc_files_CL31 = [f for f in os.listdir(base_path_CL) if f.endswith('.nc')]
nc_files_MRR = [f for f in os.listdir(base_path_MRR) if f.endswith('.nc')]

CEILO = None  # Initialize M as None to concatenate later
for file in nc_files_CL31:
    file_path = os.path.join(base_path_CL, file)
    with Dataset(file_path, 'r') as ds:
        beta_raw = ds.variables['rcs_0'][:]
        if CEILO is None:
            CEILO = beta_raw
        else:   
            CEILO = np.concatenate((CEILO, beta_raw), axis=0)

MRR = None
for file in nc_files_MRR:
    file_path = os.path.join(base_path_MRR, file)
    with Dataset(file_path, 'r') as ds:
        snowfall_rate = ds.variables['SnowfallRate'][:]
        if MRR is None:
            MRR = snowfall_rate
        else:
            MRR = np.concatenate((MRR, snowfall_rate), axis=0)

MRR = np.delete(MRR, np.s_[0:2], axis=1) 
CEILO = np.delete(CEILO, np.s_[320:770], axis=1)
CEILO = np.delete(CEILO, np.s_[0:30], axis=1)
CEILO = CEILO[::2,:]

print(np.shape(MRR))  # Print the shape of M to verify the concatenation
print(np.shape(CEILO))  # Print the shape of M to verify the concatenation
print(len(nc_files_CL31))  # Print the number of files processed
print(len(nc_files_MRR))  # Print the number of files processed

# Création des labels en fonction de MRR
LABELS = np.zeros((np.shape(CEILO)[0]), dtype=int)  # Initialize LABELS with zeros
for k in range (np.shape(MRR)[0]):
    for j in range(np.shape(MRR)[1]):
        if (MRR[k,j])<0:
            LABELS[k] = 0
        else:
            LABELS[k] = 1
            break


# Normalisation des données
# Proportion de précipitation
# proportion de 0.10838675213675214 de précipitation ( de base )
compteur = 0
Seuil = 0.78 * np.shape(LABELS)[0]  
L= np.zeros((np.shape(CEILO)[0]))
for k in range(np.shape(LABELS)[0]-1,-1,-1):
    if LABELS[k] == 0:
        L[compteur] = k
        compteur += 1
        if compteur > Seuil:
            break
CEILO = np.delete(CEILO, L.astype(int), axis=0)
LABELS = np.delete(LABELS, L.astype(int), axis=0)
print(np.shape(CEILO))
print(np.shape(LABELS))
print(Seuil)
print(compteur)
print("Données égalisées")

compteur = 0
for k in range(np.shape(LABELS)[0]):
    if LABELS[k] == 1:
        compteur += 1
prop = compteur / np.shape(LABELS)[0]
print(f"Proportion de précipitation : {prop:.2f}")


FEATURES = np.zeros((np.shape(CEILO)[0],6))
for k in range(np.shape(CEILO)[0]):
    FEATURES[k][0]=(np.mean(CEILO[k,:]))
    FEATURES[k][1]=(np.var(CEILO[k,:]))
    FEATURES[k][2]=(np.max(CEILO[k,:]))
    FEATURES[k][3]=(np.min(CEILO[k,:]))
    FEATURES[k][4]=(np.sum([value**2 for value in CEILO[k,:]]))
    FEATURES[k][5]=(np.sum(CEILO[k,:] > 0) / np.shape(CEILO)[1])

M = np.concatenate([FEATURES, CEILO], axis=1)

# Over sampling with SMOTE 
#sm = SMOTE(random_state=42)
#X_train, y_train = sm.fit_resample(X_train, y_train)
#smote pas efficace ici car les données sont déjà équilibrées
X_train, X_test, y_train, y_test = train_test_split(CEILO, LABELS, test_size=0.2, random_state=42)
clf = RandomForestClassifier(
    n_estimators=500,
    min_samples_leaf=5,
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



probs = clf.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid()
plt.show()






















"""L=[]
for file in nc_files_CL31:
    file_path = os.path.join(base_path, file)
    with Dataset(file_path, 'r') as ds:
        beta_raw = ds.variables['rcs_0'][:]
        if np.shape(beta_raw) != (2880, 770):
            print(f"Shape mismatch in file {file}: {np.shape(beta_raw)} vs {np.shape(M)}")
            L.append(file)

print(L)
"""


