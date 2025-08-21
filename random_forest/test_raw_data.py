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
CL31_1 = Dataset(os.path.join(base_path, "CALVA_CL31_202012270000.nc"))
CL31_2 = Dataset(os.path.join(base_path, "CALVA_CL31_202012271200.nc"))
MRR_data  = Dataset(os.path.join(base_path, "20201227_60s_MK_Ze-corrected_S-included.nc"))


MRR = MRR_data.variables['SnowfallRate'][:]
CEILO = np.concatenate((CL31_1.variables['beta_smooth'][:], CL31_2.variables['beta_smooth'][:]), axis=0)

MRR = np.delete(MRR, np.s_[0:2], axis=1) 
CEILO = np.delete(CEILO, np.s_[320:770], axis=1)
CEILO = np.delete(CEILO, np.s_[0:30], axis=1)

CEILO = CEILO[::2,:]

LABELS = np.zeros((np.shape(CEILO)[0],), dtype=int)  # Initialize LABELS with zeros
for k in range (np.shape(MRR)[0]):
    for j in range(np.shape(MRR)[1]):
        if (MRR[k,j])<0:
            LABELS[k] = 0
        else:
            LABELS[k] = 1
            break
FEATURES = np.zeros((np.shape(CEILO)[0],6))

X_train, X_test, y_train, y_test = train_test_split(CEILO, LABELS, test_size=0.2, random_state=42)

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