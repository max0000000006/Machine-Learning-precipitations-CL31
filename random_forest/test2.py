import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from netCDF4 import Dataset
import os


base_path = f"C:\\Users\\maxco\\OneDrive\\Bureau\\STAGE_IPSL\\data\\training_data_ML\\"
MRR_data  = Dataset(os.path.join(base_path, "20201227_60s_MK_Ze-corrected_S-included.nc"))

# déterminer la taille moyenne et la variance des épisodes de précipitations

MRR = MRR_data.variables['SnowfallRate'][:] 
print(MRR)  # Afficher la forme de MRR

liste_longueurs = []
moyenne = 0
variance = 0

for i in range(MRR.shape[0]):
    debut = 0
    fin = 0
    for j in range(MRR.shape[1]):
        if MRR[i, j] > 0:
            if debut == 0:
                debut = j
            fin = j
    if debut != 0 and fin != 0:
         liste_longueurs.append(fin - debut + 1)

moyenne = np.mean(liste_longueurs)
variance = np.var(liste_longueurs)

print(f"Moyenne des longueurs des épisodes de précipitations : {moyenne}")
print(f"Variance des longueurs des épisodes de précipitations : {variance}")
print(liste_longueurs[len(liste_longueurs)//2])  #
plt.hist(liste_longueurs, bins=range(0, 31), edgecolor='black')
plt.title("Histogramme des longueurs d’épisodes de précipitation")
plt.xlabel("taille")
plt.ylabel("Nombre d’épisodes")
plt.grid()
plt.show()
