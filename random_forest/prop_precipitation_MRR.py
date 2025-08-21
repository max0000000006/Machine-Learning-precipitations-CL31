import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os

L=[1,2,3,4,5]
L = np.delete(L,[1,2,4])
print(L)