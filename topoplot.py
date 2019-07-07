import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd

info = pd.read_csv("Elecinfo.csv")
info = np.array(info)

d = {}
for i in range(len(info)):
	row = info[i]
	d[row[0]] = [float(row[4]), float(row[5])]

channels_SMR = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
channels_SEED = ["FP1", "FPz", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2"]
channels_ERN = ["FP1", "FP2", "AF7", "AF3", "AF4", "AF8", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "POz", "PO8", "O1", "O2"]
channels_BMNIST = ["TP9", "FP1", "FP2", "TP10"]

print(len(channels_SMR), len(channels_SEED), len(channels_ERN), len(channels_BMNIST))

coords_SMR = []
for ch in channels_SMR:
	coords_SMR.append(d[ch])

coords_SEED = []
for ch in channels_SEED:
	coords_SEED.append(d[ch])

coords_ERN = []
for ch in channels_ERN:
	coords_ERN.append(d[ch])

coords_BMNIST = []
for ch in channels_BMNIST:
	coords_BMNIST.append(d[ch])

channels_SMR = np.array(channels_SMR)
channels_SEED = np.array(channels_SEED)
channels_ERN = np.array(channels_ERN)
channels_BMNIST = np.array(channels_BMNIST)
coords_SMR = np.array(coords_SMR)
coords_SEED = np.array(coords_SEED)
coords_ERN = np.array(coords_ERN)
coords_BMNIST = np.array(coords_BMNIST)

data1 = np.random.uniform(low=-1, high=1, size=(22,))
data2 = np.random.uniform(low=-1, high=1, size=(62,))
data3 = np.random.uniform(low=-1, high=1, size=(56,))
data4 = np.random.uniform(low=-1, high=1, size=(4,))
mne.viz.plot_topomap(data1, coords_SMR, names = channels_SMR, show_names = True)
mne.viz.plot_topomap(data2, coords_SEED, names = channels_SEED, show_names = True)
mne.viz.plot_topomap(data3, coords_ERN, names = channels_ERN, show_names = True)
mne.viz.plot_topomap(data4, coords_BMNIST, names = channels_BMNIST, show_names = True)