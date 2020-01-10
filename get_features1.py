from functions import features
import numpy as np
import pandas as pd
import os

PATH = '/home/pib3/Projetos/tuh_eeg_seizures/tuh_seizures/scripts/data/'

# use = ['EEG C3-REF', 'EEG C4-REF', 'EEG CZ-REF', 'EEG EKG1-REF',
#        'EEG F3-REF', 'EEG F4-REF', 'EEG F7-REF', 'EEG F8-REF',
#        'EEG FP1-REF', 'EEG FP2-REF', 'EEG FZ-REF', 'EEG O1-REF',
#        'EEG O2-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG PZ-REF',
#        'EEG T1-REF', 'EEG T2-REF', 'EEG T3-REF', 'EEG T4-REF',
#        'EEG T5-REF', 'EEG T6-REF', 'EMG-REF']


mx_n = np.load(PATH+'matriz_sem_crise.npy')
mx_w = np.load(PATH+'matriz_com_crise.npy')

mx_n = mx_n[:, :mx_w.shape[1]]

mx_n = np.delete(mx_n, [3, 22], axis=0)
mx_w = np.delete(mx_w, [3, 22], axis=0)

fs = 256

kw = features.curtose(mx_w,fs)
sw = features.assimetria(mx_w,fs)
vw = features.variancia(mx_w,fs)
ew = features.energia(mx_w,fs)

kn = features.curtose(mx_n,fs)
sn = features.assimetria(mx_n,fs)
vn = features.variancia(mx_n,fs)
en = features.energia(mx_n,fs)

sem_crise = np.array([kw,sw,vw,ew])
com_crise = np.array([kn,sn,vn,en])


np.save('data/EEG_sem_crise.npy',sem_crise)
np.save('data/EEG_com_crise.npy',com_crise)