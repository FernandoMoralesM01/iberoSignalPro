import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
import scipy.signal as sgnl

def obtenerEnvolvente(signal, string = "rms", param=[100]):
    rect = abs(signal - signal.mean())
    env = np.zeros((1, len(rect)))
    if(string == "rms"):
      squared_signal = np.pad(signal ** 2, (param[0] // 2, (param[0] - 1) // 2), mode='edge')
      env = np.sqrt(np.convolve(squared_signal, np.ones( param[0]) /  param[0], mode='valid'))
      return env
    if(string == "filtro"):
      [b, a] = sgnl.butter(param[0], [param[1]], 'low')
      env = sgnl.filtfilt(b, a, rect)
      return env
    
def normaliza(senial, min = None, max = None):
    if(min ==  None):
        min = senial.min()
    if(max ==  None):
        max = senial.max()
    norm = (senial - min) / (max - min)                
    return norm