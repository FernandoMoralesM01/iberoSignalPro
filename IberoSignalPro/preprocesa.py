import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
import scipy.signal as sgnl

def normaliza(senial, min = None, max = None):
    """
    Normaliza la señal de entrada.

    Parameters:
    senial (np.array): La señal a normalizar.
    min (float64): el valor minimo de la señal.
    max(float64): el valor máximo de la señal

    Returns:
    un arreglo de la señal normalizada.
    
    """

    if(min ==  None):
        min = senial.min()
    if(max ==  None):
        max = senial.max()
    norm = (senial - min) / (max - min)                
    return norm

def pwelch_slider(data, ventana = None, encimamiento = 0.5, noverlap =0.7):
    if(ventana == None):
        print("Indica la ventana")
        return None
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    try:
        N, M = data.shape
        data = np.transpose(data)
        
    except ValueError:
        M  = data.shape
        data = np.array([data])
        data = np.transpose(data)
        N = 1

    offset=np.round(ventana * (1-encimamiento))
    
    if(M > ventana):
        fin=M-ventana
        n_ventanas = fin//offset
        n_ventanas = int(n_ventanas)
    else:
        print(f"Tu ventana {M} es mayor al número de muestras de la señal")
        return None

    if fin % offset > 0:
        Sxx = np.zeros(((n_ventanas + 1), 1 + (ventana // 2), N))
        Pxx = np.zeros(((n_ventanas + 1), 1 + (ventana // 2)))
        n = np.zeros((int(n_ventanas + 1), 1))
    else:
        Sxx = np.zeros(((n_ventanas), 1 + (ventana // 2), N))
        Pxx = np.zeros(((n_ventanas), 1 + (ventana // 2)))
        n = np.zeros(((n_ventanas), 1))
    
    for count_v in range(n_ventanas):
        a = (count_v) * offset
        b = a + ventana - 1
        for n_sig in range(N):
            Pxx[count_v, :] = sgnl.welch(data[int(a):int(b) + 1, n_sig], nperseg=ventana, noverlap=noverlap)[1]
            Sxx[count_v, :, n_sig] = Pxx[count_v, :]
        n[count_v] = b
    
    if fin % offset < 0:
        for n_sig in range(N):
            Pxx[count_v, :] = sgnl.welch(data[M - ventana:M, n_sig], nperseg=ventana, noverlap=noverlap)
            Sxx[:,:,n_sig] = Pxx
        n[-1] = M
        
    return (Sxx), n

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
    
