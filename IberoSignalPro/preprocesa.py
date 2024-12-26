import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sgnl
import wfdb
import pywt
from scipy.interpolate import interp1d
import seaborn as sbn
import plotly.graph_objs as go
from plotly.subplots import make_subplots

## Funciones


def normaliza(senial, min_val=None, max_val=None, options=None):
    """
    Normaliza la señal de entrada.

    ## Parámetros:
    * senial (array-like): Señal a normalizar.
    * min_val (float, opcional): Valor mínimo a considerar como 0. Si no se proporciona, se usará el valor mínimo de la señal.
    * max_val (float, opcional): Valor máximo a considerar como 1. Si no se proporciona, se usará el valor máximo de la señal.
    * options (str, opcional): Opciones para la normalización. Valores posibles: "std", "morum".

    ## Return:
    array-like: Señal normalizada [N_channels, N_samples].
    """

    senial = np.array(senial)  
    
    if senial.ndim == 1:  
        if options == "std":
            min_val = np.mean(senial)
            max_val = np.std(senial)
        else:
            if min_val is None:
                min_val = senial.min()
            if max_val is None:
                max_val = senial.max()
        
        norm = (senial - min_val) / (max_val - min_val)
    
    elif senial.ndim == 2:
        N, M = senial.shape
        if(N > M):
            M, N = senial.shape
            senial = np.transpose(senial)

        norm = np.zeros_like(senial, dtype=float)
        if(options == "morum"):
            min_val = np.min(senial)
            max_val = np.max(senial)
            
        for i in range(senial.shape[0]):
            if options == "std":
                min_val = np.mean(senial[i])
                max_val = np.std(senial[i])
            elif (options == None):
                if min_val is None:
                    min_val = np.min(senial[i])
                if max_val is None:
                    max_val = np.max(senial[i])

            #else:
            #    raise ValueError("las únicas opciones disponibles son: 'std' y 'morum'.")
            
            norm[i] = (senial[i] - min_val) / (max_val - min_val)
    
    else:
        raise ValueError("El arreglo debe de ser de 1D o 2D.")
    
    return norm

def pwelch_slider(data, ventana = None, encimamiento = 0.5, noverlap =0.7, Fs = 1000):
    """
    Calcula la densidad espectral de potencia usando el periodograma de Welch en ventanas deslizantes.

    ## Parámetros:
    * data (array-like): Datos de la señal.
    * ventana (int, opcional): Tamaño de la ventana para el método de Welch. Se debe especificar.
    * encimamiento (float, opcional): Porcentaje de traslape entre ventanas (por defecto, 0.5).
    * noverlap (float, opcional): Porcentaje de traslape entre segmentos de señal (por defecto, 0.7).

    ## Return:
    tuple: Un par de matrices (PSD, n) donde:
        - PSD (array): Densidad espectral de potencia estimada.
        - f: frecuencias
        - n (array): Índices de finalización de cada ventana.
    """
    
    if(ventana == None):
        raise ValueError("No está el tamaño de la ventna :(")
        
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if data.ndim == 1:
        N = 1
        M  = data.shape[0]
        data = np.array([data])
    if data.ndim == 2:
        N, M = data.shape
        if(N > M):
            data = np.transpose(data)
            N, M = data.shape

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
            Pxx[count_v, :] = sgnl.welch(data[n_sig, int(a):int(b) + 1], nperseg=ventana, noverlap=noverlap,fs=Fs, detrend='constant')[1]
            Sxx[count_v, :, n_sig - 1] = Pxx[count_v, :]
        n[count_v] = b
    
    if fin % offset < 0:
        count_v += 1
        for n_sig in range(N):
            Pxx[count_v, :] = sgnl.welch(data[n_sig, M - ventana:M], nperseg=ventana, noverlap=noverlap, fs=Fs, detrend='constant')
            Sxx[:,:,n_sig -1] = Pxx
        n[-1] = M
        
    return (Sxx), n

def sliding_PSD(signal, N=200, step = 0.1, Nfft = 400, Fs = 1000):

    n_channels = signal.shape[0]
    n_samples = signal.shape[1]

    if (n_channels > n_samples):
        signal = signal.T
        n_channels = signal.shape[0]
        n_samples = signal.shape[1]

    n_windows = int((n_samples-N) / int(N*step))

    idx = np.arange(0,n_samples-N, int(N*step))

    PSD =  np.zeros((n_channels, int(Nfft/2)+1, n_windows+1) )

    n_counter=0
    for i in idx:
        for ch in range(n_channels):
            f, PSD[ch,:,n_counter] = sgnl.welch(signal[ch,i:i+N],
                                                fs=Fs,
                                                nperseg=N//2,
                                                noverlap=int(N*0.05),
                                                nfft=Nfft,
                                                detrend='constant')
        n_counter += 1

    PSD = np.swapaxes(PSD,1,2)
    return PSD, f, idx

def band_definition(x):
    return {
        'alpha': [8,12],
        'beta' : [12,30],
        'gamma': [30,100],
        'delta': [0.1,4],
        'theta': [4,8],
        'mu'   : [12,15]
    }.get(x, [8,12])


def eloc_reader(file, coord = 'polar'):
    """
    This function loads eloc files, this files contains electrode's: index position1 position2 name. Such fileds are tab
    separated, positions could  be on polar (on default) coordinates or cartesians.
    function returns x, y, rho, theta, ch_names

    erik [dot] bojorges [at] ibero [dot] mx
    2020-06-04
    """

    eloc = np.genfromtxt(file, names = "idx,x1,x2,ch_name", dtype=None, encoding=None)

    ch_names = eloc['ch_name']
    if coord=='polar':
        rho = eloc['x2']
        theta = eloc['x1']
        x = rho * np.cos(theta*np.pi/180)
        y = rho * np.sin(theta*np.pi/180)
    else:
        x = eloc['x1']
        y = eloc['x2']
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan(y/x) * 180/np.pi

    return x,y,rho,theta, ch_names

def band_estimation(PSD, f, band = 'alpha' ):
    # PSD is assuming as a 3D matrix  [channels, taps, frequency]
    # band limits are defined by band_definition function
    # if band is not a string, then it must be a two elemnt list
    # [min_frequency, max_frequency]

    if (isinstance(band, str)):
        f_banda = band_definition(band)
    elif (len(band)==2):
        f_banda = band
    else:
        raise ValueError('band must be specified')

    id_band = np.logical_and((f > f_banda[0]),(f < f_banda[1]))
    band_power = np.trapz(PSD[:,:,id_band], f[id_band], axis=2) / np.trapz(PSD, f, axis=2)

    return band_power


def obtenerEnvolvente(senial, options="rms", param=100):
    """
    Calcula la envolvente de una señal utilizando el método de RMS o filtrado.

    ## Parámetros:
    * senial (array-like): Señal de entrada.
    * options (str, opcional): Método de cálculo de la envolvente. Valores posibles: "rms" para RMS, "filtro" para filtrado pasa bajas.
    * param (int o list, opcional): Parámetro(s) específico(s) para el cálculo de la envolvente.
        (si options = "rms", param es un valor numérico.)
        (si options = "filtro", param es un vector de forma [N_orden, 2*frec/fs].)
    

    ## Return:
    array-like: Señal que representa la envolvente calculada [N_channels, N_samples].
    """
    senial = np.array(senial)
    
    siRMS = False
    siFiltro = False
    
    if options == "rms":
        siRMS = True
    elif options == "filtro":
        siFiltro = True
    
    if senial.ndim == 1:
        rect = np.abs(senial - senial.mean())
        env = np.zeros_like(rect)

        if siRMS:
            squared_signal = np.pad(senial ** 2, (param // 2, (param - 1) // 2), mode='edge')
            env = np.sqrt(np.convolve(squared_signal, np.ones(param) / param, mode='valid'))
        elif siFiltro:
            b, a = sgnl.butter(param[0], [param[1]], 'low')
            env = sgnl.filtfilt(b, a, rect)
    
    elif senial.ndim == 2:
        N, M = senial.shape
        if(N > M):
            M, N = senial.shape
            senial = np.transpose(senial)
        env = np.zeros_like(senial, dtype=float)
        
        for i in range(N):
            rect = np.abs(senial[i] - senial[i].mean())
            
            if siRMS:
                squared_signal = np.pad(senial[i] ** 2, (param // 2, (param - 1) // 2), mode='edge')
                env[i] = np.sqrt(np.convolve(squared_signal, np.ones(param) / param, mode='valid'))
                
            elif siFiltro:
                b, a = sgnl.butter(param[0], [param[1]], 'low')
                env[i] = sgnl.filtfilt(b, a, rect)
    else:
        raise ValueError("El arreglo debe ser de 1D o 2D.")
    
    return env

def plotSignalsDF(seniales=None, fs = None, options="cuadrado"):
        """
        Grafica las señales de ECG contenidas en un DataFrame de manera interactiva utilizando Plotly.

        Parámetros:
        seniales (DataFrame): DataFrame que contiene las señales a graficar.
        options (str, opcional): Opciones de diseño del gráfico. Valores posibles: "cuadrado", "horizontal".
            Si no se proporciona, su valor por defecto es "cuadrado".
        fs (int): Frecuencia de muestreo de las señales.

        Lanza:
        ValueError: Si la variable `seniales` no es un DataFrame que contiene las señales.
                    Si la opción de despliegue no es válida ("cuadrado" o "horizontal").
                    Si no hay señales para mostrar.
        """
           
        if not isinstance(seniales, pd.DataFrame):
            raise ValueError("La variable seniales debe ser un DataFrame que contenga las señales")
        else:
            seniales = seniales.copy()
        
        if seniales is not None:
            if fs is not None:
                t = np.arange(0, len(seniales.iloc[:, 0]), 1) / fs
                xlabel = "Tiempo (s)"
            else:
                t = np.arange(0, len(seniales.iloc[:, 0]), 1)
                xlabel = "Muestras"

            num_cols = seniales.shape[1]
            if options == "horizontal":
                fig = make_subplots(rows=num_cols, cols=1, shared_xaxes=True)

                
                for i in range(num_cols):
                    max_val = seniales.iloc[:, i].abs().max()
                    fig.add_trace(go.Scatter(x=t, y=seniales.iloc[:, i], mode='lines', name=seniales.columns[i], line=dict(color='black', width=0.5)), row=i+1, col=1)
                    fig.update_yaxes(range=[-max_val, max_val], row=i+1, col=1)  # Set y-axis limits for each subplot
                
                    fig.add_hline(y=0, line=dict(color="black", width=0.5, dash="dash"), row=i+1, col=1)  # Add a horizontal line
                
                fig.update_layout(
                    xaxis_title=xlabel,
                    height=300 * num_cols,
                    width=1500,
                    showlegend=False,
                    plot_bgcolor='white'  # Set background to white
                )

                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', ticklen=10)  # Update grid settings
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                fig.show()
            elif options == "cuadrado":
                num_rows = int(np.ceil(np.sqrt(num_cols)))
                num_cols = int(np.ceil(num_cols / num_rows))

                fig = make_subplots(rows=num_rows, cols=num_cols, shared_xaxes=True, subplot_titles=seniales.columns)

                #max_val = seniales.abs().max().max()  # Calculate the maximum absolute value in the DataFrame

                for i in range(num_rows):
                    for j in range(num_cols):
                        col_index = i * num_cols + j
                        max_val = seniales.iloc[:, col_index].abs().max()
                        
                        if col_index < len(seniales.columns):
                            fig.add_trace(go.Scatter(x=t, y=seniales.iloc[:, col_index], mode='lines', name=seniales.columns[col_index], line=dict(color='black', width=0.5)), row=i + 1, col=j + 1)
                            fig.update_yaxes(range=[-max_val, max_val], row=i + 1, col=j + 1)  # Set y-axis limits for each subplot
                            fig.add_hline(y=0, line=dict(color="black", width=0.5, dash="dash"), row=i + 1, col=j + 1)  # Add a horizontal line
                        else:
                            fig.add_trace(go.Scatter(), row=i + 1, col=j + 1)

                fig.update_layout(
                    xaxis_title=xlabel,
                    height=200 * num_rows,
                    width=400 * num_cols,
                    showlegend=False,
                    plot_bgcolor='white'  # Set background to white
                )

                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', ticklen=10)  # Update grid settings
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                fig.show()
            else:
                raise ValueError("Opción de despliegue no válida")

        else:
            raise ValueError("No hay señales que mostrar")

def peakdetect(senial, long_ventana = None, distancia = None, thresh = None, fs = None):
        """
        Detecta los picos en una señal.

        Parámetros:
        senial (array-like): Señal de entrada donde se buscarán los picos.
        long_ventana (int, opcional): Longitud de la ventana para buscar picos.
        distancia (int, opcional): Distancia mínima entre picos.
        thresh (float, opcional): Umbral para considerar un punto como pico.

        Devoluciones:
        tuple: Una tupla de arrays con los índices y los valores de los picos detectados.

        """
        senial = np.array(senial)
        if long_ventana ==  None:
            long_ventana = fs
        if thresh ==  None:
            thresh = np.std(senial) * 2
        if distancia ==  None:
            distancia = fs // (fs//2)
        max_values = []
        indices_values = []

        for i in range(0, len(senial), long_ventana):
            ventana = senial[i: i + long_ventana] if (len(senial) >= i + long_ventana) else senial[i:]
            sort = np.sort(ventana)

            maximos = sort[-distancia:]
            indices_max = np.array([i + np.where(ventana == m)[0][0] for m in maximos])
            
            indices_values.extend(indices_max)
            max_values.extend(maximos)
        
        max_values = np.array(max_values)
        indices_values = np.array(indices_values)
        
        indices_max_ordenados = np.sort(indices_values)
        indices_aux = np.where(senial[indices_max_ordenados] > thresh)[0]
        indices_max_ordenados = np.sort(indices_max_ordenados[indices_aux])
        
        signal_max_ordenados = senial[indices_max_ordenados]

        indices_max_diff =np.concatenate(([1], np.diff(indices_max_ordenados)))
        inidicesMayorque = np.where(indices_max_diff >= distancia)[0]
        inidicesMayorque = np.concatenate(([0], inidicesMayorque))

        indices_picos = []
        picos = []

        for indices in range (len(inidicesMayorque)):
            if(indices == len(inidicesMayorque)-1):
                pico =np.max(signal_max_ordenados[inidicesMayorque[indices]:])
                indice_pico = np.where(signal_max_ordenados[inidicesMayorque[indices]:] == pico)[0]
            else:
                pico =np.max(signal_max_ordenados[inidicesMayorque[indices] : inidicesMayorque[indices+1]])
                indice_pico = np.where(signal_max_ordenados[inidicesMayorque[indices]: inidicesMayorque[indices+1]] == pico)[0]
            
            indice_pico += inidicesMayorque[indices]
            indices_picos.append(indices_max_ordenados[indice_pico][0])
            picos.append(pico)

        return np.array(indices_picos), np.array(picos)
    

def baseline_correction_2(senial, ms_flt_array=[0.2, 0.6], fs=None, columns=None):
    """
    Realiza la corrección de la línea de base de una señal.

    Parámetros:
    senial (array-like): Señal de entrada.
    ms_flt_array (list): Longitud de los filtros de ajuste de la línea de base (en segundos).
    fs (float): Frecuencia de muestreo de la señal.
    columns (list): Nombres de las columnas en caso de que senial sea un DataFrame.

    Devuelve:
    array-like: Señal corregida de la línea de base.
    """

    if fs is None:
        raise ValueError("Se requiere especificar la frecuencia de muestreo (fs).")

    senial = pd.DataFrame(senial, columns=columns) if isinstance(senial, (pd.DataFrame, pd.Series)) else pd.DataFrame(senial)
    N = senial.shape[1]

    mfa = [int(fs * duration) for duration in ms_flt_array]
    mfa = [m + 1 if m % 2 == 0 else m for m in mfa]  # Asegura que las longitudes sean impares

    for j in range(N):
        X0 = senial.iloc[:, j].copy()
        for mi in range(len(mfa)):
            X0 = sgnl.medfilt(X0, mfa[mi])
        
        senial.iloc[:, j] -= X0  # Corrección de línea base
        
    return senial


########## Clases
##########

########## EMG
class EMG:
    def __init__(self, fs = None, signals_raw=None, signals_flt =None, signals_env = None, nombres = None, max_iso = None):
        self.signals_raw = signals_raw
        self.signals_flt = signals_flt
        self.fs = fs
        self.signals_env = signals_env
        self.nombres =  nombres
        self.max_iso = max_iso

    def agregaSeniales(self, seniales=None, nombres=None):
        """
        Agrega nuevas señales a `self.signals_raw`.

        ## Parámetros:
        * seniales (DataFrame o array-like, opcional): Señales a agregar. 
           - Si es un DataFrame, se concatenará con `self.signals_raw`. 
           - Si es un array, se creará un DataFrame con los datos y se añadirá a `self.signals_raw`.
        * nombres (list, opcional): Nombres de las nuevas señales para el DataFrame.

        ## Devoluciones:
        DataFrame: DataFrame actualizado `self.signals_raw` con las nuevas señales agregadas.

        Lanza:
        None
        """
        if seniales is not None:
            if isinstance(seniales, pd.DataFrame):
                if self.signals_raw is None:
                    self.signals_raw = seniales.copy()
                    
                    self.nombres = self.signals_raw.columns.values
                else:
                    self.signals_raw = pd.concat([self.signals_raw, seniales], axis=1)
                    self.nombres = np.concatenate((self.nombres, self.signals_raw.columns.values))
            else:
                seniales = np.array(seniales)
                if seniales.ndim == 1:
                    df = pd.DataFrame(data=seniales, columns=nombres)
                    if self.signals_raw is None:
                        self.signals_raw = df
                    else:
                        self.signals_raw = pd.concat([self.signals_raw, df], axis=1)
                if seniales.ndim == 2:
                    N, M = seniales.shape
                    if N < M:
                        seniales = seniales.transpose()
                        
                    df = pd.DataFrame(data=seniales, columns=nombres)
                    
                    if self.signals_raw is None:
                        self.signals_raw = df
                    else:
                        self.signals_raw = pd.concat([self.signals_raw, df], axis=1)
        if(self.nombres is None):
            self.nombres = nombres          
        return self.signals_raw
    
    def obtener_envolvente(self, options="rms", param=100):
        signals_env = obtenerEnvolvente(self.signals_raw, options=options, param=param)
        self.signals_env = pd.DataFrame(signals_env.T, columns = self.nombres)
        return self.signals_env
    
    def detecta_pancitas(sefl):
        return 0

    def normaliza(self, options = "std"):
        if(options == "std"):
            normaliza(self.signals_raw.values, options=options)
        elif(options == "max_iso"):
            for i, signals in enumerate(self.signals_raw):
                self.signals_raw[i].values = normaliza(signals, min_val= np.min(self.max_iso[i]), max_val=np.max(self.max_iso[i]))
        return self.signals_raw
    
########## ECG
class ECG:
    def __init__(self, fs = None, signalder=None, signals_flt =None, HRV = None, ppm = None, nombres = None):
        self.signalder = signalder
        self.signals_flt = signals_flt
        self.fs = fs
        self.HRV = HRV
        self.ppm = ppm
        self.nombres =  nombres
    
    def agregaSeniales(self, seniales=None, nombres=["paco"]):
        """
        Agrega nuevas señales a `self.signalder`.

        ## Parámetros:
        * seniales (DataFrame o array-like, opcional): Señales a agregar. 
           - Si es un DataFrame, se concatenará con `self.signalder`. 
           - Si es un array, se creará un DataFrame con los datos y se añadirá a `self.signalder`.
        * nombres (list, opcional): Nombres de las nuevas señales para el DataFrame.

        ## Devoluciones:
        DataFrame: DataFrame actualizado `self.signalder` con las nuevas señales agregadas.

        Lanza:
        None
        """
        if seniales is not None:
            if isinstance(seniales, pd.DataFrame):
                if self.signalder is None:
                    self.signalder = seniales.copy()
                else:
                    self.signalder = pd.concat([self.signalder, seniales], axis=1)
            else:
                seniales = np.array(seniales)
                if seniales.ndim == 1:
                    df = pd.DataFrame(data=seniales, columns=nombres)
                    if self.signalder is None:
                        self.signalder = df
                    else:
                        self.signalder = pd.concat([self.signalder, df], axis=1)
                if seniales.ndim == 2:
                    N, M = seniales.shape
                    if N < M:
                        seniales = seniales.transpose()
                        
                    df = pd.DataFrame(data=seniales, columns=nombres)
                    
                    if self.signalder is None:
                        self.signalder = df
                    else:
                        self.signalder = pd.concat([self.signalder, df], axis=1)
        self.nombres = nombres          
        return self.signalder
    
    def get_median_filter_width(self, sampling_rate, duration):
        """
        Calcula el ancho del filtro de mediana para la corrección de la línea de base.

        ## Parámetros:
        * sampling_rate (float): Tasa de muestreo de la señal.
        * duration (float): Duración del filtro en segundos.

        ## Return:
        int: Ancho del filtro de mediana.
        """
        res = int(sampling_rate * duration)
        res += ((res % 2) - 1)  # Asegura un número impar para el filtro
        return res

    def baseline_correction(self, senial= any, ms_flt_array=[0.2, 0.6], fs=None, columns = None):
        """
        Realiza la corrección de la línea de base de una señal.

        ## Parámetros:
        * senial (array-like): Señal de entrada.
        * ms_flt_array (list): Longitud de los filtros de ajuste de la línea de base (en segundos).
        * fs (float): Frecuencia de muestreo de la señal.

        ## Return:
        array-like: Señal corregida de la línea de base.

        Referencia: https://www.kaggle.com/code/nelsonsharma/ecg-02-ecg-signal-pre-processing/notebook

        """
        if fs is None:
            fs = self.fs

        if(senial is any):
            senial =  self.signalder.copy()
            N = senial.shape[1]
            columns = self.signalder.columns
        else:
            if(isinstance(senial, pd.DataFrame)):
                senial = senial.copy()
                N = senial.shape[1]
                columns = senial.columns
            else:
                senial = np.array(senial)
                if senial.ndim == 1:
                    N = 1
                if senial.ndim == 2:
                    N, M = senial.shape
                    if(N > M):
                        M, N = senial.shape
                        senial = np.transpose(senial)
                if(N == len(self.signalder.columns)):
                    columns = self.signalder.columns
                else:
                    if(columns is None):
                         raise ValueError("El numero de señales no coresponde con el número de columnas, indica el nombre de las columnas con: columns=['1', '2', '3' ...].")
            senial = pd.DataFrame(data=senial, columns=columns)
                
        mfa = np.zeros(len(ms_flt_array), dtype='int')
        
        for i in range(0, len(ms_flt_array)):
            mfa[i] = self.get_median_filter_width(sampling_rate=fs, duration=ms_flt_array[i])
        

        for j in range (N):
            X0 = senial.iloc[:, j].copy()
            for mi in range(len(mfa)):
                X0 = sgnl.medfilt(X0, mfa[mi])
            
            senial[columns[j]] = np.subtract(senial[columns[j]], X0)
            self.signals_flt = senial
        return self.signals_flt
    
    def baseline_correction_2(self, senial=None, ms_flt_array=[0.2, 0.6], fs=None, columns=None):
        """
        Realiza la corrección de la línea de base de una señal de manera optimizada.

        Parámetros:
        senial (array-like): Señal de entrada.
        ms_flt_array (list): Longitud de los filtros de ajuste de la línea de base (en segundos).
        fs (float): Frecuencia de muestreo de la señal.
        columns (list): Nombres de las columnas en caso de que senial sea un DataFrame.

        Devuelve:
        array-like: Señal corregida de la línea de base.
        """
        if fs is None:
            fs = self.fs

        if senial is None:
            senial = self.signalder.copy()
            N = senial.shape[1]
            columns = self.signalder.columns
        else:
            if isinstance(senial, pd.DataFrame):
                N = senial.shape[1]
                columns = senial.columns
            else:
                senial = np.array(senial)
                if senial.ndim == 1:
                    N = 1
                elif senial.ndim == 2:
                    N = senial.shape[1]
                    senial = senial.T
                if N != len(self.signalder.columns):
                    raise ValueError("El número de señales no corresponde con el número de columnas.")
                if columns is None:
                    columns = self.signalder.columns

        mfa = np.array([self.get_median_filter_width(sampling_rate=fs, duration=duration) for duration in ms_flt_array])
        mfa = np.round(mfa).astype(int)  # Redondea y convierte a enteros

        if isinstance(senial, pd.DataFrame):
            signals_flt = senial.apply(lambda col: col - col.rolling(window=mfa, min_periods=1, center=True).median(), axis=0)
        else:
            signals_flt = np.subtract(senial, sgnl.medfilt(senial, mfa[:, np.newaxis]))

        self.signals_flt = signals_flt
        return signals_flt


    def plotSignals(self, seniales =any, options = "cuadrado"):
        """
        Grafica las señales de ECG contenidas un DataFrae.

        Parámetros:
        seniales (DataFrame, opcional): DataFrame que contiene las señales a graficar. 
            Si no se proporciona, se usan las señales almacenadas en `self.signalder`.
        options (str, opcional): Opciones de diseño del gráfico. Valores posibles: "cuadrado", "horizontal".
            Si no se proporciona, su valor default es "cuadrado"
        
        

        Lanza:
        ValueError: Si la variable `seniales` no es un DataFrame que contiene las señales.
                    Si la opción de despliegue no es válida ("cuadrado" o "horizontal").
                    Si no hay señales para mostrar.
        """
        if(seniales is any):
            seniales = self.signalder.copy()
        elif(isinstance(seniales, pd.DataFrame)):
            seniales = seniales.copy()
        else:
            raise ValueError("la variable seniales debe de ser un dataframe donde se encuentren las señales")
        
        if seniales is not None:
            if(self.fs != None):
                t = np.arange(0, len(seniales.iloc[:, 0]), 1)/self.fs
                xlabel = "Tiempo (s)"
            else:
                t = np.arange(0, len(seniales.iloc[:, 0]), 1)
                xlabel = "Muestras"
            
            num_cols = seniales.shape[1]
            if num_cols > 1:
                num_rows = int(np.ceil(num_cols / 4)) 

                if(options == "cuadrado"):
                    fig, axes = plt.subplots(num_rows, 4, figsize=(num_rows * 8, num_rows * 6))

                elif(options=="horizontal"):
                    fig, axes = plt.subplots(num_cols, 1, figsize=(21, num_cols*3))
                
                else:
                    raise ValueError("No existe esa opcion de despliegue, solo: 'cuadrado' y 'horizontal'")
                    
                axes = axes.flatten()
               
                for i in range(num_cols):
                    ax = axes[i]
                    ax.plot(t, seniales.iloc[:, i], color='black', linewidth=0.6)
                    ax.set_title(seniales.columns[i])
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel('Amplitud (mV)')
                    ax.grid(True)
                    ax.set_ylim((-1500,1500))
                    #ax.set_xlim((0,5000))
                    ax.hlines(0,0,len(seniales.iloc[:, 0])/self.fs,color='black',linestyle='dotted')
                
            else:
                plt.figure(figsize=(21, 5))
                t = np.arange(0, len(seniales.iloc[:, 0]), 1)/self.fs
                plt.plot(t, seniales.iloc[:, 0], color='black', linewidth=0.6)
                plt.hlines(0,0,len(seniales.iloc[:, 0])/self.fs,color='black',linestyle='dotted')
                plt.grid()
                
            plt.tight_layout()
            plt.show()
            
        else:
            raise ValueError("No hay señales que mostrar")   

    def peakdetect(self, senial, long_ventana = None, distancia = None, thresh = [0, 0]):
        """
        Detecta los picos en una señal.

        Parámetros:
        senial (array-like): Señal de entrada donde se buscarán los picos.
        long_ventana (int, opcional): Longitud de la ventana para buscar picos.
        distancia (int, opcional): Distancia mínima entre picos.
        thresh (tupple, opcional): [Umbral Minimo, Umbral Máximo].

        Devoluciones:
        tuple: Una tupla de arrays con los índices y los valores de los picos detectados.

        """
        senial = np.array(senial)
        if long_ventana ==  None:
            long_ventana = self.fs
        if thresh ==  [0, 0]:
            thresh[0] = np.std(senial)
            thresh[1] = np.std(senial) * 3

        if distancia ==  None:
            distancia = self.fs // (self.fs//2)
        max_values = []
        indices_values = []

        for i in range(0, len(senial), long_ventana):
            ventana = senial[i: i + long_ventana] if (len(senial) >= i + long_ventana) else senial[i:]
            sort = np.sort(ventana)

            maximos = sort[-distancia:]
            indices_max = np.array([i + np.where(ventana == m)[0][0] for m in maximos])
            
            indices_values.extend(indices_max)
            max_values.extend(maximos)
        
        max_values = np.array(max_values)
        indices_values = np.array(indices_values)
        
        indices_max_ordenados = np.sort(indices_values)
        indices_aux = np.where((senial[indices_max_ordenados] > thresh[0]) & (senial[indices_max_ordenados] < thresh[1]))[0]

        indices_max_ordenados = np.sort(indices_max_ordenados[indices_aux])
        
        signal_max_ordenados = senial[indices_max_ordenados]

        indices_max_diff =np.concatenate(([1], np.diff(indices_max_ordenados)))
        inidicesMayorque = np.where(indices_max_diff >= distancia)[0]
        inidicesMayorque = np.concatenate(([0], inidicesMayorque))

        indices_picos = []
        picos = []

        for indices in range (len(inidicesMayorque)):
            if(indices == len(inidicesMayorque)-1):
                pico =np.max(signal_max_ordenados[inidicesMayorque[indices]:])
                indice_pico = np.where(signal_max_ordenados[inidicesMayorque[indices]:] == pico)[0]
            else:
                pico =np.max(signal_max_ordenados[inidicesMayorque[indices] : inidicesMayorque[indices+1]])
                indice_pico = np.where(signal_max_ordenados[inidicesMayorque[indices]: inidicesMayorque[indices+1]] == pico)[0]
            
            indice_pico += inidicesMayorque[indices]
            indices_picos.append(indices_max_ordenados[indice_pico][0])
            picos.append(pico)

        return np.array(indices_picos), np.array(picos)
                
    def getHRV(self, signal, a,  siPlot = False):
        """
        Calcula la variabilidad de la frecuencia cardíaca (HRV) a partir de una señal ECG.

        Parámetros:
        signal (array-like): Señal de electrocardiograma (ECG).
        a (array-like): Índices de los picos R en la señal ECG.
        siPlot (bool, opcional): Indica si se debe mostrar un gráfico del ECG y la HRV.

        Devoluciones:
        tuple: Una tupla con la señal HRV y el promedio de latidos por minuto (PPM).

        """
        
        fs = self.fs
        auxHrv = np.ones_like(signal)
        diff = np.diff(a)
        diff  = np.concatenate(([diff[0]], diff))
        diff = diff/fs
        HRV = 60/diff

        ppm = np.mean(HRV)
        auxHrv = auxHrv * ppm
        print("ppm = ", ppm)

        t = np.arange(0, len(signal)/fs, 1/(((len(HRV))/len(signal)) * fs))
        t_2 = np.arange(0, t[-1], 1/fs)

        t_3 = np.arange(0, len(signal)/fs, 1/fs)

        HRV_interpid = interp1d(t, HRV, "quadratic")
        HRV_interpid = HRV_interpid(t_2)
        auxHrv[a[0]:a[0]+len(HRV_interpid)] = HRV_interpid

        if(siPlot):
            fig, ax1 = plt.subplots(figsize=(20, 5))


            ax1.plot(t_3, signal, "black", label='ECG', linewidth = 0.5)
            ax1.set_xlabel('Tiempo (s)')
            ax1.set_ylabel('Amplitud (mV)', color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid()

            ax2 = ax1.twinx()
            ax2.plot(t_3, auxHrv, "blue", label='HRV (PPM)', alpha = 0.8, linewidth=1.1)
            ax2.set_ylabel('Amplitud (PPM)', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax1.plot(a/fs, signal[a], "o", c="r", alpha=0.3, label='picos R')

            ax2_ylim_min = np.min(auxHrv) * 0.6
            ax2_ylim_max = np.max(auxHrv) * 1.4
            ax2.set_ylim(ax2_ylim_min, ax2_ylim_max)  

            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')

            plt.xlim(a[0]/fs, a[-1]/fs)

            plt.title('ECG / HRV')

            plt.show()
        self.HRV = auxHrv
        self.ppm = ppm
        return auxHrv, ppm
    
    def plotSignals_2(self, seniales=None, options="cuadrado"):
        """
        Grafica las señales de ECG contenidas en un DataFrame de manera interactiva utilizando Plotly.

        Parámetros:
        seniales (DataFrame): DataFrame que contiene las señales a graficar.
        options (str, opcional): Opciones de diseño del gráfico. Valores posibles: "cuadrado", "horizontal".
            Si no se proporciona, su valor por defecto es "cuadrado".
        fs (int): Frecuencia de muestreo de las señales.

        Lanza:
        ValueError: Si la variable `seniales` no es un DataFrame que contiene las señales.
                    Si la opción de despliegue no es válida ("cuadrado" o "horizontal").
                    Si no hay señales para mostrar.
        """
        if(seniales is None):
            seniales = self.signalder.copy()
        if not isinstance(seniales, pd.DataFrame):
            raise ValueError("La variable seniales debe ser un DataFrame que contenga las señales")
        else:
            seniales = seniales.copy()
        
        fs = self.fs
        if seniales is not None:
            if fs is not None:
                t = np.arange(0, len(seniales.iloc[:, 0]), 1) / fs
                xlabel = "Tiempo (s)"
            else:
                t = np.arange(0, len(seniales.iloc[:, 0]), 1)
                xlabel = "Muestras"

            num_cols = seniales.shape[1]
            if options == "horizontal":
                fig = make_subplots(rows=num_cols, cols=1, shared_xaxes=True)

                
                for i in range(num_cols):
                    max_val = seniales.iloc[:, i].abs().max()
                    fig.add_trace(go.Scatter(x=t, y=seniales.iloc[:, i], mode='lines', name=seniales.columns[i], line=dict(color='black', width=0.5)), row=i+1, col=1)
                    fig.update_yaxes(range=[-max_val, max_val], row=i+1, col=1)  # Set y-axis limits for each subplot
                
                    fig.add_hline(y=0, line=dict(color="black", width=0.5, dash="dash"), row=i+1, col=1)  # Add a horizontal line
                
                fig.update_layout(
                    xaxis_title=xlabel,
                    height=300 * num_cols,
                    width=1500,
                    showlegend=False,
                    plot_bgcolor='white'  # Set background to white
                )

                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', ticklen=10)  # Update grid settings
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                fig.show()
            elif options == "cuadrado":
                num_rows = int(np.ceil(np.sqrt(num_cols)))
                num_cols = int(np.ceil(num_cols / num_rows))

                fig = make_subplots(rows=num_rows, cols=num_cols, shared_xaxes=True, subplot_titles=seniales.columns)

                #max_val = seniales.abs().max().max()  # Calculate the maximum absolute value in the DataFrame

                for i in range(num_rows):
                    for j in range(num_cols):
                        col_index = i * num_cols + j
                        max_val = seniales.iloc[:, col_index].abs().max()
                        
                        if col_index < len(seniales.columns):
                            fig.add_trace(go.Scatter(x=t, y=seniales.iloc[:, col_index], mode='lines', name=seniales.columns[col_index], line=dict(color='black', width=0.5)), row=i + 1, col=j + 1)
                            fig.update_yaxes(range=[-max_val, max_val], row=i + 1, col=j + 1)  # Set y-axis limits for each subplot
                            fig.add_hline(y=0, line=dict(color="black", width=0.5, dash="dash"), row=i + 1, col=j + 1)  # Add a horizontal line
                        else:
                            fig.add_trace(go.Scatter(), row=i + 1, col=j + 1)

                fig.update_layout(
                    xaxis_title=xlabel,
                    height=200 * num_rows,
                    width=400 * num_cols,
                    showlegend=False,
                    plot_bgcolor='white'  # Set background to white
                )

                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', ticklen=10)  # Update grid settings
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

                fig.show()
            else:
                raise ValueError("Opción de despliegue no válida")

        else:
            raise ValueError("No hay señales que mostrar")

    # Signal Filtering
    # R - peak detection
    # segmentar ECG
    # Machine learning
    