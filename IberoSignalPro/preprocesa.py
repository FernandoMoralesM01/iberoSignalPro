import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
import scipy.signal as sgnl

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

            else:
                raise ValueError("las únicas opciones disponibles son: 'std' y 'morum'.")
            
            norm[i] = (senial[i] - min_val) / (max_val - min_val)
    
    else:
        raise ValueError("El arreglo debe de ser de 1D o 2D.")
    
    return norm

def pwelch_slider(data, ventana = None, encimamiento = 0.5, noverlap =0.7):
    """
    Calcula la densidad espectral de potencia usando el periodograma de Welch en ventanas deslizantes.

    ## Parámetros:
    * data (array-like): Datos de la señal.
    * ventana (int, opcional): Tamaño de la ventana para el método de Welch. Se debe especificar.
    * encimamiento (float, opcional): Porcentaje de traslape entre ventanas (por defecto, 0.5).
    * noverlap (float, opcional): Porcentaje de traslape entre segmentos de señal (por defecto, 0.7).

    ## Return:
    tuple: Un par de matrices (Sxx, n) donde:
        - Sxx (array): Densidad espectral de potencia estimada.
        - n (array): Índices de finalización de cada ventana.
    """
    
    if(ventana == None):
        raise ValueError("ERROR: Indica la ventana guey")
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    try:

        N, M = data.shape
        if(N > M):
            aux = N
            N = M
            M = aux
        else:   
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
        raise ValueError(f"Tu ventana {M} es mayor al número de muestras de la señal")

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
    
    if fin % offset > 0:
        count_v += 1
        for n_sig in range(N):
            Pxx[count_v, :] = sgnl.welch(data[M - ventana:M, n_sig], nperseg=ventana, noverlap=noverlap)
            Sxx[:,:,n_sig] = Pxx
        n[-1] = M
        
    return (Sxx), n

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




## Clases


class ECG:
    def __init__(self, fs = None, signalder=None, signals_flt =None):
        self.signalder = signalder
        self.signals_flt = signals_flt
        self.fs = fs
    
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
                if seniales.ndim == 2:
                    N, M = seniales.shape
                    if N < M:
                        seniales = seniales.transpose()
                        
                    df = pd.DataFrame(data=seniales, columns=nombres)
                    
                    if self.signalder is None:
                        self.signalder = df
                    else:
                        self.signalder = pd.concat([self.signalder, df], axis=1)
                    
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
            num_cols = seniales.shape[1]
            num_rows = int(np.ceil(num_cols / 4)) 

            if(options == "cuadrado"):
                fig, axes = plt.subplots(num_rows, 4, figsize=(num_rows * 8, num_rows * 6))

            elif(options=="horizontal"):
                fig, axes = plt.subplots(num_cols, 1, figsize=(21, num_cols*3))
            
            else:
                raise ValueError("No existe esa opcion de despliegue, solo: 'cuadrado' y 'horizontal'")
                
            axes = axes.flatten()
            if(self.fs != None):
                t = np.arange(0, len(seniales.iloc[:, 0]), 1)/self.fs
                xlabel = "Tiempo (s)"
            else:
                t = np.arange(0, len(seniales.iloc[:, 0]), 1)
                xlabel = "Muestras"
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
                
                
            plt.tight_layout()
            plt.show()
            
        else:
            raise ValueError("No hay señales que mostrar")               
    