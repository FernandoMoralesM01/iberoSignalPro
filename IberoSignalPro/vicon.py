import pandas as pd
import numpy as np

class Vicon:
    
    def __init__(self, joints = None, segments = None, trajectories = None, devices = None, modelOutputs = None):
        self.joints = joints
        self.segments = segments
        self.trayectories = trajectories
        self.devices = devices
        self.modelOutputs = modelOutputs
        
    def procesar_dataframe_devices(self, dataframe_og):
        dataframe = dataframe_og.copy()
        dataframe.drop([1, 3], axis=0, inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        dataframe.columns = dataframe.iloc[0]
        dataframe.drop([0], axis=0, inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        dataframe = dataframe.iloc[:-1, :]
        return dataframe

    def procesar_dataframe(self, dataframe_og):
        dataframe = dataframe_og.copy()
        encabezado = ''
        contenido_actual = ''
        col_actual = 0
        fila_actual = 0

        while col_actual < len(dataframe.columns) :
            if dataframe.iloc[fila_actual, col_actual] != '' and col_actual+1 < len(dataframe.columns):

                encabezado = str(dataframe.iloc[fila_actual, col_actual])
                indice_punto = encabezado.find(':')

                if indice_punto != -1:
                    encabezado = encabezado[indice_punto + 1:]
                else:
                    encabezado = "Na"

                fila_actual = 0
                contenido_actual = dataframe.iloc[fila_actual+1, col_actual]
                dataframe.iloc[fila_actual+1, col_actual] = encabezado + '_' + contenido_actual + "_"  +dataframe.iloc[fila_actual+2, col_actual]
                col_actual += 1

                while dataframe.iloc[fila_actual, col_actual] == ''and col_actual+1 < len(dataframe.columns):
                    contenido_actual = dataframe.iloc[fila_actual+1, col_actual]
                    dataframe.iloc[fila_actual+1, col_actual] = encabezado + '_' + contenido_actual + "_"  +dataframe.iloc[fila_actual+2, col_actual]
                    col_actual += 1

                fila_actual = 0

            else:
                col_actual += 1

        dataframe.drop([1, 3], axis=0, inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        dataframe.columns = dataframe.iloc[0]
        dataframe.drop([0], axis=0, inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        dataframe = dataframe.iloc[:-1, :]
        return dataframe
    
    def convert_df_to_float32(self, df):
        if df is not None:
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.astype(np.float32)
        return df
    
    def leeViconCSV(self, file_path):
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            lines = file.readlines()
        dataframes = {}
        
        siExisteCategoria = [False, False, False, False, False]
        categorias = ['Devices', 'Joints', 'Segments', 'Trajectories', 'Model Outputs']
        current_category = None
        current_data = []
        

        for line in lines:
            if any(category in line for category in categorias):
                for i in range(0, len(categorias)):
                    if categorias[i] in line:
                        indexcat = categorias.index(categorias[i])
                        siExisteCategoria[indexcat] = True

                if current_data:
                    df = pd.DataFrame(current_data)
                    # Dividir la primera columna en múltiples columnas
                    df = df[0].str.split(',', expand=True)
                    # Usar la primera fila como encabezado
                    df.columns = df.iloc[0]
                    df = df.drop(0)
                    dataframes[current_category] = df
                
                current_category = line.strip().split(',')[0]
                current_data = []
            else:
                current_data.append([line.strip()])

        # Guardar los datos de la última categoría
        if current_data:
            df = pd.DataFrame(current_data)
            df = df[0].str.split(',', expand=True)
            df.columns = df.iloc[0]
            df = df.drop(0)
            dataframes[current_category] = df

        for category, df in dataframes.items():
            if category != "Devices":
                dataframes[category] = self.procesar_dataframe(dataframes[category])
            else:
                dataframes[category] = self.procesar_dataframe_devices(dataframes[category])
        
        self.devices = self.convert_df_to_float32(dataframes['Devices']) if siExisteCategoria[0] else None
        self.joints = self.convert_df_to_float32(dataframes['Joints']) if siExisteCategoria[1] else None
        self.segments = self.convert_df_to_float32(dataframes['Segments']) if siExisteCategoria[2] else None
        self.trajectories = self.convert_df_to_float32(dataframes['Trajectories']) if siExisteCategoria[3] else None
        self.modelOutputs = self.convert_df_to_float32(dataframes['Model Outputs']) if siExisteCategoria[4] else None

        return self.joints, self.devices, self.segments, self.trajectories, self.modelOutputs