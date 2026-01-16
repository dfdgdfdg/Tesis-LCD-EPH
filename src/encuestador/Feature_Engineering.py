from pathlib import Path
import pandas as pd
import numpy as np

# Carpeta base = ubicación del script actual
base_dir = Path(__file__).resolve().parent
file_path22 = base_dir / "../../data/training/EPHARG_train_22.csv"
file_path23 = base_dir / "../../data/training/EPHARG_train_23.csv"
file_path24 = base_dir / "../../data/training/EPHARG_train_24.csv"
file_path25 = base_dir / "../../data/training/EPHARG_train_25.csv"

data22 = pd.read_csv(file_path22)
data23 = pd.read_csv(file_path23)
data24 = pd.read_csv(file_path24)
data24.drop(columns = ["V2_01_M", "V2_02_M", "V2_03_M","V5_01_M", "V5_02_M", "V5_03_M"], inplace=True)
data25 = pd.read_csv(file_path25)
data25.drop(columns = ["V2_01_M", "V2_02_M", "V2_03_M","V5_01_M", "V5_02_M", "V5_03_M"], inplace=True)
data = pd.concat([data22, data23, data24, data25], ignore_index=True)
pd.set_option('display.max_columns', None)

data['logP47T'] = np.where((data['P47T'] > 0), np.log10(data['P47T']), np.nan)

#Intento reducir la cardinalidad de las variables categoricas
data['H07'] = data['H07'].map({0:0, 1: 1, 2: 0}) 
data['H10'] = data['H10'].map({0:0, 1: 1, 2: 0}) 
data['H11'] = data['H11'].map({0:0,1: 1, 2: 0}) 
data['P07'] = data['P07'].map({0:0,1: 1, 2: 0}) 
data['P05'] = data['P05'].map({0:0,1: 1, 2: 0}) 
data['P08'] = data['P08'].map({0:0,1: 1, 2:1, 3:3,9:0}) 
data['P09'] = data['P09'].map({0:0,9:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8}) 
data['PP07G_59'] = data['PP07G_59'].map({0:0, 5:5,1:0}) 

data = data.astype({'ANO4': 'category', 'TRIMESTRE': 'category', 'AGLOMERADO': 'category', 'V01': 'category', 'H05': 'category', 
                    'H06': 'category', 'H08': 'category', 'H09':'category','H12':'category', 'PROP':'category', 'H14':'category','H13':'category',
                    'CAT_INAC':'category', 'CAT_OCUP':'category', 'CH07':'category', 'P10':'category','P05':'category'
                    ,'PP07G1':'category', 'PP07G2':'category','PP07G3':'category', 'PP07G4':'category','PP07H':'category',
                    'PP07I':'category', 'PP07J':'category', 'PP07K':'category','Region':'category', 'P08':'category', 'P09':'category','CONDACT':'category','PP07G_59':'category', 'P02':'category'})


#Intento reducir registros por categoria, quiero reducir cardinalidad de variables categoricas
data['V01'] = data['V01'].map({0:0,1:1,2:2,3:0,4:0,5:0,6:0}) 
data['H05'] = data['H05'].map({0:0,1:1,2:2,3:0,4:0}) 
data['H06'] = data['H06'].map({0:0,1:1,2:2,3:3,4:4,5:0,6:0,7:0,9:9}) 
data['H08'] = data['H08'].map({0:0,1:1,2:0,3:0})
data['H12'] = data['H12'].map({0:0,1:1,2:2,3:0,4:0})
data['PROP'] = data['PROP'].map({0:0,1:1,2:2,3:3,4:0,5:0})
data['H14'] = data['H14'].map({0:0,1:1,2:2,3:3,4:0,5:0})
data['H13'] = data['H13'].map({0:0,1:1,2:0,4:0})
data['P10'] = data['P10'].map({0:0,1:1,2:2,9:0})
#Todas las columnas de arriba son categoricas, las convierto a category
data = data.astype({'V01': 'category', 'H05': 'category', 'H06': 'category', 'H08': 'category', 
                    'H12':'category', 'PROP':'category', 'H14':'category','H13':'category','P10':'category'})
categorical_columns = data.select_dtypes(include=['category']).columns

#Quiero hacer nuevas columnas "dummy" para ANIO y TRIMESTRE
data['ANIO_2022_Dummy'] = (data['ANO4'] == 2022).astype(int)
data['ANIO_2023_Dummy'] = (data['ANO4'] == 2023).astype(int)
data['ANIO_2024_Dummy'] = (data['ANO4'] == 2024).astype(int)
data['ANIO_2025_Dummy'] = (data['ANO4'] == 2025).astype(int)
data['TRIMESTRE_1_Dummy'] = (data['TRIMESTRE'] == 1).astype(int)
data['TRIMESTRE_2_Dummy'] = (data['TRIMESTRE'] == 2).astype(int)
data['TRIMESTRE_3_Dummy'] = (data['TRIMESTRE'] == 3).astype(int)
data['TRIMESTRE_4_Dummy'] = (data['TRIMESTRE'] == 4).astype(int)

target_cols_reg = { 'P21', 'T_VI', 'V12_M', 'V2_M', 'V3_M', 'V5_M', 'TOT_P12', 'PP08D1'}

for col in target_cols_reg:
    new_col_name = 'log' + col
    data[new_col_name] = np.where((data[col] > 0), np.log10(data[col]), np.nan)


def get_age_group(age):
    if age <= 5:
        return '0-5'
    elif age <= 14:
        return '6-14'
    elif age <= 24:
        return '15-24'
    elif age <= 64:
        return '25-64'
    else:
        return '65+'

data['Rango_Etario'] = data['P03'].apply(get_age_group)

age_group_counts = data.groupby(['CODUSU', 'Rango_Etario']).size().unstack(fill_value=0)

for age_group in age_group_counts.columns:
    data[f'Personas_{age_group}'] = data['CODUSU'].map(age_group_counts[age_group])

data.drop(columns=['Rango_Etario'], inplace=True)

#Maximo nivel educativo en el hogar

data['Max_Nivel_Educativo'] = data.groupby(['CODUSU'])['P09'].transform(lambda s: s.astype(int).max())

print(data.head(20))