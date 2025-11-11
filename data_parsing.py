import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

cwi_well_data_path = 'cwi_data/cwi5.csv'
cwi_layer_data_path = 'cwi_data/c5st.csv'

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#point = Point(479726, 4979833)

#print(gdf[gdf.geometry.contains(point)])

df = pd.read_csv(cwi_layer_data_path, low_memory=False, on_bad_lines='skip')

df = df.dropna(subset=['strat'])
df = df[df['strat'].str.startswith('P')]

print(df[df['lith_prim'] == 'MARL'])
#print(df['lith_prim'].value_counts())

#df = df[df['strat'].str.startswith('P')]

#print(df[df['strat'] == 'PUQV']['relateid'])

