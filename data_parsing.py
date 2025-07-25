import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import data_refinement

cwi_well_data_path = 'cwi_data/cwi5.csv'
cwi_layer_data_path = 'cwi_data/c5st.csv'

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.DataFrame({
    'colors': ['BLK/BLU', 'GRN/YEL/BRN', 'RED', None]
})

encoded_colors = df['colors'].apply(data_refinement.one_hot_colors)

# Optionally concatenate it to your original DataFrame
df = pd.concat([df, encoded_colors], axis=1)

print(df)

#df = pd.read_csv(cwi_layer_data_path, low_memory=False, on_bad_lines='skip')
#df = df.dropna(subset=['strat'])

#df = df[df['strat'].str.startswith('P')]

#print(df[df['strat'] == 'PUQV']['relateid'])

