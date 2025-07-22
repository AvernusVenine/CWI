import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

cwi_well_data_path = 'cwi_data/cwi5.csv'
cwi_layer_data_path = 'cwi_data/c5st.csv'

df = pd.read_csv(cwi_well_data_path, low_memory=False)
df['data_src'] = df['data_src'].str.strip()

print(df[df['data_src'] == 'MGS'].size)


