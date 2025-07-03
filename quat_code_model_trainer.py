import pandas as pd

cwi_well_data_path = ''
cwi_layer_data_path = ''

model_save_path = ''

cwi_wells = pd.read_csv(cwi_well_data_path, low_memory=False)
cwi_layers = pd.read_csv(cwi_layer_data_path, low_memory=False)

na_values = {'elevation': 0, 'utme': -1, 'utmn': -1, 'data_src': -1}
cwi_wells = cwi_wells.fillna(values=na_values)

cwi_layers = cwi_layers.dropna(subset=['strat'])
