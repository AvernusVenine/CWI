import pandas as pd

completed_atlas_counties = [
    1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14, 18, 19, 20, 23, 25, 27, 28, 29, 30, 33,
    34, 37, 38, 41, 43, 47, 49, 50, 52, 53, 55, 56, 58, 59, 61, 62, 64, 65, 66,
    67, 69, 70, 71, 72, 73, 74, 77, 79, 80, 82, 85, 86
]

in_progress_counties = [
    4, 12, 15, 16, 17, 21, 22, 24, 26,
    31, 36, 39, 40, 42, 45, 46, 51,
    57, 60, 63, 75, 76, 78, 81, 87
]

cwi_well_data_path = 'well_data/cwi5.csv'
cwi_layer_data_path = 'well_data/c5st.csv'

cwi_wells = pd.read_csv(cwi_well_data_path, low_memory=False)
cwi_layers = pd.read_csv(cwi_layer_data_path, low_memory=False, on_bad_lines='skip')

cwi_wells = cwi_wells.dropna(subset=['utme', 'utmn', 'elevation'])
cwi_wells = cwi_wells[cwi_wells['county_c'].isin(in_progress_counties)]

cwi_layers = cwi_layers.dropna(subset=['strat', 'depth_top', 'depth_bot'])
cwi_layers = cwi_layers[cwi_layers['relateid'].isin(cwi_wells['relateid'])]

print(len(cwi_layers))