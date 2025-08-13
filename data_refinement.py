import os
import pandas as pd
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import random
import geopandas as gpd
from shapely.geometry import Point

import utils
from utils import Field

cwi_well_data_path = 'cwi_data/cwi5.csv'
cwi_layer_data_path = 'cwi_data/c5st.csv'
feature_data_path = 'compiled_data/data.parquet'

depth_to_bedrock_path = 'map_data/depth_to_bedrock_2020.npa'

pca = joblib.load('trained_models/util/embedding_pca.joblib') if os.path.exists(
    'trained_models/util/embedding_pca.joblib') else None
scaler = joblib.load('trained_models/util/scaler.joblib') if os.path.exists('trained_models/util/scaler.joblib') else None
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# A custom SMOTE function that only creates faux data points from given data columns while duplicating the rest
def fit_smote(X_df : pd.DataFrame, y_df : pd.DataFrame, count : int, label : int, label_col : str, random_state : int, data_cols : list):

    random.seed(random_state)

    X_filtered_df = X_df[y_df[label_col] == label]
    df_size = X_filtered_df.shape[0]

    X_new = []
    y_new = []

    for i in range(count):
        df_index_one = random.randint(0, df_size - 1)
        df_index_two = random.randint(0, df_size - 1)

        new_row = X_filtered_df.iloc[df_index_one].copy()

        for col in data_cols:
            df_val_one = X_filtered_df.iloc[df_index_one][col]
            df_val_two = X_filtered_df.iloc[df_index_two][col]

            rand_float = random.uniform(min(df_val_one, df_val_two), max(df_val_one, df_val_two))

            new_row[col] = rand_float

        X_new.append(new_row)
        y_new.append(label)

    X_new = pd.DataFrame(X_new)
    y_new = pd.DataFrame(y_new, columns=[label_col])

    return pd.concat([X_df, X_new], ignore_index=True), pd.concat([y_df, y_new], ignore_index=True)

# Performs PCA on a given dataframe of sentence embeddings
def embed_pca(df : pd.DataFrame):
    pca = PCA(n_components=0.95)

    pca_embeddings = pca.fit_transform(df)
    pca_embeddings_df = pd.DataFrame(pca_embeddings, columns=[f'pca_emb_{i}' for i in range(pca_embeddings.shape[1])])

    joblib.dump(pca, 'trained_models/util/embedding_pca.joblib')

    return pca_embeddings_df

# Performs pca on a given dataframe of onehot colors
def color_pca(df : pd.DataFrame):
    pca = PCA(n_components=.95)

    pca_colors = pca.fit_transform(df)
    pca_colors_df = pd.DataFrame(pca_colors, columns=[f'pca_color_{i}' for i in range(pca_colors.shape[1])])

    joblib.dump(pca, 'trained_models/util/color_pca.joblib')

    return pca_colors_df

# Scales a given list of columns in a dataframe then saves the Scaler
def df_scaler(df : pd.DataFrame, cols : list):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])

    joblib.dump(scaler, 'trained_models/util/scaler.joblib')

    return df

# Embeds a given column of a DataFrame via MiniLM L6 v2
def embed_descriptions(df : pd.DataFrame):
    embeddings = embedding_model.encode(df[Field.DRILLER_DESCRIPTION].tolist(), show_progress_bar=True)
    embedding_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])

    return embedding_df

# Use to onehot encode the color of a given series
def one_hot_colors(color):
    series = pd.Series([0] * len(utils.COLORS), index=utils.COLORS)

    if pd.isna(color):
        return series

    if color in utils.COLORS:
        series[color] = 1
        return series

    color = color.replace('Dk. ', '')
    color = color.replace('Lt. ', '')
    color = color.strip()

    color_parts = color.split('/')
    translated = {utils.COLOR_ABBREV_MAP.get(p.strip()) for p in color_parts if p.strip() in utils.COLOR_ABBREV_MAP}

    for c in translated:
        series[c] = 1

    return series

def calculate_weights(df : pd.DataFrame):

    for index, row in df.iterrows():

        cuttings = True if df.at[index, Field.CUTTINGS] == 'Y' else False
        core = True if df.at[index, Field.CORE] == 'Y' else False

        if cuttings and core:
            df.at[index, Field.WEIGHT] = 10
        if cuttings or core:
            df.at[index, Field.WEIGHT] = 4
        else:
            df.at[index, Field.WEIGHT] = 1

    return df

def recalculate_weights(df : pd.DataFrame):

    df[Field.WEIGHT] = df[Field.WEIGHT].replace({
        1 : .5,
        4 : 2,
        10 : 5
    })

    return df

def find_depth_to_bedrock(df : pd.DataFrame):
    np_array = joblib.load(depth_to_bedrock_path)

    x_origin = 189750
    y_origin = 5472480
    cell_size = 90

    x_shifted = ((df[Field.UTME].astype(int) - x_origin) // cell_size).astype(int)
    y_shifted = ((y_origin - df[Field.UTMN].astype(int)) // cell_size).astype(int)

    depths = np_array[y_shifted.to_numpy(), x_shifted.to_numpy()]
    df[Field.DEPTH_TO_BEDROCK] = depths

    df[Field.TOP_DEPTH_TO_BEDROCK] = depths - df[Field.DEPTH_TOP]
    df[Field.BOT_DEPTH_TO_BEDROCK] = depths - df[Field.DEPTH_BOT]

    return df

# Finds the first bedrock layers beneath a given dataframe of wells
def find_first_bedrock_age(df : pd.DataFrame):
    for path in utils.SHAPEFILE_PATHS:
        full_path = f'map_data/s21_files_only/{path}.shp'

        gdf = gpd.read_file(full_path)

        for index, row in df.iterrows():
            point = Point(row[Field.UTME], row[Field.UTMN])
            intersection = gdf[gdf.geometry.contains(point)]

            if intersection.empty:
                continue

            if path == 'kc_pg':
                layer = intersection.iloc[0]['MAP_LABEL']
            else:
                layer = intersection.iloc[0]['MAPLABEL']


            if layer is not None and layer.startswith(('A', 'M', 'P')):
                layer = layer[0]

            age = utils.FIRST_BEDROCK_CATEGORIES.get(layer, -1)

            df.at[index, Field.FIRST_BEDROCK_CATEGORY] = age

    return df

# Loads and refines all well layer data
def load_and_refine_data():
    pd.set_option('future.no_silent_downcasting', True)

    cwi_wells = pd.read_csv(cwi_well_data_path, low_memory=False, on_bad_lines='skip')
    cwi_layers = pd.read_csv(cwi_layer_data_path, low_memory=False, on_bad_lines='skip')

    cwi_wells = cwi_wells.dropna(subset=[Field.ELEVATION, Field.UTMN, Field.UTME])

    cwi_wells[Field.CUTTINGS] = cwi_wells[Field.CUTTINGS].replace({'Y': 1, None: 0})
    cwi_wells[Field.CORE] = cwi_wells[Field.CORE].replace({'Y' : 1, None : 0})

    cwi_layers = cwi_layers.drop(
        columns=['objectid', 'c5st_seq_no', 'wellid', 'concat', 'stratcode_gen', 'stratcode_detail'])

    cwi_layers = cwi_layers.dropna(subset=[Field.STRAT])
    cwi_layers.loc[cwi_layers[Field.DEPTH_BOT].isna(), Field.DEPTH_BOT] = cwi_layers[Field.DEPTH_TOP]
    cwi_layers = cwi_layers.fillna(value={
        Field.COLOR: 'UNKNOWN',
        Field.DRILLER_DESCRIPTION: ''
    })

    # Replace certain faux 'ages' with their true meaning for easier interpretation
    cwi_layers[Field.STRAT] = cwi_layers[Field.STRAT].replace({
        'RMMF': 'FILL',
        'PITT': 'X',
        'PVMT': 'Y'
    })
    cwi_layers[Field.AGE] = cwi_layers[Field.STRAT].str[0]

    # Put all underrepresented ages into one Bucket for human interpretation
    cwi_layers[Field.AGE] = cwi_layers[Field.AGE].str.replace(r'^[WJBNI]', 'Z', regex=True)

    cwi_layers[Field.ELEVATION] = cwi_layers[Field.RELATEID].map(cwi_wells.set_index(Field.RELATEID)[Field.ELEVATION])
    cwi_layers[Field.UTME] = cwi_layers[Field.RELATEID].map(cwi_wells.set_index(Field.RELATEID)[Field.UTME])
    cwi_layers[Field.UTMN] = cwi_layers[Field.RELATEID].map(cwi_wells.set_index(Field.RELATEID)[Field.UTMN])
    cwi_layers[Field.DATA_SOURCE] = cwi_layers[Field.RELATEID].map(cwi_wells.set_index(Field.RELATEID)[Field.DATA_SOURCE])
    cwi_layers[Field.CORE] = cwi_layers[Field.RELATEID].map(cwi_wells.set_index(Field.RELATEID)[Field.CORE])
    cwi_layers[Field.CUTTINGS] = cwi_layers[Field.RELATEID].map(cwi_wells.set_index(Field.RELATEID)[Field.CUTTINGS])
    cwi_layers[Field.INTERPRETATION_METHOD] = cwi_layers[Field.RELATEID].map(cwi_wells.set_index(Field.RELATEID)[Field.INTERPRETATION_METHOD])

    cwi_layers = cwi_layers.dropna(subset=[Field.UTME, Field.UTMN, Field.ELEVATION])

    cwi_layers[Field.AGE_CATEGORY] = cwi_layers[Field.AGE].map(utils.AGE_CATEGORIES)

    cwi_wells = find_first_bedrock_age(cwi_wells)
    cwi_layers[Field.FIRST_BEDROCK_CATEGORY] = cwi_layers[Field.RELATEID].map(cwi_wells.set_index(Field.RELATEID)[Field.FIRST_BEDROCK_CATEGORY])

    cwi_wells = calculate_weights(cwi_wells)
    cwi_layers[Field.WEIGHT] = cwi_layers[Field.RELATEID].map(cwi_wells.set_index(Field.RELATEID)[Field.WEIGHT])

    embeddings = embed_descriptions(cwi_layers)
    pca_embeddings = embed_pca(embeddings)

    # All except for depth_to_bdrk, age_category and previous_age_category
    layer_features = cwi_layers[utils.LAYER_FEATURE_COLS]

    features_df = pd.concat([layer_features.reset_index(drop=True), pca_embeddings.reset_index(drop=True)], axis=1)

    features_df = features_df.sort_values(by=[Field.RELATEID, Field.DEPTH_TOP], ascending=[True, True])
    features_df[Field.PREVIOUS_AGE_CATEGORY] = (
        features_df.groupby(Field.RELATEID)[Field.AGE_CATEGORY]
        .shift(1)
        .fillna(-1)
        .astype(int)
    )

    features_df = find_depth_to_bedrock(features_df)
    #features_df = find_first_bedrock_age(features_df)

    features_df = df_scaler(features_df, utils.SCALED_COLUMNS)

    features_df.to_csv(feature_data_path, index=False)

    return features_df

def load_data():
    features_df = pd.read_parquet(feature_data_path)

    return features_df

# Creates a confusion matrix of a given dataset and their labels
def create_confusion_matrix(y_test : pd.DataFrame, y_pred : pd.DataFrame, labels : list):
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    pass

# Condenses all underrepresented Precambrian classes into one bucket for human interpretation
def condense_precambrian(df : pd.DataFrame, min_count : int):
    filtered_df = df[df[Field.STRAT].str.startswith('P')]

    strat_counts = filtered_df[Field.STRAT].value_counts()
    rare_strats = strat_counts[strat_counts < min_count].index

    df.loc[df[Field.STRAT].isin(rare_strats), Field.STRAT] = 'PUDF'
    return df

# Returns a list of labels each bedrock code is a part of
def bedrock_to_labels(df : pd.DataFrame):
    labels = list(set().union(*map(set, utils.BEDROCK_SET_MAP.values())))
    labels_df = pd.DataFrame(columns=labels)

    for label in labels:
        labels_df[label] = df[Field.STRAT].apply(lambda s : label in utils.BEDROCK_SET_MAP[s])

    return labels_df

# Condenses other underrepresented classes into their closest relative (usually direct parent)
def condense_other_bedrock(df : pd.DataFrame):
    df[Field.STRAT] = df[Field.STRAT].map(utils.BEDROCK_UNDERREP_MAP).fillna(df[Field.STRAT])
    return df