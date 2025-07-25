import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import random

import utils

cwi_well_data_path = 'cwi_data/cwi5.csv'
cwi_layer_data_path = 'cwi_data/c5st.csv'
feature_data_path = 'compiled_data/features_df.csv'

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
    embeddings = embedding_model.encode(df['drllr_desc'].tolist(), show_progress_bar=True)
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

# Loads and refines all well layer data
def load_and_refine_data():
    cwi_wells = pd.read_csv(cwi_well_data_path, low_memory=False, on_bad_lines='skip')
    cwi_layers = pd.read_csv(cwi_layer_data_path, low_memory=False, on_bad_lines='skip')

    cwi_wells = cwi_wells.dropna(subset=['elevation', 'utmn', 'utme'])

    cwi_layers = cwi_layers.drop(
        columns=['objectid', 'c5st_seq_no', 'wellid', 'concat', 'stratcode_gen', 'stratcode_detail'])

    cwi_layers = cwi_layers.dropna(subset=['strat'])
    cwi_layers.loc[cwi_layers['depth_bot'].isna(), 'depth_bot'] = cwi_layers['depth_top']
    cwi_layers = cwi_layers.fillna(value={
        'color': 'UNKNOWN',
        'hardness': 'MEDIUM',
        'drllr_desc': ''
    })

    # Replace certain faux 'ages' with their true meaning for easier interpretation
    cwi_layers['strat'] = cwi_layers['strat'].replace({
        'RMMF': 'FILL',
        'PITT': 'X',
        'PVMT': 'Y'
    })
    cwi_layers['age'] = cwi_layers['strat'].str[0]

    # Put all underrepresented ages into one Bucket for human interpretation
    cwi_layers['age'] = cwi_layers['age'].str.replace(r'^[WJBNI]', 'Z', regex=True)

    cwi_layers['elevation'] = cwi_layers['relateid'].map(cwi_wells.set_index('relateid')['elevation'])
    cwi_layers['utme'] = cwi_layers['relateid'].map(cwi_wells.set_index('relateid')['utme'])
    cwi_layers['utmn'] = cwi_layers['relateid'].map(cwi_wells.set_index('relateid')['utmn'])
    cwi_layers['data_src'] = cwi_layers['relateid'].map(cwi_wells.set_index('relateid')['data_src'])

    cwi_layers = cwi_layers.dropna(subset=['utme', 'utmn', 'elevation'])

    cwi_layers['age_cat'] = cwi_layers['age'].map(utils.AGE_CATEGORIES)


    embeddings = embed_descriptions(cwi_layers)
    pca_embeddings = embed_pca(embeddings)

    colors = cwi_layers['color'].apply(one_hot_colors)

    layer_features = cwi_layers[['depth_top', 'depth_bot', 'utme', 'utmn', 'relateid', 'age_cat',
                                 'strat', 'color', 'drllr_desc', 'elevation', 'data_src']]

    features_df = pd.concat([layer_features.reset_index(drop=True), pca_embeddings.reset_index(drop=True),
                             colors.reset_index(drop=True)], axis=1)

    features_df = features_df.sort_values(by=['relateid', 'depth_top'], ascending=[True, True])
    features_df['prev_age_cat'] = (
        features_df.groupby('relateid')['age_cat']
        .shift(1)
        .fillna(-1)
        .astype(int)
    )

    features_df = df_scaler(features_df, utils.SCALED_COLUMNS)

    features_df.to_csv(feature_data_path, index=False)

    return features_df

def load_data():
    features_df = pd.read_csv(feature_data_path, low_memory=False)

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
    filtered_df = df[df['strat'].str.startswith('P')]

    strat_counts = filtered_df['strat'].value_counts()
    rare_strats = strat_counts[strat_counts < min_count].index

    df['strat'] = df['strat'].apply(lambda x: utils.PRECAMBRIAN_UNKNOWN if x in rare_strats else x)

    return df