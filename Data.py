import pandas as pd
import torch.cuda
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import random
import matplotlib.pyplot as plt
import random
import numpy as np
import copy

import config

DATA_PATH = 'compiled_data/data.parquet'

COLORS = ['BLACK', 'BROWN', 'BLUE', 'GREEN', 'ORANGE', 'PINK', 'PURPLE', 'RED', 'GRAY', 'WHITE', 'LIGHT', 'DARK',
          'YELLOW', 'VARIED']

"""List of the most specific label the previous layer can be assigned"""

class Field:
    LENGTH = 'length'
    THICKNESS = 'thickness'
    ORDER = 'order'
    UTME = 'utme'
    UTMN = 'utmn'
    ELEVATION = 'elevation'
    ELEVATION_TOP = 'elev_top'
    ELEVATION_BOT = 'elev_bot'
    STRAT = 'strat'
    RELATEID = 'relateid'
    AGE = 'age'
    TYPE = 'type'
    DEPTH_TOP = 'depth_top'
    DEPTH_BOT = 'depth_bot'
    DEPTH = 'depth'
    COLOR = 'color'
    HARDNESS = 'hardness'
    DRILLER_DESCRIPTION = 'drllr_desc'
    TEXTURE = 'texture'
    LITH_PRIM = 'lith_prim'
    LITH_SEC = 'lith_sec'
    LITH_MINOR = 'lith_minor'
    PREVIOUS_AGE = 'prev_age'
    PREVIOUS_TEXTURE = 'prev_text'
    GROUP = 'group'
    GROUP_TOP = 'group_top'
    GROUP_BOT = 'group_bot'
    PREVIOUS_GROUP = 'prev_group'
    FORMATION = 'form'
    FORMATION_TOP = 'form_top'
    FORMATION_BOT = 'form_bot'
    PREVIOUS_FORMATION = 'prev_form'
    MEMBER = 'member'
    MEMBER_TOP = 'member_top'
    MEMBER_BOT = 'member_bot'
    PREVIOUS_MEMBER = 'prev_member'
    INTERPRETATION_METHOD = 'strat_mc'
    COUNTY = 'county_c'

def load_raw():
    return pd.read_csv(f'{config.RAW_DATA_PATH}/c5st.csv', low_memory=False, on_bad_lines='skip')

def load_well_raw():
    return pd.read_csv(f'{config.RAW_DATA_PATH}/cwi5.csv', low_memory=False, on_bad_lines='skip')

def load(path):
    """
    Load the already embedded dataset
    :param path: Data path
    :return: Dataframe
    """
    df = pd.read_parquet(f'{config.DATA_PATH}/{path}')

    return df

def load_and_embed(subset=None):
    """
    Loads the dataset from its raw form and embeds the text features then saves it to save time
    :param subset: Percentage of dataset to use
    :return: Embedded dataframe
    """

    pd.set_option('future.no_silent_downcasting', True)

    cwi_well_data_path = f'{config.RAW_DATA_PATH}/cwi5.csv'
    cwi_layer_data_path = f'{config.RAW_DATA_PATH}/c5st.csv'

    wells_df = pd.read_csv(cwi_well_data_path, low_memory=False, on_bad_lines='skip')
    layers_df = pd.read_csv(cwi_layer_data_path, low_memory=False, on_bad_lines='skip')

    if subset:
        layers_df = layers_df.sample(frac=subset)

    layers_df = layers_df.drop(columns=['objectid', 'c5st_seq_no', 'wellid', 'concat', 'stratcode_gen', 'stratcode_detail',
                                'lith_sec', 'lith_minor'])

    layers_df.loc[layers_df[Field.DEPTH_BOT].isna(), Field.DEPTH_BOT] = layers_df[Field.DEPTH_TOP]
    layers_df = layers_df.fillna(value={
        Field.COLOR: 'UNKNOWN',
        Field.HARDNESS: 'UNKNOWN',
        Field.DRILLER_DESCRIPTION: ''
    })
    """Certain classifications have non-uniform patterns, so this makes them uniform for easier interpretation"""
    layers_df[Field.STRAT] = layers_df[Field.STRAT].replace({
        'RMMF': 'FILL',
        'PITT': 'X',
        'PVMT': 'Y'
    })

    layers_df[Field.ELEVATION] = layers_df[Field.RELATEID].map(wells_df.set_index(Field.RELATEID)[Field.ELEVATION])
    layers_df[Field.UTME] = layers_df[Field.RELATEID].map(wells_df.set_index(Field.RELATEID)[Field.UTME])
    layers_df[Field.UTMN] = layers_df[Field.RELATEID].map(wells_df.set_index(Field.RELATEID)[Field.UTMN])
    layers_df[Field.INTERPRETATION_METHOD] = layers_df[Field.RELATEID].map(wells_df.set_index(Field.RELATEID)[Field.INTERPRETATION_METHOD])

    """Embed descriptions"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transformer = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    embeddings = transformer.encode(layers_df[Field.DRILLER_DESCRIPTION].tolist(), show_progress_bar=True)
    embeddings_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    layers_df = pd.concat([layers_df.reset_index(drop=True), embeddings_df.reset_index(drop=True)], axis=1)

    layers_df = layers_df.drop(columns=[Field.DRILLER_DESCRIPTION])

    """Save unlabeled datapoints for future testing purposes"""
    layers_df[layers_df[Field.STRAT].isna()].to_parquet(f'{config.DATA_PATH}/unlabelled.parquet')

    layers_df = layers_df.dropna(subset=[Field.STRAT])

    layers_df.to_parquet(f'{config.DATA_PATH}/data.parquet')

    return layers_df

def one_hot_encode(df):
    """
    Fit a one hot encoder model
    :param df: Dataframe to onehot encode
    :return: One hot encoder model
    """
    encoder = OneHotEncoder()
    encoder.fit(df)

    return encoder

def fit_pca(df, n_components=.95):
    """
    Fit a PCA model
    :param df: Dataframe to fit PCA to
    :param n_components: Number of principal components or variance explained percentage
    :return: PCA model
    """
    pca = PCA(n_components=n_components)
    pca.fit(df)

    return pca

def pca_components_graph():
    """
    Function used to determine the optimal number of components for embedding PCA
    :return:
    """
    df = load('data.parquet')

    pca = PCA()
    pca.fit(df[[f"emb_{i}" for i in range(384)]])

    variance = pca.explained_variance_ratio_

    plt.figure()
    plt.plot(range(1, len(variance)+1), variance.cumsum())
    plt.xlabel('Number of Components')
    plt.ylabel('Variance Explained')
    plt.grid(True)
    plt.savefig('pca_plot.png')

def hole_age_sets():
    """
    Function used to determine the distribution of sets of ages among holes found in the dataset
    :return: Hole age sets
    """
    df = load('data.parquet')

    hole_sets = df.groupby(Field.RELATEID)[Field.STRAT].apply(lambda s: set(s.str[0]))

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print(hole_sets.value_counts())

    return hole_sets

def sample_age_set(target):
    """
    Finds an example well that contains a given set of ages
    :param target: Target set of ages
    :return: Sampled Relate ID of target set
    """

    hole_sets = hole_age_sets()

    return hole_sets[hole_sets == target].index.to_series().sample(1).iloc[0]