import pandas as pd
import torch.cuda
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import random
import matplotlib.pyplot as plt

import Age
import config

DATA_PATH = 'compiled_data/data.parquet'

class Field:
    UTME = 'utme'
    UTMN = 'utmn'
    ELEVATION = 'elevation'
    STRAT = 'strat'
    RELATEID = 'relateid'
    AGE = 'age'
    AGE_CATEGORY = 'age_cat'
    DATA_SOURCE = 'data_src'
    #TODO: Fully remove
    #FIRST_BEDROCK_CATEGORY = 'first_bdrk_cat'
    DEPTH_TO_BEDROCK = 'depth_to_bdrk'
    TOP_DEPTH_TO_BEDROCK = 'top_depth_to_bdrk'
    BOT_DEPTH_TO_BEDROCK = 'bot_depth_to_bdrk'
    WEIGHT = 'weight'
    DEPTH_TOP = 'depth_top'
    DEPTH_BOT = 'depth_bot'
    COLOR = 'color'
    HARDNESS = 'hardness'
    DRILLER_DESCRIPTION = 'drllr_desc'
    #TODO: Fully remove
    PREVIOUS_AGE_CATEGORY = 'prev_age_cat'
    #CORE = 'core'
    #CUTTINGS = 'cuttings'
    INTERPRETATION_METHOD = 'strat_mc'
    TEXTURE = 'texture'
    LITH_PRIM = 'lith_prim'


#TODO: I actually havent added first_bedrock_age yet...
GENERAL_DROPPED_COLUMNS = [Field.RELATEID, Field.DRILLER_DESCRIPTION, Field.COLOR, Field.WEIGHT, Field.STRAT, Field.DATA_SOURCE,
                     Field.INTERPRETATION_METHOD]

def load(path):
    """
    Load the already embedded dataset
    :param path: Data path
    :return: Dataframe
    """
    df = pd.read_parquet(f'{config.DATA_PATH}/{path}')

    return df

#TODO: This no longer works with the RNN, need to create a different version
def fit_smote(X_df : pd.DataFrame, y_df : pd.DataFrame, count : int, label : int, label_col : str, random_state : int, data_cols : list):
    """
    A custom SMOTE function that only creates data points from given data columns while directly duplicating the rest
    Typically, this means duplicating spatial information while ignoring the embedded text values and categorical features
    :param X_df: Features dataframe
    :param y_df: Label dataframe
    :param count: Total amount of datapoints to return, including real ones
    :param label: Label to apply SMOTE to
    :param label_col: Label column name
    :param random_state: Random seed
    :param data_cols: Data columns to create new data points for instead of just duplicating
    :return: Dataframe containing additional datapoints
    """
    random.seed(random_state)

    X_filtered_df = X_df[y_df[label_col] == label]
    df_size = X_filtered_df.shape[0]

    X_new = []
    y_new = []

    for i in range(count):
        df_index_one = random.randint(0, df_size - 1)
        df_index_two = random.randint(0, df_size - 1)

        new_row = X_filtered_df.iloc[df_index_one].copy()

        """Draw a 'line' between two datapoints of the same field and randomly select a value on it"""
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

def reduce_quaternary(X, y, age_col, quat_max=40000, mixed_max=40000):
    """
    Attempts to balance the hole dataset by reducing the amount of holes that only contain recent layers
    :param X: Holes numpy array
    :param y: Labels array
    :param age_col: Age column
    :param quat_max: Maximum number of holes to keep with only quaternary layers
    :param mixed_max: Maximum number of holes to keep with both quaternary and recent layers
    :return: Balanced holes numpy array
    """

    X_bal = []
    y_bal = []

    quat_count = 0
    mixed_count = 0

    encoder = Age.init_encoder()

    for idx, hole in enumerate(X):
        age_set = set(encoder.inverse_transform(hole[:, age_col]))

        if age_set == {'Q'}:
            if quat_count < quat_max:
                X_bal.append(hole)
                y_bal.append(y[idx])
                quat_count = quat_count + 1
        elif age_set == {'Q', 'R'}:
            if mixed_count < mixed_max:
                X_bal.append(hole)
                y_bal.append(y[idx])
                mixed_count = mixed_count + 1
        else:
            X_bal.append(hole)
            y_bal.append(y[idx])

    return X_bal, y_bal

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