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
    FIRST_BEDROCK_CATEGORY = 'first_bdrk_cat'
    DEPTH_TO_BEDROCK = 'depth_to_bdrk'
    TOP_DEPTH_TO_BEDROCK = 'top_depth_to_bdrk'
    BOT_DEPTH_TO_BEDROCK = 'bot_depth_to_bdrk'
    WEIGHT = 'weight'
    DEPTH_TOP = 'depth_top'
    DEPTH_BOT = 'depth_bot'
    COLOR = 'color'
    DRILLER_DESCRIPTION = 'drllr_desc'
    PREVIOUS_AGE_CATEGORY = 'prev_age_cat'
    CORE = 'core'
    CUTTINGS = 'cuttings'
    INTERPRETATION_METHOD = 'strat_mc'


#TODO: I actually havent added first_bedrock_age yet...
GENERAL_DROPPED_COLUMNS = [Field.RELATEID, Field.DRILLER_DESCRIPTION, Field.COLOR, Field.WEIGHT, Field.STRAT, Field.DATA_SOURCE,
                     Field.INTERPRETATION_METHOD]

def load():
    features_df = pd.read_parquet(DATA_PATH)

    return features_df

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