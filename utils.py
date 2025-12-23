import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import re
import warnings
import matplotlib.pyplot as plt
import random

import Bedrock
import Texture
from Bedrock import GeoCode
import Precambrian
from Precambrian import PrecambrianCode
from Data import Field
import Age

def encode_weights(df):

    weight_map = {
        'A': .75,
        'B': 2,
        'C': 2,
        'D': .75,
        'E': 2,
        'F': 2,
        'G': 2,
        'H': 2,
        'N': .5,
        'O': .5,
        'P': 1.25,
        'Q': 1,
        'R': .75,
        'U': .5,
        'X': 1,
        'Z': 1,
        np.nan: 1,
    }

    df[Field.INTERPRETATION_METHOD] = df[Field.INTERPRETATION_METHOD].replace(weight_map)

    return df

def reduce_majority(X, y, percentage_dict):
    """
    Attempts to balance the hole dataset by removing labels from majority age classes (effectively masking them)
    :param X: Holes numpy array
    :param y: Labels array
    :param percentage_dict: Dictionary of ages and percentage of layers to keep
    :return: Balanced holes numpy array and labels
    """

    encoder = Age.init_encoder()

    percentage_dict = {encoder.transform([k])[0] : v for k, v in percentage_dict.items()}

    for idx, hole in enumerate(X):
        for jdx, layer in enumerate(hole):
            age = y[idx][jdx][0]

            if age in percentage_dict.keys():
                if random.random() > percentage_dict[age]:
                    y[idx][jdx][0] = -100

    return X, y

def compile_geocode(label_dict):
    """
    Converts a list of outputs from the RNN Model into an interpretable code
    :param label_dict: Label output dictionary
    :return: Stratigraphy Code
    """
    age = Age.decode_age(label_dict['age'])[0]

    """Simple one to one conversions of basic geocodes"""
    if age == 'F':
        return 'FILL' #TODO: Change this back to RMMF at some point
    if age == 'X':
        return 'PITT'
    if age == 'Y':
        return 'PVMT'
    if age == 'B':
        return 'BSMT'
    if age == 'U':
        return 'UREG'

    """Quaternary/Recent codes"""
    if age in ('Q', 'R'):
        color_dict = {
            'BROWN' : 'B',
            'GRAY' : 'G',
            'BLUE' : 'G',
            'BLACK' : 'K',
            'RED' : 'R',
            'GREEN' : 'L',
            'ORANGE' : 'O',
            'WHITE' : 'W',
            'YELLOW' : 'Y',
            'PINK' : 'U',
            'PURPLE' : 'U',
            'VARIED' : 'U'
        }

        color = 'U'

        colors = [col for col in color_dict.keys() if label_dict.get(col, 0) == 1]
        if len(set(colors)) == 1:
            color = color_dict[colors[0]]

        texture = Texture.decode_texture(label_dict['texture'])[0]

        return f'{age}{texture}U{color}'

    """Well Ordered Bedrock codes"""
    if age in ('C', 'D', 'O', 'K', 'G'):
        groups = [Bedrock.GROUP_DICT[col.name] for col in Bedrock.GROUP_LIST if label_dict.get(col.name, 0) == 1]
        formations = [Bedrock.FORMATION_DICT[col.name] for col in Bedrock.FORMATION_LIST if label_dict.get(col.name, 0) == 1]
        members = [Bedrock.MEMBER_DICT[col.name] for col in Bedrock.MEMBER_LIST if label_dict.get(col.name, 0) == 1]

        labels = groups + formations + members

        """Undifferentiated codes"""
        if not labels:
            if age == 'C':
                return 'CAMB'
            if age == 'D':
                return 'DEVO'
            if age == 'K':
                return 'KRET'
            if age == 'O':
                return 'ORDO'
            if age == 'G':
                return 'PCUU'

        geocode = GeoCode(labels)

        for key, value in Bedrock.BEDROCK_CODE_MAP.items():
            if value == geocode:
                return key

    """Precambrian codes"""
    if age in ('P', 'A', 'E', 'M'):
        categories = [col.name for col in Precambrian.CATEGORY_LIST if label_dict.get(col.name, 0) == 1]
        lithologies = [col.name for col in Precambrian.LITHOLOGY_LIST if label_dict.get(col.name, 0) == 1]

        labels = categories + lithologies

        if not labels:
            return 'PCUU'

        if age == 'P':
            age = 'PU'
        else:
            age = f'P{age}'

        geocode = PrecambrianCode(labels)

        for key, value in Precambrian.PRECAMBRIAN_MAP():
            if value == geocode:
                return f'{age}{key}'

    """Very rarely a result may yield labels that dont have a code assigned yet (like 4+ formations combined)"""
    return 'EQUIVALENT CODE NOT FOUND IN DATABASE'

def encode_hardness(df):
    """
    Encodes hardness onto a scale of 0-1, imitating the V. Soft to V. Hard range found on drill forms
    :param df: Dataframe of layers
    :return: Encoded dataframe
    """
    hardness_map = {
        'V.SOFT': 0,
        'SOFT': .25,
        'M.SOFT':.375,
        'SFT-MED': .375,
        'MEDIUM': .5,
        'UNKNOWN': .5,
        'SFT-HRD': .5,
        'M.HARD': .675,
        'MED-HRD': .675,
        'HARD': .75,
        'V.HARD': 1.0
    }

    df[Field.HARDNESS] = df[Field.HARDNESS].map(hardness_map).fillna(.5)

    return df

def encode_color(df):
    """
    Onehot encodes color
    :param df: Layer dataframe
    :return: Color encoded dataframe
    """
    color_map = {
        'BLK' : 'BLACK',
        'BRN' : 'BROWN',
        'BLU' : 'BLUE',
        'GRN' : 'GREEN',
        'OLV' : 'GREEN',
        'OLIVE' : 'GREEN',
        'ORN' : 'ORANGE',
        'PNK' : 'PINK',
        'PUR' : 'PURPLE',
        'RED' : 'RED',
        'SLV' : 'GRAY',
        'GRY' : 'GRAY',
        'SILVER' : 'GRAY',
        'TAN' : 'BROWN',
        'WHT' : 'WHITE',
        'YEL' : 'YELLOW',
        'DK.' : 'DARK',
        'LT.' : 'LIGHT',
        'VARIED' : 'VARIED'
    }

    def format_color(value):
        value = re.sub(r' ', '', value)
        colors = re.split(r'[/.]', value)

        colors = [color_map.get(c, c) for c in colors]

        return colors

    unique_colors = sorted(set(color_map.values()))

    new_cols = {
        color: df[Field.COLOR].apply(lambda x: color in format_color(x))
        for color in unique_colors
    }

    df = pd.concat([df.drop(columns=[Field.COLOR]), pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df

def sequence_individual(df, relate_id):
    """
    Helper function that extracts the layers associated with a given RelateID and returns the ordered hole
    :param df: Layer dataframe
    :param relate_id: Well RelateID
    :return: Hole data, Hole labels
    """
    hole = df[df[Field.RELATEID] == relate_id]
    hole = hole.sort_values([Field.DEPTH_BOT, Field.DEPTH_TOP], ascending=[True, True])

    y_cols = ([Field.AGE, Field.TEXTURE] + [group.name for group in Bedrock.GROUP_LIST] +
              [formation.name for formation in Bedrock.FORMATION_LIST] +
              [member.name for member in Bedrock.MEMBER_LIST] +
              [category.name for category in Precambrian.CATEGORY_LIST] +
              [lithology.name for lithology in Precambrian.LITHOLOGY_LIST])

    X = hole.drop(columns=[Field.STRAT, Field.RELATEID, Field.LITH_PRIM] + y_cols).to_numpy(dtype=float)
    y = hole[y_cols].to_numpy()

    return X, y

def sequence_layers(df):
    """
    Helper function that compiles and orders the layer dataset into ordered holes
    :param df: Layer dataframe
    :return: Hole dataset, Hole labels, Y Column List for indexing
    """
    holes = df.groupby(Field.RELATEID)

    X = []
    y = []

    y_cols = ([Field.AGE, Field.TEXTURE] + [group.name for group in Bedrock.GROUP_LIST] +
              [formation.name for formation in Bedrock.FORMATION_LIST] +
              [member.name for member in Bedrock.MEMBER_LIST] +
              [category.name for category in Precambrian.CATEGORY_LIST] +
              [lithology.name for lithology in Precambrian.LITHOLOGY_LIST])

    for _, hole in holes:
        hole = hole.sort_values([Field.DEPTH_BOT, Field.DEPTH_TOP], ascending=[True, True])

        X.append(hole.drop(columns=[Field.STRAT, Field.RELATEID, Field.LITH_PRIM] + y_cols).to_numpy(dtype=float))
        y.append(hole[y_cols].to_numpy())

    valid = [i for i in range(len(y)) if not np.all(y[i][:, 0] == -100)]
    X = [X[i] for i in valid]
    y = [y[i] for i in valid]

    return X, y

def rnn_collate_fn(batch):
    """
    Collate function for the LayerRNN Model, since it expects every item in the batch to have the same number of layers,
    so I have to pad the shorter ones
    :param batch: List of data points in a batch
    :return: Padded data, padded labels, original lengths
    """

    # Claude was used to help bug fix this based off my original design

    features, labels = zip(*batch)

    features_padded = pad_sequence(features, batch_first=True, padding_value=-1.0)

    age_padded = pad_sequence([label['age'] for label in labels], batch_first=True, padding_value=-100)
    texture_padded = pad_sequence([label['texture'] for label in labels], batch_first=True, padding_value=-100)
    group_padded = pad_sequence([label['group'] for label in labels], batch_first=True, padding_value=0.0)
    formation_padded = pad_sequence([label['formation'] for label in labels], batch_first=True, padding_value=0.0)
    member_padded = pad_sequence([label['member'] for label in labels], batch_first=True, padding_value=0.0)
    category_padded = pad_sequence([label['category'] for label in labels], batch_first=True, padding_value=0.0)
    lithology_padded = pad_sequence([label['lithology'] for label in labels], batch_first=True, padding_value=0.0)

    labels_padded = {
        'age': age_padded,
        'texture': texture_padded,
        'group': group_padded,
        'formation': formation_padded,
        'member': member_padded,
        'category': category_padded,
        'lithology': lithology_padded,
    }

    return features_padded, labels_padded