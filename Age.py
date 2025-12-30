import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import re
import warnings

import Bedrock
from Data import Field

AGE_LIST = ['B', 'C', 'D', 'F', 'K', 'O', 'P', 'Q', 'R', 'U', 'X', 'Y', 'G']
BEDROCK_AGE_LIST = ['C', 'D', 'G', 'K', 'O', 'P']
TYPE_LIST = ['Bedrock', 'Fill', 'Recent', 'Residuum', 'Quaternary', 'Pitt', 'Pavement']

def init_encoder():
    """
    Initializes and fits an age encoder
    :return: Fit age encoder
    """
    encoder = LabelEncoder()
    encoder.fit(AGE_LIST)

    return encoder

def init_type_encoder():

    encoder = LabelEncoder()
    encoder.fit(TYPE_LIST)

    return encoder

def decode_age(val):
    """
    Decodes a given age
    :param val: Encoded age(s)
    :return: Decoded age
    """
    encoder = LabelEncoder()
    encoder.fit(AGE_LIST)

    return encoder.inverse_transform(val)

def encode_type(df):
    warnings.filterwarnings("ignore")

    df[Field.TYPE] = df[Field.STRAT].astype(str).str[0]
    df[Field.TYPE] = df[Field.TYPE].replace({
        'B': None,
        'C': 'Bedrock',
        'D': 'Bedrock',
        'F': 'Fill',
        'K': 'Bedrock',
        'O': 'Bedrock',
        'P': 'Bedrock',
        'Q': 'Quaternary',
        'R': 'Recent',
        'U': 'Residuum',
        'X': 'Pitt',
        'Y': 'Pavement',
        'N': None,
        'I': None,
        'J': None,
        'W': None
    })

    df = df.dropna(subset=[Field.TYPE])

    encoder = LabelEncoder()
    encoder.fit(TYPE_LIST)

    df[Field.TYPE] = encoder.transform(df[Field.TYPE])

    df[Field.AGE] = df[Field.STRAT].astype(str).str[0]

    df[Field.AGE] = df[Field.AGE].replace({'N': None, 'I': None, 'J': None, 'W': None})

    p_exempt = ['PMHN', 'PMHF', 'PMSC', 'PMFL']

    df.loc[df[Field.STRAT].isin(p_exempt), Field.AGE] = 'G'

    encoder = LabelEncoder()
    encoder.fit(AGE_LIST)

    df.loc[df[Field.AGE].notna(), Field.AGE] = encoder.transform(df.loc[df[Field.AGE].notna(), Field.AGE])
    df.loc[~df[Field.AGE].notna(), Field.AGE] = -1

    return df

def encode_age(df):
    """
    Encodes the age part of a code
    :param df: Dataframe of layers
    :return: Age encoded dataframe
    """
    warnings.filterwarnings("ignore")

    df = df[~df['strat'].isin(['PVMT', 'PITT', 'PUDF'])]

    df[Field.AGE] = df[Field.STRAT].map(Bedrock.BEDROCK_CODE_MAP)

    df.loc[df[Field.STRAT].isin(['PMHN', 'PMHF', 'PMSC', 'PMFL']), Field.STRAT] = 'G'

    df.loc[df[Field.STRAT].astype(str).str[0] == 'P', Field.AGE] = 'P'

    df = df.dropna(subset=[Field.AGE])

    df.loc[df[Field.ORDER] == 0, Field.AGE] = df.loc[df[Field.ORDER] == 0, Field.STRAT].apply(
        lambda x : x.top_lineage[0] if isinstance(x, Bedrock.GeoCode) else x)
    df.loc[df[Field.ORDER] == 1, Field.AGE] = df.loc[df[Field.ORDER] == 1, Field.STRAT].apply(
        lambda x: x.bot_lineage[0] if isinstance(x, Bedrock.GeoCode) else x)

    encoder = LabelEncoder()
    encoder.fit(BEDROCK_AGE_LIST)

    df[Field.AGE] = encoder.transform(df[Field.AGE])

    return df