import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import re
import warnings

from Data import Field

TEXTURE_LIST = ['B', 'C', 'F', 'G', 'H', 'I', 'J', 'L', 'N', 'P', 'S', 'T', 'U', 'W']

def init_encoder():
    """
    Initializes and fits a texture encoder
    :return: Fit texture encoder
    """
    encoder = LabelEncoder()
    encoder.fit(TEXTURE_LIST)

    return encoder
def decode_texture(val):
    """
    Decodes a given texture
    :param val: Encoded texture
    :return: Decoded texture
    """
    encoder = LabelEncoder()
    encoder.fit(TEXTURE_LIST)

    return encoder.inverse_transform(val)

def encode_texture(df):
    """
    Determines whether the layer has a texture component, and if it does it extracts and encodes it
    :param df: Layer dataframe
    :return: Texture encoded dataframe
    """
    df.loc[df[Field.STRAT].astype(str).str[0].isin(['Q', 'R']), Field.TEXTURE] = df[Field.STRAT].astype(str).str[1]
    df[Field.TEXTURE] = df[Field.TEXTURE].replace({'R' : None})

    encoder = LabelEncoder()
    encoder.fit(TEXTURE_LIST)

    df.loc[df[Field.TEXTURE].notna(), Field.TEXTURE] = encoder.transform(df.loc[df[Field.TEXTURE].notna(), Field.TEXTURE])

    df = df.dropna(subset=[Field.TEXTURE])

    return df