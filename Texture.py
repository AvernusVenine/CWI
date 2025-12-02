import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import re
import warnings

from Data import Field

TEXTURE_LIST = ['B', 'C', 'F', 'G', 'I', 'J', 'H', 'L', 'W', 'P', 'N', 'S', 'T', 'U']

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
    df.loc[~df[Field.TEXTURE].notna(), Field.TEXTURE] = -1

    return df