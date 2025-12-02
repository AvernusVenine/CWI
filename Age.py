import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import re
import warnings

from Data import Field

AGE_LIST = ['B', 'C', 'D', 'F', 'K', 'O', 'P', 'Q', 'R', 'U', 'X', 'Y', 'G', 'A', 'E', 'M']

def decode_age(val):
    """
    Decodes a given age
    :param val: Encoded age
    :return: Decoded age
    """
    encoder = LabelEncoder()
    encoder.fit(AGE_LIST)

    return encoder.inverse_transform(val)

def encode_age(df):
    """
    Encodes the age part of a code
    :param df: Dataframe of layers
    :return: Age encoded dataframe
    """
    warnings.filterwarnings("ignore")

    df[Field.AGE] = df[Field.STRAT].astype(str).str[0]

    df[Field.AGE] = df[Field.AGE].replace({'N' : None, 'I' : None, 'J' : None, 'W' : None})

    df.loc[df[Field.STRAT].astype(str).str.startswith('PA'), Field.AGE] = 'A'
    df.loc[df[Field.STRAT].astype(str).str.startswith('PE'), Field.AGE] = 'E'
    df.loc[df[Field.STRAT].astype(str).str.startswith('PM'), Field.AGE] = 'M'

    p_exempt = ['PMHN', 'PMHF', 'PMSC']

    df.loc[df[Field.STRAT].isin(p_exempt), Field.AGE] = 'G'

    encoder = LabelEncoder()
    encoder.fit(AGE_LIST)

    df.loc[df[Field.AGE].notna(), Field.AGE] = encoder.transform(df.loc[df[Field.AGE].notna(), Field.AGE])
    df.loc[~df[Field.AGE].notna(), Field.AGE] = -1

    return df