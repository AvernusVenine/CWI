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

def create_bedrock(X, y, value, count, col):
    mask = y[col] == value

    X_total = X
    y_total = y

    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    X_new = []
    y_new = []

    for _ in range(count):
        idx = np.random.choice(len(X))

        utme = X.loc[idx, Field.UTME]
        utmn = X.loc[idx, Field.UTMN]

        X_temp = X.drop(X.index[idx])

        dist = np.sqrt((X_temp[Field.UTME] - utme)**2 + (X_temp[Field.UTMN] - utmn)**2)
        dist = dist[dist > 0]

        idx_one = idx
        idx_two = dist.idxmin()

        new_row = X.iloc[idx_one].copy()
        new_row[Field.INTERPRETATION_METHOD] = .5

        num = np.random.rand()

        new_row[Field.UTME] = (X.iloc[idx_one][Field.UTME] * num + X.iloc[idx_two][Field.UTME] * (1 - num))
        new_row[Field.UTMN] = (X.iloc[idx_one][Field.UTMN] * num + X.iloc[idx_two][Field.UTMN] * (1 - num))
        new_row[Field.ELEVATION] = (X.iloc[idx_one][Field.ELEVATION] * num + X.iloc[idx_two][Field.ELEVATION] * (1 - num))
        new_row[Field.DEPTH] = (X.iloc[idx_one][Field.DEPTH] * num + X.iloc[idx_two][Field.DEPTH] * (1 - num))

        X_new.append(new_row)
        y_new.append(y.iloc[idx_two])

    X_new = pd.DataFrame(X_new)
    y_new = pd.DataFrame(y_new)

    return pd.concat([X_total, X_new], ignore_index=True), pd.concat([y_total, y_new], ignore_index=True)

def create_shallow(X, y, value, count, col):
    mask = y[col] == value
    indices = np.where(mask)[0]

    X_new = []
    y_new = []

    for _ in range(count):
        idx_one = np.random.choice(indices)
        idx_two = np.random.choice(indices)

        new_row = X.iloc[idx_one].copy()

        """Create a hyperplane between the two datapoints and randomly select a value on it"""

        new_row[Field.UTME] = random.uniform(min(X.iloc[idx_one][Field.UTME], X.iloc[idx_two][Field.UTME]),
                                             max(X.iloc[idx_one][Field.UTME], X.iloc[idx_two][Field.UTME]))
        new_row[Field.UTMN] = random.uniform(min(X.iloc[idx_one][Field.UTMN], X.iloc[idx_two][Field.UTMN]),
                                             max(X.iloc[idx_one][Field.UTMN], X.iloc[idx_two][Field.UTMN]))
        new_row[Field.ELEVATION_BOT] = random.uniform(min(X.iloc[idx_one][Field.ELEVATION_BOT], X.iloc[idx_two][Field.ELEVATION_BOT]),
                                                  max(X.iloc[idx_one][Field.ELEVATION_BOT], X.iloc[idx_two][Field.ELEVATION_BOT]))
        new_row[Field.ELEVATION_TOP] = random.uniform(min(X.iloc[idx_one][Field.ELEVATION_TOP], X.iloc[idx_two][Field.ELEVATION_TOP]),
                                                  max(X.iloc[idx_one][Field.ELEVATION_TOP], X.iloc[idx_two][Field.ELEVATION_TOP]))
        new_row[Field.DEPTH_BOT] = random.uniform(min(X.iloc[idx_one][Field.DEPTH_BOT], X.iloc[idx_two][Field.DEPTH_BOT]),
                                                  max(X.iloc[idx_one][Field.DEPTH_BOT], X.iloc[idx_two][Field.DEPTH_BOT]))
        new_row[Field.DEPTH_TOP] = random.uniform(min(X.iloc[idx_one][Field.DEPTH_TOP], X.iloc[idx_two][Field.DEPTH_TOP]),
                                                  max(X.iloc[idx_one][Field.DEPTH_TOP], X.iloc[idx_two][Field.DEPTH_TOP]))

        X_new.append(new_row)
        y_new.append(y.iloc[idx_two])

    X_new = pd.DataFrame(X_new)
    y_new = pd.DataFrame(y_new)

    return pd.concat([X, X_new], ignore_index=True), pd.concat([y, y_new], ignore_index=True)