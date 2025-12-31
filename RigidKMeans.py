import joblib
import numpy as np
import pandas as pd

import Data
from Data import Field

def cluster_data(X, y, clusters, label, col, model, max_iter):

    mask = X[col] == label

    df = X.loc[mask, [Field.UTME, Field.UTMN]]

    df_min = np.min()

    centroids =

    for _ in range(max_iter):

        pass

    pass