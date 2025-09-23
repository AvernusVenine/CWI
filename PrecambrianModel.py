import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import utils
import xgboost
import cupy
from sklearn.multiclass import OneVsRestClassifier

import Data
from Data import Field

# TODO: Hyperparameter train this
PRECAMBRIAN_DROPPED_COLUMNS = [Field.BOT_DEPTH_TO_BEDROCK, Field.AGE_CATEGORY] + Data.GENERAL_DROPPED_COLUMNS

class PrecambrianModel:

    def __init__(self, path: str):
        self.model = None
        self.path = path