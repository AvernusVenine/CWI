import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import config
import utils
import xgboost
import cupy
from sklearn.multiclass import OneVsRestClassifier
import os
from torch.utils.data import DataLoader

import Data
from Data import Field
from LayerRNNModel import LayerRNN, LayerDataset

AGE_DROPPED_COLUMNS = [Field.AGE_CATEGORY, Field.AGE] + Data.GENERAL_DROPPED_COLUMNS

AGE_CLASSIFICATIONS = {
    'A': -1,
    'C': 0,
    'D': 1,
    'F': 2,
    'K': 3,
    'O': 4,
    'P': 5,
    'Q': 6,
    'R': 7,
    'U': 8,
    'X': 9,
    'Y': 10,
    'Z': 11
}

INVERSE_AGE_CLASSIFICATIONS = {v: k for k, v in AGE_CLASSIFICATIONS.items()}

class RNNAgeModel:
    def __init__(self, path, input_size, hidden_size, output_size, num_layers):
        self.model = LayerRNN(input_size, hidden_size, output_size, num_layers)

        if os.path.isfile(f'{config.MODEL_PATH}/{path}'):
            self.model.load_state_dict(torch.load(f'{config.MODEL_PATH}/{path}'))

        self.path = path

    def train(self):
        print("LOADING DATA SET")

        df = Data.load('data.parquet')
        X = df.drop(columns=[Field.STRAT])
        y = df[Field.STRAT].astype(str).str[:1]

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        pca = Data.fit_pca(X_train)

        X_train = pca.fit(X_train)
        X_test = pca.fit(X_test)

        train_loader = DataLoader(X_train, y_train)
        test_loader = DataLoader(X_test, y_test)


        pass

class AgeModel:

    def __init__(self, path : str):
        self.model = None
        self.path = path

    def train(self):
        df = Data.load('data.parquet')

        print("TRAINING AGE CLASSIFIER")

        y = df[[Field.AGE_CATEGORY]].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            df, y,
            test_size=0.2,
            random_state=127
        )

        # Apply custom SMOTE to underrepresented ages
        smote_cols = [Field.UTME, Field.UTMN, Field.ELEVATION, Field.DEPTH_TOP, Field.DEPTH_BOT]

        X_train, y_train = Data.fit_smote(X_train, y_train, 10000, AGE_CLASSIFICATIONS['X'], Field.AGE_CATEGORY,
                                          127, smote_cols)
        X_train, y_train = Data.fit_smote(X_train, y_train, 10000, AGE_CLASSIFICATIONS['Y'], Field.AGE_CATEGORY,
                                          127, smote_cols)
        X_train, y_train = Data.fit_smote(X_train, y_train, 50000, AGE_CLASSIFICATIONS['F'], Field.AGE_CATEGORY,
                                          127, smote_cols)

        # Under sample Q
        X_train = pd.concat([
            X_train[X_train[Field.AGE_CATEGORY] != 'Q'],
            X_train[X_train[Field.AGE_CATEGORY] == 'Q'].sample(frac=0.25, random_state=127)
        ], ignore_index=True)

        y_train = X_train[[Field.AGE_CATEGORY]].copy()

        X_train = X_train.drop(columns=AGE_DROPPED_COLUMNS)
        X_test = X_test.drop(columns=AGE_DROPPED_COLUMNS)

        y_test = y_test.squeeze()
        y_train = y_train.squeeze()

        age_classifier = xgboost.XGBClassifier(
            booster='dart',
            n_estimators=1000,
            verbosity=0,
            device='cuda',

            rate_drop=.1,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )

        age_classifier.fit(X_train, y_train)

        joblib.dump(age_classifier, f'{self.path}.model')
        self.model = age_classifier

        print("EVALUATING AGE CLASSIFIER")

        y_pred = age_classifier.predict(X_test)

        y_test_labels = pd.Series(y_test).map(INVERSE_AGE_CLASSIFICATIONS)
        y_pred_labels = pd.Series(y_pred).map(INVERSE_AGE_CLASSIFICATIONS)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test_labels, y_pred_labels, zero_division=0))


    def load(self):
        self.model = joblib.load(f'{self.path}.model')


    def predict(self, x : pd.Series):

        if self.model is None:
            print("NO MODEL LOADED")
            return -1, -1

        x = x.drop(columns=AGE_DROPPED_COLUMNS)

        prediction = self.model.predict(x)
        confidence = self.model.predict_proba(x)[0]

        return prediction, confidence