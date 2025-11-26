import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sympy.stats.rv import probability

import utils
import xgboost
import cupy
from sklearn.multiclass import OneVsRestClassifier

import Data
from AgeModel import AGE_DROPPED_COLUMNS
from Data import Field

QUATERNARY_DROPPED_COLUMNS = [Field.DEPTH_TOP, Field.DEPTH_BOT, Field.UTME, Field.UTMN, Field.ELEVATION,
    Field.DEPTH_TO_BEDROCK, Field.TOP_DEPTH_TO_BEDROCK, Field.BOT_DEPTH_TO_BEDROCK] + Data.GENERAL_DROPPED_COLUMNS

COLORS = {
    'BROWN': 'B', 'DK. BRN': 'B', 'LT. BRN': 'B', 'TAN': 'B',
    'GRAY': 'G', 'DK. GRY': 'G', 'LT. GRY': 'G', 'BLU/GRY': 'G',
    'BLUE': 'G', 'DK. BLU': 'G', 'LT. BLU': 'G',
    'BLACK': 'K',
    'RED': 'R',
    'GREEN': 'L',
    'ORANGE': 'O',
    'Other/Varied': 'U',
    'WHITE': 'W',
    'YELLOW': 'Y'
}

QUATERNARY_CLASSIFICATIONS = {
    'B': 0,
    'C': 1,
    'F': 2,
    'G': 3,
    'H': 4,
    'I': 5,
    'J': 6,
    'L': 7,
    'N': 8,
    'P': 9,
    'S': 10,
    'T': 11,
    'U': 12,
    'W': 13
}
INVERSE_QUATERNARY_CLASSIFICATIONS = {v: k for k, v in QUATERNARY_CLASSIFICATIONS.items()}

class QuaternaryModel:

    def __init__(self, path : str):
        self.model = None
        self.path = path

    # TODO: There are only 5 quat codes that contain 'R', and thus should be treated as a massive outlier that requiries a human to compute
    def train(self):
        df = Data.load()

        print("TRAINING QUATERNARY CLASSIFIER")

        quat_layers = df[df['strat'].str.startswith(('Q', 'R'))]

        quat_type = quat_layers['strat'].str[1]
        quat_type = quat_type.str.replace('R', 'U', regex=True)
        quat_type = quat_type.map(QUATERNARY_CLASSIFICATIONS)

        y = quat_type

        X_train, X_test, y_train, y_test = train_test_split(quat_layers, y, test_size=0.2, random_state=127)

        X_train = X_train.drop(columns=QUATERNARY_DROPPED_COLUMNS)
        X_test = X_test.drop(columns=QUATERNARY_DROPPED_COLUMNS)

        quat_classifier = xgboost.XGBClassifier(
            booster='dart',
            n_estimators=200,
            verbosity=0,
            device='cuda',

            rate_drop=.15,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )

        quat_classifier.fit(X_train, y_train)

        joblib.dump(quat_classifier, f'{self.path}.model')
        self.model = quat_classifier

        print("EVALUATING QUAT TYPE CLASSIFIER")

        y_pred = quat_classifier.predict(X_test)

        y_test_labels = pd.Series(y_test).map(INVERSE_QUATERNARY_CLASSIFICATIONS)
        y_pred_labels = pd.Series(y_pred).map(INVERSE_QUATERNARY_CLASSIFICATIONS)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test_labels, y_pred_labels, zero_division=0))


    def load(self):
        self.model = joblib.load(f'{self.path}.model')


    def predict(self, x : pd.Series):

        if self.model is None:
            print("NO MODEL LOADED")
            return

        x = x.drop(columns=QUATERNARY_DROPPED_COLUMNS)

        prediction = self.model.predict(x)
        confidence = self.model.predict_proba(x)[0]

        return prediction, confidence

    @staticmethod
    def get_description_color(description : str):
        description_colors = []

        for color in COLORS.keys():
            if color.lower() in description.lower():
                description_colors.append(COLORS[color])

        # Have to check for blue/gray combinations because they are the same code
        if len(description_colors) == 2:
            if set(description_colors) == {'BLUE', 'GRAY'}:
                return 'G'

        return description_colors[0] if len(description_colors) == 1 else 'U'


    def predict_code(self, x : pd.Series, color : str):
        prediction, confidence = self.predict(x)
        color = COLORS.get(color, 'U')

        if color == 'U':
            color = self.get_description_color(x[Field.DRILLER_DESCRIPTION])

        code = f'{x[Field.AGE]}{INVERSE_QUATERNARY_CLASSIFICATIONS[prediction]}U{color}'
        return code