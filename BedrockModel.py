import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost
from sklearn.multiclass import OneVsRestClassifier

import Bedrock
import Data
from Data import Field

BEDROCK_AGES = ('C', 'D', 'K', 'O')
ORDERED_PRECAMBRIAN = ('PMHN', 'PMFL', 'PMSC', 'PMHF')

# TODO: Hyperparameter train this
BEDROCK_DROPPED_COLUMNS = [Field.BOT_DEPTH_TO_BEDROCK, Field.AGE_CATEGORY] + Data.GENERAL_DROPPED_COLUMNS

class BedrockModel:

    def __init__(self, path: str):
        self.model = None
        self.path = path

    @staticmethod
    def encode_bedrock(series : pd.Series):
        membership_list = [item.name for item in Bedrock.AGE_LIST]
        membership_list += [item.name for item in Bedrock.GROUP_LIST]
        membership_list += [item.name for item in Bedrock.FORMATION_LIST]
        membership_list += [item.name for item in Bedrock.MEMBER_LIST]

        series = series.map(Bedrock.BEDROCK_CODE_MAP)

        rows = []
        for code in series:
            row = {name: 0 for name in membership_list}
            if code is not None:
                for stratum in code.ages + code.groups + code.formations + code.members:
                    row[stratum.name] = 1
            rows.append(row)

        return pd.DataFrame(rows, columns=membership_list)

    def train(self):
        df = Data.load()

        print("TRAINING BEDROCK CLASSIFIER")

        bedrock_layers = df[df[Field.STRAT].str.startswith(BEDROCK_AGES)]
        sorted_precamb = df[df[Field.STRAT].isin(ORDERED_PRECAMBRIAN)]

        bedrock_layers = pd.concat([bedrock_layers, sorted_precamb])

        X = bedrock_layers
        y = self.encode_bedrock(bedrock_layers[Field.STRAT])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=127)

        X_train = X_train.drop(columns=BEDROCK_DROPPED_COLUMNS)
        X_test = X_test.drop(columns=BEDROCK_DROPPED_COLUMNS)

        bedrock_classifier = xgboost.XGBClassifier(
            booster='dart',
            n_estimators=25,
            device='cuda',

            rate_drop=.1,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )

        model = OneVsRestClassifier(bedrock_classifier)
        model.fit(X_train, y_train)

        joblib.dump(model, f'{self.path}.model')
        self.model = model

        print("EVALUATING BEDROCK CLASSIFIER")

        y_pred = model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0, target_names=y_test.columns))


    def load(self):
        self.model = joblib.load(f'{self.path}.model')


    def predict(self, x : pd.Series):

        if self.model is None:
            print("NO MODEL LOADED")
            return -1, -1

        x = x.drop(columns=BEDROCK_DROPPED_COLUMNS)

        prediction = self.model.predict(x)
        confidence = self.model.predict_proba(x)

        return prediction, confidence

    def predict_code(self, x : pd.Series):
        prediction, confidence = self.predict(x)
        prediction = pd.DataFrame(prediction, columns=self.model.classes_).iloc[0]
        prediction = prediction[prediction == 1].index

        ages = [item for item in Bedrock.AGE_LIST if item.name in prediction]
        groups = [item for item in Bedrock.GROUP_LIST if item.name in prediction]
        formations = [item for item in Bedrock.FORMATION_LIST if item.name in prediction]
        members = [item for item in Bedrock.MEMBER_LIST if item.name in prediction]

        geo_code = Bedrock.GeoCode(ages + groups + formations + members)

        for key, value in Bedrock.BEDROCK_CODE_MAP.items():
            if value == geo_code:
                return key

        print(f'VALID CODE NOT FOUND: CREATE NEW CODE FOR \n')
        print(f'AGES: {geo_code.ages} \n')
        print(f'GROUPS: {geo_code.groups} \n')
        print(f'FORMATIONS: {geo_code.formations} \n')
        print(f'MEMBERS: {geo_code.members} \n')

        return None