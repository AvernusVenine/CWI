import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import data_refinement
import utils
import xgboost
import cupy
from sklearn.multiclass import OneVsRestClassifier

def train_age_classifier(features_df : pd.DataFrame):
    print("TRAINING AGE CLASSIFIER")

    y = features_df[['age_cat']].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y,
        test_size=0.2,
        random_state=127
    )

    smote_cols = ['utme', 'utmn', 'elevation', 'depth_top', 'depth_bot']

    X_train, y_train = data_refinement.fit_smote(X_train, y_train, 10000, utils.AGE_CATEGORIES['X'], 'age_cat',
                                 127, smote_cols)
    X_train, y_train = data_refinement.fit_smote(X_train, y_train, 10000, utils.AGE_CATEGORIES['Y'], 'age_cat',
                                 127, smote_cols)
    X_train, y_train = data_refinement.fit_smote(X_train, y_train, 50000, utils.AGE_CATEGORIES['F'], 'age_cat',
                                 127, smote_cols)

    # Under sample Q
    X_train = pd.concat([
        X_train[X_train['age_cat'] != 'Q'],
        X_train[X_train['age_cat'] == 'Q'].sample(frac=0.25, random_state=127)
    ], ignore_index=True)

    y_train = X_train[['age_cat']].copy()

    weights = X_train['weight']

    X_train = X_train.drop(columns=utils.AGE_DROP_COLS + utils.GENERAL_DROP_COLS)
    X_test = X_test.drop(columns=utils.AGE_DROP_COLS + utils.GENERAL_DROP_COLS)

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

    age_classifier.fit(X_train, y_train, sample_weight=weights)

    joblib.dump(age_classifier, f'trained_models/DART_Age_Model_2.joblib')

    print("EVALUATING AGE CLASSIFIER")

    y_pred = age_classifier.predict(X_test)

    y_test_labels = pd.Series(y_test).map(utils.INV_AGE_CATEGORIES)
    y_pred_labels = pd.Series(y_pred).map(utils.INV_AGE_CATEGORIES)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

#TODO: There are only 5 quat codes that contain 'R', and thus should be treated as a massive outlier that requries a human to compute
def train_quat_classifier(features_df : pd.DataFrame):
    print("TRAINING QUATERNARY CLASSIFIER")

    quat_layers = features_df[features_df['strat'].str.startswith(('Q', 'R'))]

    quat_type = quat_layers['strat'].str[1]
    quat_type = quat_type.str.replace('R', 'U', regex=True)
    quat_type = quat_type.map(utils.QUAT_CATEGORIES)

    y = quat_type

    X_train, X_test, y_train, y_test = train_test_split(quat_layers, y, test_size=0.2, random_state=127)

    weights = X_train['weight']

    X_train = X_train.drop(columns=utils.QUAT_DROP_COLS + utils.GENERAL_DROP_COLS)
    X_test = X_test.drop(columns=utils.QUAT_DROP_COLS + utils.GENERAL_DROP_COLS)

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

    quat_classifier.fit(X_train, y_train, sample_weight=weights)


    joblib.dump(quat_classifier, 'trained_models/XGB_Quat_Model.joblib')

    print("EVALUATING QUAT TYPE CLASSIFIER")

    y_pred = quat_classifier.predict(X_test)

    y_test_labels = pd.Series(y_test).map(utils.INV_QUAT_CATEGORIES)
    y_pred_labels = pd.Series(y_pred).map(utils.INV_QUAT_CATEGORIES)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

#TODO: Try by initially grouping bedrocks based on formation and see accuracy there
def train_bedrock_classifier(features_df : pd.DataFrame):
    print("TRAINING BEDROCK CLASSIFIER")

    bedrock_layers = features_df[features_df['strat'].str.startswith(utils.BEDROCK_AGES)]
    sorted_precamb = features_df[features_df['strat'].isin(utils.SORTED_PRECAMBRIAN)]

    bedrock_layers = pd.concat([bedrock_layers, sorted_precamb])

    bedrock_layers = data_refinement.condense_other_bedrock(bedrock_layers)

    utils.load_bedrock_categories(bedrock_layers)

    X = bedrock_layers
    y = data_refinement.bedrock_to_labels(bedrock_layers)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=127)

    weights = X_train['weight']

    X_train = X_train.drop(columns=utils.BEDROCK_DROP_COLS + utils.GENERAL_DROP_COLS)
    X_test = X_test.drop(columns=utils.BEDROCK_DROP_COLS + utils.GENERAL_DROP_COLS)

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

    joblib.dump(model, f'trained_models/multilabel_bedrock_model.joblib')

    print("EVALUATING BEDROCK CLASSIFIER")

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0, target_names=y_test.columns))

    #data_refinement.create_confusion_matrix(y_test, y_pred, list(utils.INV_BEDROCK_CATEGORIES.values()))

def train_precambrian_classifer(df : pd.DataFrame):
    print("TRAINING PRECAMBRIAN CLASSIFIER")


    print("EVALUATING PRECAMBRIAN CLASSIFIER")



    pass

# Can replace with just load_data() to save time if refinement isn't needed
features_df = data_refinement.load_data()

#train_age_classifier(features_df)
#train_quat_classifier(features_df)
train_bedrock_classifier(features_df)

