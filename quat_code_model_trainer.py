import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

import data_refinement
import utils

def train_age_classifier(features_df : pd.DataFrame):
    print("TRAINING AGE CLASSIFIER")

    y = features_df[['age_cat']].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y,
        test_size=0.2,
        random_state=127,
    )

    smote_cols = ['utme', 'utmn', 'elevation', 'true_depth_top', 'true_depth_bot']

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

    weights = X_train[['data_src']].copy()
    weights['data_src'] = weights['data_src'].str.strip()
    weights['data_src'] = np.where(weights['data_src'].isin(utils.TRUSTED_SOURCES), 5, 1)

    X_train = X_train.drop(columns=data_refinement)
    X_test = X_test.drop(columns=utils.AGE_DROP_COLS)

    y_test = y_test.squeeze()
    y_train = y_train.squeeze()

    age_classifier = LGBMClassifier(
        n_estimators=300,
        verbose=-1,
        boosting_type='dart',
        device='gpu',
        class_weight='balanced',
        min_child_samples=25,
        min_data_in_leaf=25,
    )
    age_classifier.fit(X_train, y_train, sample_weight=weights['data_src'])

    joblib.dump(age_classifier, 'trained_models/DART_Age_Model.joblib')

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

    y = quat_type

    X_train, X_test, y_train, y_test = train_test_split(quat_layers, y, test_size=0.2, random_state=127)

    weights = X_train[['data_src']].copy()
    weights['data_src'] = weights['data_src'].str.strip()
    weights['data_src'] = np.where(weights['data_src'].isin(utils.TRUSTED_SOURCES), 5, 1)

    X_train = X_train.drop(columns=['true_depth_top', 'true_depth_bot', 'utme', 'utmn', 'relateid', 'strat',
                                  'color', 'drllr_desc', 'elevation', 'data_src'])
    X_test = X_test.drop(columns=['true_depth_top', 'true_depth_bot', 'utme', 'utmn', 'relateid', 'strat',
                                  'color', 'drllr_desc', 'elevation', 'data_src'])

    quat_classifier = LGBMClassifier(
        n_estimators=300,
        verbose=-1,
        boosting_type='dart',
        device='gpu',
        class_weight='balanced',
        min_child_samples=25,
        min_data_in_leaf=25,
    )
    quat_classifier.fit(X_train, y_train)

    joblib.dump(quat_classifier, 'trained_models/GBT_Quat_Model.joblib')

    print("EVALUATING QUAT TYPE CLASSIFIER")

    y_pred = quat_classifier.predict(X_test)

    y_test_labels = utils.QUAT_CATEGORIES[y_test.values]
    y_pred_labels = utils.QUAT_CATEGORIES[y_pred]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

#TODO: Try by initially grouping bedrocks based on formation and see accuracy there

def train_bedrock_classifier(features_df : pd.DataFrame):
    print("TRAINING BEDROCK CLASSIFIER")

    bedrock_layers = features_df[features_df['strat'].str.startswith(utils.BEDROCK_AGES)]

    # TODO: Replace this with BEDROCK_CATEGORIES in utils
    bedrock_cat = bedrock_layers['strat'].astype('category')
    bedrock = bedrock_cat.cat.codes
    joblib.dump(bedrock_cat.cat.categories, 'trained_models/bedrock_categories.joblib')

    category_counts = bedrock_cat.value_counts().sort_index()

    for category, count in zip(bedrock_cat.cat.categories, category_counts):
        print(f"{category}: {count}")

    X = bedrock_layers
    y = bedrock

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=127)

    weights = X_train[['data_src']].copy()
    weights['data_src'] = weights['data_src'].str.strip()
    weights['data_src'] = np.where(weights['data_src'].isin(utils.TRUSTED_SOURCES), 5, 1)

    X_train = X_train.drop(columns=['relateid', 'strat', 'color', 'drllr_desc'])
    X_test = X_test.drop(columns=['relateid', 'strat', 'color', 'drllr_desc'])

    bedrock_classifier = LGBMClassifier(
        n_estimators=300,
        verbose=-1,
        boosting_type='dart',
        device='gpu',
        class_weight='balanced',
        min_child_samples=25,
        min_data_in_leaf=25,
    )
    bedrock_classifier.fit(X_train, y_train)

    joblib.dump(bedrock_classifier, 'trained_models/GBT_Bedrock_Model.joblib')

    print("EVALUATING BEDROCK CLASSIFIER")

    y_pred = bedrock_classifier.predict(X_test)

    y_test_labels = bedrock_cat.cat.categories[y_test.values]
    y_pred_labels = bedrock_cat.cat.categories[y_pred]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

# Can replace with just load_data() to save time if refinement isn't needed
features_df = data_refinement.load_and_refine_data()

train_age_classifier(features_df)
#train_quat_classifier(features_df)
#train_bedrock_classifier(features_df)

