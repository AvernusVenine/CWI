import joblib
import pandas as pd
from sympy.stats.rv import probability

import data_refinement
import utils
from utils import Field

AGE_MODEL_PATH = 'trained_models/DART_Age_Model_2.joblib'
QUAT_MODEL_PATH = 'trained_models/XGB_Quat_Model.joblib'
BEDROCK_MODEL_PATH = ''

DATA_SAVE_PATH = 'compiled_data/comparison_results.csv'

def load_models():

    age_model = joblib.load(AGE_MODEL_PATH)
    quat_model = joblib.load(QUAT_MODEL_PATH)
    bedrock_model = joblib.load(BEDROCK_MODEL_PATH)

    return age_model, quat_model, bedrock_model

def get_quat_color(desc : str, color : str):

    final_color = utils.QUAT_COLOR_MAP.get(color.strip(), 'U')

    if final_color == 'U':
        for key in utils.QUAT_COLOR_MAP.keys():
            if key in desc:
                final_color = utils.QUAT_COLOR_MAP.get(key, 'U')

    return final_color

def predict_age(model, df : pd.DataFrame, pred_df : pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=utils.AGE_DROP_COLS + utils.GENERAL_DROP_COLS)

    for _, row in df.iterrows():
        pred = model.predict(row.to_list())
        prob = model.predict_proba(row.to_list())

        pred_df.loc[int(row['relateid'])] = {
            'relateid' : row['relateid'],
            'depth_top' : row['depth_top'],
            'prim_age_prediction' : utils.INV_AGE_CATEGORIES[pred[0]],
            'prim_age_confidence' : prob[pred[0]],
            'sec_age_prediction' : utils.INV_AGE_CATEGORIES[pred[1]],
            'sec_age_confidence' : prob[pred[1]],
            'actual_age' : utils.INV_AGE_CATEGORIES[row['age_cat']],
        }

    return pred_df

def predict_quat(model, df : pd.DataFrame, pred_df : pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=utils.QUAT_DROP_COLS + utils.GENERAL_DROP_COLS)
    df = df[df['strat'].str.startswith(('Q', 'R'))]

    for _, row in df.iterrows():
        pred = model.predict(row.to_list())
        prob = model.predict_proba(row.to_list())

        color = get_quat_color(row['drllr_desc'], row['color'])

        prim_pred = f"{row['age']}U{color}{utils.INV_QUAT_CATEGORIES[pred[0]]}"
        sec_pred = f"{row['age']}U{color}{utils.INV_QUAT_CATEGORIES[pred[1]]}"

        pred_df[int(row['relateid']), [
            'prim_prediction',
            'prim_confidence',
            'sec_prediction',
            'sec_confidence',
            'actual'
        ]] = [
            prim_pred,
            prob[pred[0]],
            sec_pred,
            prob[pred[1]],
            row['strat']
        ]

    return pred_df

def predict_bedrock(model, df : pd.DataFrame, pred_df : pd.DataFrame) -> pd.DataFrame:
    utils.load_bedrock_categories(df)

    df = df.drop(columns=utils.BEDROCK_DROP_COLS + utils.GENERAL_DROP_COLS)
    df = df[df['strat'].str.startswith(utils.BEDROCK_AGES)]

    for _, row in df.iterrows():

        pred = model.predict(row.to_list())
        prob = model.predict_proba(row.to_list())

        pred_df[int(row['relateid']), [
            'prim_prediction',
            'prim_confidence',
            'sec_prediction',
            'sec_confidence',
            'actual'
        ]] = [
            utils.INV_BEDROCK_CATEGORIES[pred[0]],
            prob[pred[0]],
            utils.INV_BEDROCK_CATEGORIES[pred[1]],
            prob[pred[1]],
            row['strat']
        ]

    return pred_df

def refine_results(df : pd.DataFrame):

    df = df['actual'].replace({
        'F' : 'RMMF',
        'X' : 'PITT',
        'Y' : 'PVMT',
        'U' : 'UREG',
        'Z' : 'Underrepresented or Invalid: Class Human Interpretation Required'
    })

    df = df['prim_prediction'].replace({
        'F' : 'RMMF',
        'X' : 'PITT',
        'Y' : 'PVMT',
        'U' : 'UREG',
        'Z' : 'Underrepresented or Invalid: Class Human Interpretation Required'
    })

    df = df['sec_prediction'].replace({
        'F' : 'RMMF',
        'X' : 'PITT',
        'Y' : 'PVMT',
        'U' : 'UREG',
        'Z' : 'Underrepresented or Invalid: Class Human Interpretation Required'
    })

    return df

def find_errors(df : pd.DataFrame):
    error_df = df[df['prim_prediction'] != df['actual']]

    return error_df

df = data_refinement.load_data()

age_model, quat_model, bedrock_model = load_models()

pred_df = predict_age(age_model, df, pd.DataFrame())
pred_df = predict_quat(quat_model, df, pred_df)
pred_df = predict_bedrock(bedrock_model, df, pred_df)

pred_df = refine_results(pred_df)
pred_df = find_errors(pred_df)

pred_df.to_csv(DATA_SAVE_PATH, index=False)