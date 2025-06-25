import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

# TODO: These functions need to take in the new inputs I will be training models on like Location, Depth, etc.

label_map = {'A': -1, 'Q': 0, 'R': 1, 'B': 2, 'RMMF': 3, 'U': 4}
rev_label_map = {v: k for k, v in label_map.items()}
color_map = {'BROWN': 'B', 'GRAY': 'G', 'BLUE': 'G', 'BLACK': 'K', 'Red': 'R', 'Green': 'L', 'Orange': 'O',
             'Other/Varied': 'U', 'White': 'W', 'Yellow': 'Y'}
soil_type_map = {
  0: 'B',
  1: 'C',
  2: 'F',
  3: 'G',
  4: 'I',
  5: 'J',
  6: 'H',
  7: 'L',
  8: 'W',
  9: 'P',
  10: 'N',
  11: 'R',
  12: 'S',
  13: 'T',
  14: 'U'
}

strat_age_model = joblib.load('trained_models/LGBM_Strat_Age_Model.joblib')
quat_type_model = joblib.load('trained_models/LGBM_Quat_Type_Model.joblib')
recent_type_model = joblib.load('trained_models/LGBM_Recent_Type_Model.joblib')

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Checks to see if the color is listed in the drillers description
def get_color_from_desc(desc : str) -> str:
    colors = []

    for c in color_map.keys():
        if c.upper() in desc.upper():
            colors.append(color_map[c])

    return colors[0] if len(colors) == 1 else 'U'

# Returns the prediction for a 'Q' layer
def predict_quaternary_code(color : str, description : str) -> (str, float):
    desc_emb = embedder.encode([description])

    emb_df = pd.DataFrame(desc_emb, columns=[f"emb_{i}" for i in range(desc_emb.shape[1])])

    prediction = quat_type_model.predict(emb_df)[0]

    quat_string = 'Q' + soil_type_map[prediction] + 'U' + color_map[color]

    probs = quat_type_model.predict_proba(emb_df)[0]
    conf = np.max(probs)

    return quat_string, conf

# Returns the prediction for a 'R' layer
def predict_recent_code(color : str, description : str) -> (str, float):
    desc_emb = embedder.encode([description])

    emb_df = pd.DataFrame(desc_emb, columns=[f"emb_{i}" for i in range(desc_emb.shape[1])])

    prediction = recent_type_model.predict(emb_df)[0]

    quat_string = 'R' + soil_type_map[prediction] + 'U' + color_map[color]

    probs = recent_type_model.predict_proba(emb_df)[0]
    conf = np.max(probs)

    return quat_string, conf

# Returns the prediction as to what age a layer belongs to
def predict_age(desc : str, prev_strat : str) -> (str, float):

    desc_emb = embedder.encode([desc])

    emb_df = pd.DataFrame(desc_emb, columns=[f"emb_{i}" for i in range(desc_emb.shape[1])])
    emb_df['prev_label'] = label_map[prev_strat]

    prediction = strat_age_model.predict(emb_df)[0]

    probs = strat_age_model.predict_proba(emb_df)[0]
    conf = probs[prediction]

    return rev_label_map[prediction], conf

# Returns the Bayesian likelihood of a list of floats
def bayesian_likelihood(probs : [float]):

    num = 1
    denom = 1

    for p in probs:
        num *= p
        denom *= (1-p)

    return num / (num + denom)

# Predicts the entirety of a code
def predict_code(desc: str, prev_strat : str, color : str) -> (str, float):

    age = predict_age(desc, prev_strat)

    if age[0] == 'Q':
        code = predict_quaternary_code(desc, color)
        conf = bayesian_likelihood([age[1], code[1]])

        return code[0], conf
    elif age[0] == 'R':
        code = predict_recent_code(desc, color)
        conf = bayesian_likelihood([age[1], code[1]])

        return code[0], conf
    elif age[0] == 'RMMF':
        return age

    # TODO: Possibly make two different models, one that determines rock type and the other that determines exact code?
    if age[0] == 'B':
        pass

    return