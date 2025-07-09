import joblib
import pandas as pd

layer_df = pd.read_csv('compiled_data/layers_all_features.csv')

color_map = {'BROWN': 'B', 'DK. BRN': 'B', 'LT. BRN': 'B', 'TAN': 'B',
             'GRAY': 'G',  'DK. GRY': 'G', 'LT. GRY': 'G',
             'BLUE': 'G', 'DK. BLU': 'G', 'LT. BLU': 'G',
             'BLACK': 'K',
             'RED': 'R',
             'GREEN': 'L', 'ORANGE': 'O',
             'Other/Varied': 'U',
             'WHITE': 'W',
             'YELLOW': 'Y'}

geo_code_cat = joblib.load('trained_models/geo_code_categories.joblib')
age_cat = joblib.load('trained_models/age_categories.joblib')
quat_type_cat = joblib.load('trained_models/quat_type_categories.joblib')
bedrock_cat = joblib.load('trained_models/bedrock_categories.joblib')

strat_age_model = joblib.load('trained_models/GBT_Age_Model.joblib')
quat_type_model = joblib.load('trained_models/GBT_Quat_Model.joblib')
bedrock_model = joblib.load('trained_models/GBT_Bedrock_Model.joblib')

def predict_age(model_input):
    age_prediction = strat_age_model.predict(model_input)[0]

    age_probs = strat_age_model.predict_proba(model_input)[0]
    age_conf = age_probs[age_prediction]

    return age_prediction, age_conf

def predict_quat(model_input, color, age_prediction):

    model_input['age_cat'] = age_prediction
    model_input = model_input.drop(columns=['true_depth_top', 'true_depth_bot', 'geo_code_cat', 'utme', 'utmn'])

    quat_prediction = quat_type_model.predict(model_input)[0]

    quat_probs = quat_type_model.predict_proba(model_input)[0]
    quat_conf = quat_probs[quat_prediction]

    color = color_map.get(color.strip(), 'U')

    final_code = f'{age_cat[age_prediction]}{quat_type_cat[quat_prediction]}U{color}'

    return final_code, quat_conf

def predict_bedrock(model_input, age_prediction):
    model_input['age_cat'] = age_prediction

    bedrock_prediction = bedrock_model.predict(model_input)[0]

    bedrock_probs = bedrock_model.predict_proba(model_input)[0]
    class_index = list(bedrock_model.classes_).index(bedrock_prediction)
    bedrock_conf = bedrock_probs[class_index]

    return bedrock_cat[bedrock_prediction], bedrock_conf


mismatch_df = pd.DataFrame(columns=['relateid', 'actual', 'predicted', 'age_confidence', 'type_confidence', 'desc', 'atlas_estimate', 'color'])

for index, layer in layer_df.iterrows():

    model_features = ['true_depth_bot', 'true_depth_top', 'geo_code_cat', 'utme', 'utmn', 'prev_age_cat', 'elevation'] + \
                     [col for col in layer_df.columns if col.startswith('pca_emb_')]

    model_input = pd.DataFrame([layer[model_features].values], columns=model_features)

    age_prediction = predict_age(model_input)

    final_prediction = ''
    final_type_conf = 0
    final_age_conf = age_prediction[1]

    # Model Predicts Basement
    if age_cat[age_prediction[0]] == 'B':
        final_prediction = 'BSMT'
        final_conf = age_prediction[1]

    # Model Predicts Man-Made Fill
    elif age_cat[age_prediction[0]] == 'F':
        final_prediction = 'RMMF'
        final_conf = age_prediction[1]

    # Model Predicts Man-Made Pitt
    elif age_cat[age_prediction[0]] == 'X':
        final_prediction = 'PITT'
        final_conf = age_prediction[1]


    # Model Predicts Man-Made Pavement
    elif age_cat[age_prediction[0]] == 'Y':
        final_prediction = 'PVMT'
        final_conf = age_prediction[1]

    # Model Predicts Quaternary, proceed to next model
    elif age_cat[age_prediction[0]] in ('Q', 'R'):
        quat = predict_quat(model_input, layer['color'], age_prediction[0])
        final_prediction = quat[0]
        final_type_conf = quat[1]

    # Model Predicts Bedrock, proceed to next model
    else:
        bedrock = predict_bedrock(model_input, age_prediction[0])
        final_prediction = bedrock[0]
        final_type_conf = bedrock[1]

    if final_prediction.strip() != layer['strat'].strip():

        mismatch_df.loc[len(mismatch_df)] = {
            'relateid': layer['relateid'],
            'actual': layer['strat'],
            'predicted': final_prediction,
            'age_confidence': final_age_conf,
            'type_confidence': final_type_conf,
            'desc': layer['drllr_desc'],
            'atlas_estimate': geo_code_cat[int(layer['geo_code_cat'])],
            'color': layer['color']
        }

        if len(mismatch_df) > 20:
            mismatch_df.to_csv('compiled_data/mismatch.csv')
            break

mismatch_df.to_csv('compiled_data/mismatch.csv')