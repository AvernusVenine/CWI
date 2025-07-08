import sys
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QListWidget, QListWidgetItem, QComboBox, QFormLayout, \
    QLineEdit, QPushButton, QLabel

#TODO: What codes besides PITT, PVMT, RMMF, and BSMT are man-made, need to separate them...

color_map = {'Brown': 'B', 'Gray': 'G', 'Blue': 'G', 'Black': 'K', 'Red': 'R', 'Green': 'L', 'Orange': 'O',
             'Other/Varied': 'U', 'White': 'W', 'Yellow': 'Y'}

embedder = SentenceTransformer('all-MiniLM-L6-v2')

pca = joblib.load('trained_models/embedding_pca.joblib')

strat_age_model = joblib.load('trained_models/GBT_Age_Model.joblib')
quat_type_model = joblib.load('trained_models/GBT_Quat_Model.joblib')
bedrock_model = joblib.load('trained_models/GBT_Bedrock_Model.joblib')

geo_code_cat = joblib.load('trained_models/geo_code_categories.joblib')
age_cat = joblib.load('trained_models/age_categories.joblib')
quat_type_cat = joblib.load('trained_models/quat_type_categories.joblib')
bedrock_cat = joblib.load('trained_models/bedrock_categories.joblib')


def predict_bedrock(model_input, age_prediction):

    model_input['age_cat'] = age_prediction

    bedrock_prediction = bedrock_model.predict(model_input)[0]

    bedrock_probs = bedrock_model.predict_proba(model_input)[0]
    bedrock_conf = bedrock_probs[bedrock_prediction]

    type_confidence_text.setText(f'Model Bedrock Group Confidence: {bedrock_conf}')
    predict_text.setText(f'Model Prediction: {bedrock_cat[bedrock_prediction]}')

def predict_quat(model_input, color, age_prediction):

    model_input['age_cat'] = age_prediction
    model_input = model_input.drop(columns=['true_depth_top', 'true_depth_bot', 'geo_code_cat', 'utme', 'utmn'])

    quat_prediction = quat_type_model.predict(model_input)[0]

    quat_probs = quat_type_model.predict_proba(model_input)[0]
    quat_conf = quat_probs[quat_prediction]

    final_code = f'{age_cat[age_prediction]}{quat_type_cat[quat_prediction]}U{color_map[color]}'

    type_confidence_text.setText(f'Model Quaternary Type Confidence: {quat_conf}')
    predict_text.setText(f'Model Prediction: {final_code}')

# TODO: Add redundancy to check description for colors because some data entry failed to separate it...

def predict_stratigraphy():
    description = driller_desc.text().strip().upper()
    color = color_combobox.currentText()

    desc_emb = embedder.encode([description])
    embedding_df = pd.DataFrame(desc_emb, columns=[f"emb_{i}" for i in range(desc_emb.shape[1])])

    pca_embeddings = pca.transform(embedding_df)
    pca_embeddings_df = pd.DataFrame(pca_embeddings, columns=[f'pca_emb_{i}' for i in range(pca_embeddings.shape[1])])

    try:
        elevation = float(elevation_input.text().strip())
    except ValueError:
        print("INVALID ELEVATION INPUT")
        return

    try:
        depth_top = float(depth_top_input.text().strip())
        depth_bot = float(depth_bot_input.text().strip())
    except ValueError:
        print("INVALID DEPTH INPUTS")
        return

    true_depth_top = elevation - depth_top
    true_depth_bot = elevation - depth_bot

    try:
        utme = int(utme_input.text().strip())
        utmn = int(utmn_input.text().strip())
    except ValueError:
        print("INVALID UTM COORDINATES")
        return

    prev_age = age_combobox.currentText()

    if prev_age == 'Air':
        prev_age = -1
    else:
        prev_age = age_cat.index(prev_age)
    print("AGE")

    geo_code = geo_code_cat.index(geo_code_combobox.currentText())

    print('GEOCODE')

    numerical_features = pd.DataFrame([{
        'true_depth_bot': true_depth_bot,
        'true_depth_top': true_depth_top,
        'geo_code_cat': geo_code,
        'utme': utme,
        'utmn': utmn,
        'prev_age_cat': prev_age
    }])

    model_input = pd.concat([numerical_features.reset_index(drop=True), pca_embeddings_df.reset_index(drop=True)],
                            axis=1)

    print(model_input.columns.tolist())

    age_prediction = strat_age_model.predict(model_input)[0]

    age_probs = strat_age_model.predict_proba(model_input)[0]
    age_conf = age_probs[age_prediction]

    age_confidence_text.setText(f'Model Age Confidence: {age_conf}')

    # Model Predicts Basement
    if age_cat[age_prediction] == 'B':
        predict_text.setText('Model Prediction: BSMT')
        type_confidence_text.setText('')

    # Model Predicts Man-Made Fill
    elif age_cat[age_prediction] == 'F':
        predict_text.setText('Model Prediction: RMMF')
        type_confidence_text.setText('')

    # Model Predicts Man-Made Pitt
    elif age_cat[age_prediction] == 'X':
        predict_text.setText('Model Prediction: PITT')
        type_confidence_text.setText('')

    # Model Predicts Man-Made Pavement
    elif age_cat[age_prediction] == 'Y':
        predict_text.setText('Model Prediction: PVMT')
        type_confidence_text.setText('')

    # Model Predicts Quaternary, proceed to next model
    elif age_cat[age_prediction] in ('Q', 'R'):
        predict_quat(model_input, color, age_prediction)

    # Model Predicts Bedrock, proceed to next model
    else:
        predict_bedrock(model_input, age_prediction)

# Build app GUI

app = QApplication([])

window = QWidget()
window.setWindowTitle('CWI MN Stratigraphy Identifier')
window.setGeometry(100, 100, 600, 200)

layout = QFormLayout()

driller_desc = QLineEdit()
layout.addRow("Driller Description:", driller_desc)

color_combobox = QComboBox()
color_combobox.addItems(['Black', 'Blue', 'Brown', 'Green', 'Gray', 'Red', 'Orange', 'White', 'Yellow', 'Other/Varied'])
layout.addRow("Color:", color_combobox)

age_combobox = QComboBox()
age_combobox.addItems(['Air'])
age_combobox.addItems([str(age) for age in age_cat])
layout.addRow("Previous Age:", age_combobox)

geo_code_combobox = QComboBox()
geo_code_combobox.addItems([code for code in geo_code_cat])
layout.addRow("Atlas Estimation:", geo_code_combobox)

utme_input = QLineEdit()
utmn_input = QLineEdit()

utm_layout = QHBoxLayout()
utm_layout.addWidget(utme_input)
utm_layout.addWidget(utmn_input)

layout.addRow('UTME/UTMN:', utm_layout)

elevation_input = QLineEdit()
layout.addRow("Elevation:", elevation_input)

depth_top_input = QLineEdit()
depth_bot_input = QLineEdit()

depth_layout = QHBoxLayout()
depth_layout.addWidget(depth_top_input)
depth_layout.addWidget(depth_bot_input)

layout.addRow('Depth Top/Depth Bot:', depth_layout)

predict_button = QPushButton('Predict')
predict_button.clicked.connect(predict_stratigraphy)
layout.addRow(predict_button)

predict_text = QLabel('Model Prediction: ')
layout.addRow(predict_text)

age_confidence_text = QLabel('Model Age Confidence: ')
layout.addRow(age_confidence_text)

type_confidence_text = QLabel('Model Type Confidence: ')
layout.addRow(type_confidence_text)

window.setLayout(layout)

window.show()
sys.exit(app.exec_())
