import sys
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QListWidget, QListWidgetItem, QComboBox, QFormLayout, \
    QLineEdit, QPushButton, QLabel

label_map = {'Air': -1, 'Quaternary': 0, 'Recent': 1, 'Bedrock': 2}
rev_label_map = {-1: 'Air', 0: 'Quaternary', 1: 'Recent', 2: 'Bedrock'}
color_map = {'Brown': 'B', 'Gray': 'G', 'Blue': 'G', 'Black': 'K', 'Red': 'R', 'Green': 'L', 'Orange': 'O',
             'Other/Varied': 'U', 'White': 'W', 'Yellow': 'Y'}
quat_type_map = {
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

# TODO: Get more data to improve accuracy! Sits at 98.2% with some underrepresented types!

def get_quaternary_code(color : str, description : str):
    desc_emb = embedder.encode([description])

    emb_df = pd.DataFrame(desc_emb, columns=[f"emb_{i}" for i in range(desc_emb.shape[1])])

    type_prediction = quat_type_model.predict(emb_df)[0]

    quat_string = 'Q' + quat_type_map[type_prediction] + 'U' + color_map[color]
    predict_text.setText('Model Prediction: ' + quat_string)

    type_probs = quat_type_model.predict_proba(emb_df)[0]
    type_conf = np.max(type_probs)

    type_confidence_text.setText('Model Type Confidence: ' + str(type_conf))

# TODO: Get more data to improve accuracy! Too many underrepresented types, almost entirely unknowns!

def get_recent_code(color : str, description : str):
    desc_emb = embedder.encode([description])

    emb_df = pd.DataFrame(desc_emb, columns=[f"emb_{i}" for i in range(desc_emb.shape[1])])

    type_prediction = recent_type_model.predict(emb_df)[0]

    quat_string = 'R' + quat_type_map[type_prediction] + 'U' + color_map[color]
    predict_text.setText('Model Prediction: ' + quat_string)

    type_probs = recent_type_model.predict_proba(emb_df)[0]
    type_conf = np.max(type_probs)

    type_confidence_text.setText('Model Type Confidence: ' + str(type_conf))

# TODO: Cannot currently classify Wisconsonian or Unknown, too underrepresented!
# TODO: Add redundancy to check description for colors because some data entry failed to separate it...

def predict_stratigraphy():
    description = driller_desc.text().strip().upper()
    prev_strat = strat_combobox.currentText()
    color = color_combobox.currentText()

    desc_emb = embedder.encode([description])

    emb_df = pd.DataFrame(desc_emb, columns=[f"emb_{i}" for i in range(desc_emb.shape[1])])
    emb_df['prev_label'] = label_map[prev_strat]

    prediction = strat_age_model.predict(emb_df)[0]

    age_probs = strat_age_model.predict_proba(emb_df)[0]
    age_conf = age_probs[prediction]

    age_confidence_text.setText('Model Age Confidence: ' + str(age_conf))

    if prediction == label_map['Quaternary']:
        get_quaternary_code(color, description)
    elif prediction == label_map['Recent']:
        get_recent_code(color, description)
    else:
        predict_text.setText('Model Predicted Bedrock [Exact Classification TBA]')
        type_confidence_text.setText('Model Type Confidence: ')

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

strat_combobox = QComboBox()
strat_combobox.addItems(['Air', 'Quaternary', 'Recent', 'Bedrock'])
layout.addRow("Previous Stratigraphy:", strat_combobox)

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