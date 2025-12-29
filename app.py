from PyQt6.QtGui import QIntValidator
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QEvent
from PyQt6.QtWidgets import QWidget, QFrame, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout, QMainWindow, \
    QComboBox, QCheckBox, QBoxLayout, QScrollArea, QApplication
from sentence_transformers import SentenceTransformer
import torch
import joblib
import sys
import pandas as pd
import numpy as np

from Data import Field
import Age, Texture, Bedrock, Precambrian
import utils

class StratLayer(QWidget):
    def __init__(self):
        super().__init__()

        """Inputs"""
        self.depth_top = QLineEdit()
        self.depth_bot = QLineEdit()
        self.driller_description = QLineEdit()

        self.hardness = QComboBox()
        self.hardness.addItems([
            None,
            'HARD',
            'M.HARD',
            'M.SOFT',
            'MED-HRD',
            'MEDIUM',
            'SFT-HRD',
            'SFT-MED',
            'SOFT',
            'V.HARD',
            'V.SOFT'
        ])
        self.color = QComboBox()
        self.color.addItems([
            None,
            "BLACK",
            "BLK/BLU",
            "BLK/BRN",
            "BLK/GRN",
            "BLK/GRY",
            "BLK/OLV",
            "BLK/ORN",
            "BLK/PNK",
            "BLK/PUR",
            "BLK/RED",
            "BLK/SLV",
            "BLK/TAN",
            "BLK/WHT",
            "BLK/YEL",
            "BLU/BLK",
            "BLU/BRN",
            "BLU/GRN",
            "BLU/GRY",
            "BLU/OLV",
            "BLU/ORN",
            "BLU/PNK",
            "BLU/PUR",
            "BLU/RED",
            "BLU/SLV",
            "BLU/TAN",
            "BLU/WHT",
            "BLU/YEL",
            "BLUE",
            "BRN/BLK",
            "BRN/BLU",
            "BRN/GRN",
            "BRN/GRY",
            "BRN/OLV",
            "BRN/ORN",
            "BRN/PNK",
            "BRN/PUR",
            "BRN/RED",
            "BRN/TAN",
            "BRN/WHT",
            "BRN/YEL",
            "BROWN",
            "DARK",
            "DK. BLU",
            "DK. BRN",
            "DK. GRN",
            "DK. GRY",
            "DK. ORN",
            "DK. PNK",
            "DK. PUR",
            "DK. RED",
            "DK. TAN",
            "DK. WHT",
            "DK. YEL",
            "GRAY",
            "GREEN",
            "GRN/BLK",
            "GRN/BLU",
            "GRN/BRN",
            "GRN/GRY",
            "GRN/ORN",
            "GRN/PNK",
            "GRN/PUR",
            "GRN/RED",
            "GRN/TAN",
            "GRN/WHT",
            "GRN/YEL",
            "GRY/BLK",
            "GRY/BLU",
            "GRY/BRN",
            "GRY/GRN",
            "GRY/OLV",
            "GRY/ORN",
            "GRY/PNK",
            "GRY/PUR",
            "GRY/RED",
            "GRY/SLV",
            "GRY/TAN",
            "GRY/WHT",
            "GRY/YEL",
            "LIGHT",
            "LT. BLU",
            "LT. BRN",
            "LT. GRN",
            "LT. GRY",
            "LT. OLV",
            "LT. ORN",
            "LT. PNK",
            "LT. RED",
            "LT. TAN",
            "LT. YEL",
            "OLIVE",
            "OLV/BLK",
            "OLV/BRN",
            "OLV/GRN",
            "OLV/GRY",
            "OLV/TAN",
            "OLV/WHT",
            "OLV/YEL",
            "ORANGE",
            "ORN/BLK",
            "ORN/BRN",
            "ORN/GRN",
            "ORN/GRY",
            "ORN/PNK",
            "ORN/RED",
            "ORN/TAN",
            "ORN/WHT",
            "ORN/YEL",
            "PINK",
            "PNK/BLK",
            "PNK/BLU",
            "PNK/BRN",
            "PNK/GRN",
            "PNK/GRY",
            "PNK/OLV",
            "PNK/ORN",
            "PNK/PUR",
            "PNK/RED",
            "PNK/TAN",
            "PNK/WHT",
            "PNK/YEL",
            "PUR/BLK",
            "PUR/BRN",
            "PUR/GRN",
            "PUR/GRY",
            "PUR/ORN",
            "PUR/PNK",
            "PUR/RED",
            "PUR/WHT",
            "PUR/YEL",
            "PURPLE",
            "RED",
            "RED/BLK",
            "RED/BLU",
            "RED/BRN",
            "RED/GRN",
            "RED/GRY",
            "RED/OLV",
            "RED/ORN",
            "RED/PNK",
            "RED/PUR",
            "RED/TAN",
            "RED/WHT",
            "RED/YEL",
            "SILVER",
            "SLV/BLK",
            "SLV/BLU",
            "SLV/GRY",
            "SLV/WHT",
            "TAN",
            "TAN/BLK",
            "TAN/BLU",
            "TAN/BRN",
            "TAN/GRN",
            "TAN/GRY",
            "TAN/OLV",
            "TAN/ORN",
            "TAN/PNK",
            "TAN/PUR",
            "TAN/RED",
            "TAN/WHT",
            "TAN/YEL",
            "VARIED",
            "WHITE",
            "WHT/BLK",
            "WHT/BLU",
            "WHT/BRN",
            "WHT/GRN",
            "WHT/GRY",
            "WHT/OLV",
            "WHT/ORN",
            "WHT/PNK",
            "WHT/PUR",
            "WHT/RED",
            "WHT/SLV",
            "WHT/TAN",
            "WHT/YEL",
            "YEL/BLK",
            "YEL/BLU",
            "YEL/BRN",
            "YEL/GRN",
            "YEL/GRY",
            "YEL/OLV",
            "YEL/ORN",
            "YEL/PNK",
            "YEL/RED",
            "YEL/TAN",
            "YEL/WHT",
            "YELLOW"
        ])

        input_bot_layout = QHBoxLayout()

        input_bot_layout.addWidget(QLabel('From: '))
        input_bot_layout.addWidget(self.depth_top)

        input_bot_layout.addWidget(QLabel('To: '))
        input_bot_layout.addWidget(self.depth_bot)

        input_bot_layout.addWidget(self.hardness)
        input_bot_layout.addWidget(self.color)

        description_layout = QHBoxLayout()
        description_layout.addWidget(QLabel('Description: '))
        description_layout.addWidget(self.driller_description)

        input_layout = QVBoxLayout()

        input_layout.addLayout(description_layout)
        input_layout.addLayout(input_bot_layout)

        """Predictions and Confidences"""
        self.strat = QLabel()

        self.age_conf1 = QLabel()
        self.age_conf2 = QLabel()
        self.age_conf3 = QLabel()

        age_layout = QHBoxLayout()
        age_layout.addWidget(self.age_conf1)
        age_layout.addWidget(self.age_conf2)
        age_layout.addWidget(self.age_conf3)

        output_layout = QVBoxLayout()

        output_layout.addWidget(self.strat)
        output_layout.addLayout(age_layout)

        """Final layout"""
        layout = QHBoxLayout()

        layout.addLayout(input_layout)
        layout.addLayout(output_layout)

        self.setLayout(layout)

    def update_strat(self, strat):
        self.strat.setText(strat)

    def update_age_confidence(self, ages, probs):
        self.age_conf1.setText(f'{ages[0]} : {probs[0]:.3f}')
        self.age_conf2.setText(f'{ages[1]} : {probs[1]:.3f}')
        self.age_conf3.setText(f'{ages[2]} : {probs[2]:.3f}')

class App(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle('Minnesota CWI Stratigraphy Classifier')

        self.path = 'models/gbt'

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.pca = joblib.load(f'{self.path}.pca')
        self.utme_scaler = joblib.load(f'{self.path}.utme.scl')
        self.utmn_scaler = joblib.load(f'{self.path}.utmn.scl')
        self.elevation_scaler = joblib.load(f'{self.path}.elevation.scl')
        self.depth_scaler = joblib.load(f'{self.path}.depth.scl')
        self.pca = joblib.load(f'{self.path}.pca')

        self.age_model = joblib.load(f'{self.path}.age.mdl')
        self.age_model.set_params(device='cpu')
        self.texture_model = joblib.load(f'{self.path}.txt.mdl')
        self.texture_model.set_params(device='cpu')

        self.age_cols = joblib.load(f'{self.path}.age.fts')
        self.texture_cols = joblib.load(f'{self.path}.txt.fts')

        layout = QVBoxLayout()
        self.setLayout(layout)

        """Inputs"""
        self.relateid = QLineEdit()

        self.elevation = QLineEdit()
        self.utme = QLineEdit()
        self.utmn = QLineEdit()

        """Stratigraphy Layers"""
        self.scroll_area = QScrollArea()
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll_area.setWidgetResizable(True)

        widget = QWidget()
        widget.setLayout(layout)
        self.scroll_area.setWidget(widget)

        self.layer_layout = QVBoxLayout()
        self.layer_layout.addWidget(self.scroll_area)

        self.layer_layout.addWidget(StratLayer())
        self.layer_layout.addWidget(StratLayer())

        """Final layout"""
        layout.addLayout(self.layer_layout)

    def add_layer(self):
        pass

    def predict(self):
        age_encoder = Age.init_encoder()
        texture_encoder = Texture.init_encoder()
        group_encoder, formation_encoder, member_encoder = Bedrock.init_encoders()

        sequential_data = {
            Field.AGE: -1,
            Field.TEXTURE: -1,
            Field.GROUP_BOT: group_encoder.transform([None])[0],
            Field.FORMATION_BOT: formation_encoder.transform([None])[0],
            Field.MEMBER_BOT: member_encoder.transform([None])[0],
        }

        for idx in range(self.layer_layout.count(), 0, -1):
            layer = self.layer_layout.widget(idx)

            df = pd.DataFrame()

            embeddings = self.embedder.encode(layer.driller_description.text())
            embeddings_pca = self.pca.transform([embeddings])[0]

            df[[f"pca_{i}" for i in range(50)]] = embeddings_pca

            df[Field.UTME] = self.utme_scaler.transform([float(self.utme.text())])[0]
            df[Field.UTMN] = self.utmn_scaler.transform([float(self.utmn.text())])[0]
            df[Field.DEPTH_TOP] = self.depth_scaler.transform([float(layer.depth_top.text())])[0]
            df[Field.DEPTH_BOT] = self.depth_scaler.transform([float(layer.depth_bot.text())])[0]
            df[Field.ELEVATION_TOP] = self.elevation_scaler.transform([float(self.elevation.text()) - float(layer.depth_top.text())])[0]
            df[Field.ELEVATION_BOT] = self.elevation_scaler.transform([float(self.elevation.text()) - float(layer.depth_bot.text())])[0]

            df[Field.COLOR] = layer.color.currentText()
            df = utils.encode_color(df)

            df[Field.HARDNESS] = layer.hardness.currentText()
            df = utils.encode_hardness(df)

            df[Field.PREVIOUS_AGE] = sequential_data[Field.AGE]
            df[Field.PREVIOUS_TEXTURE] = sequential_data[Field.TEXTURE]
            df[Field.PREVIOUS_GROUP] = sequential_data[Field.GROUP_BOT]
            df[Field.PREVIOUS_FORMATION] = sequential_data[Field.FORMATION_BOT]

            age = self.age_model.predict(layer[self.age_cols])

            age_prob = self.age_model.predict_proba(layer[self.age_cols])[0]
            top3_idx = np.argsort(age_prob[-3:][::-1])
            top3_prob = age_prob[top3_idx]
            top3_classes = age_encoder.inverse_transform(self.age_model.classes_[top3_idx])

            layer.update_age_confidence(top3_classes, top3_prob)
            layer.update_age(age_encoder.inverse_transform([age])[0])

            sequential_data[Field.AGE] = age

def main():
    app = QApplication(sys.argv)
    window = App()
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())