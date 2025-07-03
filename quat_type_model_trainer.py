import sqlite3

import joblib
import pandas as pd
#from datasets import Dataset

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

print("QUERYING AND PREPARING DATA")

sql_conn = sqlite3.connect('compiled_data/hennepin.db')

query = '''
SELECT
    layers.layer_id,
    layers.relate_id,
    layers.desc,
    layers.strat
FROM
    layers
'''

data = pd.read_sql_query(query, sql_conn)

data = data.dropna(subset=['strat'])
data = data.dropna(subset=['desc'])

data = data[data['strat'].str.startswith('Q')]

label_map = {
    'B': 0, # Boulders
    'C': 1, # Clay
    'F': 2, # Sand
    'G': 3, # Gravel
    'I': 4, # Silt
    'J': 5, # Clay/Silt
    'H': 6, # Sand/Gravel
    'L': 7, # Clay/Sand
    'W': 8, # Clay/Sand/Silt
    'P': 9, # Clay/Sand/Silt/Gravel
    'N': 10, # Silt/Sand/Gravel
    'R': 11, # Sand/Gravel/Broken Rock
    'S': 12, # Organics
    'T': 13, # Till
    'U': 14, #Unknown
}

data['label'] = data['strat'].str[1].map(label_map).fillna(14)
data['label'] = data['label'].astype(int)

print("PREPARING MODEL")

model = SentenceTransformer('all-MiniLM-L6-v2')

desc_embeddings = model.encode(data['desc'].tolist(), show_progress_bar=True)

embedding_df = pd.DataFrame(desc_embeddings, columns=[f"emb_{i}" for i in range(desc_embeddings.shape[1])])
data = data.reset_index(drop=True)

data_full = embedding_df

print("TRAINING")

X = data_full
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=127)

clf = LGBMClassifier(class_weight='balanced')
clf.fit(X_train, y_train)

print("EVALUATING")

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("SAVING")

joblib.dump(clf, 'trained_models/LGBM_Quat_Type_Model.joblib')