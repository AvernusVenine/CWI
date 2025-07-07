import sqlite3
import joblib
import pandas as pd
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
    layers.strat,
    layers.previous_strat
FROM
    layers
'''

data = pd.read_sql_query(query, sql_conn)

data = data.dropna(subset=['strat'])
data['desc'] = data['desc'].fillna('')

#TODO: Need more data for Wisc and Unknown to be valid guesses!

label_map = {
    'A': -1, # Air/Nothing
    'Q': 0, # Quaternary/Pleistocene
    'R': 1, # Recent
    'W': 0, # Quaternary/Wisconsinan
    'U': 0, # Unknown/UREG
}

data['label'] = data['strat'].str[0].map(label_map).fillna(2)
data['label'] = data['label'].astype(int)

data['prev_label'] = data['previous_strat'].str[0].map(label_map).fillna(2)
prev_labels = data['prev_label'].astype(int)

print("PREPARING MODEL")

model = SentenceTransformer('all-MiniLM-L6-v2')

desc_embeddings = model.encode(data['desc'].tolist(), show_progress_bar=True)

embedding_df = pd.DataFrame(desc_embeddings, columns=[f"emb_{i}" for i in range(desc_embeddings.shape[1])])
data = data.reset_index(drop=True)

prev_labels = prev_labels.rename('prev_label')
data_full = pd.concat([embedding_df, prev_labels.reset_index(drop=True)], axis=1)

print("TRAINING")

X = data_full
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=127)

# Train a classifier
clf = LGBMClassifier()
clf.fit(X_train, y_train, categorical_feature=['prev_label'])

print("EVALUATING")

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("SAVING")

joblib.dump(clf, 'trained_models/old/LGBM_Strat_Age_Model.joblib')