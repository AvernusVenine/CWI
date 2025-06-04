import sqlite3
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import joblib


## TODO: Refine ONEHOT encoding to account for all possible values
## TODO: Refine description to possibly make it ONEHOT, likely have to ask for a list from a QUAT GEOLOGIST

sql_conn = sqlite3.connect('compiled_data/hennepin.db')

query = '''
SELECT
    layers.depth_top,
    layers.depth_bot,
    layers.desc,
    layers.color,
    layers.hardness,
    wells.utme,
    wells.utmn,
    wells.bedrock,
    layers.strat,
    layers.lith_prim,
    layers.lith_sec,
    layers.lith_minor
FROM
    layers
JOIN
    wells ON layers.relate_id = wells.relate_id;
'''

df = pd.read_sql_query(query, sql_conn)

# FILL MISSING DATA WITH UNKNOWNS AND DROP NULL STRAT ROWS

df = df.dropna(subset=['strat'])
df = df.dropna(subset=['desc'])

df['color'] = df['color'].fillna('UNKNOWN')
df['hardness'] = df['hardness'].fillna('UNKNOWN')

X = df[['depth_top', 'depth_bot', 'desc', 'color', 'hardness', 'utme', 'utmn', 'bedrock']]
y = df['strat']




preprocessor = ColumnTransformer(
    transformers=[
        ('desc', TfidfVectorizer(), 'desc'),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['color', 'hardness']),
        ('num', 'passthrough', ['depth_top', 'depth_bot', 'utme', 'utmn', 'bedrock'])
    ]
)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'trained_models/hennepin_trained_GBT.joblib')

y_pred = pipeline.predict(X_test)
wrong_predictions = X_test[y_pred != y_test]

print("Accuracy:", pipeline.score(X_test, y_test))