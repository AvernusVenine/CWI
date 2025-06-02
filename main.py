import sqlite3
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


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

#df['depth_top'] = df['depth_top'].fillna(-10000)
#df['depth_bot'] = df['depth_bot'].fillna(-10000)
df['desc'] = df['desc'].fillna('')
df['color'] = df['color'].fillna('UNKNOWN')
df['hardness'] = df['hardness'].fillna('UNKNOWN')
#df['utme'] = df['utme'].fillna(-1)
#df['utmn'] = df['utmn'].fillna(-1)
#df['bedrock'] = df['bedrock'].fillna(-10000)

df = df.dropna(subset=['strat'])

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

print("Accuracy:", pipeline.score(X_test, y_test))