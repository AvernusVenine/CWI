import pandas as pd
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import random

#TODO: Need to get a list of data_src that come directly from geologists or trusted sources

cwi_well_data_path = 'cwi_data/cwi5.csv'
cwi_layer_data_path = 'cwi_data/c5st.csv'

print("PREPARING DATA")

cwi_wells = pd.read_csv(cwi_well_data_path, low_memory=False, on_bad_lines='skip')
cwi_layers = pd.read_csv(cwi_layer_data_path, low_memory=False, on_bad_lines='skip')

#TODO: Possibly remove to allow for null location values?
cwi_wells = cwi_wells.dropna(subset=['elevation', 'utmn', 'utme'])
cwi_wells = cwi_wells.fillna(value={
    'data_src': -1
})

cwi_layers = cwi_layers.drop(columns=['objectid', 'c5st_seq_no', 'wellid', 'concat', 'stratcode_gen', 'stratcode_detail'])

cwi_layers = cwi_layers.dropna(subset=['strat'])
cwi_layers.loc[cwi_layers['depth_bot'].isna(), 'depth_bot'] = cwi_layers['depth_top']
cwi_layers = cwi_layers.fillna(value={
    'color': 'UNKNOWN',
    'hardness': 'MEDIUM',
    'drllr_desc': ''
})

# TODO: Be sure to map these outputs back!
# Process strat to minimize the amount of possible outputs by removing the color and sorting values
cwi_layers['strat'] = cwi_layers['strat'].replace({
    'RMMF': 'FILL',
    'PITT': 'X',
    'PVMT': 'Y'
})
cwi_layers['age'] =  cwi_layers['strat'].str[0]

cwi_layers['age'] = cwi_layers['age'].str.replace(r'^[WJB]', 'Z', regex=True)

cwi_layers['elevation'] = cwi_layers['relateid'].map(cwi_wells.set_index('relateid')['elevation'])
cwi_layers['utme'] = cwi_layers['relateid'].map(cwi_wells.set_index('relateid')['utme'])
cwi_layers['utmn'] = cwi_layers['relateid'].map(cwi_wells.set_index('relateid')['utmn'])

cwi_layers = cwi_layers.dropna(subset=['utme', 'utmn', 'elevation'])

cwi_layers['true_depth_top'] = cwi_layers['depth_top']
cwi_layers['true_depth_bot'] = cwi_layers['depth_bot']

# Ignore 'No Record' and 'Indeterminate' classes as they are invalid classifications
cwi_layers = cwi_layers[~cwi_layers['age'].isin(['N', 'I'])]

age_cat = cwi_layers['age'].astype('category')
cwi_layers['age_cat'] = age_cat.cat.codes
joblib.dump(list(age_cat.cat.categories), 'trained_models/age_categories.joblib')

print("EMBEDDING DESCRIPTIONS")

model = SentenceTransformer('all-MiniLM-L6-v2')

desc_embeddings = model.encode(cwi_layers['drllr_desc'].tolist(), show_progress_bar=True)
embedding_df = pd.DataFrame(desc_embeddings, columns=[f"emb_{i}" for i in range(desc_embeddings.shape[1])])

print("PERFORMING PCA")

# TODO: Try models with lower variance percentage here and compare for research
pca = PCA(n_components=0.95)
pca_embeddings = pca.fit_transform(embedding_df)
pca_embeddings_df = pd.DataFrame(pca_embeddings, columns=[f'pca_emb_{i}' for i in range(pca_embeddings.shape[1])])
joblib.dump(pca, 'trained_models/embedding_pca.joblib')

layer_features = cwi_layers[['true_depth_top', 'true_depth_bot', 'utme', 'utmn', 'relateid', 'age_cat',
                             'strat', 'color', 'drllr_desc', 'elevation']]

all_features = pd.concat([layer_features.reset_index(drop=True), pca_embeddings_df.reset_index(drop=True)], axis=1)
all_features = all_features.sort_values(by=['relateid', 'true_depth_top'], ascending=[True, False])
all_features['prev_age_cat'] = (
    all_features.groupby('relateid')['age_cat']
    .shift(1)
    .fillna(-1)
    .astype(int)
)

scale_cols = ['true_depth_top', 'true_depth_bot', 'elevation', 'utme', 'utmn']
scaler = StandardScaler()
all_features[scale_cols] = scaler.fit_transform(all_features[scale_cols])

#TODO: Remove once done with
all_features.to_csv('compiled_data/layers_all_features.csv', index=False)

# A custom SMOTE function that only creates faux data points from given data columns while duplicating the rest
def fit_smote(X_df : pd.DataFrame, y_df : pd.DataFrame, count : int, label : int, label_col : str, random_state : int, data_cols : list):

    random.seed(random_state)

    X_filtered_df = X_df[y_df[label_col] == label]
    df_size = X_filtered_df.shape[0]

    X_new = []
    y_new = []

    for i in range(count):
        df_index_one = random.randint(0, df_size - 1)
        df_index_two = random.randint(0, df_size - 1)

        new_row = X_filtered_df.iloc[df_index_one].copy()

        for col in data_cols:
            df_val_one = X_filtered_df.iloc[df_index_one][col]
            df_val_two = X_filtered_df.iloc[df_index_two][col]

            rand_float = random.uniform(min(df_val_one, df_val_two), max(df_val_one, df_val_two))

            new_row[col] = rand_float

        X_new.append(new_row)
        y_new.append(label)

    X_new = pd.DataFrame(X_new)
    y_new = pd.DataFrame(y_new, columns=[label_col])

    return pd.concat([X_df, X_new], ignore_index=True), pd.concat([y_df, y_new], ignore_index=True)

def train_age_classifier():
    print("TRAINING AGE CLASSIFIER")

    y = all_features[['age_cat']].copy()
    #X = all_features.drop(columns=['relateid', 'age_cat', 'strat', 'color', 'drllr_desc'])

    X_train, X_test, y_train, y_test = train_test_split(
        all_features, y,
        test_size=0.2,
        random_state=127,
    )

    label_mapping = dict(zip(age_cat.cat.categories, range(len(age_cat.cat.categories))))
    inv_label_mapping = {v: k for k, v in label_mapping.items()}

    X_train, y_train = fit_smote(X_train, y_train, 10000, label_mapping['X'], 'age_cat', 127,
                                 ['utme', 'utmn', 'elevation', 'true_depth_top', 'true_depth_bot']
                                 )
    X_train, y_train = fit_smote(X_train, y_train, 10000, label_mapping['Y'], 'age_cat', 127,
                                 ['utme', 'utmn', 'elevation', 'true_depth_top', 'true_depth_bot']
                                 )
    X_train, y_train = fit_smote(X_train, y_train, 50000, label_mapping['F'], 'age_cat', 127,
                                 ['utme', 'utmn', 'elevation', 'true_depth_top', 'true_depth_bot']
                                 )

    X_train = pd.concat([
        X_train[X_train['age_cat'] != 'Q'],
        X_train[X_train['age_cat'] == 'Q'].sample(frac=0.25, random_state=127)
    ], ignore_index=True)

    X_train = X_train.drop(columns=['relateid', 'age_cat', 'strat', 'color', 'drllr_desc'])
    X_test = X_test.drop(columns=['relateid', 'age_cat', 'strat', 'color', 'drllr_desc'])

    y_test = y_test.squeeze()
    y_train = y_train.squeeze()

    age_classifier = LGBMClassifier(
        n_estimators=200,
        verbose=-1,
        boosting_type='dart',
        device='gpu',
        class_weight='balanced',
        min_child_samples=20,
        min_data_in_leaf=20
    )
    age_classifier.fit(X_train, y_train)

    joblib.dump(age_classifier, 'trained_models/GBT_Age_Model.joblib')

    print("EVALUATING AGE CLASSIFIER")

    y_pred = age_classifier.predict(X_test)

    y_test_labels = pd.Series(y_test).map(inv_label_mapping)
    y_pred_labels = pd.Series(y_pred).map(inv_label_mapping)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)

    labels = list(age_cat.cat.categories)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def train_quat_classifier():
    print("TRAINING QUATERNARY CLASSIFIER")

    quat_layers = all_features[all_features['strat'].str.startswith(('Q', 'R'))]

    quat_type = quat_layers['strat'].str[1]
    quat_type_cat = quat_type.astype('category')
    quat_type = quat_type_cat.cat.codes
    joblib.dump(list(quat_type_cat.cat.categories), 'trained_models/quat_type_categories.joblib')

    X = quat_layers.drop(columns=['true_depth_top', 'true_depth_bot', 'utme', 'utmn', 'relateid', 'strat',
                                  'color', 'drllr_desc', 'elevation'])
    y = quat_type

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=127)

    quat_classifier = LGBMClassifier(class_weight='balanced', verbose=-1)
    quat_classifier.fit(X_train, y_train)

    joblib.dump(quat_classifier, 'trained_models/GBT_Quat_Model.joblib')

    print("EVALUATING QUAT TYPE CLASSIFIER")

    y_pred = quat_classifier.predict(X_test)

    y_test_labels = quat_type_cat.cat.categories[y_test.values]
    y_pred_labels = quat_type_cat.cat.categories[y_pred]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

#TODO: Try by initially grouping bedrocks based on formation and see accuracy there

def train_bedrock_classifier():
    print("TRAINING BEDROCK CLASSIFIER")

    bedrock_layers = all_features[~all_features['strat'].str.startswith(('Q', 'R', 'B', 'F', 'X', 'Y'))]

    bedrock_cat = bedrock_layers['strat'].astype('category')
    bedrock = bedrock_cat.cat.codes
    joblib.dump(bedrock_cat.cat.categories, 'trained_models/bedrock_categories.joblib')

    X = bedrock_layers.drop(columns=['relateid', 'strat', 'color', 'drllr_desc'])
    y = bedrock

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=127)

    bedrock_classifier = LGBMClassifier(class_weight='balanced', verbose=-1)
    bedrock_classifier.fit(X_train, y_train)

    joblib.dump(bedrock_classifier, 'trained_models/GBT_Bedrock_Model.joblib')

    print("EVALUATING BEDROCK CLASSIFIER")

    y_pred = bedrock_classifier.predict(X_test)

    y_test_labels = bedrock_cat.cat.categories[y_test.values]
    y_pred_labels = bedrock_cat.cat.categories[y_pred]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))


train_age_classifier()
#train_quat_classifier()
#train_bedrock_classifier()

