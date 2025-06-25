import joblib
from sentence_transformers import SentenceTransformer
import pandas as pd
from dbfread import DBF
import pyodbc

color_map = {'BROWN': 'B', 'DK. BRN': 'B', 'LT. BRN': 'B', 'TAN': 'B',
             'GRAY': 'G',  'DK. GRY': 'G', 'LT. GRY': 'G',
             'BLUE': 'G', 'DK. BLU': 'G', 'LT. BLU': 'G',
             'BLACK': 'K',
             'RED': 'R',
             'GREEN': 'L', 'ORANGE': 'O',
             'Other/Varied': 'U', 'WHITE': 'W', 'YELLOW': 'Y'}
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

county = '27'

db_path = 'compiled_data/hennepin.db'

conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=boring_data/cwidata_nps.accdb;'
)

located_wells = DBF('well_data/cwilocs_NPS.dbf')

strat_age_model = joblib.load('trained_models/LGBM_Strat_Age_Model.joblib')
quat_type_model = joblib.load('trained_models/LGBM_Quat_Type_Model.joblib')
recent_type_model = joblib.load('trained_models/LGBM_Recent_Type_Model.joblib')

embedder = SentenceTransformer('all-MiniLM-L6-v2')

mdb_conn = pyodbc.connect(conn_str)
mdb_cursor = mdb_conn.cursor()

f_out = open("comparison.txt", 'w')

for well in located_wells:

    if well['county_c'] != county:
        continue

    if well['relateid'] is None:
        continue

    relate_id = int(well['relateid'])

    mdb_cursor.execute('''
        SELECT * FROM c4st WHERE RELATEID = ?
        ''', well['relateid'])

    layers = mdb_cursor.fetchall()
    prev_strat = -1

    for layer in layers:
        true_strat = layer[6]
        description = layer[3].strip().upper()

        if true_strat is None:
            continue

        if true_strat.startswith('Q'):
            strat = 0
        elif true_strat.startswith('R'):
            strat = 1
        else:
            continue

        true_color = layer[4]

        if true_color is None:
            color = 'U'
            true_color = 'UNKNOWN'
        else:
            color = color_map.get(layer[4].strip(), 'U')

        desc_emb = embedder.encode([description])

        emb_df = pd.DataFrame(desc_emb, columns=[f"emb_{i}" for i in range(desc_emb.shape[1])])
        emb_df['prev_label'] = prev_strat

        prediction = strat_age_model.predict(emb_df)[0]

        if prediction == 0:
            emb_df = pd.DataFrame(desc_emb, columns=[f"emb_{i}" for i in range(desc_emb.shape[1])])
            type_prediction = quat_type_model.predict(emb_df)[0]

            pred_strat = 'Q' + quat_type_map[type_prediction] + 'U'+ color

            if pred_strat != true_strat:
                f_out.write('WELL ID: ' + str(relate_id) + '\n')
                f_out.write('DESCRIPTION: ' + description + '\n')
                f_out.write('COLOR: ' + true_color + '\n')
                f_out.write('PREDICTED: ' + pred_strat + '\n')
                f_out.write('EXPECTED: ' + true_strat + '\n \n')
        elif prediction == 1:
            emb_df = pd.DataFrame(desc_emb, columns=[f"emb_{i}" for i in range(desc_emb.shape[1])])
            type_prediction = recent_type_model.predict(emb_df)[0]

            pred_strat = 'R' + quat_type_map[type_prediction] + 'U' + color

            if pred_strat != true_strat:
                f_out.write('WELL ID: ' + str(relate_id) + '\n')
                f_out.write('DESCRIPTION: ' + description + '\n')
                f_out.write('COLOR: ' + true_color + '\n')
                f_out.write('PREDICTED: ' + pred_strat + '\n')
                f_out.write('EXPECTED: ' + true_strat + '\n \n')
        else:
            f_out.write('WELL ID: ' + str(relate_id) + '\n')
            f_out.write('DESCRIPTION: ' + description + '\n')
            f_out.write('COLOR: ' + true_color + '\n')
            f_out.write('PREDICTED: Bedrock' + '\n')
            f_out.write('EXPECTED: ' + true_strat + '\n\n')

        prev_strat = strat
        f_out.flush()