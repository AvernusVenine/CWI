from dbfread import DBF
import pyodbc
import sqlite3

county = '27'

db_path = 'compiled_data/hennepin.db'

conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={'boring_data/cwidata_nps.accdb'};'
)

sql_conn = sqlite3.connect(db_path)
sql_cursor = sql_conn.cursor()

## WELL TABLE ##
sql_cursor.execute('''
CREATE TABLE IF NOT EXISTS wells (
    relate_id INTEGER PRIMARY KEY,
    utme INTEGER,
    utmn INTEGER,
    elevation REAL,
    bedrock REAL
)
''')

## LAYER TABLE ##
sql_cursor.execute('''
CREATE TABLE IF NOT EXISTS layers (
    layer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    relate_id INTEGER,
    depth_top REAL,
    depth_bot REAL,
    desc TEXT,
    
    is_bedrock INTEGER DEFAULT 0,
    
    black INTEGER DEFAULT 0,
    blue INTEGER DEFAULT 0,
    brown INTEGER DEFAULT 0,
    green INTEGER DEFAULT 0,
    gray INTEGER DEFAULT 0,
    olive INTEGER DEFAULT 0,
    orange INTEGER DEFAULT 0,
    pink INTEGER DEFAULT 0,
    purple INTEGER DEFAULT 0,
    red INTEGER DEFAULT 0,
    white INTEGER DEFAULT 0,
    yellow INTEGER DEFAULT 0,
    tan INTEGER DEFAULT 0,
    silver INTEGER DEFAULT 0,
    light INTEGER DEFAULT 0,
    dark INTEGER DEFAULT 0,
    varied INTEGER DEFAULT 0,
    
    very_soft INTEGER DEFAULT 0,
    soft INTEGER DEFAULT 0,
    medium INTEGER DEFAULT 0,
    hard INTEGER DEFAULT 0,
    very_hard INTEGER DEFAULT 0,
    
    strat TEXT,
    previous_strat TEXT,
    lith_prim TEXT,
    lith_sec TEXT,
    lith_minor TEXT
)
''')

located_wells = DBF('well_data/cwilocs_NPS.dbf')

mdb_conn = pyodbc.connect(conn_str)
mdb_cursor = mdb_conn.cursor()


for well in located_wells:

    ## INSERT NEW WELL INTO wells TABLE

    if well['county_c'] != county:
        continue

    if well['relateid'] is None:
        continue

    relate_id = int(well['relateid'])
    utme = int(well['utme']) if well['utme'] is not None else -1
    utmn = int(well['utmn']) if well['utmn'] is not None else -1
    elevation = float(well['elevation']) if well['elevation'] is not None else -1
    bedrock = float(well['depth2bdrk']) if well['depth2bdrk'] is not None else -1

    sql_cursor.execute('''
    INSERT INTO wells (relate_id, utme, utmn, elevation, bedrock)
    VALUES (?, ?, ?, ?, ?)
    ''', (relate_id, utme, utmn, elevation, bedrock))

    ## COMPILE LAYERS OF WELL INTO THEIR RESPECTIVE TABLES

    mdb_cursor.execute('''
    SELECT * FROM c4st WHERE RELATEID = ?
    ''', relate_id)

    layers = mdb_cursor.fetchall()

    for layer in layers:

        depth_top = elevation - float(layer[1]) if layer[1] is not None else -1
        depth_bot = elevation - float(layer[2]) if layer[2] is not None else -1

        ## DETERMINE IF THE LAYER IS BEDROCK OR NOT
        strat = layer[6]
        is_bedrock = 1 if not strat.startswith(("Q", "W", "R", "U")) else 0

        sql_cursor.execute('''
        INSERT INTO layers (relate_id, depth_top, depth_bot, desc, is_bedrock, strat, lith_prim, lith_sec, lith_minor)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (relate_id, depth_top, depth_bot, layer[3], is_bedrock, strat, layer[7], layer[8], layer[9]))

        layer_id = sql_cursor.lastrowid

        ## ONEHOT ENCODING OF COLORS

        color = layer[4]

        color_map = {
            'blue' : ['BLU', 'BLUE'],
            'black' : ['BLK', 'BLACK'],
            'brown' : ['BRN', 'BROWN'],
            'green' : ['GRN', 'GREEN'],
            'gray' : ['GRY', 'GRAY'],
            'olive' : ['OLV', 'OLIVE'],
            'orange' : ['ORN', 'ORANGE'],
            'pink' : ['PNK', 'PINK'],
            'purple' : ['PUR', 'PURPLE'],
            'red' : ['RED'],
            'white' : ['WHT', 'WHITE'],
            'yellow' : ['YEL', 'YELLOW'],
            'tan' : ['TAN'],
            'silver' : ['SLV', 'SILVER'],
            'light' : ['LT.', 'LIGHT'],
            'dark' : ['DK.', 'DARK']
        }

        for col, keywords in color_map.items():
            if any(keyword in color for keyword in keywords):
                sql_cursor.execute(f'''
                    UPDATE layers SET {col} = ? WHERE layer_id = ?
                ''', (1, layer_id))

        ## ONEHOT ENCODING OF HARDNESS

        hardness = layer[5]

        hardness_map = {
            'very_soft' : ['V.SFT'],
            'soft' : ['SOFT', 'M.SOFT', 'SFT'],
            'medium' : ['MEDIUM', 'M.SOFT', 'M.HARD', 'MED', 'SFT-HRD'],
            'hard' : ['HARD', 'HRD', 'M.HARD'],
            'very_hard' : ['V.HRD']
        }

        for col, keywords in hardness_map.items():
            if any(keyword in hardness for keyword in keywords):
                sql_cursor.execute(f'''
                    UPDATE layers SET {col} = ? WHERE layer_id = ?
                ''', (1, layer_id))


        ## NATURAL LANGUAGE PROCESSOR TO DETERMINE ONEHOT ENCODING OF DESCRIPTION