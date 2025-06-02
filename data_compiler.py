from dbfread import DBF
import pyodbc
import sqlite3


county = '27'
county_name = 'hennepin'
utm_zone = '15'

located_well_table = DBF('well_data/cwilocs_NPS.dbf')
located_well_ids = []

# FETCH ALL LOCATED WELLS IN A GIVEN COUNTY AND CONVERT THEIR UTM TO GPS

for well in located_well_table:

    if well['county_c'] == county:

        # MAKE SURE TO CHANE ZONE_NUMBER BASED OFF OF WHICH COUNTY IS SELECTED
        #lat, lon = utm.to_latlon(int(well['utme']), int(well['utmn']), 14, northern=True)

        new_well = {
            'id': well['relateid'],
            'elevation': well['elevation'],
            'utme': well['utme'],
            'utmn': well['utmn'],
            'bedrock': well['depth2bdrk'],
            'layers': [],
        }

        located_well_ids.append(new_well)


conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={'boring_data/cwidata_nps.accdb'};'
)

conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# FETCH DRILLER DATA ABOUT EACH LOCATED WELL

sql_conn = sqlite3.connect('compiled_data/hennepin.db')
sql_cursor = sql_conn.cursor()

sql_cursor.execute('''
CREATE TABLE IF NOT EXISTS wells (
    relate_id INTEGER PRIMARY KEY,
    utme INTEGER,
    utmn INTEGER,
    bedrock REAL
)
''')

sql_cursor.execute('''
CREATE TABLE IF NOT EXISTS layers (
    layer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    relate_id INTEGER,
    depth_top REAL,
    depth_bot REAL,
    desc TEXT,
    color TEXT,
    hardness TEXT,
    strat TEXT,
    lith_prim TEXT,
    lith_sec TEXT,
    lith_minor TEXT
)
''')

i = 0

for well in located_well_ids:

    cursor.execute("SELECT * FROM c4st WHERE RELATEID = ?", well['id'])
    rows = cursor.fetchall()

    sql_cursor.execute('''
    INSERT INTO wells (relate_id, utme, utmn, bedrock)
    VALUES (?, ?, ?, ?)
    ''', (int(well['id']), int(well['utme']), int(well['utmn']), float(well['bedrock'])))

    for row in rows:

        depth_top = -1
        depth_bot = -1

        if row[1] is not None:
            depth_top = float(well['elevation']) - row[1]

        if row[2] is not None:
            depth_bot = float(well['elevation']) - row[2]
        else:
            depth_bot = depth_top




        sql_cursor.execute('''
        INSERT INTO layers (relate_id, depth_top, depth_bot, desc, color, hardness, strat, lith_prim, lith_sec, lith_minor) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (int(well['id']), depth_top, depth_bot, row[3], row[4], row[5], row[6], row[7], row[8], row[9]))

        i += 1

        if i%500 == 0:
            sql_conn.commit()

sql_conn.commit()