import pandas as pd
import arcpy

county_number = 27
county_name = 'hennepin'

cwi_well_data_path = 'cwi_data/cwi5.csv'
cwi_strat_data_path = 'cwi_data/c5st.csv'

gis_glacial_data_path = 'gis_data/hennepin_glacial.gdb'
gis_bedrock_data_path = 'gis_data/hennepin_bedrock.gdb'

cwi_wells_save_path = 'compiled_data/cwi_wells.csv'
cwi_layers_save_path = 'compiled_data/cwi_layers.csv'

cwi_wells = pd.read_csv(cwi_well_data_path, low_memory=False)
strat_layers = pd.read_csv(cwi_strat_data_path, low_memory=False, on_bad_lines='skip')

cwi_wells = cwi_wells[cwi_wells['county_c'] == county_number]
strat_layers = strat_layers[strat_layers['relateid'].isin(cwi_wells['relateid'])]

wells_df = cwi_wells[['relateid', 'elevation', 'utme', 'utmn', 'data_src', 'aquifer']]
wells_df.is_copy = False
layers_df = strat_layers[['relateid', 'depth_top', 'depth_bot', 'color', 'hardness', 'drllr_desc', 'strat',
                                 'lith_prim', 'lith_sec', 'lith_minor']]
layers_df['geo_code'] = None
layers_df['percentage'] = 0
layers_df.is_copy = False

# Returns a list of raster types found in a given workspace
def get_raster_list() -> list:
    raster_list = []

    for raster in arcpy.ListRasters('*'):
        raster = raster.split('_')

        if raster[0] not in raster_list:
            raster_list.append(raster[0])

    return raster_list

# Parses a set list of rasters to see what layers intersect with them
def parse_rasters(raster_list : list):
    for raster in raster_list:

        top_raster = arcpy.Raster(f'{raster}_top')
        top_array = arcpy.RasterToNumPyArray(top_raster, nodata_to_value=None)

        top_x_origin = top_raster.extent.XMin
        top_y_origin = top_raster.extent.YMax
        top_cell_width = top_raster.meanCellWidth
        top_cell_height = top_raster.meanCellHeight

        base_raster = arcpy.Raster(f'{raster}_base')
        base_array = arcpy.RasterToNumPyArray(base_raster, nodata_to_value=None)

        base_x_origin = base_raster.extent.XMin
        base_y_origin = base_raster.extent.YMax
        base_cell_width = base_raster.meanCellWidth
        base_cell_height = base_raster.meanCellHeight

        print(f'STARTING RASTER {raster}')

        for _, well in wells_df.dropna(subset=['utme', 'utmn', 'elevation']).iterrows():

            layers = layers_df[layers_df['relateid'] == well['relateid']]
            layers = layers.dropna(subset=['depth_top', 'depth_bot'])

            for index, layer in layers.iterrows():
                x = int(well['utme'])
                y = int(well['utmn'])

                top_x_shifted = int((x - top_x_origin) / top_cell_width)
                top_y_shifted = int((top_y_origin - y) / top_cell_height)

                base_x_shifted = int((x - base_x_origin) / base_cell_width)
                base_y_shifted = int((base_y_origin - y) / base_cell_height)

                # TODO: Fix this if an error arises, but unlikely
                '''if (top_x_shifted < 0 or top_y_shifted < 0 or base_x_shifted < 0 or base_y_shifted < 0 or
                    top_x_shifted > top_raster.width or top_y_shifted > top_raster.height or
                    base_x_shifted > base_raster.width or base_y_shifted > base_raster.height):
                    print("OUT OF RASTER BOUNDS")
                    continue '''

                gu_top = top_array[top_y_shifted, top_x_shifted]
                gu_base = base_array[base_y_shifted, base_x_shifted]

                if gu_top is None or gu_base is None or gu_top == 0 or gu_base == 0:
                    continue

                elev = float(well['elevation'])
                st_top = elev - float(layer['depth_top'])
                st_bot = elev - float(layer['depth_bot'])
                st_thick = st_top - st_bot

                depth_top = min(gu_top, st_top)
                depth_bot = max(gu_base, st_bot)
                thickness = depth_top - depth_bot

                if thickness <= 0:
                    continue

                percentage = thickness/st_thick

                if layers_df.loc[index, 'percentage'] < percentage:
                    layers_df.loc[index, 'percentage'] = percentage
                    layers_df.loc[index, 'geo_code'] = raster.upper()
    return

arcpy.env.workspace = gis_bedrock_data_path
bedrock_rasters = get_raster_list()
parse_rasters(bedrock_rasters)

arcpy.env.workspace = gis_glacial_data_path
glacial_rasters = get_raster_list()
parse_rasters(glacial_rasters)

wells_df.to_csv(cwi_wells_save_path, mode='a')
layers_df.to_csv(cwi_layers_save_path, mode='a')