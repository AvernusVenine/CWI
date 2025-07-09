import arcpy

gis_data_path = 'gis_data/dodge_bedrock.gdb'

arcpy.env.workspace = gis_data_path

print(arcpy.ListRasters('*'))
