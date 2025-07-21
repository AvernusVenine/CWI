import arcpy

gis_data_path = 'cwi_data/Benton.gdb'

arcpy.env.workspace = gis_data_path

#print(arcpy.ListRasters('*'))
print(arcpy.ListDatasets())