readme.txt

Minnesota Geological Survey
University of Minnesota

County Well Index [CWI]
CWI version 4 FTP Distribution

current update: 12 July, 2022

**NOTE: As requested by the Minnesota Department of Health, this version
        of CWI does NOT include records for public supply wells.  If you
        need access to the public supply logs, please contact the 
	Drinking Water Protection Program at 
	health.drinkingwater@state.mn.us or 651-201-4700.  


The easiest way to access CWI logs and locations for non-public supply wells is
to use the CWI OnLine website at 

     https://mnwellindex.web.health.state.mn.us/ 

The site provides search and report generation for water well logs along with 
locations viewable on screen with airphoto or USGS quad backgrounds.

The four datafiles provided here are for users that need direct access to the
CWI data tables for their own applications, queries, and reports.  Addtional
information concerning the table formats and field definitions can be obtained
by contacting the MGS as listed below.  The files included here are:

1. cwidata_nps.zip - this is a zipped version of the CWI database, but does NOT
                 include the cwi4VIEW application.  The database is in Access 2002
                 format, but will be convertible to a more current version of
                 Access if opened by said version of the software installed on
                 the target system.  To use this version, you MUST have Access 2002
                 or a more current version of Access installed on your system.
                 Since there is no viewing application included with this version,
                 all queries, reports, or other uses of the data are up to the
                 user to devise.

2. cwilocs_nps.zip - this is a zipped file containing an ArcView shapefile 
                 (cwilocs_nps.shp) of the located and digitized (coordinates
                 determined from quadrangles or plat maps or via GPS) wells.
                 This represents about one third of the wells in the entire
                 CWI database, but includes wells that have geologically
                 interpreted logs and aquifer determinations.  The attribute
                 table for the shapefile includes the attributes from the main
                 data table in the CWI database.  This simplifies the use of 
                 part of the CWI data in GIS applications. To use this dataset, 
                 un-zipping software is required along with ArcView, the free
                 ArcGIS Explorer, or some other GIS software capable of
                 reading the shapefile format.  Coordinates for well locations
                 in the shape file are in UTM zone 15 meters, NAD83.
                 

3. unlocs_nps.zip - this is a zipped file containing a shapefile (unlocs_nps.shp)
                 containing computed locations for "unlocated" water wells - 
                 those wells that have not had their locations field checked and
                 digitized or determined via GPS.  The structure of the shapefile
                 is identical to that of cwilocs_nps.

                 **CAUTION**  These locations are calculated from township-range-
                 section and subsection (if available) information recorded on 
                 well log submitted.  The coordinates may be only those of the
                 center of the section in which the well is located.  Furthermore,
                 any error in the reporting of the TRS can result in large errors
                 in the coordinates.  Do NOT assume these locations are correct
                 for your work.

4. cwi_info.zip - contains information of database table formats and field and
                 code definitions.


5. cwi4view2k_nps.zip

		 This is a "zipped" version of the CWI database with the cwi4VIEW application in 
	 	 Access 2010. Cwi4View is a data viewer developed by Fuliao Li of the Minnesota Department of Health 
		 which provides search and report-generation capabilities for the CWI database.  

		 Save this file to a folder on your machine where you would like to store the data.  Right click on the
		 file to uncompress it to this location.  Once the file has been uncompressed, the application can be 
		 run by double clicking on the "cwi4view2k.mde" file that  should have been created.

		 Cwi4View also requires that Access 2010 or higher be installed on the target system.  If you do not 
		 have Access installed see note 6 below.



Address questions, suggestions, or requests for additional information to:

  Jarrod Cicha                      
  Minnesota Geological Survey   email: cich0060@morris.umn.edu
  2609 Territorial Road                    
  Saint Paul, MN  55114         phone: 612-626-4108



