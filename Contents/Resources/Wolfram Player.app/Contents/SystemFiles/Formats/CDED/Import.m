(* ::Package:: *)

Begin["System`Convert`CDEDDump`"]


ImportExport`RegisterImport["CDED", {
 "Elements" :> CDEDElements,

   ((*"Description"|*)"SemimajorAxis"|"DataFormat"| "Dimensions"|"RasterSize"|
  "SpatialRange" | "SpatialResolution" | "ElevationResolution"|"ElevationRange"|"SemiminorAxis"|"InverseFlattening"|
  (*"PrimeMeridian"|*) "LinearUnits"|"CoordinateSystemInformation"|"CoordinateSystem"|
  "ProjectionName"|"Projection"|"Datum"|"CentralScaleFactor"|"StandardParallels"|"ReferenceModel"|"Centering"|"GridOrigin"):> evaluateElements,
	ImportCDED},
{"Graphics" :> getGraphics ,
 "SemimajorAxis" :> GISGetSemiMajorAxis,
 "SemiminorAxis" :> GISGetSemiMinorAxis,
 "InverseFlattening" :> GISGetInverseFlattening,
 "LinearUnits"  :> GISGetLinearUnits,
 "CoordinateSystemInformation":>GISGetCoordSysInfo,
 "CoordinateSystem":> GISCoordinateSysName,
 "ProjectionName":>GISProjectionName,
 "Datum":> GISGetDatum,
 "Projection":>GISProjectionParameters,
 "ProjectionName":>GISProjectionName,
 "CentralScaleFactor" :>getSubElement["CentralScaleFactor"],
 "StandardParallels"  :>getSubElement["StandardParallels"],
 "ReferenceModel"  :>getSubElement["ReferenceModel"],
 "Centering"  :>getSubElement["Centering"],
 "GridOrigin":>getSubElement["GridOrigin"],
 "Image" :> getImage,
 "ReliefImage" :> getReliefImage
 }
,
"Sources" -> {"Convert`CDED`","GDAL.exe"},
"DefaultElement" -> "Graphics",
"AvailableElements" -> {"Centering", "CentralScaleFactor", 
  "CoordinateSystem", "CoordinateSystemInformation", "Data", "Datum", 
  "Dimensions", "ElevationRange", "ElevationResolution", "Graphics", 
  "GridOrigin", "Image", "InverseFlattening", "LinearUnits", "Projection", 
  "ProjectionName", "RasterSize", "ReferenceModel", "ReliefImage", "SemimajorAxis", 
  "SemiminorAxis", "SpatialRange", "SpatialResolution", 
  "StandardParallels"},
"FunctionChannels"->{"FileNames","Directories"}
]


End[]
