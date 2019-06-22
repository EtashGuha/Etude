(* ::Package:: *)

Begin["System`Convert`GDALDump`"]


ImportExport`RegisterImport[
  "SDTS",
  {
	"Elements" :> GDALGetElements["SDTS"],
	"Data" | "Graphics"|"GraphicsList"|"LayerTypes":> ImportVectorLabeledData["SDTS"],
	"LayerNames" :> ImportVectorLabels["SDTS"],
	("CoordinateSystem"|"Projection"|"ProjectionName"|"CoordinateSystemInformation"|
	"SpatialRange"|"CoordinateSystem"|"SemimajorAxis"|"SemiminorAxis"|  "InverseFlattening"|"Datum"|
	"LinearUnits"|"CentralScaleFactor"|"StandardParallels"|"ReferenceModel"|"Centering"|"GridOrigin") :> ImportGISMetadata["SDTS"],
	ImportGISMetadata["SDTS"]
  },
  {
    "Data":>GISGetData,
    "LayerTypes":>GISGetLayerTypes,
    "SpatialRange" :> GISGetExtent,
    "CoordinateSystem":>GISGetCoordinateSystem,
    "SemimajorAxis" :> GISGetSemiMajorAxis,
    "SemiminorAxis" :> GISGetSemiMinorAxis,
    "InverseFlattening" :> GISGetInverseFlattening,
    "LinearUnits"  :> GISGetLinearUnits,
    "CoordinateSystemInformation":>GISGetCoordSysInfo,
    "CoordinateSystem":> GISCoordinateSysName,
    "ProjectionName":>GISProjectionName,
    "Projection":>GISProjectionParameters,
    "Graphics" :> GISGraphics,
    "Datum":>GISGetDatum,
    "GraphicsList" :> GISGraphicsList,
    "CentralScaleFactor" :>getSubElement["CentralScaleFactor"],
    "StandardParallels"  :>getSubElement["StandardParallels"],
    "ReferenceModel"  :>getSubElement["ReferenceModel"],
    "Centering"  :>getSubElement["Centering"],
    "GridOrigin":>getSubElement["GridOrigin"]},
 "Sources"-> ImportExport`DefaultSources["GDAL"],
 "AvailableElements" -> {"Centering", "CentralScaleFactor", "CoordinateSystem", 
			"CoordinateSystemInformation", "Data", "Datum", "Graphics", 
			"GraphicsList", "GridOrigin", "InverseFlattening", "LayerNames", 
			"LayerTypes", "LinearUnits", "Projection", "ProjectionName", 
			"ReferenceModel", "SemimajorAxis", "SemiminorAxis", "SpatialRange", 
			"StandardParallels"},
 "DefaultElement" -> "Graphics",
 "FunctionChannels" -> {"FileNames","Directories"},
 "BinaryFormat" -> True
]



End[]
