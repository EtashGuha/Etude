(* ::Package:: *)

Begin["System`Convert`GDALDump`"]


ImportExport`RegisterImport[
  "SHP",
  {
	"Elements" :> GDALGetElements["SHP"],
	"Data"|"Graphics"|"GraphicsList"|"LayerTypes" :> ImportVectorLabeledData["SHP"],
	"LayerNames" :> ImportVectrLabels["SHP"],
	("SpatialRange"|"CoordinateSystem"|"SemimajorAxis"|"SemiminorAxis"| "CoordinateSystemInformation"| "InverseFlattening"|"Datum"|
	"LinearUnits" |"CoordinateSystem"|"ProjectionName"|"Projection"|
	"CentralScaleFactor"|"StandardParallels"|"ReferenceModel"|"Centering"|"GridOrigin"):> ImportGISMetadata["SHP"],
	ImportGISMetadata["SHP"]
  },
 {
  "Data":>GISGetData,
  "LayerTypes" :>GISGetLayerTypes,
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
  "Datum":>GISGetDatum,
  "Graphics" :> GISGraphics,
  "GraphicsList" :> GISGraphicsList,
  "CentralScaleFactor" :>getSubElement["CentralScaleFactor"],
  "StandardParallels"  :>getSubElement["StandardParallels"],
  "ReferenceModel"  :>getSubElement["ReferenceModel"],
  "Centering"  :>getSubElement["Centering"],
  "GridOrigin":>getSubElement["GridOrigin"]
  },
 "Sources"->ImportExport`DefaultSources["GDAL"],
 "AvailableElements" -> {"Centering", "CentralScaleFactor", "CoordinateSystem", 
			"CoordinateSystemInformation", "Data", "Datum", "Graphics", 
			"GraphicsList", "GridOrigin", "InverseFlattening", "LayerNames", 
			"LayerTypes", "LinearUnits", "Projection", "ProjectionName", 
			"ReferenceModel", "SemimajorAxis", "SemiminorAxis", "SpatialRange", 
			"StandardParallels"},
 "DefaultElement" -> "Graphics",
 "FunctionChannels" -> {"FileNames", "Directories"},
 "BinaryFormat" -> True
]



End[]
