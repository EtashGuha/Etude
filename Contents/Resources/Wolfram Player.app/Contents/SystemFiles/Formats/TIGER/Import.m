(* ::Package:: *)

Begin["System`Convert`GDALDump`"]


ImportExport`RegisterImport[
 "TIGER",
 {
	"Elements" :> GDALGetElements["TIGER"],
	"Data"|"Graphics"|"GraphicsList"|"LayerTypes" :> ImportVectorLabeledData["TIGER"],
	"LayerNames" :> ImportVectrLabels["TIGER"],
	("SpatialRange"|"SemimajorAxis"|"SemiminorAxis"| "CoordinateSystemInformation"| "InverseFlattening"|
	 "Datum"|"ReferenceModel"):> ImportGISMetadata["TIGER"],
	ImportGISMetadata["TIGER"]
 },
 {
  "Data":>GISGetData,
  "LayerTypes":>GISGetLayerTypes,
  "SpatialRange" :> GISGetExtent,
  "SemimajorAxis" :> GISGetSemiMajorAxis,
  "SemiminorAxis" :> GISGetSemiMinorAxis,
  "InverseFlattening" :> GISGetInverseFlattening,
  "CoordinateSystemInformation":>GISGetCoordSysInfo,
  "Graphics" :> GISGraphics,
  "Datum":>GISGetDatum,
  "GraphicsList" :> GISGraphicsList,
  "ReferenceModel"  :>getSubElement["ReferenceModel"]
 },
 "Sources"-> ImportExport`DefaultSources["GDAL"],
 "AvailableElements" -> {"CoordinateSystemInformation", "Data", "Datum", "Graphics", 
			"GraphicsList", "InverseFlattening", "LayerNames", "LayerTypes", 
			"ReferenceModel", "SemimajorAxis", "SemiminorAxis", "SpatialRange"},
 "DefaultElement" -> "Graphics",
 "FunctionChannels" -> {"FileNames", "Directories"},
 "BinaryFormat" -> True
]


End[]
