(* ::Package:: *)

Begin["System`Convert`GDALDump`"]


ImportExport`RegisterImport[
    "ArcGRID",
	{ (* Raw *)
		("Graphics"|"Data"|"ElevationRange"|"Image"|"ReliefImage") :> ImportGISRasterRaw["ArcGRID"],
		 "Elements" :> GDALGetElements["ArcGRID"],
		Alternatives["DataFormat", "SpatialRange"] :> ImportGISRasterLayerMetadata["ArcGRID"],
		ImportGISRasterMetadata["ArcGRID"]
	},
    { (* Post Process *)
		"Data" :> GISRasterData,
		"DataFormat" -> GISElementsFromCSI["DataFormat"],
		"RasterSize" -> GISElementsFromCSI["RasterSize"],
		"ElevationRange"->getRasterElevationRange,
		"Graphics" :> GISRasterGraphics,
		"SpatialRange" :> GISRasterSRange,
		"SpatialResolution" :> GISRasterSResolution,
		"SemimajorAxis" :> GISGetSemiMajorAxis,
        "SemiminorAxis" :> GISGetSemiMinorAxis,
		"InverseFlattening" :> GISGetInverseFlattening,
  		"LinearUnits"  :> GISGetLinearUnits,
  		"Datum":>GISGetDatum,
 		"CoordinateSystemInformation":>GISGetCoordSysInfo,
		"CoordinateSystem":> GISCoordinateSysName,
		"ProjectionName":>GISProjectionName,
 		"Projection":>GISProjectionParameters,
 		"CentralScaleFactor" :>getSubElement["CentralScaleFactor"],
        "StandardParallels"  :>getSubElement["StandardParallels"],
        "ReferenceModel"  :>getSubElement["ReferenceModel"],
        "Centering"  :>getSubElement["Centering"],
        "GridOrigin":>getSubElement["GridOrigin"],
        "Image" :> GISRasterImage,
        "ReliefImage" :> GISReliefImage
	},
	"Sources" -> ImportExport`DefaultSources["GDAL"],
	"AvailableElements" -> {"Centering", "CentralScaleFactor", "CoordinateSystem",
			"CoordinateSystemInformation", "Data", "DataFormat", "Datum",
			"ElevationRange", "Graphics", "GridOrigin", "Image", "InverseFlattening",
			"LinearUnits", "Projection", "ProjectionName", "RasterSize",
			"ReferenceModel", "ReliefImage", "SemimajorAxis", "SemiminorAxis", "SpatialRange",
			"SpatialResolution", "StandardParallels"},
	"DefaultElement" -> "Graphics",
	"FunctionChannels" -> {"FileNames","Directories"},
	"BinaryFormat" -> True
] 


End[]
