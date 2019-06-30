(* ::Package:: *)

Begin["System`Convert`GDALDump`"]


ImportExport`RegisterImport[
	"USGSDEM",
	{	(* Raw *)
		("ReliefImage"|"Image"|"Graphics"|"Data") :> ImportGISRasterRaw["USGSDEM"],
		"Elements" :> GDALGetElements["USGSDEM"],
		 Alternatives[ "DataFormat", "SpatialRange"] :> ImportGISRasterLayerMetadata["USGSDEM"],
	    ("ElevationRange" |"ElevationResolution"):> evaluateDEMElevRnge,
		ImportGISRasterMetadata["USGSDEM"]
	},
	{ (* Post Process *)
		"Data" :> GISRasterData,
		"DataFormat" -> GISElementsFromCSI["DataFormat"],
		"RasterSize" -> GISElementsFromCSI["RasterSize"],
		"Dimensions" -> (Reverse@GISElementsFromCSI["RasterSize"][##]&),
		"Graphics" :> GISRasterGraphics,
		"SpatialRange" :> GISRasterSRange,
		"SpatialResolution" :> GISRasterSResolution,
		"CoordinateSystem" :> GISGetCoordinateSystem,
		"SemimajorAxis" :> GISGetSemiMajorAxis,
        "SemiminorAxis" :> GISGetSemiMinorAxis,
		"InverseFlattening" :> GISGetInverseFlattening,
  		"LinearUnits"  :> GISGetLinearUnits,
		"CoordinateSystemInformation" :> GISGetCoordSysInfo,
		"Datum":>GISGetDatum,
		"CoordinateSystem" :> GISCoordinateSysName,
        "ProjectionName" :> GISProjectionName,
        "Projection" :>GISProjectionParameters,
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
			"Dimensions", "ElevationRange", "ElevationResolution", "Graphics", 
			"GridOrigin", "Image", "InverseFlattening", "LinearUnits", "Projection", 
			"ProjectionName", "RasterSize", "ReferenceModel", "ReliefImage", "SemimajorAxis", 
			"SemiminorAxis", "SpatialRange", "SpatialResolution", 
			"StandardParallels"},
	"DefaultElement" -> "Graphics",
	"FunctionChannels" -> {"FileNames"},
	"BinaryFormat" -> True
]



End[]
