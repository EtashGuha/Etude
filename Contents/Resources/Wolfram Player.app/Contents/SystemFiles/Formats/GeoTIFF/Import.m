(* ::Package:: *)

Begin["System`Convert`GDALDump`"]


$sourcesForGDAL = {"Convert`CommonGraphics`"};
$sourcesForGDAL = Join[$sourcesForGDAL, ImportExport`DefaultSources["GDAL"]];


ImportExport`RegisterImport[
  "GeoTIFF",
 { (* Raw *)
	("Image"|"Graphics"|"Data"|"ElevationRange") :> ImportGISRasterRaw["GeoTIFF"],
    "Summary" :> CreateSummary,
    "BitDepth" :> GetBitDepth,
    "Channels" :> GetChannels,
	"Elements" :> GDALGetElements["GeoTIFF"],
	Alternatives[ "DataFormat", "CornerCoordinates", "ColorInterpretation",
			 "SpatialRange"] :> ImportGISRasterLayerMetadata["GeoTIFF"],
	ImportGISRasterMetadata["GeoTIFF"]
 },
 { (* Post Process *)

	"Data" :> GISRasterData,
	"DataFormat" -> GISElementsFromCSI["DataFormat"],
	"RasterSize" -> GISElementsFromCSI["RasterSize"],
	"ColorSpace" -> GISElementsFromCSI["ColorInterpretation"],
	"ElevationRange"->getRasterElevationRange,
	"Graphics" :> GISRasterGraphics,
	"SpatialRange" :> GISRasterSRange,
	"SpatialResolution" :> GISRasterSResolution,
	"SemimajorAxis" :> GISGetSemiMajorAxis,
    "SemiminorAxis" :> GISGetSemiMinorAxis,
	"InverseFlattening" :> GISGetInverseFlattening,
	"LinearUnits"  :> GISGetLinearUnits,
	"CoordinateSystemInformation":>GISGetCoordSysInfo,
	"CoordinateSystem":> GISCoordinateSysName,
	"ProjectionName":>GISProjectionName,
	"Datum":>GISGetDatum,
	"Projection":>GISProjectionParameters,
	"CentralScaleFactor" :>getSubElement["CentralScaleFactor"],
    "StandardParallels"  :>getSubElement["StandardParallels"],
    "ReferenceModel"  :>getSubElement["ReferenceModel"],
    "Centering"  :>getSubElement["Centering"],
    "GridOrigin":>getSubElement["GridOrigin"],
    "Image" :> GISRasterImage
  },
 "Sources" -> {$sourcesForGDAL},
 "AvailableElements" -> {"Centering", "CentralScaleFactor", "ColorSpace", "CoordinateSystem", 
			"CoordinateSystemInformation", "Data", "DataFormat", "Datum", 
			"ElevationRange", "Graphics", "GridOrigin", "Image", "InverseFlattening", 
			"LinearUnits", "Projection", "ProjectionName", "RasterSize", 
			"ReferenceModel", "SemimajorAxis", "SemiminorAxis", "SpatialRange", 
			"SpatialResolution", "StandardParallels", "Summary", "BitDepth", "Channels"},
 "DefaultElement" -> "Image",
 "FunctionChannels" -> {"FileNames","Directories"},
 "BinaryFormat" -> True
]


End[]
