(* ::Package:: *)

Begin["System`ConvertersDump`"]


ImportExport`RegisterExport[
    "TIFF",
	System`Convert`CommonGraphicsDump`ExportElementsToRasterFormat["TIFF", ##]&,
	"Options" -> {"ColorSpace", ColorSpace, "Comments", "ImageEncoding", "CopyrightNotice", 
		"CameraTopOrientation", "DeviceManufacturer", "Device", "Author", 
		"ImageResolution", ByteOrdering, "BitDepth"},
	"BinaryFormat" -> True,
	"AlphaChannel" -> True,
	"Sources" -> {"Convert`Exif`","Convert`CommonGraphics`" }
]

End[]
