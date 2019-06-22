(* ::Package:: *)

Begin["System`ConvertersDump`"]


ImportExport`RegisterExport[
    "XBM",
	System`Convert`BitmapDump`ExportXBitmap[##]&,
	"Sources" -> ImportExport`DefaultSources["Bitmap"],
	"FunctionChannels" -> {"Streams"},
	"Options" -> {"DataType", "BitDepth", "ColorSpace"}
	(* no "DefaultElement" needed.  converter handles it *)
]


End[]
