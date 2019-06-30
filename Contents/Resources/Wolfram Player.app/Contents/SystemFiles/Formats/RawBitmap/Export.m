(* ::Package:: *)

Begin["System`ConvertersDump`"]


ImportExport`RegisterExport[
  "RawBitmap",
  ExportRasterElements["RawBitmap", System`Convert`BitmapDump`ExportRawBitmap, ##]&,
  "Sources" -> ImportExport`DefaultSources["Bitmap"],
  "FunctionChannels" -> {"Streams"},
  "DefaultElement" -> Automatic,
  "Options" -> {"DataType", "BitDepth", "ColorSpace"},
  "BinaryFormat" -> True
]


End[]
