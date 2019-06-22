(* ::Package:: *)

Begin["System`Convert`BitmapDump`"]


ImportExport`RegisterExport[
  "BMP",
  System`ConvertersDump`ExportRasterElements["BMP", ExportBMP, ##]&,
  "FunctionChannels" -> {"Streams"},
  "DefaultElement" -> Automatic,
  "Sources" -> ImportExport`DefaultSources["Bitmap"],
  "Options" -> {"DataType", "BitDepth", "ColorSpace", "HorizontalResolution", "RLECompression", "VerticalResolution"},
  "BinaryFormat" -> True,
  "AlphaChannel" -> True
]


End[]
