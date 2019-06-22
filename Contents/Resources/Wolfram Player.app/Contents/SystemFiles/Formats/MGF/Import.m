(* ::Package:: *)

Begin["System`Convert`BitmapDump`"]


ImportExport`RegisterImport[
  "MGF",
  ImportMGF,
  {
	 "Graphics"->MGFCellToGraphics,
	 "Image" -> (System`ConvertersDump`GraphicToImage[MGFCellToGraphics[##]]&)
  },
  "Sources" -> ImportExport`DefaultSources["Bitmap"],
  "AvailableElements" -> {"Cell", "Graphics", "Image"},
  "DefaultElement" -> "Image",
  "BinaryFormat" -> True
]


End[]
