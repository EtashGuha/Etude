(* ::Package:: *)

Begin["System`Convert`EPSDump`"]


ImportExport`RegisterImport[
  "EPS",
  {
	"Graphics"		:> ImportEPSGraphics,
	"Image"			:> ImportEPSImage,
	"Plaintext"		:> ImportEPSPlaintext,
	ImportEPSPreview
  },
  "AvailableElements" -> {"Graphics", "Image", "Plaintext", "Preview"},
  "DefaultElement" -> "Graphics",
  "Sources" -> ImportExport`DefaultSources[{"EPS", "PDF"}],
  "FunctionChannels" -> {"FileNames"},
  "BinaryFormat" -> True
]


End[]
