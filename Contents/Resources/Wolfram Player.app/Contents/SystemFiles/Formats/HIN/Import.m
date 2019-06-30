(* ::Package:: *)

Begin["System`Convert`HINDump`"]


ImportExport`RegisterImport[
  "HIN",
  {
    "Elements" -> HINImportElements,
    HINImport
  },
  {
	"Graphics3D":> HINImportGraphics3D,
	"StructureDiagram" :> HINImportStructure
  },
  "FunctionChannels" -> {"Streams"},
  "AvailableElements" -> {"Comments", "EdgeRules", "EdgeTypes", "FormalCharges", "Graphics3D", "MassNumbers", "StructureDiagram", "VertexCoordinates", "VertexTypes"},
  "DefaultElement" -> "Graphics3D"
]


End[]

