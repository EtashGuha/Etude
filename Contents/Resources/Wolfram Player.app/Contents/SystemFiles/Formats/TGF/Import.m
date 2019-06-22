(* ::Package:: *)

Begin["System`Convert`TGFDump`"]


ImportExport`RegisterImport[
  (** the format name must match the folder name **)
  "TGF",

  (** raw converters **)
  {
    (* conditional raw importer *) 
    "Graph"|"VertexCount" :> ImportTGFGraph,
    (* default raw importer as the last item in the list *)
    ImportTGF
  },
  
  (** post-importers **)
  {
     "Graphics" :> ImportTGFGraphics
  },

  (** options for RegisterImport **)
  "AvailableElements" -> {"AdjacencyMatrix","EdgeAttributes",
     "EdgeRules","Graph","Graphics","VertexAttributes",
     "VertexCount", "VertexList"},
  "FunctionChannels" -> {"Streams"},
  "DefaultElement" -> "Graph"
]


End[]
