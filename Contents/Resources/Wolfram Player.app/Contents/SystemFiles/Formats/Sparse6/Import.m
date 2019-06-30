(* ::Package:: *)

Begin["System`Convert`GraphDump`"]


ImportExport`RegisterImport[
  "Sparse6",
  {
	{"AdjacencyMatrix"}:>ImportSparse6[All],
	{"AdjacencyMatrix", n_Integer}:>ImportSparse6[{n}],
	{"AdjacencyMatrix", {n__Integer}}:>ImportSparse6[{n}],

	{"EdgeRules"}:>ImportSparse6EdgeRules[All],
	{"EdgeRules", n_Integer}:>ImportSparse6EdgeRules[{n}],
	{"EdgeRules", {n__Integer}}:>ImportSparse6EdgeRules[{n}],

	{"Graph"} :> ImportSparse6Graph,

	{"GraphList"} :> ImportSparse6GraphList[All],
	{"GraphList", n_Integer} :> ImportSparse6GraphList[{n}],
	{"GraphList", {n__Integer}} :> ImportSparse6GraphList[{n}],

	{"Graphics"}:>ImportSparse6Graphics,

	{"GraphicsList"}:>ImportSparse6GraphicsList[All],
	{"GraphicsList", n_Integer}:>ImportSparse6GraphicsList[{n}],
	{"GraphicsList", {n__Integer}}:>ImportSparse6GraphicsList[{n}],

	{"VertexCount"} :> getS6VertexCount,

	{"VertexList"}:>ImportSparse6VertexList[All],
	{"VertexList", n_Integer}:>ImportSparse6VertexList[{n}],
	{"VertexList", {n__Integer}}:>ImportSparse6VertexList[{n}],

	{s_String, "Elements"}:>(s -> getGraphCountElements[##]&),
	{"Elements"} :> getTopS6Elements,
	Automatic :> getSparse6Graphs,
	getTopS6Elements
  },
  "Sources" -> ImportExport`DefaultSources["Graph"],
  "FunctionChannels" -> {"Streams"},
  "AvailableElements" -> {
	"AdjacencyMatrix", "EdgeRules", "Graph", "GraphList", "Graphics", 
	"GraphicsList", "VertexCount", "VertexList"
  }
]


End[]
