(* ::Package:: *)

Begin["System`Convert`GraphDump`"]


ImportExport`RegisterImport[
 "Graph6",
 {
	{"AdjacencyMatrix"}:>ImportGraph6[All],
	{"AdjacencyMatrix", n_Integer}:>ImportGraph6[{n}],
	{"AdjacencyMatrix", {n__Integer}}:>ImportGraph6[{n}],

	{"EdgeRules"}:>ImportGraph6EdgeRules[All],
	{"EdgeRules", n_Integer}:>ImportGraph6EdgeRules[{n}],
	{"EdgeRules", {n__Integer}}:>ImportGraph6EdgeRules[{n}],

	{"Graph"}:>ImportGraph6Graph,

	{"GraphList"}:>ImportGraph6GraphList[All],
	{"GraphList", n_Integer}:>ImportGraph6GraphList[{n}],
	{"GraphList", {n__Integer}}:>ImportGraph6GraphList[{n}],

	{"Graphics"}:>ImportGraph6Graphics,

	{"GraphicsList"}:>ImportGraph6GraphicsList[All],
	{"GraphicsList", n_Integer}:>ImportGraph6GraphicsList[{n}],
	{"GraphicsList", {n__Integer}}:>ImportGraph6GraphicsList[{n}],

	{"VertexCount"} :> getG6VertexCount,

	{"VertexList"}:>ImportGraph6VertexList[All],
	{"VertexList", n_Integer}:>ImportGraph6VertexList[{n}],
	{"VertexList", {n__Integer}}:>ImportGraph6VertexList[{n}],
	
	{s_String, "Elements"}:>(s -> getGraphCountElements[##]&),
	{"Elements"} :> getTopGraphElements,
	Automatic :> getGraph6Graphs,
	getTopGraphElements
  },
  "Sources"->ImportExport`DefaultSources["Graph"],
  "FunctionChannels" -> {"Streams"},
  "AvailableElements" -> {
	"AdjacencyMatrix", "EdgeRules", "Graph", "GraphList", "Graphics", 
	"GraphicsList", "VertexCount", "VertexList"
  }
]


End[]
