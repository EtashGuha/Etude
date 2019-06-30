Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["MXJSONGraph"]

(* This is used by the "Graph" MX file property. 
This differs from MXSymbolGraph: I don't want to disturb Seb's code or 
ideas there (e.g. adding dummy ops for the output nodes). I want this to scale
to very large graphs, and so it can't afford to be too clever graphically. *)

MXJSONGraph[expr_Association] := Scope[
	
	{nodes, argnodes, heads} = Lookup[expr, {"nodes", "arg_nodes", "heads"}, Panic[]];

	names = nodes[[All, "name"]];
	inputNames = nodes[[All, "inputs", All, 1]] + 1 /. i_Integer :> names[[i]];
	
	edges = Flatten @ MapThread[Thread[DirectedEdge[#2,#1]]&, {names, inputNames}];

	gnodes = Map[
		Property[#name, "op" -> #op, "attrs" -> Lookup[#, "attrs", Lookup[#, "param"]], "inputs" -> #inputs,
			VertexStyle -> If[#op === "null", Gray, opColor[#op]], 
			VertexShapeFunction -> If[#op === "null", "Circle", "Square"]
		]&,
		nodes
	];
	Graph[gnodes, edges, GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Left}, EdgeStyle -> GrayLevel[0.6], 
		VertexStyle -> Directive[EdgeForm[Thin]]
	]
]

(******************************************************************************)

PackageExport["MXSymbolGraph"]

MXSymbolGraph[sym_MXSymbol, OptionsPattern[]] := CatchFailure @ Scope[
	expr = MXSymbolToJSON@sym;
	{nodes, heads} = Lookup[expr, {"nodes", "heads"}, Panic[]];

	(* Add output nodes *)
	outputs = MXSymbolOutputs@sym;
	outputNodes = Map[
		<|"op" -> "null","param" -> <||>,
		"name"-> #, "inputs" -> {},"backward_source_id"->-1
		|>&, outputs
	];
	nodes = Join[nodes, outputNodes];

	(* convert to 1 indexing *)
	heads += 1;
	nodes = MapAt[#[[All, 1]] + 1&, nodes, {All, "inputs"}];
	{ids, names} = Transpose @ MapIndexed[
		Append[#2, #name]&,
		nodes
	];
	(* Get node association *)
	nodesAssociation = Association[#name -> #& /@ nodes];
	(* Get edges *)
	edges = Thread[names[[#inputs]] -> #name]& /@ nodes;
	edges = getPortInfo[#, nodesAssociation]& /@ edges;
	edges = Flatten@edges;
	
	(* Add output edges *)
	outputOperators = names[[heads[[;;, 1]]]];
	outputEdges = Thread[outputOperators -> outputs];
	edges = Join[edges, outputEdges];

	(* Add vertex colours *)
	opTypes = Union[#op& /@ nodes];
	opNums = Range@Length@opTypes/ Length@opTypes;
	opStyles = ColorData["Rainbow"]/@ opNums;
	opStyles = AssociationThread[opTypes -> opStyles];
	opStyles["null"] = Black;
	opShape = If[# === "null", "Square", "Circle"]&;
	
	(* add properties *)
	properties = Property[#name, {
		"op"-> #op, 
		"param" -> #param, 
		VertexStyle -> opStyles@#op,
		VertexShapeFunction -> opShape@#op
	}]& /@ nodes;
	graph = Graph[
		properties, 
		edges, 
		EdgeLabels -> Placed["Name", Tooltip], 
		VertexLabels -> Placed["Name", Tooltip] 
	]
]

getPortInfo[edges_, nodesAssoc_] := Scope[
	If[edges === {}, Return@edges];
	op = nodesAssoc[Last@First@edges, "op"];
	portNames = Lookup[$SymbolArgumentOrdering, op];
	If[Length@portNames < Length@edges, 
		portNames = ConstantArray[Missing[], Length@edges]
	];
	portNames = Thread["PortName" -> portNames];
	portNumber = Thread["Port" -> Range@Length@edges];
	MapThread[Property[#1, #2]&, {edges, Transpose[{portNumber, portNames}]}]
]
