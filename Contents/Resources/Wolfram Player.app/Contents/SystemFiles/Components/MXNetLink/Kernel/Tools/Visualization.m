Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["MXJSONPlot"]

Options[MXJSONPlot] = {
	"ShowTensors" -> True,
	"Rotated" -> False,
	"VertexLabels" -> Placed["ID", Above],
	"VertexOrder" -> Automatic,
	"EdgeBundling" -> True,
	"OutputTensors" -> None,
	"InternalDimensions" -> None
};

MXJSONPlot[json_String, opts:OptionsPattern[]] := 
	MXJSONPlot[
		Developer`ReadRawJSONString[json],
		opts
	];

PackageExport["$MaxGraphPlotNodes"]
$MaxGraphPlotNodes = 500;

MXJSONPlot[expr_Association, OptionsPattern[]] := CatchFailure @ Scope[
	UnpackOptions[showTensors, rotated, vertexLabels, vertexOrder, edgeBundling, outputTensors, internalDimensions];
	{nodes, argnodes, heads} = Lookup[expr, {"nodes", "arg_nodes", "heads"}, Panic[]];
	lenNodes = Length[nodes];
	If[lenNodes === 0, Return["empty graph"]];
	If[lenNodes > $MaxGraphPlotNodes, Return @ StringForm["graph size `` exceeded $MaxGraphPlotNodes", lenNodes]];
	$oldids = If[ListQ[vertexOrder],
		IndexOf[vertexOrder, #name]-1& /@ nodes,
		Range[Length[nodes]]-1
	];
	nodes = Map[Append[#, "inputs1" -> #inputs[[All, 1]]+1]&, nodes];
	nameStrings = toVertexLabelString[#name]& /@ nodes;
	typeStrings = toVertexTypeString[#op]& /@ nodes;
	argnodes += 1; heads += 1;
	edges = Join @@ MapIndexed[
		If[#op === "null", Nothing,
			If[showTensors, 
				Thread[Prepend[#2, #inputs1]],
				Thread[Prepend[#2, Complement[#inputs1, argnodes]]]
		]]&
	,
		nodes
	];
	edges = DeleteDuplicates[edges];
	nodeOps = nodes[[All, "op"]];
	If[edgeBundling && !FreeQ[nodeOps, "Concat"|"SliceChannel"],
		longRange = pickLongRangeEdges[edges, nodes],
		longRange = None;
	];

	{opTypes, opNames} = Labelling @ nodeOps;
	(* Add head tensors *)
	nullType = IndexOf[opNames, "null"];
	nodes = MapIndexed[Append[#, "id" -> First[#2]-1]&, nodes];
	If[showTensors && ListQ[outputTensors],
		opTypes = Join[opTypes, ConstantArray[nullType, Length@outputTensors]];
		argnodes = Join[argnodes, Range@Length@outputTensors + Max@edges];
		nameStrings = Join[nameStrings, outputTensors];
		blank = ConstantArray["", Length[outputTensors]];
		$oldids = Join[$oldids, blank]; typeStrings = Join[typeStrings, blank]; 
		nodes = Join[nodes, blank];
		maxIndex = Max@edges;
		MapIndexed[AppendTo[edges, {First@#1, (First[#2] + maxIndex)}]&, heads]
	];
	edgeTooltips = If[!AssociationQ[internalDimensions], None,
		nodeDims = Table[
			name = Internal`UnsafeQuietCheck[nodes[[i, "name"]], None];
			toDimLabel @ Lookup[internalDimensions, name, 
				If[StringQ[name], Lookup[internalDimensions, name <> "_output", None], None]]
			, {i, Length[nodes]}
		];
		(nodeDims[[#1]]& @@@ edges) /. BatchSize -> "b"
	];
	
	(* Do plot *)
	nops = Length[opNames];
	opStyles = opColor /@ opNames;
	opSizes = ReplacePart[ConstantArray[6, nops], nullType -> 4];
	opStyles = ReplacePart[opStyles, nullType -> Gray];
	opNames = opNames /. "null" -> "Tensor";

	vertexTypeData = <|"VertexStyles" -> opStyles|>;
	If[showTensors, vertexTypeData = Join[vertexTypeData, <|"VertexSizes" -> opSizes|>]];
	labels = vertexLabels /. {"Name" :> nameStrings, "ID" :> $oldids, "Type" :> typeStrings};
	infoGrids = Map[nodeInfoGrid, nodes];
	nnodes = Length[nodes];
	plot = LayerPlot[edges, 
		"VertexLabels" -> labels, 
		"VertexTooltips" -> infoGrids, 
		"HiddenVertices" -> If[showTensors, None, argnodes],
		"VertexTypeData" -> vertexTypeData,
		"VertexTypeLabels" -> opNames,
		"MaximumImageSize" -> None,
		"VertexSizes" -> 4,
		"EdgeTooltips" -> edgeTooltips,
		"ImageScale" -> Which[nnodes > 100, 15, nnodes > 50, 25, nnodes > 30, 30, nnodes > 20, 35, nnodes > 10, 40, nnodes > 5, 50, True, 80],
		"BaseLabelStyle" -> {FontSize -> 7},
		"DuplicateInputVertices" -> True,
		If[showTensors, "VertexTypes" -> opTypes, "VertexStyles" -> opTypes],
		"LongRangeEdges" -> longRange,
		"Rotated" -> True,
		"ArrowShape" -> "Chevron",
		"LegendLabelStyle" -> 8
	];

	If[StringQ[internalDimensions], (* if there was an error *)
		plot = Column[{plot, "",
			Style["Shape inference error:", Red, Bold],
			Style[InsertLinebreaks[internalDimensions, 120], Red, FontSize -> 8]
		}, Alignment -> Left]];

	plot
]

toDimLabel[None] := "";
toDimLabel[list_List] := Row[list, "\[Times]"];

PackageScope["opColor"]

opColor["SliceChannel"] := Darker[Green, .15];
opColor["Concat"] := Darker[Red, .1];
opColor["SwapAxis"] := Darker[Cyan, .2];
opColor["Dropout"] := Orange;
opColor["_mul" | "_plus" | "elemwise_add" | "ElementWiseSum"] := Gray;
opColor["FullyConnected"] := Black;
opColor["Reshape"] := Darker[Purple, .2];
opColor[str_] := opColor[str] = Block[{hash = Hash[str]}, ColorData[89 + Mod[hash, 5]] @ Round[hash / 5]];

pickLongRangeEdges[edges_, nodes_] := Scope[
	$nodes = nodes; $edges = edges;
	List[
		pickLRFrom["Concat", 2, Most],
		pickLRFrom["SliceChannel", 1, Rest]
	]
]

pickLRFrom[type_, n_, merge_] := 
	Catenate @ GroupBy[
		SelectIndices[$edges, MatchQ[$nodes[[#[[n]], "op"]], type]&], 
		$edges[[#, n]]&, merge
	];

symfm[e_] := Style[e, FontFamily -> "Courier", 9];

Clear[toVertexTypeString];
toVertexTypeString["null"] = "";
toVertexTypeString["_mul"] = symfm["\[Times]"]; 
toVertexTypeString["_plus" | "elemwise_add" | "ElementWiseSum"] := symfm["+"];
toVertexTypeString[s_String] := toVertexTypeString[s] = Scope[
	l = Select[Characters[s], LetterQ];
	StringJoin @ If[LowerCaseQ[First[l]], Prepend[First[l]], Identity] @ Select[l, UpperCaseQ]
];

toVertexLabelString[s_String] := Last[StringSplit[s, "."], Null];

nodeInfoGrid[""] := Null;

nodeInfoGrid[assoc_] := Grid[
	Append[
	{Style[#, Bold], fmtInfoGridItem[assoc[#]]}& /@ {"id", "name", "op", "param", "attrs"},
	{Style["inputs", Bold], fmtInfoGridItem[remapOldIDs[assoc["inputs"]]]}
	],
	Alignment -> Left, BaseStyle -> {FontFamily -> "Source Code Pro"},
	Dividers -> All, FrameStyle -> LightGray,
	Alignment -> {Left, Baseline},
	Spacings -> {1.1, 0.5}, 
	ItemSize -> {Automatic, 1.4}
];

remapOldIDs[ins_] := MapAt[$oldids[[#+1]]&, ins, {All, 1}];
remapOldIDs[{}] := {};

Clear[fmtInfoGridItem];
fmtInfoGridItem[e_] := e;
fmtInfoGridItem[<||>] := <||>;
fmtInfoGridItem[_Missing] := <||>;
fmtInfoGridItem[assoc_Association] := Grid[
	KeyValueMap[
		{Style[#1, Bold], Style[#2, ShowStringCharacters -> True]}&, 
		assoc
	],
	Alignment -> Left, Spacings -> {1.1, 0.5}, 
	Dividers -> Center, FrameStyle -> LightGray
];

(******************************************************************************)

Options[MXSymbolPlot] = Most @ Options[MXJSONPlot];

PackageExport["MXSymbolPlot"]

MXSymbolPlot[sym_MXSymbol, opts:OptionsPattern[]] := Scope[
	expr = MXSymbolToJSON[sym];
	If[!AssociationQ[expr], Return[$Failed]];
	MXJSONPlot[expr, opts, "OutputTensors" -> MXSymbolOutputs[sym]]
];

