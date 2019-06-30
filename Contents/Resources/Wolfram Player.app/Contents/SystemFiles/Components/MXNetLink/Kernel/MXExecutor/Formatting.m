Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

DeclarePostloadCode @ DefineCustomBoxes[MXExecutorData,
	exec_MXExecutorData ? System`Private`NoEntryQ :> MXExecutorDataBoxes[exec],
	"UseUpValues" -> False
];

MXExecutorDataBoxes[exec:MXExecutorData[data_]] := Scope[
	plot = Dynamic[MXSymbolPlot[data@"Symbol"], TrackedSymbols :> {}];
	context = Replace[{s_, 0} :> s] @ FromContextCode @ data["Context"];
	execID = ManagedLibraryExpressionID@data["Executor"];
	args = MXSymbolArguments[sym];
	arrayInfo = Map[
		makeArrayItems[data[#], #]&, 
		{"OutputArrays", "GradientArrays", "ArgumentArrays", "AuxilliaryArrays"}
	];
	BoxForm`ArrangeSummaryBox[
		MXExecutorData,
		exec,
		None,
		Join[
			arrayInfo,
			{makeExecItem["Context", context]}	
		],
		{makeExecItem["ExecutorID", execID], Framed[plot, FrameStyle -> LightGray]},
		StandardForm
	]
];

makeArrayItems[_Missing, _] := Nothing;
makeArrayItems[<||>, _] := Nothing;
makeArrayItems[assoc_, key_] := Block[{count = Count[assoc, Except[None]]}, If[count === 0, Nothing, makeExecItem[key, count]]];
makeArrayItems[assoc_, key:"InputArrays"|"OutputArrays"|"SpecialArrays"] := 
	makeExecItem[key, Style[KeyValueMap[arrayItem, assoc], "Code", FontWeight -> "Plain"]];

arrayItem[name_, array_] := Tooltip[name, NDArrayDimensions[array]];

makeExecItem[name_, value_] := BoxForm`MakeSummaryItem[{Pane[name <> ": ", {90, Automatic}], value}, StandardForm];
