Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["$EnableNDArrayTooltip"]

$EnableNDArrayTooltip = True;

DeclarePostloadCode @ DefineCustomBoxes[NDArray, 
	nd_NDArray ? NDArrayExistsQ :> MakeNDArrayBoxes[nd],
	"UseUpValues" -> False
]

MakeNDArrayBoxes[nd_] := Block[
	{interior, dims, info, device, extra, type},
	id = MLEID @ nd;
	dims = NDArrayDimensions @ nd;
	device = Replace[{s_, 0} :> s] @ NDArrayContext[nd];
	type = NDArrayDataType @ nd;
	interior = RowBox @ Riffle[dims, StyleBox["\[Times]", Gray]];
	interior = StyleBox[interior, FontFamily -> "Source Code Pro", FontSize -> 11];
	extra = {};
	If[device =!= "CPU", AppendTo[extra, device]];
	If[type =!= "Real32", AppendTo[extra, type]];
	If[extra =!= {}, 
		extra = RowBox[{"(", RowBox[Riffle[extra, ", "]], ")"}];
		interior = RowBox[{interior, "   ", StyleBox[extra, Gray]}];
	];
	interior = PanelBox[interior, ContentPadding -> False, BaselinePosition -> Baseline];
	If[$EnableNDArrayTooltip && (Times @@ dims) < 2^20,
		interior = TooltipBox[
			interior, 
			DynamicBox[ToBoxes @ NDArraySummaryGrid @ nd, TrackedSymbols :> {}],
			TooltipDelay -> 0.2
		];		
	];
	InterpretationBox @@ {
		RowBox[{"NDArray", "[", id, ":", interior, "]"}], nd,
		Selectable -> False, Editable -> False, SelectWithContents -> True
	}
]