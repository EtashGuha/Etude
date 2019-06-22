Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

DeclarePostloadCode @ DefineCustomBoxes[MXSymbol,
	sym_MXSymbol ? System`Private`NoEntryQ :> MXSymbolBoxes[sym],
	"UseUpValues" -> False
]

MXSymbolBoxes[sym:MXSymbol[id_]] := Scope[
	plot = Dynamic[MXSymbolPlot[sym], TrackedSymbols :> {}];
	outs = MXSymbolOutputs[sym];
	args = MXSymbolArguments[sym];
	BoxForm`ArrangeSummaryBox[
		MXSymbol,
		sym,
		None,
		{makeSymItem["ID", id], 
		 makeSymItem["Outputs", Column @ outs]},
		{makeSymItem["Arguments", If[Length[args] < 16, Row[args, ","], Skeleton[Length[args]]]],
		 makeSymItem["Plot", plot]},
		StandardForm
	]
]

makeSymItem[name_, value_] := BoxForm`MakeSummaryItem[{Pane[name <> ": ", {60, Automatic}], value}, StandardForm]
