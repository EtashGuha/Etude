BeginPackage["GraphStore`Entity`RDFEntityStoreFormatting`", {"GraphStore`", "GraphStore`Entity`"}];
Begin["`Private`"];

RDFEntityStore /: MakeBoxes[store_RDFEntityStore, fmt_] := With[{res = Catch[iRDFEntityStoreMakeBoxes[store, fmt], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iRDFEntityStoreMakeBoxes];
iRDFEntityStoreMakeBoxes[store : RDFEntityStore[_, backend_, ___], fmt_] := BoxForm`ArrangeSummaryBox[
	RDFEntityStore,
	store,
	None,
	{
		BoxForm`MakeSummaryItem[
			{
				"Type count: ",
				Length[store[]]
			},
			fmt
		],
		BoxForm`MakeSummaryItem[
			{
				"RDFStore: ",
				OpenerView[{
					If[MatchQ[backend, _RDFStore], "in-memory", "remote"],
					backend
				}]
			},
			fmt
		]
	},
	{
		BoxForm`MakeSummaryItem[
			{
				"Types: ",
				store[]
			},
			fmt
		]
	},
	fmt
];

End[];
EndPackage[];
