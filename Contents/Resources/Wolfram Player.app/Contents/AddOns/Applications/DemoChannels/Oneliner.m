(* $Id$ *)

(* :Title: *)

(* :Context: *)

(* :Author: Igor Bakshee *)

(* :Summary: *)

(* :Copyright: Copyright 2016 Wolfram Research, Inc. *)

(* :Package Version: 1.0 ($Revision$/$Date$) *)

(* :Mathematica Version: 11.0 *)

(* :History: *)

(* :Keywords: *)

(* :Sources: *)

(* :Warning: *)

(* :Discussion: *)

System`Private`NewContextPath[{"ChannelFramework`", "System`"}];

(* let the package be loaded by a simpler context, for the time being *)
BeginPackage["DemoChannels`Oneliner`"];
EndPackage[]

BeginPackage["ChannelFramework`Channels`Demos`Oneliner`"];

(* for reloading/debugging *)
Unprotect[Evaluate[Context[]<>"*"]];
ClearAll @@ ({#<>"*",#<>"Private`*"}& @ Context[])

OnelinerSetAlias::usage = "OnelinerSetAlias[\"alias\"]";
OnelinerPreSend::usage = "OnelinerPreSend[cmnd]";
OnelinerAppendToLog::usage = "OnelinerAppendToLog[assoc]";
OnelinerNotebookOpen::usage = "OnelinerNotebookOpen[]";

Begin["`Private`"];

OnelinerPreSend::connect = "Please connect to the Wolfram Cloud to send a message.";

SetAttributes[OnelinerPreSend, HoldFirst]
OnelinerPreSend[e_,opts___] := (
	<|"From" -> $alias, "WL_Compressed" -> Compress[Hold[e]]|>
)/;$CloudConnected

OnelinerPreSend[args__] := (
	Message[OnelinerPreSend::connect];
	$Failed
)

(***
OnelinerPreSend[args__] := ChannelFramework`Private`forceCloudLogin[OnelinerPreSend[args], 
	None
]
*)

OnelinerSetAlias[s_] := $alias = s

$alias := If[$CloudConnected,$WolframID,"Anonymous"];

OnelinerNotebookOpen[] := RunScheduledTask[onelinerNotebook[],{0},
	AutoRemove -> True]

invisibleMarker = Cell["", "Text",
	(* might be a good idea, but not really needed:
		CellFrame -> 0, 
		ShowCellBracket -> False,	
	*)
	Selectable -> True,
	CellElementSpacings -> {"CellMinHeight" -> 0, "ClosedCellHeight" -> 0},
	CellMargins -> -2,
	CellOpen -> False
];

markerCell[] := $mco /; MemberQ[Notebooks[], $tnb]
markerCell[] := (onelinerNotebook[]; markerCell[])

setMarker[] := $mco = Cells[$tnb]//Last

onelinerNotebook[] := $tnb /; MemberQ[Notebooks[], $tnb]

onelinerNotebook[] := (
	$tnb = CreateDocument[{
			Cell["Recent One-Liners", "Title"], 
			invisibleMarker
		},
		WindowSize -> {500, Automatic}, 
		WindowMargins -> {{Automatic, 0}, {Automatic, 0}},
		StyleDefinitions -> FrontEnd`FileName[{"Report"}, "StandardReport.nb"],
		ShowCellTags->True
	];
	setMarker[];
	MathLink`CallFrontEnd[FrontEnd`AttachCell[$tnb, 
		ToBoxes[
			ExpressionCell[
				DynamicModule[{in = Null}, 
					Panel @ Column[{
						Text[Style["Send to channel:", Medium]],
						(* prevent firing when focus leaves the input field *)
						EventHandler[
							InputField[Dynamic[in], String,
								ImageSize -> {Dynamic[CurrentValue["WindowSize"][[1]] - 50], 20}
							],
							{"ReturnKeyDown" :> (
								ChannelSend["Demos:Oneliner", ToExpression[in, InputForm, Unevaluated]];
								in = Null
							)}
						]
					}]
				],
				StripOnInput -> True
			]
		],
		{0, {Center, Bottom}}, {Center, Bottom}
	]];
	$tnb
)

OnelinerAppendToLog[assoc_Association] := (
	onelinerNotebook[];
	onelinerWrite0[ #Message["WL_Compressed"], #Message["From"]]& @ assoc
)/;KeyExistsQ[assoc["Message"],"WL_Compressed"]

onelinerWrite0[HoldComplete[e_],user_] := (
	NotebookWrite[markerCell[], 
		{
			Cell[BoxData[MakeBoxes[InputForm[e]]], "Input", CellTags -> user<>":"],
			invisibleMarker
		}			
	];
	setMarker[];
)

End[];

EndPackage[]

System`Private`RestoreContextPath[];
