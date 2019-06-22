BeginPackage["Compile`Core`Lint`Utilities`"]

LintFailure

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]


(* ::Subsection:: *)
(* Utilities *)

ClearAll[toString]
toString[s_?ObjectInstanceQ] := s["toString"]
toString[s_] := ToString[s]


getBasicBlocks[fm_] :=
	Module[{bbs = CreateReference[<||>]},
		fm["topologicalOrderScan",
			Function[{bb},
				bbs["associateTo", bb["fullName"] -> bb]
			]
		];
		bbs
	]
	
(* ::Subsection:: *)
(* Error Message *)

toFailure[type_, tag_, inst_, msg_] :=
	LintFailure[<|
		"Type" -> type,
		"Tag" -> tag,
		"Instruction" -> inst,
		"Message" -> msg	
	|>]
printIfNotQuiet[msg_] :=
	If[Lookup[Internal`QuietStatus[], "Global", "Unquiet"] === "Unquiet",
		Print[msg]
	]
error[tag_, inst_, msg___] := 
	printIfNotQuiet[
		toFailure["Error", tag, inst, msg]
	]
	
(* ::Subsection:: *)
(* Formatting *)



getIcon["Error"] := "\[WarningSign]"
getIcon["Warning"] := "\[LightBulb]"
makeIcon[type_] :=
	Framed[
		Style[getIcon[type], Directive["Message", 35]],
		ContentPadding -> False,
		FrameStyle->None,
		FrameMargins -> {{0,0},{0,0}}
	]
LintFailure /: MakeBoxes[f:LintFailure[data_], fmt_] :=
	With[{
		type = data["Type"],
		tag = data["Tag"],
		inst = data["Instruction"],
		msg = data["Message"]
	},
		BoxForm`ArrangeSummaryBox[
	        StringJoin["Lint", type],
	        f,
	        makeIcon[type],
	        {
                BoxForm`SummaryItem[{"Message: ", msg}],
                BoxForm`SummaryItem[{"Tag: ", tag}]
	        },
	        {
	        	If[InstructionQ[inst],
	        		BoxForm`SummaryItem[{"Instruction: ", inst}],
	        		Nothing
	        	]
	        },
	        fmt,
			"Interpretable" -> False
		]
	];
End[]

EndPackage[]
