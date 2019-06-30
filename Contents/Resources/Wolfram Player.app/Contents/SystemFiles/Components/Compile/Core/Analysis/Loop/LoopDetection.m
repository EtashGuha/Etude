
BeginPackage["Compile`Core`Analysis`Loop`LoopDetection`"]

LoopDetectionPass

Begin["`Private`"]

Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]



(*
	Scan through all basic blocks and annotates the ones that are interior to a loop with
	the property "isPartOfLoop" -> True.
*)

run[fm_, opts_] := Module[{changed},
	fm["scanBasicBlocks", Function[{bb},
		(* Initialize the "freshChildren" property to be the direct children of each bb.
		   The fresh children are defined to be the unvisited indirect children of a given
		   basic block. The freshChildren are accumulated in "indirectChildren". If the
		   final set of indirect children contains `bb`, then `bb` must be part of a loop.
		*)
		bb["setProperty", "freshChildren" -> bb["getChildren"]];
		bb["setProperty", "indirectChildren" -> bb["getChildren"]];
	]];
	changed = True;
	While[changed,
		changed = False;
		fm["topologicalOrderScan", Function[{bb}, Module[{grandChildren, grandChildrenIds, freshChildrenIds, freshChildren},
			grandChildren = Flatten[Map[#["getChildren"]&, bb["getProperty", "freshChildren"]]];
			grandChildrenIds = #["id"]& /@ grandChildren;
			(* Fresh children are all the indirect children that haven't been visited before *)
			freshChildrenIds = Complement[grandChildrenIds, #["id"]& /@ bb["getProperty", "indirectChildren"]];
			freshChildren = Select[grandChildren, MemberQ[freshChildrenIds, #["id"]]&];

			If[Length[freshChildren] != 0,
				changed = True;
				bb["setProperty", "indirectChildren" -> Join[bb["getProperty", "indirectChildren"], freshChildren]];
				bb["setProperty", "freshChildren" -> freshChildren];
			];
		]]];
	];

	fm["scanBasicBlocks", Function[{bb},
		(* If one of the indrect children of `bb` is `bb` itself, then `bb` must be part of a loop *)
		If[MemberQ[bb["getProperty", "indirectChildren"], bb],
			bb["setProperty", "isPartOfLoop" -> True];
		];
		bb["removeProperty", "indirectChildren"];
		bb["removeProperty", "freshChildren"];
	]];
];



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"LoopDetection",
	"This pass annotates basic blocks with information about their role in forming a loop."
];

LoopDetectionPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"passClass" -> "Analysis"
|>];

RegisterPass[LoopDetectionPass]
]]

End[]

EndPackage[]
