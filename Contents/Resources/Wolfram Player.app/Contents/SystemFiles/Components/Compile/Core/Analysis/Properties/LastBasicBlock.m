
BeginPackage["Compile`Core`Analysis`Properties`LastBasicBlock`"]

LastBasicBlockPass

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]



(*
  Pass that updates the last BasicBlock of each Function Module.
  
  Scan through all BasicBlocks and pick the one with no children.
  If there is less or more than one that is an error.
*)

run[fm_, opts_] :=
	Module[{bb, ref},
		ref = CreateReference[ {}];
		fm["scanBasicBlocks", 
			If[#["children"]["length"] === 0, ref["appendTo", #]]&];
		If[ ref["length"] =!= 1,
			ThrowException[{"LastBasicBlock Update error, found other than 1 BasicBlock with no children", ref["get"]}]
		];
		bb = ref["getPart", 1];
		fm["setLastBasicBlock", bb];
	]

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"LastBasicBlock",
		"This pass recomputes the last BasicBlock for each Function Module."
];

LastBasicBlockPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"passClass" -> "Analysis"
|>];

RegisterPass[LastBasicBlockPass]
]]


End[]

EndPackage[]
