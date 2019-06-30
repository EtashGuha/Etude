
BeginPackage["Compile`Core`Analysis`Properties`HasClosure`"]

HasClosurePass

Begin["`Private`"]

Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`Transform`Closure`Utilities`"]
Needs["Compile`Core`PassManager`PassRegistry`"]

run[fm_, opts_] := 
	If[CapturesVariablesQ[fm],
		With[{
			pm = fm["programModule"]
		},
			pm["setProperty", "hasClosure" -> True]
		];
		fm
	]




RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"HasClosure",
	"This pass stores a property in the program module to tell later passes whether this program module contains a closure."
];

HasClosurePass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"passClass" -> "Analysis"
|>];

RegisterPass[HasClosurePass];

]]

End[]

EndPackage[]
