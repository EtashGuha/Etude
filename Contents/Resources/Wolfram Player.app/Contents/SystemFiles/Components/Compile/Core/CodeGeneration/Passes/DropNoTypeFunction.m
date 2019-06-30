
BeginPackage["Compile`Core`CodeGeneration`Passes`DropNoTypeFunction`"]

DropNoTypeFunctionPass

Begin["`Private`"]

Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`ProgramModulePass`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Callback`"]



run[pm_, opts_] :=
	Module[{fms},
		fms = pm["getFunctionModules"];
		pm["setFunctionModules", CreateReference[{}]];
		Scan[ 
			If[ !#["getProperty", "noTypeFunction", False],
				pm["addFunctionModule", #]
			]&,
			fms
		];
		pm
	]



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"DropNoTypeFunction",
		"The pass drops local functions that don't have any type information because we cannot code generate for them."
];

DropNoTypeFunctionPass = CreateProgramModulePass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[DropNoTypeFunctionPass]
]]


End[]

EndPackage[]
