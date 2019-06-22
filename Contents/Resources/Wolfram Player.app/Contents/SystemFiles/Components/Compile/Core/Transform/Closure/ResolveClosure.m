BeginPackage["Compile`Core`Transform`Closure`ResolveClosure`"]

ResolveClosurePass

Begin["`Private`"] 

Needs["Compile`Core`IR`ProgramModule`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]

Needs["Compile`Core`Optimization`CopyPropagation`"]
Needs["Compile`Core`Optimization`ConstantPropagation`"]
Needs["Compile`Core`Transform`ResolveTypes`"]

Needs["Compile`Core`Transform`Closure`DeclareClosureEnvironmentType`"]
Needs["Compile`Core`Transform`Closure`PackClosureEnvironment`"]
Needs["Compile`Core`Transform`Closure`UnpackClosureEnvironment`"]
Needs["Compile`Core`Transform`Closure`DesugarClosureEnvironmentStructure`"]
Needs["Compile`Core`Transform`Closure`Utilities`"]
Needs["Compile`Core`Analysis`Properties`ClosureVariablesProvided`"]
Needs["Compile`Core`Transform`Closure`PassClosureArguments`"]

run[fm_, opts_] :=
	Module[{},
		fm["removeProperty", "closureVariablesProvided"];
		fm["removeProperty", "closureVariablesConsumed"];
		fm
	];
	
constraint[arg_?ProgramModuleQ, opts_] := (
	HasClosureQ[arg]
)
constraint[arg_?FunctionModuleQ, opts_] := (
	HasClosureQ[arg["programModule"]]
)

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"ResolveClosure",
	"The pass runs the code to introduce the closure environment type, packing the closure variables, and unpacking the closure environment."
];

ResolveClosurePass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		ClosureVariablesProvidedPass,
		DeclareClosureEnvironmentTypePass,
		PackClosureEnvironmentPass,
		UnpackClosureEnvironmentPass,
		PassClosureArgumentsPass,
		DesugarClosureEnvironmentStructurePass
	},
	"postPasses" -> {
		ResolveTypesPass,
		CopyPropagationPass,
		ConstantPropagationPass
	},
	"constraint" -> constraint
|>];

RegisterPass[ResolveClosurePass]
]]


End[]
	
EndPackage[]
