BeginPackage["Compile`Core`Transform`Closure`DeclareClosureEnvironmentType`"]

DeclareClosureEnvironmentTypePass

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]
Needs["Compile`Core`IR`TypeDeclaration`"]
Needs["Compile`Core`Transform`Closure`Utilities`"]

createEnvironmentType[fm_, opts_] :=
	Module[{tyEnv, ty, pm = fm["programModule"], tyName, capturedVars, typeDecl, tyExpr},
		tyEnv = pm["typeEnvironment"];
		tyName = ClosureEnvironmentTypeName[fm];
		If[tyEnv["resolvableQ", TypeSpecifier[tyName]],
			Return[]
		];
		capturedVars = CapturedVariables[fm];
		tyExpr = MetaData[<|"Fields" -> AssociationThread[(#["name"]& /@ capturedVars) -> Range[0, Length[capturedVars] - 1]] |>]@
			TypeConstructor[tyName, ConstantArray["*", Length[capturedVars]] -> "*", "ShortName" -> "$CEnv"];
		tyEnv["reopen"];
		tyEnv["declareType", tyExpr];
		tyEnv["finalize"];
		typeDecl = CreateTypeDeclaration[TypeSpecifier[tyName]];
		typeDecl["setProperty", "private" -> True];
		pm["typeDeclarations"]["appendTo", typeDecl];
	    fm
	]
	
run[fm_, opts_] :=
	If[CapturesVariablesQ[fm],
		createEnvironmentType[fm, opts],
		fm
	]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"DeclareClosureEnvironmentType",
	"The pass declares the closure environment for each function module that is a closure. " <>
	"Subsequent passes use this type information. It is safe to run this pass multiple times."
];

DeclareClosureEnvironmentTypePass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[DeclareClosureEnvironmentTypePass]
]]

End[]
	
EndPackage[]
