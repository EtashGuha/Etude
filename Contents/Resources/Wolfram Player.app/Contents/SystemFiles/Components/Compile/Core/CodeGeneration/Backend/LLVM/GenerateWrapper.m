
BeginPackage["Compile`Core`CodeGeneration`Backend`LLVM`GenerateWrapper`"]

GenerateWrapperPass

Begin["`Private`"]

Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`ProgramModulePass`"]
Needs["Compile`Core`IR`ProgramModule`"]
Needs["Compile`Core`IR`FunctionModule`"] (* For FunctionModuleQ *)
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["LLVMCompileTools`CreateWrapper`"]
Needs["Compile`"]
Needs["CompileUtilities`Callback`"]



getTypeDeclaration[ pm_, fun_] :=
	Module[ {name, funTy},
		name = fun["name"];
		funTy = UnResolveType[fun["type"]];
		{Primitive`ExternalFunction["LocalLink", name], funTy}
	]


switchFuns[ pm_, pmSrc_] :=
	Module[ {funs},
		funs = pmSrc["getFunctionModules"];
		Scan[ 
			(
			#["setProgramModule", pm];
			pm["addFunctionModule", #];
			)&, funs]
	]

run[pm_, opts_] :=
	If[ TrueQ[Lookup[opts, "CreateWrapper", False]],
		runWork[pm, opts]
		,
		pm
	]


runWork[pm_, opts_] :=
	Module[{
		wrapperFun, pmSub, tyEnv, wrapperName, typeDecs, assoc, topLevelFunList
	},
		topLevelFunList = pm["exportedFunctions"];
		Assert[MatchQ[topLevelFunList, {___?FunctionModuleQ}]];

		If[ Length[topLevelFunList] === 0,
			ThrowException[CompilerException[{"Cannot find any toplevel functions"}]]
		];
		
		assoc = Association[Map[ #["name"] -> <| "type" -> UnResolveType[#["type"]]|> &, topLevelFunList]];
		typeDecs = Map[getTypeDeclaration[pm, #]&, topLevelFunList];
		wrapperFun = CreateWrapperFunction[ assoc];
		wrapperName = "WrapperMain";
		wrapperFun = ProgramExpr["TypeDeclarations" -> typeDecs, "Functions" -> {{wrapperName, wrapperFun}}];
		tyEnv = pm["typeEnvironment"];
		pmSub = With[ {funSub = wrapperFun},
			CompileExprRecurse[funSub, "OptimizationLevel"->1]];
		If[ !ProgramModuleQ[pmSub],
			ThrowException[CompilerException[{"Problem compiling wrapper function", pmSub}]]
		];
		switchFuns[pm, pmSub];
		pm["setProperty", "mainFunctionWrapperName" -> wrapperName];
		pm
	]



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"GenerateWrapper",
		"The pass creates any wrapper functions needed to interface with the generated functions."
];

GenerateWrapperPass = CreateProgramModulePass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[GenerateWrapperPass]
]]


End[]

EndPackage[]
