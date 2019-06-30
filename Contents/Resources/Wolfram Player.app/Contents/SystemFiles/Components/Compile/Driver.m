BeginPackage["Compile`Driver`"]

Begin["`Private`"]
(* Implementation of the package *)

Needs["Compile`Utilities`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["Compile`"]
Needs["Compile`Core`CodeGeneration`Backend`LLVM`"]
Needs["Compile`Utilities`Serialization`"]
Needs["Compile`Values`CompileValues`"]
Needs["LLVMCompileTools`CreateWrapper`"]
Needs["LLVMCompileTools`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for CatchException *)
Needs["CompileAST`Create`Construct`"]
Needs["Compile`Core`IR`Lower`CreateIR`"]
Needs["Compile`AST`Transform`ElaborateFunctionSlots`"]
Needs["Compile`AST`Transform`MExprConstant`"]
Needs["Compile`AST`Transform`InitializeMExpr`"]
Needs["Compile`AST`Macro`Expand`"]
Needs["Compile`AST`Semantics`Binding`"]
Needs["Compile`AST`Transform`NativeUnchecked`"]
Needs["Compile`AST`Transform`InitializeType`"]
Needs["CompileAST`Class`Base`"]
Needs["Compile`Core`PassManager`PassRunner`"]
Needs["Compile`Core`IR`CompiledProgram`"]
Needs["Compile`Core`IR`ProgramModule`"]
Needs["LLVMTools`"]
Needs["LLVMCompileTools`Finalize`"]
Needs["Compile`AST`Semantics`ClosureVariablesConsumed`"]
Needs["Compile`AST`Semantics`ClosureVariables`"]
Needs["Compile`AST`Semantics`EscapingScopedVariables`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["LLVMLink`LLVMInformation`"]
Needs["CompiledLibrary`"]
Needs["Compile`API`FunctionCompile`"]

Compile`Utilities`$CreateWrapper = "Legacy"
Compile`Utilities`$ExpressionInterface = Automatic
Compile`Utilities`$AddRuntime = "SharedLibrary"


Compile`Utilities`$LazyJIT =
	Which[
		$SystemWordLength === 32,
			False
		,
		$LLVMInformation["LLVM_VERSION"] >= 7.0,
			False,
		True,
			True]


Compile`Utilities`$LLVMOptimization = Automatic
Compile`Utilities`$ExceptionsModel = Automatic
Compile`Utilities`$AbortHandling = Automatic

$sharedOptions = {
	"TypeSystem" -> Automatic,
	"TargetArchitecture" -> Automatic,
	"OptimizationLevel" -> 2,
	"SaveIR" -> False,
	"TypeEnvironment" :> Automatic,
	"PassLogger" -> Automatic,
	"MacroRecursionLimit" -> 5,
	"ResetVariableID" -> True,
	"MacroEnvironment" :> $DefaultMacroEnvironment,
	"ExternalLibraries" -> {},
	"TargetSystemID" -> Automatic, 
	"TargetTriple" -> Automatic, 
	"DataLayout" -> Automatic, 
	"ExceptionsModel" :> Compile`Utilities`$ExceptionsModel,
    "AbortHandling" :> Compile`Utilities`$AbortHandling,
	"PassOptions" -> {},
	"EntryFunctionName" -> Automatic
};

Options[CompileExpr] = Join[$sharedOptions, {
	"Debug" -> False
}]

Options[CompileToIR] = Join[$sharedOptions, {
	"Debug" -> False
}]

Options[CompileToCodeFunction] = Join[$sharedOptions, {
	"Debug" -> False,
	"CacheExpr" -> False,
	"AddRuntime" -> Compile`Utilities`$AddRuntime,
	"LazyJIT" -> Compile`Utilities`$LazyJIT,
	"LLVMOptimization" -> Compile`Utilities`$LLVMOptimization,
	"ExpressionInterface" -> Compile`Utilities`$ExpressionInterface,
	"CreateWrapper" -> Compile`Utilities`$CreateWrapper,
	"ErrorFunction" -> Null,
	"ExecutionEngine" -> Automatic
}]

Options[CompileToExternalFunction] = Join[$sharedOptions, {
	"Debug" -> False,
	"CacheExpr" -> False,
	"AddRuntime" -> Compile`Utilities`$AddRuntime,
	"LazyJIT" -> Compile`Utilities`$LazyJIT,
	"LLVMOptimization" -> Compile`Utilities`$LLVMOptimization,
	"ExpressionInterface" -> Compile`Utilities`$ExpressionInterface,
	"CreateWrapper" -> Compile`Utilities`$CreateWrapper,
	"ErrorFunction" -> Null
}]

Options[CompileToLLVMIR] = Join[$sharedOptions, {
	"Debug" -> False,
	"CreateWrapper" -> False,
	"ExpressionInterface" -> Compile`Utilities`$ExpressionInterface,
	"AddRuntime" -> Compile`Utilities`$AddRuntime,
	"LazyJIT" -> Compile`Utilities`$LazyJIT,
	"LLVMOptimization" -> Compile`Utilities`$LLVMOptimization
}]

Options[CompileToLLVMString] = Join[$sharedOptions, {
	"Debug" -> False,
	"CreateWrapper" -> False,
	"ExpressionInterface" -> Compile`Utilities`$ExpressionInterface,
	"AddRuntime" -> Compile`Utilities`$AddRuntime,
	"LazyJIT" -> Compile`Utilities`$LazyJIT,
	"LLVMOptimization" -> Compile`Utilities`$LLVMOptimization
}]

Options[CompileToLLVMModule] = Join[$sharedOptions, {
	"Debug" -> False,
	"CreateWrapper" -> False,
	"ExpressionInterface" -> Compile`Utilities`$ExpressionInterface,
	"AddRuntime" -> Compile`Utilities`$AddRuntime,
	"LazyJIT" -> Compile`Utilities`$LazyJIT,
	"LLVMOptimization" -> Compile`Utilities`$LLVMOptimization
}]

Options[CompileToWIRSerialization] = Join[$sharedOptions, {
	"Debug" -> False,
	"CreateWrapper" -> False,
	"ExpressionInterface" -> Compile`Utilities`$ExpressionInterface,
	"AddRuntime" -> Compile`Utilities`$AddRuntime,
	"LazyJIT" -> Compile`Utilities`$LazyJIT,
	"LLVMOptimization" -> Compile`Utilities`$LLVMOptimization
}]



startProcess[ arg_, opts_] :=
	Module[ {pLogger},
		pLogger = Lookup[opts, "PassLogger"];
		If[ObjectInstanceQ[pLogger],
			pLogger["startProcess", arg]];
	]


endProcess[ arg_, opts_] :=
	Module[ {pLogger},
		pLogger = Lookup[opts, "PassLogger"];
		If[ObjectInstanceQ[pLogger],
			pLogger["endProcess", arg]];
	]

(*
  Run the LLVM Passes if the option has been set.  If it has
  and if the module is not known to be valid then run the 
  verifier. Don't run the passes if the module is not valid.
*)

getPassHandler[_List] :=
	LLVMRunPasses
	
getPassHandler["ClangOptimization"[_Integer]] :=
	LLVMRunOptimizationPasses
	
getPassHandler[_] :=
	None


	
runLLVMOptimizationOption[mod_LLVMModule, optionValue_, validQ_] :=
	Module[{handler = getPassHandler[optionValue], verifyRes},
		If[ handler === None,
			Return[False]];
		If[ !validQ,
			verifyRes = LLVMVerifyModule[mod];
			If[ !TrueQ[verifyRes["valid"]],
				Message[Compile::valid];
				Return[False]]];
		handler[mod, "OptimizationLevel" -> optionValue]
	]

getOptions[ head_, expr_, optsIn:OptionsPattern[]] :=
	Module[ {typeSys, targetArch,
			macroRecursionLimit, opts = Flatten[{optsIn}], optLevel, debug, optsRet, passLogger,
			resetVarID, extLibs, targetSystemID, tyEnv, exceptModel, abortModel, passOptions, entryFunctionName},
		opts = FilterRules[ opts, Options[head]];
		entryFunctionName = OptionValue[head, opts, "EntryFunctionName"];
		passOptions = OptionValue[head, opts, "PassOptions"];
		passLogger = OptionValue[head, opts, "PassLogger"];
		optLevel = OptionValue[head, opts, "OptimizationLevel"];
		typeSys = OptionValue[head, opts, "TypeSystem"];
		targetArch = OptionValue[head, opts, "TargetArchitecture"];
		macroRecursionLimit = OptionValue[head, opts, "MacroRecursionLimit"];
		debug = TrueQ[OptionValue[head, opts, "Debug"]];
		resetVarID = OptionValue[head, opts, "ResetVariableID"];
		extLibs = OptionValue[head, opts, "ExternalLibraries"];
		targetSystemID = OptionValue[head, opts, "TargetSystemID"];
		If[ targetSystemID === Automatic,
			targetSystemID = $SystemID];
		exceptModel = OptionValue[head, opts, "ExceptionsModel"];
        abortModel = OptionValue[head, opts, "AbortHandling"];
		optLevel = Which[
			MemberQ[{None, 0, 1, 2, 3}, optLevel],
				optLevel,
			ListQ[optLevel],
				optLevel,
			True,
				2
		];
		optsRet = <|
			"TypeSystem" -> typeSys,
			"OptimizationLevel" -> optLevel,
			"Debug" -> debug,
			"LLVMDebug" -> debug,
			"TargetArchitecture" -> targetArch,
			"MacroRecursionLimit" -> macroRecursionLimit,
			"PassLogger" -> passLogger,
			"ResetVariableID" -> resetVarID,
			"ExternalLibraries" -> extLibs,
			"TargetSystemID" -> targetSystemID,
			"ExceptionsModel" -> exceptModel,
            "AbortHandling" -> abortModel,
			"PassOptions" -> passOptions,
			"EntryFunctionName" -> entryFunctionName
		|>;
		tyEnv = OptionValue[head, opts, "TypeEnvironment"];
		optsRet["TypeEnvironment"] = getTypeEnvironment[targetSystemID, tyEnv, expr];
		optsRet["MacroEnvironment"] = OptionValue[head, opts, "MacroEnvironment"];
		optsRet
	]



(*
 TypeEnvironment Functions
*)

(*
  Used to determine if a Type Environment should be created or not.
*)
AutoTypeEnvironmentQ[expr_] :=
	CompiledProgramQ[expr] || Head[expr] === Program

(*
 No TargetSystemID and no type environment, use the default.
*)
getTypeEnvironment[ None, Automatic, expr_] :=
	SelectTypeEnvironment[$SystemID, AutoTypeEnvironmentQ[expr]]

(*
 No type environment, but targetSystemID given,  use that.
*)
getTypeEnvironment[ targetSystemID_, Automatic, expr_] :=
	Module[ {},
		SelectTypeEnvironment[targetSystemID, AutoTypeEnvironmentQ[expr]]
	]

(*
 No targetSystemID given, but there is a type environment, 
 use that.
*)
getTypeEnvironment[ None, tyEnv_, expr_] :=
	Module[ {},
		tyEnv
	]

(*
 Both type environment and targetSystemID given check they are compatible.
*)
getTypeEnvironment[ targetSystemID_, tyEnv_, expr_] :=
	Module[ {targ = tyEnv["getProperty", "TargetSystemID", Null]},
		If[ targ === Null,
			ThrowException[{"The type environment does not have a TargetSystemID set."}]];
		If[ targ =!= targetSystemID,
			ThrowException[{"The TargetSystemID does not match that from the TypeEnvironment.", targetSystemID, targ}]];
		tyEnv
	]

mexprPasses := mexprPasses = {
	InitializeMExprPass,
	
	ElaborateFunctionSlotsPass,
	MExprBindingRewritePass,
	MacroExpandPass,
	MExprConstantPass,
	NativeUncheckedPass,
	(* We rerun all the passes again
	 * for safe measure (some might
	 * not really be needed, but we
	 * certainly need to run the
	 * MExprClosureConversionPass
	 *)
	ElaborateFunctionSlotsPass,
	MExprBindingRewritePass,
	MExprClosureVariablesPass,
	MExprClosureVariablesConsumedPass,
	MExprEscapingScopedVariablesPass,
	MExprInitializeTypePass,
	FinalizeMExprPass
}

CompileToAST[ expr_, opts_:<||>] :=
	Module[ {mexpr},
		InitializeCompiler[];
		mexpr = If[ MExprQ[expr], expr, CreateMExpr[expr]];
		mexpr = RunPasses[
			mexprPasses,
			mexpr,
			opts
		];
		mexpr
	]



iCompileToIR[ CompileValues[ sym_Symbol], opts_] :=
	Module[ {progExpr},
		progExpr = ToProgramExpr[ sym];
		If[FailureQ[progExpr],
			progExpr,
			With[ {p1 = progExpr},
				iCompileToIR[p1, opts]
			]
		]
	]


iCompileToIR[ inp_, opts_] :=
	Module[ {cp, pm, libs},
		cp = If[ CompiledProgramQ[inp], inp, CreateCompiledProgram[inp]];
		cp = RunPasses[
			mexprPasses,
			cp,
			opts
		];
		pm = RunPass[
			CreateIRPass,
			cp,
			Join[
			   opts,
			   <|
					"OptimizationLevel" -> opts["OptimizationLevel"],
					"EnvironmentOptions" -> opts,
					"Debug" -> opts["Debug"],
					"LLVMDebug" -> opts["Debug"]
			   |>
			]
		];
		libs = Lookup[ opts, "ExternalLibraries", {}];
		If[ MatchQ[libs, {__String}],
			pm["setProperty", "externalLibraries" -> libs]];
		pm
	]


CompileExpr[ expr_, opts:OptionsPattern[]] :=
	CompileToIR[ expr, opts]

CompileToIR[ expr_, opts:OptionsPattern[]] :=
	CatchException[
		Module[ {settings, pm},
			InitializeCompiler[];
			settings = getOptions[CompileToIR, expr, opts];
			startProcess[expr, settings];
			pm = iCompileToIR[expr, settings];
			endProcess[expr, settings];
			pm
		]
		,
		{{_, CreateFailure}}
	]


CompileExprRecurse[ expr_, opts:OptionsPattern[]] :=
	CompileToIR[ expr, "ResetVariableID" -> False, opts]

getCodeGenOptions[ head_, optsIn:OptionsPattern[]] :=
	Module[ {opts = Flatten[{optsIn}], passLogger, addRuntime, createWrapper, 
				expressionInterface, trgtArch, lazyJIT, optLevel, llvmOpt, debug, 
				exceptionModel, abortModel, trgtSystemID, trgtTriple, dataLayout, passOptions, 
				loopHint, machArch},
		opts = FilterRules[ opts, Options[head]];
		passLogger = OptionValue[head, opts, "PassLogger"];
		addRuntime = OptionValue[head, opts, "AddRuntime"];
		If[addRuntime == "Link" && $OperatingSystem == "Windows",
			ThrowException["Option value AddRuntime->Link is disallowed on Windows"]
		];
		passOptions = OptionValue[head, opts, "PassOptions"];
		lazyJIT = OptionValue[head, opts, "LazyJIT"];
		optLevel = OptionValue[head, opts, "OptimizationLevel"];
		llvmOpt = OptionValue[head, opts, "LLVMOptimization"];
		createWrapper = OptionValue[head, opts, "CreateWrapper"];
		expressionInterface = OptionValue[head, opts, "ExpressionInterface"];
		trgtArch = OptionValue[head, opts, "TargetArchitecture"];
		trgtSystemID = OptionValue[head, opts, "TargetSystemID"];
		trgtTriple = OptionValue[head, opts, "TargetTriple"];
		dataLayout = OptionValue[head, opts, "DataLayout"];
		debug = OptionValue[head, opts, "Debug"];
		exceptionModel = OptionValue[head, opts, "ExceptionsModel"];
        abortModel = OptionValue[head, opts, "AbortHandling"];
        If[ createWrapper =!= "Legacy" && abortModel === Automatic,
        	abortModel = False];
        {llvmOpt, loopHint, machArch} = processLLVMOptimization[llvmOpt];
 		<|
			"PassLogger" -> passLogger,
			"AddRuntime" -> addRuntime,
			"LazyJIT" -> lazyJIT,
			"LLVMOptimization" -> llvmOpt,
			"LoopHints" -> loopHint,
			"MachineArchitecture" -> machArch,
			"CreateWrapper" -> createWrapper,
			"ExpressionInterface" -> expressionInterface,
			"TargetArchitecture" -> trgtArch,
			"TargetSystemID" -> trgtSystemID,
			"TargetTriple" -> trgtTriple,
			"DataLayout" -> dataLayout,
			"Debug" -> debug,
			"LLVMDebug" -> debug,
			"ExceptionsModel" -> exceptionModel,
            "AbortHandling" -> abortModel,
			"PassOptions" -> passOptions
		|>
	]


getClangOptimization[ Automatic] :=
	"ClangOptimization"[1]

getClangOptimization[ val_] :=
	val

processLLVMOptimization[ Automatic] :=
	{getClangOptimization[Automatic], Automatic, Automatic}
	
processLLVMOptimization[ "ClangOptimization"[lev_]] :=
	{"ClangOptimization"[lev], Automatic, Automatic}

processLLVMOptimization[ {clangOpt_:Automatic, opts:OptionsPattern[]}] :=
	Module[{loopHints, machArch},
		loopHints = Lookup[{opts}, "LoopHints", Automatic];
		machArch = Lookup[{opts}, "MachineArchitecture", Automatic];
		{getClangOptimization[clangOpt], loopHints, machArch}
	]
	
processLLVMOptimization[ {clangOpt_:Automatic, opts_?AssociationQ}] :=
	Module[{loopHints,machArch},
		loopHints = Lookup[opts, "LoopHints", Automatic];
		machArch = Lookup[opts, "MachineArchitecture", Automatic];
		{getClangOptimization[clangOpt], loopHints, machArch}
	]
	
processLLVMOptimization[ _] :=
	{None, None, None}



(*
  CompileToLLVMIR
*)

CompileToLLVMIR[ expr_, opts:OptionsPattern[]] :=
	Module[{mod},
		mod = CompileToLLVMModule[expr, opts];(* temporary variable is used to simplify reasoning about Finalization scoping *)
		Replace[mod, {
			LLVMModule[id_Integer] :> LLVMToString[LLVMModule[id]],
			other_ :> other
		}]
	]

(*
  CompileToLLVMString
*)

CompileToLLVMString[ expr_, opts:OptionsPattern[]] :=
	Module[{mod},
		mod = CompileToLLVMModule[expr, opts];(* temporary variable is used to simplify reasoning about Finalization scoping *)
		Replace[mod, {
			LLVMModule[id_Integer] :> LLVMToString[LLVMModule[id]],
			other_ :> other
		}]
	]


(*
  CompileToLLVMModule
*)


CompileToLLVMModule[ arg_WIRSerialization, opts:OptionsPattern[]] :=
	CatchException[
		Module[ {pm, data, settings, mod, tyEnv, meta},
			InitializeCompiler["InstallLLVM" -> True];
			settings = getCodeGenOptions[ CompileToLLVMModule, opts];
			startProcess[arg, settings];
			meta = GetSerializationData[arg, "metaInformation"];
			tyEnv = getTypeEnvironment[ Lookup[meta, "TargetSystemID", None], OptionValue["TypeEnvironment"], arg];
			pm = WIRDeserialize[ tyEnv, arg];
			RunPass[CreateLLVMIROnlyPass, pm, settings];
			data = pm["getProperty", "LLVMLinkData"];
			mod = LLVMModule[data["moduleId"]];
			runLLVMOptimizationOption[mod, settings["LLVMOptimization"], False];
			endProcess[arg, settings];

			DisposeBuilders[data];
			
			EnableLLVMModuleFinalization[mod];

			mod
		]
		,
		{{_, CreateFailure}}
	]

CompileToLLVMModule[ pm_?ProgramModuleQ, opts:OptionsPattern[]] :=
	CatchException[
		Module[ {data, settings, mod},
			InitializeCompiler["InstallLLVM" -> True];
			settings = getCodeGenOptions[ CompileToLLVMModule, opts];
			startProcess[pm, settings];
			RunPass[CreateLLVMIRPass, pm, settings];
			If[
				Lookup[settings, "CreateWrapper"] === "Legacy",
				createFunctionWrappers[pm, True]];
			data = pm["getProperty", "LLVMLinkData"];
			mod = LLVMModule[data["moduleId"]];

			runLLVMOptimizationOption[mod, settings["LLVMOptimization"], False];
			endProcess[pm, settings];

			DisposeBuilders[data];

			EnableLLVMModuleFinalization[mod];

			mod
		]
		,
		{{_, CreateFailure}}
	]

CompileToLLVMModule[ CompileValues[ sym_Symbol], opts:OptionsPattern[]] :=
	Module[ {progExpr},
		InitializeCompiler["InstallLLVM" -> True];
		progExpr = ToProgramExpr[ sym];
		If[FailureQ[progExpr],
			progExpr,
			CompileToLLVMModule[progExpr, opts]]
	]

CompileToLLVMModule[ expr_, opts:OptionsPattern[]] :=
	CatchException[
		Module[ {settings, pm, mod},
			InitializeCompiler["InstallLLVM" -> True];
			settings = getOptions[CompileToIR, expr, opts];
			startProcess[expr, settings];
			pm = iCompileToIR[expr, settings];
			mod = CompileToLLVMModule[pm, opts];
			endProcess[expr, settings];
			mod
		]
		,
		{{_, CreateFailure}}
	]






Compile::notfound = "The function module `1` was not found."
Compile::tynotfound = "The function module type `1` was not found."
Compile::mainnotfound = "The main function cannot be found."
Compile::wrapper = "CreateWrapper of True is not supported."
Compile::valid = "The LLVM Module is not valid and LLVM passes cannot be run."

Off[Compile::valid]

CompileSymbols[ sym_Symbol, opts:OptionsPattern[]] :=
	Module[ {fun},
		fun = CompileToExternalFunction[ CompileValues[sym], opts];
		If[ FailureQ[ fun],
			fun,
			FixDownValues[ {sym, fun}]]
	]


CompileToExternalFunction[ CompileValues[ sym_Symbol], opts:OptionsPattern[]] :=
	Module[ {progExpr},
		progExpr = ToProgramExpr[ sym];
		If[FailureQ[progExpr],
			progExpr,
			CompileToExternalFunction[progExpr, "CreateWrapper" -> True, opts]]
	]


CompileToExternalFunction[expr_, opts:OptionsPattern[]] :=
	CompileToCodeFunction[expr, opts]



(*
 Get the EE environment
*)
getExecutionEngine[ a_] :=
	a

(*
  Test whether to use the Evaluator EE
*)
getExecutionEngine[ Automatic] :=
	If[ TrueQ[Compile`Utilities`$UseEvaluatorExecution],
		"Evaluator", "Compiler"]



CompileToCodeFunction[ expr_, opts:OptionsPattern[]] :=
	Module[{list, execEngine = getExecutionEngine[ OptionValue["ExecutionEngine"]]},
		If[ execEngine === "Evaluator",
			Return[CreateEvaluationCodeFunction[expr, opts]]];
		list = CompileToCodeFunctionList[expr, opts];
		If[ MatchQ[ list, {__}],
				First[list],
				list]
	]

CompileToCodeFunctionList[ expr_, opts:OptionsPattern[]] :=
	CatchException[
		Module[ {settings, genOpts, pm, errorFunction, ef, data},
			InitializeCompiler["InstallLLVM" -> True];
			genOpts = getCodeGenOptions[ CompileToCodeFunction, opts];
			settings = getOptions[CompileToCodeFunction, expr, opts];
			startProcess[expr, genOpts];
			pm = iCompileToIR[expr, settings];
			(*
			 Creates the data and stores in a property.
			*)
			pm = RunPass[CreateLLVMIRPass, pm, genOpts];

            errorFunction = OptionValue[CompileToCodeFunction, Flatten[{opts}], "ErrorFunction"];
			ef = createExternalFunctionList[ pm, CompileToCodeFunction, expr, errorFunction, genOpts];
			endProcess[expr, genOpts];

			data = pm["getProperty", "LLVMLinkData"];
			DisposeBuilders[data];
			pm["dispose"];
			ef
		]
		,
		{{_, CreateFailure}}
	]


createFunctionWrappers[pm_, legacy_?BooleanQ] :=
	Module[ {llvmData, firstName, names, funs, initName, initId},
		llvmData = pm["getProperty", "LLVMLinkData", Null];
		If[llvmData === Null,
			Message[Compile::llvmlinkdata];
			Return[$Failed]
		];
		If[!TrueQ[legacy],
			Message[Compile::wrapper];
			Return[$Failed]
		];
		funs = pm["exportedFunctions"];
		(*
		 Temporary fix to make sure one function is exported. 
		 Really this should be dealt with elsewhere.
		*)
		If[ funs === {},
			funs = {pm["getElement",1]}];
		
		firstName = First[funs]["name"];
		{initName, initId} = InitializeWrapper[llvmData, firstName];
		names = Map[createFunctionWrapper[llvmData, #]&, funs];
		FinalizeWrapper[ llvmData, initId];
		If[MemberQ[names, $Failed],
			$Failed,
			{initName, names}]
	]

createFunctionWrapper[llvmData_, fun_?FunctionModuleQ] := 
	Module[{ ty},
		ty = fun["type"]["unresolve"];
		Replace[ty, {
			TypeSpecifier[args_List -> res_] :> 
				{fun, CreateWrapper[llvmData, fun["name"], args, res],fun["name"]},
			Type[args_List -> res_] :> 
				{fun, CreateWrapper[llvmData, fun["name"], args, res],fun["name"]},
			_ :> (
				Message[Compile::tynotfound, ty];
				$Failed
				)
		}]
]

(*
 This will be simpler when we can drop the Legacy CreateWrapper code
*)
createExternalFunctionList[ pm_, head_, func_, errorFunction_, opts_] :=
    CatchException[
    	Module[ {legacy, funWrapperData, initName, ef, data, mod},
			data = pm["getProperty", "LLVMLinkData"];
			legacy = Lookup[opts, "CreateWrapper"] === "Legacy";
			If[!legacy,
				Throw[{"Unimplemented"}]];
			funWrapperData = createFunctionWrappers[pm, True];
			If[!MatchQ[funWrapperData, {_, {{_,_,_}..}}],
				(* Rely on messages created by createFunctionWrapper *)
				Return[funWrapperData]
			];
			VerifyModule[data];
			mod = LLVMModule[data["moduleId"]];
			runLLVMOptimizationOption[mod, opts["LLVMOptimization"], True];
			InitializeCodeGeneration[data];
			initName = First[funWrapperData];
			ef = Map[createExternalFunction[data, func, errorFunction, initName, #, opts]&, Last[funWrapperData]];
			ef
		]
		,
		{{_, CreateFailure}}
	]


	

createExternalFunction[data_, func_, errorFunction_, initName_, {fm_, name_, rawName_}, opts_] :=
	Module[ {funData,  ty, str, ef},
			ty = fm["type"]["unresolve"];
            funData = 
            	Join[<| "Signature" -> ty, "Input" -> func, "ErrorFunction" -> errorFunction|>, 
            		ExtraCodeFunctionData[]];
			If[ TrueQ[Lookup[opts, "SaveIR"]],
				str = LLVMToString[LLVMModule[data["moduleId"]]];
				AssociateTo[funData, "LLVMIR" -> str]
			];
			ef = CreateExternalFunction[ data, {name, initName, rawName}, funData];
			If[FailureQ[ef],
			    Return[ef]
			];
			EnableCompiledCodeFunctionFinalization[ef];
			ef
	]

(*
  This should not create the wrapper.  That would be done by a caller,  eg CompileToCodeFunction.
  No need for any special code here,  CreateLLVMIRPreprocessPass doesn't create it.
*)
CompileToWIRSerialization[ expr_, opts:OptionsPattern[]] :=
	CatchException[
		Module[ {settings, genOpts, pm, wir},
			InitializeCompiler[];
			settings = getOptions[CompileToIR, expr, opts];
			genOpts = getCodeGenOptions[ CompileToWIRSerialization, opts];
			startProcess[expr, genOpts];
			pm = iCompileToIR[expr, settings];
			RunPass[CreateLLVMIRPreprocessPass, pm, genOpts];
			wir = WIRSerialize[pm["typeEnvironment"], pm];
			endProcess[expr, genOpts];
			wir
		]
		,
		{{_, CreateFailure}}
	]


CompileToExternalFunction[ arg_WIRSerialization, opts:OptionsPattern[]] :=
	CompileToCodeFunction[arg, opts]

CompileToCodeFunction[ arg_WIRSerialization, opts:OptionsPattern[]] :=
	CatchException[
		Module[ {pm, genOpts, errorFunction, ef, tyEnv},
			InitializeCompiler["InstallLLVM" -> True];
			genOpts = getCodeGenOptions[ CompileToCodeFunction, opts];
			startProcess[arg, genOpts];
			tyEnv = OptionValue["TypeEnvironment"];
			pm = WIRDeserialize[tyEnv, arg];
			RunPass[CreateLLVMIROnlyPass, pm, genOpts];
            errorFunction = OptionValue[CompileToCodeFunction, Flatten[{ opts}], "ErrorFunction"];
			ef = createExternalFunction[pm, CompileToCodeFunction, Missing["NotAvailable"] (* original expr unavailable *), errorFunction, genOpts];
			endProcess[arg, genOpts];
			ef
		]
		,
		{{_, CreateFailure}}
	]


CompileToExternalFunction[ pm_?ProgramModuleQ, opts:OptionsPattern[]] :=
	CompileToCodeFunction[pm, opts]

CompileToCodeFunction[ pm_?ProgramModuleQ, opts:OptionsPattern[]] :=
	Module[{list},
		list = CompileToCodeFunctionList[pm, opts];
		If[ MatchQ[ list, {__}],
				First[list],
				list]
	]
	
CompileToCodeFunctionList[ pm_?ProgramModuleQ, opts:OptionsPattern[]] :=
	CatchException[
		Module[ {genOpts, errorFunction, ef},
			InitializeCompiler["InstallLLVM" -> True];
			genOpts = getCodeGenOptions[ CompileToCodeFunction, opts];
			If[pm["exportedFunctions"] === {},
				pm["getElement", 1]["setProperty", "exported" -> True]];
			startProcess[pm, genOpts];
			RunPass[CreateLLVMIRPass, pm, genOpts];
            errorFunction = OptionValue[CompileToCodeFunction, Flatten[{opts}], "ErrorFunction"];
			ef = createExternalFunctionList[ pm, CompileToCodeFunction, Missing["NotAvailable"] (* original expr unavailable *), errorFunction, genOpts];
			endProcess[expr, genOpts];
			ef
		]
		,
		{{_, CreateFailure}}
	]





End[]

EndPackage[]

