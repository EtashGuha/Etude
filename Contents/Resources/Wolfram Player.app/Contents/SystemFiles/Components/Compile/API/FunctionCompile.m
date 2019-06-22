

BeginPackage["Compile`API`FunctionCompile`"]

CreateEvaluationCodeFunction

Begin["`Private`"]

Needs["Compile`"]
Needs["LLVMTools`"]
Needs[ "Compile`API`Utilities`"]
Needs["CompiledLibrary`"]
Needs["Compile`API`RuntimeErrors`"]


prots = Unprotect[FunctionCompile, Compile`Internal`RepairCompiledCodeFunction]

Clear[FunctionCompile]
Clear[Compile`Internal`RepairCompiledCodeFunction]

FunctionCompile::fun = "Argument `1` is expected to be a Function."


(*
 FunctionCompile
*)

Options[FunctionCompile] = 
	{
	CompilerOptions -> Automatic	
	}


FunctionCompile[ func_, opts:OptionsPattern[]] :=
	Module[ {res = Catch[
					iFunctionCompileDriver[ functionCompileData, func, opts],
					_functionCompileException,
					Part[#2,1]&]
	},
		res /; res =!= Null
	]

FunctionCompile[args___ /; (compileArgumentError[{args}, FunctionCompile, 1]; False)] :=
	Null



functionCompileData = <| 
	"head" -> FunctionCompile
|>
	

(*
  On error throw functionCompileException[ Null] to return unevaluated and 
  functionCompileException[ arg] to return arg. 
*)

iFunctionCompileDriver[ data_, func_, opts:OptionsPattern[]] :=
	Module[ {ef, ccFOpts = getCompilerOptions[data["head"], functionCompileException, {opts}]},
		checkFunction[ data, func];
		ef = CompileToCodeFunction[func, ccFOpts, "ErrorFunction" -> Automatic];
		ef
	]

	

checkFunction[data_, HoldPattern[Function][ args_, body_]] :=
	Null
	
checkFunction[data_, func_] :=
	With[ {h = data["head"]},
		Message[h::fun, func];
		Throw[Null, functionCompileException[Null]]
	]

CompiledCodeFunction::form = "`1` is not recognised as a CompiledCodeFunction that was created by the Wolfram Compiler."
CompiledCodeFunction::rest = "`1` could not be restored to a functioning CompiledCodeFunction."


(*
  RepairCompiledCodeFunction
*)

(*
 Try to repair a CompiledCodeFunction that has an association for argument 1.
 See if this is loaded from a library or if there is an Input Field.
  
 If not give the CompiledCodeFunction::form  message.
 
 The return value doesn't matter,  this is going back to internal code which 
 will make the result uncompiled. Since this is used as a head, that might be 
 better.
*)
Compile`Internal`RepairCompiledCodeFunction[ data_?AssociationQ, args_] :=
	Module[ {},
		Which[
			useEvaluatorExecutionEngine[],
				createExecutionFunction[data, args]
			,
			KeyExistsQ[data, "LibraryPath"] && KeyExistsQ[data, "FunctionName"],
				reloadLibraryWork[data, args]
			,
			KeyExistsQ[data, "Input"],
				recompileFunction[data, args]
			,
			True,
				Message[CompiledCodeFunction::form, makeMessageArgument[data, args]];
				Null]
	]


(*
  Test whether to use the Evaluator EE
*)
useEvaluatorExecutionEngine[] :=
	TrueQ[Compile`Utilities`$UseEvaluatorExecution]

createExecutionFunction[data_, args_] :=
	Module[ {originalInput, errorFunction},
		originalInput = Lookup[data, "Input", Null];
		errorFunction = Lookup[data, "ErrorFunction", Null];
		If[originalInput =!= Null,
			CreateEvaluationCodeFunction[originalInput, "ErrorFunction" -> errorFunction]
			,
			Message[CompiledCodeFunction::form, makeMessageArgument[data, args]];
			Null]
	]

reloadLibraryWork[ data_, args_] :=
	Module[{ef = ReloadFromLibrary[data, args]},
		If[ef === Null,
			Message[CompiledCodeFunction::form, makeMessageArgument[data, args]];
			Null
			,
			ef]
	]

(*
 There is known to be an Input field,  use this to try and recompile the function.
 If this doesn't work then give the CompiledCodeFunction::rest message.
*)
recompileFunction[ data_, args_] :=
	Module[{errorFun = Lookup[data, "ErrorFunction", Null],
			originalInput = Lookup[data, "Input", Null], ef},
		ef = Quiet[CompileToCodeFunction[originalInput, "ErrorFunction" -> errorFun]];
		If[Head[ef] =!= CompiledCodeFunction,
			Message[CompiledCodeFunction::rest, makeMessageArgument[data, args]]];
		ef
	]


(*
 Called when CompiledCodeFunction with no arguments is found. Just issue a message,
 see comments above about return value.
*)
Compile`Internal`RepairCompiledCodeFunction[ ] :=
	Module[ {},
		Message[CompiledCodeFunction::form, HoldForm[CompiledCodeFunction][]];
	]

(*
 Called when CompiledCodeFunction with no association argument is found. Just issue a message,
 see comments above about return value.
*)
Compile`Internal`RepairCompiledCodeFunction[ data_, args_] :=
	Module[ {},
		Message[CompiledCodeFunction::form, makeMessageArgument[data, args]];
	]

makeMessageArgument[data_, args_] :=
	Apply[ HoldForm[CompiledCodeFunction], Prepend[args, data]]



(*
  Called from CompileToCodeFunction when "TargetFunction" -> "Evaluation"
*)
CreateEvaluationCodeFunction[fun_, opts:OptionsPattern[]] :=
	Module[{funData, evalFun = ProcessUncompiledFunction[fun], errorFunction},
		errorFunction = Lookup[ Flatten[{opts}], "ErrorFunction", Null];
		funData = 
            	Join[<| "Input" -> fun, "ErrorFunction" -> errorFunction|>, 
            		ExtraCodeFunctionData[]];
		Compile`CreateCompiledCodeFunction[{funData, evalFun}]
	]


End[]

EndPackage[]
