BeginPackage["Compile`CompileToLibrary`"]

Begin["`Private`"]


Needs["Compile`"]
Needs["CCompilerDriver`"]
Needs["Compile`Core`IR`ProgramModule`"]
Needs["CompileAST`Class`Base`"] (* For MExprQ *)
Needs["LLVMTools`"] (* for LLVMToObjectFile *)
Needs["LLVMCompileTools`"] (* for $LLVMCompilerResources *)
Needs["CompiledLibrary`"];
Needs["TypeFramework`"]

(* private imports *)
Needs["Compile`Driver`"]



$sharedOptions = Compile`Driver`Private`$sharedOptions;

CompileToLibrary::usage = "CompileToLibrary[(_File | _String), opts:OptionsPattern[]]";

CompileToLibrary::addressfunctions = "Option \"AddressFunctions\" setting `1` expected to be a list of strings.";
CompileToLibrary::addresswrappers = "Option \"AddressWrappers\" setting `1` expected to be True or False.";
CompileToLibrary::notfound = "Could not find function with name `1` in the input.";
CompileToLibrary::createlibfailed = "The library `1` could not be created. The CompilerOption \"TraceFunction\" -> Print may provice more information."
CompileToLibrary::libname = "Option \"LibraryName\" setting `1` expected to be Automatic | _String.";

$cnt = 1;

getCounter[] :=
	ToString[$cnt++]


Options[CompileToLibrary] = Join[$sharedOptions, {
	(* Which functions should we generate LibraryLink address-getters for? *)
	"AddressFunctions" -> Automatic,
	(* Should we generate address-getters for the wrapper functions as well? *)
	"AddressWrappers" -> True,
	"LibraryName" -> Automatic,

	"LLVMDebug" -> False,
	"LazyJIT" -> Compile`Utilities`$LazyJIT,
	"LLVMOptimization" -> Compile`Utilities`$LLVMOptimization,
	"TraceFunction" -> None,
	"CreateLibraryOptions" -> {"SystemLibraries" -> {}},
	"IncludeInput" -> False
}];

initializeDebugger[] :=
	(
	Debugger`DebugOn[True];
	RuntimeTools`SetRuntimeMode[False];
	initializeDebugger[] = True;
	)


CompileToLibrary[File[path_], optsIn:OptionsPattern[]] := Module[{func, debug},
	InitializeCompiler["InstallLLVM" -> True];
	debug = OptionValue["LLVMDebug"];
	If[ debug,
		initializeDebugger[];
		RuntimeTools`SetRuntimeMode[True]];
	func = Get[path];
	If[ debug,
		RuntimeTools`SetRuntimeMode[False]];
	CompileToLibrary[func, optsIn]
];


ProgramFunctionQ[x_Function] :=
	True

ProgramFunctionQ[MetaData[___]@x_Function] :=
	True

ProgramFunctionQ[x_Program] :=
	True

ProgramFunctionQ[_] :=
	False

CompileToLibrary[inp_?ProgramFunctionQ, optsIn:OptionsPattern[]] := Module[{opts = Flatten[{optsIn}], pm, compileExprOpts},
	InitializeCompiler["InstallLLVM" -> True];
	compileExprOpts = FilterRules[opts, Options[CompileToIR]];
	pm = CompileToIR @@ {inp, compileExprOpts};
	CompileToLibrary[ pm, inp, optsIn]
]

CompileToLibrary[func_?MExprQ, optsIn:OptionsPattern[]] := Module[{opts = Flatten[{optsIn}], pm,
                                                                   compileExprOpts},
	InitializeCompiler["InstallLLVM" -> True];
	compileExprOpts = FilterRules[opts, Options[CompileExpr]];
	pm = CompileExpr @@ {func, compileExprOpts};
	CompileToLibrary[pm, optsIn]
];

(* Create a library from `pm`, which will include generated LibraryFunctionLoad'able functions
   to get the address of each non-local function in `pm`. If "AddressWrappers" -> True,
   address-getters will be generated for the wrappers as well. *)
CompileToLibrary[pm_?ProgramModuleQ, inp_:None, optsIn:OptionsPattern[]] := Module[{
	    opts = Flatten[{optsIn}], functions, entryFunction, llvmMod, libName, compileToLLVMStringOpts, llvmToObjectFileOpts,
        objectFile, libLinkSource, lib, traceFun, runtimeLibrary, debug, functionsData,
		functionPointerTemplate, libData, libDataString, libDataTemplate, extLibs,
		createLibOpts
	},
	InitializeCompiler["InstallLLVM" -> True];

	(* TODO: Check that if "AddressWrappers" == True (and we're not using "Legacy" wrappers)
             `pm` actually contains the wrappers. We could run the GenerateWrapper pass if not. *)

	functions = Replace[OptionValue["AddressFunctions"], {
		(* By default we generate an address-getter for every exported function. *)
		Automatic :> Map[#["name"]&, pm["exportedFunctions"]],
		funcs:{___String} :> funcs,
		other_ :> (
			Message[CompileToLibrary::addressfunctions, other];
			Return[$Failed]
		)
	}];
	entryFunction = pm["entryFunctionName"];
	libName = Replace[OptionValue["LibraryName"], {
		Automatic :> "lib" <> getCounter[],
		(* TODO: Check that the string doesn't contain any illegal filename characters? *)
		name_String :> name,
		_ :> (
			Message[CompileToLibrary::libname, libName];
			Return[$Failed]
		)
	}];

	debug = OptionValue["LLVMDebug"];

	(*Create the object file for the PM -- this will be linked with the generated address-getters.*)

	compileToLLVMStringOpts = FilterRules[opts, Options[CompileToLLVMString]];
	Replace[OptionValue["AddressWrappers"], {
		(* If we're creating address getters for the wrappers, make sure we're creating wrappers *)
		True :> AppendTo[compileToLLVMStringOpts, "CreateWrapper" -> "Legacy"],
		False :> Null,
		other_ :> (
			Message[CompileToLibrary::addresswrappers, other];
			Return[$Failed]
		)
	}];
	llvmMod = CompileToLLVMModule @@ {pm, System`Sequence @@ compileToLLVMStringOpts};

	objectFile = CreateFile[] <> ".obj";
	llvmToObjectFileOpts = FilterRules[opts, Options[LLVMToObjectFile]];
	objectFile = LLVMToObjectFile[llvmMod, objectFile, System`Sequence @@ llvmToObjectFileOpts];
	(* TODO: Can we delete this object file after creating the library? *)

	(* Create the LibraryLink library *)

	libLinkSource = "
	#include \"WolframLibrary.h\"

	/* Return the version of Library Link */
	DLLEXPORT mint WolframLibrary_getVersion( ) {
		return WolframLibraryVersion;
	}

	/* Initialize Library */
	DLLEXPORT int WolframLibrary_initialize( WolframLibraryData libData) {
		return LIBRARY_NO_ERROR;
	}

	/* Uninitialize Library */
	DLLEXPORT void WolframLibrary_uninitialize( WolframLibraryData libData) {
		return;
	}
	";

	(* This template will be filled in for every function in the PM which we want
       to get a pointer to. *)
	functionPointerTemplate = StringTemplate["
	extern void * `funcName`;

	EXTERN_C DLLEXPORT int `funcName`_Pointer(WolframLibraryData libData, mint argc,
	                                          MArgument* args, MArgument res) {
		MArgument_setInteger(res, (mint)&`funcName`);
		return LIBRARY_NO_ERROR;
	}
	"];

	(* This ends up looking like: <|
		"funcName" -> <| "Type" -> fm["type"]["unresolve"],
		                 "Unwrapped" -> funcName <> "_Pointer",
		                 "Wrapped" -> funcName <> "_Wrapper_Pointer"
		|>,
		...
	|> *)
	functionsData = <||>;

	Scan[
		Function[funcName,
			Module[{fm = pm["getFunctionModule", funcName], data = <||>, typeAssoc = <||>},
				If[MissingQ[fm],
					Message[CompileToLibrary::notfound, funcName];
					Return[$Failed]
				];

				typeAssoc = CompiledLibrary`Private`typeToAssociation[fm["type"]["unresolve"]];
				Assert[AssociationQ[typeAssoc]];

				AssociateTo[data, "Type" -> typeAssoc];
				AssociateTo[data, "Unwrapped" -> funcName <> "_Pointer"];
				AssociateTo[data, "Initialization" -> entryFunction <> "_Initialization_Pointer"];

				(* Append LibraryLink C code for the unwrapped function *)
				libLinkSource = libLinkSource <>
					functionPointerTemplate[<| "funcName" -> funcName |>];

				If[TrueQ[OptionValue["AddressWrappers"]],
					(* Append LibraryLink C code for the wrapper function *)
					libLinkSource = libLinkSource <>
						functionPointerTemplate[<| "funcName" -> funcName <> "_Wrapper" |>];
					(* Append LibraryLink C code for the wrapper init function *)
					
					(*
					 There is only one initialization function
					*)
					If[ funcName === entryFunction,
						libLinkSource = libLinkSource <>
							functionPointerTemplate[<| "funcName" -> funcName <> "_Initialization" |>]];

					(* libLinkSource = libLinkSource <>
						functionPointerTemplate[<| "funcName" -> "initializationDone" |>]; *)

					AssociateTo[data, "Wrapped" -> funcName <> "_Wrapper_Pointer"];
				];

				AssociateTo[functionsData, funcName -> data];
			]
		],
		functions
	];

	libLinkSource = libLinkSource <>
			functionPointerTemplate[<| "funcName" -> "initializeRTL" |>];

	(*
		Create the function data getter
	*)


	libDataTemplate = StringTemplate["
	EXTERN_C DLLEXPORT int GetLibraryInformation(WolframLibraryData libData, mint argc,
	                                             MArgument* args, MArgument res) {
		MArgument_setUTF8String(res, \"`libData`\");
		return LIBRARY_NO_ERROR;
	}
	"];

	Assert[MatchQ[functions, {___String}]];

	libData = Join[
			<|
			"ExportedFunctions" -> functions,
			"FunctionData" -> functionsData,
			"UtilityFunction" -> "initializeRTL_Pointer"
			|>, 
			GetInput[ inp, opts],
			ExtraCodeFunctionData[]];
	

	libDataString = ExportString[libData, "JSON"];
	(* Escape the " characters, for embedding in the C source template *)
	libDataString = StringReplace[libDataString, "\\" -> "\\\\"];
	libDataString = StringReplace[libDataString, "\"" -> "\\\""];
	(* Remove the new line characters, we can't embed a multiline string into C source code. Also
	   replace indents with single spaces, just for readability *)
	libDataString = StringReplace[libDataString, {"\r\n" -> " ", "\n" -> " ", " ".. -> " "}];

	libLinkSource = libLinkSource <> libDataTemplate[<|"libData" -> libDataString|>];

	(*
		Create the library
	*)

	runtimeLibrary = FindWolframRTLStub[];
	
	runtimeLibrary = StringReplace[runtimeLibrary, "dll" -> "lib"];

	traceFun = OptionValue["TraceFunction"];

	extLibs = pm["getProperty", "externalLibraries", {}];
	If[ !MatchQ[extLibs, {__String}],
		extLibs = {}];
	AppendTo[extLibs, runtimeLibrary];
	createLibOpts = OptionValue["CreateLibraryOptions"];
	lib = CreateLibrary[libLinkSource, libName, 
				"ExtraObjectFiles" -> {objectFile},
				 createLibOpts,
				"ShellCommandFunction" -> traceFun, "ShellOutputFunction" -> traceFun, 
				"Debug" -> debug, "Libraries" -> extLibs];
	If[FailureQ[lib],
		Message[CompileToLibrary::createlibfailed, libName];
		Return[$Failed];
	];

	CompiledLibrary`CompiledLibrary[lib]
]


GetInput[ inp_, opts:OptionsPattern[]] :=
	Module[{includeInput = OptionValue[CompileToLibrary, opts, "IncludeInput"]},
		If[!TrueQ[includeInput] || inp === None,
			Return[<||>]];
		<|
			"InputExpression" -> ExportString[inp, "ExpressionJSON", "Compact" -> True]
		|>
	]
		

End[]

EndPackage[]

