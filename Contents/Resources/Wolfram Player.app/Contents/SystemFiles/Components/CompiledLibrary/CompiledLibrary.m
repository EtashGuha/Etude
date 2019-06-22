BeginPackage["CompiledLibrary`"]

FunctionCompileLibraryFunctionLoad
CompiledLibraryLoadFunction
CompiledLibraryLoadFunctions
CompiledLibraryInformation

CompiledLibraryLoadFunction::usage = "Load a function from a compiled library"
CompiledLibraryLoadFunctions::usage = "Load all of the functions in a compiled library"
CompiledLibraryInformation::usage = "Get an Association with information about a compiled library"

CompiledLibraryLoadFunction::notfound = "A function with the name `1` was not found"
CompiledLibraryLoadFunction::notexported = "A function with the name `1` was found, but is not exported"
CompiledLibraryLoadFunction::nowrapper = "A function with the name `1` was found, but does not have a wrapper \
(hint: Use the \"AddressFunctions\" and \"AddressWrappers\" options on CompileToLibrary to control wrapper generation)"

CompiledLibraryInformation::nofile = "The library specified does not exist: `1`"
CompiledLibraryInformation::datagetter = "The library specified does not contain a GetLibraryInformation function"

LibraryFunctionLoad::noform = "`1` is not a structured library that can be loaded."
LibraryFunctionLoad::nofun = "A function with the name `1` was not found."

(* This symbol has no definition, but is used to wrap the path to a dynamic library which was
   created using CompileToLibrary *)
CompiledLibrary

ExtraCodeFunctionData
ReloadFromLibrary

Begin["`Private`"]

Needs["CompileUtilities`Error`Exceptions`"]
Needs["Compile`API`CompiledCodeFunction`"]

(*
  Called when creating a CompiledCodeFunction loaded from a library.
  If there is are "FunctionName" and "LibraryPath" then add this to the 
  LibraryFunction mechanism.
*)
setupLibraryFunction[ func:HoldPattern[CompiledCodeFunction][data_?AssociationQ, ___]] :=
	Module[{name, path},
		name = Lookup[data, "FunctionName", Null];
		path = Lookup[data, "LibraryPath", Null];
		If[ name =!= Null && path =!= Null,
			Compile`SetupCompiledCodeFunction[ path, name, func]];
		func
	]

setupLibraryFunction[arg_] :=
	arg

FunctionCompileLibraryFunctionLoad[args___] :=
	Module[{fun},
		fun = CompiledLibraryLoadFunction[args, "ErrorFunction" -> Automatic];
		setupLibraryFunction[ fun]
	]

Options[CompiledLibraryLoadFunctions] = {
	(* TODO: Possible option? See FIXME above CompiledLibraryLoadFunctions *)
	"ExportedFunctionsOnly" -> True,
	"ErrorFunction" -> Null
}

Options[CompiledLibraryLoadFunction] = {
	"ErrorFunction" -> Null
}

Unprotect[ CompiledLibraryLoadFunction]
Clear[CompiledLibraryLoadFunction]

CompiledLibraryLoadFunction[lib_CompiledLibrary, opts:OptionsPattern[]] :=
	CompiledLibraryLoadFunction[lib, "Main", opts]
		
CompiledLibraryLoadFunction[libPath_, opts:OptionsPattern[]] :=
		CompiledLibraryLoadFunction[ CompiledLibrary[libPath], "Main", opts]

CompiledLibraryLoadFunction[libPath_String, funcName_String, opts:OptionsPattern[]] :=
		CompiledLibraryLoadFunction[ CompiledLibrary[libPath], funcName, opts]

CompiledLibraryLoadFunction[CompiledLibrary[File[libPath]], funcName_String, opts:OptionsPattern[]] :=
		CompiledLibraryLoadFunction[ CompiledLibrary[libPath], funcName, opts]

CompiledLibraryLoadFunction[_, _, opts:OptionsPattern[]] :=
		Null

CompiledLibraryLoadFunction[CompiledLibrary[libPath_String], funcName_String, optsIn:OptionsPattern[]] := Module[{
	libData, funcData, exported, wrapped, wrapperAddrFun, wrapperAddr, initAddr, unwrappedAddr, typeSpec, sysOpts,
	ccfData, opts = Flatten[{optsIn}], errorFunction
},
	opts = FilterRules[ opts, Options[CompiledLibraryLoadFunction]];
	errorFunction = OptionValue[CompiledLibraryLoadFunction, opts, "ErrorFunction"];
	libData = CompiledLibraryInformation[CompiledLibrary[libPath]];
	If[libData === $Failed,
		(* CompiledLibraryInformation will already have messaged for us *)
		Return[$Failed];
	];

	funcData = Lookup[libData["FunctionData"], funcName, $Failed];
	If[funcData === $Failed,
		Message[LibraryFunctionLoad::nofun, funcName];
		Return[$Failed]
	];

	exported = MemberQ[libData["ExportedFunctions"], funcName];
	If[!exported,
		Message[LibraryFunctionLoad::nofun, funcName];
		Return[$Failed]
	];

	wrapped = Lookup[funcData, "Wrapped", $Failed];
	If[wrapped === $Failed,
		Message[LibraryFunctionLoad::noform, libPath];
		Return[$Failed]
	];
(*
 Make sure that "DynamicLibraryGlobal" is set to False,  this prevents 
 different compiled libraries from interfering with each other.
*)
	sysOpts = SystemOptions["DynamicLibraryOptions"];
	SetSystemOptions["DynamicLibraryOptions" -> {"DynamicLibraryGlobal" -> False}];
	wrapperAddrFun = LibraryFunctionLoad[libPath, wrapped, {}, Integer];
	wrapperAddr = wrapperAddrFun[];
	initAddr = LibraryFunctionLoad[libPath, funcData["Initialization"], {}, Integer][];
	unwrappedAddr = LibraryFunctionLoad[libPath, funcData["Unwrapped"], {}, Integer][];
	SetSystemOptions[sysOpts];
	
	processUtility[ libPath, Lookup[ libData, "UtilityFunction"]];
	Assert[IntegerQ[wrapperAddr]];
	Assert[IntegerQ[initAddr]];
	Assert[IntegerQ[unwrappedAddr]];

	typeSpec = Lookup[funcData, "Type", Null];
	If[MatchQ[typeSpec, TypeSpecifier[_]],
		typeSpec = First[typeSpec]];
	typeSpec = ToString[typeSpec, InputForm];
	(* The LoadedFunction" field ensures that the LibraryFunction is not unloaded -- this is a hack,
	   but it works for now. If we don't do this, calling the CompiledCodeFunction later may cause a
	   crash *)
	ccfData = Join[<| 
		"Signature" -> funcData["Type"], "ErrorFunction" -> errorFunction,
	     "LoadedFunction" -> wrapperAddrFun, "FunctionName" -> funcName,
	     "LibraryPath" -> First[wrapperAddrFun]
	     |>, getExtraData[libData], getInput[libData]];
	Compile`CreateCompiledCodeFunction[{ccfData,
	                     wrapperAddr, initAddr, unwrappedAddr, typeSpec}]
]


(*
 Return the extra data to pass into the CCF.
 KeyTake doesn't include the data if the key is absent.
*)
getExtraData[libData_] :=
	KeyTake[libData, {"VersionData", "SystemID"}]


getInput[libData_] :=
	Module[ {inputExpr = Lookup[libData, "InputExpression", Null]},
		If[inputExpr === Null,
			Return[<||>]];
			
		inputExpr = Quiet[ ImportString[inputExpr, "ExpressionJSON"]];
		If[ inputExpr === $Failed,
			Return[<||>]];
		<| "Input" -> inputExpr |>
	]


ExtraCodeFunctionData[] :=
	<|"SystemID" -> $SystemID,
		"VersionData" -> {$VersionNumber, $ReleaseNumber, $MinorReleaseNumber}|>


getAddressFromKernel[ addrList_, name_] :=
	Module[{addr},
		addr = SelectFirst[ addrList, MatchQ[#, {name, _}]&, 0];
		If[addr === 0,
			ThrowException[{"Cannot find " <> name}]];
		Last[addr]
	]

processUtility[ libPath_, utilityPointer_String] :=
	Module[ {addrList = Compile`GetFunctionAddresses[], allocListAddr, watchAddr, watchProcess,
			utilityFunAddr = LibraryFunctionLoad[libPath, utilityPointer, {}, Integer][]},
		allocListAddr = getAddressFromKernel[addrList, "AllocatorList"];
		watchAddr = getAddressFromKernel[addrList, "WatchAddress"];
		watchProcess = getAddressFromKernel[addrList, "WatchProcess"];
		Compile`InitializeCompileUtilities[utilityFunAddr, allocListAddr, watchAddr, watchProcess];
	]


(* FIXME: This WILL currently error with a bunch of LibraryFunctionLoad messages. This is because
          We currently have no way of distinguishing the top-level functions in a PM which were
		  generated by compiling an expr (and have wrappers) from those which were generated 
		  (like System`Plus_Integer64_Integer64_Integer64). So this will attempt to load a
		  non-existant wrapper for the generated function. *)

CompiledLibraryLoadFunctions[CompiledLibrary[libPath_String], opts:OptionsPattern[]] := Module[{
	libData, functionData, rules
},
	libData = CompiledLibraryInformation[CompiledLibrary[libPath]];
	If[FailureQ[libData],
		(* CompiledLibraryInformation will already have messaged, so lets not do anything *)
		Return[$libData]
	];
	Assert[AssociationQ[libData]];

	functionData = libData["FunctionData"];
	Assert[AssociationQ[functionData]];

	If[TrueQ[OptionValue["ExportedFunctionsOnly"]],
		(* Filter the non-exported functions out of `functionData` *)
		functionData = functionData[[ libData["ExportedFunctions"] ]]
	];

	rules = Map[# -> CompiledLibraryLoadFunction[CompiledLibrary[libPath, opts], #]&, Keys[functionData]];
	Association @@ rules
]

(* The data returned by this function is determined by the implementation of CompileToLibrary *)
CompiledLibraryInformation[libPath_String, opts:OptionsPattern[]] :=
	CompiledLibraryInformation[CompiledLibrary[libPath], opts]
	
CompiledLibraryInformation[CompiledLibrary[libPath_String], opts:OptionsPattern[]] := 
	Module[{
		dataGetter, data, sysOpts
	},

(*
 Make sure that "DynamicLibraryGlobal" is set to False,  this prevents 
 different compiled libraries from interfering with each other.
*)
		sysOpts = SystemOptions["DynamicLibraryOptions"];
		SetSystemOptions["DynamicLibraryOptions" -> {"DynamicLibraryGlobal" -> False}];
		dataGetter = Quiet[LibraryFunctionLoad[libPath, "GetLibraryInformation", {}, "UTF8String"],{LibraryFunction::libload} ];
		SetSystemOptions[sysOpts];
		If[Head[dataGetter] =!= LibraryFunction,
			Message[LibraryFunctionLoad::noform, libPath];
			Return[$Failed];
		];
	
		data = dataGetter[];
		(* If there is a LibraryLink function called GetLibraryInformation in the lib, we'll just assume
		   it's not a coincidence, and consequently not do any checking *)
		Assert[StringQ[data]];
	
		(* TOOD: Might have to un-escape the double quotes? *)
		data = ImportString[data, "JSON"];
		(* Replace lists of rules with associations, working from the inner expressions outwards (doing
		   this from the outside moving inwards doesn't work because Association[{a -> {b -> d}}] will
		   evaluate to itself). *)
		data = Replace[data, r : {__Rule} :> Association[r], {0, Infinity}];
		Assert[AssociationQ[data]];
	
		(* Fix the function types, so they are in "unresolve"'ed form. We go from a form like:
		       <| "Kind" -> "TypeArrow", "Arguments" -> {___}, "Result" -> _ |>
		   to:
		       Type[{___} -> _]
		*)
		
		AssociateTo[data, "FunctionData" -> AssociationMap[fixType, data["FunctionData"]]];
	
		data
	]


(*
  Reload data from a CompiledLibrary
  Make sure to set the ErrorFunction to be the same.
  Also call setupLibraryFunction (see comments above).
*)
ReloadFromLibrary[ data_, args_] :=
	Module[{path = Lookup[data, "LibraryPath", Null],
			name = Lookup[data, "FunctionName", Null], 
			version = Lookup[data, "VersionData", Null], 
			errorFun = Lookup[data, "ErrorFunction", Null], ef},
		If[ path === Null || name === Null || !MatchQ[ version, {_,_,_}],
			Return[ Null]];
		If[ First[version] =!= $VersionNumber,
			Return[Null]];
		ef = Quiet[ CompiledLibraryLoadFunction[CompiledLibrary[path], name, "ErrorFunction" -> errorFun]];
		If[ Head[ef] =!= CompiledCodeFunction,
			Null,
			setupLibraryFunction[ef]]
	]



fixType[name_ -> funcDataArg_Association] := Module[{funcData, typeAssoc, type},
	funcData = funcDataArg;
	(* Extract the Association-form of the function type. *)
	typeAssoc = funcData["Type"];
	Assert[AssociationQ[typeAssoc]];
	type = typeFromAssociation[typeAssoc];
	(* Rewrite the "Type" field of the data for this particular function with the converted
		to "unresolve" form of the type *)
	AssociateTo[funcData, "Type" -> type];
	name -> funcData
]

(* Helpers to convert the "unresolve" form of a type object to and from a representation made up
   only of Associations and Strings. These association forms are then converted to and from JSON
   in CompileToLibrary and CompiledLibraryInformation. *)

typeToAssociation[Type[arg_]] :=
	typeToAssociation[ arg]

typeToAssociation[TypeSpecifier[arg_]] :=
	typeToAssociation[ arg]

typeToAssociation[ {args___} -> res_] := 
		<| "Kind" -> "TypeArrow",
			"Arguments" -> typeToAssociation /@ {args},
			"Result" -> typeToAssociation[res] |>

typeToAssociation[TypeFramework`TypeLiteral[val_Integer, _]] := 
	val

typeToAssociation[f_[args___]] :=
	<|"Kind" -> "TypeApplication", "Type" -> typeToAssociation[f], "Arguments" -> typeToAssociation /@ {args}|>

typeToAssociation[atom_String] := 
	atom

typeToAssociation[args___] := 
	Throw[StringForm["Unimplemented: typeToAssociation of: ``", {args}]]

(* ================
   From Association
   ================ *)

typeFromAssociation[assoc_Association] := Switch[assoc["Kind"],
	"TypeArrow",
		TypeSpecifier[typeFromAssociation /@ assoc["Arguments"] -> typeFromAssociation[assoc["Result"]]],
	"TypeApplication",
		TypeSpecifier[typeFromAssociation[assoc["Type"]][ typeFromAssociation /@ assoc["Arguments"]]],
	_,
		Throw["CompileToLibrary implementation error: Unknown type Kind: ", assoc["Kind"]]
]
typeFromAssociation[atom_String] := atom
typeFromAssociation[val_Integer] := TypeFramework`TypeLiteral[val, "MachineInteger"]

End[]

EndPackage[]
