Package["MXNetLink`"]


(* NOTE: This file should be loaded first via Get. According to new-style
package semantics, the other files will be loaded in alphabetical order. *)

(******************************************************************************)

PackageScope["DeclarePostloadCode"]

SetAttributes[DeclarePostloadCode, HoldAll];
$postloadCode = Internal`Bag[];
DeclarePostloadCode[expr_] := Internal`StuffBag[$postloadCode, Unevaluated[expr]];

(******************************************************************************)

PackageImport["GeneralUtilities`"]
PackageImport["Developer`"]
PackageImport["PacletManager`"]

(******************************************************************************)

PackageScope["$MXNetLibraryResourcesPath"]

$MXNetLibraryResourcesPath::usage = "$MXNetLibraryResourcesPath determines where to various library resources related to MXNet are located. It will be set automatically during paclet loading, or can be set in advance."

(******************************************************************************)

PackageScope["$MXNetLibraryPath"]

$MXNetLibraryPath::usage = "$MXNetLibraryPath determines where to the MXNet library is located. It will be set automatically during paclet loading, or can be set in advance."

(******************************************************************************)

PackageScope["$MXNetLinkLibraryPath"]

$MXNetLinkLibraryPath::usage = "$MXNetLinkLibraryPath determines where to the MXNetLink library is located. It will be set automatically during paclet loading, or can be set in advance."

(******************************************************************************)

PackageExport["$ExpectedMXNetVersion"]

$ExpectedMXNetVersion = 10400;
$ExpectedMXNetVersion::usage = "$ExpectedMXNetVersion should be set to the version of MXNet that is expected to be loaded. If this version is not found, the loading will be aborted."

(******************************************************************************)

PackageExport["$MXRestartRequiredQ"]

(* reloading the paclet shouldn't reset this value, only restarting the kernel should *)
If[!ValueQ[$MXRestartRequiredQ], $MXRestartRequiredQ = False];
$MXRestartRequiredQ::usage = "$MXRestartRequiredQ is True if the WL kernel needs to be restarted so MXNetLink can load the GPU libmxnet version.";

(******************************************************************************)

PackageExport["$MXGPUEnabledQ"]

$MXGPUEnabledQ::usage = "$MXGPUEnabledQ is True if MXNetResources has been downloaded, False otherwise."

(******************************************************************************)

PackageExport["$MXForceCPU"]

$MXForceCPU::usage = "$MXForceCPU should be set to True if the MXNetResources paclet (if any) should be ignored, and the CPU-only library used instead.";

(******************************************************************************)

$MXResourcesName = "MXNetResources";

(******************************************************************************)

PackageExport["MXNetResourcesInstall"]

DeclarePostloadCode[
General::mxrescorr = "The GPU library paclet is corrupted.";
General::mxdloff = "The Wolfram Language is currently configured not to use the internet and could not download the GPU library paclet. To allow internet access, use Help > Internet Connectivity...";
General::mxresnotavail = "Could not find the GPU library paclet on the Wolfram server.";
General::mxresnoinst = "Installing paclets is disabled in this Wolfram Language session. This feature is required to install GPU libraries.";
General::mxresdlfail = "An unknown error occurred while downloading the GPU library paclet.";
General::trgrestart = "TargetDevice -> \"GPU\" requires a restart of your Wolfram Language session.";
];

Options[MXNetResourcesInstall] = {"AllowUpdate" -> False}

SetUsage @ "
MXNetResourcesInstall[head] downloads and installs the MXNetResources paclet. \
It returns True if it succeeds, and False otherwise. If the paclet already exists, \
no download will take place, even if an updated version is on the paclet server. \
It has a side-effect of setting $MXRestartRequiredQ to True when successful.
The following options are available:
|'AllowUpdate'| False | If True, will download an update of the paclet if available. |"

MXNetResourcesInstall[head_, OptionsPattern[]] := Scope[
	UnpackOptions[allowUpdate];

	Quiet[
		$MessageList = {};
		paclet = UpdatePacletWithProgress[$MXResourcesName, "Downloading GPU libraries...", "AllowUpdate" -> allowUpdate];
		messages = $MessageList;
	];

	If[Head[paclet] =!= Paclet,
		Which[
			(* if a paclet doesn't exist on remote, getPacletWithProgress issues PacletManager::dlfail *)
			(* this should never be encountered by users, but very useful for debugging if they do *)
 			($AllowInternet === False) || MemberQ[messages, HoldForm @ PacletManager::dloff],
				Message[head::mxdloff],
			MemberQ[messages, HoldForm @ PacletManager::rdonly],
				Message[head::mxresnoinst],				
			Length[PacletFindRemote @ $MXResourcesName] == 0,
				Message[head::mxresnotavail],
			True,
				Message[head::mxresdlfail]
		];
		Return[False];
	];

	If[!VerifyPaclet[paclet],
		Message[head::mxrescorr];
		PacletUninstall[paclet];
		Return[False];
	];

	(* set global variable *)
	Message[head::trgrestart];
	$MXRestartRequiredQ ^= True;
	True
]

(******************************************************************************)

PackageExport["$LibrariesLoaded"]

$LibrariesLoaded = False;
$LibrariesLoaded::usage = "$LibrariesLoaded will be False if libraries have not been loaded, True if they have, and $Failed if an attempt was made that failed."

(******************************************************************************)

PackageExport["MXNetLink`Bootstrap`LoadLibraries"]

LoadLibraries[] := If[$LibrariesLoaded =!= True,
	$LibrariesLoaded = $Failed; 
	(* ^ we'll set this to True after we're done, early return will keep it $Failed *)

	(* find MXNetLink library *)
	$libExt = Switch[$OperatingSystem, "Windows", "dll", "MacOSX", "dylib", "Unix", "so"];
	If[!StringQ[$MXNetLinkLibraryPath], $MXNetLinkLibraryPath = FileNameJoin[{
			ParentDirectory[DirectoryName @ $InputFileName], 
			"LibraryResources", $SystemID, "MXNetLink." <> $libExt
		}]
	];
	If[!FileExistsQ[$MXNetLinkLibraryPath],
		libMissingFail["MXNetLink library", $MXNetLinkLibraryPath];
		Return @ $Failed
	];

	$originalLibError = LibraryLink`$LibraryError;
	(* ^ allows us to only report new errors *)
	(* If the MXNetResources paclet exists on users machine, always use it *)
	$MXNetResourcesPaclet = PacletFind[$MXResourcesName];
	If[$MXNetResourcesPaclet =!= {} && $MXForceCPU =!= True,
		(* first is the latest compatible paclet if multiple are installed *)
		$MXNetResourcesPaclet = First[$MXNetResourcesPaclet];
		If[!ValueQ[$MXNetLibraryResourcesPath],
			$MXNetLibraryResourcesPath = 
				FileNameJoin[{Lookup[PacletInformation[$MXNetResourcesPaclet], "Location"], "LibraryResources", $SystemID}]
		];
		$MXGPUEnabledQ = True;
	,
		If[!ValueQ[$MXNetLibraryResourcesPath],
			$MXNetLibraryResourcesPath = FileNameDrop[$MXNetLinkLibraryPath];
		];
		$MXGPUEnabledQ = False;
	];

	If[!FileExistsQ[$MXNetLibraryResourcesPath], 
		libMissingFail["MXNet library directory", $MXNetLibraryResourcesPath];
		Return @ $Failed
	];

	(* explicitly load CUDA dependencies for 64-bit Windows, (32-bit doesn't support GPU training) *)
	If[$SystemID === "Windows-x86-64",
		$WinLibs = FileNames[
			{"cublas*", "cufft*", "cudart*", "curand*", "nvrtc*", "cusolver*"}, 
			$MXNetLibraryResourcesPath
		];
		$Win10CUDNN = FileNameJoin[{
			$MXNetLibraryResourcesPath, 
			If[Windows10Q[], "win10", Nothing]
		}];
		$WinLibs = Join[FileNames["cudnn*", $Win10CUDNN, 1], $WinLibs]; 
		Scan[LibraryLoad, $WinLibs];
	];
	
	(* load MXNet *)
	If[!StringQ[$MXNetLibraryPath],
		(* TODO: work with RE to unify these to just libmxnet.XXX *)
		$mxname = Switch[$OperatingSystem,
			"Windows", "mxnet.dll",
			"Unix", "libmxnet.so",
			"MacOSX", "libmxnet.dylib"
		];
		$MXNetLibraryPath = FileNameJoin[{$MXNetLibraryResourcesPath, $mxname}];
	];
	If[!FileExistsQ[$MXNetLibraryPath],
		libMissingFail["MXNet library", $MXNetLibraryPath];
		Return @ $Failed
	];
	If[FailureQ @ CheckedLibraryLoad @ $MXNetLibraryPath,
		libLoadFail["MXNet library",  $MXNetLibraryPath];
		Return @ $Failed
	];

	If[FailureQ @ CheckedLibraryLoad @ $MXNetLinkLibraryPath,
		libLoadFail["MXNetLink library",  $MXNetLinkLibraryPath];
		Return @ $Failed
	];

	(* sanity test: check we can load at least one function, and that we are seeing the right version of mxnet *)
	getVersionFunc = Quiet @ Check[LibraryFunctionLoad[$MXNetLinkLibraryPath, "mxlMXGetVersion", {}, _Integer], $Failed];
	If[getVersionFunc === $Failed,
		libLoadFail["MXNetLink library", $MXNetLinkLibraryPath];
		Return @ $Failed
	];
	actualVersion = getVersionFunc[];
	If[actualVersion =!= $ExpectedMXNetVersion, 
		Message[General::nnwrongver, actualVersion, $ExpectedMXNetVersion];
	];

	$LibrariesLoaded = True;
];

(******************************************************************************)

libMissingFail[name_, path_] := 
	Message[General::nnlibmiss, name, path];

libLoadFail[name_, path_] := Scope[
	linkError = LibraryLink`$LibraryError;
	If[StringQ[linkError] && StringFreeQ[linkError, "WolframCompileLibrary_wrapper"] && linkError =!= $originalLibError,
		Message[General::nnlibload2, name, path, linkError],
		Message[General::nnlibload1, name, path]
	];
];

defineFailureMessages[] := (
	General::nnlibmiss = "The ``, required by the neural network runtime, cannot be found at the expected location \"``\".";
	General::nnlibload1 = "The ``, required by the neural network runtime, cannot be loaded from \"``\".";
	General::nnlibload2 = "The ``, required by the neural network runtime, cannot be loaded from \"``\".\nThe operating system reported the following error:\n``";
	General::nnwrongver = "The wrong version of the neural network runtime was loaded (loaded version ``, expected version ``). Change the variable $ExpectedMXNetVersion in LibraryLink.m.";
	General::nnlibunavail = "The neural network runtime is unavailable.";
);

(* we need these to be defined in time for loading *)
defineFailureMessages[];

DeclarePostloadCode[
	defineFailureMessages[]
]

(******************************************************************************)

Windows10Q[] := Windows10Q[] = Module[
	{output, version},
	If[$OperatingSystem =!= "Windows", Return[False]];
	If[MatchQ[$LicenseType, "Player" | "Player Pro"],
		(* see 333787 -- Import of pipe / processlink doesn't work here, 
		but player doesn't work in standalone mode so using FE is okay. *)
		output = Quiet @ SystemInformation["FrontEnd", "OperatingSystemVersion"],
		output = Quiet @ Import["!cmd /c ver", "String"]
	];
	If[!StringQ[output], Return[$Failed]];
	version = First[StringCases[output, DigitCharacter..], ""];
	StringStartsQ[version, "10"]
];

(******************************************************************************)

CheckedLibraryLoad[e_] := Quiet[Check[LibraryLoad[e], $Failed], LibraryFunction::load];

(******************************************************************************)

CheckedLibraryFunctionLoad[name_, {largs___, "Array", rargs___}, ret_] := 
	With[{n = Length[{largs}] + 1, narrayq = NumericArrayQ, 
	 freal = CheckedLibraryFunctionLoad[name, {largs, {Real, _, "Constant"}, rargs}, ret],
	 fint  = CheckedLibraryFunctionLoad[name, {largs, {Integer, _, "Constant"}, rargs}, ret],
	 farr  = CheckedLibraryFunctionLoad[name, {largs, {"RawArray", "Constant"}, rargs}, ret]},
		Function[
			Which[
				NumericArrayQ[Slot[n]], farr,
				Developer`PackedArrayQ[Slot[n], Real], freal,
				Developer`PackedArrayQ[Slot[n], Integer], fint,
				ArrayQ[Slot[n], _, NumberQ], freal,
				True, LibraryFunctionError["NotAnArray", 6]&
			][##]
		]
	]

CheckedLibraryFunctionLoad[name_, args_, "Tensor"] := With[
	{fun = CheckedLibraryFunctionLoad[name, args, {Real, _}]},
	Replace[
		Quiet[fun[##], LibraryFunction::numerr],
		LibraryFunctionError[_, 4] :> mxlGetTempIntegerTensor[]
	]&
]

$failingStubFunction = Function[ThrowFailure["nnlibunavail"]];
(* ^ this function is used when we can't load any of the NN libraries:
calling NN functions will trigger this stub and the entire function will abort,
allowing the rest of the system to continue mostly as normal *)

CheckedLibraryFunctionLoad[name_, args_, ret_] /; ($LibrariesLoaded === $Failed) := $failingStubFunction;
(* ^ if we failed to load the library itself, the function load will fail too, so don't try, just produce a stub *)

CheckedLibraryFunctionLoad[name_, args_, ret_] := Quiet[
	Replace[
		Quiet[LibraryFunctionLoad[$MXNetLinkLibraryPath, name, args, ret], LibraryFunction::libload],
		e:Except[_LibraryFunction] :> (
			$LibrariesLoaded = $Failed;
			libLoadFail[StringForm["function \"``\"", name], $MXNetLinkLibraryPath]; 
			$failingStubFunction
		)
	],
	LibraryFunction::overload
];

(******************************************************************************)

PackageScope["mxlDeclare"]

SetUsage @ "
mxlDeclare[symbol$, args$, return$] loads a function from MXNetLink into the given symbol.
* The function must have the same name as the symbol. 
	* Any suffix '$xxx' in the symbol name will be dropped, which allows type overloading.
* The arguments can be of the following forms. This table includes also the MXArgMan \
methods that can be used to read the value:
| mxl type | WL expression | C++ getter |
| --- | --- | --- |
| 'Boolean' | True or False | getBoolean |
| 'Integer' | single integer | getInteger |
| 'Real' | single real number | getReal |
| 'IntegerVector', 'RealVector' | list of numbers | getUnifiedArray, getVector<t$>, getTensor, getMTensor |
| 'IntegerMatrix', 'RealMatrix' | matrix of numbers | getUnifiedArray, getTensor<t$>, getMTensor, |
| 'IntegerTensor', 'RealTensor' | tensor of integers (any rank) | getUnifiedArray, getTensor<t$>, getMTensor |
| 'NumericArray' | NumericArray | getUnifiedArray, getMRawArray |
| 'Array' | NumericArray or packed array | getUnifiedArray |
| 'String' | a string | getString, getCString |
* The special type 'Array' will result in a multiplexed library function \
that can handle any form of array. Multiple arguments can use 'Array'.
* The return value can be any of the above, as well as 'Void' (or just leave \
off the return value).
* setUnifiedArray allows a NumericArray or packed array of any type to be returned \
but overloads should then be used to call the right form for what kind of array \
will be returned.
* The following values do not have their own types, but they do have special 
WL functions to encode them to be passed as arguments, and MXArgMan methods to get them:
| WL expression | passed as | WL encoder | C++ getter |
| --- | --- | --- | --- |
| list of strings | 'String' | mxlPackStringVector | getCStringVector |
| single NDArray | 'Integer' | MLEID | getNDArray |
| single MXSymbol | 'Integer' | MLEID | getMXSymbol |
| single MXExecutor | 'Integer' | MLEID | getMXExecutor |
| single KVStore | 'Integer' | MLEID | getKVStore |
| single Optimizer | 'Integer' | MLEID | getOptimizer |
| list of NDArrays | 'IntegerVector' | Map[MLEID] | getNDArrayVector |
| list of MXSymbols | 'IntegerVector' | Map[MLEID] | getMXSymbolVector |
* The following values are the opposite: they have methods of MXargMan to set them as
return values, and WL functions to decode them once returned:
| WL expression | returned as | C++ setter | WL decoder |
| --- | --- | --- | --- |
| list of strings | 'String' | setStringVector | mxlUnpackStringVector |
| JSON-like data | 'String' | setJSON | ReadRawJSONString |
"

$lfArgTypes = {
	"Boolean" -> True|False,
	"Integer" -> Integer, "IntegerVector" -> {Integer, 1, "Constant"}, "IntegerMatrix" -> {Integer, 2, "Constant"}, "IntegerTensor" -> {Integer, _, "Constant"},
	"Real" -> Real, "RealVector" -> {Real, 1, "Constant"}, "RealMatrix" -> {Real, 2, "Constant"}, "RealTensor" -> {Real, _, "Constant"},
	"NumericArray" -> {"RawArray", "Constant"},
	"Array" -> "Array", (* this type is resolved by CheckedLibraryFunctionLoad *)
	"Void" -> "Void",
	"String" -> "UTF8String",
	s_ :> Panic["InvalidLibArgType", "``: `` is not one of the valid library argument types, which are ``.", $libfname, s, Most @ Keys @ $lfArgTypes]
};

toArgType[e_] := Replace[e, $lfArgTypes];

toRetType["SharedNumericArray"] = {"RawArray", "Shared"}; (* unused i think *)
toRetType["Array"] := Panic["InvalidLibRetType", "``: Array is not allowed.", $libfname];
toRetType["Tensor"] := "Tensor";
toRetType[e_] := toArgType[e] /. "Constant" -> Automatic;

SetAttributes[mxlDeclare, HoldFirst];

mxlDeclare[symbol_, args_] := 
	mxlDeclare[symbol, args, "Void"];

mxlDeclare[symbol_Symbol, arg_String, ret_] := 
	mxlDeclare[symbol, {arg}, ret];

mxlDeclare[symbol_Symbol, args_List, ret_] := (Clear[symbol]; With[{name = StringExtract[SymbolName[symbol], "$" -> 1]}, Block[{$libfname = name},
	symbol := symbol = CheckedLibraryFunctionLoad[name, Map[toArgType, args], toRetType[ret]]
]]);

mxlDeclare[mxlGetTempIntegerTensor, {}, "IntegerTensor"];

(******************************************************************************)

PackageScope["OnLibError"]

SetHoldAll[OnLibError];

DefineMacro[OnLibError,
OnLibError[else_, libf_] := Quoted @ If[Internal`UnsafeQuietCheck[libf] =!= Null, else]
]

(******************************************************************************)

PackageScope["tryCall"]

tryCall[failf_, data_, _LibraryFunctionError] := failf[data];
tryCall[_, _, res_] := res;

(******************************************************************************)

PackageScope["mxlCall"]

DeclarePostloadCode[
General::mxneterr = "MXNet encountered an error: ``";
]

mxlCall[func_, args___] :=
	Replace[
		If[$MXNetLogger =!= Hold, log[func, args]];
		func[args], 
		{
			_LibraryFunctionError :> MXLibraryError[],
			_LibraryFunction[___] :> Panic["LibraryFunctionUnevaluated", 
				"Library function `` with args `` did not evaluate.", toLibForm @ func, {args}]
		}
	];

toLibForm[lib_] := FirstCase[lib, lb_LibraryFunction :> Part[lb, 2], lib, {0, Infinity}, Heads -> True];

log[func_, args___] := 
	$MXNetLogger[toLibForm[func][args] /. {
		l_List  /; ByteCount[l] > 256 :> elideArray[l],
		na_NumericArray :> elideArray[na]
	}];

elideArray[l_List] := If[ArrayQ[l, 1|2|3, MachineIntegerQ] && LeafCount[l] < 20, l,
	(AngleBracket @@ Dimensions[l])];

elideArray[na_NumericArray] := 
	"NumericArray"[AngleBracket @@ Dimensions[na], NumericArrayType[na]];

DeclarePostloadCode[
General::netseqlen = "A sequence was provided that was less than the minimum required length.";
]

(******************************************************************************)

PackageScope["mxParameterToString"]

mxParameterToString[l_List] := StringJoin["(", Riffle[Map[mxParameterToString, l], ","], ")"];
mxParameterToString[True] := "true";
mxParameterToString[False] := "false";
mxParameterToString[e_Real | e_Rational] := HighPrecisionDoubleString[e];
mxParameterToString[e_Integer] := If[Negative[e], "-" <> IntegerString[e], IntegerString[e]];
mxParameterToString[e_String] := e;
mxParameterToString[e_] := Panic["Unserializable", "`` is unserializable.", e];

(******************************************************************************)

PackageScope["mxlPackStringVector"]
PackageScope["mxlUnpackStringVector"]

(* note, this encoding has the weakness that it does support empty strings.
don't use it if you require this feature *)

(* this is used to combine lists of strings into a single string,
for use with MXArgMan::getStringVector *)
mxlPackStringVector[list_] := StringJoin @ Riffle[list, "\:0001"];
mxlPackStringVector[{}] := "";

(* this does the opposite, and works with results produced via
MXArgMan::setStringVector *)
mxlUnpackStringVector[str_] := StringSplit[str, "\:0001"];
mxlUnpackStringVector[""] := {};

(******************************************************************************)

PackageScope["MLEID"]

DefineAlias[MLEID, ManagedLibraryExpressionID]

(******************************************************************************)

PackageScope["WriteCompactJSON"]

WriteCompactJSON[expr_] := Developer`WriteRawJSONString[expr, "Compact" -> True];

(******************************************************************************)

PackageExport["$SetupUpValues"]

$SetupUpValues::usage = "$SetupUpValues, if set to True when MXNetLink loads, will \
create upvalues for NDArrays and MXSymbols."

(******************************************************************************)

PackageExport["MXNetLink`Bootstrap`RunPostloadCode"]

RunPostloadCode[] := Internal`BagPart[$postloadCode, All, List];

(******************************************************************************)

PackageExport["MXNetLink`Bootstrap`SetupNullHandles"]

PackageExport["$NullNDArray"]
PackageExport["$NullExecutor"]

MXNetLink`Bootstrap`SetupNullHandles[] := (
	MXGetLastError[]; (* <- it appears to be a bug in LibraryLink, but we have to call
	one function before we can setup managed expressions *)
	$NullNDArray = CreateManagedLibraryExpression["NDArray", NDArray]; 
	$NullExecutor = CreateManagedLibraryExpression["MXExecutor", MXExecutor];
);