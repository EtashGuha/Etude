(* Mathematica Test File *)

BeginPackage["GPUTools`Utilities`"]

GPUTools`Utilites`ByteCount::usage = "GPUTools`Utilites`ByteCount[elem] byte count of element"
GPUTools`Utilities`TimingBlock::usage = "GPUTools`Utilities`TimingBlock[outputFunction, runCode]"
GPUTools`Internal`TimingFunction::usage "GPUTools`Internal`TimingFunction function used to time CUDA code."

GPUTools`Utilities`ArrayType::usage = "GPUTools`Utilities`ArrayType gets the type of the input array."
GPUTools`Utilities`ArrayDim::usage = "GPUTools`Utilities`ArrayType gets the dimensions of the input array."
GPUTools`Utilities`ArrayDimMatchQ::usage = "GPUTools`Utilities`ArrayDimMatchQ checks if two lists have matching dimensions."

GPUTools`Utilities`ValidLibraryFunctionReturnQ::usage = "GPUTools`Utilities`VValidLibraryFunctionReturnQret] checks if the return value from LibraryLink is valid."

GPUTools`Utilities`LibraryGetDirectory::usage = "GPUTools`Utilities`LibraryGetDirectory[ver] gets the Library directory from LibraryVersionInformation"
GPUTools`Utilities`LibraryGetMajorVersion::usage = "GPUTools`Utilities`LibraryGetMajorVersion[ver] gets the major version from LibraryVersionInformation" 
GPUTools`Utilities`LibraryGetMinorVersion::usage = "GPUTools`Utilities`LibraryGetMinorVersion[ver] gets the minor version from LibraryVersionInformation" 
GPUTools`Utilities`LibraryGetRevisionVersion::usage = "GPUTools`Utilities`LibraryGetRevisionVersion[ver] gets the revision version from LibraryVersionInformation" 
GPUTools`Utilities`LibraryVersionInformationString::uasge = "GPUTools`Utilities`LibraryVersionInformationString[ver] gets the Library version information as a string"

GPUTools`Utilities`OptionsCheck::usage = "GPUTools`Utilities`OptionsCheck[fn, opts, lhs] checks for any invalid options passed to fn"

GPUTools`Internal`LibraryFunctionSafeLoad::usage = "GPUTools`Internal`LibraryFunctionSafeLoad  "

GPUTools`Internal`LibrariesSafeLoad::usage = "GPUTools`Internal`LibrariesSafeLoad  "

GPUTools`Utilities`DefineMessage::usage = "GPUTools`Utilities`DefineMessage[symbols, tag, messageText]"

GPUTools`Utilities`PrettyPrintMemory::usage = "GPUTools`Utilities`PrettyPrintMemory[mem] pretty prints memory in terms of GB, MB, or KB"

GPUTools`Utilities`Logger::usage = "GPUTools`Utilities`Logger[fun, msg] logs message using fun."

GPUTools`Utilities`VerboseLogger::usage = "GPUTools`Utilities`Logger[fun, msg] logs message using fun."

GPUTools`Utilities`LogPrinter::usage = "GPUTools`Utilities`LogPrinter[msg] prints log messages."

GPUTools`Utilities`VerboseLogPrinter::usage = "GPUTools`Utilities`LogPrinter[msg] prints log messages."

GPUTools`Utilities`IsSandboxedQ::usage = "GPUTools`Utilities`IsSandboxedQ[msg] checks if you are in a sandboxed environment"
 
Begin["`Private`"]

Needs["LibraryLink`"]

GPUTools`Utilites`ByteCount[x:(_List | _Integer | _Real)] :=
	ByteCount[x]

SetAttributes[GPUTools`Utilities`TimingBlock, HoldRest]
GPUTools`Utilities`TimingBlock[outputFunction_, fun_] :=
	If[outputFunction === None,
		fun,
		Module[{res},
			res = GPUTools`Internal`TimingFunction[fun];
			If[Length[res] == 2,
				outputFunction[First@res];
				res[[2]],
				$Failed
			]
		]
	]

ClearAll[GPUTools`Utilities`ValidLibraryFunctionReturnQ]
GPUTools`Utilities`ValidLibraryFunctionReturnQ[$Failed] = False
GPUTools`Utilities`ValidLibraryFunctionReturnQ[_LibraryFunction] := False
GPUTools`Utilities`ValidLibraryFunctionReturnQ[_LibraryFunctionError] := False
GPUTools`Utilities`ValidLibraryFunctionReturnQ[_] := True
		
	
GPUTools`Utilities`ArrayType[e_?Developer`PackedArrayQ] := 
	Switch[Part[e, Apply[Sequence, ConstantArray[1, ArrayDepth[e]]]],
		_Integer, Integer,
		_Real, Real,
		_, Complex
	]
GPUTools`Utilities`ArrayType[_List, 0] :=
	Symbol 
GPUTools`Utilities`ArrayType[e_?ListQ, n_:1] := 
	Which[
		MemberQ[e, _Complex, Infinity],	Complex,
		MemberQ[e, _Real, Infinity],	Real,
		MemberQ[e, _Integer, Infinity],	Integer,
		True, GPUTools`Utilities`ArrayType[N[e], n-1]
	]
GPUTools`Utilities`ArrayType[_Image] := Integer
GPUTools`Utilities`ArrayType[___] := Symbol;

GPUTools`Utilities`ArrayDim[e_?Developer`PackedArrayQ] := Dimensions[e]
GPUTools`Utilities`ArrayDim[e_List] := Dimensions[e]
GPUTools`Utilities`ArrayDim[e_Image] := Flatten[Join[{ImageDimensions[e]}, {ImageChannels[e]}]]
GPUTools`Utilities`ArrayDim[___] := False

GPUTools`Utilities`ArrayDimMatchQ[ a_, b_] := GPUTools`Utilities`ArrayDim[a] === GPUTools`Utilities`ArrayDim[b] 

GPUTools`Utilities`OptionsCheck[fn_, opts_List, lhs_] := 
	Module[{fnOpts, leftover},
		fnOpts = FilterRules[opts, Options[fn]];
		leftover = Complement[opts, fnOpts];
		If[MatchQ[leftover, {_Rule ..}],
			Message[fn::optx, leftover[[1,1]], HoldForm[lhs]];
			$Failed,
			Null
		]
	]

$LibraryExtension =
	Switch[$OperatingSystem,
		"Windows",
			".dll",
		"Unix",
			".so",
		"MacOSX",
			".dylib",
		_,
			".so"
	]

GPUTools`Internal`TimingFunction = AbsoluteTiming

(* ::Section:: *)
(* LibraryLoading Functions *)

If[!ListQ[GPUTools`Internal`LoadedLibraries],
	GPUTools`Internal`LoadedLibraries = {}
]

LibraryFunctionSafeLoad[libName_String, funName_String, inputArgs_List, outputArgs_] :=
	Module[{res = Quiet[LibraryFunctionLoad[libName, funName, inputArgs, outputArgs]]},
		If[Head[res] =!= LibraryFunction,
			GPUTools`Utilities`Logger["Failed to load libraryfunction=", funName, " from libraryfile=", libName, "."];
			Throw[$Failed],
			GPUTools`Utilities`VerboseLogger["Loading libraryfunction=", funName, " from libraryfile=", libName, "."];
			If[FreeQ[GPUTools`Internal`LoadedLibraries, FindLibrary[libName]],
				PrependTo[GPUTools`Internal`LoadedLibraries, FindLibrary[libName]]
			];
			res
		]
	]
LibraryFunctionSafeLoad[varName_, libName_String, funName_String, inputArgs_List, outputArgs_] :=
	(varName = LibraryFunctionSafeLoad[libName, funName, inputArgs, outputArgs])
GPUTools`Internal`LibraryFunctionSafeLoad[libName_String, funName_String, inputArgs_List, outputArgs_] :=
	LibraryFunctionSafeLoad[libName, funName, inputArgs, outputArgs]
GPUTools`Internal`LibraryFunctionSafeLoad[varName_, libName_String, funName_String, inputArgs_List, outputArgs_] :=
	GPUTools`Internal`LibraryFunctionSafeLoad[{{varName, libName, funName, inputArgs, outputArgs}}]
GPUTools`Internal`LibraryFunctionSafeLoad[x_List] :=
	If[Catch[Scan[Apply[LibraryFunctionSafeLoad, #]&, x]] === $Failed,
		$Failed,
		True
	] 	

LibrariesSafeLoad[x_String] /; FreeQ[GPUTools`Internal`LoadedLibraries, x] := 
	If[Quiet[LibraryLoad[x]] === $Failed,
		GPUTools`Utilities`Logger["Failed to load libraryfile=", x, "."];
		Throw[$Failed],
		GPUTools`Utilities`VerboseLogger["Loading libraryfile=", x, "."];
		PrependTo[GPUTools`Internal`LoadedLibraries, x];
		True
	]
LibrariesSafeLoad[x_String] := True

GPUTools`Internal`LibrariesSafeLoad[Null] := False 
GPUTools`Internal`LibrariesSafeLoad[x_String] :=
	GPUTools`Internal`LibrariesSafeLoad[{x}]
GPUTools`Internal`LibrariesSafeLoad[x_List] :=
	If[Catch[Scan[LibrariesSafeLoad, x]] === $Failed,
		$Failed,
		True
	] 

If[$VersionNumber <= 9.0,
	GPUTools`Utilities`IsSandboxedQ[errHd_] :=
		TrueQ[Developer`CheckProtectedMode[errHd]],
	GPUTools`Utilities`IsSandboxedQ[errHd_] :=
		TrueQ[Developer`ProtectedModeBlockedQ[errHd]]
]

GPUTools`Utilities`LibraryGetDirectory[ver_] :=
	Quiet[Check[
		If[StringQ[ver],
			ver,
			If[TrueQ[Quiet[ver =!= $Failed && FileExistsQ["Name" /. ver]]],
				DirectoryName["Name" /. ver],
				$Failed
			]
		],
		$Failed
	]]
GPUTools`Utilities`LibraryGetMajorVersion[ver_] :=
	Quiet[Check[
		If[StringQ[ver],
			ver,
			If[ver === $Failed,
				$Failed,
				If[("MajorVersion" /. ver) === "MajorVersion",
					"Major Version" /. ver,
					"MajorVersion" /. ver
				]
			]
		],
		$Failed
	]]
GPUTools`Utilities`LibraryGetMinorVersion[ver_] :=
	Quiet[Check[
		If[StringQ[ver],
			ver,
			If[ver === $Failed,
				$Failed,
				If[("MinorVersion" /. ver) === "MinorVersion",
					"Minor Version" /. ver,
					"MinorVersion" /. ver
				]
			]
		],
		$Failed
	]]
GPUTools`Utilities`LibraryGetRevisionVersion[ver_] :=
	Quiet[Check[
		If[StringQ[ver],
			ver,
			If[ver === $Failed,
				$Failed,
				If[("RevisionNumber" /. ver) === "RevisionNumber",
					"Revision Number" /. ver,
					"RevisionNumber" /. ver
				]
			]
		],
		$Failed
	]]


GPUTools`Utilities`LibraryVersionInformationString[ver_] :=
	Quiet[Check[
		LibraryVersionString[ver],
		$Failed
	]]

(* ::Section:: *)
(* Pretty Print Memory *)

KiloByte = 1024
MegaByte = 1024 * KiloByte
GigaByte = MegaByte * 1024

PrettyN[x_] :=
	With[{str = ToString[N[x]]},
		If[StringTake[str, -1] === ".",
			str <> "0",
			str
		]
	]

GPUTools`Utilities`PrettyPrintMemory[mem_Integer] :=
	Which[
		mem >= GigaByte,
			PrettyN[mem/GigaByte] <> " GB",
		mem >= MegaByte,
			PrettyN[mem/MegaByte] <> " MB",
		mem >= KiloByte,
			PrettyN[mem/KiloByte] <> " KB",
		True,
			ToString[mem] <> "B"
	]

(* ::Section:: *)
(* Batch Message Definition *)

GPUTools`Utilities`DefineMessage[symbol_Symbol, tag_, text_] :=
	MessageName[Evaluate@symbol, tag] = text
GPUTools`Utilities`DefineMessage[symbols_List, tag_, text_] :=
	Do[
		MessageName[Evaluate@sym, tag] = text
		, {sym, symbols}
	]
GPUTools`Utilities`DefineMessage[symbols_List, tagTextList_List] :=
	Do[
		Do[
			MessageName[Evaluate@sym, tt[[1]]] = tt[[2]]
			, {tt, tagTextList}
		]
		, {sym, symbols}
	]
GPUTools`Utilities`DefineMessage[messages_List] :=
	Map[GPUTools`Utilities`DefineMessage, messages]
	
(* ::Section:: *)
(* Documentation *)

GPUTools`Message`MakeDocumentationLink[text_String, hyperlink_String] :=
	"\!\(\*\nButtonBox[\"" <> text <> "\",ButtonStyle->\"Link\",ButtonData:>\"paclet:" <> hyperlink <> "\"]\)"

(* ::Section:: *)
(* Documentation *)
	
GPUTools`Utilities`Logger[msg___] :=
	(
		If[GPUTools`Utilities`VerboseLogPrinter =!= Unevaluated[GPUTools`Utilities`VerboseLogPrinter],
			GPUTools`Utilities`LogPrinter = Print
		];
		GPUTools`Utilities`LogPrinter["LOG: ", msg]
	)
	
GPUTools`Utilities`VerboseLogger[msg___] :=
	GPUTools`Utilities`VerboseLogPrinter["LOG: ", msg]
	
End[]

EndPackage[]
