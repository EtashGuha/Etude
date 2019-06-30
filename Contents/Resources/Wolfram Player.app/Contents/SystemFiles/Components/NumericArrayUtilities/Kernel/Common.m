(* ::Package:: *)

(*******************************************************************************

Common: global variables plus common utility functions

*******************************************************************************)

Package["NumericArrayUtilities`"]

PackageImport["GeneralUtilities`"]
PackageImport["Developer`"]

(******************************************************************************)
(****** Library Loader ******)

PackageExport["$NAULib"]
PackageExport["LoadLibraries"]

LibraryLoadChecked[$Failed] = $Failed;
LibraryLoadChecked[e_] := Quiet[Check[LibraryLoad[e], $Failed]];

$LibraryResources = FileNameJoin[{ParentDirectory[DirectoryName @ $InputFileName], "LibraryResources", $SystemID}];

$NAULib::loadfail = "Failed to load NumericArrayUtilities library at ``.";
$NAULib::locate = "No NumericArrayUtilities library was found at ``.";

(* Load NumericArrayUtilities library *)
LoadLibraries[] := (
	Clear[$NAULib, LoadLibraries];
	If[!MemberQ[$LibraryPath, $LibraryResources], AppendTo[$LibraryPath, $LibraryResources]];
	(* Check whether library exists *)
	If[!StringQ[$NAULib], $NAULib = FindLibrary["NumericArrayUtilities"]];
	If[FailureQ[$NAULib], Message[$NAULib::locate, $LibraryResources]; Return[$Failed]];
	If[FailureQ @ LibraryLoadChecked[$NAULib], Message[$NAULib::loadfail, $NAULib]; Return[$Failed]];
	$NAULib
);

$NAULib := LoadLibraries[];

(******************************************************************************)
(* Checked Library Function Loading *)
CheckedLibraryFunctionLoad[name_, args___] :=
	Replace[
		LibraryFunctionLoad[$NAULib, name, args],
		Except[_LibraryFunction] :> 
			Panic["NumericArrayUtilitiesFunctionLoadError", 
				"Couldn't load `` from NumericArrayUtilities.", name
				]
	];

PackageScope["DeclareLibraryFunction"]
SetAttributes[DeclareLibraryFunction, HoldFirst];

DeclareLibraryFunction[symbol_Symbol, name_String, args___] := 
	SetDelayed[symbol, Set[symbol, CheckedLibraryFunctionLoad[name, args]]];

DeclareLibraryFunction[___] := Panic["MalformedDeclareLibraryFunction"];


(******************************************************************************)
PackageScope["NAUInvoke"]

General::nauerror = "Numeric Array Utilities encountered an error: ``";

NAUInvoke[func_, args___] :=
    Replace[
        func[args], 
        {
            _LibraryFunctionError :> (
                Panic["NumericArrayUtilitiesError"];
            ),
            _LibraryFunction[___] :> (
                Panic["LibraryFunctionUnevaluated", "Library function `` with args `` did not evaluate.", func[[2]], {args}]
            )
        }
    ];

(******************************************************************************)
PackageExport["ReturnErrorString"]
PackageExport["PopErrorString"]
DeclareLibraryFunction[ReturnErrorString, "return_error_string",
	{},
	"UTF8String"
]
DeclareLibraryFunction[PopErrorString, "pop_error_string",
	{},
	"UTF8String"
]
