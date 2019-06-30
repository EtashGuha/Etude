(* ::Package:: *)

(*******************************************************************************

Common functions + Global Variables

*******************************************************************************)

Package["LightGBMLink`"]

PackageImport["GeneralUtilities`"]

PackageScope["$lightGBMLinkLib"]

(******************************************************************************)
(**** Global Variables ******)

(******************************************************************************)
(****** Library Loader ******)

$lightGBMLinkLib::loadfail = "Failed to load LightGBMLink library at ``.";
$lightGBMLinkLib::locate = "Failed to locate LightGBMLink library at ``.";

LibraryLoadChecked[e_] := Quiet[Check[LibraryLoad[e], $Failed]];

$libraryResources = FileNameJoin[{
        ParentDirectory[DirectoryName @ $InputFileName], 
        "LibraryResources", $SystemID}];

If[!MemberQ[$LibraryPath, $libraryResources],
   AppendTo[$LibraryPath, $libraryResources];
];

LoadLibraries[] := (
	If[!FileExistsQ[$libraryResources], 
           Message[$lightGBMLinkLib::locate, $libraryResources];
           Return[$Failed]];
        Clear[$lightGBMLib];
	$lightGBMLib = FindLibrary["lib_lightgbm"];
	If[FailureQ @ LibraryLoadChecked[$lightGBMLib], 
           Message[$lightGBMLib::loadfail, $lightGBMLib];
           Return[$Failed]];
	Clear[$lightGBMLinkLib];
	$lightGBMLinkLib = FindLibrary["LightGBMLink"];
	If[FailureQ @ LibraryLoadChecked[$lightGBMLinkLib], 
           Message[$lightGBMLinkLib::loadfail, $lightGBMLinkLib];
           Return[$Failed]];
        );

$lightGBMLinkLib = (LoadLibraries[]; $lightGBMLinkLib);


(******************************************************************************)
(****** Load Library Functions ******)

PackageExport["LGBMGetLastError"]

LGBMGetLastError =
LibraryFunctionLoad[$lightGBMLinkLib,
                    "WL_GetLastError", 
	            {} , 
	            "UTF8String"
];	

(******************************************************************************)
(* Error handling: all LightGBM libs should be called with LGBMInvoke *)

PackageScope["LGBMInvoke"]

General::lgbmerr = "LightGBM encountered an error: ``";

LGBMInvoke[func_, args___] := Module [
	{args2, result},
	args2 = {args} /. (mobj_LGBMDataset | mobj_LGBMBooster) :> ManagedLibraryExpressionID[mobj];
	result = Apply[func, args2];
	Match[result,
	      _LibraryFunctionError :> (
		      lastError = LGBMGetLastError[];
		      ThrowFailure["lgbmerr", lastError];
	       ),
	      _LibraryFunction[___] :> (
		      Panic["LibraryFunctionUnevaluated"]
	       ),
              result
	]
]; 

LibraryFunctionFailureQ[call_] :=
	If[Head@call === LibraryFunctionError, True, False]
	
LGBMErrorCheck[parentSymbol_, result_] :=
        If[
	        LibraryFunctionFailureQ@result, 
	        parentSymbol::mxnetError = MXFailureDescription[];
	        Panic[parentSymbol::mxnetError]
        ];

(******************************************************************************)

