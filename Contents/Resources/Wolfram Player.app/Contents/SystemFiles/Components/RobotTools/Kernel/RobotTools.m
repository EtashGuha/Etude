(* ::Package:: *)

(* ::Title:: *)
(*RobotTools Package*)

(* ::Section:: *)
(*Annotations*)

(* :Title: RobotTools.m *)

(* :Author:
        Brenton Bostick
        brenton@wolfram.com
*)

(* :Package Version: 1.2 *)

(* :Mathematica Version: 7.0 *)

(* :Copyright: RobotTools source code (c) 2005-2007 Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:
   RobotTools is a package for automating keyboard, mouse, and screenshot usage.
   
   RobotTools generally follows the J/Link style for package layout, with several source files, a `Package` context, etc.
*)

(* ::Section:: *)
(*BeginPackage*)

BeginPackage["RobotTools`", {"JLink`"}]

(* ::Section:: *)
(*Information*)

`Information`$VersionNumber = 1.2

`Information`$ReleaseNumber = 0

`Information`$Version = "RobotTools Version 1.2.0"

`Information`CVS`$RobotToolsID = "$Id: RobotTools.wl,v 1.1 2014/08/27 20:49:59 carlosy Exp $"

(* ::Section:: *)
(*Messages*)

General::d60 =
"The delay must be between 0 and 60 seconds: `1`."

General::menupath =
"List of non-empty strings expected in position `1` of `2`."

General::nbobj =
"Notebook object expected in position `1` of `2`."

General::pnt =
"Point expected in position `1` of `2`."

General::pntl =
"Point or list of points expected in position `1` of `2`."

General::menumoved =
"That elusive Menu symbol has moved again!
That poor symbol still is not design reviewed!
To fix RobotTools:
1. scan all of the .m files in RobotTools for FrontEnd`.`Menu and replace them with System`.`Menu (or whatever the new context is)
2. Fix the code in RobotTools.m below the definition of this message to acommodate the new situation."

If[!NameQ["Menu"],
	Message[General::menumoved];
	Throw[$Failed]
]


(* ::Section:: *)
(*Package*)

Begin["`Package`"]

$RobotToolsDirectory

$RobotToolsTextResourcesDirectory

End[] (*`Package`*)

AppendTo[$ContextPath, "RobotTools`Package`"]

(* ::Section:: *)
(*Private*)

(* ::Subsection:: *)
(*Begin*)

Begin["`RobotTools`Private`"]

(* ::Subsection:: *)
(*$RobotToolsDirectory*)

$RobotToolsDirectory =
	DirectoryName[$InputFileName]

$RobotToolsTextResourcesDirectory =
	FileNameJoin[{$RobotToolsDirectory, "TextResources"}]

(* ::Subsection:: *)
(*$implementationFiles*)

$implementationFiles =
	{
		(* Patterns.m must be loaded before everything else, since all other files depend on it *)
		FileNameJoin[{$RobotToolsDirectory, "Patterns.m"}],
		FileNameJoin[{$RobotToolsDirectory, "AppleScript.m"}],
		FileNameJoin[{$RobotToolsDirectory, "CharacterData.m"}],
		FileNameJoin[{$RobotToolsDirectory, "Delay.m"}],
		FileNameJoin[{$RobotToolsDirectory, "Dialogs.m"}],
		FileNameJoin[{$RobotToolsDirectory, "FrontEnd.m"}],
		FileNameJoin[{$RobotToolsDirectory, "InstallRobotTools.m"}],
		FileNameJoin[{$RobotToolsDirectory, "Keyboard.m"}],
		FileNameJoin[{$RobotToolsDirectory, "Menu.m"}],
		FileNameJoin[{$RobotToolsDirectory, "Mouse.m"}],
		FileNameJoin[{$RobotToolsDirectory, "RobotExecute.m"}],
		FileNameJoin[{$RobotToolsDirectory, "Scaling.m"}],
		FileNameJoin[{$RobotToolsDirectory, "ScreenShot.m"}]
	}

(* ::Subsection:: *)
(*$processDecls*)

processDecls[file_] :=
	Module[{strm, e, moreLines = True},
		strm = OpenRead[file];
		If[Head[strm] =!= InputStream,
			Return[$Failed]
		];
		While[moreLines,
			e = Read[strm, Hold[Expression]];
			ReleaseHold[e];
			If[e === $Failed || MatchQ[e, Hold[_End]],
				moreLines = False
			]
		];
		Close[file]
	]

(* ::Subsection:: *)
(*End*)

End[] (*`RobotTools`Private`*)

(* ::Section:: *)
(*Implementation*)

Scan[`RobotTools`Private`processDecls, `RobotTools`Private`$implementationFiles]

Scan[Get, `RobotTools`Private`$implementationFiles]

Needs["RobotTools`CaptureScreenshot`"]

(* ::Section:: *)
(*EndPackage*)

EndPackage[]
