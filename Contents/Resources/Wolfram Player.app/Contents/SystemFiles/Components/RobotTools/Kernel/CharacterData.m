(* ::Package:: *)

(* ::Title:: *)
(*CharacterData*)

(* ::Section:: *)
(*Annotations*)

(* :Title: CharacterData.m *)

(* :Author:
        Brenton Bostick
        brenton@wolfram.com
*)

(* :Package Version: 1.2 *)

(* :Mathematica Version: 7.0 *)

(* :Copyright: RobotTools source code (c) 2005-2008 Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:
   Character info functionality.
   CharacterData is a System` symbol in version 7.0, but it does not have the needed
   functionality. So have a littleinternal version.
*)

(* ::Section:: *)
(*Information*)

`Information`CVS`$CharacterDataId = "$Id: CharacterData.m,v 1.1 2008/05/19 15:55:26 brenton Exp $"

Begin["`Package`"]

characterData

End[]

Begin["`CharacterData`Private`"]

$TextResourcesDirectory :=
	FileNameJoin[{$InstallationDirectory, "SystemFiles", "FrontEnd", "TextResources"}]

toHexString[n_Integer] :=
	"0x" <> ToUpperCase[IntegerString[n, 16, 4]]

$unicodeCharacters :=
	$unicodeCharacters =
	Module[{fileName = FileNameJoin[{$TextResourcesDirectory,"UnicodeCharacters.tr"}]},
		ReadList[fileName, Word, WordSeparators -> {"\t"}, RecordLists -> True]
	]

characterQ[c_] :=
	Head[c] === String && StringLength[c] == 1

unicodeCharactersLine[c_String] :=
	Module[{code, hex, pos},
		code = First[ToCharacterCode[c]];
		hex = toHexString[code];
		pos = Position[$unicodeCharacters[[All, 1]], hex];
		If[pos != {},
			Extract[$unicodeCharacters, First[pos]]
			,
			(* there is no entry for c in UnicodeCharacters.tr *)
			$Failed
		]
	]
	
characterData[c_?characterQ, "LongName"] :=
	Module[{line, longName},
		line = unicodeCharactersLine[c];
		If[line =!= $Failed,
			longName = line[[2]];
			If[longName != "\\[]",
				longName
				,
				Missing["NotApplicable"]
			]
			,
			Missing["NotApplicable"]
		]
	]

characterData[c_?characterQ, "Aliases"] :=
	Module[{line, aliases},
		line = unicodeCharactersLine[c];
		If[line =!= $Failed,
			aliases = line[[3]];
			If[aliases != "()",
				StringCases[aliases, ShortestMatch["$" ~~ a__ ~~ "$"] :> a]
				,
				{}
			]
			,
			(* if a character isn't in UnicodeCharacters.tr, then it doesn't have any aliases *)
			{}
		]
	]
	
End[] (* `CharacterData`Private` *)
