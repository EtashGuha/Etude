(* This file is not currently used *)

BeginPackage["CompileAST`Export`Format`Information`Helper`"]


Begin["`Private`"] 

Needs["CompileAST`Export`Format`Information`Character`"]

(*
   This creates the CharacterInformation function which gives information on the complete list of characters 
   from the UnicodeCharacters.tr file. These are needed since Mathematica does not currently have functions for testing 
   things like OperatorQ, PrefixQ, etc.
*)
parse[
   {
      theCode_,
      theCharacter_,
      shortforms_,
      theFixity_,
      thePrecedence_,
      theGrouping_,
      theRightSpacing_,
      theLeftSpacing_,
      other___
   }
] :=
   (ToExpression[StringJoin["\"", theCharacter, "\""]] ->
      <|
         "Code" -> theCode,
         "Associativity" -> theFixity,
         "Precedence" -> thePrecedence,
         "Grouping" -> theGrouping,
         "RightSpacing" -> theRightSpacing,
         "LeftSpacing" -> theLeftSpacing,
         "ShortForms" -> shortforms,
         "Character" -> StringDrop[StringDrop[theCharacter, 2], -1],
         "Other" -> {other}
      |>
    );

parse[{theCode_, theCharacter_, shortforms_, theFixity_}] :=
   (ToExpression[StringJoin["\"", theCharacter, "\""]] -> 
      <|
      	"Code" -> theCode,
      	"Associativity" -> theFixity,
        "ShortForms" -> shortforms,
      	"Character" -> StringDrop[StringDrop[theCharacter, 2], -1]
      |>
   );


(*
   We need to handle the cases when the characters are not actually characters at all.
*)
parse[
   {
      theCode_,
      "\\[]",
      shortforms_,
      theFixity_,
      thePrecedence_,
      theGrouping_,
      theRightSpacing_,
      theLeftSpacing_,
      other___
   }
] :=
   Nothing;
   
parse[___] :=
	Nothing;

init[] :=
	Module[{stream},		
		stream = OpenRead[System`Dump`unicodeCharactersTR];
		Quiet[
			 $SyntaxCharacterInformation = Association[Map[
		         parse,
		         ReadList[
		            stream,
		            Word,
		            RecordLists -> True,
		            WordSeparators -> {FromCharacterCode[9]},
		            RecordSeparators -> {FromCharacterCode[13], FromCharacterCode[10]}
		         ]
		      ]
		]];
		Close[stream];
		$SyntaxCharacterInformation
	];
(* init[] *)

End[]

EndPackage[]
