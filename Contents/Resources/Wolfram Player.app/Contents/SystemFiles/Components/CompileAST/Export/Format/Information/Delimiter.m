

BeginPackage["CompileAST`Export`Format`Information`Delimiter`"]


Begin["`Private`"] 

Needs["CompileAST`Export`Format`Information`Character`"]

characterQ[a_String] := StringLength[a] == 1;
characterQ[other___] := False;


(*   delimiterQ: exceptions  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
delimiterQ["\[InvisibleComma]"] = True;
delimiterQ[","] = True;
delimiterQ["["] = True;
delimiterQ["]"] = True;
delimiterQ["("] = True;
delimiterQ[")"] = True;
delimiterQ["{"] = True;
delimiterQ["}"] = True;


(*   delimiterQ: the general case  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
delimiterQ[(char_)?characterQ] :=
   Module[
      {theFixity = SyntaxCharacterInformation[char, "Fixity"]},
      theFixity === "Open" || theFixity === "InfixOpen" || theFixity === "Close"
   ];
delimiterQ[other_] := False;

SyntaxCharacterInformation[tok_, "DelimiterQ"] :=
	TrueQ[With[{info = SyntaxCharacterInformation[tok]},
		If[FailureQ[info],
			False,
			delimiterQ[info["Character"]]
		]
	]]

End[]
EndPackage[]
