

BeginPackage["CompileAST`Export`Format`Information`Operator`"]


Begin["`Private`"] 

Needs["CompileAST`Export`Format`Information`Character`"]

characterQ[a_String] := StringLength[a] == 1;
characterQ[other___] := False;


(*   infixOperatorQ: exceptions  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

infixOperatorQ["+"] = True;
infixOperatorQ["*"] = True;
infixOperatorQ["^"] = True;
infixOperatorQ["."] = True;
infixOperatorQ["-"] = True;
infixOperatorQ["->"] = True;
infixOperatorQ[":>"] = True;
infixOperatorQ["="] = True;
infixOperatorQ[":="] = True;
infixOperatorQ["^="] = True;
infixOperatorQ["^:="] = True;
infixOperatorQ["+="] = True;
infixOperatorQ["-="] = True;
infixOperatorQ["*="] = True;
infixOperatorQ["/="] = True;
infixOperatorQ["/."] = True;
infixOperatorQ["//."] = True;
infixOperatorQ["//"] = True;
infixOperatorQ["/;"] = True;
infixOperatorQ["/"] = True;
infixOperatorQ[":"] = True;
infixOperatorQ[";"] = True;
infixOperatorQ["<="] = True;
infixOperatorQ["<"] = True;
infixOperatorQ[">"] = True;
infixOperatorQ[">="] = True;
infixOperatorQ["=="] = True;
infixOperatorQ["==="] = True;
infixOperatorQ["!="] = True;
infixOperatorQ["=!="] = True;
infixOperatorQ["&&"] = True;
infixOperatorQ["||"] = True;
infixOperatorQ["?"] = True;
infixOperatorQ["@@"] = True;
infixOperatorQ["@"] = True;
infixOperatorQ["/@"] = True;
infixOperatorQ["||"] = True;
infixOperatorQ["|"] = True;

(*   infixOperatorQ: the general case  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

infixOperatorQ[(char_)?characterQ] := "Infix" === SyntaxCharacterInformation[char, "Fixity"];
infixOperatorQ[other_] := False

(*   postfixOperatorQ: exceptions  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

postfixOperatorQ[";"] = True;
postfixOperatorQ["&"] = True;
postfixOperatorQ["!!"] = True;
postfixOperatorQ["!"] = True;
postfixOperatorQ["'"] = True;
postfixOperatorQ["--"] = True;
postfixOperatorQ["++"] = True;
postfixOperatorQ["=."] = True;
postfixOperatorQ[".."] = True;
postfixOperatorQ["..."] = True;
postfixOperatorQ[other_] = False;

(*   postfixOperatorQ: the general case  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)

postfixOperatorQ[(char_)?characterQ] := "Postfix" === SyntaxCharacterInformation[char, "Fixity"];
postfixOperatorQ[other_] := False;

(*   prefixOperatorQ: exceptions  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
prefixOperatorQ["\[PartialD]"] = True;
prefixOperatorQ["\[Integral]"] = True;
prefixOperatorQ["\[ContourIntegral]"] = True;
prefixOperatorQ["\[CounterClockwiseContourIntegral]"] = True;
prefixOperatorQ["\[ClockwiseContourIntegral]"] = True;
prefixOperatorQ["\[DoubleContourIntegral]"] = True;
prefixOperatorQ["\[Sum]"] = True;
prefixOperatorQ["\[Product]"] = True;


(*   prefixOperatorQ: the general case  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *)
prefixOperatorQ["!"] = True;
prefixOperatorQ[(char_)?characterQ] := "Prefix" === SyntaxCharacterInformation[char, "Fixity"];
prefixOperatorQ[other_] := False;

SyntaxCharacterInformation[tok_, "OperatorQ"] :=
	TrueQ[With[{info = SyntaxCharacterInformation[tok]},
		MemberQ[
			{"Infix", "Prefix", "Postfix"},
			Lookup[info, "Fixity", "Letter"]
		]
	]]
SyntaxCharacterInformation[tok_, "InfixOperatorQ"] :=
	TrueQ[With[{info = SyntaxCharacterInformation[tok]},
		If[FailureQ[info],
			False,
			infixOperatorQ[info["Character"]]
		]
	]]
SyntaxCharacterInformation[tok_, "PostfixOperatorQ"] :=
	TrueQ[With[{info = SyntaxCharacterInformation[tok]},
		If[FailureQ[info],
			False,
			postfixOperatorQ[info["Character"]]
		]
	]]

SyntaxCharacterInformation[tok_, "PrefixOperatorQ"] :=
	TrueQ[With[{info = SyntaxCharacterInformation[tok]},
		If[FailureQ[info],
			False,
			prefixOperatorQ[info["Character"]]
		]
	]]

End[]
EndPackage[]
