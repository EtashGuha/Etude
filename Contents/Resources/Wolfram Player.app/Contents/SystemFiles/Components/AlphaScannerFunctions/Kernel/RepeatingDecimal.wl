(* ::Package:: *)

(* ::Chapter:: *)
(*prolog*)


BeginPackage["AlphaScannerFunctions`", {"AlphaScannerFunctions`CommonFunctions`"}]


RepeatingDecimalToRational::usage = "Converts a repeating decimal number into a rational."


Begin["`Private`"]


(* ::Chapter:: *)
(*main code*)


(* ::Section::Closed:: *)
(*RepeatingDecimalToRational*)


(********************************)
(*	RepeatingDecimalToRational  *)
(********************************)
ClearAll[RepeatingDecimalToRational]
Attributes[RepeatingDecimalToRational] = {HoldFirst};
(* If second argument is an integer, take it to mean the number of digits repeating. If it is a list of
   integers, we assume it to be the numbers that are repeating. *)
RepeatingDecimalToRational[number_Real, repeatingdigs_ : "None", base_ : 10]:= Module[
	{
		num, whatisrepeating, res
	},
	If[base != 10, Return[$Failed]];
	num = RepeatingDecPrecision[number];
	whatisrepeating = WhatIsRepeating[num, repeatingdigs];
		
	If[base < 10 && ContainsQ[HonestDigitList[num], _?(# >= base &)], Return[$Failed]];
		
	If[MatchQ[whatisrepeating, $Failed], whatisrepeating = RealDigits[FractionalPart[num]][[1]]];
	If[!VectorQ[whatisrepeating, NumericQ], Return[number]];
		
	res = RationalApprox[num, whatisrepeating, base];
		
	If[!NumericQ[res], number, Sign[number] res]
]

RepeatingDecimalToRational[number_Rational, ___] := number
RepeatingDecimalToRational[number_Integer, ___] := number
RepeatingDecimalToRational[number_Times, ___] := number


(* ::Section::Closed:: *)
(*Helpers*)


(* ::Subsection::Closed:: *)
(*RepeatingDecPrecision*)


ClearAll[RepeatingDecPrecision]
Attributes[RepeatingDecPrecision] = {HoldFirst};
RepeatingDecPrecision[num_Real ] :=(
 SetPrecision[num, #] & [StringLength[StringReplace[ToString[num, InputForm], {(StartOfString~~"0." | "-0.") ~~ Repeated["0"] -> "", (StartOfString~~"0." | "-0.") -> "", "." -> "" }]] ]
)


(* ::Subsection::Closed:: *)
(*WhatIsRepeating*)


(* head Integer is for RepeatingDecimalToRational[0.`, {9}] *)
ClearAll[WhatIsRepeating]
WhatIsRepeating[num : (_Real  | _Integer), repeatingdigs_] /; MatchQ[repeatingdigs, {__?NumericQ}] := repeatingdigs
WhatIsRepeating[num : (_Real | _Integer), repeatingdiglength_] /; MatchQ[repeatingdiglength, _?NumericQ] := RealDigits[num][[1]][[-repeatingdiglength;;]]
WhatIsRepeating[num : (_Real), repeatingdigs___] := Module[{digslist = HonestDigitList[num], repdigs},
	repdigs = digslist /. GoodRepDigRulesList;
	
	repdigs = If[(*Length[repdigs] == Length[digslist] ||*) MatchQ[ repdigs, {0 ..}] ,
   		digslist,
   		(* Commented out code below causes major problems for RepeatingDecimal[CalculateReal[0.75920033854533527837188392306739624117797687748006081741640000, "NumDigits" -> 58]] *)
   		repdigs /. {a_ ..} :> {a} (*/. {a__..} :> {a}*)
   	](*;
   	
   	If[MatchQ[repdigs, {__?NumericQ}], Flatten[Table[repdigs, {OptionValue["RepeatingDigitLength"] /. Automatic -> 1}]], repdigs]*)
]

WhatIsRepeating[num : (_Rational | _Integer), repeatingdigs___] := With[{digslist = RealDigits[FractionalPart[num]][[1]]},
	Which[
		(*MatchQ[digslist, {0}],
   			{0},*)
   		MatchQ[digslist, {{__}}],
   			digslist /. {{b__}} :> {b},
   		MatchQ[digslist, {__}],
   			digslist /. {b__} :> {b},
   		True,
   		{}
   	] 
]


(* ::Subsection::Closed:: *)
(*HonestDigitList*)


ClearAll[HonestDigitList]
HonestDigitList[number_] := With[{list = PadLeft[#1, Length[#1] + Abs[#2]] & @@ (RealDigits[FractionalPart[number]] /. {a_Integer, 0} :> {a})},
	If[MatchQ[list, {_, 0}], list[[;;1]], list]	
]

HonestDigitList[number_, "All"] := Module[{list = HonestDigitList[number]},
	{Flatten[Prepend[list, #]], Length[#]} & [IntegerDigits[IntegerPart[number]]]
]


(* ::Subsection::Closed:: *)
(*GoodRepDigRulesList*)


GoodRepDigRulesList = {
						{___, a__, a__} /; {a} != {0..} :> {a},
						{___, a__, a__, b__} /; Length[{b}] <= Length[{a}] && MatchQ[{b}, {a}[[;;Length[{b}]]]] :>  {a},
					 (* {___, a__, a__, b__} :> {a, a, b}, *)
						{foo___, a__, b__} /; Length[{b}] <= Length[{a}] && 5 Length[{b}] >= Length[{foo, a, b}] && MatchQ[{b}, {a}[[;;Length[{b}]]]] :> {a}
					};


(* ::Subsection::Closed:: *)
(*RationalApprox*)


RationalApprox[number : (_Real | _CalculateReal), whatisrepeating_, base_] := FromDigits[HonestDigitList[number, "All"] /. {
												{{PatternSequence @@ whatisrepeating, 0}, c_} :> {{whatisrepeating}, c},
												{{PatternSequence @@ whatisrepeating..}, c_} :> {{whatisrepeating}, c},
												{{a___, Repeated[PatternSequence @@ whatisrepeating]}, c_} :> {{a, whatisrepeating}, c},
												{{a__, PatternSequence @@ whatisrepeating, b__}, c_} :> {{a, whatisrepeating}, c}, 
												{{a___, PatternSequence @@ whatisrepeating, b__}, c_} /; Length[{b}] <= Length[whatisrepeating] && MatchQ[{b}, whatisrepeating[[;;Length[{b}]]]] :>  {{a, whatisrepeating}, c},
												{{a__}, c_} :> {{a, whatisrepeating}, c}}
												, base]


(* ::Chapter::Closed:: *)
(*epilog*)


End[]


EndPackage[]
