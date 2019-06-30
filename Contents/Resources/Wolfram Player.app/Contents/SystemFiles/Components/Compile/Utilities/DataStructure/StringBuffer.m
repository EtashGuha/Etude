
BeginPackage["Compile`Utilities`DataStructure`StringBuffer`"]

StringBuffer;
StringBufferQ;
StringBufferClass;
CreateStringBuffer;

Begin["`Private`"] 

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



initialize[self_] :=
	self["setBuffer", Internal`Bag[]]
stringJoin[self_, str_?StringQ] :=
	If[StringLength[str] === 1,
		Internal`StuffBag[self["buffer"], str],
		stringJoin[self, Characters[str]]
	]
stringJoin[self_, str_?ListQ] :=
	Scan[stringJoin[self, #]&, str]
stringLength[self_] :=
	Internal`BagLength[self["buffer"]]
characters[self_] :=
	Internal`BagPart[self["buffer"], All]
stringDrop[self_, n_?IntegerQ] :=
	Which[
		n === 0,
			Nothing,
		True,
			With[{data = self["characters"]},
				With[{
						dropped = Drop[data, n],
						rest = Take[data, n]
					},
					self["setBuffer", Internal`Bag[]];
					Scan[Internal`StuffBag[self["buffer"], #]&, dropped];
					StringJoin[rest]
				]
			]
	]
stringSplit[self_, char_?StringQ /; StringLength[char] === 1] :=
	With[{str = self["toString"]},
		StringSplit[str, char]
	]
stringTake[self_, n_?IntegerQ] :=
	Which[
		n === 0,
			"",
		n < 0,
			StringJoin[
				Internal`BagPart[self["buffer"], n ;;]
			],
		n > 0,
			StringJoin[
				Internal`BagPart[self["buffer"], ;; n]
			]
	]
stringContainsQ[self_, char_?StringQ /; StringLength[char] === 1] :=
	With[{data = self["characters"]},
		MemberQ[data, char]
	]
toString[self_] :=
	StringJoin[
		"\"",
		self["characters"],
		"\""
	]
clone[self_, ___] :=
	CreateStringBuffer[self["characters"]]

RegisterCallback["DeclareCompileClass", Function[{st},
StringBufferClass = DeclareClass[
	StringBuffer,
	<|
		"initialize" -> (initialize[Self, ##]&),
		"stringJoin" -> (stringJoin[Self, ##]&),
		"stringLength" -> (stringLength[Self, ##]&),
		"stringDrop" -> (stringDrop[Self, ##]&),
		"stringSplit" -> (stringSplit[Self, ##]&),
		"stringTake" -> (stringTake[Self, ##]&),
		"stringContainsQ" -> (stringContainsQ[Self, ##]&),
		"characters" -> (characters[Self, ##]&),
		"clone" -> (clone[Self]&),
		"toString" -> (toString[Self]&),
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"buffer"
	},
	Predicate -> StringBufferQ
]
]]

CreateStringBuffer[] :=
	CreateObject[StringBuffer]
CreateStringBuffer[bytes_?ListQ] :=
	With[{obj = CreateObject[StringBuffer]},
		obj["stringJoin", bytes]
	]
CreateStringBuffer[str_?StringQ] :=
	With[{obj = CreateObject[StringBuffer]},
		obj["stringJoin", str]
	]
CreateStringBuffer[str_?StringBufferQ] :=
	str["clone"]

(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)

icon := Graphics[Text[
  Style["STR\nBUF", GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];  
     
toBoxes[var_?StringBufferQ, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		StringBuffer,
		var,
  		icon,
  		{
  		    BoxForm`SummaryItem[{"string: ", var["toString"]}]
  		},
  		{}, 
  		fmt
  	]
End[]

EndPackage[]
