(*NewContextPath is used to set the $ContextPath and avoid unwanted symbols from leaking out into Global symbols with the same name*)
System`Private`NewContextPath[{"System`", "NaturalLanguage`"}];

Needs["NaturalLanguage`Customizations`"];

(*essentially all code should be written in a private context, so nothing leaks out into the user's session*)
Begin["System`NaturalLanguage`Private`"];

(*symbols to be protected/read protected*)
$readProtectedSymbols={NaturalLanguage};
	
(*ensure there are no latent definitions to the symbols*)
CompoundExpression[
	Unprotect[#],
	Clear[#]
]& /@ $readProtectedSymbols;


(* ----- Implementation -----  Code Goes Here ----- *)

(*
At the time of this writing there are no strict code-style guidelines for Wolfram Language source code, 
however described below is a general outline of what I feel like is good practice in terms of code structure, 
which includes some examples of argument handling, messaging, and unevaluated returns.
*)

(*
here are two 'tag' symbols, used in this package for Catch and Throw.
Catch and Throw should ONLY be used in their 2-argument form, never in their 1-argument form.
These symbols need not be defined, but having string definitions makes it easier to diagnose if things happen to escape back to top-level.
*)
$unevaluatedTag = "examplePacletUnevaluatedTag";
$failureTag = "examplePacletUnevaluatedTag";

(*
This is an example of using an interface function to do all the argument handling for a user-facing function 'ExampleFunction'.
This is done so that only 1 DownValue(basicaly function definition) is used, and all so that the function has a single exit-point.

Catch[iExampleFunction[args], $failureTag] is used so that at any point during the evaluation a utility function can 'bail-out' 
by using Throw[$unevaluatedTag, $failureTag] which will end evaluation and have the function exit gracefully.

With[{res = ...}, res /; res =!= $unevaluatedTag] is used so that bad inputs can be returned unevaluated.
the condition res /; res =!= $unevaluatedTag indicates "return res, unless it is $unevaluatedTag", 
which allows the function to return unevaluated without recursing or wrapping things with Hold or Unevaluated.
*)
NaturalLanguage[args___] := With[{res = Catch[iNaturalLanguage[args], $failureTag]},
	res /; res =!= $unevaluatedTag
]

(*typically the 'interface' function here handles checking arguments, and then passes the standardized, validated, arguments
on into the 'guts' of the function, which should be able to handle everything without needing to re-check the arguments along the way

In this case 'ExampleFunction' just adds two numbers together, or adds 1 if only 1 number is given
*)

SetAttributes[iNaturalLanguage, HoldAll]

iNaturalLanguage[expr_] := FormattingBlock[
	asciifystring @ basestringify @ MakeBoxes[expr, TraditionalForm]
]

(*any other input should be returned unevaluated*)
iNaturalLanguage[args___] := CompoundExpression[
	(*this will issue an appropriate message if there are too many, or too few, arguments*)
	System`Private`Arguments[NaturalLanguage[args], {1, 1}],
	Throw[$unevaluatedTag, $failureTag]
]

(* basestringify *)
Options[basestringify] = {"Times"->Automatic};

basestringify::unkbox = "The boxes `1` in `2` are not supported";

basestringify[boxes_, OptionsPattern[]] := Module[{working,bad},
	(* eliminate graphics *)
	working=boxes/._GraphicsBox|_GraphicsBox3D|_SliderBox->GraphicsBox[{}];

	(* remove _Spacer objects after graphics *)
	working=working/.TemplateBox[{a___,g_GraphicsBox,spacerBoxPattern,b___},tag:"Row"|"RowDefault",t___]->TemplateBox[{a,g,"",b},tag,t]/.spacerBoxPattern->" ";

	(* stip some structures *)
	working=working/.CleanUnusualStructures/.InterpretationBox->Function[a,a,HoldAll];

	(* remove labeled graphics and dynamic structures *)
	working=working//.TemplateBox[{""|_GraphicsBox|_Graphics3DBox|TagBox[_GraphicsBox|_GraphicsBox3D,__]|OverlayBox[{__GraphicsBox},___],__},"Labeled",___]|_DynamicBox|_DynamicModuleBox->"";

	(* convert matrices to a TemplateBox *)
	working=working//.RowBox[{"(","\[NoBreak]",GridBox[m_List,___],"\[NoBreak]",")"}]->TemplateBox[{m},"Matrix",DisplayFunction->(RowBox[{"(","\[NoBreak]",GridBox[#1],"\[NoBreak]",")"}]&)];

	(* convert template boxes to ordinary boxes *)
	working=working/.TemplateBox->TemplateBoxToStringBoxes;

	(* add spacing for row box operators, e.g., +, - etc *)
	working=working/.RowBox->FixRowBoxOperators;

	(* unevaluated symbols that shouldn't appear in output, e.g., If *)
	working=working/.InvalidSymbols;

	working=working//.PaneSelectorBox[rules_,___]:>(False/.rules);

	working=working/.TagBox|FormBox|Cell|TooltipBox|ItemBox|FrameBox|RotationBox|PaneBox|PaneSelectorBox|ButtonBox->Function[a,a,HoldAll];

	working=working/.StyleBox->StripStyleBox;

	working=working/.s_String:>ToCalculateNumberString[s];

	If[OptionValue["Times"]==="*",working=working/.RowBox->AddStar];

	working=working/.s_String:>fixstring[s];

	working=working/._GraphicsBox|_Graphics3DBox|_OverlayBox->"";

	working=working/.GridBox[{{""..}..},___]->"";

	working=working/.OverlayBox->reduceOverlayBox;

	working=working/.RowBox->AddScriptBoxSpace;

	working=working/.FractionBox->(RowBox[{ParenthesizeBox[#1],"/",ParenthesizeBox[#2]}]&);

	working=working/.SqrtBox->(RowBox[{"sqrt","(",#1,")"}]&);

	working=working/.RowBox->AddRowBoxSpace;

	working=working/.SubscriptBox|UnderscriptBox->(RowBox[{#1,"_",ParenthesizeBox[#2]}]&);

	working=working//.SuperscriptBox[a_,"\[Prime]",___]:>RowBox[{a,"'"}];

	working=working//.SuperscriptBox[a_,"\[Prime]\[Prime]",___]:>RowBox[{a,"''"}];

	working=working//.RadicalBox[a_,b_,___]:>SuperscriptBox[a,RowBox[{"1","/",b}]];

	working=working/.SuperscriptBox|OverscriptBox->(RowBox[{ParenthesizeBox[#1],"^",ParenthesizeBox[#2]}]&);

	working=working/.SubsuperscriptBox|UnderoverscriptBox->(RowBox[{#1,"_",ParenthesizeBox[#2],"^",ParenthesizeBox[#3]}]&);

	working=working/.s_String:>StringReplace[s,basicasciirules];

	working=working/.GridBox[{{""..}..},___]->"";

	working=working//.GridBox[a_,___]:>RowBox[Riffle[RowBox/@(Riffle[#," | "]&/@a),"\n"]];

	working=working//.BoxData[b_List]:>RowBox[b];

	working=working/.BoxData->Identity/.TextData->textbox;

	working=working/.RowBox->stringbox;

	If[StringQ[working],
		(* might issue an error message *)
		If[BackTickWordQ[working], "", working],
		
		bad=Union@Cases[working,s_Symbol/;StringMatchQ[SymbolName[Unevaluated@s],__~~"Box"],{0,Infinity},Heads->True];
		Which[
			Length[bad]>0,
			Message[basestringify::unkbox,bad,working],
			
			!FreeQ[working, s_String?BackTickWordQ],
			(* might issue an error message *)
			"",
			
			True,
			""
		]
	]
]

spacerBoxPattern = TemplateBox[_, "Spacer1"];

CleanUnusualStructures = {
	TagBox[StyleBox[s_String, r___], HoldForm] :> TagBox[s, r],
	Cell[s_String, "Text"] :> s
};

InvalidSymbols = {
	"If" -> "System`If", "Which" -> "System`Which", "Switch" -> "System`Switch", "$Failed" -> "System`Failed", "Entity"->"System`Entity",
	"DateListPlot" -> "System`DateListPlot"
};

basicasciirules = {
	"\[InvisibleSpace]"|"\[Null]"|"\[NoBreak]"|"\[InvisibleApplication]"|"\[InvisiblePrefixScriptBase]"->"",
	"\[NegativeMediumSpace]"|"\[NegativeThinSpace]"|"\[NegativeThickSpace]"|"\[NegativeVeryThinSpace]"->"",
	"\[VeryThinSpace]"->"",
	FromCharacterCode[63484]->",",
	"\[ThinSpace]"|"\[MediumSpace]"|"\[ThickSpace]"|"\[LetterSpace]"|" "|"\[SpanFromLeft]"|"\[SpanFromAbove]"|"\[SpanFromBoth]"|"\[DiscretionaryLineSeparator]" -> " "
};

(* custom handling of TraditionalForm TemplateBox's *)

(stringifyType[#] = "prefix")&/@
{"Abs", "Beta", "Binomial", "Ceiling", "FactorialPower", "CubeRoot",
"Floor", "Log", "Multinomial", "Subfactorial", "Superfactorial"};

(stringifyType[#[[1]]] = #[[2]])&/@
{"AiryAiPrime" -> "Ai'", "AiryBiPrime" -> "Bi'", 
 "Hypergeometric0F1" -> "0F1", "Hypergeometric1F1" -> "1F1", 
 "Hypergeometric2F1" -> "2F1", "JacobiAmplitude" -> "am",
(* Alternate template boxes for 'prefix' functions *)
 "Beta3" -> "beta", "Beta4" -> "beta", 
 "BetaRegularized4" -> "betaregularized", 
 "FactorialPower3" -> "factorialpower", "Fibonacci2" -> "fibonacci", 
 "PolyGamma2" -> "polygamma"
 };

stringifyType["InterpretationForm"] = "interpretationfunction";
stringifyType[_] = Automatic;
stringifyType[f_Function] = f;
stringifyType["LinearSubscript"] = RowBox[{#1, #2}]&;

TemplateBoxToStringBoxes[data_, tag_, opts___] := Module[{string, func},
	string = stringifyType[tag];
	Switch[string,
		"prefix",
		func = RowBox[{ToLowerCase@tag, "(", RowBox[Riffle[data, ","]], ")"}]&,

		"stdname",
		func = RowBox[{tag, "(", RowBox[Riffle[data, ","]], ")"}]&,
		
		"interpretationfunction",
		func = InterpretationFunction /. {opts};
		If[func === InterpretationFunction, func = RowBox[{tag, "(", RowBox[Riffle[data, ","]], ")"}]&]
		,

		_String,
		func = RowBox[{string, "(", RowBox[Riffle[data, ","]], ")"}]&,

		_Function,
		func = string,

		_,
		func = DisplayFunction /. {opts};
		If[func === DisplayFunction, 
			func = CurrentValue[{StyleDefinitions, tag, "TemplateBoxOptionsDisplayFunction"}];
			If[MatchQ[func, _Function], func = BoxForm`normalizeTemplateBoxFunction[func]]
		];
		If[!FreeQ[func, TemplateSlotSequence],
			func = BoxForm`normalizeTemplateBoxFunction[func]
		]
	];
	Which[
		MatchQ[func, _Function],
		func @@ data,

		ListQ[data],
		RowBox[{tag, "[", RowBox[Riffle[data,","]], "]"}],

		True,
		RowBox[{tag, "[", data, "]"}]
	]
]

FromTemplateBox[data_, tag_, opts___] := Module[{func},
	func = DisplayFunction /. {opts};
	If[func === DisplayFunction, 
		func = CurrentValue[{StyleDefinitions, tag, "TemplateBoxOptionsDisplayFunction"}];
		If[MatchQ[func, _Function], func = BoxForm`normalizeTemplateBoxFunction[func]]
	];

	If[!FreeQ[func, TemplateSlotSequence],
		func = BoxForm`normalizeTemplateBoxFunction[func]
	];

	Which[
		MatchQ[func, _Function],
		func @@ data,

		ListQ[data],
		RowBox[{tag, "[", RowBox[Riffle[data,","]], "]"}],

		True,
		RowBox[{tag, "[", data, "]"}]
	]
]

FixRowBoxOperators[a_List] := Module[{row},
	row = Replace[a, {op : Longest[Repeated["+" | "-", {0, 1}]], r___} :> Join[{op}, FixSpacing[{r}]]];
	RowBox[row]
]

FixSpacing[a_List] := Replace[a, {","->", ", op:Alternatives["+", "-", "\[PlusMinus]", "\[MinusPlus]", "\[Xor]", "\[Or]", "\[And]", "\[Nand]", "\[Nor]"] :> " "<>op<>" "}, {1}]

StripTagBox[a_, ___] := a

StripStyleBox[_, ___, ShowContents->False, ___] = "";
StripStyleBox[a_, ___] := a

AddStar[list_] := RowBox @ Replace[
	list,
	{
	{a___, " ", s_String?(StringMatchQ[#, NumberString] &), " ", b___} -> {a, "*", s, "*", b},
	{a___, s_String?(StringMatchQ[#, NumberString] &), " ", b___} -> {a, s, "*", b},
	{a___, " ", s_String?(StringMatchQ[#, NumberString] &), b___} -> {a, "*", s, b}
 	}
]

fixstring[s_] := StringReplace[
	s,
	StartOfString ~~ "\"" ~~ a___ ~~ "\"" ~~ EndOfString :> StringJoin[
		StringSplit[
			a, 
			"\\" ~~ x_ :> Switch[x, "n", "\n", "t", "\t", "\\", "\\", "\"", "\"", _, "\\" <> x]
		]
	]
]

reduceOverlayBox[a_List, opts___] := Module[{stripped = DeleteCases[a, ""]},
	Replace[stripped,
		{
		{boxes_} :> boxes,
		_ -> OverlayBox[a, opts]
		}
	]
]

AddScriptBoxSpace[a_List] := Module[{spaces},
	spaces = ScriptBoxSpace @@@ Partition[a, 2, 1];
	RowBox[DeleteCases[Riffle[a, spaces], ""]]
]

AddRowBoxSpace[a_List] := Module[{spaces},
	spaces = RowBoxSpace @@@ Partition[a, 2, 1];
	RowBox[Riffle[a, spaces]]
]

RowBoxSpace[first:("%" | _RowBox), next_] := If[LetterBoxQ[next] && !TrailingSpaceQ[first], " ", ""]
RowBoxSpace[_, _] := ""

ScriptBoxSpace[first:(_SubscriptBox | _SuperscriptBox | _SubsuperscriptBox | _UnderscriptBox | _OverscriptBox | _UnderoverscriptBox), next_] := If[!BracketBoxQ[next] && !TrailingSpaceQ[first], " ", ""]
ScriptBoxSpace[_, _] := ""

(* a horrible hack to work around the fact that \[Integral] later gets converted to " integral " *)
BracketBoxQ[(SubscriptBox | SuperscriptBox | SubsuperscriptBox | UnderscriptBox | OverscriptBox | UnderoverscriptBox)["\[Integral]", __]] = True;
BracketBoxQ[s_String] := StringMatchQ[s, "\[RightBracketingBar]"|"{"|"["|"("|"/"|" "|"<"|">"|"+"|"-"|"\[LongEqual]"|"\[TildeTilde]"|"}"|"]"|")"|"\[InvisibleSpace]"|" "|"\[MediumSpace]"|"\[ThinSpace]"|"\[NegativeMediumSpace]"|"\[Cross]"|"\[CenterDot]"|"\[Times]"|"\[NoBreak]"|"," ~~ ___]
BracketBoxQ[RowBox[{RowBox[{"(", ___}], ___}]] = False;
BracketBoxQ[RowBox[{s_, ___}]] := BracketBoxQ[s]
BracketBoxQ[_] = False;

LetterBoxQ[s_String] := StringMatchQ[s, LetterCharacter ~~___]
LetterBoxQ[RowBox[{s_, ___}]] := LetterBoxQ[s]
LetterBoxQ[(SubscriptBox|SuperscriptBox|SubsuperscriptBox)[s_,__]] := LetterBoxQ[s]
LetterBoxQ[_] = False;

TrailingSpaceQ[s_String] := StringMatchQ[s, ___~~(" "|"-"|"\[Hyphen]")]
TrailingSpaceQ[RowBox[{___,s_}]] := TrailingSpaceQ[s]
TrailingSpaceQ[_] = False;

ParenthesizeBox[ boxes_RowBox ] := If[unneededParentheses[First @ boxes], boxes, RowBox[{"(", boxes, ")"}]]
ParenthesizeBox[ boxes_FractionBox ] := RowBox[{"(",boxes,")"}]
ParenthesizeBox[ boxes_ ] := boxes

unneededParentheses[{"(", __}] = True;
unneededParentheses[r_] := MatchQ[DeleteCases[r, ""|"\[InvisibleSpace]"], {_String} | {_String | _RowBox, "(", _String | _RowBox, ")"} | {"[", _String | _RowBox, "]"}]

BackTickWords[s_String] := StringCases[s, WordBoundary~~LetterCharacter.. ~~ Repeated["`" ~~ LetterCharacter.., {1,Infinity}] ~~WordBoundary]
BackTickWordQ[s_String] := BackTickWords[s] =!= {}

stringbox[a:{___String}] := StringJoin[a]
stringbox[a:{(_String | _Row | _Hyperlink | _HTMLInfo) ..}] := Row[a //. {x_HTMLInfo:>x, Shortest[b___], c__String, Shortest[d___]} :> {b, StringJoin[{c}], d}]
stringbox[a_] := RowBox[a]

textbox[a_String | a_RowBox] := RowBox[{a}]
textbox[a__] := RowBox[a]

ToCalculateNumberString[s_String] := Module[{e},
	If[!StringMatchQ[s, NumberString~~__], Return[s]];
	e = Quiet@Check[ToExpression[s, InputForm, Hold],{$Failed}];
	If[TrueQ[BoxForm`HeldNumberQ@@e],
		e = ReleaseHold[e];
		Quiet@ToBoxes[NumberForm[e]] /. TagBox|InterpretationBox -> (#&),
		s
	]
]

Options[asciifystring] = {"ASCIILevel" -> "MoreUnicode", "ModifySpaceCharacters" -> Automatic};
asciifystring[baseIn_String, OptionsPattern[]] := Block[{level=OptionValue["ASCIILevel"],res, base=baseIn,pre,post},
	If[level===Automatic && !TrueQ[$AlphaDevModeQ], Return[asciifystring[base]]];
	(* preserve whitespace at start/end *)
	{pre,base,post}=StringCases[baseIn, RegularExpression["(?ms)^(\\s*+)(.*?)(\\s*+)$"] :> {"$1", "$2", "$3"}][[1]];
	res = Switch[level,
		"Strict7Bit", ToString[base, OutputForm, CharacterEncoding -> "ASCII"],
		"Unicode", base,
		"LessUnicode", asciifystring[base],
		"MoreUnicode"|_, StringReplace[base, {
			RegularExpression["["<>$asciifymoreunicodeexceptionsPattern<>"]"]:> asciifymoreunicode["$0"],
			(* Use asciifychar for character codes in private use area *)
			RegularExpression["[\:e000-\:f8ff]"]:>asciifychar["$0"]
		}]
	];
	(* Replace .7f characters by a space when between letters *)
	res = StringReplace[res, {RegularExpression["(?<=["<>$AllLetters<>"])++(?=["<>$AllLetters<>"])"]->" ", ""->""}];
	(* Get rid of double space and add single space after commas *)
	If[OptionValue["ModifySpaceCharacters"]=!=False, 
		res = StringReplace[res, {RegularExpression["  +"]->" ", RegularExpression[",\\s*"]:>", "}]
	];
	(* Get rid newly introduced leading/trailing space *)
	res = StringReplace[res, {RegularExpression["^ +| +$"]->""}];
	pre<>res<>post
	
]

asciifychar[char_] := StringReplace[ToString[FullForm[char]],
	{"\"\\[" ~~ c__ ~~ "]\"" :> fromnamedcharacter[c], "\""~~x___~~"\"":>x}
]

(* take a \[CharacterName] character and produce a reasonable replacement string *)  
fromnamedcharacter[c_] := With[{res=StringReplace[
	c,
	  {
		StartOfString~~"Capital"~~r__ :> capitalize[namedcharacter[r]],
		StartOfString~~("DoubleStruck"|"Doubled"|"Script"|"Gothic")~~r__ :> namedcharacter[r],
		StartOfString~~r_~~s___~~EndOfString /; StringFreeQ[s, CharacterRange["A","Z"]] :> ToLowerCase[r]<>s,
		StartOfString~~s__~~EndOfString :> ToLowerCase[decamel[s]]
	  }
	]},
	If[StringLength[res] > 1, " "<>res<>" ", res]
]

(* Use asciifychar for character codes in private use area *)
asciifymoreunicode[x_] := x

(* Exceptions from letting through the characters for very "canonical" asciifications.
   These can use  to represent an optional whitespace, see above.
   Also includes some private use area characters with decent asciifications *)
$asciifymoreunicodeexceptions = 
{
{"\[ThickSpace]"," "},{"\[ThinSpace]"," "},{"\[VeryThinSpace]"," "},{"\[Dash]","-"},{"\[LongDash]","--"},
{"\[Ellipsis]","..."},{"\[Prime]",","},{"\[DoublePrime]",",,"},{"\[ReversePrime]","`"},
{"\[ReverseDoublePrime]","``"},{"\[MediumSpace]"," "},{"\[InvisibleTimes]",""},{"\[LeftArrow]","<-"},
{"\[RightArrow]","->"},{"\[LeftRightArrow]","<->"},{"\[Minus]","-"},{"\[DivisionSlash]","/"},
{"\[Backslash]","\\"},{"\[Colon]",":"},{"\[Tilde]","~"},
{"\[NotEqual]", "!="}, {"\[LessEqual]", "<="}, {"\[GreaterEqual]", ">="},
{"\[LessLess]","<<"},{"\[GreaterGreater]",">>"},{"\[Star]","*"},{"\[CenterEllipsis]","..."},
{"\[LeftDoubleBracket]", "[["}, {"\[RightDoubleBracket]", "]]"}, {"\[InvisibleSpace]", ""},
{"\[NegativeVeryThinSpace]",""},{"\[NegativeThinSpace]",""},
{"\[NegativeMediumSpace]",""},{"\[NegativeThickSpace]",""},{"\[Null]",""},{"\[IndentingNewLine]","\n"},
{"\[Continuation]","\\"},{"\[InvisiblePrefixScriptBase]",""},{"\[InvisiblePostfixScriptBase]",""},
{"\[Equal]", " = "}, {"\[RuleDelayed]", ":>"}, {"\[Rule]", "->"},
{"\[NumberSign]","#"},{"\[ExponentialE]","e"},{"\[ImaginaryI]","i"},
{"\[DifferentialD]"," d"},{"\[TripleDot]","..."},{"\[LongEqual]"," = "},
{"ﬁ","fi"},{"ﬂ","fl"},

{"\[Hyphen]", "-"}, {"\[OpenCurlyQuote]", "'"}, {"\[CloseCurlyQuote]", "'"}, 
{"\[OpenCurlyDoubleQuote]", "\""}, {"\[CloseCurlyDoubleQuote]", "\""}, 
{"\[Equilibrium]", " equilibrium "}, {"\[ForAll]", " for all "}, 
{"\[PartialD]", "d"}, {"\[Exists]", "exists"}, 
{"\[NotExists]", " not exists "}, {(*Long name changed in M9;must use unicode*) "\[Laplacian]", " laplacian"}, 
{"\[Del]", " del "}, {"\[Element]", " element "}, {"\[NotElement]", " not element "}, 
{"\[SuchThat]", " such that "}, {"\[Product]", " product"}, {"\[Coproduct]", "coproduct"}, 
{"\[Sum]", " sum"}, {"\[Sqrt]", " sqrt"}, {"\[Integral]", " integral"}, 
{"\[Congruent]", " congruent "}, {"\[NotCongruent]", " not congruent "}, 
{"\[Precedes]", " precedes "}, {"\[Succeeds]", " succeeds "}, 
{"\[Subset]", " subset "}, {"\[Superset]", " superset "}, {"\[Intersection]", " intersection "}, 
{"\[Union]", " union "}, {"\[FilledSmallCircle]", "*"},
{"\[VerticalSeparator]", "|"},{"\[Cross]", "x"}, 
{"\[CapitalDifferentialD]", "D"}, {"\[ImaginaryJ]", "I"}, 
{"\[LeftSkeleton]", "<<"}, {"\[RightSkeleton]", ">>"}, 
{"\[AlignmentMarker]", ""}, {"\[VerticalBar]", "|"},
{"\[DirectedEdge]", "->"}, {"\[UndirectedEdge]", "<->"}
};

$asciifymoreunicodeexceptionsPattern = StringJoin[$asciifymoreunicodeexceptions[[All,1]]];

(asciifymoreunicode[#[[1]]] = #[[2]])&/@$asciifymoreunicodeexceptions;

$AllLetters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\:0144\[CapitalAGrave]\
\[CapitalAAcute]\[CapitalAHat]\[CapitalATilde]\[CapitalADoubleDot]\
\[CapitalARing]\[CapitalAE]\[CapitalCCedilla]\[CapitalEGrave]\
\[CapitalEAcute]\[CapitalEHat]\[CapitalEDoubleDot]\[CapitalIGrave]\
\[CapitalIAcute]\[CapitalIHat]\[CapitalIDoubleDot]\[CapitalEth]\
\[CapitalNTilde]\[CapitalOGrave]\[CapitalOAcute]\[CapitalOHat]\
\[CapitalOTilde]\[CapitalODoubleDot]\[CapitalOSlash]\[CapitalUGrave]\
\[CapitalUAcute]\[CapitalUHat]\[CapitalUDoubleDot]\[CapitalYAcute]\
\[CapitalThorn]\[SZ]\[AGrave]\[AAcute]\[AHat]\[ATilde]\[ADoubleDot]\
\[ARing]\[AE]\[CCedilla]\[EGrave]\[EAcute]\[EHat]\[EDoubleDot]\
\[IGrave]\[IAcute]\[IHat]\[IDoubleDot]\[Eth]\[NTilde]\[OGrave]\
\[OAcute]\[OHat]\[OTilde]\[ODoubleDot]\[OSlash]\[UGrave]\[UAcute]\
\[UHat]\[UDoubleDot]\[YAcute]\[Thorn]\[YDoubleDot]\[CapitalABar]\
\[ABar]\[CapitalACup]\[ACup]\[CapitalCAcute]\[CAcute]\[CapitalCHacek]\
\[CHacek]\[CapitalEBar]\[EBar]\[CapitalECup]\[ECup]\[CapitalICup]\
\[ICup]\[DotlessI]\[CapitalLSlash]\[LSlash]\[CapitalODoubleAcute]\
\[ODoubleAcute]\[CapitalSHacek]\[SHacek]\[CapitalUDoubleAcute]\
\[UDoubleAcute]\[CapitalAlpha]\[CapitalBeta]\[CapitalGamma]\
\[CapitalDelta]\[CapitalEpsilon]\[CapitalZeta]\[CapitalEta]\
\[CapitalTheta]\[CapitalIota]\[CapitalKappa]\[CapitalLambda]\
\[CapitalMu]\[CapitalNu]\[CapitalXi]\[CapitalOmicron]\[CapitalPi]\
\[CapitalRho]\[CapitalSigma]\[CapitalTau]\[CapitalUpsilon]\
\[CapitalPhi]\[CapitalChi]\[CapitalPsi]\[CapitalOmega]\[Alpha]\[Beta]\
\[Gamma]\[Delta]\[CurlyEpsilon]\[Zeta]\[Eta]\[Theta]\[Iota]\[Kappa]\
\[Lambda]\[Mu]\[Nu]\[Xi]\[Omicron]\[Pi]\[Rho]\[FinalSigma]\
\[Sigma]\[Tau]\[Upsilon]\[CurlyPhi]\[Chi]\[Psi]\[Omega]\[CurlyTheta]\
\[CurlyCapitalUpsilon]\[Phi]\[CurlyPi]\[CapitalStigma]\[Stigma]\
\[CapitalDigamma]\[Digamma]\[CapitalKoppa]\[Koppa]\[CapitalSampi]\
\[Sampi]\[CurlyKappa]\[CurlyRho]\[Epsilon]\[ScriptG]\[ScriptCapitalH]\
\[GothicCapitalH]\[ScriptCapitalI]\[GothicCapitalI]\[ScriptCapitalL]\
\[ScriptL]\[ScriptCapitalR]\[GothicCapitalR]\[GothicCapitalZ]\
\[ScriptCapitalB]\[GothicCapitalC]\[ScriptE]\[ScriptCapitalE]\
\[ScriptCapitalF]\[ScriptCapitalM]\[ScriptO]\[Aleph]\[Bet]\[Gimel]\
\[Dalet]\[ScriptA]\[ScriptB]\[ScriptC]\[ScriptD]\[ScriptF]\[ScriptH]\
\[ScriptI]\[ScriptJ]\[ScriptK]\[ScriptM]\[ScriptN]\[ScriptP]\
\[ScriptQ]\[ScriptR]\[ScriptS]\[ScriptT]\[ScriptU]\[ScriptV]\
\[ScriptW]\[ScriptX]\[ScriptY]\[ScriptZ]\[GothicA]\[GothicB]\
\[GothicC]\[GothicD]\[GothicE]\[GothicF]\[GothicG]\[GothicH]\
\[GothicI]\[GothicJ]\[GothicK]\[GothicL]\[GothicM]\[GothicN]\
\[GothicO]\[GothicP]\[GothicQ]\[GothicR]\[GothicS]\[GothicT]\
\[GothicU]\[GothicV]\[GothicW]\[GothicX]\[GothicY]\[GothicZ]\
\[DoubleStruckA]\[DoubleStruckB]\[DoubleStruckC]\[DoubleStruckD]\
\[DoubleStruckE]\[DoubleStruckF]\[DoubleStruckG]\[DoubleStruckH]\
\[DoubleStruckI]\[DoubleStruckJ]\[DoubleStruckK]\[DoubleStruckL]\
\[DoubleStruckM]\[DoubleStruckN]\[DoubleStruckO]\[DoubleStruckP]\
\[DoubleStruckQ]\[DoubleStruckR]\[DoubleStruckS]\[DoubleStruckT]\
\[DoubleStruckU]\[DoubleStruckV]\[DoubleStruckW]\[DoubleStruckX]\
\[DoubleStruckY]\[DoubleStruckZ]\[DotlessJ]\[ScriptDotlessI]\
\[ScriptDotlessJ]\[ScriptCapitalA]\[ScriptCapitalC]\[ScriptCapitalD]\
\[ScriptCapitalG]\[ScriptCapitalJ]\[ScriptCapitalK]\[ScriptCapitalN]\
\[ScriptCapitalO]\[ScriptCapitalP]\[ScriptCapitalQ]\[ScriptCapitalS]\
\[ScriptCapitalT]\[ScriptCapitalU]\[ScriptCapitalV]\[ScriptCapitalW]\
\[ScriptCapitalX]\[ScriptCapitalY]\[ScriptCapitalZ]\[GothicCapitalA]\
\[GothicCapitalB]\[GothicCapitalD]\[GothicCapitalE]\[GothicCapitalF]\
\[GothicCapitalG]\[GothicCapitalJ]\[GothicCapitalK]\[GothicCapitalL]\
\[GothicCapitalM]\[GothicCapitalN]\[GothicCapitalO]\[GothicCapitalP]\
\[GothicCapitalQ]\[GothicCapitalS]\[GothicCapitalT]\[GothicCapitalU]\
\[GothicCapitalV]\[GothicCapitalW]\[GothicCapitalX]\[GothicCapitalY]\
\[DoubleStruckCapitalA]\[DoubleStruckCapitalB]\[DoubleStruckCapitalC]\
\[DoubleStruckCapitalD]\[DoubleStruckCapitalE]\[DoubleStruckCapitalF]\
\[DoubleStruckCapitalG]\[DoubleStruckCapitalH]\[DoubleStruckCapitalI]\
\[DoubleStruckCapitalJ]\[DoubleStruckCapitalK]\[DoubleStruckCapitalL]\
\[DoubleStruckCapitalM]\[DoubleStruckCapitalN]\[DoubleStruckCapitalO]\
\[DoubleStruckCapitalP]\[DoubleStruckCapitalQ]\[DoubleStruckCapitalR]\
\[DoubleStruckCapitalS]\[DoubleStruckCapitalT]\[DoubleStruckCapitalU]\
\[DoubleStruckCapitalV]\[DoubleStruckCapitalW]\[DoubleStruckCapitalX]\
\[DoubleStruckCapitalY]\[DoubleStruckCapitalZ]\[FiLigature]\
\[FlLigature]";

(*
This sets the list of $readProtectedSymbols to be both ReadProtected and Protected, which is done for any user-facing function.
With is used because SetAttributes is HoldFirst
*)
With[{s=$readProtectedSymbols},SetAttributes[s,{ReadProtected}]];
Protect@@$readProtectedSymbols;

(* this ends the private context where functions are defined*)
End[];

(*this restores the context and context path to what they were prior to running the source file*)
System`Private`RestoreContextPath[];