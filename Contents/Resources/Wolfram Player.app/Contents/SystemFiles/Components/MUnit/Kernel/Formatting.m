Begin["`Package`"]

(*boxes -> ReadableForm string*)
BoxesToReadableFormString::usage =
"BoxesToString[boxes] converts boxes to a ReadableForm string."

(*data(boxes) -> LinearOutputForm string*)
DataToString::usage =
"DataToString[data] outputs strings of the LinearOutputForm of the given data."

End[]

Begin["`Formatting`Private`"]

(*
takes single char,
leave ASCII chars untouched, while converting non-ASCII to ASCII
*)
charToReadableForm =
	Module[{c},
		Switch[#,
			"\[Rule]", "->",
			"\[RuleDelayed]", ":>",
			"\[Equal]", "==",
			"\[And]", "&&",
			"\[Or]", "||",
			"\[Not]", "!",
			"\[NotEqual]", "!=",
			"\[LeftDoubleBracket]", "[[",
			"\[RightDoubleBracket]", "]]",
			_,
			c = ToCharacterCode[#, "Unicode"][[1]];
			If[c > 127,
				StringTake[ToString[#, InputForm, CharacterEncoding -> "ASCII"], {2, -2}]
				,
				#
			]
		]
	]&

stringToReadableFormString[s_String] :=
StringJoin[charToReadableForm /@ Characters[s]]

BoxesToReadableFormString[box_, env_:{}] :=
	Block[{$RecursionLimit = Infinity},
		boxesToReadableFormString[box, env]
	]

(*
Second arg to BoxesToReadableFormString is the lexical environment.
Currently, it is only used to communicate the immediate parent box 
*)

boxesToReadableFormString[FractionBox[op1_, op2_], env_:{}] :=
	Module[{s},
		s = StringJoin[{boxesToReadableFormString[op1, {FractionBox}], "/", boxesToReadableFormString[op2, {FractionBox}]}];
		If[env =!= {},
			s = "(" <> s <> ")";
		];
		s
	]

boxesToReadableFormString[StyleBox[content_, ___], env_:{}] :=
	boxesToReadableFormString[content, {StyleBox}]

boxesToReadableFormString[RowBox[row_], env_:{}] :=
	Module[{s},
		s = Replace[row, {
			{UnderoverscriptBox["\[Sum]", RowBox[{index_, "=", init_}], end_], body_} :>
				StringJoin["Sum[", boxesToReadableFormString[body, {RowBox}], ", {", index, ", ", boxesToReadableFormString[init, {RowBox}], ", ", boxesToReadableFormString[end, {RowBox}], "}]"],
			{UnderoverscriptBox["\[Product]", RowBox[{index_, "=", init_}], end_], body_} :>
				StringJoin["Product[", boxesToReadableFormString[body, {RowBox}], ", {", index, ", ", boxesToReadableFormString[init, {RowBox}], ", ", boxesToReadableFormString[end, {RowBox}], "}]"],
			
			(*
			this is to get around the issue that 1`-2 and 1` - 2 are two very different expressions
			*)
			{op1_, "-", op2__} :> StringJoin[{boxesToReadableFormString[op1, {RowBox}], " - ", boxesToReadableFormString[RowBox[{op2}], env]}],
			{op1_, "+", op2__} :> StringJoin[{boxesToReadableFormString[op1, {RowBox}], " + ", boxesToReadableFormString[RowBox[{op2}], {env}]}],
			
			{op1_, ",", op2__} :> StringJoin[{boxesToReadableFormString[op1, {RowBox}], ", ", boxesToReadableFormString[RowBox[{op2}], {env}]}],
			a_ :> StringJoin[boxesToReadableFormString[#, {RowBox}]& /@ a]
		}];
		If[env == {FractionBox},
			s = "(" <> s <> ")";
		];
		s
	]

boxesToReadableFormString["\[IndentingNewLine]", env_:{}] =
	"\n"
boxesToReadableFormString[s_String, env_:{}] :=
	stringToReadableFormString[s]
(* can occur when there are multiple lines in a cell *)
boxesToReadableFormString[list_List, env_:{}] :=
	boxesToReadableFormString[#, {List}]& /@ list
boxesToReadableFormString[box_, env_:{}] :=
	Block[{$OutputForms = {}},
		ToExpression[box, StandardForm,
			Function[{e}, stringToReadableFormString[ToString[Unevaluated[e], InputForm]], {HoldFirst}]]
	]

(*
DataToString

text -> return
boxes like SqrtBox, FormBox, etc. -> return pseudo-linear syntax string thing 
boxes like GraphicsBox, GridBox -> -GraphicsBox-, -GridBox-, etc.
*)

DataToString::baddata =
"Invalid data given to DataToString: `1`"

DataToString::badtext =
"Invalid text data given to DataToString: `1`"

DataToString::badbox =
"Invalid box data given to DataToString: `1`"

(*
strings go through because Cell["title", "Title"] is like implicit TextData
*)

DataToString[d_] :=
	Switch[d,
		_String,
		d
		,
		TextData[_List, ___],
		StringJoin[dtsTextData /@ d[[1]]]
		,
		_TextData,
		dtsTextData[d[[1]]]
		,
		_BoxData,
		dtsBoxData[d[[1]]]
		,
		_,
		Throw[{"Value" -> d, "Messages" -> {"Bad Data", "DataToString does not understand this type of Data"}}, MUnitErrorTag]
	]

(*
dtsTextData accepts inline Cells that come from TextData
*)
dtsTextData[s_String] := s
dtsTextData[c_Cell] := DataToString[c[[1]]]
(*seems kind of inconsistent that things like ButtonBox can just appear by themselves in TextData*)
dtsTextData[b_ButtonBox] := dtsBoxData[b]
dtsTextData[b_StyleBox] := dtsBoxData[b]
dtsTextData[t_] := Throw[{"Value" -> t, "Messages" -> {"Bad Text", "DataToString does not understand this type of Text"}}, MUnitErrorTag]

dtsBoxData[ButtonBox[b_, ___]] := dtsBoxData[b]
dtsBoxData[FormBox[b_, _]] := dtsBoxData[b]
dtsBoxData[RowBox[list_List]] := StringJoin[dtsBoxData /@ list]
dtsBoxData[StyleBox[b_, ___]] := dtsBoxData[b]
dtsBoxData[SubscriptBox[a_String, b_String]] := a<>"_"<>b
dtsBoxData[SuperscriptBox[a_String, b_String]] := a<>"^"<>b
dtsBoxData[SqrtBox[a_String]] := "@"<>a
dtsBoxData[s_String] := s
dtsBoxData[b_] := Throw[{"Value" -> Head[b], "Messages" -> {"Bad Box", "DataToString does not understand this type of box"}}, MUnitErrorTag]


End[]
