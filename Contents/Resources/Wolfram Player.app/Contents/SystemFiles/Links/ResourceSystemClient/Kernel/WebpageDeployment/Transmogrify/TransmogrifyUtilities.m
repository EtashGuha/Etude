
Begin["`Private`"]

PrependOptions::usage = "PrependOptions[expr, newopts] - adds newopts to the beginning\
 of the options in expr."
DeleteOptions::usage = "DeleteOptions[expr, optname] - removes all instances of optname\
 from the options in expr.\nDeleteOptions[expr, opt->val] - removes only the instance\
 of opt-val from the options in expr."
ResetOptions::usage = "ResetOptions[expr, newopts] - replaces (or adds) newopts in the\
 options of expr."

createDirectory::usage = "Mimic pre-10.4 behavior of CreateDirectory: if the requested
directory cannot be created, return the name of the directory.";

PrependOptions[h_[stuff___,opts___?OptionQ],newopts:{__?OptionQ}]:=
  h[stuff,Sequence@@Flatten@newopts,opts]
PrependOptions[h_[stuff___,opts___?OptionQ], newopts__?OptionQ]:=
  h[stuff,newopts,opts]

DeleteOptions[h_[stuff___,opts___?OptionQ], newopts__?OptionQ]:=
  DeleteOptions[h[stuff,opts],{newopts}]

(* delete the exact option,value pair *)
DeleteOptions[h_[stuff___,opts___?OptionQ], newopts:{__?OptionQ}]:=
  h[stuff, Sequence @@ DeleteCases[{opts},Alternatives@@newopts]]

(* delete all options with the optioname specified in newopts *)
DeleteOptions[h_[stuff___,opts___?OptionQ], newopts:{__}]:=
  h[stuff, Sequence @@ DeleteCases[{opts},_[Alternatives@@newopts,_]]]

ResetOptions[h_[stuff___,opts___?OptionQ], newopts__?OptionQ]:=
    h[stuff, Sequence@@Flatten@{newopts}, Sequence @@ System`Utilities`FilterOutOptions[
      {newopts},{opts}]]

ResetOptions[h_[stuff___],newopts__?OptionQ]:=h[stuff,newopts]

createDirectory[dir_] := dir /; DirectoryQ[dir];
createDirectory[dir_] := CreateDirectory[dir, CreateIntermediateDirectories->True];

InlineFormattedQ[s_String]:= Or[
  Not[StringFreeQ[ToString[FullForm[s]],
    RegularExpression["\\\\!\\\\(.*?\\\\)"]]],
  $ShowStringCharacters===False && StringMatchQ[s, "\"*\""]
];

FromInlineString[s_, showstring_:Automatic] := 
  Block[{nb = NotebookCreate[Visible -> False],
         str, res, qbef = False, qaft = False, show},
    str = s;
    If[showstring === Automatic, show=($ShowStringCharacters=!=False), show=showstring];
    If[show===False(*StringMatchQ[str, RegularExpression[".*\\\\\\!\\\\\\(.*\\\\\\).*"]]*),
      str = Cell[BoxData[str], ShowStringCharacters->False],
      (* Else *)
      If[StringMatchQ[str, "\"*"],
        qbef = True,
        str = "\"" <> str];
      If[StringMatchQ[str, "*\""],
        qaft = True,
        str = str <> "\""];
    ];
    NotebookWrite[nb, str];
    SelectionMove[nb, All, Notebook];
    SelectionMove[nb, Before, CellContents];
    FrontEndTokenExecute[nb, "DeleteNext"];
    SelectionMove[nb, After, CellContents];
    FrontEndTokenExecute[nb, "DeletePrevious"];
    SelectionMove[nb, All, CellContents];
    res = NotebookRead[nb];
    NotebookClose[nb];
    If[qbef || qaft,
      TextData[Join[
        If[qbef, {"\""}, {}],
        {res},
        If[qaft, {"\""}, {}]
      ]],
      res]];


(* in need a stringifyer... *)
ConvertToString[c_NamespaceBox]:= 
	Cases[c, DynamicModuleBox[{Set[Typeset`query$$, t_], ___}, ___] :> t, Infinity];
ConvertToString[c_String] := c;
ConvertToString[c_Cell] := ConvertToString@c[[1]];
ConvertToString[c_List] := StringJoin[ConvertToString /@ c];
ConvertToString[c_ButtonBox] := ConvertToString@c[[1]];
ConvertToString[c_BoxData] := ConvertToString@c[[1]];
ConvertToString[c_StyleBox] := ConvertToString@c[[1]];
ConvertToString[c_RowBox] := StringJoin[ConvertToString /@ c[[1]]];
ConvertToString[c_TextData] := ConvertToString@c[[1]];
ConvertToString[c_FormBox] := ConvertToString@c[[1]];
ConvertToString[c_AdjustmentBox] := ConvertToString@c[[1]];
ConvertToString[c_SuperscriptBox] := StringJoin[ConvertToString@c[[1]] <> "^"<>ConvertToString@c[[2]]];
ConvertToString[c_SubscriptBox] := StringJoin[ConvertToString@c[[1]] <> "_"<>ConvertToString@c[[2]]];
ConvertToString[f_FractionBox] := 
	StringJoin[" ( ", ConvertToString@f[[1]], " ) / ( ", ConvertToString@f[[2]] , " ) "];
ConvertToString[c : TemplateBox[_, "RefLink"|"OrangeLink"|"BlueLink", ___]] := ConvertToString[c[[1,1]]];
ConvertToString[c_] := ToString@c;
(* warn if a sequence *)
ConvertToString[c__] := (Message[ConvertToString::sequence]; StringJoin[ConvertToString /@ {c}]);
ConvertToString::sequence = "Warning: Sequence used as an argument, should be List";
(* legacy support *)
stringifyContent[c_]:= (Message[ConvertToString::legacy]; ConvertToString[c]);
ConvertToString::legacy = "stringifyContent[] has been deprecated. Use ConvertToString[] instead.";

End[] 
