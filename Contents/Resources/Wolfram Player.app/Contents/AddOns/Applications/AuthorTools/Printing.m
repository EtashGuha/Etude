(* :Context: AuthorTools`Printing` *)

(* :Author: Louis J. D'Andria *)

(* :Summary:
    Package to support the Printing palette for AuthorTools,
    including headers and footers.
    
*)

(* :Copyright: *)

(* :Package Version: $Revision: 1.35 $, $Date: 2015/11/24 19:54:52 $ *)

(* :Mathematica Version: 4.2 *)

(* :History:
    
*)

(* :Keywords:
     
*)

(* :Discussion:
    
*)

(* :Warning:
    Some SetOptions commands do not respect a value of True or False,
    but rather convert this to a value of Inherited.  This explains
    some of the behavior of the headers and footers dialog.
*)



BeginPackage["AuthorTools`Printing`", "AuthorTools`Common`"]


(*
  The term "Feader" occurs repeatedly in this package. This invented word
   serves as a shorthand for the 'Header and Footer'.
*)

ModifyPrintingOption::usage = "ModifyPrintingOption[nb, str] changes the printing option corresponding to str for nb, where each allowed value of str has a rule written for it in the package.";

HeadersDialog::usage = "HeadersDialog[nb] opens a dialog for changing header-related and footer-related options for nb.";

RunningHead::usage = "RunningHead[nb] returns a TextData expression containing a counter box built from the content of Section style cells in the notebook nb. RunningHeads[nb,style] uses the content of cells with the specified style. RunningHead[nb,style,target] pastes the counter box at the current insertion point in the notebook target.";


Begin["`Private`"]



(* ************** HeadersDialog ***************** *)

gridOpts = {
  Background->$Resource["WhiteBackground"],
  FontColor->GrayLevel[0],
  FontWeight->"Plain",
  FontFamily->"Times",
  GridBoxOptions->{GridFrameMargins->{{1,1},{2,2}}}
};

grid1 = StyleBox[GridBox[{
              {
                StyleBox[$Resource["Headers", "page1"],
                  FontSize->18]}
              }, GridFrame->True],
          Sequence @@ gridOpts];

grid12 = RowBox[{
        StyleBox[GridBox[{
              {
                StyleBox[$Resource["Headers", "page1"],
                  FontSize->18]}
              },
            GridFrame->True],
            Sequence @@ gridOpts], " ", 
        StyleBox[GridBox[{
              {
                StyleBox[$Resource["Headers", "page2"],
                  FontSize->18,
                  FontColor->GrayLevel[1]]}
              },
            GridFrame->True],
            Sequence @@ gridOpts]}];

grid21 = RowBox[{
        StyleBox[GridBox[{
              {
                StyleBox[$Resource["Headers", "page2"],
                  FontSize->18,
                  FontColor->GrayLevel[1]]}
              },
            GridFrame->True],
            Sequence @@ gridOpts], " ", 
        StyleBox[GridBox[{
              {
                StyleBox[$Resource["Headers", "page1"],
                  FontSize->18]}
              },
            GridFrame->True],
            Sequence @@ gridOpts]}];





insertSpecialButton[] := ButtonBox[
  $Resource["Headers", "Insert Value"],
  ButtonFunction:>FrontEnd`MessagesToConsole[OpenAuthorTool["InsertValue.nb"]],
  ButtonEvaluator->"Local",
  Active->True]


insertRunningHeadButton[nb_] := ButtonBox[
  $Resource["Headers", "Running Head"],
  ButtonFunction:>
    FrontEnd`MessagesToConsole[RunningHead[nb, {"CellStyle"}, ButtonNotebook[]]],
  ButtonEvaluator->"Local",
  Active->True]


closeButton[] := ButtonBox[
  $Resource["Cancel"],
  ButtonFunction:>FrontEndExecute[NotebookClose[ButtonNotebook[]]],
  ButtonEvaluator->None,
  Active->True]


okButton[nb_] := ButtonBox[
  $Resource["OK"],
  Active->True,
  ButtonEvaluator->"Local",
  ButtonFunction:>FrontEnd`MessagesToConsole[
    setNotebookOpts[nb, readOptsFromDialog[ButtonNotebook[]]];
    NotebookClose[ButtonNotebook[]]]]

applyButton[nb_] := ButtonBox[
  $Resource["Apply"],
  Active->True,
  ButtonEvaluator->"Local",
  ButtonFunction:>FrontEnd`MessagesToConsole[
    setNotebookOpts[nb, readOptsFromDialog[ButtonNotebook[]]];
    SelectionMove[ButtonNotebook[], Before, Notebook]]]


toggleButton[values_] :=
(
  SelectionMove[ButtonNotebook[], All, ButtonCell];
  r = NotebookRead[ButtonNotebook[], "WrapBoxesWithBoxData" -> True];
  lis = ToString /@ values;
  currentValue=Cases[r,ButtonBox[x_, ___]:>x,\[Infinity]][[1]];
  pos = Position[lis, currentValue][[1,1]];
  NotebookWrite[ButtonNotebook[], 
    ReplacePart[r,
      Part[lis, 1+Mod[pos, Length[values]]],
      Position[r, currentValue]
    ],
    All
  ]
)



topCells[nb_, nbOpts_] := 
Module[{name},
  name = NotebookName[nb];
  
  {
  Cell[BoxData[GridBox[{
        {
          $Resource["Headers", "Title"], 
          ButtonBox[
            RowBox[{" ", 
              StyleBox["?",
                FontSize->12], " "}],
            ButtonFunction:>(FrontEndExecute[ {
                FrontEnd`HelpBrowserLookup["AddOns", 
                  "Headers and Footers"]}]&),
            ButtonEvaluator->None,
            Active->True,
            ButtonMargins->$Resource["SlimMargin"],
            Background->$Resource["Button2Background"]]}
        },
      ColumnWidths->{0.9, 2},
      ColumnAlignments->{
        Left, Right}]], "DefinitionBox",
    CellMargins->{{4, 0}, {Inherited, 0}},
    FontFamily->$Resource["Font"],
    FontSize->14,
    FontWeight->"Bold",
    FontColor->$Resource["Button1Text"],
    Background->$Resource["Button1Background"]]
  ,
  Cell[$Resource["Headers", "for"] <> name, "Text",
    TaggingRules->{"NotebookObject" -> nb},
    CellMargins -> {{Inherited,Inherited},{Inherited,0}},
    TextAlignment->Left,
    FontSlant -> $Resource["Italic"]]
  }
]



facingPagesCell[nb_, nbOpts_] :=
Module[{setting, k, lis},
  setting = List @@ nbOpts[[{5,6}]];
  k = Switch[setting,
    {True, "Left" | Left}, 2,
    {True, "Right" | Right}, 3,
    _, 1
  ];
    
  lis = {
    {grid1, {False,"Right"}},
    {grid12, {True,"Left"}},
    {grid21, {True,"Right"}}
  };

  (* The following block is needed because these radio buttons must
  refresh the dialog -- if there are facing pages, we must add the
  entry blanks for left and right pages. *)
  
  Block[{radioButton},
    radioButton[activeQ_, radios_,n_]:= Module[{str},
      str = $Resource["Headers", If[activeQ, "active", "inactive"]];
      ButtonBox[RowBox[{str, "  ", Part[radios,n,1]}], ButtonData -> refresh[nb, n]]
    ];
  
    radioButtonCell[RadioButtonData[
      $Resource["Headers", "Set Facing"],
      "FacingPages+FirstPageFace",
      lis,
      k
    ]]
  ]
]




(**** Radio Button Support *****)


radioButton[activeQ_, radios_, n_] := Module[{str},
  str = $Resource["Headers", If[activeQ, "active", "inactive"]];
  ButtonBox[RowBox[{str, "  ", Part[radios,n,1]}], ButtonData -> n]
]


radioButtonCell[r:RadioButtonData[prefix_String, opt_, radios_, n_]] :=
  Cell[BoxData[GridBox[{
      Flatten[{
        Cell[prefix,
          FontFamily -> $Resource["Font"],
          FontColor -> $Resource["Button2Background"],
          TaggingRules -> {r}],
        Table[
          radioButton[k===n, radios, k],
          {k,1,Length @ radios}
        ]}]}]],
    "Text",
    Active -> True,
    FontSize -> 10,
    ButtonBoxOptions  ->  {
        ButtonEvaluator  ->  "Local",
        ButtonFunction  :> (radioButtonFunction[#2]&),
        ButtonFrame  ->  None
        },
    ShowCellBracket -> False,
    FontFamily -> $Resource["Font"],
    TextAlignment -> Center,
    CellMargins -> {{0,0},{Inherited,Inherited}},
    FontWeight->"Plain",
    TaggingRules -> {opt -> Part[radios,n,2]},
    CellTags -> opt]


radioButtonFunction[refresh[nb_, n_]]:=
Module[{nn=ToExpression[n], nb2, f5, f6, newOpts},
  (* This one is specifically for the facing pages buttons *)
  nb2 = ButtonNotebook[];
  {f5,f6} = Switch[nn,
    2, {True, "Left"},
    3, {True, "Right"},
    _, {False, "Left"}
  ];
    
  newOpts = readOptsFromDialog[nb2];
  newOpts = ReplacePart[ReplacePart[newOpts, f5, 5], f6, 6];
  NotebookPut[feaderDialog[nb, newOpts], nb2]
]
  

radioButtonFunction[n_]:=
  Module[{nn = ToExpression[n], nb = ButtonNotebook[]},
    SelectionMove[nb, All, ButtonCell];
    data = Cases[NotebookRead[nb, "WrapBoxesWithBoxData" -> True], _RadioButtonData, Infinity];
    If[Length[data] ===0, Return[]];
    data=First[data];
    data=ReplacePart[data, n, -1];
    NotebookWrite[nb, radioButtonCell[data]]
    ]

(*********)


firstPageFeaderCell[str_, nb_, nbOpts_] :=
Module[{setting, lis},
  setting = ToString[Part[nbOpts, If[str==="Header", 7, 8] ]];
  lis = {{$Resource["Yes"], True}, {$Resource["No"], False}};
  radioButtonCell[RadioButtonData[
    $Resource[str, "on first page"], 
    "FirstPage" <> str,
    lis,
    If[setting === "False", 2, 1]
  ]]
]



subsectionCell[str_] :=
Cell[BoxData[
    FormBox[
      ButtonBox[GridBox[{
            {" ", 
              RowBox[{
                StyleBox[str,
                      FontFamily->$Resource["Font"],
                      FontSize->10], 
                    StyleBox[
                      AdjustmentBox["\[DownPointer]",
                        BoxMargins->{{0, 0}, {-0.25, 0.25}},
                        BoxBaselineShift->0.25],
                      FontFamily->$Resource["Font"],
                      FontSize->18]}], " "}
            },
          ColumnWidths->{0.13, 0.72, 0.13}],
        ButtonFunction:>FrontEndExecute[ {
            FrontEnd`SelectionMove[ 
              FrontEnd`ButtonNotebook[ ], All, ButtonCell], 
            FrontEndToken[ 
              FrontEnd`ButtonNotebook[ ], "OpenCloseGroup"]}],
        ButtonNote->str], TraditionalForm]],
  "Subsubsection",
  CellDingbat->None,
  CellMargins->{{11,11}, {0, 6}},
  CellElementSpacings->{"ClosedGroupTopMargin"->0},
  FontColor -> $Resource["Button1Text"],
  TextAlignment->Left,
  SingleLetterItalics->False,
  ButtonBoxOptions->{
    Active->True,
    ButtonMargins->$Resource["SlimMargin"],
    Background->$Resource["Button1Background"]
  }
]



inputPair[str_, value_, style_, tags_List] :=
{Cell[If[Head[value]===Cell, First[value], value], style,
  CellFrame->True,
  CellMargins->{{70, 11}, {0, 0}},
  Background->$Resource["WhiteBackground"],
  CellDingbat -> str,
  CellTags->tags]}



includeLineCell[feader_, LorR_, extra_, lineQ_] := 
Module[{setting, lis},
  setting = lineQ;
  lis = {{$Resource["Yes"],True}, {$Resource["No"],False}};
  
  radioButtonCell[RadioButtonData[
    $Resource[feader <> "Line", extra],
    If[feader === "Header", "PageHeaderLines", "PageFooterLines"],
    lis,
    If[MemberQ[{False, "False"}, setting], 2, 1]
  ]]
]







inputCells[feader_, extra_, n_, nbOpts_, nb_, closed_] := 
Module[{type, page, values},

  type = feader <> "s";
  page = If[n===1, "LeftPage", "RightPage"];
  If[feader==="Header",
    values = nbOpts[[1,n]]; lineQ = nbOpts[[3,n]],
    values = nbOpts[[2,n]]; lineQ = nbOpts[[4,n]]
  ];
  Cell[CellGroupData[Flatten[{
    subsectionCell[ $Resource[feader, extra] ],
    includeLineCell[feader, page, extra, lineQ],
    inputPair[$Resource["Headers", "Left aligned"],
              values[[1]],
              feader, 
              {page, "LeftAligned", feader}],
    inputPair[$Resource["Headers", "Centered"],
              values[[2]],
              feader,
              {page, "CenterAligned", feader}],
    inputPair[$Resource["Headers", "Right aligned"],
              values[[3]],
              feader,
              {page, "RightAligned", feader}],
    insertCell[nb]
    }], closed]]
]



facingInput[nb_, nbOpts_] :=
If[TrueQ @ Part[nbOpts, 5],
  (* then we have facing pages *)
  {
  inputCells["Header", "Left", 1, nbOpts, nb, Open],
  inputCells["Header", "Right", 2, nbOpts, nb, Closed],
  inputCells["Footer", "Left", 1, nbOpts, nb, Closed],
  inputCells["Footer", "Right", 2, nbOpts, nb, Closed]
  }
  ,
  (* else, we don't have facing pages *)
  {
  inputCells["Header", "None", 1, nbOpts, nb, Open],
  inputCells["Footer", "None", 1, nbOpts, nb, Closed]
  }
]

insertCell[nb_] :=
Cell[BoxData[RowBox[{
          insertSpecialButton[],
          insertRunningHeadButton[nb]
        }]], "Text",
  TextAlignment->Center,
  FontFamily->$Resource["Font"],
  FontWeight->"Bold",
  FontSize->10,
  FontColor->$Resource["Button2Text"],
  ButtonBoxOptions->{Background->$Resource["Button2Background"]}
]


okCell[nb_, nbOpts_] :=
Cell[BoxData[GridBox[{
          {closeButton[],
          "",
          applyButton[nb],
          okButton[nb]}
        }, ColumnsEqual -> True]], "Subsubsection",
  CellDingbat->None,
  CellMargins -> {{Inherited,11},{Inherited,Inherited}},
  TextAlignment->Right,
  FontFamily->$Resource["Font"],
  FontWeight->"Bold",
  FontSize -> 11,
  FontColor->$Resource["Button2Text"],
  ButtonBoxOptions->{
    Background->$Resource["Button2Background"],
    ButtonMargins -> $Resource["WideMargin"]}
]







readOptsFromNotebook[nb_] := 
Module[{opts = AbsoluteOptions[nb, {PageHeaders, PageFooters, PageHeaderLines,
  PageFooterLines, PrintingOptions}], h, f, hl, fl, fp, fpf, fphead, fpfoot, i},
  i[vars___] := Table[Inherited, vars];
  h = PageHeaders /. opts /. PageHeaders -> i[{2},{3}];
  f = PageFooters /. opts /. PageFooters -> i[{2},{3}];
  hl = PageHeaderLines /. opts /. PageHeaderLines -> i[{2}];
  fl = PageFooterLines /. opts /. PageFooterLines -> i[{2}];
  {fp, fpf, fphead, fpfoot} = 
    {"FacingPages", "FirstPageFace", "FirstPageHeader", "FirstPageFooter"} /.
      (PrintingOptions /. opts) /. 
      {"FacingPages" -> Inherited, "FirstPageFace" -> Left,
       "FirstPageHeader" -> Inherited, "FirstPageFooter" -> Inherited};
  printOptions[h, f, hl, fl, fp, fpf, fphead, fpfoot]
]



feaderDialog[nb_, nbOpts_printOptions] :=
Notebook[Flatten @ {
    $Resource["Logo"],
    topCells[nb, nbOpts],
    facingPagesCell[nb, nbOpts],
    firstPageFeaderCell["Header", nb, nbOpts],
    firstPageFeaderCell["Footer", nb, nbOpts],
    facingInput[nb, nbOpts],
    okCell[nb, nbOpts]},
  StyleDefinitions -> $AuthorToolsStyleDefinitions,
  WindowToolbars->{},
  WindowTitle->$Resource["Headers", "Title"],
  PageWidth->WindowWidth,
  Sequence @@ $userOptions,
  WindowElements->{"HorizontalScrollBar", "VerticalScrollBar", "MenuBar"},
  ScrollingOptions -> {"VerticalScrollRange" -> Fit},
  Saveable -> False,
  ShowCellBracket->False,
  ShowCellLabel->False,
  ShowCellTags->False,
  ImageMargins->{{0, Inherited}, {Inherited, 0}},
  Background->$Resource["GrayBackground"]
]


$userOptions = {
WindowSize->{400, 500},
WindowMargins->{{Automatic, 131}, {Automatic, 0}},
Magnification->1
};



HeadersDialog[nbFile_String, args___] :=
  HeadersDialog[NotebookOpen[nbFile], args] /; FileType[nbFile] === File;

HeadersDialog[nb_NotebookObject, nbOpts_printOptions] :=
(
  If[Not[MemberQ[Notebooks[], nb]], Return[$Failed]];
  NotebookPut @ feaderDialog[nb, nbOpts]
)

HeadersDialog[nb_NotebookObject] :=
(
  If[Not[MemberQ[Notebooks[], nb]], Return[$Failed]];
  HeadersDialog[nb, readOptsFromNotebook[nb]]
)

HeadersDialog[___] :=
messageDialog[$Resource["Printing", "no nb"]] /;
  ButtonNotebook[] =!= $Failed



readInfo[tag_, dialog_] :=
(NotebookFind[dialog, tag, All, CellTags, AutoScroll->False];
Switch[tag,
  "Header" | "Footer",
    DeleteCases[#, _[CellFrame | CellMargins | CellDingbat | 
      GeneratedCell | CellAutoOverwrite | CellTags, _], \[Infinity]]& @
    Partition[NotebookRead[dialog], 3],
  "FacingPages+FirstPageFace",
    "FacingPages+FirstPageFace" /. (TaggingRules /.
      AbsoluteOptions[NotebookSelection[dialog], TaggingRules]),
  "FirstPageHeader" | "FirstPageFooter",
    tag /. (TaggingRules /. 
      AbsoluteOptions[NotebookSelection[dialog], TaggingRules]),
  "PageHeaderLines" | "PageFooterLines",
    Cases[Flatten[{NotebookRead[dialog]}],
      _[TaggingRules, {___, tag -> x_, ___}]:>ToExpression[x], \[Infinity]],
  _,
    $Failed
]
)



fixFeaders[{{a_,b_,c_},{d_,e_,f_}}] := 
  {{a,b,c},{d,e,f}} //. {
    Cell["" | _[""] | _[_[""]], ___] :> None,
    Cell["Inherited" | _["Inherited"] | _[_["Inherited"]], ___] :> Inherited,
    Cell["Automatic" | _["Automatic"] | _[_["Automatic"]], ___] :> Automatic,
    Cell["None" | _["None"] | _[_["None"]], ___] :> None,
        
    _[CellFrame | CellMargins | Background | GeneratedCell | 
      CellAutoOverwrite | CellTags ,_] :> Sequence[]}

fixFeaders[{{a_,b_,c_}}] := fixFeaders[{{a,b,c},{a,b,c}}];



fixLines[a_Symbol] := {a, a};
fixLines[a_String] := {a, a};
fixLines[{a_}] := {a, a};
fixLines[{a_, b_}] := {a, b};



readOptsFromDialog[dialog_] :=
(
  $userOptions = Options[dialog,
    {WindowSize, WindowMargins, Magnification}];
  printOptions[
    fixFeaders[readInfo["Header", dialog]],
    fixFeaders[readInfo["Footer", dialog]],
    fixLines[readInfo["PageHeaderLines", dialog]],
    fixLines[readInfo["PageFooterLines", dialog]],
    Sequence @@ readInfo["FacingPages+FirstPageFace", dialog],
    readInfo["FirstPageHeader", dialog],
    readInfo["FirstPageFooter", dialog]
  ]
)


setNotebookOpts[nb_, nbOpts_printOptions] :=
Module[{oldPrintingOptions},
  oldPrintingOptions = PrintingOptions /. 
    Options[nb, PrintingOptions];
  oldPrintingOptions = DeleteCases[oldPrintingOptions,
    _["FirstPageHeader" | "FirstPageFooter" | 
    "FacingPages" | "FirstPageFace", _]];
  SetOptions[nb,
    PageHeaders -> Part[nbOpts, 1],
    PageFooters -> Part[nbOpts, 2],
    PageHeaderLines -> Part[nbOpts, 3],
    PageFooterLines -> Part[nbOpts, 4],
    PrintingOptions -> Join[oldPrintingOptions,
      {"FacingPages" -> Part[nbOpts, 5],
       "FirstPageFace" -> Part[nbOpts, 6],
       "FirstPageHeader" -> Part[nbOpts, 7],
       "FirstPageFooter" -> Part[nbOpts, 8]}]
  ]
]

(* ************** end HeadersDialog ***************** *)



(* changeNotebookOption and changeNotebookSubOption are
   used throughout ModifyPrintingOption*)


changeNotebookOption[nb_NotebookObject, sym_, str_, explanation_] :=
Module[{value},
  value = sym /. AbsoluteOptions[nb, sym];
  AuthorTools`Common`Private`iSetOptionsDialog @@
  {
    sym,
    str,
    value,
    SetOptions[nb, sym -> #]&,
    "Explanation" -> explanation,
    "SpecialValueButtons" -> 
      If[MemberQ[{True,False},value], {True,False},{}],
    WindowTitle -> str
  }
] /; MemberQ[Notebooks[], nb];


changeNotebookOption[___] :=
messageDialog[$Resource["Printing", "no nb"]] /;
  ButtonNotebook[] =!= $Failed



changeNotebookSubOption[nb_NotebookObject, sym_, sub_, str_, explanation_] :=
Module[{value, rest},
  rest = sym /. AbsoluteOptions[nb, sym];
  value = Cases[rest, _[sub, x_] :> x];
  If[value === {}, Return[$Failed], value = First @ value];
  
  AuthorTools`Common`Private`iSetOptionsDialog @@
  {
    sym,
    str,
    value,
    SetOptions[nb, sym -> Join[
      DeleteCases[sym /. AbsoluteOptions[nb, sym], _[sub,_]], {sub -> #}]]&,
    "Explanation" -> explanation,
    "SpecialValueButtons" ->
      If[MemberQ[{True,False},value], {True,False},{}],
    WindowTitle -> str
  }
] /; MemberQ[Notebooks[], nb];


changeNotebookSubOption[___] :=
messageDialog[$Resource["Printing", "no nb"]] /;
  ButtonNotebook[] =!= $Failed



(* individual cases *)

ModifyPrintingOption[nb_, "StartingPageNumber"] :=
changeNotebookOption[
  nb,
  PrintingStartingPageNumber,
  "StartingPageNumber",
  $Resource["Printing", "StartingPageNumber"]
]


ModifyPrintingOption[nb_, "PrintingMargins"] :=
changeNotebookSubOption[
  nb,
  PrintingOptions,
  "PrintingMargins",
  "PrintingMargins",
  $Resource["Printing", "PrintingMarings"]
]


ModifyPrintingOption[nb_, "CellBrackets"] :=
changeNotebookSubOption[
  nb,
  PrintingOptions,
  "PrintCellBrackets",
  "CellBrackets",
  $Resource["Printing", "CellBrackets"]
]


ModifyPrintingOption[nb_, "SelectionHighlighting"] :=
changeNotebookSubOption[
  nb,
  PrintingOptions,
  "PrintSelectionHighlighting",
  "SelectionHighlighting",
  $Resource["Printing", "Highlighting"]
]


ModifyPrintingOption[nb_, "RegistrationMarks"] :=
changeNotebookSubOption[
  nb,
  PrintingOptions,
  "PrintRegistrationMarks",
  "RegistrationMarks",
  $Resource["Printing", "Crop"]
]


ModifyPrintingOption[nb_, "MultipleHorizontalPages"] :=
changeNotebookSubOption[
  nb,
  PrintingOptions,
  "PrintMultipleHorizontalPages",
  "MultipleHorizontalPages",
  $Resource["Printing", "MultipleHorizontal"]
]


(************** Running Head ********************)


CustomCounterBox[sty_String, strs:{__String}] :=
  CounterBox[sty, CounterFunction -> (Part[strs, #]&)]


RunningHead::nosty = "There were no cells of style \"`1`\" in the specified notebook.";

RunningHead[nbFile_String, args___] :=
  RunningHead[NotebookOpen[nbFile], args] /; FileType[nbFile] === File;

RunningHead[nb_] := RunningHead[nb, {"CellStyle", "Section"}];

RunningHead[nb_, sty_String, res___] :=
  RunningHead[nb, {"CellStyle", sty}, res];

RunningHead[nb_, {"CellStyle"}, res___] :=
  RunningHead[nb,
    {"CellStyle", InputString @ $Resource["Printing", "Pick a style"]}, res];

RunningHead[nb_, {"CellTags"}, res___] :=
  RunningHead[nb,
    {"CellTags", InputString @ $Resource["Printing", "Pick a tag"]}, res];

(*
  If someone cancels out of the InputString input of a custom
  CellStyle etc, InputString will return an empty string. In that
  case, simply end quietly.
*)

RunningHead[nb_, {type_, ""}, ___] := Null;

RunningHead[nb_NotebookObject, {type_String, sty_String}] :=
Module[{lis},
  If[type === "CellTags",
    NotebookFind[nb, sty, All, CellTags],
    NotebookFind[nb, sty, All, CellStyle]
  ];
  lis = Flatten[{NotebookRead[nb]}];
  If[lis === {},
    MessageDisplay[RunningHead::nosty, sty];
    Return[""]
  ];
  lis = fixData[First[#]]& /@ lis;
  TextData[{ CustomCounterBox[sty, lis] }]
]

RunningHead[nb_NotebookObject, data_, target_NotebookObject] := 
With[{rh = RunningHead[nb, data]},
  If[rh =!= Null && rh =!= "", NotebookWrite[target, rh]];
  rh
];


fixData[str_String] := str;
fixData[TextData[x___]] := fixData[{x}];
fixData[List[x__]] := StringJoin[fixData /@ {x}];
fixData[x_ -> y_] := "";
fixData[x_ :> y_] := "";
fixData[ButtonBox[x_,___]] := fixData[x];
fixData[StyleBox[x_,___]] := fixData[x];
fixData[z_] := StringJoin[Cases[z, _String, \[Infinity]]];




End[]

EndPackage[]
