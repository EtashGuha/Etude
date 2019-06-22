(* :Context: AuthorTools`MakeIndex` *)

(* :Author: Louis J. D'Andria *)

(* :Summary:
    This package defines functions for creating and 
    manipulating indices and index entries.
*)

(* :Copyright: *)

(* :Package Version: $Revision: 1.89 $, $Date: 2005/10/10 18:33:39 $ *)

(* :Mathematica Version: 5.0 *)

(* :History:

*)

(* :Keywords:
    document, notebook, formatting 
*)

(* :Discussion:
    
*)

(* :Warning:
    
*)


BeginPackage["AuthorTools`MakeIndex`", 
  {"AuthorTools`Common`",
   "AuthorTools`MakeProject`"}];

IndexingDialog::usage = "IndexingDialog[] opens a new dialog for adding or editing index entries.";

IndexCellOnSelection::usage = "IndexCellOnSelection[nb] reads in the selected text or cell, and adds that text as an index entry for the cell.";

AddIndexEntry::usage = "AddIndexEntry[nb, {main, sub}] adds an index entry with the given main and sub entry to the currently selected cells in nb.";

CleanIndex::usage = "CleanIndex[nb] removes index entries which refer to cell tags that don't exist in nb.";

RemoveIndex::usage = "RemoveIndex[nb] removes the cell tags and index entries from nb.";

IndexFileName::usage = "IndexFileName is an option to MakeIndex that specifies the name of the generated index file.";

MakeIndex::usage = "MakeIndex[nb|proj, format] creates a new index file for the given notebook or project in the specified format, and opens it in the front end.";

MakeIndexNotebook::usage = "MakeIndexNotebook[nb|proj, format] returns a Notebook expression containing an index for the specified notebook or project in the given format.";

ColumnHeights::usage = "ColumnHeights is an option to MakeIndex that specifies the height of the grids used for the \"TwoColumn\" format. Setting it to {n1, n2, ...} indicates that the first grid should have n1 rows, the second n2, etc.";



Begin["`Private`"];




(* constructFileName, constructDirectoryName, and windowsPathFix added
   August 11, 1998 to work around a bug in the Windows version, which 
   causes
     ToFileName[{$RootDirectory, ...}, ...]
   to return
     \a:\....
   instead of 
     a:\....
*)

constructFileName::usage = "constructFileName[fefn] turns a \
FrontEnd`FileName object into a string representing the path to the file.";

constructFileName[FrontEnd`FileName[dir_List, file_String, ___]] :=
  constructFileName[dir, file];

constructFileName[dir_List, file_String, ___] :=
  windowsPathFix @ ToFileName[dir, file];

constructDirectoryName::usage = "constructDirectoryName[expr] turns a \
FrontEnd`FileName object into a string representing the path to the directory \
containing it.";

constructDirectoryName[FrontEnd`FileName[dir_List, ___]] :=
  constructDirectoryName[dir];

constructDirectoryName[dir_List, ___] :=
  windowsPathFix @ ToFileName @ dir;

windowsPathFix::usage = "windowsPathFix[str] removes an initial pathname \
separator from path str if necessary.";

windowsPathFix[str_String] :=
If[StringMatchQ[$OperatingSystem, "Win*"] && StringTake[str, 1] === $RootDirectory,
  StringDrop[str, 1],
  str
];



pathToIndex::usage = "pathToIndex[file or files, str] returns a string
of the form \"path:to:fileIND.nb\" if fed one file, and a string of the 
form \"path:to:str\" otherwise.";

pathToIndex[{FrontEnd`FileName[d_List, n_String, __]}, str_] := 
  constructFileName[d, StringReplace[n, ".nb" -> "IND.nb"]];

pathToIndex[{s_String}, str_] :=
With[{d = DirectoryName[s]},
  ToFileName[d, StringReplace[s, {d -> "", $PathnameSeparator->"", ".nb" -> "IND.nb"}]]
];

pathToIndex[{FrontEnd`FileName[d_List, n_String, ___], __}, str_] :=
  constructFileName[d, str];

pathToIndex[{s_String, __}, str_] :=
  ToFileName[DirectoryName[s], str];

pathToIndex[___] := $Failed;



cellCellTags[nb_, opts___] :=
Module[{},
  If[HorizontalInsertionPointQ[nb], Return[$Failed]];
  SelectionMove[nb, All, Cell];
  CellTags /. Options[NotebookSelection[nb]] /. CellTags -> None
]


cellIndexingCellTag::usage = "cellIndexingCellTag[nb] reads in
the options of the current cell in nb, and returns one that
starts with $IndexingCellTagPrefix.";
(* added 1998.05.28 by Lou *)

cellIndexingCellTag[nb_, opts___] :=
Module[{tags = cellCellTags[nb, opts]},
  If[tags === $Failed || tags === None, Return[tags]];

  tags = Select[Flatten[{tags}], StringMatchQ[#, $IndexingCellTagPrefix <> "*"]&];
  If[tags === {}, None, First[tags]]
];


notebookIndexAndTaggingRules::usage = "notebookIndexAndTaggingRules[nb]
returns a pair of lists.  The first item is a list of the index entries
for the notebook, and the second is the remainder of the notebook's
TaggingRules setting.";
(* added 1998.05.28 by Lou *)

notebookIndexAndTaggingRules[nb_, opts___] :=
Module[{tRules},
  tRules = Cases[Options[nb], _[TaggingRules, x_]:>x]//Flatten;
  Join[
    If[#==={}, {{}}, {#}]& @
      Flatten[Cases[tRules, _["IndexEntries", x_]:>x], 1],
    {DeleteCases[tRules, _["IndexEntries", x_]]}
  ]
]


getNotebookIndexEntries::usage = "getNotebookIndexEntries[nb] returns
a list of the index entries from the TaggingRules of nb.";
(* added 1998.05.28 by Lou *)

getNotebookIndexEntries[nb_, opts___] :=
Module[{lis},
  lis = Cases[Options[nb], 
          _[TaggingRules, {___, _["IndexEntries", x_],___} ]:>x];
  If[lis === {}, {}, First[lis]]
]


lookupNotebookIndexEntries[nbfile_] :=
Module[{lis},
  lis = Cases[NotebookFileOptions[nbfile], 
          _[TaggingRules, {___, _["IndexEntries", x_],___} ]:>x];
  If[lis === {}, {}, First[lis]]
]



setNotebookIndexEntries::usage = "setNotebookIndexEntries[nb,lis]
sets the \"IndexEntries\" suboption of TaggingRules to lis in nb.";
(* added 1998.05.28 by Lou *)

setNotebookIndexEntries[nb_, lis_, opts___] :=
Module[{discardentries, tRules},
  {discardentries, tRules} = notebookIndexAndTaggingRules[nb];
  SetOptions[nb, TaggingRules ->
    Join[tRules, {"IndexEntries" -> lis}]]
]


cellIndexEntries::usage = "cellIndexEntries[nb] returns a list of
index entries for the current cell.";
(* altered 1998.05.28 by Lou to take into account the new
   storage location for the index entries *)

cellIndexEntries[nb_, opts___] :=
Module[{tag = cellIndexingCellTag[nb, opts]},
  If[tag === $Failed, Return[$Failed]];
  If[tag === None, Return[{}]];
  Cases[getNotebookIndexEntries[nb, opts], {tag, x___}:>{x}]
];
  

$IndexingCellTagPrefix = "i:";


cellNeedsIndexingCellTagQ[nb_, opts___] := 
  cellIndexingCellTag[nb, opts] === None;


setNextIndexingNumber::usage = "setNextIndexingNumber[nb,n] sets the \
TaggingRule \"NextIndexingNumber\" for nb to n.";

setNextIndexingNumber[nb_, n_, opts___] :=
Module[{tRules},
  tRules = TaggingRules /. Options[nb, TaggingRules];
  If[tRules === None, tRules = {}];
  SetOptions[nb, TaggingRules -> 
    Join[
      DeleteCases[tRules, _["NextIndexingNumber",_]],
      {"NextIndexingNumber" -> n}
    ]
  ]
]


nextIndexingNumber::usage = "nextIndexingNumber[nb] returns the value of \
the notebook TaggingRules option \"NextIndexingNumber\".  If one doesn't \
exist, it adds one set to 1, and returns the number 1.";

nextIndexingNumber[nb_, opts___] :=
Module[{n},
  n = Cases[Options[nb, TaggingRules], _["NextIndexingNumber",x_]:>x, Infinity];
  If[n =!= {},
    First[n],
    setNextIndexingNumber[nb, 1];
    1
  ]
]


incrementNextIndexingNumber::usage = "incrementNextIndexingNumber[nb] increases \
the value of the setting for \"NextIndexingNumber\" by 1.";

incrementNextIndexingNumber[nb_, opts___] :=
  setNextIndexingNumber[nb, nextIndexingNumber[nb]+1];


nextIndexingCellTag::usage = "nextCellTag[nb] returns a string built from \
$IndexingCellTagPrefix and nextIndexingNumber[nb].";

nextIndexingCellTag[nb_, opts___] := 
StringJoin[$IndexingCellTagPrefix,
  ToString @ nextIndexingNumber @ nb]




(*
  addTags[nb] determines if a CellTag is needed for indexing in the current cell in nb.  If not, it does nothing; if a setting is needed, it determines the next available setting, and uses it.
*)

addTags[nb_, opts___] :=
Module[{oldIndexingCellTag},
  oldIndexingCellTag = cellIndexingCellTag[nb];
  If[oldIndexingCellTag =!= None,
    (* if it's not None, then it's either $Failed or a string *)
    If[oldIndexingCellTag === $Failed, Return[$Failed], Return[]]];
  AddCellTags[nb, nextIndexingCellTag[nb]];
  incrementNextIndexingNumber[nb];
]


(*
  addIndexEntry[nb, entry] adds the entry for the current cell in nb
*)

addIndexEntry[nb_, {entry__}, opts___] :=
Module[{tag, entries, tRules},
  tag = cellIndexingCellTag[nb];
  If[Head[tag]=!=String, Return[$Failed]];
  {entries, tRules} = notebookIndexAndTaggingRules[nb];
  entries = Join[entries, {{tag, entry} //. {x___, ""} :> {x} }];
  SetOptions[nb, TaggingRules ->
    Join[tRules, {"IndexEntries" -> Union[entries]}]
  ]
]


(*
   The exported version of addIndexEntry should add the cell
   tags automatically, when necessary.
*)

AddIndexEntry[nb_, {entry__}] :=
  (addTags[nb]; addIndexEntry[nb, {entry}])


removeIndexEntry::usage = "removeIndexEntry[nb, entry] removes all copies \
of {tag, entry} from nb's index, where tag is the indexing tag of the 
current cell.";

removeIndexEntry[nb_, {entry__}, opts___] :=
Module[{tag, entries, tRules},
  tag = cellIndexingCellTag[nb];
  If[Head[tag]=!=String, Return[$Failed]];
  {entries, tRules} = notebookIndexAndTaggingRules[nb];
  If[!MemberQ[entries, {tag, entry}], Return[$Failed]];
  entries = DeleteCases[entries, {tag, entry}];
  
  (* if it's the last entry refering to that tag, remove the tag from the cell *)
  If[!MatchQ[entries, {___, {tag, ___}, ___}],
    SelectionRemoveCellTags[nb, tag]
  ];
  
  SetOptions[nb, TaggingRules -> Join[tRules, {"IndexEntries" -> entries}]]
]


changeIndexEntry::usage = "changeIndexEntry[nb, old, new] removes all \
copies of {tag, old} from the \"IndexEntries\", and adds a copy of \
{tag, new}.  If there are no olds in the setting, it returns $Failed";

changeIndexEntry[nb_, {oldentry__}, {newentry__}, opts___] :=
Module[{tag, entries, tRules},
  tag = cellIndexingCellTag[nb];
  If[Head[tag]=!=String, Return[$Failed]];
  {entries, tRules} = notebookIndexAndTaggingRules[nb];
  If[!MemberQ[entries, {tag, oldentry}], Return[$Failed]];
  entries = Join[
    DeleteCases[entries, {tag, oldentry}],
    {{tag, newentry}}];
  SetOptions[nb, TaggingRules ->
    Join[tRules, {"IndexEntries" -> Union[entries]}]
  ]
]


nbOptions := {
StyleDefinitions -> $AuthorToolsStyleDefinitions,
Sequence @@ $userOptions,
Background->$Resource["GrayBackground"],
Saveable -> False,
ShowCellBracket->False,
WindowElements->{"HorizontalScrollBar", "VerticalScrollBar", 
  "MenuBar"},
ScrollingOptions -> {"VerticalScrollRange" -> Fit},
WindowTitle->$Resource["Index","Title"]
};


$userOptions = {
WindowSize->{320, 400},
WindowMargins->{{Automatic, 131}, {Automatic, 0}},
Magnification->1
};


textFieldCells::usage = "textFieldCells[lis] returns a list of cells \
for use in the indexing dialog, with default values given in lis.";

textFieldCells[contents_List] :=
Module[{x, y, z},
{x, y, z} = Take[Join[contents, {"","",""}], 3];
{
Cell[$Resource["Index", "Main Entry"], "Text",
  CellMargins->{{Inherited, Inherited}, {2, 5}},
  Editable->False,
  FontFamily->$Resource["Font"],
  FontWeight->"Bold",
  FontSize->10],

Cell[TextData[{x}], "Text",
  CellFrame->True,
  CellMargins->{{Inherited, Inherited}, {5, 0}},
  Background->$Resource["WhiteBackground"],
  CellTags->"MainEntry"],

Cell[$Resource["Index", "Sub Entry"], "Text",
  CellMargins->{{Inherited, Inherited}, {2, 5}},
  Editable->False,
  FontFamily->$Resource["Font"],
  FontWeight->"Bold",
  FontSize->10],

Cell[TextData[{y}], "Text",
  CellFrame->True,
  CellMargins->{{Inherited, Inherited}, {5, 0}},
  Background->$Resource["WhiteBackground"],
  CellTags->"SubEntry"],

Cell[$Resource["Index", "Short Form"], "Text",
  CellMargins->{{Inherited, Inherited}, {2, 5}},
  Editable->False,
  FontFamily->$Resource["Font"],
  FontWeight->"Bold",
  FontSize->10],

Cell[TextData[{z}], "Text",
  CellFrame->True,
  CellMargins->{{Inherited, Inherited}, {2, 0}},
  Background->$Resource["WhiteBackground"],
  CellTags->"MainCategory"]
}
];



displayIndexEntry::usage = "displayIndexEntry[lis] returns a list of \
objects suitable for displaying within a TextData construct.";


displayIndexEntry[{entry__}] :=
BoxForm`Intercalate[
  Join[{entry}, {"","",""}] //. {x___, ""} :> {x},
  ", "
];


headerCells[contents_, opts___] := {
Cell[BoxData[GridBox[{
        {
          $Resource["Index", "Title"], 
          ButtonBox[
            RowBox[{" ", 
              StyleBox["?",
                FontSize->12], " "}],
            ButtonFunction:>(FrontEndExecute[ {
                FrontEnd`HelpBrowserLookup["AddOns", 
                  "Make Index: Introduction"]}]&),
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
Cell[contents, "Text",
  TextAlignment -> "Right",
  FontSlant -> $Resource["Italic"],
  Editable->False]
}


header[1] := $Resource["Index", "Add Tags"];
header[2, lis_] := 
TextData[{
  $Resource["Index", "Editing"],
  StyleBox[displayIndexEntry[lis],
    FontWeight->"Plain"]
}];
header[3] :=
TextData[{StyleBox[$Resource["Index", "Error"],
  FontColor->$Resource["ErrorText"]]}];


buttonCellOptions = 
{
  Editable->False,
  CellMargins -> {{Inherited,Inherited},{Inherited,0}},
  TextAlignment->Right,
  FontFamily->$Resource["Font"],
  FontSize->11,
  FontWeight->"Bold",
  FontColor->$Resource["Button2Text"],
  ButtonBoxOptions->{
    ButtonEvaluator->Automatic,
    Background->$Resource["Button2Background"],
    Active->True}
};

buttonCell[1] :=
Cell[BoxData[GridBox[{
        {
          ButtonBox[$Resource["Index", "Tag button"], 
            ButtonFunction:>FrontEnd`MessagesToConsole[addTagsToCurrentCell[]]], 
          ButtonBox[$Resource["Close"], 
            ButtonEvaluator->None,
            ButtonFunction:>(FrontEndExecute[{
              NotebookClose[ButtonNotebook[], Interactive->False]}]&)
              ]}
        }]], "Text",
  Sequence @@ buttonCellOptions];

buttonCell[2, lis_] :=
Cell[BoxData[GridBox[{
        {
          ButtonBox[$Resource["Index", "Tag button"],
            ButtonFunction:>FrontEnd`MessagesToConsole[editTagsOfCurrentCell[lis]]], 
          ButtonBox[$Resource["Index", "Cancel edit"],
            ButtonFunction:>FrontEnd`MessagesToConsole[refreshIndexingDialog[]]]}
        }]], "Text",
  Sequence @@ buttonCellOptions];

buttonCell[3] :=
Cell[BoxData[
  ButtonBox[$Resource["Index", "Update button"],
    ButtonFunction:>FrontEnd`MessagesToConsole[refreshIndexingDialog[]]]
  ], "Text",
  Sequence @@ buttonCellOptions];



editButton::usage = "editButton[ind] creates a button that when \
clicked, will (a) allow the user to edit ind, an index entry for the \
current cell, and (b) refresh the index dialog";

editButton[lis_, opts___] :=
ButtonBox[$Resource["Index", "Edit"], ButtonStyle->"Hyperlink", 
  ButtonFunction:>FrontEnd`MessagesToConsole[refreshIndexingDialog[lis, header[2]] ]];


copyButton[lis_] :=
ButtonBox[$Resource["Index", "Copy"], ButtonStyle->"Hyperlink",
  ButtonFunction:>FrontEnd`MessagesToConsole[ refreshIndexingDialog[lis, header[1]] ]];


removeButton::usage = "removeButton[ind] creates a button that when \
clicked, will (a) remove ind from the current cell's \"IndexEntries\" \
setting, and (b) refresh the index dialog.";

removeButton[lis_, opts___] :=
ButtonBox[$Resource["Index", "Remove"], ButtonStyle->"Hyperlink",
  ButtonFunction:>FrontEnd`MessagesToConsole[removeTagsOfCurrentCell[lis]]];


tagListCells::usage = "tagListCells[] returns the cells that contains \
the Edit/Remove list of indexing tags for the current cell.";

tagListCells[opts___] :=
Module[{lis, contents},
  lis = cellIndexEntries[foremostUserNotebook[]];
  contents = If[lis === {} || lis === None || lis === $Failed, "None",
    Most @
    Flatten[{editButton[#], " / ", copyButton[#], " / ", removeButton[#],
             ":  ", displayIndexEntry[#], "\n"}& /@ lis]];
  {Cell[$Resource["Index", "List of entries"],
     "Text", 
     CellMargins->{{Inherited, Inherited}, {2, 12}},
     Editable->False,
     FontFamily->$Resource["Font"],
     FontWeight->"Bold",
     FontSize->10],
   Cell[TextData[contents], "Text",
     CellFrame->True,
     Editable->False,
     ParagraphSpacing -> {0,0},
     CellMargins->{{Inherited, Inherited}, {0, 0}},
     ButtonBoxOptions->{ButtonEvaluator->Automatic}],
   buttonCell[3]}
]


extractArguments[nb_] := 
{NotebookFind[nb,"MainEntry",All,CellTags];
  (* 
     changed the following line from NotebookRead[nb][[1,1]]
     because in 3.0, TextData[str] is returned from NotebookRead,
     but in 3.5 and beyond, str is returned without Text Data.
  *)
  If[Head[#] === String, #, First @ #]& @ First @ NotebookRead[nb, "WrapBoxesWithBoxData" -> True],
  NotebookFind[nb,"SubEntry",All,CellTags];
  If[Head[#] === String, #, First @ #]& @ First @ NotebookRead[nb, "WrapBoxesWithBoxData" -> True],
  NotebookFind[nb,"MainCategory",All,CellTags];
  If[Head[#] === String, #, First @ #]& @ First @ NotebookRead[nb, "WrapBoxesWithBoxData" -> True]
}


createIndexingDialogNotebook::usage = "createIndexingDialogNotebook[] \
returns the notebook expression ready to add indexing tags to the current \
cell.  createIndexingDialogNotebook[lis] returns a notebook ready to edit \
the indexing tags in lis.";


createIndexingDialogNotebook[defaults_:{"","",""}, msg_:header[1]] := 
Block[{editingQ},
  editingQ = (msg === header[2]) && (First[defaults] =!= "");
  Notebook[{
    $Resource["Logo"],
    Sequence @@ headerCells[If[editingQ, header[2, defaults], msg]],
    Sequence @@ textFieldCells[defaults],
    If[editingQ, buttonCell[2, defaults], buttonCell[1]],
    Sequence @@ tagListCells[]
  }, 
  Sequence @@ nbOptions
  ]
]


(*
  IndexingDialog creates the dialog window which allows users to
  add/edit/remove index entries for the current cell, and sets up
  a pointer to that notebook in $IndexDialogNB. This pointer can
  only point to one notebook, so there can only be one of these
  dialogs working at a time.
*)

IndexingDialog[lis_:{"","",""}, msg_:header[1]] :=
If[MemberQ[Notebooks[], $IndexDialogNB],
  SetSelectedNotebook[$IndexDialogNB],
  $IndexDialogNB = NotebookPut[createIndexingDialogNotebook[lis, msg]]
];


refreshIndexingDialog::usage = "refreshIndexingDialog[lis] refreshes the \
dialog box to match the current user selections.";

refreshIndexingDialog[lis_:{"","",""}, msg_:header[1]] :=
Block[{winsize, winmargins},
  If[Head[$IndexDialogNB] =!= NotebookObject,
    IndexingDialog[lis, msg]
    ,
    (* without the SelectionMove, the NotebookPut doesn't work -- fe bug *)
    SelectionMove[$IndexDialogNB, All, Notebook];
    $userOptions = Options[$IndexDialogNB,
      {WindowSize, WindowMargins, Magnification}];
    (* the Insert doesn't alter the window's size and location on screen *)
    NotebookPut[createIndexingDialogNotebook[lis, msg], $IndexDialogNB]
  ];
  SelectionMove[$IndexDialogNB, Before,Notebook]
];


(* indexCellWithString[nb, str] is an easy way to add str to the
   index entries for the current cell in nb. *)

indexCellWithString[nb_, str_] :=
If[str === "", Return[$Failed],
  addTags[nb];
  addIndexEntry[nb, {str}]
];

(* IndexCellOnSelection reads in the selected text and 
   adds that text as an index entry for the cell containing it. *)

(* made "tag on selection" to work if whole cell is selected
   at Pavi and Andre's request *)

IndexCellOnSelection[nb_] :=
Module[{entry = NotebookRead[nb, "WrapBoxesWithBoxData" -> True]},
  If[Head[entry] =!= Cell,
    indexCellWithString[nb, entry],
    indexCellWithString[nb,
      If[entry[[1,0]] === String, entry[[1]], entry[[1,1]]]]
  ]
];


(* An "inline index" entry is one that is typed in a special style
   which will be interpreted as an index entry by a notebook post-
   processor in this package.  *)

$InlineIndexingStyle = "Indexing";

divisionRules = {
  {a___, b_String, " ", c___} :> {a, b<>" ", c},
  {a___, b_String, c_String,d___} :> {a, b<>c, d} /; 
    MemberQ[{" ", ","},StringTake[b,-1]],
  {a___, b_String, c_String, d___} :> {a, b<>c, d} /;
    MemberQ[{" ", ","},StringTake[c,1]]};

convertCellFromInline[nb_] :=
Module[{orig},
  SelectionMove[nb, All, Cell];
  orig = NotebookRead[nb, "WrapBoxesWithBoxData" -> True];
  inlines = Cases[orig, StyleBox[x_, $InlineIndexingStyle, ___]:>x, Infinity];
  If[inlines === {}, Return[]];
  NotebookWrite[nb,
    DeleteCases[orig, StyleBox[x_, $InlineIndexingStyle, ___], Infinity], All];
  indexCellWithString[nb, #]& ~Scan~ (inlines //. divisionRules);
]

convertNotebookFromInline[nb_] := (
  SelectionMove[nb, Before, Notebook];
  SelectionMove[nb, Next, Cell];
  While[Options[NotebookSelection[nb]] =!= $Failed,
    convertCellFromInline[nb];
    SelectionMove[nb, Next, Cell]]);


(* addTagsToCurrentCell[] is used by the dialog interface *)

addTagsToCurrentCell[] :=
Module[{args, res, nb},
  nb = foremostUserNotebook[];
  args = extractArguments[ButtonNotebook[]];
  If[First @ args =!= "",
    res = {addTags[#], addIndexEntry[#, args]}& @ nb,
    res = {$Failed, $Failed}];
  If[First[res]===$Failed || Last[res]===$Failed,
    refreshIndexingDialog[{"","",""}, header[3]],
    refreshIndexingDialog[]];
]


editTagsOfCurrentCell[old_] :=
Module[{args, res, nb},
  nb = foremostUserNotebook[];
  args = extractArguments[ButtonNotebook[]];
  If[First @ args =!= "",
    res = changeIndexEntry[nb, old, args],
    res = $Failed];
  If[res === $Failed,
    refreshIndexingDialog[{"","",""},header[3]],
    refreshIndexingDialog[]]
]


removeTagsOfCurrentCell[old_] :=
Module[{res, nb},
  nb = foremostUserNotebook[];
  res = removeIndexEntry[nb, old];
  If[res === $Failed,
    refreshIndexingDialog[{"","",""}, header[3]],
    refreshIndexingDialog[]]
]



(*
  foremostUserNotebook is needed because the IndexingDialog is a
  notebook but *not* a palette. That means that *it* is the
  InputNotebook at the time these buttons are used. We can't just
  use Notebooks[][[2]], because that might be a palette or
  something else. What we want is the *second* InputNotebook, if
  there is one. And we specifically exclude the messages and help
  notebooks because they sometimes get in the way.
*)

foremostUserNotebook[] :=
Module[{nbs},
  nbs = DeleteCases[Notebooks[],
    MessagesNotebook[] | $IndexDialogNB
  ];
  nbs = Select[nbs,
    WindowClickSelect /. AbsoluteOptions[#, WindowClickSelect]&, 1];
  If[nbs === {}, Abort[], First @ nbs]
]







(* ****

IndexData[
  "File" -> nbname,
  "iCellTag" -> str,
  "MainEntry" -> expr,
  "SubEntry" -> expr,
  "ShortEntry" -> str,
  "CellTags" -> {strs},
  "CellIndex" -> k,
  "CellPage" -> p,
  maybe more....
]

***** *)



RawIndex::usage = "RawIndex[nb|proj] returns a list of IndexData expressions representing index information from the specified notebook or project.";

IndexData::usage = "IndexData is the head used in the internal representation of an index entry before formatting.";



Options[RawIndex] =
  {IncludeCellPage -> False,
   IncludeCellTags -> False};


RawIndex[file_String, opts___?OptionQ] :=
Block[{pn, pd, pf},
  {pn, pd, pf} = {"Name", "Directory", "Files"} /. 
       ProjectInformation[file];
  
  RawIndex[pn, pd, pf, opts]
] /; FileType[file] === File


RawIndex[pn_String, pd_String, {pf__String}, opts___?OptionQ] :=
Block[{nb, result, c=0},
  nb = ProgressDialog[$Resource["Index", "Caption"],"", {1, Length @ {pf}}];
  result =
  Map[
    (ProgressDialogSetSubcaption[nb, #];
     ProgressDialogSetValue[nb, ++c];
     RawIndex[pn, pd, #, opts])&,
    {pf}
  ]//Flatten;
  ProgressDialogClose[nb];
  result
]


RawIndex[pn_String, pd_String, pf_String, opts___?OptionQ] :=
Block[{nbfile, nb, raw, cp, ct, ci, outline, pos, openNotebooks,
       obsolete, num},
  {cp, ct, ci} = {IncludeCellPage, IncludeCellTags, IncludeCellIndex} /.
    Flatten[{opts, Options[RawIndex]}];
  {cp, ct, ci} = TrueQ /@ {cp, ct, ci};

  nbfile = ToFileName[{pd}, pf];
  
  If[NotebookCacheValidQ[nbfile],
    raw = lookupNotebookIndexEntries[nbfile]
    ,
    RememberOpenNotebooks[];
    nb = NotebookOpen[nbfile];
    raw = getNotebookIndexEntries[nb];  
    NotebookCloseIfNecessary[nb];
  ];
  
  raw = Map[
    IndexData[
      "File" -> pf,
      "iCellTag" -> #[[1]],
      "MainEntry" -> #[[2]],
      "SubEntry" -> If[Length[#]>2, #[[3]], ""],
      "ShortEntry" -> If[Length[#]>3, #[[4]], ""]]&,
    raw];
  
  If[raw === {}, Return[raw]];
  
  If[ci || ct || cp, 
    outline = NotebookLookup[nbfile, "CellOutline"];
    
    (* Now that we have the outline, we can tell if there are any
       index entries that are obsolete - that is, if the cell tag
       specified by the entry in the index list still exists in 
       the notebook or not. Remove those that do not exist, and 
       issue a message the the user may want to run 
       RemoveObsoleteIndexEntries
    *)
    
    tags = "iCellTag" /. (List @@@ raw);
    pos = Position[outline, #]& /@ tags;
    
    obsolete = Position[pos, {}, {1}]//Flatten;
    obsolete = Part[tags, obsolete];
    If[obsolete === {},
      pos = #[[1,1]]& /@ pos;
      ,
      Message[MakeIndex::obs, Length[obsolete]];
      obsolete =
        If[Length[obsolete]===1, First @ obsolete, Alternatives @@ obsolete];
      raw = DeleteCases[raw, IndexData[___, "iCellTag" -> obsolete, ___]];
      tags = DeleteCases[tags, obsolete];
      pos = Position[outline, #][[1,1]]& /@ tags;
    ];
    
  ];

  If[raw === {}, Return[raw]];
  
  If[ci,
    raw = MapThread[Append[#1, "CellIndex" -> #2]&, {raw, pos}]
  ];
  
  If[cp,
    num = NotebookLookup[nbfile, "Numeration"];
    raw = MapThread[Join[#1, IndexData["CellPage" -> #2, "Numeration" -> num]]&,
     {raw, NotebookLookup[nbfile, "CellPage"][[pos]]}]
   ];
  
  If[ct, raw = MapThread[Append[#1, "CellTags" -> #2]&,
    {raw, Flatten[Cases[#, _[CellTags,x_]:>x]]& /@ outline[[pos]]}]
  ];
  
  raw
  
]  

MakeIndex::obs = "Obsolete index entries detected. Ignoring `1` obsolete entries.";


(*
   MakeIndex[nbfile, format, opts] determines where an index file
   will be saved, calls MakeIndexNotebook[nbfile, format, opts],
   and opens and saves the resulting notebook.
*)


Options[MakeIndex] =
{
  IndexFileName -> "Index.nb",
  ColumnHeights -> {20, 30}
};


MakeIndex[nb_NotebookObject, format_String, opts___] :=
Module[{nbfile, result},
  NotebookSaveWarning[nb, MakeIndex];
  nbfile = NotebookFilePath[nb];
  result = MakeIndex[nbfile, format, opts];
  SetSelectedNotebook[result]
] /; !ProjectDialogQ[nb]


MakeIndex[nb_NotebookObject, format_String, opts___]:=
Block[{},
  AuthorTools`MakeProject`Private`ProjectDialogFunction["SaveWarning", nb];
  MakeIndex[ProjectFileLocation[nb], format, opts]
]/; ProjectDialogQ[nb]


MakeIndex[nbfile_String, format_String, opts___] :=
Block[{nb, indexname},
  indexname = IndexFileName /. Flatten[{opts, Options[MakeIndex]}];
  indexname = ToFileName[ProjectDirectory[nbfile], indexname];
  
  nb = NotebookPut[ MakeIndexNotebook[nbfile, format, opts] ];
  NotebookSave[nb, indexname];
  nb
] /; FileType[nbfile] === File


MakeIndex[arg_NotebookObject | arg_String, opts___?OptionQ] :=
  MakeIndex[arg, "Simple", opts]


(**
  MakeIndexNotebook[_, "Expression", __]
**)


MakeIndexNotebook[file_String, "Expression", opts___] :=
Block[{raw, pn},
  raw = RawIndex[file,
    IncludeCellPage -> False, IncludeCellTags -> False];
  pn = ProjectName[file];
  
  Notebook[{
      Cell[pn <> $Resource["Index", "List"], "ContentsTitle"],
      Cell[BoxData @ MakeBoxes[#, StandardForm]& @ raw, "Input"]
    },
    StyleDefinitions -> $AuthorToolsStyleDefinitions,
    ScreenStyleEnvironment -> "Brackets"
  ]
]



(**
  MakeIndexNotebook[_, "Simple", __]
**)


MakeIndexNotebook[file_String, "Simple", opts___] :=
Block[{raw, pn},
  raw = RawIndex[file,
    IncludeCellPage -> False, IncludeCellTags -> False];
  pn = ProjectName[file];
  
  Notebook[Flatten[{
      Cell[pn <> $Resource["Index", "Index"], "ContentsTitle"],
      If[raw === {}, noIndexEntries, simpleCell /@ raw]
    }],
    StyleDefinitions -> $AuthorToolsStyleDefinitions,
    ScreenStyleEnvironment -> "Brackets"
  ]
]


noIndexEntries =
  Cell[$Resource["Index", "No index"], "Index"];


displayIndexEntry[e1_,  ""     ] := {e1, ", "};
displayIndexEntry[e1_, e2_     ] := {e1, ", ", e2, ", "};

displayIndexEntry[e1_,  "",  ""] := {e1, ", "};
displayIndexEntry[e1_, e2_,  ""] := {e1, ", ", e2, ", "};
displayIndexEntry[e1_,  "", e3_] := {e1, ", , ", e3, ", "};
displayIndexEntry[e1_, e2_, e3_] := {e1, ", ", e2, ", ", e3, ", "};


hyperlinkButton[contents_, nb_, tag_] :=
ButtonBox[contents,
  ButtonStyle->"Hyperlink",
  ButtonData->{FrontEnd`FileName[{}, nb], tag}
]


addOnsLinkButton[None | "None", itag_] :=
ButtonBox[
  StyleBox[$Resource["Index", "not found"],
    FontWeight->"Bold",
    FontColor->$Resource["ErrorText"]
  ],
  ButtonStyle -> "AddOnsLink",
  ButtonData -> "Note on Browser Index",
  ButtonNote -> itag
]


addOnsLinkButton[bctag_, itag_] :=
ButtonBox[bctag,
  ButtonStyle -> "AddOnsLink",
  ButtonData -> {bctag, itag}
]


simpleCell[IndexData[opts___]] :=
Block[{nbfile, itag, e1, e2, e3},
  {nbfile, itag, e1, e2, e3} = 
    {"File", "iCellTag", "MainEntry", "SubEntry", "ShortEntry"} /. {opts};
  
  Cell[TextData[Flatten[{
    displayIndexEntry[e1, e2, e3],
    hyperlinkButton[nbfile <> " (" <> itag <> ")", nbfile, itag]
  }]], "Index"]
]


(**
  MakeIndexNotebook[_, "Book", __]
**)


MakeIndexNotebook[nbfile_String, "Book", opts___] :=
Block[{raw, pn},
  raw = RawIndex[nbfile,
    IncludeCellPage -> True, IncludeCellTags -> False];
  raw = addSortKeys /@ raw;
  pn = ProjectName[nbfile];
  $datafield = "File";
  
  Notebook[Flatten[{
      Cell[pn <> $Resource["Index", "Index"], "ContentsTitle"],
      If[raw === {}, noIndexEntries, bookCells @ raw]
    }],
    StyleDefinitions -> $AuthorToolsStyleDefinitions
  ]
]


(*
  addSortKeys uses the fact that the built-in Sort function
  will place "foo00bar" ahead of "foo bar". In fact, the 00's
  will come before any other characters. This is a trick so that
  we sort on a different key using the default sorting function,
  rather than customizing the sorting function and paying for it
  with a speed degradation.
  
  The StringReplace technique is to make sure that within strings,
  the sorted order is spaces first, then hyphens, then other 
  letters.
  
  We provide two versions of addSortKeys, because the BrowserIndex
  needs to sort based on the "ShortEntry" field, if one exists.
*)


addSortKeys[IndexData[opts___]]:=
Block[{e1, e2},
  {e1, e2} = {"MainEntry", "SubEntry"} /. {opts};
  IndexData[opts, "SortKey1" -> toSortKey[e1], "SortKey2" -> toSortKey[e2]]
]


toSortKey[s_String] :=
StringReplace[s,
    {" "->"0",
     "-"->"1",
     "\[Hyphen]"->"1",
     "."->"",

     "\[ScriptA]" -> "a", "\[ScriptB]" -> "b", "\[ScriptC]" -> "c",
     "\[ScriptD]" -> "d", "\[ScriptE]" -> "e", "\[ScriptF]" -> "f",
     "\[ScriptG]" -> "g", "\[ScriptH]" -> "h", "\[ScriptI]" -> "i",
     "\[ScriptJ]" -> "j", "\[ScriptK]" -> "k", "\[ScriptL]" -> "l",
     "\[ScriptM]" -> "m", "\[ScriptN]" -> "n", "\[ScriptO]" -> "o",
     "\[ScriptP]" -> "p", "\[ScriptQ]" -> "q", "\[ScriptR]" -> "r",
     "\[ScriptS]" -> "s", "\[ScriptT]" -> "t", "\[ScriptU]" -> "u",
     "\[ScriptV]" -> "v", "\[ScriptW]" -> "w", "\[ScriptX]" -> "x",
     "\[ScriptY]" -> "y", "\[ScriptZ]" -> "z",
     
     "\[ScriptCapitalA]" -> "A", "\[ScriptCapitalB]" -> "B", "\[ScriptCapitalC]" -> "C",
     "\[ScriptCapitalD]" -> "D", "\[ScriptCapitalE]" -> "E", "\[ScriptCapitalF]" -> "F",
     "\[ScriptCapitalG]" -> "G", "\[ScriptCapitalH]" -> "H", "\[ScriptCapitalI]" -> "I",
     "\[ScriptCapitalJ]" -> "J", "\[ScriptCapitalK]" -> "K", "\[ScriptCapitalL]" -> "L",
     "\[ScriptCapitalM]" -> "M", "\[ScriptCapitalN]" -> "N", "\[ScriptCapitalO]" -> "O",
     "\[ScriptCapitalP]" -> "P", "\[ScriptCapitalQ]" -> "Q", "\[ScriptCapitalR]" -> "R",
     "\[ScriptCapitalS]" -> "S", "\[ScriptCapitalT]" -> "T", "\[ScriptCapitalU]" -> "U",
     "\[ScriptCapitalV]" -> "V", "\[ScriptCapitalW]" -> "W", "\[ScriptCapitalX]" -> "X",
     "\[ScriptCapitalY]" -> "Y", "\[ScriptCapitalZ]" -> "Z",
     
     "\[GothicA]" -> "a", "\[GothicB]" -> "b", "\[GothicC]" -> "c",
     "\[GothicD]" -> "d", "\[GothicE]" -> "e", "\[GothicF]" -> "f",
     "\[GothicG]" -> "g", "\[GothicH]" -> "h", "\[GothicI]" -> "i",
     "\[GothicJ]" -> "j", "\[GothicK]" -> "k", "\[GothicL]" -> "l",
     "\[GothicM]" -> "m", "\[GothicN]" -> "n", "\[GothicO]" -> "o",
     "\[GothicP]" -> "p", "\[GothicQ]" -> "q", "\[GothicR]" -> "r",
     "\[GothicS]" -> "s", "\[GothicT]" -> "t", "\[GothicU]" -> "u",
     "\[GothicV]" -> "v", "\[GothicW]" -> "w", "\[GothicX]" -> "x",
     "\[GothicY]" -> "y", "\[GothicZ]" -> "z",

     "\[GothicCapitalA]" -> "A", "\[GothicCapitalB]" -> "B", "\[GothicCapitalC]" -> "C",
     "\[GothicCapitalD]" -> "D", "\[GothicCapitalE]" -> "E", "\[GothicCapitalF]" -> "F",
     "\[GothicCapitalG]" -> "G", "\[GothicCapitalH]" -> "H", "\[GothicCapitalI]" -> "I",
     "\[GothicCapitalJ]" -> "J", "\[GothicCapitalK]" -> "K", "\[GothicCapitalL]" -> "L",
     "\[GothicCapitalM]" -> "M", "\[GothicCapitalN]" -> "N", "\[GothicCapitalO]" -> "O",
     "\[GothicCapitalP]" -> "P", "\[GothicCapitalQ]" -> "Q", "\[GothicCapitalR]" -> "R",
     "\[GothicCapitalS]" -> "S", "\[GothicCapitalT]" -> "T", "\[GothicCapitalU]" -> "U",
     "\[GothicCapitalV]" -> "V", "\[GothicCapitalW]" -> "W", "\[GothicCapitalX]" -> "X",
     "\[GothicCapitalY]" -> "Y", "\[GothicCapitalZ]" -> "Z",
     
     "\[DoubleStruckA]" -> "a", "\[DoubleStruckB]" -> "b", "\[DoubleStruckC]" -> "c",
     "\[DoubleStruckD]" -> "d", "\[DoubleStruckE]" -> "e", "\[DoubleStruckF]" -> "f",
     "\[DoubleStruckG]" -> "g", "\[DoubleStruckH]" -> "h", "\[DoubleStruckI]" -> "i",
     "\[DoubleStruckJ]" -> "j", "\[DoubleStruckK]" -> "k", "\[DoubleStruckL]" -> "l",
     "\[DoubleStruckM]" -> "m", "\[DoubleStruckN]" -> "n", "\[DoubleStruckO]" -> "o",
     "\[DoubleStruckP]" -> "p", "\[DoubleStruckQ]" -> "q", "\[DoubleStruckR]" -> "r",
     "\[DoubleStruckS]" -> "s", "\[DoubleStruckT]" -> "t", "\[DoubleStruckU]" -> "u",
     "\[DoubleStruckV]" -> "v", "\[DoubleStruckW]" -> "w", "\[DoubleStruckX]" -> "x",
     "\[DoubleStruckY]" -> "y", "\[DoubleStruckZ]" -> "z",

     "\[DoubleStruckCapitalA]" -> "A", "\[DoubleStruckCapitalB]" -> "B",
     "\[DoubleStruckCapitalC]" -> "C", "\[DoubleStruckCapitalD]" -> "D",
     "\[DoubleStruckCapitalE]" -> "E", "\[DoubleStruckCapitalF]" -> "F",
     "\[DoubleStruckCapitalG]" -> "G", "\[DoubleStruckCapitalH]" -> "H",
     "\[DoubleStruckCapitalI]" -> "I", "\[DoubleStruckCapitalJ]" -> "J",
     "\[DoubleStruckCapitalK]" -> "K", "\[DoubleStruckCapitalL]" -> "L",
     "\[DoubleStruckCapitalM]" -> "M", "\[DoubleStruckCapitalN]" -> "N",
     "\[DoubleStruckCapitalO]" -> "O", "\[DoubleStruckCapitalP]" -> "P",
     "\[DoubleStruckCapitalQ]" -> "Q", "\[DoubleStruckCapitalR]" -> "R",
     "\[DoubleStruckCapitalS]" -> "S", "\[DoubleStruckCapitalT]" -> "T",
     "\[DoubleStruckCapitalU]" -> "U", "\[DoubleStruckCapitalV]" -> "V",
     "\[DoubleStruckCapitalW]" -> "W", "\[DoubleStruckCapitalX]" -> "X",
     "\[DoubleStruckCapitalY]" -> "Y", "\[DoubleStruckCapitalZ]" -> "Z",

     "\\[" -> ""
     }
]


toSortKey[t_] := toSortKey[
  StringJoin[BoxForm`Intercalate[Cases[{t},_String,Infinity],"00"]]
]



(*
To create a book index, we must:

+ get a list of unique main entries and sort them alphabetically.

+ all those elements with identical main entries and sub entries 
  can be combined into a single entry with multiple pages or a page 
  range.

+ all the remaining elements with identical main entries can be 
  grouped together so that the sub entries all fall beneath it.

+ all entries whose first character of the SortKey1 is the same 
  should be separated from the next group with an empty cell
  (separate the A's from the B's, etc.)
*)



bookCells[{}] := {};

(*
   The following version of bookCells[raw_] is superceded by the
   one below it, which separates alphabetic categories with blank
   cells.
*)

(*
bookCells[raw_] := 
Block[{sortedMainEntries, groupedByMainEntry},
  
  sortedMainEntries = Last /@ Union[{"SortKey1", "MainEntry"} /. (List @@@ raw)];

  groupedByMainEntry =
  Map[
    GroupedByMainEntry[Cases[raw, _[___,"MainEntry" -> #, ___]]]&,
    sortedMainEntries
  ];

  Flatten[bookCell /@ groupedByMainEntry]
]
*)


bookCells[raw_] := 
Block[{bins, sortedMainEntries, groupedByMainEntry},
  
  bins = Union[{"SortKey1", "MainEntry"} /. (List @@@ raw)];
  
  sortedMainEntries = Last /@ bins;
  bins = First /@ bins;
  bins = If[Head[#]=!=String, "", StringTake[#,1]//ToLowerCase]& /@ bins;
  
  groupedByMainEntry =
  Map[
    GroupedByMainEntry[Cases[raw, _[___,"MainEntry" -> #, ___]]]&,
    sortedMainEntries
  ];

  (*
  Don't insert an empty cell before the first cell, or before a
  cell whose sort key starts with the same letter as the previous
  one.
  *)

  Flatten[Table[
      {If[i===1 || bins[[i]] === bins[[i-1]], {}, Cell[" ", "Index"]],
       bookCell @ groupedByMainEntry[[i]]},
     {i, 1, Length @ bins}
    ]
  ]
]




(*
  GroupedByMainEntry contains a list that needs to be grouped by
  subentry and also by page.
  
  If we have entries of the form:

  foo, bar, p3
  foo, bar, p3
  foo, car, p5
  foo, car, p6
  foo, bar, p17
  foo,      p100
  foo, car, p101
  
  They should be formatted thusly:

  foo, 100
    bar, 3, 17
    car, 5-6, 101

  Make sense?
  
  Here's a more detailed spec from Andre for when to place the first
  subentry on the second line, and when to place it on the first
  line:

  Entry, 22
  
  Entry, subentry, 23
  
  Entry, 22
    subentry, 23
  
  Entry,
    subentry, 23
    another subentry, 24
  
  Entry, 22
    subentry, 23
    another subentry, 24
  
  Summary: always put the first subentry on the second line, unless
  there is only one.
*)




allPageButtons[lis_] :=
Module[{pageInfo},
  pageInfo = {"Numeration", "CellPage", $datafield, "iCellTag"} /. (List @@@ lis);
  pageInfo = Split[Sort @ pageInfo, First[#1] === First[#2] && #2[[2]] - #1[[2]] < 2 &];
  BoxForm`Intercalate[pageButtons /@ pageInfo, ", "]
]

pageButtons[{{num_, page_, nb_, tag_}}] :=
  hyperlinkButton[PageString[page, num], nb, tag];

pageButtons[{{num_, page1_, nb_, tag_}, ___, {_, page1_, _, _}}] :=
  pageButtons[{{num, page1, nb, tag}}];

pageButtons[{{num1_, page1_, nb_, tag_}, ___, {num2_, page2_, nb2_, tag2_}}] := 
{
  hyperlinkButton[PageString[page1, num1], nb, tag],
  $Resource["Index", "Dash"],
  hyperlinkButton[PageString[page2, num2], nb2, tag2]
};



bookCell[GroupedByMainEntry[lis_List]] :=
Block[{e1, e2, sortedSubEntries, groupedBySubEntry},

  e1 = "MainEntry" /. (List @@ First[lis]);
  
  sortedSubEntries = Last /@ Union[{"SortKey2", "SubEntry"} /. (List @@@ lis)];
  
  groupedBySubEntry =
  Map[
    GroupedBySubEntry[Cases[lis, _[___,"SubEntry" -> #, ___]]]&,
    sortedSubEntries
  ];
  
  (* If there's no entry without a subentry, pretend there is one. *)
  If[Length[sortedSubEntries] > 1 && First[sortedSubEntries] =!= "",
    PrependTo[sortedSubEntries, ""];
    PrependTo[groupedBySubEntry, GroupedBySubEntry[{}]];
  ];
  
  Flatten[{
    firstBookCell[e1, First @ sortedSubEntries, First @ groupedBySubEntry]
    ,
    MapThread[restBookCell[e1, #1, #2]&,
      {Rest @ sortedSubEntries, Rest @ groupedBySubEntry}
    ]
  }]
]



(*
   DeleteCases[#, ""] below works around a display bug in the
   front end. If there is an empty string within a TextData
   list, the display of hyperlinks is messed up.
*)

firstBookCell[e1_, "", GroupedBySubEntry[{}]] :=
  Cell[TextData @ DeleteCases[#, ""]& @ Flatten @
    {e1}, "Index"]

firstBookCell[e1_, "", GroupedBySubEntry[lis_List]] :=
  Cell[TextData @ DeleteCases[#, ""]& @ Flatten @
    {e1, ", ", allPageButtons @ lis}, "Index"]

firstBookCell[e1_, e2_, GroupedBySubEntry[lis_List]] :=
  Cell[TextData @ DeleteCases[#, ""]& @ Flatten @
    {e1, ", ", e2, ", ", allPageButtons @ lis}, "Index"]

restBookCell[e1_, e2_, GroupedBySubEntry[lis_List]] :=
  Cell[TextData @ DeleteCases[#, ""]& @ Flatten @
    {e2, ", ", allPageButtons @ lis}, "IndexSubentry"]






bookCell[x_] := Cell[ToString[x,InputForm],"Index"]




(**
  MakeIndexNotebook[_, "AddOnsBook", __]
**)


(*
   This format is identical to the "Book" format, except the
   hyperlinks point to locations in the help browser instead of
   directly to other notebook files. It uses most of the code for
   "Book", except with a custom link function so that the links
   are of the form

   ButtonBox[bctag, ButtonStyle -> "AddOnsLink",
   ButtonData->{bctag,itag}]

   instead of

   ButtonBox[page, ButtonStyle -> "Hyperlink",
   ButtonData->{file,tag}]

   Since the actual button creation happens several layers down,
   we use Block to replace the old hyperlinkButton function with
   the new addOnsBookButton.


   The rest of the work is done by the $datafield switch, which
   determines which raw database field is used as the first
   element of the ordered pair setting of ButtonData. For links
   outside the help browser, it should use "File"; for links
   inside the help browser it should use "BrowCatsTag".
*)


MakeIndexNotebook[nbfile_String, "AddOnsBook", opts___] :=
Block[{raw, pn, pd, hyperlinkButton},
  
  {pd, pn} = {"Directory", "Name"} /. ProjectInformation[nbfile];

  raw = RawIndex[nbfile,
    IncludeCellPage -> True, IncludeCellTags -> True];
  raw = addSortKeys /@ raw;
  raw = addCategoriesToRawIndex[raw, pd];
  
  $datafield = "BrowCatsTag";
  hyperlinkButton = addOnsBookButton;
  
  Notebook[Flatten[{
      Cell[pn <> $Resource["Index", "Index"], "ContentsTitle"],
      If[raw === {}, noIndexEntries, bookCells @ raw]
    }],
    StyleDefinitions -> $AuthorToolsStyleDefinitions
  ]
]

addOnsBookButton[contents_, nb_, tag_] :=
ButtonBox[contents,
  ButtonStyle->"AddOnsLink",
  ButtonData->{nb, tag}
]

addOnsBookButton[contents_, None | "None", tag_] :=
ButtonBox[
  StyleBox[contents, FontColor->$Resource["ErrorText"], FontWeight->"Bold"],
  ButtonStyle->"AddOnsLink",
  ButtonData -> "Note on Browser Index",
  ButtonNote -> tag
]








(**
  MakeIndexNotebook[_, "TwoColumn", __]
**)


MakeIndexNotebook[nbfile_String, "TwoColumn", opts___] :=
Block[{raw, pn, ch, cells},
  ch = ColumnHeights /. Flatten[{opts, Options[MakeIndex]}];

  raw = RawIndex[nbfile,
    IncludeCellPage -> True, IncludeCellTags -> False];
  raw = addSortKeys /@ raw;
  pn = ProjectName[nbfile];
  $datafield = "File";
  
  cells = bookCells[raw];
  cells = simplifyCells[cells];
  cells = twoColumnCells[cells, ch];
    
  Notebook[Flatten[{
      Cell[pn <> $Resource["Index", "Index"], "ContentsTitle"],
      If[cells === {}, noIndexEntries, cells]
    }],
    StyleDefinitions -> $AuthorToolsStyleDefinitions
  ]
]

simplifyCells[lis_] :=
Replace[lis,
  {
    Cell[s_String,___] :> s,
    Cell[TextData[{s__}], "IndexSubentry",___] :> {"\t", s},
    Cell[TextData[s_],___] :> s,
    BoxData[x_] :> Cell[BoxData[x]]
  },
  {1}
]


twoColumnTemplate[lis1_, lis2_] := 
Cell[BoxData @ FormBox[
    GridBox @ {
      {Cell @ TextData @ BoxForm`Intercalate[lis1,"\n"],
       Cell @ TextData @ BoxForm`Intercalate[lis2,"\n"]}},
  None],
"Index2Column",
GridBoxOptions -> {ColumnWidths -> 0.5, ColumnAlignments -> Left},
TaggingRules -> {"ColumnHeights" -> Map[Length, {lis1, lis2}]}
]



twoColumnCells[cells_, n_Integer] := twoColumnCells[cells, {n}];

twoColumnCells[cells_, {n__Integer}] := twoColumnCells[cells, {#, #}& /@ {n}];

(*
   Here's the algorithm for working with the list of pairs {a,b}:

   o If there are no cells left, you're done.

   o If there are at least a+b cells left, create the current
   page and continue.

   o If there are less than a+b cells left, recalculate a and b
   so that there are exactly a+b cells lieft, then create the
   current page and continue.
*)

twoColumnCells[cells_, lis:{{_Integer,_Integer}..}] :=
Block[{cellList, result, countLists, a, b},

  cellList = cells;
  countLists = lis;
  result = {};
  
  While[Length[cellList] > 0,
    {a,b} = First[countLists];
    If[Length[cellList] < a,
      {a,b} = {Length[cellList], 0},
      If[Length[cellList] < a+b, b = Length[cellList] - a]
    ];
    
    result = Join[result, {
      twoColumnTemplate[Take[cellList, a], Take[cellList, {a+1,a+b}]],
      Cell["", "RuledPageBreak"]
    }];
    
    cellList = Drop[cellList, a+b];
    If[Length[countLists] > 1, countLists = Rest[countLists]];
  ];
  result
]

(* Fall through case for bad input: *)
twoColumnCells[cells_, x_] := twoColumnCells[cells, {20,30}]






(**
  MakeIndexNotebook[_, "BrowserIndex", __]
**)

(*
  Read in the relevant browser category file.
  Add CopyTag, IndexTag, and other relevant info to the associated 
    items in the raw index.
  Build the browser index based on that.
*)


MakeIndexNotebook[nbfile_String, "BrowserIndex", opts___] :=
Block[{browCats, pd, pn, pf, raw},
  {pd, pn, pf} = {"Directory", "Name", "Files"} /.
    ProjectInformation[nbfile];
    
  raw = RawIndex[nbfile,
    IncludeCellPage -> False, IncludeCellTags -> True];
  raw = addShortEntry /@ raw;
  raw = addSortKeys /@ raw;
  raw = addCategoriesToRawIndex[raw, pd];
    
  Notebook[Flatten[{
      Cell[pn, "IndexSection", CellTags -> "MasterIndexHeading"],
      If[raw === {}, noIndexEntries, browserCells @ raw]
    }],
    StyleDefinitions->$AuthorToolsStyleDefinitions
  ]
]



browserCells[{}] := {};


(*
   We used to divide this up just by main entry for
   "BrowserIndex", as is done in the "Book" index. However, this
   doesn't allow the "ShortEntry" to be used to distinguish
   between index entries. Instead, we now divide this up first by
   main entry, and then those groups are divided by short entry.
   Each main/short grouping is then formatted separately.
*)


(*
browserCells[raw_] :=
Block[{sortedMainEntries, groupedByMainEntry},
  
  sortedMainEntries = Last /@ Union[{"SortKey1", "MainEntry"} /. (List @@@ raw)];

  groupedByMainEntry =
  Map[
    GroupedByMainEntry[Cases[raw, _[___,"MainEntry" -> #, ___]]]&,
    sortedMainEntries
  ];

  Flatten[browserCell /@ groupedByMainEntry]
]
*)



browserCells[raw_] :=
Block[{sortedMainEntries, lis, sortedShortEntries, groupedByMainEntry},
  
  sortedMainEntries = Last /@
    Union[{"SortKey1", "MainEntry"} /. (List @@@ raw)];
  
  (*
   As commented above, groupedByMainEntry is somewhat of a
   misnomer here. It would more accurately be
   groupedByMainAndShortEntry, but that's much too long.
  *)
  
  groupedByMainEntry =
  Map[(
    lis = Cases[raw, _[___,"MainEntry" -> #, ___]];
    sortedShortEntries = Union["ShortEntry" /. (List @@@ lis)];    
    Map[
      GroupedByMainEntry[Cases[lis, _[___,"ShortEntry" -> #, ___]]]&,
      sortedShortEntries
    ])&,
    sortedMainEntries
  ]//Flatten;

  Flatten[browserCell /@ groupedByMainEntry]
]




(* We need to separate each entry out by subentries. *)

browserCell[GroupedByMainEntry[lis_List]] :=
Block[{e1, e2, e3, sortedSubEntries, groupedBySubEntry},
  {e1, e3} = {"MainEntry", "ShortEntry"} /. (List @@ First[lis]);
  
  sortedSubEntries = Last /@ Union[{"SortKey2", "SubEntry"} /. (List @@@ lis)];
  
  groupedBySubEntry =
  Map[
    GroupedBySubEntry[Cases[lis, _[___,"SubEntry" -> #, ___]]]&,
    sortedSubEntries
  ];
  
  (* If there's no entry without a subentry, pretend there is one. *)
  If[Length[sortedSubEntries] > 1 && First[sortedSubEntries] =!= "",
    PrependTo[sortedSubEntries, ""];
    PrependTo[groupedBySubEntry, GroupedBySubEntry[{}]];
  ];
  
  Flatten[{
    firstBrowserCell[e1, First @ sortedSubEntries, e3, First @ groupedBySubEntry]
    ,
    MapThread[restBrowserCell[e1, #1, e3, #2]&,
      {Rest @ sortedSubEntries, Rest @ groupedBySubEntry}
    ]
  }]
]


firstBrowserCell[e1_, "", e3_, GroupedBySubEntry[{}]] :=
  Cell[TextData @ DeleteCases[#, ""]& @ Flatten @
    {e1}, "Index", CellTags -> e3]

firstBrowserCell[e1_, "", e3_, GroupedBySubEntry[lis_List]] :=
  Cell[TextData @ DeleteCases[#, ""]& @ Flatten @
    {e1, ", ", allAddOnsLinkButtons @ lis}, "Index", CellTags -> e3]

firstBrowserCell[e1_, e2_, e3_, GroupedBySubEntry[lis_List]] :=
  Cell[TextData @ DeleteCases[#, ""]& @ Flatten @
    {e1, ", ", e2, ", ", allAddOnsLinkButtons @ lis}, "Index", CellTags -> e3]

restBrowserCell[e1_, e2_, e3_, GroupedBySubEntry[lis_List]] :=
  Cell[TextData @ DeleteCases[#, ""]& @ Flatten @
    {e2, ", ", allAddOnsLinkButtons @ lis}, "IndexSubentry", CellTags -> e3]



allAddOnsLinkButtons[lis_]:=
Module[{pageInfo},
  pageInfo = {"BrowCatsTag", "iCellTag"} /. (List @@@ lis);
  pageInfo = First /@ Split[sortLinksByBrowCatsTag @ Union[pageInfo], First[#1] === First[#2]&];
  BoxForm`Intercalate[addOnsLinkButton @@@ pageInfo, ", "]
]


sortLinksByBrowCatsTag[lis_] := Part[lis, Ordering @ Map[stringToListWithIntegers, First /@ lis]]

stringToListWithIntegers[str_String] :=
Block[{res},
  res = Split[Characters[str], DigitQ[#1] === DigitQ[#2]&];
  res = Map[StringJoin, res] //. d_String?DigitQ :> ToExpression[d]
]




browserCellTag[x_, ""] := browserCellTag[x];
browserCellTag[x_, y_] := browserCellTag[y];
browserCellTag[x_String] := x;
browserCellTag[x_List] := StringJoin[browserCellTag /@ x];
browserCellTag[x_StyleBox] := browserCellTag[First @ x];
browserCellTag[x_] := ToString[x];


addShortEntry[IndexData[opts__]]:=
Block[{e1, e3},
  {e1, e3} = {"MainEntry", "ShortEntry"} /. {opts};
  IndexData[opts] /.
    Rule["ShortEntry", _] :> Rule["ShortEntry", browserCellTag[e1, e3]]
]


MakeIndex::nobc = "There must be a BrowserCategories.m file in the directory `1` to do that.";


(*
   addCategoriesToRawIndex is more generally useful than just
   Indexing. For example, it is used for creating the AddOnsBook
   TOC.
*)

addCategoriesToRawIndex[raw_, pd_] := 
Block[{browCats, tags, tagpat, browItems},

  browCats = ToFileName[{pd}, "BrowserCategories.m"];
  If[FileType[browCats] =!= File,
    MessageDisplay[MakeIndex::nobc, pd];
    Abort[]
  ];
  browCats = Cases[Get[browCats], _Item, Infinity];
  
  (* 
     Don't rely on defaults - expand settings for IndexTag and CopyTag
     to a list for all browcats items.
  *)
  
  browCats = Map[
    If[FreeQ[#, IndexTag], Append[#, IndexTag -> {First @ #}], #]&,
    browCats
  ];
  browCats = Map[
    If[FreeQ[#, CopyTag], Append[#, CopyTag -> First[Cases[#,_[IndexTag,x_]:>x]]], #]&,
    browCats
  ];
  browCats = browCats //. {
    _[CopyTag, s_String | s_Symbol] :> Rule[CopyTag, {s}],
    _[IndexTag, s_String | s_Symbol] :> Rule[IndexTag, {s}]
  };

  Map[(
    tags = ("CellTags" /. List @@ #);
    tagpat = If[Length[tags]===1,
      {___, First @ tags, ___},
      {___, Alternatives @@ tags, ___}
    ];
    browItems = Cases[browCats, Item[_, "File" /. List @@ #, ___]];

    browItems = Cases[browItems,
        Item[a_, ___, CopyTag -> {None} | tagpat, ___, IndexTag -> x_, ___] |
        Item[a_, ___, IndexTag -> x:tagpat, ___] :>
      Thread[{a,x}]
    ]//Flatten;
    
    Append[#, "BrowCatsTag" -> If[Length[browItems]>=2, browItems[[2]], None]]
    )&,
    raw
  ]

]





convertEntriesToNewFormat::usage = "convertEntriesToNewFormat[nb] copies
index entries from old locations in nb to new locations in nb.  The old
locations were used in version 0.00 through 0.94 of this package; the
new locations are in effect starting with version 0.95.";
(* added 1998.05.28 by Lou *)

convertEntriesToNewFormat[nb_] :=
Module[{tRules, entries, tag, allEntries = {}},
  SelectionMove[nb, Before, Notebook];
  SelectionMove[nb, Next, Cell];
  SelectionMove[nb, All, Cell];
  While[(tRules = Options[NotebookSelection[nb]]) =!= $Failed,
    tRules = Cases[tRules, _[TaggingRules, x_]:>x];
    tag = Cases[tRules, _["IndexingCellTag", x_]:>x, Infinity];
    entries = Cases[tRules, _["IndexEntries", x_]:>x, Infinity];
    If[tag =!= {} && entries =!= {},
      tag = First[tag];
      entries = First[entries];
      allEntries = Join[allEntries, Join[{tag}, #]& /@ entries]
    ];
    SelectionMove[nb, Next, Cell]
  ];
  setNotebookIndexEntries[nb, allEntries]
];




CleanIndex::none = "No obsolete entries found.";
CleanIndex::n = "`1` obsolete entries removed.";


CleanIndex[arg_] :=
Block[{result},
  result = ExtendNotebookFunction[{cleanIndex, "CleanIndex"}, arg];
  result = Total[result];
  If[ result > 0,
    MessageDisplay[CleanIndex::n, result],
    MessageDisplay[CleanIndex::none]
  ];
  result
]/; ProjectDialogQ[arg] || (Head[arg] === String && FileType[arg] === File)


CleanIndex[nb_NotebookObject] := 
Block[{result},
  result = cleanIndex[nb];
  If[ result > 0,
    MessageDisplay[CleanIndex::n, result],
    MessageDisplay[CleanIndex::none]
  ];
  result
]



cleanIndex[nb_NotebookObject] :=
Module[{pre, post, allTags, obsolete},
  pre = getNotebookIndexEntries[nb];
  allTags = Flatten @ NotebookCellTags[nb];
  post = Select[pre, MemberQ[allTags, First[#]]&];
  obsolete = Complement[First /@ pre, First /@ post];
  If[obsolete =!= {}, setNotebookIndexEntries[nb, post]];
  Length[obsolete]
]



RemoveIndex::done1 = "Index removed from `1`. Close the notebook without saving to restore its index entries.";
RemoveIndex::done2 = "All index entries have been removed from the project.";


RemoveIndex[file_String] :=
Block[{},
  ExtendNotebookFunction[{removeIndex, "RemoveIndex"}, file];
  file
] /; FileType[file] === File


RemoveIndex[proj_NotebookObject] :=
Block[{},
  ExtendNotebookFunction[{removeIndex, "RemoveIndex"}, proj];
  MessageDisplay[RemoveIndex::done2];
  proj
] /; ProjectDialogQ[proj]


RemoveIndex[nb_NotebookObject] :=
Block[{},
  removeIndex[nb];
  MessageDisplay[RemoveIndex::done1, NotebookName @ nb];
  nb
]



removeIndex[nb_NotebookObject] :=
Module[{n, nb2, t1, t2, otherrules,tags},
  n = nextIndexingNumber[nb];
  
  {t1, otherrules} = notebookIndexAndTaggingRules[nb];
  t1 = First /@ t1;
  t2 = Table[$IndexingCellTagPrefix <> ToString[i], {i,1,n}];
  tags = Union[t1, t2];
  
  SelectionMove[nb, All, Notebook];
  SelectionRemoveCellTags[nb, tags];
  
  SetOptions[nb, TaggingRules -> DeleteCases[
    otherrules, _["NextIndexingNumber",_]]
  ];
]


End[];

EndPackage[];


