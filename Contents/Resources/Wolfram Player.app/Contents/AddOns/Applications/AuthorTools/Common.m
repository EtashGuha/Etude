(* :Context: AuthorTools`Common` *)

(* :Author: Louis J. D'Andria *)

(* :Summary:
    This package defines some commonly used notebook
    manipulation commands.
*)

(* :Copyright: *)

(* :Package Version: $Revision: 1.113 $ $Date: 2017/01/11 23:16:58 $ *)

(* :Mathematica Version: 5.0 *)

(* :History:

*)

(* :Keywords:
     
*)

(* :Discussion:
    
*)

(* :Warning:
    
*)


(*
   Some symbols used by the front end may not exist, or may exist in
   the Global` context, which can cause problems. The following code
   fixes that.
*)

Scan[ {Remove @@ Names["Global`"<>#], ToExpression["System`"<>#]}&,
  {"BrowserCategory", "Delimiter", "IndexTag", "Item",
   "LayoutInformation", "Page", "Saveable", "CounterValues",
   "CopyTag", "InlineCell", "NotebookReleaseHold",
   "HelpDirectoryListing"}];

(* One more fix necessary, because of the coreliance of MakeProject.m and Common.m *)

AuthorTools`Common`messageDialog;
AuthorTools`Common`MessageDisplay;

AuthorTools`MakeProject`Private`ProjectDialogFunction
AuthorTools`MakeProject`ProjectDataQ
AuthorTools`MakeProject`ProjectDialogQ
AuthorTools`MakeProject`ProjectFileLocation
AuthorTools`MakeProject`ProjectInformation





BeginPackage["AuthorTools`Common`", "AuthorTools`MakeProject`"]


HorizontalInsertionPointQ::usage = "HorizontalInsertionPointQ[nb] returns True if a notebook's insertion point is outside any cell or cell bracket, and False otherwise.";

CellsSelectedQ::usage = "CellsSelectedQ[nb] returns True if the selection in the indicated notebook is one or more cells, and False otherwise. CellsSelectedQ[nb, {min, max}] checks that the number of selected cells is in the indicated range.";

SelectionMoveAfterCell::usage = "SelectionMoveAfterCell[nb] moves the insertion point after the current cell in the given notebook. If the current selection is already between cells, the selection is unchanged.";

TemplateCell::usage = "TemplateCell[\"Styled\", style] returns a template cell expression with the given cell style. TemplateCell[\"Styled\", style, content] uses the given content for the template cell.";

NotebookWriteTemplateCell::usage = "NotebookWriteTemplateCell[nb, args] writes TemplateCell[args] after the current cell in the given notebook.";

IncludeCellIndex::usage = "IncludeCellIndex is an option that determines whether a cell's numeric index (the first cell in a notebook has an index of 1, the second an index of 2, etc.) should be included in the output.";

IncludeCellPage::usage = "IncludeCellIndex is an option that determines whether the page on which a cell ends should be included in the output.";

IncludeCellTags::usage = "IncludeCellTags is an option that determines whether the list of cell tags from a cell should be included in the output.";

NotebookFolder::usage = "NotebookFolder[nb] returns the path to the directory containing nb, or $Failed if there is no such directory.";

NotebookName::usage = "NotebookName[nb] returns the file name, not the full path, for the specified notebook, and returns \"Untitled-n\" if nb is not yet saved.";

NotebookModifiedQ::usage = "NotebookModifiedQ[nb] returns True if the indicated notebook has been edited since it was last saved, and False otherwise.";

NotebookSaveWarning::usage = "NotebookSaveWarning[nb, func] issues a warning message if the given notebook isn't saved before being passed to the specified function.";

NotebookRevert::usage = "NotebookRevert[nb] opens a fresh copy of nb, discarding any changes that have not been saved.  NotebookRevert[nb, Interactive->True] first asks the user if they wish to save the changes.";

RememberOpenNotebooks::usage = "RememberOpenNotebooks[] caches the currently open notebooks for NotebookCloseIfNecessary.";

NotebookCloseIfNecessary::usage = "NotebookCloseIfNecessary[nbObj] closes the given notebook if it was not open the last time RememberOpenNotebooks was called.";

NotebookScan::usage = "NotebookScan[nb,func] scans nb cell by cell, and executes the function func[nb] as each cell is selected.";

FlattenCellGroups::usage = "FlattenCellGroups[nbExpr] returns the notebook expression with all the cell groupings structures removed.";

messageDialog::usage = "messageDialog[symbol::tag, e1, e2, ...] displays the indicated message in a dialog-like notebook with an OK button.";

MessageDisplay::usage = "MessageDisplay[symbol::tag, e1, e2, ...] displays the indicated message in a dialog-like notebook if the evaluation began from a button click, and through the normal message stream otherwise.";

ProgressDialog::usage = "ProgressDialog[caption, subcaption] displays caption and subcaption in a dialog-looking notebook with no buttons.  See also ProgressDialogClose, ProgressDialogSetSubcaption.";

ProgressDialogSetSubcaption::usage = "ProgressDialogSetSubcaption[nb, subcaption] sets the subcaption in the given progress dialog to the desired string.";

ProgressDialogSetRange::usage = "ProgressDialogSetRange[nb, range] sets desired range of the given progress dialog.";

ProgressDialogSetValue::usage = "ProgressDialogSetValue[nb, value] sets the current value for the given progress dialog.";

ProgressDialogClose::usage = "ProgressDialogClose[nbobj] closes the given progress dialog.";


SelectedCellStyles::usage = "SelectedCellStyles is an option that determines what cells from the user's notebook are itemized when building a table of contents or browser categories.  It can be set to a list of style names.";

VersionCheck::usage = "VersionCheck[v] aborts the current evaluation if $VersionNumber is less than v, and notifies the user that they need a more recent copy of Mathematica.";


NotebookLookup::usage = "NotebookLookup[nb,obj] looks up in notebook nb the objects specified by obj, which can be any one of {\"CellOutline\", \"CellIndex\", \"CellExpression\", \"CellPage\"}.  NotebookLookup[nb,obj,pat] returns only those objects whose summary cell matches the given pattern.";

NotebookFileOutline::usage = "NotebookFileOutline[nb] returns the notebook file outline cache for the saved notebook nb.";

NotebookFileOptions::usage = "NotebookFileOptions[nb] returns a list of the notebook options read from from the saved notebook nb.";

NotebookCacheValidQ::usage = "NotebookCacheValidQ[nb] returns true if the given notebook contains what appears to be valid cache data, and false otherwise.";

ExtractCells::usage = "ExtractCells[nb,{cells}] returns the cell expresssions corresponding to the list of cell summaries extracted from the notebook file outline.";

NotebookFilePath::usage = "NotebookFilePath[nbObj] returns the full path to the given notebook object as specified by NotebookInformation.";


AddCellTags::usage = "AddCellTags[nbObj,tags] adds the string or list of strings to the cell tags of the currently selected cell(s) in nbObj. AddCellTags[nbObj,tags,n] adds the tags to the nth cell. AddCellTags[nbObj,{tags1,...,tagsk},{n1,...,nk}] adds tagsi to the ni-th cell.";

NotebookCellTags::usage = "NotebookCellTags[nb] returns a list of the cell tags present in nb.";

RemoveCellTags::usage = "RemoveCellTags[nbObj,pat] removes all cell tags from nbObj that match the string pattern pat. RemoveCellTags[nbObj,All] removes every cell tag from every cell in nbObj.";

SelectionRemoveCellTags::usage = "SelectionRemoveCellTags[nbObj,tags] removes the string or list of strings from the cell tags of the currently selected cell(s) in nbObj. SelectionRemoveCellTags[nbObj,All] removes all cell tags from the current selection.";


SetOptionsDialog::usage = "SetOptionsDialog[function, option] allows the user to set the specified option in a dialog-like window.";

optionValues::usage = "optionValues[opt] can be set to a list of possible option settings which are presented by SetOptionsDialog[f, opt]. optionValues[f,opt] can store different groups of values for the same option used by several functions.";

OpenAuthorTool::usage = "OpenAuthorTool[name] opens the palette with the specified name from AuthorTools layout. OpenAuthorTool[] returns links for opening any of the main palettes.";

ExtendNotebookFunction::usage = "ExtendNotebookFunction[function, nb, args] is a convenience that defines function[nb, args] when nb is a notebook file, project file, or project notebook, in terms of function[_NotebookObject, args].";

$Resource::usage = "$Resource is an internal symbol.";

$AuthorToolsStyleDefinitions::usage = "$AuthorToolsStyleDefinitions determines the style sheet used by any notebook file AuthorTools generates.";

PageString::usage = "PageString is an internal symbol.";


Begin["`Private`"]


(* determining whether the insertion point is in a cell *)


HorizontalInsertionPointQ[nb_NotebookObject] := 
  Developer`CellInformation[nb] === $Failed


SelectionMoveAfterCell[nb_NotebookObject] :=
(
  If[Not @ HorizontalInsertionPointQ @ nb, SelectionMove[nb, After, Cell]];
  Null
)


$CellsSelected = {1, Infinity}

CellsSelectedQ[nb_NotebookObject] := CellsSelectedQ[nb, $CellsSelected]

CellsSelectedQ[nb_NotebookObject, n_Integer] := CellsSelectedQ[nb, {n, n}]

CellsSelectedQ[nb_NotebookObject, {min_, max_}] :=
  MatchQ[
    Developer`CellInformation[nb],
    info : {{___, "CursorPosition" -> "CellBracket", ___}..} /;
      min <= Length[info] <= max
  ]

CellsSelectedQ[___] := False





TemplateCell["Styled", style_String, data_, opts___]:= Cell[data, style, opts]

TemplateCell["Styled", style_String]:= Cell[StringJoin[style <> " styled text..."], style]

TemplateCell["Table", style_String]:= Cell[StringJoin[style <> " styled text..."], style]

TemplateCell[___] := {}

NotebookWriteTemplateCell[nb_NotebookObject, fmt_, style_, data_, opts___]:=
(
  SelectionMoveAfterCell[nb];
  NotebookWrite[nb, TemplateCell[fmt, style, data, opts]];
)





(* getting the name of a notebook *)

stringToFilename[str_] :=
  StringReplace[str, {DirectoryName[str] -> "", $PathnameSeparator -> ""}]

NotebookName[nbFile_String] :=
If[FileType[nbFile] =!= File,
  $Failed,
  stringToFilename[nbFile]
] 

(*
   To get the name of a notebook object, use the "WindowTitle" return
   value from NotebookInformation, which works even for untitled
   notebooks. If the WindowTitle setting for the notebook is set to
   "FullFileName", this return value needs to be whittled down to just
   the name.
*)

NotebookName[nb_NotebookObject?NotebookOpenQ] :=
If["FullFileName" === (WindowTitle /. Options[nb, WindowTitle]),
  stringToFilename["WindowTitle" /. NotebookInformation[nb]],
  "WindowTitle" /. NotebookInformation[nb]
]


(* getting the location of a notebook.
   The names NotebookPath and NotebookDirectory
   are already taken by front end options.
*)

NotebookFolder[nbFile_String] :=
Switch[FileType[nbFile],
  File, DirectoryName[nbFile],
  Directory, nbFile,
  _, $Failed
];

NotebookFolder[nb_NotebookObject?NotebookOpenQ] := 
  If[FileType[#]===Directory, #, $Failed]& @ ToFileName @
    First["FileName" /. NotebookInformation[nb] /.
      "FileName" -> {"$Failed"}]


NotebookFilePath[nbFile_String] :=
  If[FileType[nbFile] === File, nbFile, $Failed]

NotebookFilePath[nb_NotebookObject?NotebookOpenQ] :=
  If[# === $Failed, $Failed,
    ToFileName[{#}, NotebookName @ nb]]& @ NotebookFolder[nb]
    



(*
   NotebookOpenQ used to check Notebooks[] for its return value in all
   cases. Now, when given a notebook object, it checks
   NotebookInformation[nbObj] instead. This is so NotebookOpenQ will
   return True when passed a style sheet notebook object, which is
   invisible to Notebooks[]. That means NotebookName, NotebookFolder,
   and NotebookFilePath, eg, will work with style sheet notebook
   objects.

   It's a bit inconsistent that NotebookOpenQ use Notebooks[] for
   notebook files and NotebookInformation for notebook objects, but I
   think that's the best we can do for now.
*)


(*
NotebookOpenQ[nbObj_NotebookObject] := MemberQ[Notebooks[], nbObj];
*)

NotebookOpenQ[nbObj_NotebookObject] := ListQ[NotebookInformation @ nbObj];

NotebookOpenQ[nbFile_String] :=
  MemberQ[NotebookFilePath /@ Notebooks[], nbFile];



NotebookModifiedQ::cant = "This version of Mathematica does not support notebook modification detection.";


NotebookModifiedQ[nbObj_NotebookObject?NotebookOpenQ] :=
Block[{mod},
  mod = "ModifiedInMemory" /.
    Flatten[{NotebookInformation[nbObj], "ModifiedInMemory" -> $Failed}];
  If[mod === $Failed, Message[NotebookModifiedQ::cant]];
  mod
]

(*
   If NotebookModifiedQ returns $Failed, then NotebookSaveWarning
   must save the notebook without asking -- there's really no
   alternative.
*)

NotebookSaveWarning::nbmod = "The notebook must be saved to use the function `1`."

NotebookSaveWarning[nb_NotebookObject?NotebookOpenQ, func_] :=
Switch[NotebookModifiedQ[nb],
  True,
    MessageDisplay[NotebookSaveWarning::nbmod, func];
    Abort[],
  False,
    Null,
  $Failed,
    NotebookSave[nb]
]



(* discarding changes in a notebook *)

Options[NotebookRevert] = {Interactive -> False};

NotebookRevert::uns = "NotebookRevert cannot work on the notebook `1` because it has never been saved.";

NotebookRevert[nb_NotebookObject?NotebookOpenQ, opts___] := 
Block[{path, asktosave},
  If[ NotebookModifiedQ[nb] === False, Return[nb]];
  
  path = NotebookFilePath[nb];
  If[path === $Failed || FileType[path] =!= File,
    Message[NotebookRevert::uns, nb];
    Return[$Failed]
  ];
  
  asktosave = TrueQ[Interactive /. Flatten[{opts, Options[NotebookRevert]}]];
  NotebookClose[nb, Interactive -> asktosave];
  NotebookOpen[path]
]




(* closing only notebooks that weren't already open *)

RememberOpenNotebooks[] := ($RememberedNotebooks = Notebooks[])

NotebookCloseIfNecessary[nb_NotebookObject] := NotebookClose[nb] /;
  !MemberQ[$RememberedNotebooks, nb]




(**************** ProgressDialog Functions ****************)


(* generic dialog objects and options *)

dialogOpts = {
  Active->True,
  Editable->False,
  Background->$Resource["GrayBackground"],
  WindowFrame->"Normal",
  WindowSize->{300, 150},
  WindowElements->{},
  Saveable -> False,
  WindowFrameElements->{},
  WindowFloating->True,
  WindowTitle -> None,
  ShowCellBracket->False
};

$userprogopts = {};

messageCell[message_] :=
Cell[BoxData[GridBox[{{
      $Resource["WarningIcon"],
      Cell[message,FontFamily -> "Times"]}},
    RowAlignments -> "Top",
    ColumnWidths -> {Automatic, 15}]],
  "Text",
  FontSize->14,
  CellTags->"caption",
  CellMargins -> {{15,Inherited},{Inherited, Inherited}}];

captionCell[caption_, opts___] :=
Cell[BoxData[GridBox[{{
      $Resource["ClockIcon"],
      Cell[caption, FontFamily -> "Times"]}},
    RowAlignments -> "Center",
    ColumnWidths -> {Automatic, 15}]],
  "Text",
  FontSize->14,
  CellTags->"caption",
  CellMargins -> {{15,Inherited},{Inherited, Inherited}},
  opts];

subcaptionCell[subcaption_, opts___] :=
Cell[BoxData[DynamicBox[$ProgressSubcaption]],
  "Text",
  TextAlignment->Right,
  CellMargins->{{10, Inherited},{Inherited,Inherited}},
  FontSize->12,
  FontFamily -> "Times",
  FontSlant->$Resource["Italic"],
  CellTags->"subcaption",
  FontColor->$Resource["Button1Background"],
  AutoSpacing->False,
  opts];


barberpoleCell[opts___] :=
Cell[BoxData[AnimatorBox[0, {0, Infinity},
    AppearanceElements->{"ProgressSlider"}, ImageSize->{280, Automatic}]],
 "Text",
  TextAlignment->Center,
  CellMargins->{{0,0},{0,0}},
  opts];


thermometerCell[opts___] :=
Cell[BoxData[ProgressIndicatorBox[
    Dynamic[$ProgressValue],
    Dynamic[$ProgressRange],
    ImageSize->{280, Automatic}]],
 "Text",
  TextAlignment->Center,
  CellMargins->{{0,0},{0,0}},
  opts];



okCell[ok_, opts___] :=
Cell[BoxData[ButtonBox[ok,
    Active->True,
    ButtonEvaluator->None,
    ButtonFunction:>(FrontEndExecute[{NotebookClose[ButtonNotebook[]]}]&),
    Background->$Resource["Button2Background"],
    ButtonMargins -> $Resource["WideMargin"]]],
  "Text",
  TextAlignment->Right,
  FontFamily->$Resource["Font"],
  FontSize->12,
  FontWeight->"Bold",
  FontColor->$Resource["Button2Text"],
  CellTags->"ok",
  opts];
  

(* messageDialog *)

(*
   ProgressDialog appears and disappears under code control. But
   if the user writes a script, they don't want to see
   unnecessary dialogs like messageDialog remaining on their
   screen. In that case, a Message[] is enough, or maybe even the
   return value of the function is enough.

   In most applications of messageDialog, you should check
   whether the current evaluation started with a button click. If
   it did, use messageDialog. If it didn't -- that is, if
   ButtonNotebook[] evaluates to $Failed -- then just use
   Message[] or the return value or whatever.
*)



messageDialog[caption_, args___] :=
Block[{str = caption},
  If[Head[str] === MessageName, str = ReplacePart[str, General, 1]];
  str = ToString @ StringForm[str, args];
  NotebookPut @ Notebook[
    {
      messageCell[str],
      okCell[$Resource["WideOK"]]
    },
    WindowElements -> {"VerticalScrollBar"},
    ScrollingOptions -> {"VerticalScrollRange" -> Fit},
    WindowFrameElements-> {"CloseBox", "ResizeArea"},
    WindowSize -> {FitAll,FitAll},
    Sequence @@ dialogOpts
  ]
]

SetAttributes[MessageDisplay, {HoldFirst}];

MessageDisplay[caption_, args___] :=
  If[ButtonNotebook[] === $Failed, Message, messageDialog][caption, args]


(* ProgressDialog *)

ProgressDialog[caption_, subcaption_] :=
(
  $ProgressSubcaption = subcaption;
  NotebookPut @ Notebook[
    {
      captionCell[caption],
      subcaptionCell[subcaption],
      barberpoleCell[]
    },
    WindowFrameElements -> "CloseBox",
    Sequence @@ dialogOpts
  ]
)


ProgressDialog[caption_, subcaption_, range_] :=
(
  $ProgressSubcaption = subcaption;
  $ProgressValue = 0;
  $ProgressRange = range;
  NotebookPut @ Notebook[
    {
      captionCell[caption],
      subcaptionCell[subcaption],
      thermometerCell[]
    },
    WindowFrameElements -> "CloseBox",
    Sequence @@ dialogOpts,
    Sequence @@ $userprogopts
  ]
)


ProgressDialogSetSubcaption[nb_, subcaption_] :=
  ($ProgressSubcaption = subcaption)


ProgressDialogSetRange[nb_, r_] :=
  ($ProgressRange = r)
    

ProgressDialogSetValue[nb_, n_] :=
  ($ProgressValue = n)
    

ProgressDialogClose[nbobj_] :=
  (If[ListQ[#], $userprogopts = #]& @ (* 58299 *) AbsoluteOptions[nbobj, WindowMargins];
   NotebookClose[nbobj, Interactive -> False])


(**************** End ProgressDialog Functions ****************)




(* scanning a notebook cell by cell, 
   doing/extracting something at each step 
*)


NotebookScan[file_String, func_] :=
  ExtendNotebookFunction[NotebookScan, file, func] /; FileType[file] === File


NotebookScan[proj_NotebookObject?ProjectDialogQ, func_] :=
  ExtendNotebookFunction[NotebookScan, proj, func]


NotebookScan[nb_NotebookObject?NotebookOpenQ, func_] :=
Module[{lis = {}},
  SelectionMove[nb, Before, Notebook];
  SelectionMove[nb, Next, Cell];
  SelectionMove[nb, All, Cell];
  While[Developer`CellInformation[nb] =!= $Failed,
    lis = Join[lis, {func[nb]}];
    SelectionMove[nb, Next, Cell]
  ];
  lis
]



(*
   It's fairly common to want just the block-level cells from a
   notebook, not Cell expression that happen to be inline, in button
   functions, tagging rules, style sheets, etc.
*)

FlattenCellGroups[nb_NotebookObject?NotebookOpenQ] := FlattenCellGroups[NotebookGet @ nb]

FlattenCellGroups[Notebook[cells_, opts___]] := Notebook[FlattenCellGroups[cells], opts]

FlattenCellGroups[{c__Cell}] := Flatten[FlattenCellGroups /@ {c}]

FlattenCellGroups[Cell[CellGroupData[{c__Cell}, ___], ___]] := FlattenCellGroups /@ {c}

FlattenCellGroups[x_] := x



(**************** NotebookLookup Functions ****************)

(*
   NotebookLookup and related functions NotebookFileOutline and
   NotebookFileOptions rely on the saved notebook's cache which
   the front end updates every time it saves the notebook.
*)

eol[st_InputStream] := Module[{str, pos, a, b},
  SetStreamPosition[st, 0];
  str = StringJoin @ DeleteCases[Read[st, Table[Character, {2000}]], EndOfFile];
  pos = StringPosition[str, "NotebookFileLineBreakTest"];
  If[!MatchQ[pos, {{_,_},{_,_}}],
    Return[$Failed]
  ];
  
  a = pos[[1,2]];
  b = pos[[2,1]];
  StringTake[str, {a+1, b-1}]
]


(*
   keyCacheLines returns a list of the special lines from the notebook's
   header which indicate if there is a cache, and where it should be if
   there is one. It only reads the first 40 lines, which should be more
   than sufficient to read the whole header. Notice the use of Record
   instead of String in Read, so that the end of line characters can be
   specified (and essentially ignored in this case).
*)

keyCacheLines[st_InputStream] :=
Module[{lines},
  SetStreamPosition[st, 0];
  lines = Table[
    Read[st, Record, RecordSeparators -> {"\n", "\r"}, NullRecords -> False],
    {40}
  ];
  {Select[lines, StringMatchQ[#, "(*CacheID:*)"] &],
   Select[lines, StringMatchQ[#, "*NotebookOptionsPosition[*]*"] &],
   Select[lines, StringMatchQ[#, "*NotebookOutlinePosition[*]*"] &], 
   Select[lines, StringMatchQ[#, "*CellTagsIndexPosition[*]*"] &]}
]


(*
   extractCacheValues takes one of the strings returned by keyCacheLines
   and reduces it to just a pair of integers. Each of those strings should
   be of a very particular structure, which is where the values in the
   second arg to StringTake came from.

   CacheLocations returns three integers, provided the cache is valid.
*)

extractCacheValues[str_String] := 
  ToExpression /@ StringCases[str, DigitCharacter..]
  

CacheLocations[st_InputStream] :=
Module[{lis},
  If[! NotebookCacheValidQ[st], Return[$Failed]];
  lis = extractCacheValues /@ Flatten[Rest @ keyCacheLines @ st];
  If[StringLength[eol[st]] > 1, Plus @@@ lis, First /@ lis]
]



(*
   Determining whether a notebook's cache is valid is a little tricky.
   Through discussions with John Fultz, I've decided on the following
   simple procedure:

   o Check to make sure that exactly one copy of each key cache line exists.
  
   o Make sure that the stream values actually exist in the notebook,
     and that the first few characters at those locations are what they
     should be.
*)


NotebookCacheValidQ[nbObj_NotebookObject?NotebookOpenQ] :=
  NotebookCacheValidQ[NotebookFilePath[nbObj]];

NotebookCacheValidQ[nb_String]:=
Block[{st, tmp},
  st = OpenRead[nb];
  If[st === $Failed, Return[False]];
  tmp = NotebookCacheValidQ[st];
  Close[st];
  tmp
]

NotebookCacheValidQ[st_InputStream] :=
Module[{lines, vals, pos, n},
  lines = keyCacheLines[st];
  
  If[! MatchQ[lines, {{_}, {_}, {_}, {_}}], Return[False]];
    
  vals = extractCacheValues /@ Flatten[Rest @ lines];
  If[!MatchQ[vals,
    {{_Integer, _Integer}, {_Integer, _Integer}, {_Integer, _Integer}}],
    Return[False]
  ];
  
  n = StringLength[eol[st]];
  
  (* FrontEndVersion is no longer guaranteed to be the first option.... 
  pos = If[n === 2, Total @ vals[[1]], First @ vals[[1]]];
  SetStreamPosition[st, pos];
  If[StringJoin[Table[Read[st, Character], {15}]] =!= "FrontEndVersion", 
    Return[False]
  ];
  *)
  
  pos = If[n === 2, Total @ vals[[2]], First @ vals[[2]]];
  SetStreamPosition[st, pos];
  If[StringJoin[Table[Read[st, Character], {10}]] =!= "Notebook[{", 
    Return[False]
  ];
    
  pos = If[n === 2, Total @ vals[[3]], First @ vals[[3]]];
  SetStreamPosition[st, pos];
  If[StringJoin[Table[Read[st, Character], {13}]] =!= "CellTagsIndex", 
    Return[False]
  ];
  
  True
]



(*
   Some efficiency can be gained by only reading the a notebook's cache
   when necessary. If you've already read it, and the notebook hasn't 
   changed since then, then there's no need to read it again. Simply use
   the version you read in last time. I can't envision why a user would
   need to turn this "caching the cache" off, but here's a switch just
   in case.
*)


$CacheOutlinesQ=True;


(*
   Here's the payoff from all the cache gymnastics above. Read
   the file's outline cache by jumping to it in the notebook's
   file stream, and then read the following expression. Store it
   using CachedOutline and DateOfCachedOutline if $CacheOulinesQ
   is True.

   NotebookFileOutline and NotebookFileOptions avoid
   Read[st,Expression] due to performance issues. The
   implementations below use the fact that using RecordSeparators
   to read in a large chunk of data is about the fastest you can
   hope for without abandoning Read.

   A notebook's file outline will be everything between the
   position that the front end specifies and the following close
   comment deliminter '*' followed by ')' (which I can't write
   properly in this comment because M- would think I'm done
   talking).

   The new ParseFileToLinkPacket ties up the front end when
   handling large notebooks, so we stick with a kernel-only
   implementation using Read.
   
*)


(*
   ToExpression won't return properly if the wrong kind of new
   line is at the end of the string. So strip 'em:
*)

stripTrailingNewlines[str_String] :=
If[ MatchQ[StringTake[str,-1], "\n" | "\r"],
  stripTrailingNewlines[StringDrop[str,-1]],
  str
]


NotebookFileOutline[nb_] := 
Block[{nbfile, fd, st, expr},
  nbfile = NotebookFilePath[nb];
  If[FileType[nbfile] =!= File, Return[$Failed]];
  fd = FileDate[nbfile];
  
  If[$CacheOutlinesQ &&
      fd =!= $Failed &&
      DateOfCachedOutline[nbfile] === fd,
    Return[CachedOutline[nbfile]]
  ];
  
  st = OpenRead[nbfile];
  If[st === $Failed, Return[$Failed]];
  
  If[!NotebookCacheValidQ[st],
    Close[st];
    Message[NotebookFileOutline::nocache,nbfile];
    Return[$Failed]
  ];
  
  SetStreamPosition[st, CacheLocations[st][[2]] ];
  expr = ToExpression @ stripTrailingNewlines @
    Read[st, Record, RecordSeparators -> {"*)"}];
  Close[st];
  
  If[$CacheOutlinesQ,
    DateOfCachedOutline[nbfile] = fd;
    CachedOutline[nbfile] = expr
  ];
  
  expr
]


(*
   NotebookFileOptions reads the notebook's options directly
   from the file, relying on the key cache lines to determine
   where in the file the options begin. The options list is
   terminated when the function encounters a line with just a
   close bracket ']'. The result is cached as described above if
   $CacheOutlinesQ.
*)


NotebookFileOptions[nb_] := 
Module[{nbfile, fd, st, lf, str},
  nbfile = NotebookFilePath[nb];
  If[FileType[nbfile] =!= File, Return[$Failed]];
  fd = FileDate[nbfile];
  
  If[$CacheOutlinesQ &&
      fd =!= $Failed && 
      DateOfCachedOptions[nbfile] === fd,
    Return[CachedOptions[nbfile]]
  ];
  
  st = OpenRead[nbfile];
  If[st === $Failed, Return[$Failed]];
  
  If[!NotebookCacheValidQ[st],
    Close[st];
    Message[NotebookFileOptions::nocache, nbfile];
    Return[$Failed]
  ];
  
  lf = eol[st];
  
  SetStreamPosition[st, First @ CacheLocations[st]];
  str = Read[st, Record, RecordSeparators -> {lf<>"]"<>lf}];
  
  Close[st];
  
  If[$CacheOutlinesQ,
    DateOfCachedOptions[nbfile] = fd;
    CachedOptions[nbfile] = ToExpression @ StringJoin["{", str, "}"]
    ,
    ToExpression @ StringJoin["{", str, "}"]
  ]
]


NotebookFileOptions[nb_, opts_] := 
  Cases[NotebookFileOptions[nb], _[Alternatives @@ Flatten[{opts}], _]]


NotebookFileOutline::nocache = NotebookFileOptions::nocache =
"The notebook `1` has no cache or an invalid cache.";





(*
   ExtractCells assumes that each Cell expression in the notebook's
   file outline cache will have integers as their first four
   arguments, corresponding to: the stream position of the cell, the
   number of newlines before the cell, the length of the cell in
   bytes, and the number of newlines within the cell. The stream
   position and byte length count all newlines as one byte long, which
   is why we add the number of newlines to the other offsets when the
   file uses double-byte newlines.
*)



ExtractCells[_, {}] := {};

ExtractCells[nb_NotebookObject?NotebookOpenQ, c_] :=
  ExtractCells[NotebookFilePath[nb], c]

ExtractCells[nbFile_String, c_Cell]:=ExtractCells[nbFile, {c}]


ExtractCells[nbFile_String, {c__Cell}]:=
Block[{st, cells, lf, pos, len, func, str},
  
  st = OpenRead[nbFile];
  If[st === $Failed, Return[{}]];
  
  lf = eol[st];
  If[StringLength[lf] > 1,
    pos = Plus @@@ {c}[[All, {1,2}]];
    len = Plus @@@ {c}[[All, {3,4}]];
    func = StringReplace[#, lf -> "\n"]&;
    ,
    pos = {c}[[All, 1]];
    len = {c}[[All, 3]];
    func = Identity;
  ];
  cells = Table[
      SetStreamPosition[st, pos[[i]] ]; 
      str = StringJoin @ Read[st, Table[Character, { len[[i]] }]];
      ToExpression @ func @ str,
    {i, 1, Length @ {c}}];
  Close[st];
  Flatten[cells]
]


(*
   In several of the NotebookLookup functions below, there are
   efficiency tweaks when the user -- or more likely this package
   -- performs a lookup with pattern Cell[_Integer, ___]. This is
   the default third argument which will match every cell, so
   actually performing the pattern match is unnecessary.

   Even with this efficiency tweak, calling
   NotebookLookup[nb,"CellPage"] will still be slower than
   calling NotebookPaginationCache[nb] directly, since the
   NotebookLookup version checks to make sure that the length of
   the cache matches the number of cells in the notebook.
*)


NotebookLookup[nb_, what_] := NotebookLookup[nb, what, Cell[_Integer, ___]]

NotebookLookup[nb_, "NotebookOutline", ___] := NotebookFileOutline[nb]

NotebookLookup[
    nb_, what:("CellOutline" | "CellIndex" | "CellExpression"), pat_] :=
Block[{celloutlines, result},
  celloutlines=
    Cases[NotebookLookup[nb,"NotebookOutline"], Cell[_Integer, ___], Infinity];
  If[celloutlines === {}, Return[{}]];
  
  If[what === "CellIndex",
    If[pat === Cell[_Integer, ___],
      Return[Range @ Length @ celloutlines],
      Return[Flatten @ Position[celloutlines, pat, {1}]]
    ]
  ];
  
  If[pat =!= Cell[_Integer, ___], celloutlines = Cases[celloutlines, pat]];
  If[what === "CellOutline", Return[celloutlines]];
  If[what === "CellExpression", Return[ExtractCells[nb, celloutlines]]];
]


NotebookLookup[nb_, "CellPage", pat_]:=
Block[{pages, n, ind},
  Needs["AuthorTools`Pagination`"];
  n = Length[NotebookLookup[nb, "CellIndex"]];
  If[pat === Cell[_Integer, ___],
    ind = All,
    ind = NotebookLookup[nb, "CellIndex", pat]
  ];

  pages = AuthorTools`Pagination`NotebookPaginationCache @ nb;
  pages = If[MatchQ[pages, {_,{___Integer}, _}], pages[[2]], {}];
  
  If[Length[pages] === n,
    Part[pages, ind],
    Table["?", {If[ind === All, n, Length @ ind]}]]
]


NotebookLookup[nb_, "Numeration"]:=
Block[{num},
  Needs["AuthorTools`Pagination`"];
  num = AuthorTools`Pagination`NotebookPaginationCache @ nb;
  num = If[MatchQ[num, {_,{___Integer}, _}], Last @ num, Automatic];
  
  If[MemberQ[{"RomanNumerals", "CapitalRomanNumerals"}, num],
    num,
    Automatic
  ]
]



PageString[n_Integer, "CapitalRomanNumerals"] :=
  PageString[n, "RomanNumerals"]//ToUpperCase

PageString[n_Integer, "RomanNumerals"] :=
  StringJoin[ Table["m", {Quotient[n, 1000]}], toRoman[Mod[n, 1000]] ]

PageString[n_, _] := ToString[n];


(*
   Thanks to Robby Villegas for allowing me to use the following
   arabic-to-roman numeral utility, which can be found amid many more
   such converters here:
   http://library.wolfram.com/infocenter/Demos/4952/
*)

$romanDigitValues = {1, 5, 10, 50, 100, 500, 1000};

$romanDigitSymbols = {"i", "v", "x", "l", "c", "d", "m"};

$romanDigitDecompRules = Append[
      {4 -> {1, 5}, 9 -> {1,10}}, 
      digit_ :> Table[5, {Quotient[digit, 5]}] ~Join~
           Table[1, {Mod[digit, 5]}]
      ];

romanDigitDecomp[n_Integer /; 0 <= n <= 9] := Replace[n, $romanDigitDecompRules]

toRoman[n_Integer /; 0 <= n <= 999] :=
  Module[{digitList, romanValueLists, romanDigitLists},
    digitList = IntegerDigits[n, 10, 3];
    romanValueLists = romanDigitDecomp /@ digitList;
    romanDigitLists = MapThread[Replace[#1, Thread[{1, 5, 
        10} -> #2], {1}]&,
        {
          romanValueLists,
          Reverse @ Partition[$romanDigitSymbols, 3, 2]
          }
        ];
    StringJoin[romanDigitLists]
    ]



(**************** End NotebookLookup Functions ****************)




(* "You need a newer Mathematica to do that" *)

VersionCheck::"tooold" = "You need a more recent version of Mathematica to do that.  Please contact Wolfram Research Inc. (http://www.wolfram.com) for assistance.";

VersionCheck[v_] :=
If[$VersionNumber < v,
  MessageDisplay[VersionCheck::tooold];
  Abort[]
]




(**************** CellTagging Functions ****************)

NotebookCellTags[nb_] :=
Block[{lis = "check", f},

  lis = MathLink`CallFrontEnd[FrontEnd`GetCellTagsPacket[nb]];
    
  If[Not @ MatchQ[lis, {___String}],
    (* unsorted union *)
    f[x_] := (f[x]=Sequence[]; x);
    lis = Cases[NotebookLookup[nb,"CellOutline"], _[CellTags, x_] :> x, {2}];
    lis = Map[f, Flatten @ lis]
  ];
  lis
]

(*
   Using AddCellTags on a notebook *file* makes sense as long as
   you're using the three argument form -- the two argument form
   requires a selection.
*)

AddCellTags[nbFile_String, tags_, n_] :=
Block[{nb, nbs},
  nbs = Notebooks[];
  nb = NotebookOpen[nbFile];
  AddCellTags[nb, tags, n];
  NotebookSave[nb];
  If[!MemberQ[nbs, nb], NotebookClose[nb]];
  nbFile
] /; FileType[nbFile] === File;

AddCellTags[nb_NotebookObject, tag_String, args___] :=
  AddCellTags[nb, {tag}, args];

AddCellTags[nb_NotebookObject?NotebookOpenQ, {tags__String}] := 
(
  MathLink`CallFrontEnd[FrontEnd`SelectionAddCellTags[nb, {tags}]];
  nb
)

AddCellTags[nb_NotebookObject?NotebookOpenQ, {tags__String}, n_Integer] :=
(
  SelectionMove[nb, Before, Notebook]; 
  SelectionMove[nb, Next, Cell, n];
  MathLink`CallFrontEnd[FrontEnd`SelectionAddCellTags[nb, {tags}]];
  nb
)


(*
   Efficiency concerns have caused this function to be a little
   more complex.
*)

AddCellTags[nb_NotebookObject?NotebookOpenQ, tags_, {n__Integer}]:=
Block[{k, taglist, lis},
  
  k = Length[{n}];
  
  (* 
     A single string or list of strings will be added to every
     indicated cell. Otherwise, the i-th tag will be associated
     with the i-th indicated cell
  *)
  
  Switch[tags,
    _String,
      taglist = Table[{tags}, {k}],
    {__String},
      taglist = Table[tags, {k}],
    _List,
      If[Length[tags] < k,
        taglist = Table[Flatten[tags], {k}],
        taglist = Take[tags, k]
      ]
  ];
    
  lis = Transpose[{{n}, taglist}];
  
  (*
    The following Split command groups together ranges of
    consecutive cells that are to be tagged identically. This
    makes some sorts of tagging much more efficient. Note that k
    may have changed as a result, so we reset it below.
  *)
  lis = Split[lis, First[#1]+1===First[#2] && Last[#1]===Last[#2]&];
  k = Length[lis];
  taglist = #[[-1,-1]]& /@ lis;
  
  (*
     Now that we have the tags for each consecutive cell range in
     taglist, we can focus on how we're going to select those
     cell ranges. The following gymnastics returns a list of
     pairs of the form {offset from previous selection, number of
     additional cells to select}.
  *)
  lis = Map[First, lis, {2}];
  lis = {First[#],Last[#]}& /@ lis;
  lis = Flatten[{0,lis}];
  lis = Abs[Subtract @@@ Partition[lis, 2, 1]];
  lis = Partition[lis, 2];
      
  SelectionMove[nb, Before, Notebook];
  Do[
    SelectionMove[nb, Next, Cell, lis[[i,1]] ];
    FrontEndExecute[
      Table[FrontEndToken[nb, "SelectNextCell"], { lis[[i,2]] }]
    ];
    AddCellTags[nb, taglist[[i]] ]    
    ,
    {i, 1, k}
  ];
  
  nb
]



RemoveCellTags[file_String, arg_]:=
Block[{},
  ExtendNotebookFunction[RemoveCellTags, file, arg];
  file
]/; FileType[file] === File


RemoveCellTags[proj_NotebookObject?ProjectDialogQ, args__]:=
Block[{},
  ExtendNotebookFunction[RemoveCellTags, proj, args];
  proj
]


RemoveCellTags[nb_NotebookObject?NotebookOpenQ, All] := 
(
  SelectionMove[nb, All, Notebook]; 
  SetOptions[NotebookSelection[nb], CellTags -> {}];
  nb
)


(*
   The following definition is needed so that the toc and browser
   categories palettes can call RemoveCellTags directly.
*)

RemoveCellTags[nb_NotebookObject?NotebookOpenQ, func_, opt_]:=
Block[{pat},
  pat = opt /. Options[func];
  If[Head[pat] =!= String, Abort[]];
  pat = pat <> "*";
  RemoveCellTags[nb, pat]
]

RemoveCellTags[nb_NotebookObject?NotebookOpenQ, pat_String] :=
Block[{tags},
  tags = NotebookCellTags[nb]//Flatten//Union;
  tags = Select[tags, StringMatchQ[#, pat]&];
  SelectionMove[nb, All, Notebook];
  SelectionRemoveCellTags[nb, tags];
  nb
]


SelectionRemoveCellTags[nb_NotebookObject?NotebookOpenQ, All] :=
(
  SelectionMove[nb, All, Cell];
  SetOptions[NotebookSelection[nb], CellTags -> {}];
  nb
)

SelectionRemoveCellTags[nb_NotebookObject?NotebookOpenQ,
                        tags:((_String) | {__String})   ] := 
(
  MathLink`CallFrontEnd[FrontEnd`SelectionRemoveCellTags[nb, tags]];
  nb
)


(**************** End CellTagging Functions ****************)




(**************** SetOptionsDialog Functions ****************)

$febbuttonstyleoptions = Sequence[
  FontColor -> $Resource["Button2Text"],
  FontWeight -> "Bold",
  FontFamily -> $Resource["Font"],
  FontSize -> 12];

$febbuttonstyleoptions2 = Sequence[
  FontColor -> $Resource["Button3Text"],
  FontWeight -> "Bold",
  FontFamily -> $Resource["Font"],
  FontSize -> 11];





pasteButton[val_String] := pasteButton[val, val];
pasteButton[val_] := pasteButton[ToString[val], val];

pasteButton["Browse...", _]:=
ButtonBox[StyleBox[$Resource["Browse..."], $febbuttonstyleoptions2],
  ButtonEvaluator -> "Local",
  ButtonFunction :> FrontEnd`MessagesToConsole[
    NotebookFind[ButtonNotebook[],
      "GetThisCell", All, CellTags, AutoScroll -> False];
    SelectionMove[ButtonNotebook[], All, CellContents, AutoScroll->False];
    NotebookWrite[ButtonNotebook[],
      "\"" <> DirectoryName[ToString[SystemDialogInput["FileOpen"]]] <> "\"",
      All]
  ]
]

pasteButton[str_, val_] :=
With[{valbox = MakeBoxes[val]},
  ButtonBox[StyleBox[str, $febbuttonstyleoptions2],
    ButtonFunction :> FrontEndExecute[{
      NotebookFind[ButtonNotebook[],
        "GetThisCell", All, CellTags, AutoScroll -> False];
      SelectionMove[ButtonNotebook[], All, CellContents, AutoScroll->False],
      NotebookWrite[ButtonNotebook[], valbox, All]
    }]
  ]
]

okButton[func_] := 
  ButtonBox[StyleBox[$Resource["OK"], $febbuttonstyleoptions],
    ButtonEvaluator -> "Local",
    ButtonFunction :> FrontEnd`MessagesToConsole[
      NotebookFind[ButtonNotebook[],
        "GetThisCell", All, CellTags, AutoScroll -> False];
      AuthorTools`Common`Private`tempCell = NotebookRead[ButtonNotebook[]];
      storeUserOptions[ButtonNotebook[]];
      NotebookClose[ButtonNotebook[]];
      func @@ MakeExpression @ First @ First @ AuthorTools`Common`Private`tempCell
    ]
  ]


applyButton[func_] := 
  ButtonBox[StyleBox[$Resource["Apply"], $febbuttonstyleoptions],
    ButtonEvaluator -> "Local",
    ButtonFunction :> FrontEnd`MessagesToConsole[
      NotebookFind[ButtonNotebook[],
        "GetThisCell", All, CellTags, AutoScroll -> False];
      AuthorTools`Common`Private`tempCell = NotebookRead[ButtonNotebook[]];
      storeUserOptions[ButtonNotebook[]];
      func @@ MakeExpression @ First @ First @ AuthorTools`Common`Private`tempCell
    ]
  ]


cancelButton[] := 
ButtonBox[$Resource["Cancel"],
  ButtonFunction:>FrontEndExecute[{
    NotebookClose[ButtonNotebook[], Interactive->False]
  }]
]


helpButton[cat_String, f:(_String | {_String,_String})] :=
ButtonBox["?",
  ButtonEvaluator -> None,
  ButtonFunction :> FrontEndExecute[{FrontEnd`HelpBrowserLookup[cat, f]}]
]




Options[iSetOptionsDialog] = {
  "Explanation" -> "",
  "SpecialValueButtons" -> {}
}


SetAttributes[iSetOptionsDialog, HoldAll];


iSetOptionsDialog[f_, optionName_String, expr_, func_, opts___] :=
Module[{explanation, morebuttons, logo, headerContent},
  
  {explanation, morebuttons} = {"Explanation", "SpecialValueButtons"} /.
    Flatten[{opts, Options[iSetOptionsDialog]}];
  
  logo = $Resource["Logo"];
  logo = If[Head[logo] === Cell, 
    Append[DeleteCases[logo, _[CellMargins, _]], CellMargins -> {{4,4},{0,8}}],
    $Failed
  ];
  
  headerContent = BoxData @ GridBox[{{
      optionName,
      If[StringMatchQ[Context[f], "AuthorTools`*"],
        helpButton["AddOns", {SymbolName @ f, optionName}],
        helpButton["RefGuide", optionName]
      ]
    }},
    ColumnWidths -> {0.9, 2},
    ColumnAlignments -> {Left, Right}
  ];
  
  NotebookPut[Notebook[Flatten[{

    If[logo === $Failed, {}, logo],
    
    Cell[headerContent,
      "PaletteTitle",
      FontFamily -> $Resource["Font"],
      FontWeight -> "Bold",
      FontColor -> $Resource["Button1Text"],
      Background -> $Resource["Button1Background"],
      Selectable -> False,
      CellMargins->{{4,4},{Inherited,0}}],
    
    Cell[explanation, "Text",
      Selectable -> False],
    
    Cell[BoxData[MakeBoxes[##]& @@ {optionName <> " \[Rule]"}], "Text",
      Selectable -> False],
    
    Cell[BoxData[MakeBoxes[expr]],
      "Input",
      CellFrame -> True,
      ShowCellBracket -> False,
      CellTags -> "GetThisCell",
      CellMargins -> {{20, 2}, {2, 2}},
      AutoItalicWords->{},
      Evaluatable -> False,
      Background -> $Resource["WhiteBackground"]],
    
    Cell[BoxData[GridBox[{{
      cancelButton[],
      "",
      applyButton[func],
      okButton[func]
      }},
        ColumnWidths->0.24,
        ColumnSpacings->0]], 
      "Input",
      ShowCellBracket->False,
      $febbuttonstyleoptions,
      ButtonBoxOptions->{
        ButtonMargins-> $Resource["WideMargin"],
        Active->True,
        ButtonEvaluator->None,
        Background -> $Resource["Button2Background"]}
    ],
    
    If[morebuttons === {}, {},      
      morebuttons = Partition[pasteButton /@ morebuttons, 4, 4, {1,1}, ""];
      
      Cell[BoxData[GridBox[morebuttons,
        ColumnWidths->0.24,
        RowSpacings->0,
        ColumnSpacings->0]], 
      "Input",
      ShowCellBracket->False,
      $febbuttonstyleoptions,
      ButtonBoxOptions->{
        ButtonMargins-> $Resource["WideMargin"],
        Active->True,
        ButtonEvaluator->None,
        Background -> $Resource["Button3Background"]}
    ]]

  }],

  WindowTitle -> $Resource["Set Option", "Title"],

  Sequence @@ $userOptions,
  WindowElements -> {"VerticalScrollBar"},
  ScrollingOptions -> {"VerticalScrollRange" -> Fit},
  WindowFrameElements -> {"CloseBox", "ResizeArea"},
  Saveable -> False,
  ShowCellBracket -> False,
  Magnification -> 1,
  StyleDefinitions -> $AuthorToolsStyleDefinitions,
  Deletable -> False,
  Background -> $Resource["GrayBackground"],
  Deletable -> False
  ]]
]


storeUserOptions[nb_] := $userOptions =
  Options[nb, {WindowSize, WindowMargins, Magnification}];

$userOptions = {
  WindowSize -> {440, 300},
  WindowMargins -> {{Inherited, Inherited}, {Inherited, Inherited}},
  Magnification -> 1
};


booleanOptionQ[f_, opt_] :=
  MemberQ[{True, False}, opt /. Options[f]]


SetOptionsDialog[f_, opt_] := 
iSetOptionsDialog @@
{
  f,
  ToString[opt],
  opt /. Options[f],
  SetOptions[f, opt -> #]&,

  "Explanation" -> ToString[opt::usage],
  "SpecialValueButtons" ->
    Which[
      booleanOptionQ[f,opt], {True, False},
      MatchQ[optionValues[f,opt], {_,___}], optionValues[f, opt],
      MatchQ[optionValues[opt], {_,___}], optionValues[opt],
      True, {}
    ]
}


(**************** End SetOptionsDialog Functions ****************)


$AuthorToolsStyleDefinitions = "HelpBrowser.nb";


$AuthorToolsDirectory = DirectoryName[System`Private`$InputFileName];


paletteNameToPath[name_String] :=
Block[{dir, path},
  (* All palettes are in TextResources, except one *)
  If[StringMatchQ[name, "AuthorTools*"],
    dir = {$AuthorToolsDirectory, "FrontEnd", "Palettes"},
    dir = {$AuthorToolsDirectory, "FrontEnd", "TextResources"}
  ];
  (* 
     After discussing this with Yoshi, we use the same language for pre-
     generated palettes as we do for code-generated ones (see $Resource).
  *)
  Which[
    $AuthorToolsLanguage =!= "English" &&
    FileType[path = ToFileName[Append[dir, $AuthorToolsLanguage], name]] === File,
    path
    ,
    FileType[path = ToFileName[dir, name]] === File,
    path
    ,
    True,
    $Failed
  ]
]


OpenAuthorTool[name_String] :=
  If[# === $Failed, $Failed, NotebookOpen[#]]& @ paletteNameToPath[name]


OpenAuthorTool[] := 
Block[{data},
  data = {"AuthorTools", "MakeProject", "MakeIndex", "MakeContents",
    "MakeCategories", "MakeBilateralCells", "NotebookDiff",
    "NotebookRestore", "Paginate", "ExportCells", "InsertValue",
    "SetPrintingOptions" };
  
  data = {ButtonBox[#, ButtonData :> {#2, None}, ButtonStyle -> "Hyperlink"]}& @@@
    Map[{#, paletteNameToPath[# <> ".nb"]}&, data];
  
  CellPrint[Cell[BoxData[GridBox[data, ColumnAlignments -> Left]], "Print"]]
]


(*
   Yoshi was concerned that even users with $Language set to
   "Japanese" might not want to see Japanese in their dialogs if
   the language kit isn't installed. So instead of switching
   resources via $Language, we switch via the symbol
   $AuthorToolsLanguage, which is the same as $Language if there's
   a directory matching $Language in the AuthorTools palettes
   location, and "English" otherwise.
*)


$AuthorToolsLanguage =
If[
  FileType[ToFileName[{$AuthorToolsDirectory,
    "FrontEnd", "Palettes", $Language}]] === Directory,
  $Language,
  "English"
]



(*
   English resources are in AuthorTools/Resources.m.
   Other language resources are in AuthorTools/<lang>/Resources.m
*)

Needs["AuthorTools`Resources`"];
If[$AuthorToolsLanguage =!= "English",
  Needs["AuthorTools`" <> $AuthorToolsLanguage <> "`Resources`"]
];



$Resource[x__] := $Resource[x] = 
Module[{val, r = AuthorTools`Resources`Private`Resource},
  Which[
    $AuthorToolsLanguage =!= "English" &&
    Head[val = r[x, $AuthorToolsLanguage]] =!= r,
    val,
    
    Head[val = r[x]] =!= r,
    val,
    
    True,
    $Failed
  ]
]


(*
   Message Redirection:

   It is desirable to have messages not appear in palette
   windows, especially when they have options specified to
   automatically save them each time they are closed. I had used
   the following global switch to redirect all messages
   everywhere to the messages window:

     SetOptions[$FrontEnd,
       First[Options[$FrontEnd, MessageOptions]] //.
       "PrintToNotebook" -> "PrintToConsole"
     ];

   but that's overkill. Instead, we'll wrap each button's
   function with a block that sends all messages to the messages
   window during evaluation, and then restores normal message
   behavior when it's done.
  
   MessagesToConsole function is based on John Novak's contribution
   of the same name. I made it a bit stronger, making sure that
   any error messages that would have been printed to the
   notebook to the console (messages window) instead, not just
   kernel error messages. John's notes:
   
   MessagesToConsole[expr] is a hopefully robust temporary redirection
   of messages emitted by expr to the Messages notebook window (instead
   of the input notebook, per default). Note that it is essentially a
   programming structure; hence, it is HoldAll. It contains a CheckAll
   clause so that the value for MessageOptions will get set back to
   what it was even if expr exits abnormally.
   
   
   Update:
   
   In V6, I've removed my local definition of MessagesToConsole in
   favor of the (same) one in FrontEnd`MessagesToConsole.
*)






ExtendNotebookFunction[func_, proj_NotebookObject?ProjectDialogQ, args___] :=
Block[{},
  AuthorTools`MakeProject`Private`ProjectDialogFunction["SaveWarning", proj];
  ExtendNotebookFunction[func, ProjectFileLocation[proj], args]
]


ExtendNotebookFunction[{func_}, file_String, args___] :=
  ExtendNotebookFunction[{func, ToString[func]}, file, args]


ExtendNotebookFunction[func_, file_String, args___] :=
  ExtendNotebookFunction[{func, ToString[func]}, file, args] /;
  Head[func] =!= List


ExtendNotebookFunction::run = "Running `1` on \"`2`\" project.";


ExtendNotebookFunction[{func_, str_String}, file_String, args___] :=
Module[{pn, pd, pf, nb, nbs, prog, result},
  nbs = Notebooks[];
  If[ProjectDataQ[file],
    {pn, pd, pf} = {"Name", "Directory", "Files"} /. ProjectInformation[file];
    If[FileType[pd] =!= Directory || pf === {},
      Return[$Failed]
    ];
    prog = ProgressDialog[
      ToString @ StringForm[ExtendNotebookFunction::run, str, pn],
      ""
    ];
    result = Map[
      (ProgressDialogSetSubcaption[prog, #];
       ExtendNotebookFunction[{func, str}, ToFileName[{pd}, #], args])&,
      pf
    ];
    ProgressDialogClose[prog];
    result
    ,
    (* otherwise, it's a notebook *)
    nb = NotebookOpen[file];
    result = func[nb, args];
    NotebookSave[nb];
    If[!MemberQ[nbs, nb], NotebookClose[nb]];
    result
  ]
] /; FileType[file] === File


End[]

EndPackage[]
