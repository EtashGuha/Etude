(* :Context: AuthorTools`NotebookDiff` *)

(* :Author: Dale R. Horton *)

(* :Summary:
    Package to compare notebooks.
*)

(* :Copyright: *)

(* :Package Version: 2.0 $Revision: 1.54 $, $Date: 2017/01/11 22:09:43 $ *)

(* :Mathematica Version: 5.0 *)

(* :History:
    
*)

(* :Keywords:
     
*)

(* :Discussion:
    
*)

(* :Warning:
    
*)


(* Limitations: 

Some cells have contents that depend on the appearance of the
notebook. For example, fractions depend on the window width
(FractionBox or RowBox).

A small window width can squeeze out GridBox[]es

Assumes local kernel (fe and kernel have same file system)

*)



BeginPackage["AuthorTools`NotebookDiff`", 
  {"AuthorTools`DiffReport`", 
  "AuthorTools`Common`", "AuthorTools`MakeProject`"}]


(* Usage messages *)

NotebookDiff::usage = "NotebookDiff[nb1, nb2] creates a report of the differences between nb1 and nb2."

DiffRecursionDepth::usage = "DiffRecursionDepth is an option of NotebookDiff which specifies how many levels of subdirectories to process when comparing two directories."

ExcludeCellsOfStyles::usage = "ExcludeCellsOfStyles is an option of NotebookDiff which specifies that cells with these styles are excluded from the NotebookDiff."

ExcludeCellsWithTag::usage = "ExcludeCellsWithTag is an option of NotebookDiff which specifies that cells with these tags that are excluded from the NotebookDiff."

IgnoreCellStyleDiffs::usage = "IgnoreCellStyleDiffs is an option of NotebookDiff and CellDiff which specifies whether to ignore diffs in cell styles."

IgnoreOptionDiffs::usage = "IgnoreOptionDiffs is an option of NotebookDiff and CellDiff which specifies whether to ignore diffs in notebook and cell options."

IgnoreContentStructure::usage = "IgnoreContentStructure is an option to NotebookDiff which specifies whether to discard all structure from each cell's content, keeping only strings.";

CellDiff::usage = "CellDiff[cell1, cell2] prints cell1 and cell2 with differences highlighted and a comparison of cellstyles and options."

(* Error messages *)

NotebookDiff::mangr = "The notebook `1` uses Manual CellGrouping. If you replace the content of this notebook from the other notebook, the grouping may not be preserved."

NotebookDiff::nodir = "`1` is not a directory."

NotebookDiff::noproj = "`1` is not a project."

NotebookDiff::nofile = "The files `1` do not exist."

NotebookDiff::noint = "The filesets `1` and `2` have no filenames in common."

NotebookDiff::nonb = "The notebookobject `1` is not an open notebook." 

NotebookDiff::optval = "The value `1` for the `2` option does not match one of the possible patterns `3`."

NotebookDiff::nbval = "The value `1` is not a filename or NotebookObject."

NotebookDiff::dirval = "The value `1` is not a project or directory."

(* Other messages *)

CellDiff::two = "You must select two and only two cells.";

NotebookDiff::proc = "Processing File: `1`";

NotebookDiff::proc1 = "Processing `1` of `2`";

NotebookDiff::empty = "You must select two notebooks.";

NotebookDiff::empty1 = "You must select two projects or directories.";

NotebookDiff::notnb = "You must choose a notebook ending in \".nb\".";

NotebookDiff::notproj = "You must choose a project file."

NotebookDiff::pos = "`1`:Cell `2`";

NotebookDiff::pos1 = "`1` :Cell `2` to `3`";

NotebookDiff::before = "`1` :Before Cell `2`";

NotebookDiff::values = "`1`: `2` in first, `3` in second.";

NotebookDiff::nodiff = "No differences in `1`.";



Begin["`Private`"]  

(* Common routines *)

(*find if nb is an open notebook*)
NotebookClosedQ[nb_NotebookObject] := Options@nb===$Failed
NotebookClosedQ[nb_String] := NotebookClosedQ[NBObj@nb]
NotebookClosedQ[_] := True
      
nbname[nb_] := nbstring[nb]
  
nbstring[nb_NotebookObject] := FullOptions[nb, WindowTitle]

nbstring[nb_Notebook] := 
  WindowTitle/.Options[nb]/.WindowTitle->$Resource["Diff", "No name"]

nbstring[nb_String] := nb
  
myFileBrowse[bool_] := 
 Module[{file = If[bool === False, SystemDialogInput["FileOpen"], SystemDialogInput["FileSave"]]}, 
   If[file =!= $Canceled && $InterfaceEnvironment === "Windows", 
      SetOptions[$FrontEnd, NotebookBrowseDirectory -> DirectoryName@file]
   ];
   file
] /; $Notebooks

(* standalone kernel *)
myFileBrowse[bool_] := 
Module[{file = InputString[$Resource["Diff", "Kernel prompt"]]},
(* don't worry about bool, just do it *)
  If[FileType@file===File, file, $Canceled]
]
    
(*POST-PROCESSORS OF DiffReport -------------------------------------------*)

(* removeContent: remove from diff any cells with cell styles or cell tags *)
removeContent[diff_, {exsty_, extag_}, meth_] :=
  Module[{del, ins, up, sty, tag},
    {del, ins, up} = {"Delete", "Insert", "Update"} /. diff;
    sty = If[Head[exsty] =!= List, {exsty}, exsty];
    tag = If[Head[extag] =!= List, {extag}, extag];
(* Remove excluded cells from each type of diff *)
    del = Map[trim[ #, {sty, tag}] &, del];
    ins = Map[trim[ #, {sty, tag}] &, ins];
    up = Map[trim[ #, {sty, tag}] &, up];
(* An "Update" can be turned into a "Delete" or an "Insert" *)    
    {del, ins, up} = revert[{del, ins, up}, meth];
    {"Delete" -> del, "Insert" -> ins, "Update" -> up}
  ]
    
(*remove content from Delete differences*)
trim[{p1_List, p2_Integer, cont_List}, {exsty_List, extag_List}] := 
  Module[{pairs, newp1, newcont},
    pairs = makepairs[{p1, cont}, {exsty, extag}]; 
    If[pairs === {},
      Sequence @@ {}, (* entirety is excluded *)
      {newp1, newcont} = Transpose@pairs; 
      {newp1, p2, newcont} (* reduced or unchanged diff *)
    ]
  ]
    
(*remove content from Insert differences*)
trim[{p1_Integer, p2_List, cont_List}, {exsty_List, extag_List}] := 
  Module[{pairs, newp2, newcont},
    pairs = makepairs[{p2, cont}, {exsty, extag}];
    If[pairs === {},
      Sequence @@ {}, (* entirety is excluded *)
      {newp2, newcont} = Transpose@pairs;
      {p1, newp2, newcont} (* reduced or unchanged diff *)
    ]
  ]
    
(*remove content from Update differences*)
trim[{p1_List, p2_List, cont1_List, cont2_List}, {exsty_List, extag_List}] := 
  Module[{pairs1, pairs2, newp1, newp2, newcont1, newcont2},
    pairs1 = makepairs[{p1, cont1}, {exsty, extag}];
    pairs2 = makepairs[{p2, cont2}, {exsty, extag}];
    Which[
      pairs1 === pairs2 === {},
        Sequence @@ {}, (* entirety is excluded *)
      pairs1 === {},
        {newp2, newcont2} = Transpose@pairs2;
        newp1 = Floor[(Last@p1 - First@p1)/2];
        {newp1, newp2, newcont2}, (* Update becomes Insert *)
      pairs2 === {},
        {newp1, newcont1} = Transpose@pairs1;
        newp2 = Floor[(Last@p2 - First@p2)/2];
        {newp1, newp2, newcont2}, (* Update becomes Delete *)
      True,
        {newp1, newcont1} = Transpose@pairs1;
        {newp2, newcont2} = Transpose@pairs2;
        {newp1, newp2, newcont1, newcont2} (* reduced or unchanged diff *)
    ]
  ]
    
(*delete any pairs {position, cell} that are undesired*)    
makepairs[{p_, cont_}, {exsty_, extag_}] :=
  Module[{sty, tag, pairs},
    sty = Alternatives @@ exsty;
    tag = Alternatives @@ extag;
    pairs = Transpose@{p, cont};
    (* exclude styles *)
    pairs = DeleteCases[pairs, {_, Cell[_, sty, ___]}];
    (* exclude tags *) 
    pairs = DeleteCases[pairs, {_, Cell[__, CellTags -> tag, ___]}];
    DeleteCases[pairs, {_, Cell[__, CellTags -> {___, tag, ___}, ___]}]
  ]
    
(*convert Update to Delete or Insert if necessary. Linear diff.*)   
revert[{del_, ins_, up_}, Linear] :=
  Module[{newdel, newins, newup},
    (* add Delete's *)
    newdel = Join[del, Cases[up, {_List, _Integer, _List}]];
    (* add Inserts's *)
    newins = Join[ins, Cases[up, {_Integer, _List, _List}]];
    (* remove Update's *)
    newup = 
      DeleteCases[up, {_List, _Integer, _List} | {_Integer, _List, _List}];
    {newdel, newins, newup}
  ]
    
(*makeMove: combine Delete and Insert blocks with identical content*)
makeMove[diff_] :=
  Module[{del, ins, up, lost, gained, lostpos, gainedpos, moves},
    {del, ins} = {"Delete", "Insert"} /. diff;
    lost = Last /@ del ;  (* Delete contents *)
    gained = Last /@ ins; (* Insert contents *)
    int = Intersection[lost, gained]; (* Move contents *)
    (* location of Move content in Insert and Delete *)
    lostpos = Map[First@Position[lost, #] &, int]; 
    gainedpos = Map[First@Position[gained, #] &, int]; 
    (* list of Move diffs *)
    moves =  
      Map[Flatten[#, 1] &, 
        Transpose@{Map[Take[#, 2] &, Extract[ins, gainedpos]], 
            Map[Take[#, 2] &, Extract[del, lostpos]], List /@ int}];
    (* remove Move from Delete and Insert *)
    del = Delete[del, lostpos];
    ins = Delete[ins, gainedpos]; 
    Join[DeleteCases[diff, _["Delete" | "Insert", _]], {"Delete" -> del, 
        "Insert" -> ins, "Move" -> moves}]
  ]
    
(*makeAppearance: convert Update of identical content into Appearance*)
makeAppearance[diff_] :=
  Module[{up, app, new},
    up = "Update" /. diff;
    new = Sort@Flatten[compare/@up];
      (* compare content in Update *)
    up = Cases[new, x_update :> First@x]; (* filter out Update diffs *)
    app = Cases[new, x_appearance :> First@x]; (* filter out Appearance diffs *)
    (* all None are discarded *)
    Join[DeleteCases[diff, _["Update", _]], {"Update" -> up, "Appearance" -> app}]
  ]
  
(*comparison of content for linear diff*)
compare[{p1 : {_Integer ..}, p2 : {_Integer ..}, c1_, c2_}] := 
  Module[{cont1 = First /@ c1, cont2 = First /@ c2},
    If[cont1=!=cont2 && (cont1=!=BoxData/@cont2) && (BoxData/@cont1=!=cont2),
      update@{p1, p2, c1, c2},
      appearance@{p1, p2, c1, c2}
    ]
  ]

(*NotebookDiff -------------------------------------------------------------*)
(*zero nbs*)
NotebookDiff[opts___?OptionQ] := 
  Module[{nb1, nb2},
    nb1 = myFileBrowse[False];
    (* $Canceled is returned if user cancels dialog *)
    If[nb1 === $Canceled, Return[$Canceled]];
    nb2 = myFileBrowse[False];
    If[nb2 === $Canceled, $Canceled, NotebookDiff[nb1, nb2, opts]]
  ]

(*one nb*)
NotebookDiff[nb1_, opts___?OptionQ] := 
  Module[{nb2},
    nb2 = myFileBrowse[False];
    (* $Canceled is returned if user cancels dialog *)
    If[nb2 === $Canceled, $Canceled, NotebookDiff[nb1, nb2, opts]]
  ]
    
(* two nbs*)
(* cannot diff the same nb *)
NotebookDiff[nb1_, nb1_, ___] := 
  (MessageDisplay[NotebookDiff::same, nb1]; $Failed)

NotebookDiff[nb1_Notebook, nb2_Notebook, opts___?OptionQ] :=
  NotebookDiff[NotebookPut[nb1], NotebookPut[nb2], opts]

NotebookDiff[nb1_, nb2_, opts___?OptionQ] :=
If[NotebookQ@nb1 && NotebookQ@nb2,
  notebookDiff[nb1, nb2, opts],
  NotebookDiffFiles[nb1, nb2, opts]
]

NotebookQ[nb_] := 
 Head@nb===NotebookObject || (StringQ@nb && StringMatchQ[nb, "*.nb"])

notebookDiff[nba_, nbb_, opts___?OptionQ] := 
Module[{cells, DR, nb1, nb2},
  (* create raw report of differences *)
  DR = RawNotebookDiff[nba, nbb, opts];
  If[DR===$Failed, Return[$Failed]];
  (* convert raw report to nb *)
  {nb1, nb2} = Map[absolutenb, {nba, nbb}];
  cells = diffCells[{nb1, nb2}, DR, opts];
  Notebook[cells,
    InitializationCellEvaluation -> False,
    CellGrouping -> Manual, 
    WindowTitle->$Resource["Diff", "Notebook title"],
    TaggingRules->{"Notebooks"->{nb1,nb2}},
    WindowSize->{600,600},
    importss[nb1]
  ]
]
    
RawNotebookDiff[nba_, nbb_, opts___?OptionQ] :=
Module[{NB1, NB2},
  (* test nbs and get nb expr *)
  {NB1, NB2} = nbexpr/@{nba, nbb};
  If[NB1===$Failed||NB2===$Failed, Return[$Failed]];
  (* test options *)
  If[!goodopts[NotebookDiff, opts], Return[$Failed]];
  (* test for manual cellgrouping *)
  Which[
   (CellGrouping/.Options[NB1])===Manual,
     MessageDisplay[NotebookDiff::mangr, nba],
   (CellGrouping/.Options[NB2])===Manual,
     MessageDisplay[NotebookDiff::mangr, nbb]
  ];
  (* create report of differences *)
  NotebookDiffReport[NB1, NB2, opts]
]

NotebookDiffReport[NB1_, NB2_, opts___?OptionQ] :=
  Module[{cells, exsty, extag, igopt, meth, f,
      cells1, cells2, DR, optdiff},
    {exsty, extag, igopt, meth, f} = {ExcludeCellsOfStyles, ExcludeCellsWithTag, 
      IgnoreOptionDiffs, Method, ShowDiffProgress} /. Flatten[{opts, Options[NotebookDiff]}];
    
    cells1 = First @ FlattenCellGroups @ NB1;
    cells2 = First @ FlattenCellGroups @ NB2;
    
    If[TrueQ@f,
      f = If[TrueQ@$FromPalette,
       $PaletteTrackingFunction,
       $KernelTrackingFunction
     ]
    ];
    CheckAbort[
      DR = DiffReport[cells1, cells2, Method -> meth, ShowDiffProgress -> f, opts],
      If[ValueQ[$dialog], ProgressDialogClose@$dialog; $dialog=.]; Abort[]
    ];
    ProgressDialogClose@dialog;
    (* Post-Process *)
    DR = removeContent[DR, {exsty, extag}, meth];
    DR = makeMove@DR;
    DR = makeAppearance@DR;
    (* Notebook option diffs*)
    optdiff = If[igopt =!= All && meth === Linear,
      diffOptions[NB1, NB2, igopt],
      {}
    ];
    Append[DR, "Option"->optdiff]
  ]
    

Options[NotebookDiff] = {
  DiffRecursionDepth -> Infinity,
  ExcludeCellsOfStyles -> {}, ExcludeCellsWithTag -> {}, 
  IgnoreCellStyleDiffs -> False,
  IgnoreOptionDiffs -> None, 
  IgnoreContentStructure -> False,
  Method -> Linear,
  ShowDiffProgress -> True}


optionValues[IgnoreOptionDiffs] = {None, All, {CellTags}, {CellLabel}}
      
(* test nbs *)
nbexpr[nb_] :=
Module[{tmp},
  Switch[nb,
        _String, 
          If[FileType@nb=!=File,
            MessageDisplay[NotebookDiff::noopen, nb]; $Failed,
            tmp = myImport[nb, "Notebook"];
            If[tmp===$Failed, MessageDisplay[NotebookDiff::noopen, nb]];
            tmp
          ],
        _NotebookObject, 
          If[NotebookClosedQ@nb, 
            MessageDisplay[NotebookDiff::nonb, nb]; $Failed,
            NotebookGet@nb
          ],
        _Notebook, 
          nb,
        _, 
          MessageDisplay[NotebookDiff::nbval, nb]; $Failed
  ]
]

absolutenb[nb_String] := System`Private`ExpandFileName@nb
absolutenb[nb_] := nb 


(* choose Import or Get *)
myImport[nb_, "Notebook"] := Get[nb] /; $UseGetForNotebookDiff
myImport[nb_, fmt_] := Import[nb, fmt]

 
(* test all options except ShowDiffProgress *)
goodopts[f_, opts___?OptionQ] :=
  Module[{exsty, extag, igsty, igopt, meth},
    {exsty, extag, igsty, igopt, meth} = {ExcludeCellsOfStyles, ExcludeCellsWithTag, 
      IgnoreCellStyleDiffs, IgnoreOptionDiffs, Method} /. {opts} /. Options[f];
    testopt[ExcludeCellsOfStyles, exsty, {{}, _String, {_String ..}}, f] && 
      testopt[ExcludeCellsWithTag, extag, {{}, _String, {_String ..}}, f] && 
      testopt[IgnoreCellStyleDiffs, igsty, {True, False}, f] && 
      testopt[IgnoreOptionDiffs, igopt, {All,None,{},_Symbol,{_Symbol ..}}, f] &&
      testopt[Method, meth, {Linear}, f] 
  ]
    
(* test an option *)
testopt[opt_, val_, patterns_List, f_] :=
  If[Or @@ Map[MatchQ[val, #] &, patterns], True, 
    MessageDisplay[f::optval, val, opt, patterns]; False]
    
(*Misc Tools: used in several places -----------------------------------*)

originalNBs[] :=
  "Notebooks"/.FullOptions[ButtonNotebook[], TaggingRules]
  
Attributes[BlockContext] = {HoldAll}

(* Temporarily hide package contexts so NotebookPut and NotebookWrite
   will use full context names. *)
BlockContext[f_] := 
  Block[{$ContextPath = {"System`"}}, f]
  
Attributes[UnBlockContext] = {HoldAll}
  
(* Unhide package contexts for commands like ToString *)
UnBlockContext[f_] := 
  Block[{$ContextPath = 
    {"System`", "AuthorTools`NotebookDiff`", "AuthorTools`DiffReport`"}}, f
  ]
  
NBObj[nb_] :=
  Switch[nb,
    _NotebookObject, nb,
    _String, NotebookOpen@nb,
    _Notebook, NotebookPut@nb
  ]
  
(* Platform specific constants used in nb layout.  
   delta=width between nbs. hdelta=height between nbs. border=border around nbs. *)
{delta, hdelta, border} = 
Switch[$InterfaceEnvironment,
  "Windows", {8, 28, 0},
  "Macintosh", {2, 23, 0},
  "X", {8, 95, 25},
  _, {8, 28, 0}
]






(*diffCells: list of Cells that represent the differences*)
diffCells[{nb1_, nb2_}, DR_, opts___?OptionQ] :=
  Module[{cells, pat, name1, name2,del, ins, up, move, app, opt},
    {exsty, extag, igsty, igopt, meth, f} = {ExcludeCellsOfStyles, ExcludeCellsWithTag, 
      IgnoreCellStyleDiffs, IgnoreOptionDiffs, Method, ShowDiffProgress} /. {opts} /.
      Options[NotebookDiff];
    {name1, name2} = nbname/@{nb1, nb2};
    {del, ins, up, move, app, opt} = 
      {"Delete", "Insert", "Update", "Move", "Appearance", "Option"} /. DR;
    (* check for no differences *)
    cells = If[del===ins===up===move===app==={}, {diffCell["None"]},
    (* else scan through differences and create a cell for each one *)
    nums = Sort[getnum/@Join[del, ins, up, app, move]];
    Map[
      Function[i, Which[
        pat={{i, ___}, __}; MatchQ[del, {___, pat, ___}], 
          diffCell["Delete", First@Cases[del, pat], opts],
        pat={i, __}; MatchQ[ins, {___, pat, ___}], 
          diffCell["Insert", First@Cases[ins, pat], opts],
        pat={{i, ___}, __}; MatchQ[up, {___, pat, ___}], 
          diffCell["Update", First@Cases[up, pat], opts],
        pat={{i, ___}, __}; MatchQ[app, {___, pat, ___}], 
          diffCell["Appearance", First@Cases[app, pat], opts],
        pat={_, _, {i, ___}, __}; MatchQ[move, {___, pat, ___}],
          diffCell["Move", First@Cases[move, pat], opts]
      ]], nums]
    ];
    (* Notebook option diffs*)
    cells = Join[cells, diffCell["Option", opt]];
    (* add header *)
    cells = Join[ 
      {Cell[$Resource["Diff", "Notebook title"], "ReportName", ShowCellBracket->False],
      Cell[BoxData[FormBox[GridBox[{
        {Cell[TextData[{StyleBox[$Resource["Diff", "New"] <> ": ", FontWeight->"Bold"], name1}]],
         Cell[TextData[{StyleBox[$Resource["Diff", "Old"] <> ": ", FontWeight->"Bold"], name2}]]}
      }], TextForm]], "SideBySide"],
      Cell[BoxData[FormBox[GridBox[{{
        ButtonBox[$Resource["Diff", "Change View"], ButtonStyle->"View"],
        ButtonBox[$Resource["Diff", "Update"], ButtonStyle->"Update", ButtonData:>{opts}],
        ButtonBox[$Resource["Diff", "Close"], ButtonStyle->"Close"],
        ButtonBox[$Resource["Diff", "Help"], ButtonStyle->"Help"]
      }}], TextForm]], "Controls"]},
      cells
    ];
    cells
  ]
  
getnum[{{i_Integer,___}, __}] := i (*del, up, app*)
getnum[{i_Integer, _, _}] := i (*ins*)
getnum[{_, _, {i_Integer, ___}, __}] := i (*mv*)
    
viewdiff["SideBySide"] := 
  selectDiff[{1, 1}, "Working", Highlight->False]

viewdiff["Working"] := 
( RemoveColor[ButtonNotebook[]];
  SetOptions[ButtonNotebook[], WindowSize->{600, 600}, ScreenStyleEnvironment->"Working", 
    WindowMargins->{{Automatic, Automatic}, {Automatic, Automatic}}];
) 

updatediff[{opts___}] :=
  reDiff[opts]
   
closediff[] :=
Module[{bnb},
  bnb=ButtonNotebook[];
  SaveAndClose/@originalNBs[];
  SaveAndClose@bnb
]

SaveAndClose[nb_NotebookObject] :=
(
  NotebookSave[nb, Interactive->True];
  NotebookClose@nb
)

SaveAndClose[nb_String]:=
Module[{nbobj},
  nbobj = Select[Notebooks[], filenameQ[#, nb]&];
  If[nbobj=!={},
    SaveAndClose@First@nbobj
  ]; 
]

$PaletteTrackingFunction[{1, total_/;(total>100)}] :=
  ($dialog = ProgressDialog[$Resource["Diff", "Computing"], ""])
          
$PaletteTrackingFunction[{total_, total_/;(total>100)}] :=
  (ProgressDialogClose@$dialog; $dialog=.)

$PaletteTrackingFunction[{count_/;(Mod[count,10]===0), total_/;(total>100)}] :=
  ProgressDialogSetSubcaption[$dialog, 
    ToString @ StringForm[NotebookDiff::proc1, count, total]]

$KernelTrackingFunction[{1, total_/;(total>100)}] :=
  Print[$Resource["Diff", "Starting nb"]]
          
$KernelTrackingFunction[{total_, total_/;(total>100)}] :=
  Print[$Resource["Diff", "Finished"]]

$KernelTrackingFunction[{count_, total_/;(total>100)}] :=
  If[Mod[count, Ceiling[total/10]] === 0,
    Print[$Resource["Diff", "Processing"], " ", Floor[100*count/total], "%" ]
  ]
    
(*diffCell: a Cell that represents a difference. *)
(* Delete in source or Insert in target *)
diffCell["Delete",  
  {pos1_List, pos2_Integer, cont_List}, opts___] := 
  Cell[CellGroupData[Flatten@{
   Cell[BoxData@FormBox[GridBox@{{
    RowBox@{
     ReportHighlightButton[
      Cell[TextData[{CounterBox["Diff"], ") ", $Resource["Diff", "Extra in new"]}]],
      pos1, pos2
     ],
     positiontext[pos1, "New"], ",", positiontext[pos2, "Old"], "."},
    ButtonBox[$Resource["Diff", "Apply left"], ButtonStyle -> "ChangeDiff",
     ButtonData :> change["Delete", pos1, 1, opts]],
    ButtonBox[$Resource["Diff", "Apply right"], ButtonStyle -> "ChangeDiff",
     ButtonData :> change["Insert", {pos2, 2}, {pos1, 1}, opts]]
   }}, TextForm],"DiffReport"], 
   cont}, Closed]
  ]

(* Insert in source or Delete in target *)
diffCell["Insert", 
  {pos1_Integer, pos2_List, cont_List}, opts___] := 
  Cell[CellGroupData[Flatten@{
   Cell[BoxData@FormBox[GridBox@{{
    RowBox@{
     ReportHighlightButton[
      Cell[TextData[{CounterBox["Diff"], ") ", $Resource["Diff", "Extra in old"]}]],
      pos1, pos2
     ],
     positiontext[pos1, "New"], ",", positiontext[pos2, "Old"], "."},
    ButtonBox[$Resource["Diff", "Apply left"], ButtonStyle -> "ChangeDiff",
     ButtonData :> change["Insert", {pos1, 1}, {pos2, 2}, opts]],
    ButtonBox[$Resource["Diff", "Apply right"], ButtonStyle -> "ChangeDiff",
     ButtonData :> change["Delete", pos2, 2, opts]]
   }}, TextForm],"DiffReport"], 
   cont}, Closed]
  ]

(* Insert in source or different Insert in target *)
diffCell["Update", 
  {pos1_List, pos2_List, cont1_List, cont2_List}, opts___] := 
  Cell[CellGroupData[Flatten@{
   Cell[BoxData@FormBox[GridBox@{{
    RowBox@{
     ReportHighlightButton[
      Cell[TextData[{CounterBox["Diff"], ") ", $Resource["Diff", "Different cells"]}]],
      pos1, pos2
     ],
     positiontext[pos1, "New"], ",", positiontext[pos2, "Old"], ".  ",
     CellDiffButton[]},
    ButtonBox[$Resource["Diff", "Apply left"], ButtonStyle -> "ChangeDiff",
     ButtonData :> change["Insert", {pos1, 1}, {pos2, 2}, opts]],
    ButtonBox[$Resource["Diff", "Apply right"], ButtonStyle -> "ChangeDiff",
     ButtonData :> change["Insert", {pos2, 2}, {pos1, 1}, opts]]
   }}, TextForm],"DiffReport"], 
   Cell[$Resource["Diff", "In new"], "PositionMiddle"],
   cont1, 
   Cell[$Resource["Diff", "In old"], "PositionMiddle"],
   cont2
   }, Closed]
  ]

(* Insert in source or different Insert in target *)
diffCell["Appearance", 
  {pos1_List, pos2_List, cont1_List, cont2_List}, opts___] := 
  Cell[CellGroupData[Flatten@{
   Cell[BoxData@FormBox[GridBox@{{
    RowBox@{
     ReportHighlightButton[
      Cell[TextData[{CounterBox["Diff"], ") ", $Resource["Diff", "Different opts"]}]],
      pos1, pos2
     ],
     positiontext[pos1, "New"], ",", positiontext[pos2, "Old"], ".  ",
     CellDiffButton[]},
    ButtonBox[$Resource["Diff", "Apply left"], ButtonStyle -> "ChangeDiff",
     ButtonData :> change["Insert", {pos1, 1}, {pos2, 2}, opts]],
    ButtonBox[$Resource["Diff", "Apply right"], ButtonStyle -> "ChangeDiff",
     ButtonData :> change["Insert", {pos2, 2}, {pos1, 1}, opts]]
   }}, TextForm],"DiffReport"], 
   Cell[$Resource["Diff", "In new"], "PositionMiddle"],
   cont1, 
   Cell[$Resource["Diff", "In old"], "PositionMiddle"],
   cont2
   }, Closed]
  ]

(* Move within source or Move within target *)
diffCell["Move",
  {ins1_Integer, ins2_List, del1_List, del2_Integer, cont_List}, opts___] := 
  Cell[CellGroupData[Flatten@{
   Cell[BoxData@FormBox[GridBox@{{
    RowBox@{
     ReportHighlightButton[
      Cell[TextData[{CounterBox["Diff"], ") ", $Resource["Diff", "Moved"]}]],
      del1, ins2
     ],
     positiontext[del1, "New"], ",", positiontext[ins2, "Old"], "."},
    ButtonBox[$Resource["Diff", "Apply left"], ButtonStyle -> "ChangeDiff",
     ButtonData :> change["Move", del1, ins1, 1, opts]],
    ButtonBox[$Resource["Diff", "Apply right"], ButtonStyle -> "ChangeDiff",
     ButtonData :> change["Move", ins2, del2, 2, opts]]
   }}, TextForm],"DiffReport"], 
   cont}, Closed]
  ]
  
diffCell["None", opts___] := 
  Cell[$Resource["Diff", "Same cells"], "NoDifference"]

(*change: how to change one nb to be like the other*)
change["Delete", pos_, num_, opts___] := 
Module[{nbs, nbobj},
   nbs = originalNBs[];
   nbobj = NBObj@nbs[[num]];
   highlightContent[nbobj, pos]; 
   (* should ensure deletable *)
   NotebookDelete[nbobj];
   reDiff[opts]
]

(* use FrontEnd`NotebookWrite to write the cell without the GeneratedCell option *)
(* Insert also used for updating *)
change["Insert", {pos_, num_}, {pos2_, num2_}, opts___] :=
Module[{cont, nbs, nbobj, nbobj2},
   nbs = originalNBs[];
   nbobj = NBObj@nbs[[num]];
   nbobj2 = NBObj@nbs[[num2]];
   highlightContent[nbobj, pos];
   highlightContent[nbobj2, pos2];
   cont = NotebookRead[nbobj2];
   FrontEndExecute[{FrontEnd`NotebookWrite[nbobj, cont, GeneratedCell -> False]}];
   reDiff[opts]
]
 
change["Move", del_, ins_, num_, opts___] :=
Module[{cont, nbs, nbobj},
   nbs = originalNBs[];
   nbobj = NBObj@nbs[[num]];
   highlightContent[nbobj, del];
   cont = NotebookRead[nbobj];
   If[First@del > ins,
    NotebookDelete[nbobj];
     highlightContent[nbobj, ins]; 
     FrontEndExecute[{FrontEnd`NotebookWrite[nbobj, cont, GeneratedCell -> False]}],
    highlightContent[nbobj, ins]; 
     FrontEndExecute[{FrontEnd`NotebookWrite[nb, cont, GeneratedCell -> False]}];
     highlightContent[nbobj, del];
     NotebookDelete[nbobj]
   ];
   reDiff[opts]
]

(* No Option differences*)
diffCell["Option", {}] := 
  {Cell[$Resource["Diff", "Same opts"], "NoDifference"]}

(* Option differences *)
diffCell["Option", up_] := 
  Map[optCell, up]

optCell[{opt1_ , opt2_}] := 
  Cell[BoxData@FormBox[GridBox@{{
    RowBox@{
      Cell[TextData[{CounterBox["Diff"], ")"}]],
     ToString @ StringForm[NotebookDiff::values, First@opt1, optval@opt1, optval@opt2]},
    ButtonBox[$Resource["Diff", "Apply left"], ButtonStyle -> "ChangeDiff",
     ButtonData :> setOptions[1, opt2]],
    ButtonBox[$Resource["Diff", "Apply right"], ButtonStyle -> "ChangeDiff",
     ButtonData :> setOptions[2, opt1]]
   }}, TextForm],"DiffReport"]
   
(* SetOptions in source or different SetOptions in target *)
setOptions[num_, opt_->"(No value)"] :=
Module[{nb},
  nb = originalNBs[][[num]];
  SetOptions[NBObj@nb, opt->Inherited];
  removecell[]
]

setOptions[num_, opt_] :=
Module[{nb},
  nb = originalNBs[][[num]];
  SetOptions[NBObj@nb, opt];
  removecell[]
]

removecell[] :=
module[{bnb=ButtonNotebook[]},
  SelectionMove[bnb, All, ButtonCell];
  SetOptions[NotebookSelection@bnb, Editable->True, Deletable->True];
  NotebookDelete[bnb]
]

(* get string for RHS of option without evaluating RHS of :> *)
optval[opt_] := 
  Module[{str, pos},
    str = ToString@opt;
    pos = Last@First@StringPosition[str, {"->", ":>"} ];
    str = StringDrop[str, pos + 1];
    If[StringLength[str] <= 25, str, 
      StringTake[str, 12] <> "\[Ellipsis]" <> StringTake[str, -12]]
  ]
    
(*diffOptions: create list of different options*)
(* diff in fe *)
diffOptions[nb1_NotebookObject, nb2_NotebookObject, igopt_] :=
  Module[{listofopts, opt1, opt2, up, 
      notopts = {WindowMargins,WindowSize,ScreenRectangle,FrontEndVersion,
        WindowTitle}}, 
    notopts = Switch[igopt, 
      None, notopts,
      _List, Join[notopts, igopt],
      _, Append[notopts, igopt]
    ];
    listofopts = 
      Complement[
        Flatten@{First/@Options[nb1], First/@Options[nb2]}, notopts
    ];
    opt1 = If[listofopts==={}, {}, Options[nb1, listofopts]];
    opt2 = If[listofopts==={}, {}, Options[nb2, listofopts]];
    up = "Update" /. DiffReport[opt1, opt2];
    Flatten[Map[Transpose@Take[#, -2] &, up], 1]
]
    
(* diff in kernel *)    
diffOptions[nb1_Notebook, nb2_Notebook, igopt_] :=
Module[{opt1, opt2, pairs1, pairs2, 
    notopts = {WindowMargins,WindowSize,ScreenRectangle,FrontEndVersion}},  
    notopts = Switch[igopt, 
      None, notopts,
      _List, Join[notopts, igopt],
      _, Append[notopts, igopt]
    ];
  {opt1, opt2} = List @@@ 
    Map[Sort[DeleteCases[Rest[#], _[Alternatives@@notopts, _]], 
      (OrderedQ[First /@ {#1, #2}]) &] &, {nb1, nb2}
  ];
  pairs1 = Map[{#, element[opt2, #]} &, opt1];
  pairs2 = Map[{element[opt1, #], #} &, opt2];
  DeleteCases[Union[pairs1, pairs2], {_[a_, b_], _[a_, b_]}]
]

(* broken *)
diffOptions[nb1_String, nb2_String, igopt_] := {}

element[opts_, elem_] :=
  Module[{sel},
    sel = Select[opts, (First[#] === First[elem]) &, 1];
    If[sel === {}, First[elem]->"(No value)", First@sel]
  ]

(* ReportHighlightButton: highlight a difference *)
ReportHighlightButton[txt_, pos1_, pos2_] := 
  ButtonBox[txt, 
    ButtonStyle -> "DiffReportButton", 
    ButtonData :> {pos1, pos2}
]

(* highlight single cell *)
highlightContent[nb_, {pos_Integer}] := 
Module[{nbobj = NBObj@nb},
   SetSelectedNotebook[nbobj];
    SelectionMove[nbobj, Before, Notebook, AutoScroll -> False];
    SelectionMove[nbobj, Next, Cell, pos];
    FrontEndExecute[{FrontEndToken[nbobj, "OpenSelectionParents"]}]
]
(* "OpenSelectionParents" token opens only selected cellgroups *)

(* highlight between cells *)
highlightContent[nb_, pos_Integer] := 
Module[{nbobj = NBObj@nb},
  highlightContent[nbobj, {pos}]; SelectionMove[nbobj, Before, Cell];
]

(* highlight multiple cells *)
highlightContent[nb_, {p1_Integer, ___, p2_Integer}] := 
Module[{nbobj = NBObj@nb},
   highlightContent[nbobj, {p2}]; 
    FrontEndExecute[Table[FrontEndToken[nbobj, "SelectPreviousCell"], {p2 - p1}]]; 
    FrontEndExecute[{FrontEndToken[nbobj, "OpenSelectionParents"]}]
]

(* selectDiff: show selected difference *)
selectDiff[{pos1_, pos2_}, mode_, opts___?OptionQ] := 
Module[{NB1, NB2, bnb, sw, w, sh, h, diffh=200, high},
  high = Highlight /. {opts} /. Highlight->True;
  {NB1, NB2} = NBObj/@originalNBs[];
  If[NB1===$Failed||NB2===$Failed, Return[]];
  bnb=ButtonNotebook[];
  If[mode==="Working", 
    {sw, sh} = ScreenSize[];
    w = Floor[ (sw - delta - 2*border)/2];
    h = sh - diffh - hdelta;
    SetOptions[NB1, WindowSize->{w, h}, 
      WindowMargins->{{border, Automatic}, {Automatic, border}}];
    SetOptions[NB2, WindowSize->{w, h}, 
      WindowMargins->{{Automatic, border}, {Automatic, border}}];
    SetOptions[bnb, WindowSize->{Automatic, diffh}, ScreenStyleEnvironment->"SideBySide", 
      WindowMargins->{{border, border}, {border, Automatic}}];
    SelectionMove[bnb, All, Notebook, AutoScroll->False];
    FrontEndExecute[{FrontEndToken[bnb, "SelectionCloseAllGroups"]}]
  ];
  highlightContent[NB1, pos1];
  highlightContent[NB2, pos2];
  If[high,
    RemoveColor[bnb];
    SelectionMove[bnb, All, ButtonCell];
    SetOptions[NotebookSelection[bnb], Editable -> True];
    SetOptions[NotebookSelection[bnb], Background -> RGBColor[1, 1, 0]];
    SetOptions[NotebookSelection[bnb], Editable -> False]
  ];
  SetSelectedNotebook[bnb]
]

RemoveColor[bnb_] :=
(
  NotebookFind[bnb, "DiffReport", All, CellStyle, AutoScroll -> False];
  SetOptions[NotebookSelection[bnb], Editable -> True];
  SetOptions[NotebookSelection[bnb], Background -> RGBColor[1, 1, 1]];
  SetOptions[NotebookSelection[bnb], Editable -> False];
)

ScreenSize[] :=
Module[{nb, sw},
  nb=NotebookCreate[Visible->False,
    WindowSize->Automatic,
    WindowMargins->{{0,0},{0,0}}
  ];
  sw = FullOptions[nb, WindowSize];
  NotebookClose@nb;
  sw
]
          
(* reDiff: find differences again, possibly after changes to nbs *)
reDiff[opts___] :=
Module[{NB1, NB2, nb1, nb2, nba, nbb, diffnb},
      {nba, nbb} = originalNBs[];
      diffnb = ButtonNotebook[];
      (* hide diffnb to indicate changes are yet to be made *)
      SetOptions[diffnb, Visible -> False]; 
      (* place new list of diffs in nb *)
      {NB1, NB2} = getnb/@{nba, nbb};
      DR = RawNotebookDiff[NB1, NB2, opts];
      {nb1, nb2} = Map[absolutenb, {nba, nbb}];
      cells = diffCells[{nb1, nb2}, DR, opts];
      NotebookPut[
        Notebook[cells, Sequence@@Options[diffnb]], 
        diffnb
      ];
      SelectionMove[diffnb, Before, Notebook];
      (* unhide nb to indicate that reDiff is complete *)
      SetOptions[diffnb, Visible -> True]
]

filenameQ[nb_NotebookObject, file_] := NotebookFilePath[nb] === file;

getnb[nb_] :=
Module[{tmp, nbobj},
  Switch[nb,
        _String, 
          (* get open nb if possible because could include changes *)
          tmp = Select[Notebooks[], filenameQ[#, nb]&];
          If[tmp === {},
            myImport[nb, "Notebook"],
            NotebookGet@First@tmp
          ],
        _NotebookObject, 
          If[NotebookClosedQ@nb, 
            MessageDisplay[NotebookDiff::nonb, nb]; {$Failed, $Failed},
            NotebookGet@nb
          ],
        _Notebook, 
          nb,
        _, 
          MessageDisplay[NotebookDiff::nbval, nb]; $Failed
  ]
]      
  
(*NotebookDiffDialog ---------------------------------------------------*)

NotebookDiffDialog[] := 
Module[{nb},
  nb = InputNotebook[];
  If[nb===$Failed, nb = ""];
  notebookDiffDialog[Notebook][{nb, ""}]
]

NotebookDiffFilesDialog[] := 
Module[{nb},
  nb = InputNotebook[];
  If[nb===$Failed || !ProjectDialogQ@nb, nb = ""];
  notebookDiffDialog[Files][{nb, ""}]
]
  
notebookDiffDialog[type_][{nb1_, nb2_}] :=
Module[{newnb, txt},
        txt = If[type===Notebook, "Notebooks", "Files"];
        newnb = BlockContext@NotebookPut@Notebook[{
            DialogHeader[type],
            Choose[type]["NB1", "New"], 
            Choice["NB1", nb1], 
            Choose[type]["NB2", "Old"], 
            Choice["NB2", nb2], 
            ControlButtons[type]}, 
          ButtonBoxOptions :> {ButtonStyle -> "Paste", Active -> True, 
            ButtonEvaluator -> Automatic}, 
          Deletable->False,
          ShowCellBracket -> False, 
          WindowElements -> {"StatusArea", "VerticalScrollBar", "MenuBar"}, 
          WindowTitle -> $Resource["Diff", "Selector title", txt], 
          Editable -> False,
          WindowSize -> {500, 300}, 
          ShowSelection -> False,
          TaggingRules:>{},
          Background -> GrayLevel[0.85],
          ScrollingOptions->{"VerticalScrollRange"->Fit}
        ];
        SelectionMove[newnb, Before, Notebook];
        newnb
]

(* Make sure content is editable before acting on it *)
Attributes[MakeEditable] = {HoldAllComplete}

MakeEditable[f_] := MakeEditable[ButtonNotebook[], f]

MakeEditable[bnb_, f_] :=
(
  SetOptions[bnb, Editable->True]; 
  f;
  SetOptions[bnb, Editable->False]; 
)

DialogHeader[type_] :=
Module[{txt, lnk},
txt = $Resource["Diff", If[type===Notebook, "Notebook title", "Other title"]];
lnk = If[type===Notebook, 
  "Finding Notebook Differences", 
  "Finding Project and Directory Differences"
];
Sequence@@{
$Resource["Logo"],

Cell[BoxData[GridBox[{
        {txt, 
          ButtonBox[
            RowBox[{" ", 
              StyleBox["?",
                FontSize->12], " "}],
            ButtonData->lnk,
            ButtonFunction:>(FrontEndExecute[ {
                FrontEnd`HelpBrowserLookup[ 
                "AddOns", #]}]&),
            ButtonEvaluator->None,
            ButtonMargins->3,
            Background->RGBColor[0.329412, 0.584314, 0.694118]]}
        },
      ColumnWidths->{0.9, 2},
      ColumnAlignments->{Left, Right}]], "DefinitionBox",
  CellMargins->{{4, 4}, {Inherited, 0}},
  FontFamily -> "Helvetica",
  FontWeight -> "Bold",
  FontColor -> GrayLevel[1],
  FontSize->14,
  Background->RGBColor[0, 0.32549, 0.537255]]
}]
    
(* ChooseNotebook: interface to selecting an open nb *)
Choose[Notebook][NB_, str_] :=
  Cell[BoxData[GridBox[{
        {Cell[$Resource["Diff", str, "Notebook"],
            FontFamily->"Helvetica",
            FontSize->10,
            FontWeight->"Bold",
            FontSlant->"Plain"],
          Cell[TextData[$Resource["Diff", "Select notebook", NB]],
            FontFamily->"Times",
            FontWeight->"Plain",
            FontSlant->$Resource["Italic"]
          ]}},
          ColumnWidths->{0.3, 0.7},
          ColumnAlignments->{Left, Right}
        ]], NotebookDefault,
  CellMargins->{{12, 10}, {0, 4}},
  ShowSelection->True
]
  
Choose[Files][NB_, str_] :=
  Cell[BoxData[GridBox[{
        {Cell[$Resource["Diff", str, "Files"],
            FontFamily->"Helvetica",
            FontSize->10,
            FontWeight->"Bold",
            FontSlant->"Plain"],
          Cell[TextData[$Resource["Diff", "Select files", NB]],
            FontFamily->"Times",
            FontWeight->"Plain",
            FontSlant->$Resource["Italic"]
          ]}},
          ColumnWidths->{0.3, 0.7},
          ColumnAlignments->{Left, Right}
        ]], NotebookDefault,
  CellMargins->{{12, 10}, {0, 4}},
  ShowSelection->True
]

openNBFunction[NB_, bnb_] :=
Module[{nbs, cont, tmpfile=Close@OpenTemporary[]},
 nbs = DeleteCases[Notebooks[], MessagesNotebook[]|bnb|_?PaletteQ];
 cont = If[nbs === {}, 
    $Resource["Diff", "Selector No Notebooks"],
    GridBox@Map[openNB[NB, bnb, #]&, nbs]
 ];
 tmpfile = tmpfile <> ".nb"; (* 50793 *)
 Export[tmpfile, 
   Notebook[{Cell[BoxData@cont, 
     "Text", CellMargins->{{0,0},{0,0}}]}, 
     GridBoxOptions :> {RowSpacings->0, ColumnWidths->.99},
     ButtonBoxOptions :> {ButtonStyle -> "Paste", Active -> True, 
       ButtonEvaluator -> Automatic}, 
     TextAlignment->Center, 
     ShowCellBracket -> False, 
     WindowElements -> {}, WindowFloating->True,
     WindowTitle -> $Resource["Diff", "Selector title", "Notebook"], WindowFrame -> "Palette", 
     WindowFrameElements -> {"CloseBox"}, Editable -> False,
     WindowSize->{200, Fit}, Background->RGBColor[0.5, 0.6, 0.6875]
   ], "Notebook"
 ];
 NotebookOpen@tmpfile
]

PaletteQ[nb_NotebookObject] :=
 FullOptions[nb, WindowClickSelect]===False
 
(* Button to select an open nb *)
openNB[NB_, bnb_, val_] :=
   {ButtonBox[nbstring@val, ButtonFunction:>
          (
          Needs["AuthorTools`NotebookDiff`"];
          NotebookClose[ButtonNotebook[]];
          SetOptions[NotebookFind[bnb, NB, All, CellTags],
            Editable->True, Deletable->True];
          MakeEditable[bnb, NotebookWrite[bnb, Choice[NB, val]]]
          )
   ]}

(* choose nb by browsing *)     
browseButtonFunction[NB_, bnb_, file_] :=
  If[file =!= $Canceled,
    If[StringLength[file]>2 && StringTake[file, -3] === ".nb",
      SetOptions[NotebookFind[bnb, NB, All, CellTags], 
        Editable->True, Deletable->True];
        MakeEditable@NotebookWrite[bnb, Choice[NB, file]],
      messageDialog[NotebookDiff::notnb]
    ]
  ]
  
openDirectoryFunction[NB_, bnb_] :=
Module[{file},
  file = myFileBrowse[False];
  If[file =!= $Canceled,
    SetOptions[NotebookFind[bnb, NB, All, CellTags], 
      Editable->True, Deletable->True];
      MakeEditable@NotebookWrite[bnb, Choice[NB, DirectoryName@file]]
  ]
]

openProjectFunction[NB_, bnb_] :=
Module[{file},
  file = myFileBrowse[False];
  If[file =!= $Canceled,
    If[StringLength[file]>1 && StringTake[file, -2] === ".m" &&
     ProjectDataQ[file],
      SetOptions[NotebookFind[bnb, NB, All, CellTags], 
        Editable->True, Deletable->True];
        MakeEditable@NotebookWrite[bnb, Choice[NB, file]],
      messageDialog[NotebookDiff::notproj]
    ]
  ]
]    

(* Choice: cell that holds nb choice. *)
Choice[NB_, val_] :=
  Module[{name},
    name = If[Head[val] === String, val, BoxData@ToBoxes@val/.
      InterpretationBox[stuff__]:>InterpretationBox[stuff, Editable->False]];
    Cell[name, "Text", CellFrame->True, FontColor->GrayLevel[0],
      CellTags->NB, CellMargins->{{Inherited, Inherited}, {7,5}},
      Editable->True, ShowSelection -> True, Background->GrayLevel[1],
      FontWeight->"Plain", FontFamily->"Times"]
  ]
  
(* ControlButtons: Close dialog and cancel or execute NotebookDiff *)
ControlButtons[Notebook] := controlbuttons["notebooks", doNotebookDiff]
ControlButtons[Files] := controlbuttons["files", doNotebookDiffFiles]

controlbuttons[txt_, f_] :=
Cell[BoxData[
    ButtonBox[
      $Resource["Diff", "Compare"],
      ButtonFunction:>CompoundExpression[ 
          Needs[ "AuthorTools`NotebookDiff`"], f[ ]],
      ButtonNote->$Resource["Diff", "Compare note", txt],
      Background->RGBColor[0.329412, 0.584314, 0.694118]
  ]], "Text", 
  CellMargins->{{Inherited, Inherited}, {Inherited, 0}},
  TextAlignment->Right,
  FontFamily->"Helvetica",
  FontSize->11,
  FontWeight->"Bold",
  FontColor->GrayLevel[1]]
  
doNotebookDiff[] :=
Module[{in, nb1, nb2},
  in = NotebookGet@ButtonNotebook[];
  {nb1, nb2} = Cases[in, 
    Cell[cont_, ___, CellTags -> "NB1"|"NB2", ___] :> cont, Infinity];
  {nb1, nb2} = {nb1, nb2} /. {BoxData[str_String] :> str, 
    BoxData[intbox_InterpretationBox] :> intbox[[2]],
    BoxData[tmplbox_TemplateBox] :> ReleaseHold@MakeExpression[tmplbox, StandardForm]};
  Which[
    nb1 === "" || nb2 === "", 
     messageDialog[NotebookDiff::empty],
    nb1 === nb2,
     messageDialog[NotebookDiff::same, nb1],
    Head@nb1===String && FileType@nb1=!=File,
     messageDialog[NotebookDiff::nofile, {nb1}],
    Head@nb2===String && FileType@nb2=!=File,
     messageDialog[NotebookDiff::nofile, {nb2}],
    FreeQ[{String, NotebookObject}, Head@nb1],
     messageDialog[NotebookDiff::nbval, nb1],
    FreeQ[{String, NotebookObject}, Head@nb2],
     messageDialog[NotebookDiff::nbval, nb2],
    True,
     NotebookClose[ButtonNotebook[ ], Interactive -> False];
     Block[{$FromPalette=True}, NotebookPut@NotebookDiff[nb1, nb2]]
  ]
]

doNotebookDiffFiles[] :=
Module[{in, nb1, nb2},
  in = NotebookGet@ButtonNotebook[];
  {nb1, nb2} = Cases[in, 
    Cell[cont_, ___, CellTags -> "NB1"|"NB2", ___] :> cont, Infinity];
  {nb1, nb2} = {nb1, nb2} /. {BoxData[str_String] :> str, 
    BoxData[intbox_InterpretationBox] :> intbox[[2]],
    BoxData[tmplbox_TemplateBox] :> ReleaseHold@MakeExpression[tmplbox, StandardForm]};
  Which[
    nb1 === "" || nb2 === "", 
     messageDialog[NotebookDiff::empty1],
    nb1 === nb2,
     messageDialog[NotebookDiff::same, nb1],
    Last@TestFiles@nb1===$Failed,
     Null,
    Last@TestFiles@nb2===$Failed,
     Null,
    True,
     NotebookClose[ButtonNotebook[ ], Interactive -> False];
     Block[{$FromPalette=True}, NotebookPut@NotebookDiffFiles[nb1, nb2]]
  ]
]
  
(*CellDiff -------------------------------------------------------*)
CellDiffButton[] :=
ButtonBox[$Resource["Diff", "View diffs"], ButtonStyle->"CellDiff"]

CellDiffButtonFunction[] :=
Module[{cont1, cont2},
  {cont1, cont2} = GetContent[];
  If[Length@cont1===Length@cont2===1,
    ViewCellDiff[First@cont1, First@cont2],
    SelectCellDiff[{cont1, 1}, {cont2, 1}]
  ]
]

GetContent[]:= 
Module[{bnb = ButtonNotebook[], in, cells},
  SelectionMove[bnb, All, ButtonCell, AutoScroll->False];
  SelectionMove[bnb, All, CellGroup, AutoScroll->False];
  in=NotebookRead[bnb];
  SetOptions[bnb, PrintingStyleEnvironment -> "Working"];
  SetOptions[bnb, PrintingStyleEnvironment -> "Printout"];
  cells = Rest@in[[1, 1]];
  Rest/@Split[cells, !MatchQ[#2, Cell[_, "PositionMiddle"]]& ]
]

SelectCellDiff[{cont1_, num1_}, {cont2_, num2_}] :=
Module[{len, c1, c2, nb},
  len = Max[Length/@{cont1, cont2}];
  {c1, c2} = Map[PadRight[#, len, ""]&, {cont1, cont2}/.$LargeItemRules];
  nb = NotebookPut@Notebook[{
    Cell[$Resource["Diff", "Select one"]],
    Cell[BoxData@RowBox@{
      ButtonBox[$Resource["Show"], Active->True, ButtonEvaluator->Automatic,
        ButtonFunction:>ShowSelectedCellDiff[cont1, cont2]],
      ButtonBox[$Resource["Close"], Active->True,
        ButtonFunction:>NotebookClose[ButtonNotebook[]]]
    }, "Text", TextAlignment -> Right],
    SelectionGrid[{c1, num1}, {c2, num2}]    
   },
   Saveable->False,
   Deletable->False,
   WindowTitle->$Resource["Diff", "Select cells"],
   WindowSize->{Automatic, 400},
   WindowElements->{"VerticalScrollBar"},
   ScrollingOptions->{"VerticalScrollRange"->FitAll},
   ShowCellBracket->False
  ];
  SetSelectedCells[nb, {1,1}]
]

ShowSelectedCellDiff[cont1_, cont2_] :=
Module[{pos1, pos2, bnb = ButtonNotebook[]},
  {pos1, pos2} = "Selected" /. FullOptions[bnb, TaggingRules];
  ViewCellDiff[ cont1[[pos1]], cont2[[pos2]] ]
]

SelectionGrid[{cont1_, num1_}, {cont2_, num2_}] :=
  Cell[BoxData@GridBox[
      Transpose@{MapIndexed[SelectCell[{#, {cont1, cont2}}, #2, {num1, num2}, 1]&, cont1], 
        MapIndexed[SelectCell[{#, {cont1, cont2}}, #2, {num1, num2}, 2]&, cont2]},
        ColumnWidths->.5, GridFrame->True, RowLines->True, ColumnLines->True
    ], CellTags->"SelectionGrid"]

SelectCell[{cell_, cont_}, {pos_}, selpos_, n_] :=
Which[
  cell==="",
    "",
  pos===selpos[[n]], 
    ButtonBox[ReduceCell@cell, Background -> GrayLevel[.4], Active->False],
  True,
    ButtonBox[ReduceCell@cell, Active->True, ButtonEvaluator->Automatic, 
      ButtonFunction:>SelectCellFunction[cont, ReplacePart[selpos, pos, n]]]
]

ReduceCell[Cell[cont_String, ___]] := cont

ReduceCell[Cell[cont_, ___]] := Cell[cont]

SelectCellFunction[{cont1_, cont2_}, {pos1_, pos2_}] :=
Module[{bnb = ButtonNotebook[]},
  NotebookFind[bnb, "SelectionGrid", All, CellTags];
  SetOptions[bnb, Editable->True, Deletable->True];
  NotebookWrite[bnb, SelectionGrid[{cont1, pos1}, {cont2, pos2}]];
  SelectionMove[bnb, Before, Notebook];
  SetSelectedCells[bnb, {pos1, pos2}]
]

SetSelectedCells[nb_, {pos1_, pos2_}] :=
  SetOptions[nb, TaggingRules->{"Selected"->{pos1, pos2}}]
  
ViewCellDiff[cont1_, cont2_] :=
Module[{l, r, both},
  {l, r, both} = cellDiff[cont1, cont2]/.$LargeItemRules;
  l = Join[Flatten@l, both];
  r = Join[Flatten@r, both];
  NotebookPut@Notebook[{
    Cell[BoxData@GridBox[Transpose@{l, r}, GridFrame->True, 
      RowLines->True, ColumnLines->True, ColumnWidths->.5], 
    "Text", CellMargins->{{0,0},{0,0}}]
  }, 
  CellHorizontalScrolling->True,
  ShowCellBracket->False,
  Editable->False,
  Deletable->False,
  WindowTitle->$Resource["Diff", "CellDiffs"],
  WindowSize->{Automatic, Fit},
  WindowElements->{"VerticalScrollBar"},
  ScrollingOptions->{"VerticalScrollRange"->FitAll}
  ]
]

Options[CellDiff] = {IgnoreCellStyleDiffs -> False, IgnoreOptionDiffs -> None}

(* kernel interface to cellDiff *)
CellDiff[cell1_Cell, cell2_Cell, opts___] := 
  If[$Notebooks,
     CellPrint/@Flatten@cellDiff[cell1, cell2, opts];,
    {cell1, cell2}
  ]
  
(*cellDiff: return lists of Cells that identify difference between two cells.
{list1, list2, list3}: 
list1 are cells specific to cell1. 
list2 are cells specific to cell2. 
list3 are cells common to both cells.
CellDiff CellPrints all three lists.
CellDiffButton puts them in a side-by-side grid.*)

(* identical cells *)
cellDiff[cell_, cell_, ___] := 
  {{}, {}, {Cell[$Resource["Diff", "Identical cells"], "Message"]}}
    
$LargeItemRules = {
  gd_GraphicsData:>BoxData[TagBox["Graphic", Hash@gd]],
  gd_GridBox:>TagBox["GridBox", Hash@gd]
}    

RawCellDiff[cell1_, cell2_, opts___?OptionQ] := 
  Module[{igsty, igopt, cont1, cont2, newcont, style, del, ins, up, 
      cellopts, opt1, opt2},
    {igsty, igopt} = {IgnoreCellStyleDiffs, IgnoreOptionDiffs}/.{opts}/.Options[CellDiff];
    (* content *)
    cont1 = First@cell1;
    cont2 = First@cell2;
    (* break the contents into elements based on form. Then diff the elements.*)
    newcont = makeMove[DiffReport@@toelements/@{cont1, cont2}];  
    (* style diff based on IgnoreCellStyleDiffs *)
    style = Map[#[[2]]&, {cell1, cell2}];
    style = Which[
      igsty, {}, 
      First@style===Last@style, {"Style"->{}},
      True, {"Style" -> style} 
    ];
    (* option diffs based on IgnoreOptionDiffs *)
    cellopts = If[igopt === All, {},
        cellopts = sortopts /@ {cell1, cell2};
        If[igopt=!=None && igopt=!={},
          cellopts=DeleteCases[cellopts, _[Alternatives@@Flatten@{igopt}, _], {2}];
        ];
        {del, ins, up} = {"Delete", "Insert", "Update"}/.DiffReport@@cellopts;
        del = Flatten@Map[Last, del];
        ins = Flatten@Map[Last, ins];
        If[del==ins==up=={}, {"Options" -> {}},
          {opt1, opt2} = Flatten /@ Transpose@Map[Take[#, -2]&, up];
          {"Options" -> {Join[opt1, del], Join[opt2, ins]}}
        ]
    ];
    Join[newcont, style, cellopts]
  ]


(* special case for when styles are stripped. *)  
cellDiff[Cell[cont1_, opt1___?OptionQ], Cell[cont2_, opt2___?OptionQ], 
  opts___?OptionQ] :=
cellDiff[Cell[cont1, "", opt1], Cell[cont2, "", opt2], 
  IgnoreCellStyleDiffs->True, opts]
  
cellDiff[cell1_, cell2_, opts___?OptionQ] := 
  Module[{igsty, igopt, cont, cont2, newcont1, newcont2, style, cellopts, newopts,
           both = {}, txt},
    {igsty, igopt} = {IgnoreCellStyleDiffs, IgnoreOptionDiffs}/.{opts}/.Options[CellDiff];
    (* reduce any graphics in the cells to make the cells manageable *)
    cont1 = First@cell1/.$LargeItemRules;
    cont2 = First@cell2/.$LargeItemRules;
    (* break the contents into elements based on form. Then diff the elements.*)
    {newcont1, newcont2} = 
     If[cont1===cont2, (*Identical content*)
       AppendTo[both, nodiff["cell content"]]; {{},{}},
       MapThread[tocontent, 
        {{cont1, cont2}, elementDiff@@toelements/@{cont1, cont2} }
       ]
     ];  
    If[newcont1===$Failed, AppendTo[both, 
          Cell[$Resource["Diff", "Can't compare"], "Message"]];
        {newcont1, newcont2} = {{},{}} 
    ];
    (* style diff based on IgnoreCellStyleDiffs *)
    style = If[igsty, {{}, {}},
        style = Map[#[[2]] &, {cell1, cell2}];
        If[First@style === Last@style, 
          AppendTo[both, nodiff["cell style"]];{{}, {}}, 
          Map[Cell[TextData[{"Style", ":", " ", 
            sty[RGBColor[1, 0, 1], "\[LeftAngleBracket]"], #,
            sty[RGBColor[1, 0, 1], "\[RightAngleBracket]"]}], "Text",
            CellGroupingRules->"OutputGrouping"] &, style]
        ]
    ];
    (* option diffs based on IgnoreOptionDiffs *)
    cellopts = If[igopt === All, {{}, {}},
        cellopts = sortopts /@ {cell1, cell2};
        txt = "cell options";
        If[igopt=!=None && igopt=!={},
          cellopts=DeleteCases[cellopts, _[Alternatives@@Flatten@{igopt}, _], {2}];
          txt = "allowed cell options"
        ];
        cellopts = Map[ToString[#, StandardForm] &, cellopts, {2}];
        newopts = elementDiff @@ cellopts/.
          (FontColor->_RGBColor)->(FontColor->RGBColor[1,0,1]);
        If[newopts === cellopts, AppendTo[both, nodiff[txt]]; {{}, {}},  
          cellopts = Map[
            createOptList, newopts];         
          Map[Cell[BoxData@RowBox[Flatten@{"Options", ":", #}], "Text", 
            FontFamily->"Times", CellGroupingRules->"OutputGrouping"] &, cellopts]
        ]
    ];
    {{newcell[cell1, newcont1], First@style, 
        First@cellopts}, {newcell[cell2, newcont2], Last@style, Last@cellopts}, 
      both}
  ]
  
createOptList[{}] := "None"
  
createOptList[opts_List] :=
  Drop[Flatten@MapThread[insertbreaks, {opts,RotateLeft@opts}], -1]
  
nodiff[x_] := 
  Cell[ToString @ StringForm[NotebookDiff::nodiff, x], "SmallText", 
    CellGroupingRules->"OutputGrouping"]
    
toelements[cont_] := Switch[cont,
  _String, toWords@cont,
  TextData[_List]|BoxData[_List], Flatten[toelements/@First@cont],
  BoxData[RowBox[_]], First@reduce@First@cont,
  BoxData[_String], toWords@First@cont,
  _TextData|_BoxData, List@@cont,
  _OutputFormData, toelements@Last@cont,
  _RawData, toelements@First@cont,
  _StyleData, toelements@First@cont,
  _, cont
]
  
tocontent[oldcont_, newelem_] := Switch[oldcont,
  _String|TextData[_List], TextData@newelem,
  _TextData, TextData@newelem,
  BoxData[_List], BoxData@newelem,
  _BoxData, BoxData@RowBox@newelem,
  _OutputFormData, TextData@newelem,
  _RawData, TextData@newelem,
  _StyleData, TextData@newelem,
  _, newelem
]
  
insertbreaks[x:StyleBox["\[LeftAngleBracket]", ___], _] = x
insertbreaks[x_, StyleBox["\[RightAngleBracket]", ___]] = x
insertbreaks[x_, _] = {x, ","}
  
toWords[str_String] :=
Module[{strm=StringToStream@str, words},
  words = ReadList[strm, Word, TokenWords -> {" ", "\n", "\r", "\t"}];
  Close[strm];
(* This clumps the spaces on the end of the word. *)
  words //. {a___, b_, sp:" "|"\n"|"\r"|"\t", c___} :> {a, b<>sp, c} 
]
    
(*newcell: remove styles from Cell level so that they don't 
  conflict with the colors created by cellDiff *)
newcell[cell_, cont_] := 
  Cell[cont, "Text", ShowAutoStyles->False, CellGroupingRules->"OutputGrouping"]

newcell[_, {}] := {}
      
(* sort options so that there can be no "Move" differences *)
sortopts[cell_Cell] := 
  Sort[Drop[List @@ cell, 2], (OrderedQ[First /@ {#1, #2}]) &]
  
(* diff the elements and mark the difference *) 
elementDiff[$Failed, _] = elementDiff[_, $Failed] = {$Failed, $Failed}
 
elementDiff[el1_, el2_] := 
  Module[{pos, del, ins, up, mv, new1, new2, mark1, 
      mark2}, {del, ins, up, mv} = {"Delete", "Insert", "Update", "Move"} /. 
        makeMove@DiffReport[el1, el2];
    new1 = If[el1==={}, {},
     {t1, t2, t3} = {Map[First, del], Map[First, up], Map[#[[3]] &, mv]};
     {l1, l2, l3} = Length /@ {t1, t2, t3};
     mark1 = 
      Sort[Join[Transpose[{t1, Table[RGBColor[0, 1, 0], {l1}]}], 
          Transpose[{t2, Table[RGBColor[0, 1, 0], {l2}]}], 
          Transpose[{t3, 
              Table[RGBColor[0, 1, 0], {l3}]}]], (#1[[1, 1]] > #2[[1, 
                  1]]) &];
     Fold[mark, el1, mark1]
    ];
    new2 = If[el2==={}, {},
     {t1, t2, t3} = {Map[#[[2]] &, ins], Map[#[[2]] &, up], 
        Map[#[[2]] &, mv]};
     {l1, l2, l3} = Length /@ {t1, t2, t3};
     mark2 = 
      Sort[Join[Transpose[{t1, Table[RGBColor[0, 1, 0], {l1}]}], 
          Transpose[{t2, Table[RGBColor[0, 1, 0], {l2}]}], 
          Transpose[{t3, 
              Table[RGBColor[0, 1, 0], {l3}]}]], (#1[[1, 1]] > #2[[1, 
                  1]]) &];
     Fold[mark, el2, mark2]
    ];
    {new1, new2}]
    
mark[el_, {pos_, color_}] := 
  Insert[Insert[el, sty[color, "\[RightAngleBracket]"], Last@pos + 1], 
    sty[color, "\[LeftAngleBracket]"], First@pos]
  
sty[color_, text_] :=  
  StyleBox[text, FontColor->RGBColor[0, 1, 0], FontSize->16, FontWeight->"Bold"]

(* flatten RowBox[]es to have any hope at a useful linear diff *)      
reduce[boxes_] := 
  boxes //. RowBox[{x___, RowBox[{y__}], z___}] :> RowBox[{x, y, z}]

positiontext[{begin_Integer, ___, end_Integer}, nb_] :=
  ToString @ StringForm[NotebookDiff::pos1, $Resource["Diff", nb], begin, end]
    
positiontext[{pos_Integer}, nb_] :=
  ToString @ StringForm[NotebookDiff::pos, $Resource["Diff", nb], pos]
  
positiontext[pos_Integer, nb_] :=
  ToString @ StringForm[NotebookDiff::before, $Resource["Diff", nb], pos]

(* :STYLESHEETS: *)

StyleSheet[nb_NotebookObject] :=
Module[{sd},
  sd = StyleDefinitions/.Options[nb];
  sd = If[sd===StyleDefinitions, 
    DefaultStyleDefinitions/.Options[$FrontEnd, DefaultStyleDefinitions],
    sd
  ]
]
  
StyleSheet[nb_Notebook] := 
Module[{sd},
  sd = StyleDefinitions/.List@@Rest[nb];
  sd = Which[
    sd===StyleDefinitions && $Notebooks, 
      DefaultStyleDefinitions/.Options[$FrontEnd, DefaultStyleDefinitions],
    sd===StyleDefinitions, (* && !$Notebooks*)
      "Default.nb",
    True,  
      sd
  ]
]

StyleSheet[nb_String] := StyleSheet[myImport[nb, "Notebook"]]

(* make new stylesheet. *)	
importss[nb_] := 
Module[{ss1, ss, dummy},
  ss1 = StyleSheet@nb;
  ss = makenewstylesheet@ss1;
  StyleDefinitions -> ss
]

(* Create  from a shared style sheet *)
makenewstylesheet[nb_FrontEnd`FileName] := 
  makenewstylesheet[ToFileName@nb]
   
makenewstylesheet[nb_String] := 
Module[{file, ss, dirs},
   dirs = If[$Notebooks,
     AbsoluteCurrentValue[$FrontEnd, StyleSheetPath],
     Append[
      Map[ToFileName[{#, "SystemFiles", "FrontEnd", "StyleSheets"}]&,
       {$UserAddOnsDirectory, $AddOnsDirectory, $TopDirectory}],
       ToFileName[{$TopDirectory, "Configuration", "FrontEnd", "StyleSheets"}]
     ]
   ];
(* Search for relative filename along StyleSheetPath and absolute filename *)
   file = First@Flatten[{Map[listfilesindir[#, nb]&, dirs], nb}];
   ss = If[FileType@file===File, 
    Get@file, 
    "Default.nb"
   ];
   makenewstylesheet@ss
]

(* Get filenames for directories that include _ *)
listfilesindir[FrontEnd`FileName[{begin___, Verbatim[_], end___}, ___], 
    name_] :=
  Module[{choices}, choices = FileNames["*", ToFileName[{begin}]];
    FileNames[ToFileName[{end}, name], choices]
    ]

(* Get filenames for directories that do not include _ *)
listfilesindir[FrontEnd`FileName[{all___}, ___], name_] :=
  FileNames[name, ToFileName[{all}]]
  
listfilesindir[dir_String, name_] :=
  FileNames[name, dir]

(* Append new styles to existing stylesheet *)
makenewstylesheet[nb_Notebook] := 
  Notebook@Join[First@nb, 
    First@Get["DiffStyles.nb", Path->{
     ToFileName@{$TRDir, AuthorTools`Common`Private`$AuthorToolsLanguage}, $TRDir
    }]
  ]
  
$TRDir = ToFileName[{AuthorTools`Common`Private`$AuthorToolsDirectory,
  "FrontEnd", "TextResources"}]
  
makenewstylesheet[nb_] := makenewstylesheet["Default.nb"]

(* NotebookDiffFiles *)

NotebookDiffFiles[f1_, f2_, opts___?OptionQ] :=
Module[{file1, file2, type, type1, type2, short1, short2, common, cells, f, g, dialog, depth}, 
  depth = DiffRecursionDepth /. {opts} /. Options[NotebookDiff];
  (* testing and convert input to file lists *)
  {{type1, file1}, {type2, file2}} = TestFiles[#, depth]&/@{f1,f2};
  If[file1===$Failed || file2===$Failed, Return[$Failed]];
  type = If[type1===type2, type1<>" ", ""];
  If[!goodopts[NotebookDiff, opts], Return[$Failed]];
  (* diff tracking. Automatic is handled internally. *)
  f = ShowDiffProgress /. {opts} /. Options[NotebookDiff];
  g = If[TrueQ@f, 
    If[TrueQ@$FromPalette,
      dialog = ProgressDialog[$Resource["Diff", "Computing"], ""]
    ];
    False, f
  ];
  (* remove files not in common *)
  If[ type1 === type2 === "Directory",
    {short1, short2} = {ShortName[file1, f1], ShortName[file2, f2]},
    {short1, short2} = {ShortName[file1], ShortName[file2]}
  ];
  common = Intersection[short1, short2];
  If[common==={}, 
    MessageDisplay[NotebookDiff::noint, short1, short2];
    If[TrueQ@f && TrueQ@$FromPalette, ProgressDialogClose@dialog];
    Return[$Failed]
  ];
  file1 = Map[Part[file1, Position[short1, #][[1,1]]]&, common];
  file2 = Map[Part[file2, Position[short2, #][[1,1]]]&, common];
  (* find differences *)
  cells = Map[
    (SideEffect[f, TrueQ@$FromPalette, dialog, ShortName@First@#];
     ChangedNotebook[#, ShowDiffProgress->g, opts])&, 
    Transpose@{file1, file2}
  ];
  {c1, c2} = {DeleteCases[cells, _same], Cases[cells, same[file_]->Cell[file,"Text"]]};
  cells = Flatten@{
    Cell[$Resource["Diff", type, "Differences"], "Title", ShowCellBracket->False],
    Cell[$Resource["Diff", "Changed notebooks"], "Section"], 
    c1,
    Cell[$Resource["Diff", "Identical notebooks"], "Section"], 
    c2,
    Cell[
      If[Head@f1===List,
        $Resource["Diff", "Only first list"],
        StringJoin@$Resource["Diff", "Only prefix", f1]
      ], "Section"],
    Map[Cell[#,"Text"]&, Complement[short1, common]],
    Cell[If[Head@f1===List,
        $Resource["Diff", "Only first list"],
        StringJoin@$Resource["Diff", "Only prefix", f2]
      ], "Section"],
    Map[Cell[#,"Text"]&, Complement[short2, common]]
  };
  If[TrueQ@f && TrueQ@$FromPalette, ProgressDialogClose@dialog];
  Notebook[cells,
        WindowTitle->type<>"Differences",
        ShowCellBracket->False
      ]
]


SideEffect[True, True, dialog_, file_] := 
  ProgressDialogSetSubcaption[dialog, ToString @ StringForm[NotebookDiff::proc, file]]
SideEffect[True, False, dialog_, file_] := 
  Print[ToString @ StringForm[NotebookDiff::proc, file]]

ShortName[files_List, res___] := ShortName[#, res]& /@ files

ShortName[file_String] := 
  StringReplace[file, DirectoryName@file->""]

ShortName[file_String, dir_String] := 
  StringReplace[file, dir -> ""]

TestFiles[files:{__String?(StringMatchQ[#, "*.nb"]&)}, ___] :=
{"Notebook List",
Module[{badfiles},
  badfiles = Select[files, FileType[#]=!=File&];
  If[badfiles=!={},
    MessageDisplay[NotebookDiff::nofile, badfiles]; $Failed,
    files
  ]
]}

TestFiles[file_String/;StringMatchQ[file, "*.nb"], ___] :=
{"Notebook List",
If[FileType[file]=!=File,
  MessageDisplay[NotebookDiff::nofile, {file}]; $Failed,
  {file}
]}

TestFiles[project_String/;StringMatchQ[project, "*.m"], ___] :=
{"Project",
If[!ProjectDataQ[project],
  MessageDisplay[NotebookDiff::noproj, project]; $Failed,
  Last@TestFiles@Map[ToFileName[ProjectDirectory@project, #]&, ProjectFiles@project]
]}

TestFiles[dir_String] := TestFiles[dir, Infinity]

TestFiles[dir_String, depth_] :=
{"Directory",
If[FileType@dir=!=Directory,
  MessageDisplay[NotebookDiff::nodir, dir]; $Failed,
  FileNames["*.nb", dir, depth]
]}

TestFiles[else_, ___] := 
(
  MessageDisplay[NotebookDiff::dirval, else];
  {"None", $Failed}
)

ChangedNotebook[{file1_, file2_}, opts___] :=
Module[{diff, dummy},
  diff = NotebookDiff[file1, file2, opts];
  If[Drop[First@diff, 3] === Flatten@{diffCell["None"], diffCell["Option",{}]},
    same[ShortName@file1],
    Cell[TextData@{ButtonBox[ShortName@file1,
      ButtonEvaluator->Automatic,
      ButtonFunction:>NotebookPut@dummy,
      Active->True,
      ButtonStyle->"Hyperlink"
    ]/.dummy->diff}, "Text"]
  ]
]
            
End[]

EndPackage[]
