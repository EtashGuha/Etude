(* :Context: AuthorTools`MakeBilateralCells` *)

(* :Author: Louis J. D'Andria *)

(* :Summary:
    This package defines functions for creating and 
    manipulating two-column cells, useful for laying
    out input/output next to exposition.
*)

(* :Copyright: *)

(* :Package Version: $Revision: 1.19 $, $Date: 2004/11/19 17:03:31 $*)

(* :Mathematica Version: 4.2 *)

(* :History:
    
*)

(* :Keywords:
    
*)

(* :Discussion:
    
*)

(* :Warning:
    This package requires the HelpBrowser.nb style sheet
    included with V4.2 and later versions of Mathematica.
*)
  


BeginPackage["AuthorTools`MakeBilateralCells`", "AuthorTools`Common`"]


PasteBilateralTemplate::usage = "PasteBilateralTemplate[nb] pastes a properly formatted cell with some placeholders into nb.";


MakeBilateral::usage = "MakeBilateral[NotebookSelection[nb]] replaces the selected cells with a single bilateral cell, whose left column contains the first cell, and whose right column contains the remaining cells. MakeBilateral[nb] replaces all sequences of cells indicated by FirstBilateralStyles and RestBilateralStyles.";

DivideBilateral::usage = "DivideBilateral[NotebookSelection[nb]] replaces any bilateral cells in the current selection with their constituent cells. DivideBilateral[nb] does so for the entire notebook.";


ToBilateral::usage = "ToBilateral[cellList] returns the bilateral cell form of the given cells.";

FromBilateral::usage = "FromBilateral[cell] returns a list of the constituent cells from the given bilateral cell. FromBilateral[cellList] returns the list with each bilateral cell so expanded.";


FirstBilateralStyles::usage = "FirstBilateralStyles is an option to MakeBilateral which determines which cell styles should indicate the caption of a bilateral cell. Default value is $FirstBilateralStyles.";

RestBilateralStyles::usage = "RestBilateralStyles is an option to MakeBilateral which determines which cell styles should indicate the content of a bilateral cell. Default value is $RestBilateralStyles.";

$FirstBilateralStyles::usage = "$FirstBilateralStyles gives the default setting for the option FirstBilateralStyles in MakeBilateral.";

$RestBilateralStyles::usage = "$RestBilateralStyles gives the default setting for the option RestBilateralStyles in MakeBilateral.";


Begin["`Private`"]



(* ToBilateral and FromBilateral *)

formatLeftCell[x_Cell] := x;

formatRightCell[Cell[x_, y___, CellLabel->cl_, z___], opts___] :=
Block[{lab},
  lab = StringReplace[cl, "(Dialog) " -> ""];
  {Cell[lab, "CellLabel"], Cell[x,y,z]}
]

formatRightCell[Cell[x_, y___], opts___] :=
  {Cell["", "CellLabel"], Cell[x,y]}



$BilateralStyle = "BilateralCell";

ToBilateral[{c1_Cell, c2__Cell}] :=
Block[{cellList, cellTags},
  cellList = FlattenCellGroups[{c2}];
  
  cellTags = Union @ Flatten @
    Cases[Append[cellList,c1], _[CellTags, t_]:>t, {2}];
  
  addAndRemoveCellTags[#, cellTags, {}]& @ 
  Cell[BoxData[FormBox[GridBox[{{
    formatLeftCell @ c1,
    GridBox[
      formatRightCell /@ cellList,
      ColumnAlignments->Left,
      RowAlignments->Baseline,
      ColumnWidths->Automatic
    ]}},
    RowAlignments->Top,
    ColumnAlignments->Left], InlineCell]],
  $BilateralStyle
  ]
]



unformatLeftCell[x_Cell] := x;

unformatRightCell[{x_Cell}] := x;

unformatRightCell[{Cell["", "CellLabel"], Cell[y___]}] :=
  Cell[y]

unformatRightCell[{Cell[TextData[x_], "CellLabel"], Cell[y___]}] :=
  Cell[y, CellLabel->x]

unformatRightCell[{Cell[x_, "CellLabel"], Cell[y___]}] :=
  Cell[y, CellLabel->x]



(* Case 1: the cell has one or more cell tags to begin with. *)

addAndRemoveCellTags[Cell[x___, CellTags -> t_String, y___], add_, remove_] :=
  addAndRemoveCellTags[Cell[x, CellTags -> {t}, y], add, remove]

addAndRemoveCellTags[Cell[x___,CellTags -> lis_List, y___], add_, remove_] :=
Block[{newlis},
  newlis = Union[add, Complement[lis, remove]];
  Switch[newlis,
    {}, Cell[x, y],
    {_}, Cell[x, CellTags -> First[newlis], y],
    {_,__}, Cell[x, CellTags -> newlis, y]
  ]
]

(* Case 2: it doesn't. Note the 'remove' argument is irrelevant. *)

addAndRemoveCellTags[c_Cell, {}, _] := c;

addAndRemoveCellTags[Cell[x___], add_, _] :=
  addAndRemoveCellTags[Cell[x, CellTags -> {}], add, {}]



FromBilateral[Cell[BoxData[FormBox[
    GridBox[{{c1_Cell, GridBox[c2_, ___]}}, ___], InlineCell]],
    $BilateralStyle, opts___]] :=
Block[{btags, ctags, addtags, removetags, tmp, result},
  result = Flatten[{
    unformatLeftCell[c1],
    unformatRightCell /@ c2
  }];
  
  btags = CellTags /. Flatten[{opts, CellTags -> {}}];
  btags = Flatten[{btags}];
  ctags = Union @ Flatten @ Cases[result, _[CellTags, t_]:>t, {2}];
  
  (* 
    If there are tags in the bilateral cell that aren't in the
    subcells, the user added them. Add them to every subcell. If
    there are tags in the subcells that aren't in the bilateral
    cell, the user removed them. Remove them from the subcells.
  *)
  
  addtags = Complement[btags, ctags];
  removetags = Complement[ctags, btags];
    
  addAndRemoveCellTags[#, addtags, removetags]& /@ result

] /; Apply[Union, Map[Head,c2,{2}]]==={Cell}


FromBilateral[{cells__Cell}] :=
  cells //. c:Cell[___, $BilateralStyle, ___] :> FromBilateral[c]



(* MakeBilateral and DivideBilateral *)


currentCell[nb_] := 
  If[#[[0]] =!= Cell, $Failed, # ]&[ NotebookRead[nb] ]

currentCellStyle[nb_] :=
  If[Length[#] =!= 1, $Failed, "Style" /. First[#]]&[ Developer`CellInformation[nb] ]


findNextBeginning::usage = "findNextBeginning[nb,styles] searches nb for the next cell with a style in styles.  It then deletes that cell from the notebook, and returns that cell's Cell expression.";

findNextBeginning[nb_, styles_] := 
Block[{cellExpr, sty},
  While[
    sty = currentCellStyle[nb];
    sty =!= $Failed && !MemberQ[styles, sty]
    ,
    SelectionMove[nb, Next, Cell];
  ];
  cellExpr = currentCell[nb];
  If[cellExpr =!= $Failed, NotebookDelete[nb]];
  cellExpr
]



findRest::usage = "findRest[nb,styles] gathers a consecutive sequence of cells whose styles are among styles, deleting them from nb and saving their Cell expressions.  When it's finished, it returns a list of the Cell expressions for the cells it deleted.";

findRest[nb_, styles_] :=
Block[{cellList = {}, cellExpr, sty},
  While[
    SelectionMove[nb, Next, Cell];
    sty = currentCellStyle[nb];
    If[MemberQ[styles, sty],
      cellExpr = currentCell[nb];
      NotebookDelete[nb];
      AppendTo[cellList, cellExpr];
      True
      ,
      SelectionMove[nb, Before, Cell];
      False
    ]
  ];
  cellList
]  



$FirstBilateralStyles =
  {"MathCaption"};
$RestBilateralStyles = 
  {"Input", "Output", "Graphics", "Picture", "Print", "Message"};


Options[MakeBilateral] =
{
  FirstBilateralStyles :> $FirstBilateralStyles,
  RestBilateralStyles :> $RestBilateralStyles
};


MakeBilateral::fail = "MakeBilateral failed.  Make sure you have selected the cells on which to operate.";


MakeBilateral[NotebookSelection[nb_NotebookObject], opts___] :=
Block[{cellList},
  cellList = NotebookRead[nb];
  cellList = FlattenCellGroups[cellList];
  If[!MatchQ[cellList, {_Cell, __Cell}], 
    Message[MakeBilateral::fail];
    $Failed
    ,
    NotebookWrite[nb, ToBilateral[cellList], All]
  ]
]


MakeBilateral[file_String, opts___] :=
  ExtendNotebookFunction[MakeBilateral, file, opts] /; FileType[file] === File


MakeBilateral[proj_NotebookObject, opts___] :=
  ExtendNotebookFunction[MakeBilateral, proj, opts] /; ProjectDialogQ[proj]


MakeBilateral[nb_NotebookObject, opts___] :=
Block[{leftCell, rightCells, sty1, sty2},
  
  {sty1, sty2} = {FirstBilateralStyles, RestBilateralStyles} /. 
    Flatten[{opts, Options[MakeBilateral]}];

  SelectionMove[nb, Before, Notebook];
  SelectionMove[nb, Next, Cell];
  
  While[(leftCell = findNextBeginning[nb, sty1]) =!= $Failed,
    rightCells = findRest[nb, sty2];
    If[rightCells === {},
      NotebookWrite[nb, leftCell],
      NotebookWrite[nb, ToBilateral @ Flatten @ {leftCell, rightCells}]
    ];
    SelectionMove[nb, Next, Cell]
  ]
]





DivideBilateral::fail = "DivideBilateral failed.  Make sure you have selected a bilateral cell.";

DivideBilateral[NotebookSelection[nb_NotebookObject], opts___] :=
Block[{cells},
  cells = NotebookRead[nb];
  If[Position[{cells}, Cell[___, $BilateralStyle, ___]] === {},
    Message[DivideBilateral::fail];
    Return[$Failed]
  ];
  
  cells =
    cells //. c:Cell[___, $BilateralStyle, ___] :> FromBilateral[c];
    
  NotebookWrite[nb, Flatten @ {cells}, All]
]


DivideBilateral[file_String, opts___] :=
  ExtendNotebookFunction[DivideBilateral, file, opts] /; FileType[file] === File


DivideBilateral[proj_NotebookObject, opts___] :=
  ExtendNotebookFunction[DivideBilateral, proj, opts] /; ProjectDialogQ[proj]


DivideBilateral[nb_NotebookObject, opts___] :=
(
  SelectionMove[nb, Before, Notebook];
  While[NotebookFind[nb, $BilateralStyle, Next, CellStyle] =!= $Failed,
    NotebookWrite[nb, #]& @ FromBilateral @ NotebookRead[nb];
  ]
)






PasteBilateralTemplate[nb_NotebookObject] := 
(
  SelectionMove[nb, After, Cell];
  NotebookWrite[nb, 
    ToBilateral[{
      Cell[$Resource["Bilateral", "SampleText"],
        "MathCaption"],
      Cell[$Resource["Bilateral", "SampleInput"],
        "Input", CellLabel -> "In[1]:="],
      Cell[BoxData[$Resource["Bilateral", "SampleOutput"]],
        "Output", CellLabel -> "Out[1]="]
    }]
  ]
) 




End[]

EndPackage[]
