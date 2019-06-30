(* :Context: AuthorTools`NotebookRestore` *)

(* :Author: Louis J. D'Andria *)

(* :Summary:
   This package attempts to extract useable cells from otherwise
   corrupted notebooks.
 *)

(* :Copyright: *)

(* :Package Version: $Revision: 1.6 $ $Date: 2009/04/20 19:17:48 $ *)

(* :Mathematica Version: 4.2 *)

(* :History:
   + Started life as OpenCorruptedNotebooks.nb (97/01)
   + Then became OpenCorruptedNotebooks.m (97/01)
   + Then because Corruption.m (97/02/27)
   + Added DeleteTypesetCells and DeleteGraphicCells options at the
     suggestion of StanWagon (97/06/17)
   + Worked around a bug in SetStreamPosition reported by Xah Lee
     (97/09/10)
   + Simplied interface and added to AuthorTools as NotebookRestore.m, 2002
   + Simplified function/option interface, 2003
*)

(* :Keywords:
   Notebook, FrontEnd, Corruption, Salvage, Restore
*)

(* :Discussion:
    
*)

(* :Warning:
    
*)

(* :Limitations:
   + Notebook options are lost, including style sheet.
   + Comments at the top and bottom of notebook are lost, including
     cache information.
   + Uses raw strings to determine where to break cells, which has
     the possibility of being fooled by notebooks containing certain
     strings.
   + Uses Read[st, String], so line endings must be native for your
     platform.
*)



BeginPackage["AuthorTools`NotebookRestore`", "AuthorTools`Common`"]


NotebookRestore::usage = "NotebookRestore[file.nb] recreates the given notebook by salvaging individual cells out of file.nb and writing them, one by one as they are found, into a new notebook window.";

SalvageCells::usage = "SalvageCells[file.nb] returns a list of cells salvaged from file.nb."

NextCorruptCell::usage = "NextCorruptCell[nbObj] highlights the next cell in the given nbObj with cell tag \"Corrupt\"";

DeleteCorruptCells::usage = "DeleteCorruptCells is an option that determines whether NotebookRestore and SalvageCells will preserve the corrupt data or discard it.";

IgnoreGraphicsCells::usage = "IgnoreGraphicsCells is an option that determines whether NotebookRestore and SalvageCells should check the syntax of graphics cells or discard them without checking.";

IgnoreTypesetCells::usage = "IgnoreTypesetCells is an option that determines whether NotebookRestore and SalvageCells functions should check the syntax of typeset cells or discard them without checking.";


Begin["`Private`"]



Options[NotebookRestore] = Options[SalvageCells] =
{
  DeleteCorruptCells -> True,
  IgnoreGraphicsCells -> True,
  IgnoreTypesetCells -> False
}


NotebookRestore[] :=
Module[{file},
  file = SystemDialogInput["FileOpen"];
  If[Head[file] =!= String || FileType[file] =!= File, Return[]];  
  NotebookRestore[file]
]


NotebookRestore::prog = "Processing `1`\[Ellipsis]";

NotebookRestore[file_, opts___] :=
Module[{nb},
  If[Head[$FrontEnd] =!= FrontEndObject,
    Abort[],
    $interactive = True;
    $SalvageNotebook = NotebookCreate[TaggingRules -> {"Salvaged" -> True}];
    nb = ProgressDialog[
      ToString @ StringForm[NotebookRestore::prog, NotebookName @ file], ""];
    SalvageCells[file, opts, Sequence @@ Options[NotebookRestore]];
    ProgressDialogClose[nb];
    $interactive = False;
    $SalvageNotebook
  ]
] /; FileType[file] === File



SalvageCells[file_, opts___] := 
Module[{st, cellList, pos, temp, nextCell},
  
  If[(st = OpenRead[file])===$Failed, Abort[]];
  
  cellList = {};
  
  {$deleteCorruptCells, $ignoreGraphicsCells, $ignoreTypesetCells} =
    {DeleteCorruptCells, IgnoreGraphicsCells, IgnoreTypesetCells} /.
      Flatten[{opts, Options[SalvageCells]}];
  
  (*skip the comment at the top of the notebook*)
  moveToStartingLine[st];
    
  (* main loop *)
  While[(pos = StreamPosition[st];
         temp = Read[st,String];
         SetStreamPosition[st,pos];
         temp) =!= EndOfFile,
    nextCell = getNextCell[st];
    If[Head[nextCell] === Cell,
      If[TrueQ[$interactive],
        NotebookWrite[$SalvageNotebook, nextCell],
        cellList = Join[cellList, {nextCell}]
      ]
    ]
  ];
  Close[st];
  cellList
] /; FileType[file] === File






groupingDataQ[lis_] :=
Module[{lis2},
  lis2 = DeleteCases[lis, "" |
                          "\n" |
                          "Notebook[{" |
                          "Cell[CellGroupData[{" |
                          "}, Open]]" |
                          "}, Open]]," |
                          "}, Open  ]]" |
                          "}, Open  ]]," |
                          "}, Closed]]" |
                          "}, Closed]],"];
  If[lis2==={}, True, False]
]


startingLineQ[str_] := 
  StringMatchQ[str, "Cell[*"] && !StringMatchQ[str,"Cell[CellGroupData[{*"];
  

endingLineQ[str_] := 
Or[
  StringMatchQ[str, "Cell[*"],
  StringMatchQ[str, "Cell[CellGroupData[{*"],
  StringMatchQ[str, "}, Open]]*"],
  StringMatchQ[str, "}, Open  ]]*"],
  StringMatchQ[str, "}, Closed]]*"],
  StringMatchQ[str, "},"],
  StringMatchQ[str, "Cached data follows*"],
  StringMatchQ[str, "(* End of Notebook Content *)"]
];


gData = "Cell[GraphicsData[*";
bData = "Cell[BoxData[*";




moveToStartingLine[st_] := 
Module[{temp="", pos},
  
  While[temp=!=EndOfFile && Not[startingLineQ[temp]],
    pos = StreamPosition[st];
    temp = Read[st,String]
  ];
  
  If[temp =!= EndOfFile,
    SetStreamPosition[st, pos]
  ];
];



getNextCell[st_] := 
Module[{strList, temp, pos, c},
    
  (* read the first, good line off the stream immediately *)
  
  strList = {Read[st, String]};
  pos = StreamPosition[st];
  temp = Read[st, String];
  
  (* if the first line indicates a graphics cell or typeset cell
     follows, then simply keep count of the lines as they pass
     by, provided the relevant flag is True *)
  
  If[($ignoreGraphicsCells && StringMatchQ[First @ strList, gData]) ||
     ($ignoreTypesetCells && StringMatchQ[First @ strList, bData]),
     c=1;
     While[temp =!= EndOfFile && !endingLineQ[temp],
       c++;
       pos = StreamPosition[st];
       temp = Read[st, String]
     ],
  
  (* otherwise, keep reading until you encounter an endingLine *)

    While[temp =!= EndOfFile && !endingLineQ[temp],
      strList = Join[strList, {temp}];
      pos = StreamPosition[st];
      temp = Read[st, String]
    ]
  ];
  
  (* replace strList with a custom cell if the user wants 
     typeset or graphics cells removed *)

  If[$ignoreGraphicsCells && StringMatchQ[First @ strList, gData],
    strList = {ignoredGraphicsCell[c]}
  ];
    
  If[$ignoreTypesetCells && StringMatchQ[First @ strList, bData],
    strList = {ignoredTypesetCell[c]}
  ];

  (* "Cached data follows" is the marker for the end of usable cells *)

  If[temp =!= EndOfFile && (
      StringMatchQ[temp, "Cached data follows*"] ||
      StringMatchQ[temp, "(* End of Notebook Content *)"]),
    SetStreamPosition[st, Infinity];
    
    (* There's a bug in the 3.0.1 kernel, fixed in 3.1, that prevents 
       SetStreamPosition[st,Infinity] from setting the position to the 
       end of the file for some (large) files.  In those cases, the 
       following line is also needed. In the non-buggy cases, the 
       following won't slow things up that much. *)
    
    While[Read[st, String] =!= EndOfFile];
    temp = EndOfFile
  ];
  
  (* If the search stopped because it hit the end of the file, it
     did not find a valid cell. Otherwise, check the syntax, and
     return a good or bad cell accordingly. *)
  
  If[temp === EndOfFile,
    createCorruptCell[strList],
    SetStreamPosition[st, pos];
    checkSyntax[strList]
  ]
];




combineLines[lis_] := StringJoin @ BoxForm`Intercalate[lis, "\n"]


checkSyntax[lis_] := 
Module[{str = combineLines[lis], cellstr, cell, synlen},
  synlen = SyntaxLength[str];
  If[synlen > StringLength[str],
    createCorruptCell[lis],
    (*otherwise, there's a chance the syntax is ok*)
    cellstr = StringTake[str, synlen];
    cell = ToExpression[cellstr];
    If[cell===$Failed || Head[cell] =!= Cell,
      createCorruptCell[lis],
      cell
    ]
  ]
];



NotebookRestore::gra = "This was a graphics cell spanning `1` lines.";
NotebookRestore::typ = "This was a typeset cell spanning `1` lines.";
NotebookRestore::sty = "Style sheet and other notebook options ignored.";
NotebookRestore::cor = "`1` lines of corrupt data deleted.";


ignoredGraphicsCell[n_] := StringJoin[
  "Cell[\"",
  ToString @ StringForm[NotebookRestore::gra, n],
  "\", \"Text\", FontColor->",
  ToString @ $Resource["Restore", "Graphic color"],
  ",CellTags->\"Ignored Graphic\"]"
]


ignoredTypesetCell[n_] := StringJoin[
  "Cell[\"",
  ToString @ StringForm[NotebookRestore::typ, n],
  "\", \"Text\", FontColor->",
  ToString @ $Resource["Restore", "Typeset color"],
  ",CellTags->\"Ignored Typsetting\"]"
]


createCorruptCell[lis_] := 
(* if the info is just grouping info, there's no error*)
If[groupingDataQ[lis],
  {},
  Cell[
    Which[
      Length[lis] > 1 && StringMatchQ[lis[[2]], "FrontEndVersion*"],
      ToString @ StringForm[NotebookRestore::sty]
      ,
      $deleteCorruptCells,
      ToString @ StringForm[NotebookRestore::cor, Length[lis]]
      ,
      True,
      combineLines[lis]
    ],
    "Text",
    FontFamily->$Resource["Restore", "Corrupt font"],
    FontColor->$Resource["Restore", "Corrupt color"],
    Background->$Resource["Restore", "Corrupt background"],
    FontWeight->"Bold",
    CellTags->"Corrupt"]
];


NextCorruptCell::sal = "The notebook `1` was not created by NotebookRestore.";

NextCorruptCell[nb_NotebookObject] :=
Module[{salvagedQ},
  salvagedQ = TrueQ["Salvaged" /. (TaggingRules /. Options[nb, TaggingRules])];
  If[salvagedQ,
    NotebookFind[nb, "Corrupt", Next, CellTags],
    MessageDisplay[NextCorruptCell::sal, nb];
    $Failed
  ]
]


End[ ]

EndPackage[ ]
