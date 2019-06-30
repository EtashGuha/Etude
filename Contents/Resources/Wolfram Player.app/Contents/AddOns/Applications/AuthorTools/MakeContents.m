(* :Context: AuthorTools`MakeContents` *)

(* :Author: Rolf Mertig
            Louis J. D'Andria *)

(* :Summary:
    This package defines functions for creating and 
    manipulating tables of contents and browser
    categories.
*)

(* :Copyright: *)

(* :Package Version: $Revision: 1.57 $ $Date: 2005/02/15 18:30:43 $*)

(* :Mathematica Version: 4.2 *)

(* :History:

*)

(* :Keywords:

*)

(* :Discussion:
    
*)

(* :Warning:
    
*)


BeginPackage["AuthorTools`MakeContents`",
  {"AuthorTools`Common`",
   "AuthorTools`MakeProject`"}];


ContentsFileName::usage = "ContentsFileName is an option to MakeContents that specifies the name of the generated contents file.";

CellTagPrefix::usage = "CellTagPrefix is an option for MakeContents which specifies the string that will start all cell tags added by MakeContents.";

MakeContents::usage = "MakeContents[nb|proj, format] writes a table of contents for the specified notebook or project in the given format."; 

MakeContentsNotebook::usage = "MakeContentsNotebook[nb|proj, format] returns a Notebook expression containing a table of contents for the specified notebook or project in the given format.";



Begin["`Private`"]; 


RawContents::usage = "RawContents[nbfile] returns a list of ContentsData expressions, one for each apopropriate cell in nbfile.  The ContentsData contains enough information to format a TOC.";


ContentsData::usage = "ContentsData is the head used in the internal representation of contents information.";


Options[RawContents] = 
{
  IncludeCellIndex -> True,
  IncludeCellPage -> False,
  FixCounters -> True
};



RawContents[nbfile_String, opts___?OptionQ] := 
Block[{pn, pd, pf},
  {pn, pd, pf} = {"Name", "Directory", "Files"} /.
    ProjectInformation[nbfile];
    
  RawContents[pn, pd, pf, opts]
] /; FileType[nbfile] === File



RawContents[pn_String, pd_String, {pf__String}, opts___?OptionQ] :=
Block[{nb, result, c=0},
  nb = ProgressDialog[$Resource["Contents", "Reading..."],"", {0, Length @ {pf}}];
  result =
  Map[
    (ProgressDialogSetSubcaption[nb, #];
     ProgressDialogSetValue[nb, ++c];
     RawContents[pn, pd, #, opts])&,
    {pf}
  ]//Flatten;
  ProgressDialogClose[nb];
  result
]


RawContents[pn_String, pd_String, pf_String, opts___?OptionQ] :=
Block[{stylePattern, ci, cp, styles, raw, nbfile, num, fc, pre},
  {ci, cp, fc, pre} = {IncludeCellIndex, IncludeCellPage, FixCounters, CellTagPrefix} /.
    Flatten[{opts, Options[RawContents], Options[MakeContents]}];
  {ci,cp,fc} = {TrueQ[ci], TrueQ[cp], TrueQ[fc]};
  styles = SelectedCellStyles /. Flatten[{opts, Options[MakeContents]}];
  
  stylePattern = Cell[___, Alternatives @@ styles, ___];
  nbfile = ToFileName[{pd}, pf];
  
  raw = ContentsData[
      "File" -> pf,
      "StyleIndex" -> First @ First @ Position[styles, #[[2]]],
      "CellTags" -> Flatten[Cases[Rest[#1], _[CellTags, x_]:>x]],
      "CellExpression" -> If[fc, fixCounters[pf, pre, #], #]
    ]& /@ NotebookLookup[nbfile, "CellExpression", stylePattern];
    
  If[raw === {}, Return[raw]];
  
  If[ci, raw = MapThread[Append[#1, "CellIndex" -> #2]&,
    {raw, NotebookLookup[nbfile, "CellIndex", stylePattern]}]
  ];

  If[cp,
    num = NotebookLookup[nbfile, "Numeration"];
    raw = MapThread[Join[#1, ContentsData["CellPage" -> #2, "Numeration" -> num]]&,
      {raw, NotebookLookup[nbfile, "CellPage", stylePattern]}]

  ];

  raw

]



(*
   The fixCounters utility replaces counters in the user's original
   content with counters that point back to the user's original
   content. This might not get the CounterFunction right, but the rest
   is ok.
*)

fixCounters[pf_String, pre_String, c:Cell[___,CellTags -> t_, ___]] :=
Block[{tag},
  tag = Select[Flatten[{t}], StringMatchQ[#, pre<>"*"]&,1];
  If[tag === {}, Return[c], tag = First @ tag];
  c //. {
    CounterBox[sty_] :> CounterBox[sty, {pf, tag}],
    CounterBox[sty_, x_String] :> CounterBox[sty, {pf, x}]
  }
]

fixCounters[_, _, x_] := x


(*
   MakeContents[nbfile, format, opts] constructs a path to
   the destination toc, creates the notebook expression by calling
   MakeContentsNotebook[nbfile, format, opts], then saves
   the resulting notebook, leaving it open.
*)


Options[MakeContents] = 
{
  ContentsFileName -> "TOC.nb",
  CellTagPrefix -> "c:",
  SelectedCellStyles -> {"Title", "Section", "Subsection", "Subsubsection"}
};


MakeContents[nb_NotebookObject, format_String, opts___] :=
Block[{nbfile, result},
  NotebookSaveWarning[nb, MakeContents];
  nbfile = NotebookFilePath[nb];
  result = MakeContents[nbfile, format, opts];
  SetSelectedNotebook[result]
] /; !ProjectDialogQ[nb]


MakeContents[nb_NotebookObject, format_String, opts___]:=
Block[{},
  AuthorTools`MakeProject`Private`ProjectDialogFunction["SaveWarning", nb];
  MakeContents[ProjectFileLocation[nb], format, opts]
]/; ProjectDialogQ[nb]



MakeContents[nbfile_String, format_String, opts___] :=
Block[{nb, tocname},
  tocname = ContentsFileName /. Flatten[{opts, Options[MakeContents]}];
  tocname = ToFileName[ProjectDirectory[nbfile], tocname];
  NotebookClose /@ Select[Notebooks[], NotebookFilePath[#] === tocname & ];
  
  Export[tocname, MakeContentsNotebook[nbfile, format, opts], "Notebook"];
  nb = NotebookOpen[tocname];
  NotebookSave[nb];
  nb
] /; FileType[nbfile] === File


MakeContents[arg_NotebookObject | arg_String, opts___?OptionQ] :=
  MakeContents[arg, "Simple", opts]


(**
  MakeContentsNotebook[_, "Expression", ___]
**)


MakeContentsNotebook[file_String, "Expression", opts___] :=
Block[{raw, pn},
  raw = RawContents[file,
    IncludeCellIndex -> False, IncludeCellPage -> False, opts];
  pn = ProjectName[file];
  
  Notebook[{
      Cell[pn <> $Resource["Contents", "List"], "ContentsTitle"],
      Cell[BoxData @ MakeBoxes[#, StandardForm]& @ raw, "Input"]
    },
    StyleDefinitions -> $AuthorToolsStyleDefinitions,
    ScreenStyleEnvironment -> "Brackets"
  ]
]



(**
  MakeContentsNotebook[_, "Simple", ___]
**)


MakeContentsNotebook[file_String, "Simple", opts___] :=
Block[{raw, pn},
  raw = RawContents[file,
    IncludeCellIndex -> False, IncludeCellPage -> False, opts];  
  pn = ProjectName[file];
  
  Notebook[Flatten[{
      Cell[pn <> $Resource["Contents", "Contents"], "ContentsTitle"],
      simpleCell /@ raw
    }],
    StyleDefinitions -> $AuthorToolsStyleDefinitions,
    ScreenStyleEnvironment -> "Brackets"
  ]
]


$OutlineStyles = {"Outline1", "Outline2", "Outline3", "Outline4"};

simpleCell[ContentsData[opts___]] :=
Block[{file, sty, exp},
  {file, sty, exp} = {"File", "StyleIndex", "CellExpression"} /. {opts};
  Flatten[{
    If[sty=!=1, {},
      Cell[TextData[ButtonBox[$Resource["Contents", "Open"] <> file,
        ButtonData->{FrontEnd`FileName[{},#]& @ file, None},
        ButtonStyle->"Hyperlink"]],"Text"]
    ],
  
  sty = If[MemberQ[{1,2,3,4}, sty], $OutlineStyles[[ sty ]], "Text"];
  
  Cell[First[exp], sty]
  }]
]



(**
  MakeContentsNotebook[_, "Book", ___]
  MakeContentsNotebook[_, "BookCondensed", ___]
  MakeContentsNotebook[_, "AddOnsBook", ___]
**)


(*
  These are each a little different, as they will require us to
  add cell tags to the notebook.

  "BookCondensed" is similar in look to "Book", but combines
  multiple consecutive subsections into a single cell, separated
  by bullets. Inspired by the toc in The Matheamtica Book.

  "AddOnsBook" is identical to the "Book" format, except the
  hyperlinks point to locations in the help browser instead of
  directly to other notebook files.
*)


MakeContentsNotebook[file_String,
    format:("Book" | "BookCondensed" | "AddOnsBook"), opts___] :=
Block[{raw, pn, pd, pf, nb, pre, max, prognb, c=0},
  
  {pn, pd, pf} = {"Name", "Directory", "Files"} /.
    ProjectInformation[file];
  
  prognb = ProgressDialog[
    $Resource["Contents", "Tagging..."], "", {0, Length @ Flatten @ {pf}}];
  RememberOpenNotebooks[];
  Scan[
    (ProgressDialogSetSubcaption[prognb, #];
     ProgressDialogSetValue[prognb, ++c];
     updateContentsCellTags[ToFileName[{pd}, #], opts])&,
    Flatten[{pf}]
  ];
  ProgressDialogClose[prognb];
  
  raw = RawContents[file,
    IncludeCellIndex -> True, IncludeCellPage -> True, opts];
  
  If[format === "AddOnsBook",
    Needs["AuthorTools`MakeIndex`"];
    raw = AuthorTools`MakeIndex`Private`addCategoriesToRawIndex[raw, pd];
  ];
  
  (* Now, every item in the index has a cell tag of the form pre<>k,
     and we can use that tag in the hyperlink. *)
  
  pre = CellTagPrefix /. Flatten[{opts, Options[MakeContents]}];
  tocCellTags =
    First[Select["CellTags" /. List @@ #, StringMatchQ[#, pre<>"*"]&]]& /@ raw;

  raw = MapThread[Append[#1, "tocCellTag" -> #2]&, {raw, tocCellTags}];
  
  (* If we need things condensed, gather consecutive bottom level
     entries into a sublist
  *)
  If[format === "BookCondensed",
    max = Max["StyleIndex" /. (List @@@ raw)];
    raw = Map[
      If[MatchQ[#, _[___,"StyleIndex" -> max, ___]], List[#], #]&,
      raw
    ];
    raw = raw //. {a___, b_List, c_List, d___} :> {a, Join[b,c], d};
  ];
  
  Notebook[Flatten[{
      Cell[pn <> $Resource["Contents", "Contents"], "ContentsTitle"],
      bookTOCCell[#, format]& /@ raw
    }],
    StyleDefinitions -> $AuthorToolsStyleDefinitions
  ]
];


(* 
   updateContentsCellTags checks every specified heading cell
   (SelectedCellStyles) for a unique cell tag starting with 'pre'
   (CellTagPrefix). If each heading cell has such a tag, this
   function does nothing. Otherwise, it removes any exisitng
   'pre' cell tags, and then tags every heading cell accordingly.
   
   AddCellTags is defined in Common.m
   
   Note: this function could be further improved by tagging only
   those heading cells that have no 'pre' tag, or no *unique* pre
   tag.
*)


updateContentsCellTags[nbfile_, opts___] :=
Block[{nb, pre, tags, sty, ind, outlines},

  {sty, pre} = {SelectedCellStyles, CellTagPrefix} /. 
    Flatten[{opts, Options @ MakeContents}];
  sty = Cell[___, Alternatives @@ sty, ___];
  
  (* If every heading cell already has a unique 'pre' tag, do nothing *)
  
  outlines = NotebookLookup[nbfile, "CellOutline", sty];
  tags = Flatten[Cases[#, _[CellTags, t_] :> t]]& /@ outlines;
  tags = DeleteCases[tags, x_String /; !StringMatchQ[x, pre <> "*"], {2}];
  If[Not[MemberQ[tags, {}]] &&
       Length[Union[First /@ tags]] === Length[outlines],
    Return[First /@ tags]
  ];
  
  (* otherwise, remove all existing 'pre' tags, and add the right ones *)

  ind = NotebookLookup[nbfile, "CellIndex", sty];

  nb = NotebookOpen[nbfile];
  
  RemoveCellTags[nb, StringJoin[pre, "*"]];
  tags = {StringJoin[pre, ToString @ #]}& /@ Range[Length[ind]];
  AddCellTags[nb, tags, ind];
  NotebookSave[nb];
  NotebookCloseIfNecessary[nb];
  tags
]



bookTOCCell[ContentsData[opts___], format_] := 
Block[{cellexpr, page, file, tag, style, bctag, cellsty, adbq, num},

  {cellexpr, page, file, tag, style, bctag, num} =
    {"CellExpression", "CellPage", "File", "tocCellTag", "StyleIndex", "BrowCatsTag", "Numeration"} /.
       {opts};
  
  file = FrontEnd`FileName[{},#]& @ file;
  
  style = Part[#, style]& @ If[format === "BookCondensed",
    {"TOCTitleCondensed", "TOCSectionCondensed",
     "TOCSubsectionCondensed", "TOCSubsubsectionCondensed"},
    {"TOCTitle", "TOCSection", "TOCSubsection","TOCSubsubsection"}
  ];
  
  (* If it's a TextData cell, whittle it down to just the contents.
     Otherwise, leave it as an inline cell
  *)
  cellexpr = Switch[cellexpr,
    Cell[x_String, ___], First @ cellexpr,
    Cell[TextData[x_],___], First @ First @ cellexpr,
    _, cellexpr];
  
  adbq = SameQ[format, "AddOnsBook"];
  
  Cell[TextData[
  Flatten[{
    cellexpr,
    " ",
    StyleBox["\t", "Leader"],
    Cell[TextData[
      If[adbq && ToString[bctag]==="None",
        PageString[page, num],
        ButtonBox[PageString[page, num],
          ButtonData -> If[adbq, {bctag,tag}, {file, tag}],
          ButtonStyle -> If[adbq, "AddOnsLinkText", "PageLink"]]
      ]], "TOCPage"]
    }]],
    style
  ]
]


bookTOCCell[{c__ContentsData}, format:("Book"|"AddOnsBook")] :=
  bookTOCCell[#, format]& /@ {c}


(* 
   Note we don't need page numbers for the condensed style, since the text 
   itself will be the content of the link
*)


bookTOCCell[{c__ContentsData}, "BookCondensed"] := 
Block[{lis={}, cellexpr, page, file, tag, style, cellsty},

  Do[
    {cellexpr, file, tag} =
      {"CellExpression", "File", "tocCellTag"} /.
      List @@ {c}[[i]];
    
    file = FrontEnd`FileName[{},#]& @ file;
    
    (* If it's a TextData cell, whittle it down to just the contents.
       Otherwise, leave it as an inline cell
    *)
    cellexpr = Switch[cellexpr,
      Cell[x_String, ___], First @ cellexpr,
      Cell[TextData[x_],___], First @ First @ cellexpr,
      _, cellexpr];
    
    lis = Flatten[{lis, 
      "\[Bullet] ",
      ButtonBox[cellexpr,
        ButtonData -> {file, tag},
        ButtonStyle -> "PageLink"],
      " "}]
    ,
    {i, 1, Length @ {c}}
  ];
 
 style = "StyleIndex" /. List @@ First[{c}];
 style = Part[#, style]& @
   {"TOCTitleCondensed", "TOCSectionCondensed",
    "TOCSubsectionCondensed", "TOCSubsubsectionCondensed"};
 
 Cell[TextData[Flatten[{lis}]], style]
]



(**
  MakeContentsNotebook[_, "Categories", ___]
**)

(*
   This format simply reads in the BrowserCategories.m file from
   the project's directory, and formats it as a notebook.

   Currently we use the whole BrowserCategory expression, but it
   might be worth considering only using those categories and
   items that refer to files actually in the current project.
*)

MakeContents::nobc = "There must be a BrowserCategories.m file in the directory `1` to do that.";

MakeContentsNotebook[file_String, "Categories", opts___] :=
Block[{pn, pd, pf, browCats, nb},
  
  {pn, pd, pf} = {"Name", "Directory", "Files"} /.
    ProjectInformation[file];
  
  browCats = ToFileName[{pd}, "BrowserCategories.m"];
  If[FileType[browCats] =!= File,
    MessageDisplay[MakeContents::nobc, pd];
    Abort[]
  ];
  browCats = Get[browCats];
  
  Notebook[Flatten[{
      Cell[pn <> $Resource["Contents", "Categories"], "ContentsTitle"],
      catsToCells[browCats, 1]
    }],
    StyleDefinitions -> $AuthorToolsStyleDefinitions
  ]
];


catsToCells[BrowserCategory[str_String, None, lis_List], d_]:=
Flatten[{
  Cell[str,"Outline" <> ToString[d]],
  catsToCells[#, d+1]& /@ lis
}]

catsToCells[Item[Delimiter, ___], d_] :=
  Cell["\[LongDash]", "Outline" <> ToString[d]]

catsToCells[Item[args___], d_] := catsToCells[Item[args, IndexTag ->
  First[{args}]], d] /; FreeQ[{args},IndexTag]

catsToCells[Item[args___], d_] := catsToCells[Item[args, CopyTag ->
  First[Cases[{args},_[IndexTag,x_]:>x]]], d] /; FreeQ[{args},CopyTag]

catsToCells[Item[str_String, file_, opts___], d_]:=
Block[{itag, ctag},
  {itag, ctag} = {IndexTag, CopyTag} /. {opts};
  Cell[TextData[{ButtonBox[str,
      ButtonData -> {file, ctag},
      ButtonStyle -> "AddOnsLinkText"]}],
    "Outline" <> ToString[d]
  ]
]

catsToCells[x_,___] := Cell[ToString[x,InputForm], "SmallText"]





End[]; 

EndPackage[]; 

