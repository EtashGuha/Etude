(* :Context: AuthorTools`MakeCategories` *)

(* :Author: Louis J. D'Andria *)

(* :Summary:
    
*)

(* :Copyright: *)

(* :Package Version: $Revision: 1.39 $, $Date: 2005/02/15 18:30:43 $ *)

(* :Mathematica Version: 5.0 *)

(* :History:
    This started as embedded code in MakeContents.m.
*)

(* :Keywords:
    document, notebook, formatting 
*)

(* :Discussion:
    
*)

(* :Warning:
    
*)



BeginPackage["AuthorTools`MakeCategories`",
  {"AuthorTools`Common`",
   "AuthorTools`MakeProject`"}]


CategoriesFileName::usage = "CategoriesFileName is an option to MakeBrowserCategories that specifies the name of the generated browser categories file.";

CopyTagPrefix::usage = "CopyTagPrefix is an option for MakeBrowserCategories which specifies the start of the string used in CopyTag and CellTags settings.";

IndexTagPrefix::usage = "IndexTagPrefix is an option for MakeBrowserCategories which specifies the start of the strings used in your items' IndexTag settings.  This tag needs to be unique among all other tags in the help browser, so this option should be set to something unique to your project. The default value of Automatic uses the project name.";

IndexTagFormats::usage = "IndexTagFormats is an option for MakeBrowserCategories which specifies what format or list of formats to use for IndexTag settings in the generated browser categories. Recognized formats include \"Counters\", \"CellContents\" and \"TaggingRules\".";

StartingCounterValues::usage = "StartingCounterValues is an option for MakeBrowserCategories which determines how the chapters, sections, etc. will be numbered.  The default {0,0,0,0} numbers everything starting at zero.  A setting of {1,0,1,0} for example will number the chapters starting at 1, sections starting at 0, subsections starting at 1, etc.";

MakeBrowserCategories::usage = "MakeBrowserCategories[nbfile, format] creates a browser categories file in the same directory as nbfile, and opens it in the front end.";

WriteBrowserCategories::usage = "WriteBrowserCategories[path, expr] writes the browser categories expression to a file with the specified full path.";

GetBrowserCategory::usage = "GetBrowserCategory[nb|proj, format] returns the BrowserCategory expression for the given notebook or project in the given format. Note that the resulting expression may contain references to CellTags that don't exist yet, and other information in TaggingRules that is only used by MakeBrowserCategories.";





Begin["`Private`"];


RawBrowserCategories::usage = "RawBrowserCategories[nb|proj] returns a list of CategoryData expressions that will be used to create the browser categories file for the specified notebook or project.";

CategoryData::usage = "CategoryData is the head used for the internal representation of browser category information before being written to a file.";



RawBrowserCategories[file_String, opts___] := 
Block[{pn, pd, pf},
  {pn, pd, pf} = {"Name", "Directory", "Files"} /.
    ProjectInformation[file];
    
  RawBrowserCategories[pn, pd, pf, opts]
] /; FileType[file] === File



RawBrowserCategories[pn_String, pd_String, {pf__String}, opts___] :=
Block[{nb, result, c=0},
  nb = ProgressDialog[$Resource["Categories", "Reading..."],"", {1, Length @ {pf}}];
  result =
  Map[
    (ProgressDialogSetSubcaption[nb, #];
     ProgressDialogSetValue[nb, ++c];
     RawBrowserCategories[pn, pd, #, opts])&,
    {pf}
  ]//Flatten;
  ProgressDialogClose[nb];
  result
]



RawBrowserCategories[pn_String, pd_String, pf_String, opts___] :=
Block[{nbfile, raw, styles, stylePattern, inds, struct},
  styles = SelectedCellStyles /. Flatten[{opts, Options[MakeBrowserCategories]}];
  stylePattern = Cell[___, Alternatives @@ styles, ___];
  nbfile = ToFileName[{pd}, pf];
  
  raw = MapThread[
    CategoryData[
      "File" -> pf,
      "CellIndex" -> #1,
      "StyleIndex" -> First @ First @ Position[styles, #2[[2]]],
      "CellTags" -> Flatten[Cases[#2, _[CellTags,x_]:>x, Infinity]],
      "CellContents" -> First[#2],
      "CellTaggingRules" -> Flatten[Cases[#2, _[TaggingRules,x_]:>x]]
    ]&,
    {NotebookLookup[nbfile, "CellIndex", stylePattern],
     NotebookLookup[nbfile, "CellExpression", stylePattern]}
  ];
  
  If[raw === {}, Return[raw]];
    
  struct = validStructure[raw];
  If[IntegerQ @ struct, Message[MakeBrowserCategories::valid, pf, struct]];
  
  inds = "CellIndex" /. List @@@ raw;
  inds = Flatten[{inds, 1 + Last[NotebookLookup[nbfile, "CellIndex"]]}];
  inds = Partition[inds, 2, 1];
  inds = # - {0, 1}& /@ inds;
  
  raw = MapThread[Append[#1, "IndexRange" -> #2]&, {raw, inds}]

]



(* 
   It would be a good idea to warn users when there are cells
   that will not appear anywhere in the BrowserCategory items
   because of the structure of their document. For example, if a
   particular notebook has as its first five cells:
   
   Title
   Text
   Text
   Text
   Section
   
   Then the Text cells will appear nowhere in the category display.

   Here's a function - validStructure - that checks whether the
   structure of the notebook is such that all cells will appear
   under some browser category item (True) or not (False)
   
   The algorithm is as follows:
   
   Once you encounter some non-heading cell within a heading, the 
   next cell should be either:
     (a) another non-heading cell,
     (b) another heading of the same level, or
     (c) a higher level heading to pop back out.
   
   If, on the other hand, you have a non-heading cell followed by
   a lower level heading, that's a problem. Essentially, you're
   asking that the heading which contains the non-headings and
   the lower level headings be used both as an item (to display
   the text) and a category (to display the lower level heading).
   It can't do both.
*)


validStructure[raw_] :=
Module[{lis, f, indices, lastHead, pat, n, val},
  indices = {"CellIndex", "StyleIndex"} /. List @@@ raw;
  (f[#1] = #2) & @@@ indices;
  f[_] = Infinity;
  lis = f /@ Range[First @ Last @ indices];
    
  lastHead = 0;
  pat = _;
  n = Do[
    val = lis[[i]];
    If[!MatchQ[val, pat], Break[i]];
    If[IntegerQ[val],
      pat = _;
      lastHead = val,
      pat = Alternatives @@ Flatten[{Range[lastHead], Infinity}]
    ],
    {i, 1, Length @ lis}
  ];

  (* If n is Null, then the Break never fired, and all is well *)
  n
]


MakeBrowserCategories::valid = "The cells in the notebook `1` do not follow strict outline format (first detected near cell `2`), so some cells will not be displayed in any browser category item. \!\(\*ButtonBox[\"Details\[Ellipsis]\",ButtonStyle->\"AddOnsLink\",ButtonFrame->\"None\",ButtonData->\"Note on Outline Format\"]\)";



Options[MakeBrowserCategories] = 
{
  CategoriesFileName -> "NewBrowserCategories.m",
  CopyTagPrefix -> "b:",
  IndexTagPrefix -> Automatic,
  IndexTagFormats -> {"Counters"},
  SelectedCellStyles -> {"Title", "Section", "Subsection"},
  StartingCounterValues -> {0,0,0,0}
};


MakeBrowserCategories[nb_NotebookObject, format_String, opts___] :=
Block[{nbfile, result},
  NotebookSaveWarning[nb, MakeBrowserCategories];
  nbfile = NotebookFilePath[nb];
  result = MakeBrowserCategories[nbfile, format, opts];
  SetSelectedNotebook[result]
] /; !ProjectDialogQ[nb]


MakeBrowserCategories[nb_NotebookObject, format_String, opts___]:=
Block[{},
  AuthorTools`MakeProject`Private`ProjectDialogFunction["SaveWarning", nb];
  MakeBrowserCategories[ProjectFileLocation[nb], format, opts]
]/; ProjectDialogQ[nb]


MakeBrowserCategories[nbfile_String, format_String, opts___] :=
Block[{browCats, pd, file},
  browCats = GetBrowserCategory[nbfile, format, opts];
  pd = ProjectDirectory[nbfile];

  tagNotebooksForBrowCats[
    pd,
    Cases[browCats, _[TaggingRules, x_] :> x, Infinity],
    opts
  ];
  
  browCats = DeleteCases[browCats, _[TaggingRules, _], Infinity];
  
  file = CategoriesFileName /. Flatten[{opts, Options[MakeBrowserCategories]}]; 
  file = ToFileName[{pd}, file];
  WriteBrowserCategories[file, browCats];
  NotebookOpen[file]
] /; FileType[nbfile] === File


MakeBrowserCategories[arg_NotebookObject | arg_String, opts___?OptionQ] :=
  MakeBrowserCategories[arg, "Simple", opts]



tagNotebooksForBrowCats[_, {}, opts___] := {};

tagNotebooksForBrowCats[pd_, lis_, opts___] := 
Block[{nb, files, c=0},
  files = Union[First /@ lis];
  nb = ProgressDialog[$Resource["Categories", "Tagging..."], "", {1, Length @ files}];
  RememberOpenNotebooks[];
  Scan[
    (ProgressDialogSetSubcaption[nb, #];
     ProgressDialogSetValue[nb, ++c];
     tagNotebookForBrowCats[pd, #, Cases[lis, {#,___}], opts])&,
    files
  ];
  ProgressDialogClose[nb];
  Null;
]


(*
   In tagNotebookForBrowCats below, the tags only get added to
   the notebook if they need to. If all the tags are already in
   place, nothing is done.
   
   Note: this function could be improved by tagging only  those
   individual cells that need to be tagged, instead of retagging
   a whole notebook to pick up potentially just a few cells, but
   that requires a lot more cleverness with adding and removing
   cell tags.
*)

tagNotebookForBrowCats[pd_String, pf_String, lis_, opts___] :=
Block[{nbfile, nb, pre, tags, needsTagsQ},
  pre = CopyTagPrefix /.
    Flatten[{opts, Options[MakeBrowserCategories]}];
  nbfile = ToFileName[{pd}, pf];
  
  (* If all the cell tags already exist, don't do anything *)
  
  tags = Flatten[Cases[#, _[CellTags,t_]:>t]]& /@
    NotebookLookup[nbfile, "CellOutline"];
  needsTagsQ = False;
  Scan[
    If[Count[Take[tags, #[[3]] ], #[[2]], {2}] < Length[ Range @@ #[[3]] ],
      needsTagsQ = True; Return[], Null
    ]&,
    lis
  ];
  If[Not @ needsTagsQ, Return[]];
  
  (* otherwise, set the necessary cell tags *)
  
  nb = NotebookOpen[nbfile];
  RemoveCellTags[nb, StringJoin[pre, "*"]];
  Scan[
    AddCellTags[nb, {#[[2]]}, Range @@ #[[3]]]&,
    lis
  ];
  NotebookSave[nb];
  NotebookCloseIfNecessary[nb]
]



(**
  GetBrowserCategory[_, "Simple", ___]
**)


GetBrowserCategory[nbfile_, "Simple", opts___] :=
Block[{pn, pf},
  {pn, pf} = {"Name", "Files"} /. ProjectInformation[nbfile];  
  BrowserCategory[pn, None,
    Item[#, #, CopyTag -> None]& /@ pf
  ]
]



(**
  GetBrowserCategory[_, "Full", ___]
**)


(*
   First, we read all the relevant category information using
   RawBrowserCategories.  Then, we can build the browser
   categories with the appropriate cell tags, and add the cell
   tags to the appropriate notebooks either at the same time or
   one after the other.
*)


addBCTagToRawBrowserCategories[lis:{__CategoryData}, startValues_]:=
Block[{counters, incrementCounter, m},

  m = Length @ startValues;
  counters = Table[-1, {m}];

  incrementCounter[n_] := counters = Flatten[{
    Take[counters, n - 1],
    If[counters[[n]] === -1, startValues[[n]], 1 + counters[[n]]],
    Table[-1, {m - n}]
  }];
  
  styleIndex = "StyleIndex" /. List @@@ lis;
  
  MapThread[
    Append[#1, "BCTag" -> StringJoin[#2]]&,
    {lis, 
     BoxForm`Intercalate[ToString /@ #, "."]& /@ 
       DeleteCases[incrementCounter /@ styleIndex, -1, Infinity]
    }
  ]
];



(*
   People might have weird things in their heading cells, but
   only strings can be displayed in the category and item
   listings. The following tries to clean things up.
*)

cellContentsToString[Cell[expr_,___]] := cellContentsToString[expr];
cellContentsToString[TextData[expr_]] := cellContentsToString[expr];
cellContentsToString[List[exprs___]] := StringJoin[cellContentsToString /@ {exprs}];
cellContentsToString[StyleBox[expr_,___]] := cellContentsToString[expr];
cellContentsToString[str_String] :=
  StringReplace[str, {"\t"->" ", "\n" -> " ", "\r" -> " ", "\[IndentingNewLine]" -> " "}];
cellContentsToString[expr_] := ToString[expr, InputForm];


GetBrowserCategory[nbfile_String, "Full", opts___] :=
Block[{raw, bc, sty, cc, ii, pre1, startvals, pre2, pn, cellstr, tagfmt},

  {pre1,startvals,pre2,tagfmt} =
    {CopyTagPrefix, StartingCounterValues, IndexTagPrefix, IndexTagFormats} /.
      Flatten[{opts, Options[MakeBrowserCategories]}];
  
  pn = ProjectName @ nbfile;
  If[pre2 === Automatic, pre2 = pn];
  
  raw = RawBrowserCategories[nbfile, opts];
  If[raw === {}, Return @ BrowserCategory[pn, None, {}]];
  raw = addBCTagToRawBrowserCategories[raw, startvals];
  
  sty = "StyleIndex" /. List @@@ raw;
  sty = Append[sty, Last[sty]];
  sty = Partition[sty, 2, 1];
  
  (* 
     The following builds the proper hierarchy of categories and
     items by first using dummy heads - cc and ii for BrowserCategory
     and Item resp, then nesting those heads according to what cell
     style the refer to, and finally, building the actual Item and
     BrowserCategory expressions based on the CategoryData's.
     
     The TaggingRules are used when cell tags need to be added to
     the source notebooks.
  *)
  
  bc = 
  MapThread[
    If[Less @@ #1, cc[First @ #1, #2, {}], ii[First @ #1, #2]]&,
    {sty, raw}
  ];
  
  bc = bc //.  
    {x___, cc[a_, b_, {c___}], d_[e_, f___], y___} :> 
      {x, cc[a, b, {c, d[e, f]}], y} /; a < e;
  
  bc = bc //. {
    
    cc[a_, CategoryData[b___], c_] :> Sequence @@ Flatten[{
      If[delimiterAboveQ[b], Item[Delimiter], {}],
      BrowserCategory[
        cellContentsToString["CellContents" /. {b}],
        None,
        c],
      If[delimiterBelowQ[b], Item[Delimiter], {}]
    }],
        
    ii[a_,CategoryData[b___]] :> Sequence @@ Flatten[{
      If[delimiterAboveQ[b], Item[Delimiter], {}],
      Item[
        cellstr = cellContentsToString["CellContents" /. {b}],
        "File" /. {b},
        CopyTag -> StringJoin[pre1, "BCTag" /. {b}],
        IndexTag -> indexTagSetting[tagfmt, pre2, cellstr, b],
        TaggingRules -> {
          "File" /. {b},
          StringJoin[pre1, "BCTag" /. {b}],
          "IndexRange" /. {b}
        }],
      If[delimiterBelowQ[b], Item[Delimiter], {}]
    }]
  };
  
  BrowserCategory[pn, None, bc]
]


delimiterAboveQ[___,"CellTags" -> {___, "DelimiterAbove", ___}, ___] = True;
delimiterAboveQ[___] = False;
delimiterBelowQ[___, "CellTags" -> {___, "DelimiterBelow", ___}, ___] = True;
delimiterBelowQ[___] = False;


indexTagSetting[tagfmt_, prefix_, cellstr_, opts___] :=
Block[{f, lis},
  
  f["Counters"] := StringJoin[prefix, "BCTag" /. {opts}];
  f["CellContents"] := StringJoin[prefix, cellstr];
  f["TaggingRules"] := Block[{usertags},
    usertags = Flatten[Cases["CellTaggingRules" /. {opts} , _["IndexTag", t_] :> t]];
    If[MatchQ[usertags, {__String}], StringJoin[prefix, #]& /@ usertags, {}]
  ];
  
  lis = Flatten[{tagfmt}];
  If[lis === {All}, lis = {"Counters", "CellContents", "TaggingRules"}];
  lis = Cases[lis, "Counters" | "CellContents" | "TaggingRules"];
  
  lis = Flatten[{f /@ lis}];
  
  Switch[lis,
    {}, f["Counters"],
    {_}, First[lis],
    _, lis
  ]
]




(**
  GetBrowserCategory[_, "FullNoTags", ___]
**)

(*
  The cell tags are only added to the notebook for every Item in
  the BrowserCategory returned by
  GetBrowserCategory[_,"Full",___] which contains a TaggingRules
  setting.  Thus, removing the TagginRules settings will prevent
  the notebooks from being tagged.
*)


GetBrowserCategory[nbfile_String, "FullNoTags", opts___] :=
DeleteCases[
  GetBrowserCategory[nbfile, "Full", opts],
  _[TaggingRules, _],
  Infinity
]






(* *********** WriteBrowserCategories **************** *)


blanks[n_] := blanks[n] = StringJoin[Table[" ", {n}]];

recursiveWriteBC[st_, BrowserCategory[str_, x_:"", {}], d_:0] :=
  (
    WriteString[st, blanks[d]];
    WriteString[st, "BrowserCategory[", toStr[str], ", "];
    If[x=!="", WriteString[st, quoteIfString @ x, ", "]];
    WriteString[st, "{ }]"];
  );

recursiveWriteBC[st_, BrowserCategory[str_, x_:"", lis_List], d_:0] :=
  (
    WriteString[st, blanks[d]];
    WriteString[st, "BrowserCategory[", toStr[str], ", "];
    If[x=!="", WriteString[st, quoteIfString @ x, ", "]];
    WriteString[st, "\n", blanks[d+2], "{\n"];
    (recursiveWriteBC[st, #, d+4]; WriteString[st, ",\n"])& /@ Most[lis];
    recursiveWriteBC[st, Last[lis], d+4];
    WriteString[st, "\n", blanks[d+2], "}\n", blanks[d], "]"];
  );

recursiveWriteBC[st_, itm_Item, d_] :=
    WriteString[st, blanks[d], toStr[itm]];

(* Also write out HelpDirectoryListings, etc. *)
recursiveWriteBC[st_, other_, d_] :=
    WriteString[st, blanks[d], toStr[other]];


WriteBrowserCategories[path_, expr_] :=
Block[{st},
  st = OpenWrite[path, PageWidth -> Infinity];
  recursiveWriteBC[st, expr];
  Close[st];
  path
]

quoteIfString[x_String] := "\"" <> x <> "\"";
quoteIfString[x_] := x;
toStr[x_] := ToString[x, InputForm, CharacterEncoding -> "ASCII"]


End[]; 

EndPackage[]; 
