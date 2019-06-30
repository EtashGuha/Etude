(* :Context: AuthorTools`Experimental` *)

(* :Author: Louis J. D'Andria *)

(* :Summary: *)

(* :Copyright: Copyright 2003, Louis J. D'Andria *)

(* :Package Version: Beta $Revision: 1.71 $ $Date: 2013/12/10 19:04:24 $ *)

(* :Mathematica Version: 5.0 *)

(* :History: *)

(* :Keywords: *)

(* :Discussion: *)

(* :Warning:
      These functions are likely to change in the next version.
*)

(* :Limitations:
      NotebookSearch uses Find to scan notebook files for content.
      Searching content that is broken across more than one line is
      not directly possible, though in such cases you can do a
      conjuctive search.
      
      Searches that return links to the Help Browser will discard
      information in notebooks that are not in any browser item, or in
      notebooks that are in the Help Browser but in a location
      inaccessible to direct linking, or in notebooks that are in the
      Help Browser but which are overlooked by a bug in this package.

*)


(* :Future:
      Add easy ability to limit a search to the notebooks / items 
      found by a previous search.
      
      Optional pre-indexing for speed.
      
      Implement full boolean searches (and, or, *not*, other?)
      
      Add ability to sort by date, for matches in recently edited files.
      
      Add ability to do partial searches (return first k matches, k+1 - 2k, etc).
      
      Add abililty to show matches as they are found instead of waiting 'til the end.
      
      Add abort button to progress dialog.
      
      Add size (string or number) to output.
      
      Request to search .m files too.
      
      Add tokens for collections of cell styles (headings, body text, graphics, etc).
      
      Add tokens for collections of help notebooks from an application (HelpNotebooks["JLink"]),
      or any user-specified category (Built-in Functions > Numerical Computation)
      
      Emit a warning when the user does a HelpBrowser* search with no
      help browser notebooks in the search space.
*)


(* :Examples:

$DefaultSearchFormat = "Expressions"

NotebookSearch[]

NotebookSearch["AbortProtect"]

NotebookSearch[$TopDirectory, "AbortProtect"]

NotebookSearch[HelpNotebooks[], "AbortProtect"]

NotebookSearch[HelpNotebooks["MainBook"], "Abort"]

NotebookSearch[Rest @ Notebooks[], "Abort"]

NotebookSearch[{dir1, dir2, dir3, nbfile1, nbfile2}, "Abort"]

NotebookSearch[HelpNotebooks["MainBook"], {"AbortProtect", "CheckAbort"}]

NotebookSearch[HelpNotebooks["MainBook"], {"AbortProtect", "CheckAbort"},
  SelectedCellStyles -> {"Text", "MathCaption"}]

NotebookSearch[HelpNotebooks["MainBook"], {"AbortProtect", "CheckAbort"},
  ExcludedCellStyles -> {"Text", "MathCaption"}]

NotebookSearch[HelpNotebooks["MainBook"], {"AbortProtect", "CheckAbort"},
  MultiWordSearch -> And]

--

$NotebookSearchFormats

NotebookSearch[HelpNotebooks[], "Polya", "ShortOutput"]
NotebookSearch[HelpNotebooks[], "Polya", "Output"]
NotebookSearch[HelpNotebooks[], "Polya", "HelpBrowserOutput"]
NotebookSearch[HelpNotebooks[], "Polya", "LongOutput"]

NotebookSearch[HelpNotebooks["MainBook"], "Abort", "CellLinks"]
NotebookSearch[HelpNotebooks["MainBook"], "Abort", "NotebookLinks"]
NotebookSearch[HelpNotebooks["MainBook"], "Abort", "HelpBrowserLinks"]
NotebookSearch[HelpNotebooks["MainBook"], "Abort", "HelpBrowserCellLink"]
NotebookSearch[HelpNotebooks["MainBook"], "Abort", "HelpBrowserCellLinks"]
NotebookSearch[HelpNotebooks["MainBook"], "Abort", "HelpBrowserExpressions"]

--

$DefaultSearchFormat = "HelpBrowserExpressions"

RebuildBrowserLookupTable[]

NotebookSearch["Beethoven", SelectedCellStyles -> {"Picture"}]

NotebookSearch[HelpNotebooks["RefGuide", "MasterIndex"], "Integrate"]

NotebookSearch[HelpNotebooks["MasterIndex"], "Plot",
  SortByHitCount -> False,
  IgnoreCase -> True,
  Verbosity -> 6,
  HighlightSearchStrings -> False]

NotebookSearch["Plot" && "Compiled", SelectedItems -> {
  {"Help Browser", "The Mathematica Book", "Principles of Mathematica", ___},
  {"Help Browser", "Built-in Functions", "Graphics and Sound", "3D Plots", ___}},
  Verbosity -> 0,
  CategorizeResults -> False]

(NotebookSearch[HelpNotebooks[], "Plot", "ShortOutput"];
NotebookSearch[HelpNotebooks[], "Compiled", "NotebookLinks", SearchInResults -> True])

*)



BeginPackage["AuthorTools`Experimental`", {"AuthorTools`Common`"}]


NotebookSearch::usage = "NotebookSearch[str] searches the notebooks from the Help Browser and opens a new notebook with links to any cell containing the given string pattern. NotebookSearch[list, str] searches the given list of notebook objects, files, and directories. NotebookSearch[list, str, format] returns the results in the given format.";

MultiWordSearch::usage = "MultiWordSearch is an option for NotebookSearch that determines whether to return cells that contain any (Or) or all (And) of the given search strings. Default: Or";

$NotebookSearchFormats::usage = "$NotebookSearchFormats is a list containing all formats available to NotebookSearch.";

$DefaultSearchFormat::usage = "$DefaultSearchFormat is the default format generated by NotebookSearch.";

$DefaultSearchNotebooks::usage = "$DefaultSearchNotebooks is the default list of notebooks scanned by NotebookSearch.";

HelpNotebooks::usage = "HelpNotebooks[category1, category2, ...] returns a list of all notebooks used in the given Help Browser categories. HelpNotebooks[] lists notebooks in all categories.";

$HelpCategories::usage = "$HelpCategories is a list containing the names of the top-level help categories.";

RebuildBrowserLookupTable::usage = "RebuildBrowserLookupTable[] refreshes the value of $BrowserLookupTable based on the currently available browser category files.";

GetBrowserLookupTable::usage = "Read the browser item cache if one exists, or run RebuildBrowserLookupTable[] if not.";

$BrowserLookupTable::usage = "A cache of every item in every browser configuration file visible to the current Help Browser.";

ExcludedCellStyles::usage = "ExcludedCellStyles is an option to NotebookSearch. If SelectedCellStyles is set to a list of strings, this option's setting is ignored.";

SortByHitCount::usage = "SortByHitCount is an option to NotebookSearch which indicates how the resutls should be sorted.";

Verbosity::usage = "Verbosity is an option to NotebookSearch that can be set to a positive integer, indicating how much speed to trade for progress messages.";

$BrowserCacheFile::usage = "$BrowserCacheFile is the file path used to store Help Browser information in a format that can be read quickly into the kernel in subsequent sessions with NotebookSearch.";

SearchInResults::usage = "SearchInResults is an option to NotebookSearch which confines the given search space to the notebooks or browser items found by the immediately preceeding search.";

$CacheOffsetsQ::usage = "$CacheOffsetsQ determines whether to cache byte offset information for notebook searches run in the current kernel session.";

SelectedItems::usage = "SelectedItems is an option to NotebookSearch that can be set to All or a list of patterns indicating Help Browser locations to search.";

ShowResultsInBrowser::usage = "ShowResultsInBrowser is an option to NotebookSearch that indicates whether to open a new notebook with the results, or open the results as a help browser item.";

CategorizeResults::usage = "CategorizeResults is an option to NotebookSearch that indicates whether to present Help Browser search results grouped by category or as a flat list.";

HighlightSearchStrings::usage = "HighlightSearchStrings is an option to NotebookSearch that indicates whether to reformat cell expressions with the search strings highlighted.";


ItemLookup::usage = "ItemLookup[str] displays links to those Help Browser items whose item name contains the given string. ItemLookup[str, True] also looks in category names.";

ItemLookupCategories::usage = "ItemLookupCategories is an option to ItemLookup that determines which top-level categories are scanned.";

PartialMatch::usage = "PartialMatch is an option to ItemLookup that determines whether to look inside item names for the given string.";


InstallSearchMenus::usage = "InstallSearchMenus[] adds menu items to the Help menu via $UserBaseDirectory/Autoload/NotebookSearchMenus/FrontEnd/init.m";

UninstallSearchMenus::usage = "UninstallSearchMenus[] deletes the file $UserBaseDirectory/Autoload/NotebookSearchMenus/FrontEnd/init.m";


Begin["`Private`"]


(* shield the $ package globals from package reloading *)

If[Not @ ListQ @ $NotebookSearchFormats,
  $NotebookSearchFormats =
    {"CellLinks", "NotebookLinks", "Expressions", "Output", "ShortOutput", "LongOutput",
     "HelpBrowserLinks", "HelpBrowserCellLinks", "HelpBrowserOutput",
     "HelpBrowserLongOutput", "HelpBrowserExpressions", "HelpBrowserCellLink",
     "HelpBrowserSmallLink"}
]


If[Not @ MemberQ[$NotebookSearchFormats, $DefaultSearchFormat],
  $DefaultSearchFormat = If[$ParentLink === Null, "Output", "HelpBrowserExpressions"]
]



If[Not @ ListQ @ $DefaultSearchNotebooks,
  $DefaultSearchNotebooks := HelpNotebooks[]
]




$HelpCategories = {
  "RefGuide",
  "AddOns",
  "MainBook",
  "OtherInformation",
  "GettingStarted",
  "Tour",
  "Demos",
  "MasterIndex"
}



beginProgress[str_] := 
If[$Verbosity > 0,
  If[$Notebooks,
    $ProgressNotebook = ProgressDialog["Searching for " <> str, "", Indeterminate],
    Print["Searching for " <> str <> "..."]
  ]
]


endProgress[] :=
If[$Verbosity > 0 && $Notebooks,
  ProgressDialogClose[$ProgressNotebook]
]


(* Only issue level 5 messages if a time threshhold is exceeded
   for $Verbosity === 5. Always issue them if $Verbosity > 5. *)

$lastUpdate = AbsoluteTime[];
$updateInterval = 0.4;

showProgress[5][args___] :=
If[AbsoluteTime[] > $lastUpdate + $updateInterval,
  $lastUpdate = AbsoluteTime[];
  If[MatchQ[{args}, {_List, _}],
    progressUpdateFunction[ ToString @ (StringForm @@ {args}[[1]]), {args}[[2]] ],
    progressUpdateFunction @ ToString @ StringForm[args]
  ]
] /; $Verbosity === 5


showProgress[n_Integer][str_String, args___] :=
If[$Verbosity >= n,
  progressUpdateFunction @ ToString @ StringForm[str, args]
]

showProgress[n_Integer][{str_String, args___}, n_] :=
If[$Verbosity >= n,
  progressUpdateFunction[ ToString @ StringForm[str, args], n]
]
 

progressUpdateFunction[str_String] := 
If[$Notebooks,
  ProgressDialogSetSubcaption[$ProgressNotebook, str];
  AuthorTools`Common`Private`ProgressDialogSetRange[$ProgressNotebook, Indeterminate],
  Print[str]
]


progressUpdateFunction[str_String, n_] := 
If[$Notebooks,
  ProgressDialogSetSubcaption[$ProgressNotebook, str];
  ProgressDialogSetValue[$ProgressNotebook, n];
  ProgressDialogSetRange[$ProgressNotebook, {0, 1}],
  Print[str]
]






(*
   The progress meter runs quite a bit faster in the status line of a
   notebook than in the progress dialog. If when we have a palette
   interface, or even if not, we should consider these:

setNotebookStatus[nbObj_, str_] :=
  MathLink`CallFrontEnd[FrontEnd`SetNotebookStatusLine[nbObj, str]]

beginProgress[] := ($statusNotebook = ButtonNotebook[])
endProgress[] := setNotebookStatus[$statusNotebook, ""]
progressUpdateFunction[str_] := setNotebookStatus[$statusNotebook, str]
*)




NotebookSearch::vscan = "Searching `1` notebook files"
NotebookSearch::vcond = "Condensing `1` hits";
NotebookSearch::vform = "Formatting"



Options[NotebookSearch] =
{
  IgnoreCase -> False,
  WordSearch -> False,
  MultiWordSearch -> Or,
  SelectedCellStyles -> All,
  ExcludedCellStyles -> {},
  SortByHitCount -> True,
  SearchInResults -> False,
  Verbosity -> 5,
  SelectedItems -> All,
  ShowResultsInBrowser -> False,
  CategorizeResults -> True,
  HighlightSearchStrings -> True
};


Options[rawSearch] =
{
  IncludeIndex -> False,
  IncludeCount -> True,
  IncludeOffsets -> False,
  MaxRawHits -> 20000
};


Options[nbSearch] =
{
  IncludeCount -> True,
  IncludeIndex -> False,
  IncludeOutline -> False,
  IncludeExpression -> False
};





(*

Division of labor:

NotebookSearch calls nbSearch, which in turn calls rawSearch.

rawSearch returns information based on raw hits from Find or FindList.
No attempt is made to figure out if the hits were in an actual cell,
in the content of a cell vs its options, etc.

nbSearch takes the result of rawSearch and filters it. It handles
boolean searches, it condenses duplicate entries, it discards matches
outside of actual notebook content.

*)


rawSearch[{}, ___] := {}

rawSearch[{nbFiles__String}, {strs__String}, opts___] :=
Block[{nbFile, indexQ, countQ, offsetsQ, index, count, offsets, lis, st,
    findOpts, total, n, max, messQ=True},


  {indexQ, countQ, offsetsQ, max} = {IncludeIndex, IncludeCount, IncludeOffsets, MaxRawHits} /.
      Flatten[{opts, Options[rawSearch], Options[NotebookSearch]}];
  
  {indexQ, countQ, offsetQ} = TrueQ /@ {indexQ, countQ, offsetsQ};  
  
  findOpts = FilterRules[
    {opts, Sequence @@ Options[NotebookSearch], RecordSeparators -> {"\n", "\r"}},
    Options[Find]
  ];
  
  total = 0;
  n = Length[{nbFiles}];
  
  Table[
  nbFile = Part[{nbFiles},i];  
  
  Which[
    total > max,
    If[messQ, Message[NotebookSearch::max1]; messQ = False];
    count = 0
    ,
    showProgress[5][{"`1`% `2` -- `3` candidate lines found", 
      Round[100 i/n], $progstring, total}, i/n];
    $CacheOffsetsQ && (offsetsQ || indexQ) && cacheExists[nbFile, {strs}, opts],
    {offsets, index} = rawCache[nbFile, {strs}, opts];
    count = Length[offsets]
    ,
    offsetsQ || indexQ,
    offsets = {}; index = {};
    st = OpenRead[nbFile];
    If[st === $Failed,
      offsets = {},
      While[
        Find[st,{strs}, findOpts] =!= EndOfFile,
        offsets = {offsets, StreamPosition[st]}
      ];
      Close[st]
    ];
    offsets = Flatten[offsets];
    count = Length[offsets];
    If[indexQ,
      index = byteOffsetsToCellIndexes[{nbFile, offsets}];
      
      (* There might be room for an efficiency tweak here, to read all
      the relevant cell outlines when the byte offsets are being
      calculated -- since the cell outlines are available to
      byteoffsetsToCellIndexes. Look at this later *)
    ];
    If[$CacheOffsetsQ, setRawCache[{nbFile, {strs}, opts}, {offsets, index}]];
    ,    
    True,
    (* if you don't need offsets or indexes, just use FindList *)
    lis = FindList[nbFile, {strs}, findOpts];
    If[lis === $Failed, lis = {}];
    count = Length[lis];
  ];
  
  total+=count;
  
  If[count === 0,
    Unevaluated[Sequence[]],
    Flatten[{
      nbFile,
      If[countQ, "Count" -> count, {}],
      If[offsetsQ, "Offsets" -> offsets, {}],
      If[indexQ, "Index" -> index, {}]
    }]
  ]
  ,
  {i, 1, Length[{nbFiles}]}]
]


$CacheOffsetsQ = True

cacheExists[keys__] := ListQ @ rawCache[keys]

setRawCache[{keys__}, data_] := Set[rawCache[keys], data]

clearRawCache[] := Clear[rawCache]



nbSearch[{}, ___] := {}

nbSearch[{nbFiles__String}, {strs__String}, opts___]:=
Block[{indexQ, countQ, outlineQ, exprQ, styleQ,
        multi, styles, pat, xstyles, xstyleQ, xpat,
        index, outlines, findOpts, raw, rawFiles,
        nbFile, total, c},
  
  {multi, styles, xstyles} = {MultiWordSearch, SelectedCellStyles, ExcludedCellStyles} /. 
    Flatten[{opts, Options[nbSearch], Options[NotebookSearch]}];
  
  {indexQ, countQ, outlineQ, exprQ} =
    {IncludeIndex, IncludeCount, IncludeOutline, IncludeExpression} /.
      Flatten[{opts, Options[nbSearch], Options[NotebookSearch]}];
  
  {indexQ, countQ, outlineQ, exprQ} = TrueQ /@ 
    {indexQ, countQ, outlineQ, exprQ};
  
  styles = Flatten[{styles}];
  If[MatchQ[styles, {__String}],
    styleQ = True; pat = Alternatives @@ styles,
    styleQ = False; pat = _
  ];
  
  xstyles = Flatten[{xstyles}];
  If[MatchQ[xstyles, {__String}],
    xstyleQ = True; xpat = Alternatives @@ xstyles,
    xstyleQ = False; xpat = _
  ];
  
  
  findOpts = FilterRules[{opts}, Options[Find]];
  
  showProgress[1][NotebookSearch::vscan, Length[{nbFiles}]];
  
  If[
    (multi === And || multi === "NotebookAnd") && Length[{strs}] > 1,
    (* And: do each search separately and intersect the results *)
    (* Note: this is not the most efficient implementation, but it works for now *)
    raw = {};
    rawFiles = {nbFiles};
    Do[
      (* dynamically reduce the search space for each subsequent string *)
      $progstring = ToString @ StringForm["for `1` of `2` words", i, Length[{strs}]];
      AppendTo[raw, rawSearch[rawFiles, {Part[{strs}, i]}, IncludeIndex -> True, findOpts]];
      rawFiles = Intersection[rawFiles, First /@ Last[raw]],
      {i, Length @ {strs}}
    ];
    total = Total[Cases[raw, Rule["Count", c_Integer] :> c, Infinity]];
    rawFiles = Intersection @@ Map[First, raw, {2}];
    raw = Apply[If[multi === "NotebookAnd", Union, Intersection],
      Cases[raw, {#, ___, "Index" -> lis_List, ___}:>lis, {2}]]& /@ rawFiles;
    ,
    (* Or: lump everything together *)
    $progstring = "complete";
    raw = rawSearch[{nbFiles}, {strs}, IncludeIndex -> True, findOpts];    
    total = Total[Cases[raw, Rule["Count", c_Integer] :> c, Infinity]];
    rawFiles = Map[First, raw];
    raw = "Index" /. Map[Rest, raw];
  ];

  (* remove duplicates, and matches before (0) or after (Infinity) notebook content *)
  raw = DeleteCases[Union /@ raw, 0 | Infinity, Infinity];

  showProgress[1][NotebookSearch::vcond, total];
  
  
  c=0;
  
  Table[
  nbFile = Part[rawFiles, i];
  index = Part[raw, i];
  
  showProgress[5]["`1` candidate lines -- collecting cells: `2`", total, c];
  
  If[styleQ || xstyleQ || outlineQ || exprQ,
    outlines = NotebookLookup[nbFile, "CellOutline"];
    Which[
      styleQ, (* including styles *)
      (* when searching limited styles, reduce the index and outline list together *)
      tmp = Cases[Transpose @ {index, Part[outlines, index]}, {_, Cell[___, pat, ___]}];
      {index, outlines} = If[tmp === {}, {{}, {}}, Transpose @ tmp],
      
      xstyleQ, (* excluding styles *)
      (* again, reduce the index and outline list together *)
      tmp = DeleteCases[Transpose @ {index, Part[outlines, index]}, {_, Cell[___, xpat, ___]}];
      {index, outlines} = If[tmp === {}, {{}, {}}, Transpose @ tmp],
      
      True, (* otherwise *)
      (* reduce the outline list based on the index list *)
      outlines = Part[outlines, index]
    ];
  ];
  
  c+=Length[index];
  
  If[index === {},
    Unevaluated[Sequence[]],
    Flatten[{
      nbFile,
      If[countQ, "Count" -> Length[index], {}],
      If[indexQ, "Index" -> index, {}],
      If[outlineQ, "Outline" -> outlines, {}],
      If[exprQ, "Expression" -> ExtractCells[nbFile, outlines], {}]
    }]
  ]
  ,
  {i, Length @ rawFiles}]
  
];



byteOffsetsToCellIndexes[{nbFile_String, {}}] := {}

byteOffsetsToCellIndexes[{nbFile_String, lis_List}] :=
Module[{cellOutlines, n, st, nbOptLoc, res = {}},
    
  cellOutlines = NotebookLookup[nbFile, "CellOutline"];
  n = Length[cellOutlines];
  
  st = OpenRead[nbFile];
  lf = AuthorTools`Common`Private`eol[st];
  nbOptLoc = First @ AuthorTools`Common`Private`CacheLocations @ st;
  Close[st];
  
  cellOutlines = If[
    StringLength[lf] > 1,
    Plus @@@ cellOutlines[[All, {1, 2}]],
    First /@ cellOutlines
  ];
  AppendTo[cellOutlines, nbOptLoc]; (* the notebook content ends at nbOptLoc *)
  
  (* The fastest method for calculating positions used to be procedural: *)
  (*
  a = Sort[lis]; ap = Length[a]; a1 = a[[ap]];
  b = Sort[cellOutlines]; bp = bn = Length[b]; b1 = b[[bp]];
  While[
    ap >= 0 && bp >= 0,
    While[b1 < a1, res = {res, bp}; a1 = a[[--ap]] ];
    b1 = b[[--bp]];
  ];
  *)
  
  (* Here's a faster, functional method *)
  res = Position[Sort[Flatten[{N @ lis, cellOutlines}]], _Real];
  res = Flatten[res] - Range[Length @ res];
    
  (* In the return value, 0 indicates a match before the first cell of
  the notebook, and  Infinity indicates a match after the last cell. *)
  res /. Length[cellOutlines] -> Infinity
  
] /; FileType[nbFile] === File





(* End search functions *)



(* Formatting functions *)



stringToWordList[str_String] :=
  DeleteCases[StringSplit[str, {" ", ",", "\t", "-", "&", "|", "+", "-", "!"}], ""];



NotebookSearch::nonbs = "There were no valid notebooks specified.";
NotebookSearch::nohelp = "There were no valid help notebooks specified.";
NotebookSearch::nostrs = "There were no valid search strings specified.";
NotebookSearch::noform = "The specified format is invalid. Using $DefaultSearchFormat instead.";
NotebookSearch::nofe = "Help Browser searches require a front end. Using $DefaultSearchFormat instead."
NotebookSearch::max1 = "Maximum raw hit count exceeded. Truncating search."




NotebookSearch[opts___?OptionQ] :=
Block[{str, strs},
  str = InputString["Search for cells containing:"];
  strs = stringToWordList[str];
  
  If[StringMatchQ[str, "*&*"],
    NotebookSearch[strs, MultiWordSearch -> And, opts],
    NotebookSearch[strs, opts]
  ] /; strs =!= {}
]


NotebookSearch[strs_, opts___?OptionQ] :=
  NotebookSearch[$DefaultSearchNotebooks, strs, opts];

NotebookSearch[nbFiles_, strs_, opts___?OptionQ] := 
  NotebookSearch[nbFiles, strs, $DefaultSearchFormat, opts];

NotebookSearch[nbFiles_, Verbatim[And][s__String], fmt_String, opts___?OptionQ] :=
  NotebookSearch[nbFiles, {s}, fmt, MultiWordSearch -> And, opts]

NotebookSearch[nbFiles_, Verbatim[Or][s__String], fmt_String, opts___?OptionQ] :=
  NotebookSearch[nbFiles, {s}, fmt, MultiWordSearch -> Or, opts]

NotebookSearch[nbFiles_, strs_, format_String, opts___?OptionQ]:=
Block[{dirs, files, nbs, strings, fmt, res, cells, headerCells, footerCells,
  n, c, s, t, nb, focus, sort, searchResults, verb, inBrowser, selectedItems,
  nbExpr, categorize},

  Which[
    !MemberQ[$NotebookSearchFormats, format],
    Message[NotebookSearch::noform];
    fmt = $DefaultSearchFormat,
    StringMatchQ[format, "HelpBrowser*"] && Not[$Notebooks],
    Message[NotebookSearch::nofe];
    fmt = $DefaultSearchFormat,
    True,
    fmt = format
  ];
  
  files = Cases[Flatten @ {nbFiles}, _String];
  dirs = Select[files, FileType[#] === Directory&];
  files = Select[files, FileType[#] === File&];
  
  nbs = Cases[Flatten @ {nbFiles}, _NotebookObject];
  nbs = NotebookFilePath /@ nbs;
  
  files = Union[Flatten[{
    If[dirs == {}, {}, FileNames["*.nb", dirs, Infinity]],
    files,
    Cases[nbs, _String]
  }]];
  If[files === {},
    Message[NotebookSearch::nonbs];
    Return @ $Failed
  ];
  
  strings = Select[Flatten @ {strs}, StringQ[#] && StringLength[#]>0 &];
  If[strings === {},
    Message[NotebookSearch::nostrs];
    Return @ $Failed
  ];
  
  {sort, s, searchResults, verb, inBrowser, selectedItems, categorize, $highlightQ} =
    {SortByHitCount, MultiWordSearch, SearchInResults, Verbosity, ShowResultsInBrowser,
     SelectedItems, CategorizeResults, HighlightSearchStrings} /.
    Flatten[{opts, Options[NotebookSearch]}];
  
  sort = TrueQ[sort];
  searchResults = TrueQ[searchResults];
  $highlightQ = TrueQ[$highlightQ];
  s = ToString[s @@ strings, InputForm];
  
  $Verbosity = If[IntegerQ[verb], verb, 1];
  beginProgress[s];
  
  If[searchResults,
    files = Intersection[files, FoundNotebooks[]];
    If[files === {},
      Message[NotebookSearch::nonbs];
      endProgress[];
      Return @ $Failed
    ];
  ];
  
  If[
    StringMatchQ[fmt, "HelpBrowser*"],
    If[Not @ ListQ @ $BrowserLookupTable,
      showProgress[1][RebuildBrowserLookupTable::config];
      GetBrowserLookupTable[]
    ];
    files = Intersection[files, HelpNotebooks[]];
    If[ListQ @ selectedItems,
      files = Intersection[files, itemListToNotebookList[selectedItems]]
    ];
    If[searchResults, files = Intersection[files, FoundNotebooks[]]];
    If[files === {},
      Message[NotebookSearch::nohelp];
      endProgress[];
      Return @ $Failed
    ];
  ];
  
  {t, res} = Timing[nbSearch[files, encodeSearchString /@ strings,
    IncludeCount -> True,
    IncludeIndex -> True,
    IncludeOutline -> True,
    IncludeExpression -> MemberQ[{"Expressions", "LongOutput", "HelpBrowserLongOutput", "HelpBrowserExpressions"}, fmt],
    opts]];
  
  n = Length[res];
  c = If[n === 0, 0, Total["Count" /. Map[Rest, res]]];

  FoundNotebooks[] = First /@ res;
  
  showProgress[1][NotebookSearch::vform];
  
  Switch[fmt,
    "ShortOutput",
    res = "-Suppressed-";
    
    ,
    
    "HelpBrowserOutput" | "HelpBrowserLongOutput" | "HelpBrowserLinks" |
    "HelpBrowserCellLinks" | "HelpBrowserCellLink" | "HelpBrowserExpressions" |
    "HelpBrowserSmallLink",
    (* transform the results into categories/items *)
    showProgress[4]["Adding browser data"];
    res = addBrowserData /@ res;
    showProgress[4]["Resorting by browser data"];
    res = Flatten[regroupHitsByBrowserData /@ res,1];
    (* remove any hits outside the browser *)
    res = DeleteCases[res, {{}, ___}];
    (* remove any hits outside selectedItems *)
    If[ListQ @ selectedItems,
      res = Cases[res, {Alternatives @@ selectedItems, ___}]
    ];
    (* if necessary, remove hits outside of the previous item set *)
    If[searchResults, res = Cases[res, {Alternatives @@ FoundItems[], ___}]];
    (* gather the master index hits together by category instead of file *)
    showProgress[4]["Merging master index hits"];
    res = mergeMasterIndexHits[res];
    (* regardless of sorting preference, start by sorting by category string *)
    showProgress[4]["Sorting"];
    res = Last /@ Sort[Transpose[{StringJoin[First[#]]& /@ res, res}]];
    If[sort,
      (* if necessary, resort by the number of 'hits' per item *)
      res = Last /@ Sort[
        Transpose[{"Count" /. Map[Rest, res] /. "Count" -> {}, res}],
        OrderedQ[{First[#2], First[#1]}]&
      ]
    ];
    FoundItems[] = First /@ res;
    (* adjust the counts to refer to help items *)
    n = Length[res];
    c = If[n === 0, 0, Plus @@ ("Count" /. Map[Rest, res] /. "Count" -> {})];
    
    ,
    
    _, (* other formats *)    
    (* hits are already sorted alphabetically by file name *)
    If[sort,
      (* if neceesary, resort by the number of 'hits' per file *)
      showProgress[4]["Sorting"];
      res = Last /@ Sort[
        Transpose[{"Count" /. Map[Rest, res] /. "Count" -> {}, res}],
        OrderedQ[{First[#2], First[#1]}]&
      ]
    ]
  ];
  
  If[StringMatchQ[fmt, "*Output"],
    endProgress[];
    Return @ {res, t, c, n, Length @ files}
  ];
  
  focus = If[
    StringMatchQ[fmt, "HelpBrowser*"],
    "browser item",
    "notebook"
  ];
  
  $strs = strings;

  headerCells = {
    Cell["Search Results", "Title"],
    Cell[ToString @ 
      Which[
        c===0,
        StringForm["No cells found containing `1`", s],
        c===1,
        StringForm["Found 1 cell containing `1` in 1 `2`:", s, focus],
        n===1,
        StringForm["Found `1` cells containing `2` in 1 `3`:", c, s, focus],
        True,
        StringForm["Found `1` cells containing `2` in `3` `4`s:", c, s, n, focus]
      ],
      "Text"
    ]//If[StringMatchQ[fmt, "*Expressions"], highlightSearchStrings, Identity],
    horizontalRule
  };
  
  footerCells = {
    horizontalRule,
    Cell[ToString @
      StringForm["Searched `1` notebooks (`2`) in `3` seconds",
        Length @ files,
        sizeString @ Total @ Map[FileByteCount, files],
        t /. Second -> 1],
      "Text",
      TextAlignment -> Right,
      FontSlant -> "Italic"
    ],
    Cell[ToString @
      StringForm["Memory in use: `1`", sizeString[MemoryInUse[]]],
      "Text",
      TextAlignment -> Right,
      FontSlant -> "Italic"
    ],
    styleSheetButtonCell
  };
  
  showProgress[1][NotebookSearch::vform];
  
  
  cells = If[TrueQ[categorize] && StringMatchQ[fmt, "HelpBrowser*"],
    $trimCategories = False; (* might switch to True later *)
    formatResults[res, "Categories", fmt],
    $trimCategories = False;
    formatResults[#, fmt]& /@ res
  ];
  
  showProgress[4]["Opening notebook"];
  
  nbExpr = Notebook[
      Flatten @ {headerCells, cells, footerCells},
    WindowTitle -> StringJoin["Search Results: ", s],
    CellGrouping -> Manual,
    ShowCellTags -> False,
    StyleDefinitions -> "HelpBrowser.nb",
    ScreenStyleEnvironment -> "Working",
    InitializationCellEvaluation -> False
  ];
  
  Which[
    TrueQ @ inBrowser,
    nb=Export[
      ToFileName[{$UserBaseDirectory, "Applications", "NotebookSearch", "Documentation", "English"}, "SearchResults.nb"],
      nbExpr
    ];
    NotebookClose[HelpBrowserNotebook[]];
    HelpBrowserLookup["AddOns", "Search: Results"]
    ,
    ListQ @ inBrowser,
    nb=Export[First @ inBrowser, nbExpr];
    NotebookClose[HelpBrowserNotebook[]];
    HelpBrowserLookup @@ Last[inBrowser]
    ,
    True,
    nb=NotebookPut[nbExpr]
  ];

  SetOptions[$FrontEnd, FindSettings -> {"FindString" -> First[strings]}];
  
  endProgress[];
  
  {nb, t, c, n, Length @ files}
];




encodeSearchString[str_String] :=
  StringTake[ToString[str, InputForm, CharacterEncoding -> "ASCII"], {2, -2}]

sizeString[n_Integer] :=
StringJoin[
  Which[
    n < 1024, {ToString[n//N], " bytes"},
    n < 1024^2, {ToString[n/1024//N], " K"},
    n < 1024^3, {ToString[n/(1024^2//N)], " MB"},
    True, {ToString[n/(1024^3)//N], " GB"}
  ]
]







resultsHeadingCell[c_, content__]:=
Cell[TextData @ Flatten @ {
    ToString[c],
    If[c === 1, " cell in ", " cells in "],
    content
  },
  "Text",
  PageWidth -> Infinity,
  ShowGroupOpenCloseIcon -> True,
  ShowSpecialCharacters -> False,
  CellTags -> "SearchResultsHeadingCell"
]

resultsHeadingCell2[c_, content_, tip_]:=
Cell[BoxData @ FormBox[
    TooltipBox[
      content,
      ToString[c] <> If[c === 1, " cell in: ", " cells in: "] <> ToString[tip],
      ActionDelay -> 0.5
    ], TextForm],
  "Text",
  PageWidth -> Infinity,
  CellMargins -> {{Inherited,Inherited},{0,0}},
  ShowGroupOpenCloseIcon -> True,
  ShowSpecialCharacters -> False,
  CellTags -> "SearchResultsHeadingCell",
  FontFamily -> CurrentValue["PanelFontFamily"],
  FontSize -> CurrentValue["PanelFontSize"]
]



formatResults[{nbFile_, opts__}, "CellLinks"] := 
Block[{c, i, o},
  {c, i, o} = {"Count", "Index", "Outline"} /. {opts};
  
  {Cell[CellGroupData[{
    resultsHeadingCell[c, linkToNotebookFile[nbFile]],
    
    Cell[TextData @ BoxForm`Intercalate[
        MapThread[linkToCellN[nbFile, #1, #2]&, {i, o}], "\n"],
      "Text",
      ShowCellBracket -> False]
    },
  Closed]]}
];


formatResults[{nbFile_, opts__}, "NotebookLinks"] := 
  { resultsHeadingCell["Count" /. {opts}, linkToNotebookFile[nbFile]] }



formatResults[{nbFile_, opts__}, "Expressions"] := 
Block[{c, i, o, e},
  {c, i, o, e} = {"Count", "Index", "Outline", "Expression"} /. {opts};
  
  {Cell[CellGroupData[{
    resultsHeadingCell[c, linkToNotebookFile[nbFile]],

    MapThread[
      {Cell[TextData[linkToCellN[nbFile, #1, #2]], "Text",
        ShowCellBracket -> False],
       highlightSearchStrings @ #3,
       horizontalRule2}&, {i, o, e}]
    }//Flatten,
  Closed]]}
];



highlightSearchStrings[x_] := If[$highlightQ, highlight[x, $strs], x]

(*
highlightSearchStrings[x_] :=
If[$highlightQ,
  highlight[System`Convert`CommonDump`RemoveLinearSyntax[x,
    System`Convert`CommonDump`Recursive -> False], $strs],
  x
]
*)

highlight["",_] := ""

highlight[str_String, target_String] := 
Block[{pos, q},
  q = If[StringTake[str,1] === StringTake[str,-1] === "\"", "\"", ""];
  pos=StringPosition[str,target];
  If[pos==={},
    str,
    pos=First[pos];
    Flatten @ {
      If[First[pos] === 1, {}, StringTake[str,-1+First @ pos] <> q],
      highlight[target],
      If[Last[pos] === StringLength[str], {}, highlight[q <> StringDrop[str,Last @ pos],target]]
    }
  ]
]




highlight[StyleBox[x_,opts___],target_]:= StyleBox[#,opts]& /@ Flatten[{highlight[x,target]}]

(*highlight[StyleBox[x_,opts___],target_]:= StyleBox[highlight[x,target],opts]*)

highlight[ButtonBox[x_,opts___],target_]:= ButtonBox[#,opts]& /@ Flatten[{highlight[x,target]}]

(*highlight[ButtonBox[x_,opts___],target_]:= ButtonBox[highlight[x,target],opts]*)

highlight[TextData[x_],target_]:=TextData[Flatten @ {highlight[x,target]}]

highlight[{x__},target_]:= Flatten[highlight[#,target]& /@ {x}]

highlight[Cell[BoxData[x_], "Message", opts___], _] := Cell[BoxData[x], "Message", opts] (* careful with strings *)

highlight[Cell[x_String,opts___],target_]:= Cell[highlight[TextData[x],target],opts]

highlight[Cell[TextData[x_],opts___],target_]:= Cell[highlight[TextData[x],target],opts]

highlight[Cell[BoxData[x_], opts___], target_] := Cell[highlight[BoxData[x], target], opts]

highlight[BoxData[x_], target_] := BoxData[highlight[x,target]]

highlight[RowBox[lis_List], target_] := RowBox[Flatten[highlight[#, target]& /@ lis]]

highlight[InterpretationBox[x_, opts___], target_] := InterpretationBox[#, opts]& @@ {highlight[RowBox[{x}], target]}

highlight[(box:(SqrtBox | FormBox | TagBox | AdjustmentBox | ErrorBox))[
  x_, opts___], target_] := box[highlight[RowBox[{x}], target], opts]

highlight[(box:(SuperscriptBox | SubscriptBox | UnderscriptBox | OverscriptBox | FractionBox | RadicalBox))[
  x_, y_, opts___], target_] := box[highlight[RowBox[{x}], target], highlight[RowBox[{y}], target], opts]

highlight[(box:(SubsuperscriptBox | UnderoverscriptBox))[
  x_, y_, z_, opts___], target_] := box[highlight[RowBox[{x}], target], highlight[RowBox[{y}], target], highlight[RowBox[{z}], target], opts]

highlight[GridBox[x_, opts___], target_] := GridBox[Map[highlight[RowBox[{#}],target]&, x, {2}], opts]

(*highlight[x_, s:{__String}] := Fold[highlight, x, s]*)

highlight[x_, s:{__String}] := 
Block[{colors},
  colors = PadRight[$HighlightColors, Length[s], $HighlightColors];
  MapThread[(highlight[#1] = StyleBox[#1, Background -> #2])&, {s, colors}];
  Fold[highlight, x, s]
]

highlight[x_,___] := x


$HighlightColors = Hue[#, 0.2, 1]& /@ {0.8, 0.6, 0.4, 0.1, 0.2, 0.0};





formatResults[{{cats__}, opts__}, "HelpBrowserLinks"]:=
Block[{c, indexTag},
  {c, indexTag} = {"Count", "IndexTag"} /. {opts};
  
  { resultsHeadingCell[c, linkToBrowserItem[{cats}, indexTag]] }
];







formatResults[{{cats__}, opts__}, "HelpBrowserCellLinks"]:=
Block[{c, i, o, f, e, indexTag},
  {c, i, o, f, indexTag} = {"Count", "Index", "Outline", "File", "IndexTag"} /. {opts};
      
  {Cell[CellGroupData[{
    resultsHeadingCell[c, linkToBrowserItem[{cats}, indexTag]],
    
    If[{cats}[[2]] === "Master Index",
      Cell["", "Text", ShowCellBracket -> False],
      Cell[TextData @ Flatten @ BoxForm`Intercalate[
          MapThread[linkToBrowserCell[f, #1, #2, {cats}, indexTag]&, {i, o}], "\n"],
        "Text",
        ShowCellBracket -> False
      ]
    ]
    },
  Closed]]}
];



formatResults[{cats:{_, "Master Index", ___}, opts__}, "HelpBrowserCellLink"]:=
  formatResults[{cats, opts}, "HelpBrowserLinks"]


formatResults[{{cats__}, opts__}, "HelpBrowserCellLink"]:=
Block[{c, i, o, f, e, indexTag, text, link},
  {c, i, o, f, indexTag} = {"Count", "Index", "Outline", "File", "IndexTag"} /. {opts};
  
  {i,o} = First /@ {i,o};
  
  text = First @ linkToBrowserItem[{cats}, indexTag];
  
  link = First @ linkToBrowserCell[f, i, o, {cats}, indexTag];
  
  link = ReplacePart[link, Last @ {cats}, 1];
  
  {resultsHeadingCell[c, {text, link}]}
  
];




formatResults[{{cats__}, opts__}, "HelpBrowserSmallLink"]:=
Block[{c, i, o, f, e, indexTag, text, link},
  {c, i, o, f, indexTag} = {"Count", "Index", "Outline", "File", "IndexTag"} /. {opts};
  
  {i,o} = First /@ {i,o};
  
  text = First @ linkToBrowserItem[{cats}, indexTag];
  text = text <> indexTag;

  If[{cats}[[2]] === "Master Index",
    link = Last @ linkToBrowserItem[{cats}, indexTag]
    ,
    link = First @ linkToBrowserCell[f, i, o, {cats}, indexTag];
    link = ReplacePart[link, Last @ {cats}, 1];
    AppendTo[link, Method -> "Preemptive"];
  ];
  
  {resultsHeadingCell2[c, link, text]}
  
];








(* Master Index hits have "File" -> list *)
formatResults[{cats:{_,"Master Index",___}, opts__}, "HelpBrowserExpressions"]:=
Block[{c, i, o, f, e, indexTag},
  {c, f, e, indexTag} = {"Count", "File", "Expression", "IndexTag"} /. {opts};
    
  {Cell[CellGroupData[Flatten @ {
    resultsHeadingCell[c, linkToBrowserItem[cats, indexTag]],
    
    MapThread[
      {$MasterIndexHeading[#1],
       highlightSearchStrings @ showCellTags @ #2,
       horizontalRule2}&, {f, e}]
    },
  Closed]]}
];

showCellTags[Cell[a_,"IndexSubentry",b___]] :=
  Cell[a, "IndexSubentry", b, ShowCellTags -> True]
showCellTags[x_]:=x


(* other hits have "File" -> string *)
formatResults[{{cats__}, opts__}, "HelpBrowserExpressions"]:=
Block[{c, i, o, f, e, indexTag},
  {c, i, o, f, e, indexTag} = {"Count", "Index", "Outline", "File", "Expression", "IndexTag"} /. {opts};
  
  {Cell[CellGroupData[Flatten @ {
    resultsHeadingCell[c, linkToBrowserItem[{cats}, indexTag]],
    
    MapThread[
      {Cell[TextData[linkToBrowserCell[f, #1, #2, {cats}, indexTag]], "Text",
        ShowCellBracket->False],
       highlightSearchStrings @ #3, horizontalRule2}&, {i, o, e}]
    },
  Closed]]}
];




formatResults[res_, "Categories", fmt_] :=
Block[{cats, lis, i, c=0},
  cats = #[[1,2]]& /@ res;
  cats = Last /@ Sort[{-Length[#],First[#]}& /@ Split[Sort @ cats]];
  
  ( lis = Cases[res, {{_, #, ___}, ___}];
    i = Length[lis];
    Cell[CellGroupData[{
      Cell[TextData[{
        #,
        StyleBox[
          ToString @ StringForm["  (`1` `2`)", i, If[i>1,"items","item"]],
          "Text",
          FontWeight -> "Plain"
        ]}],
        "Section",
        ShowGroupOpenCloseIcon -> True
      ],
      
      formatResults[#, fmt]& /@ lis
    }//Flatten, If[(*++c===1*) Length[cats]===1, Open, Closed]]]
  )& /@ cats
  
  
]







horizontalRule = 
Cell["", "Text",
    CellFrame->{{0, 0}, {0, 0.5}},
    ShowCellBracket -> False,
    CellMargins->{{0, 0}, {1, 1}},
    CellElementSpacings->{"CellMinHeight"->1},
    CellFrameMargins->False,
    CellFrameColor->GrayLevel[0.8],
    CellSize->{Inherited, 3}
];

horizontalRule2 =
Cell["", "Text",
    CellFrame->{{0, 0.5}, {0.5, 0}},
    ShowCellBracket -> False,
    CellElementSpacings->{"CellMinHeight"->1},
    CellFrameMargins->False,
    CellFrameColor->GrayLevel[0.8],
    CellSize->{Inherited, 3}
];


styleSheetButtonCell = 
Cell[TextData[{
    "Style Sheet: ",
    ButtonBox["Default",
      ButtonFunction:>FrontEndExecute[ {
        SetOptions[ButtonNotebook[ ], StyleDefinitions -> "Default.nb"]}],
      ButtonStyle->"Hyperlink"],
    " or ",
    ButtonBox["Help Browser",
      ButtonFunction:>FrontEndExecute[ {
        SetOptions[ButtonNotebook[ ], StyleDefinitions -> "HelpBrowser.nb"]}],
      ButtonStyle->"Hyperlink"]
    }],
  "Text",
  TextAlignment->Right,
  FontSlant->"Italic"
];




linkToCellN[nbFile_String, n_Integer] := 
  linkToCellN[nbFile, n, "Cell " <> ToString[n]];

linkToCellN[nbFile_String, n_Integer, Cell[__Integer, "GraphicsData", _String, sty_String, ___]]:=
  linkToCellN[nbFile, n, "Cell " <> ToString[n] <> " (" <> sty <> ")"]

linkToCellN[nbFile_String, n_Integer, Cell[__Integer,sty_String,___]]:=
  linkToCellN[nbFile, n, "Cell " <> ToString[n] <> " (" <> sty <> ")"]

linkToCellN[nbFile_String, n_Integer, Cell[__Integer,sty_Symbol,___]]:=
  linkToCellN[nbFile, n, "Cell " <> ToString[n] <> " (" <> ToString[sty] <> ")"]


linkToCellN[nbFile_String, n_Integer, str_String] :=
ButtonBox[str,
  ButtonEvaluator -> Automatic,
  ButtonFunction :> 
    Module[{nb = NotebookOpen[nbFile]}, 
      FrontEndExecute[{
        FrontEnd`SelectionMove[nb, Before, Notebook],
        FrontEnd`SelectionMove[nb, Next, Cell, n],
        FrontEnd`FrontEndToken[nb, "OpenSelectionParents"]
      }]
    ],
  ButtonStyle -> "Hyperlink"
]




linkToNotebookFile[nbFile_String] := 
ButtonBox[
    StringReplace[nbFile, {
      $TopDirectory -> "$TopDirectory",
      $BaseDirectory -> "$BaseDirectory",
      $UserBaseDirectory -> "$UserBaseDirectory"}],
  ButtonData :> {nbFile, None}, 
  ButtonStyle -> "Hyperlink"
]






(* HelpPanelSearch *)


$HelpPanelSearchCategories = $HelpCategories;
$HelpPanelSearchFormat = "HelpBrowserSmallLink";

HelpPanelSearch[opts___?OptionQ] := HelpPanelSearch[ NotebookCreate[], opts]


HelpPanelSearch[nb_NotebookObject, opts___?OptionQ] :=
(
  NotebookPut[ HelpPanelNotebook[""], nb ];
  NotebookFind[nb, "SearchButtonCell", All, CellTags, AutoScroll -> False];
  SelectionMove[nb, Before, CellContents];
  SelectionMove[nb, Next, Character];
)


HelpPanelSearch[nb_NotebookObject, str_, opts___?OptionQ] :=
  HelpPanelSearch[nb, str, $HelpPanelSearchFormat, opts]


HelpPanelSearch[nb_NotebookObject, str_, format_String, opts___]:=
Block[{strs, $trimCategories=False, $highlightQ = False},
  
  strs = stringToWordList[str];
  If[strs === {}, Message[NotebookSearch::nostrs]; Return[$Failed]];  
  
  If[StringMatchQ[str, "*&*"], strs = And @@ strs, strs = Or @@ strs];
  
  NotebookPut[ HelpPanelNotebook[str], nb];
  helpPanelBeginProgress[nb, cat, str];
  If[Not @ ListQ @ $BrowserLookupTable,
    helpPanelShowProgress[nb, "Reading browser configuration"];
    GetBrowserLookupTable[]
  ];
  Scan[helpPanelSearchCategory[#, nb, strs, format, opts]&, $HelpPanelSearchCategories];
  helpPanelEndProgress[nb, cat, strs];
]



helpPanelSearchCategory[cat_, nb:_[_,n_], strs_, format_, opts___] :=
Block[{res, outputfmt},
  helpPanelShowProgress[nb, cat, strs];
  
  outputfmt = If[StringMatchQ[format, "*Expression*"], "HelpBrowserLongOutput", "HelpBrowserOutput"];
  
  res = NotebookSearch[HelpNotebooks[cat], strs, outputfmt, Verbosity -> 0, opts];
  If[res[[4]] === 0, Return[]];
  $total[n] += res[[4]];
  SelectionMove[nb, After, Notebook, AutoScroll -> False];
  NotebookWrite[nb, formatResults[First[res], "Categories", format], All, AutoScroll -> False];
]


helpPanelBeginProgress[nb:NotebookObject[_, n_], _, _] := 
(
  $searchinprogress[n] = True;
  $total[n] = 0;
  $progmsg[n] = "";
  SelectionMove[nb, After, Notebook];
  NotebookWrite[nb, Cell[
    BoxData[FormBox[DynamicBox[ToBoxes[$progmsg[n], StandardForm]], TextForm]],
    "Text",
    FontFamily -> CurrentValue["PanelFontFamily"],
    FontSize -> CurrentValue["PanelFontSize"],
    PageWidth -> Infinity
  ], All]
)

helpPanelShowProgress[NotebookObject[_, n_], msg_] := $progmsg[n] = ToString[msg]

helpPanelShowProgress[NotebookObject[_, n_], cat_, str_] :=
  $progmsg[n] = helpPanelProgressString[n, cat, ToString[str, InputForm]]


helpPanelEndProgress[NotebookObject[_, n_], cat_, str_] :=
(
  $searchinprogress[n] = False;
  $progmsg[n] = helpPanelProgressString[n, cat, ToString[str, InputForm]];
)



helpPanelProgressString[n_, cat_, str_] :=
StringJoin[
  If[$searchinprogress[n],
    ToString @ StringForm["Searching: `1`", First @ nameToCategoryList[cat]],
    "Search complete."
  ],
  Switch[{$total[n], $searchinprogress[n]},
    {0, True}, {"\n\n", ToString @ StringForm["`1`", str]},
    {0, False}, {"\n\n", ToString @ StringForm["`1` not found.", str]},
    
    {1, True}, {"\n\n", ToString @ StringForm["`1` found in 1 browser item so far.", str]},
    {1, False}, {"\n\n", ToString @ StringForm["`1` found in 1 browser item.", str]},
    
    {_, True}, {"\n\n", ToString @ StringForm["`1` found in `2` browser items so far.", str, $total[n]]},
    {_, False}, {"\n\n", ToString @ StringForm["`1` found in `2` browser items.", str, $total[n]]}
  ]
]






HelpPanelNotebook[str_String] :=
Notebook[{
    SearchButtonCell[str]
  },
  StyleDefinitions -> "HelpBrowser.nb",
  ScrollingOptions -> {"VerticalScrollRange" -> Fit},
  WindowTitle -> "Extended Search",
  CellGrouping -> Manual
]



SearchButtonFunction[nb_, c_Cell] :=
Block[{str},
  str = StringJoin[Cases[c, InputFieldBox[x_, ___] :> Cases[{x}, _String, Infinity], Infinity, 1]];
  HelpPanelSearch[nb, str]
]


SearchButtonCell[str_String] :=
Cell[BoxData @ FormBox[RowBox @ Flatten @ {
   InputFieldBox[str, String, FieldSize -> {20, 1}], " ", 
   ButtonBox["Search",
    ButtonFunction:>(CompoundExpression[
      Needs["AuthorTools`Experimental`"],
      Symbol["FrontEnd`MessagesToConsole"][
        Symbol["AuthorTools`Experimental`Private`SearchButtonFunction"][ButtonNotebook[], #1]]
      ]&),
    ButtonEvaluator->Automatic,
    ButtonSource->Cell,
    Method -> "Queued",
    ButtonFrame->"DialogBox"]}, TextForm],
  "Text",
  FontFamily -> CurrentValue["ControlsFontFamily"],
  System`ShowSyntaxStyles->False,
  AutoItalicWords->{},
  AutoSpacing -> False,
  AutoIndent -> False,
  CellTags -> "SearchButtonCell",
  ShowStringCharacters -> True
]










(* Reading information from browser configuration files *)


removeIndexFiles[lis_] := Select[lis, !StringMatchQ[#, "*BrowserIndex.nb"]&]


(*
  Note that the lists initially returned by HelpNotebooks are
  approximations only. They might include notebooks that don't appear
  in the browser, miss notebooks that do appear in the browser, and
  also grab style sheets or other configuration files. A better
  algorithm -- that actually walks the BrowserCategory expressions,
  gathering all and only those notebooks that are called by some item
  -- is invoked in RebuildBrowserLookupTable.
*)


HelpNotebooks["RefGuide"] = removeIndexFiles @ FileNames["*.nb",
  ToFileName[{$TopDirectory, "Documentation", "English", "RefGuide"}],
  Infinity]

HelpNotebooks["AddOns"] = removeIndexFiles @ FileNames["*.nb", {
  ToFileName[{$TopDirectory, "Documentation", "English", "AddOns"}],
  ToFileName[{$TopDirectory, "AddOns"}],
  $BaseDirectory,
  $UserBaseDirectory}, Infinity]

HelpNotebooks["MainBook"] = removeIndexFiles @ FileNames["*.nb",
  ToFileName[{$TopDirectory, "Documentation", "English", "MainBook"}],
  Infinity]

HelpNotebooks["GettingStarted"] = removeIndexFiles @ FileNames["*.nb",
  ToFileName[{$TopDirectory, "Documentation", "English", "GettingStarted"}],
  Infinity]

HelpNotebooks["OtherInformation"] = removeIndexFiles @ FileNames["*.nb",
  ToFileName[{$TopDirectory, "Documentation", "English", "OtherInformation"}],
  Infinity]

HelpNotebooks["FrontEnd"] = HelpNotebooks["OtherInformation"]

HelpNotebooks["Demos"] = removeIndexFiles @ FileNames["*.nb",
  ToFileName[{$TopDirectory, "Documentation", "English", "Demos"}],
  Infinity]

HelpNotebooks["Tour"] = removeIndexFiles @ FileNames["*.nb",
  ToFileName[{$TopDirectory, "Documentation", "English", "Tour"}],
  Infinity]

HelpNotebooks["MasterIndex"] = FileNames["BrowserIndex.nb", {
  ToFileName[{$TopDirectory, "Documentation", "English"}],
  ToFileName[{$TopDirectory, "AddOns"}],
  $BaseDirectory,
  $UserBaseDirectory}, Infinity]

HelpNotebooks["NewDocs"] = FileNames["*.nb",
  ToFileName[{$TopDirectory, "Documentation", "English", "NewDocumentation"}],
  Infinity]

HelpNotebooks[x___String] := HelpNotebooks[{x}]

HelpNotebooks[{x___String}] := Union @@ Map[HelpNotebooks, {x}]

HelpNotebooks[] = HelpNotebooks[$HelpCategories];

HelpNotebooks[__] := {}


FoundNotebooks[] = {};

FoundItems[] = {};



itemListToNotebookList[{it__}]:=
Join[
  Union @ Cases[$BrowserLookupTable, {f_, __, Alternatives[it]} :> f],
  If[!FreeQ[{it}, "Master Index"], $MasterIndexNotebooks, {}]
]





FrontEndVersionString[] :=
Block[{version = $NotebookVersionNumber, int},
  If[StringQ @ version, Return @ version];
  If[Not @ NumericQ @ version, Return @ "$Failed"];
  StringJoin[
    ToString[int = IntegerPart[version]],
    ".",
    ToString[IntegerPart[10 * IntegerPart[version - int]]]
  ]
]
    

(* Delay this definition until it's used, to avoid init.m load sequence errors *)
$BrowserCacheFile := $BrowserCacheFile = 
Block[{dir},

  (* On windows, there's way to get the version-number specific directory
     where the help browser caches are, so we have to punt for now. *)
  If[StringMatchQ[$System, "*Windows*"], Return @
    ToFileName[{$UserBaseDirectory}, "NotebookSearchBrowserData.mx"]
  ];
  
  dir = ToFileName[$NotebookUserBaseDirectory];
  If[FileType[dir] =!= Directory, Return[$Failed]];
  If[StringMatchQ[$System, "*Mac*"],
    dir = ToFileName[{dir, "FrontEnd", FrontEndVersionString[] <> " Caches"}],
    dir = ToFileName[{dir, "FrontEnd", FrontEndVersionString[] <> "_Caches"}]
  ];
  If[FileType[dir] =!= Directory, Return[$Failed]];
  dir = ToFileName[{dir, "HelpBrowserData"}];
  If[FileType[dir] === None, CreateDirectory[dir]];
  If[FileType[dir] =!= Directory, Return[$Failed]];
  ToFileName[{dir}, "NotebookSearchBrowserData.mx"]
]



GetBrowserLookupTable[] :=
Block[{},
  Which[
    Length[$BrowserLookupTable] > 0,
    Null,
    
    FileType[$BrowserCacheFile] === File &&
    (Get[$BrowserCacheFile]; Length[$BrowserLookupTable] > 0),
    Null,
    
    True,
    RebuildBrowserLookupTable[]; Null
  ];
]




RebuildBrowserLookupTable::config = "First run: caching browser configuration";

RebuildBrowserLookupTable[] :=
Module[{bcfile, notebooksInCategory},
  bcfile = ToFileName[{$TopDirectory, "Documentation", "English"}, "BrowserCategories.m"];
  If[FileType[bcfile] =!= File, Return[$Failed]];
  
  clearRawCache[];
  
  $ItemBag = Internal`Bag[];
  $BadBag = Internal`Bag[];
  $IndexBag = Internal`Bag[];
  indexListing[ToFileName[{$TopDirectory, "Documentation", "English"}, "BrowserIndex.nb"]];
  itemListing[Get[bcfile], {}, bcfile];
  
  (*
    Now, $IndexBag contains a list of all BrowserIndex.nb notebooks,
    and $ItemBag contains sublists of the form {0|1, nbFile, indexTag,
    copyTag, category path}. The 0|1 entry is used in sorting
    MainEntry -> True (0) items before other (1) items.
  *)
  
  (* throw away duplicate pointers to the same info *)
  (* old *)
  (* $BrowserLookupTable = Union[Internal`BagPart[$ItemBag, All]]; *)
  (* new *)
  $BrowserLookupTable = First /@ Split[
    Rest /@ Sort[
      Internal`BagPart[$ItemBag, All],
      OrderedQ[{Take[#1, 3], Take[#2, 3]}] &
    ],
    Take[#1, 2] === Take[#2, 2] &
  ];
  (* don't search the search *results* *)
  $BrowserLookupTable = DeleteCases[$BrowserLookupTable,
    {ToFileName[{$UserBaseDirectory, "Applications", "NotebookSearch", "Documentation", "English"}, "SearchResults.nb"],__}
  ];
  
  $MasterIndexNotebooks = Union[Internal`BagPart[$IndexBag, All]];
  
  (* this $MasterIndexHeading stuff expensive -- a few seconds, and a few MB of memory *)
  Clear[$MasterIndexHeading];
  $MasterIndexHeading[indfile_] := $MasterIndexHeading[indfile] = Block[{lis},
    lis = NotebookLookup[indfile, "CellExpression",
      Cell[___, CellTags -> ("MasterIndexHeading" | {___, "MasterIndexHeading", ___}), ___]
    ];
    If[lis === {},
      Cell["No affiliation found", "IndexSection"],
      First @ lis
    ]
  ];
  Scan[$MasterIndexHeading, $MasterIndexNotebooks];

  $MasterIndexEntries = Union @ Flatten @ Map[
    Cases[NotebookFileOutline[#], _[CellTags, t_] :> t, Infinity]&,
    $MasterIndexNotebooks
  ];
  
  Clear[$ItemBag, $IndexBag];
  
  (* redefine HelpNotebooks based on actual browser information, now that we have it *)
  HelpNotebooks[] = Union[First /@ $BrowserLookupTable, $MasterIndexNotebooks];
  
  notebooksInCategory[s_]:= Union[Cases[
    $BrowserLookupTable,
    {f_, ___, {_, Alternatives @@ nameToCategoryList[s], ___}} :> f
  ]];
  Scan[
    (HelpNotebooks[#] = notebooksInCategory[#])&,
    DeleteCases[$HelpCategories, "MasterIndex"]
  ];
  HelpNotebooks["MasterIndex"] = $MasterIndexNotebooks;  
  
  (* splitting this list by notebook makes later use very efficient *)
  Clear[$LookupCache];
  $LookupCache[nbFile_String] :=
    $LookupCache[nbFile] = Cases[$BrowserLookupTable, {nbFile, __}];
  Scan[$LookupCache, Union[First /@ $BrowserLookupTable]];
  
  (* store for fast access next time *)
  DumpSave[$BrowserCacheFile, {
    $BrowserLookupTable,
    $MasterIndexNotebooks,
    $MasterIndexHeading,
    $MasterIndexEntries,
    $LookupCache,
    HelpNotebooks
  }];
  
  Length /@ {$BrowserLookupTable, HelpNotebooks[], $MasterIndexNotebooks, $MasterIndexEntries}
];


(* Browser index support *)

indexListing[indfile_String] :=
  If[FileType[indfile] === File, Internal`StuffBag[$IndexBag, indfile]];

itemListing[Global`HelpMasterIndex[], ___] := Null



(*
   Empty categories contribute nothing.

   Populated categories contribute their category path to all enclosed
   items, as long as their second argument is None.

   Populated categories with a string second argument pass four
   arguments through itemListing, so the enclosed items can use that
   string as a default second argument.
*)


itemListing[b:BrowserCategory[cat_, _, {}], catpath_, bcfile_]:=Null

itemListing[BrowserCategory[cat_, None, lis_List], catpath_, bcfile_]:=
  itemListing[lis, Append[catpath, cat], bcfile]

itemListing[BrowserCategory[cat_, f_String, lis_List], catpath_, bcfile_]:=
  itemListing[lis, Append[catpath, cat], bcfile, f]

itemListing[lis_List, args__]:=
  Scan[itemListing[#, args]&, lis]



(*
   Delimiters contribute nothing.

   Items with a string for a second argument contribute a sublist:
   {copytag, indextag, item expr, catpath, directory}

   Items without a string for a second argument look to the enclosing
   category for a file name.

   Items with MainEntry -> True are sorted before other items.
*)


itemListing[Item[Delimiter], catpath_, bcfile_] := Null;


getIndexTag[Item[___, _[IndexTag, s_String], ___]] := s;
getIndexTag[Item[___, _[IndexTag, {s_String, ___}], ___]] := s;
getIndexTag[Item[s_String, ___]] := s;

getCopyTag[Item[___, _[CopyTag, t_], ___]] := First @ Flatten @ {t};
getCopyTag[Item[___, _[IndexTag, t_], ___]] := First @ Flatten @ {t};
getCopyTag[Item[t_, ___]] := t

itemNotebookPath[Item[_, f_String, ___], bcfile_, ___] :=
  ToFileName[{DirectoryName @ bcfile}, f]

itemNotebookPath[Item[_, FrontEnd`FileName[{dirs___String}, f_String],___], bcfile_, ___] :=
  ToFileName[{DirectoryName @ bcfile, dirs}, f]

itemNotebookPath[_Item, bcfile_, f_] :=
  ToFileName[{DirectoryName @ bcfile}, f]

itemNotebookPath[x___] := $Failed



itemListing[i:Item[a_, ___], {cats___}, args___] :=
Block[{main, nbfile},
  nbfile = itemNotebookPath[i, args];
  main = If[ MatchQ[i, Item[___, Symbol["Global`MainEntry"] -> True, ___]], 0, 1];
  
  If[StringQ[nbfile] && FileType[nbfile] === File,
    (* only bag this information if the source notebook actually exists *)
    Internal`StuffBag[$ItemBag, {main, nbfile, getIndexTag[i], getCopyTag[i], {cats, a}}],
    (* otherwise, store for other use *)
    Internal`StuffBag[$BadBag, {main, nbfile, getIndexTag[i], getCopyTag[i], {cats,a}}]
  ]; 
]




(*
   HelpDirectoryListings contribute the items in the categories to
   which they point. Although, figuring out exactly which files they
   point to is rather tricky.

   The syntax of this symbol, according to John Fultz, is as follows:

   arg1: Path(s) to search

   arg2: Look in subdirectories (True/False)

   arg3: Add Documentation/<Language> to directory names (True/False)

   arg4: Process only first item found (True/False)

   arg5: List of paths to *exclude* from the search

   Args 4 and 5 are not implemented for now, but 1-3 are. And their
   implementation is based on Pavi's HelpDirectoryListing docs.
   Essentially, it says the following three things.

   1. Any path given as the first argument that is not an absolute
   path is relative to $TopDirectory/Documentation/English

   2. The second and third arguments to HelpDirectoryListing designate
   where in dir one should look for browser config files:


   arg2     arg3      Location searched

   False    False     directory

   False    True      directory/Documentation/English

   True     False     directory/*

   True     True      directory/*/Documentation/English


   Note: Here directory/ * refers to all subdirectories located one
   level below directory.

   3. HelpDirectoryListing[dir] is equivalent to
   HelpDirectoryListing[dir,True], and HelpDirectoryListing[dir,arg2]
   is equivalent to HelpDirectoryListing[dir,arg2,arg2]

   The main function here is bcDirs, which takes a
   HelpDirectoryListing as input, and returns all directories that
   might contain browser configuration files.
   
*)



itemListing[h_HelpDirectoryListing, catpath_, bcfile_] := 
Block[{dirs = Flatten @ List @ bcDirs @ h},
  Scan[indexListing, FileNames["BrowserIndex.nb", dirs]];
  Scan[itemListing[Get[#], catpath, #]&, FileNames["BrowserCategories.m", dirs]]
];




absDirPath[str_] := ToFileName[{$TopDirectory, "Documentation", "English", str}];

absDirPath[f:FrontEnd`FileName[{$TopDirectory, ___}, ___]] := ToFileName[f];
absDirPath[f:FrontEnd`FileName[{$InstallationDirectory, ___}, ___]] := ToFileName[f];
absDirPath[f:FrontEnd`FileName[{$PreferencesDirectory, ___}, ___]] := ToFileName[f];
absDirPath[f:FrontEnd`FileName[{$UserAddOnsDirectory, ___}, ___]] := ToFileName[f];
absDirPath[f:FrontEnd`FileName[{$UserBaseDirectory, ___}, ___]] := ToFileName[f];
absDirPath[f:FrontEnd`FileName[{$AddOnsDirectory, ___}, ___]] := ToFileName[f];
absDirPath[f:FrontEnd`FileName[{$BaseDirectory, ___}, ___]] := ToFileName[f];

absDirPath[FrontEnd`FileName[{args___}, ___]] :=
  ToFileName[{$TopDirectory, "Documentation", "English", args}]



bcDirs[HelpDirectoryListing[arg1_]] := bcDirs[HelpDirectoryListing[arg1, True]];

(* this line ignores certain uses -- must be fixed someday *)
bcDirs[HelpDirectoryListing[arg1_, arg2_List]] := bcDirs[HelpDirectoryListing[arg1]];

bcDirs[HelpDirectoryListing[arg1_, arg2_]] := bcDirs[HelpDirectoryListing[arg1, arg2, arg2]];

bcDirs[HelpDirectoryListing[AddOnHelpPath]] := Flatten[
  bcDirs[HelpDirectoryListing[{#}]]& /@
  (AbsoluteCurrentValue[$FrontEnd, AddOnHelpPath])
]

bcDirs[HelpDirectoryListing[{dir_}, False, False]] :=
  absDirPath[dir]

bcDirs[HelpDirectoryListing[{dir_}, False, True]] := 
  ToFileName[{absDirPath[dir], "Documentation", "English"}]

bcDirs[HelpDirectoryListing[{dir_}, True, False]] := 
  FileNames["*", absDirPath[dir]]

bcDirs[HelpDirectoryListing[{dir_}, True, True]] := 
  ToFileName[{#, "Documentation", "English"}]& /@ FileNames["*", absDirPath[dir]]



(*
   All other elements in the browser category expressions contribute
   nothing.
*)


RebuildBrowserLookupTable::badin = "Encountered unknown input to RebuildBrowserLookupTable. Skipping `1`.";


itemListing[x_, catpath_, bcfile_] :=
  (showProgress[10][RebuildBrowserLookupTable::badin, x]; Null)






nameToCategoryList["AddOns"] = {"Add-ons & Links", "Add-ons and Links", "Add-ons"};
nameToCategoryList["Demos"] = {"Demos"};
nameToCategoryList["GettingStarted"] = {"Getting Started", "Getting Started/Demos"};
nameToCategoryList["MainBook"] = {"The Mathematica Book"};
nameToCategoryList["MasterIndex"] = {"Master Index"};
nameToCategoryList["OtherInformation"] = {"Front End", "Other Information"};
nameToCategoryList["RefGuide"] = {"Built-in Functions"};
nameToCategoryList["Tour"] = {"Tour"};
nameToCategoryList[cat_] := {"Unknown Category (" <> ToString[cat] <> ")"};

nameToButtonStyle[x_String] := StringJoin[x, "Link"]


(* categoryToName is essentially the inverse of nameToCategoryList *)
categoryToName[x_] := categoryToName[x] =
Block[{lis},
  lis = Select[$HelpCategories, MemberQ[nameToCategoryList[#], x]&];
  If[lis === {}, "MasterIndex", First[lis]]
]

categoryToButtonStyle[x_] := nameToButtonStyle @ categoryToName[x]



linkToBrowserItem[cats_, indexTag_]:=
Flatten[{
  StringJoin[{#," > "}& /@ Take[cats,{If[$trimCategories, 3, 2],-2}]],
  ButtonBox[Last @ cats,
    ButtonData -> indexTag,
    ButtonNote -> indexTag,
    ButtonStyle -> categoryToButtonStyle[ cats[[2]] ]
  ]
}]//DeleteCases[#,""]&



linkToBrowserCell[nbFile_, n_, o_, cats:{_,"Master Index", ___}, indexTag_]:=
  linkToBrowserItem[cats, indexTag]

linkToBrowserCell[nbFile_, n_, o_, cats_, indexTag_]:=
With[{searchStrings = $strs},
{
  ButtonBox[First @ linkToCellN[nbFile, n, o],
    ButtonStyle -> "Hyperlink",
    ButtonEvaluator -> Automatic,
    ButtonFunction :> GoToBrowserCell[nbFile, n, o, cats, indexTag, searchStrings],
    ButtonFrame -> None
  ]
}
]



GoToBrowserCell[nbFile_, i_, o_, cats_, indexTag_, searchStrings_:{}]:=
Module[{copyTag, browserOffset=1, lis, nb, oldCell, newCell},
  copyTag = Alternatives @@ Flatten[{None, Cases[o, _[CellTags, t_] :> t]}];
  copyTag = Cases[$LookupCache @ nbFile, {_, indexTag, copyTag, cats}];
  If[copyTag === {}, Return["ERROR"], copyTag = copyTag[[1,3]]];
  
  If[copyTag === None,
    (* if CopyTag is None, then the browser offset is the cell offset *)
    browserOffset = i,
    (* otherwise we must calculate it *)
    lis = NotebookLookup[nbFile, "CellIndex",
      Cell[___,CellTags -> (copyTag | {___, copyTag, ___}),___]];
    lis = Position[lis, i];
    If[lis === {}, browserOffset = 1, browserOffset = lis[[1,1]]]
  ];
  
  HelpBrowserLookup[categoryToName @ cats[[2]], indexTag];
  nb = HelpBrowserNotebook[];
  
  FrontEndExecute[{
    FrontEnd`SelectionMove[nb, Before, Notebook],
    FrontEnd`SelectionMove[nb, Next, Cell, browserOffset],
    FrontEnd`FrontEndToken[nb, "OpenSelectionParents"],
    FrontEnd`SetOptions[FrontEnd`NotebookSelection[nb], ShowCellBracket -> True]
  }];
  
  If[searchStrings =!= {} &&
     $highlightQ &&
     !MemberQ[
       {"Graphics", "Picture", "PictureGroup", "ItemizedPicture", 
        "OpenCloseItemizedPicture", "ImportPict", "Sound"},
       "Style" /. First[Developer`CellInformation[nb]]
     ],
    oldCell = NotebookRead[nb];
    newCell = highlight[oldCell, searchStrings];
    If[oldCell =!= newCell, NotebookWrite[nb, newCell, All]];
  ];
  
  nb
]




(*
   Utilities nba2hba and hba2nba translate back and forth between the
   "Notebook Address" (nba) and "Help Browser Address" (hba) of an
   individual cell.
*)

nba2hba[ {nbFile_String, cellOffset_Integer}] :=
Module[{o, tags, tagpat, matches, nb, indexTag, copyTag, cats, lis, browserOffset},
  o = NotebookLookup[nbFile, "CellOutline"];
  If[ Length[o] < cellOffset, Message[nba2hba::toofew]; Return[$Failed]];
  o = o[[ cellOffset ]];

  tags = Flatten[Cases[o, _[CellTags, t_] :> t]];
  If[tags === {}, tagpat = None, tagpat = Alternatives @@ tags];
  
  GetBrowserLookupTable[];
  matches = Cases[$LookupCache @ nbFile, {_, _, tagpat, _}];
  Switch[matches,
    {}, Message[nba2hba::nohelp]; Return[$Failed],
    {_, __}, Message[nba2hba::multi]
  ];
  {nb, indexTag, copyTag, cats} = First @ matches;
  
  If[copyTag === None,
    (* if CopyTag is None, then the browser offset is the cell offset *)
    browserOffset = cellOffset,
    (* otherwise we must calculate it *)
    lis = NotebookLookup[nbFile, "CellIndex",
      Cell[___,CellTags -> (copyTag | {___, copyTag, ___}),___]];
    lis = Position[lis, cellOffset];
    If[lis === {}, browserOffset = 1, browserOffset = lis[[1,1]]]
  ];
  
  {categoryToName @ cats[[2]], indexTag, browserOffset}
]


hba2nba[ {cat_String, indexTag_String, browserOffset_Integer}] :=
Module[{ category, matches, nbFile, itag, copyTag, cats, cellOffset, lis},
  GetBrowserLookupTable[];
  category = First @ nameToCategoryList[cat];
  matches = Cases[$BrowserLookupTable, {_, indexTag, _, {_, category, ___}}];
  Switch[ matches,
    {}, Message[hba2nba::none]; Return[$Failed],
    {_, __}, Message[hba2nba::multi]
  ];
  {nbFile, itag, copyTag, cats} = First @ matches;
  
  If[copyTag === None,
    (* if CopyTag is None, then the notebook offset is the browser offset *)
    i = browserOffset,
    (* otherwise we must calculate it *)
    lis = NotebookLookup[ nbFile, "CellIndex",
      Cell[___,CellTags -> (copyTag | {___, copyTag, ___}),___]];
    If[ Length[lis] < browserOffset,
      Message[hba2nba::toofew]; cellOffset = $Failed,
      cellOffset = Part[lis, browserOffset]
    ]
  ];
  {nbFile, cellOffset}
]





Scan[(init[#] = #) &, CharacterRange["A", "Z"]];
Scan[(init[#] = ToUpperCase[#]) &, CharacterRange["a", "z"]];
Scan[(init[#] = "123...") &, CharacterRange["0", "9"]];
init["$"] = "$";
init[___] := "!@#...";

getInitial[""] = "?";
getInitial[str_String] := init[StringTake[str, 1]];



findInLookupTable[nbFile_, Cell[args___]]:=
Block[{tags, t},
  tags = Flatten[Cases[{args}, _[CellTags, t_] :> t]];
  If[tags === {},
    {{}, None},
    t = First[tags];
    {{"Help Browser", "Master Index", getInitial[t], t}, t}
  ]
] /; MemberQ[$MasterIndexNotebooks, nbFile]


findInLookupTable[nbFile_, Cell[args___]]:=
Block[{copyTagPat, x},
  copyTagPat = Alternatives @@ Flatten[{None, Cases[{args}, _[CellTags, t_] :> t]}];
  x = Cases[$LookupCache @ nbFile, {_, i_, copyTagPat, cats_} :> {cats, i}];
  If[x === {}, {{},None}, First[x]]
]


findInLookupTable[___] := {}




addBrowserData[{nbFile_String, opts__}]:=
Block[{o}, 
  o = "Outline" /. {opts};
  {nbFile, opts,
    "Browser" -> Map[findInLookupTable[nbFile, #]&, o]
  }
]


regroupHitsByBrowserData[{nbFile_, opts___}]:=
Block[{data, exprQ},
  exprQ = MatchQ[{opts}, {___, "Expression" -> _, ___}];

  If[exprQ,
    data = {"Browser", "Index", "Outline", "Expression"}  /. {opts},
    data = {"Browser", "Index", "Outline"} /. {opts}
  ];
  
  data = Split[Sort[Transpose[data]], First[#1] === First[#2]&];

  Map[{
      #[[1,1,1]],
      "IndexTag" -> #[[1,1,2]],
      "File" -> nbFile,
      "Count" -> Length[#],
      "Index" -> #[[All, 2]],
      "Outline" -> #[[All, 3]],
      If[exprQ, "Expression" -> #[[All, 4]], Unevaluated @ Sequence[]]
    }&,
    data
  ]
]



(*
   Since the master index hits might have multiple notebooks pointing
   to a single mater index entry (eg, Master Index > P > Plot has
   information from the Book and the GSP), we need to take special
   care in combining them. This is the only time a single help browser
   destination can be made up of information from more than one
   notebook.
*)


mergeMasterIndexHits[lis_] := 
Block[{data, indexHits, indexHitsGrouped, uniqueTargets},

  indexHits = Cases[lis,{{_,"Master Index",___},___}];
  
  indexHitsGrouped = Split[Sort[indexHits], First[#1] === First[#2]&];
  
  uniqueTargets = #[[1,1]]& /@ indexHitsGrouped;
  
  indexHits = MapThread[mergeOneIndexEntry, {uniqueTargets, indexHitsGrouped}];
  
  Join[
    DeleteCases[lis, {{_,"Master Index",___},___}],
    indexHits
  ]
]


mergeOneIndexEntry[cat_, lis_] :=
Module[{data, exprQ, indexTag, i, f, o, e},
  exprQ = MatchQ[First[lis], {___, "Expression" -> _, ___}];

  If[exprQ,
    data = {"IndexTag", "File", "Index", "Outline", "Expression"}  /. Map[Rest,lis];
    {indexTag, f, i, o, e} = Transpose[data],
    data = {"IndexTag", "File", "Index", "Outline"}  /. Map[Rest,lis];
    {indexTag, f, i, o} = Transpose[data];
  ];
      
  {cat,
    "IndexTag" -> Flatten[indexTag][[1]],
    "File" -> Flatten[MapThread[Table[#1, {#2}]&, {f, Length /@ i}]],
    "Count" -> Length[Flatten[i]],
    "Index" -> Flatten[i],
    "Outline" -> Flatten[o],
    If[exprQ, "Expression" -> Flatten[e], Unevaluated @ Sequence[]]
  }
]





Options[ItemLookup] =
{
  ItemLookupCategories :> DeleteCases[$HelpCategories, "MasterIndex"],
  PartialMatch -> True,
  IgnoreCase -> True
}




ItemLookup::nomatch = "No item matching \"`1`\" found.";

ItemLookup[opts___?OptionQ] :=
  ItemLookup[InputString["Search for item names containing:"], opts]

ItemLookup[str_String, opts___?OptionQ] :=
  ItemLookup[str, False, opts]

ItemLookup[str_String, catQ:(True | False), opts___?OptionQ] :=
  ItemLookup[str, catQ, False, opts]

ItemLookup[str_String, catQ:(True|False), outputQ:(True|False), opts___?OptionQ] :=
Block[{lis, lis2, f, cats, partialQ, ignoreCaseQ, matchQ, strPat, catPat},

  If[Not @ ListQ @ $BrowserLookupTable, GetBrowserLookupTable[]];
  
  {cats, partialQ, ignoreCaseQ} = {ItemLookupCategories, PartialMatch, IgnoreCase} /. 
    Flatten[{opts, Options[ItemLookup]}];
  
  {partialQ, ignoreCaseQ} = TrueQ /@ {partialQ, ignoreCaseQ};
  
  If[cats === All, cats = $HelpCategories];
  cats = Map[First[nameToCategoryList @ #]&, cats];
  catPat = Alternatives @@ cats;
  
  If[catQ,
    If[partialQ,
      matchQ[{_, catPat, c__}] := matchQ @ StringJoin[c],
      matchQ[{_, catPat, c__}] := Or @@ Map[matchQ, {c}]
    ],
    matchQ[{_, catPat, ___, c_}] := matchQ[c]
  ];
  
  strPat = If[partialQ, "*" <> str <> "*", str];
  
  matchQ[c_String] := StringMatchQ[c, strPat, IgnoreCase -> ignoreCaseQ];
  matchQ[___] := False;
  
  lis = Select[$BrowserLookupTable, matchQ @ Last @ # &];
  lis = Part[#, {4, 2}]& /@ lis;
  (* add master index hits, if necessary *)
  If[MemberQ[cats, "Master Index"],
    lis2 = Select[$MasterIndexEntries, matchQ];
    lis2 = findInLookupTable[First[$MasterIndexNotebooks], Cell[CellTags -> #]]& /@ lis2;
    lis = Join[lis, lis2]
  ];
  
  (* sort by category / item names *)
  lis = Rest /@ Sort[Prepend[#, StringJoin @ First @ #]& /@ lis];
  (* group by category name, sorted to the user's category order *)
  lis = DeleteCases[Cases[lis, {{_,#,___},___}]& /@ cats, {}];
  
  If[outputQ,
    Flatten[lis, 1],
    printItemLookup[strPat, lis]; Null
  ]
]


printItemLookup[str_, {}] := Message[ItemLookup::nomatch, str];

printItemLookup[_, lis_] := CellPrint @ itemLookupCell @ lis;

itemLookupCell[lis_] :=
(
  $trimCategories = False;
  
  Cell[TextData @ Flatten @ {
      BoxForm`Intercalate[itemLookupLines /@ lis, "\n\n"] },
    "Print",
    FontFamily->"Times",
    CellMargins -> {{20, Inherited},{Inherited,Inherited}},
    PageWidth -> Infinity,
    Background->RGBColor[0.964706, 0.929412, 0.839216],
    ShowSpecialCharacters -> False
  ]
)


itemLookupLines[lis_] :=
  BoxForm`Intercalate[linkToBrowserItem @@@ lis, "\n"]


ItemLookupMenu[] :=
Block[{str, lis, n},
  str = InputString["Search for item names containing:"];
  If[str === "", Return @ Null];
  lis = ItemLookup[str, True, True, ItemLookupCategories -> All];
  n = Length[lis];
  lis = Split[lis, #1[[1, 2]] === #2[[1, 2]] &];
  NotebookPut[Notebook[{
    If[n === 0,
      Cell["No matching items.", "Text"],
      itemLookupCell[lis]
    ]},
    WindowSize -> {Fit, 250},
    ShowCellBracket -> False,
    WindowMargins -> {{0, Automatic}, {0, Automatic}},
    WindowTitle -> ToString[StringForm["Lookup Results: `1` (`2`)", str, n]],
    Background -> RGBColor[0.964706, 0.929412, 0.839216],
    ScrollingOptions -> {"VerticalScrollRange" -> Fit}
  ]]
]




InstallSearchMenus::error = "An error occurred. The menus were not installed.";

InstallSearchMenus::restart = "The search menu items will be present the next time you launch the front end.";

InstallSearchMenus[] := InstallSearchMenus[$UserBaseDirectory]

InstallSearchMenus[dir_String] := 
Block[{d, file},
  d = dir;
  If[FileType[d] =!= Directory, Message[InstallSearchMenus::error]; Return[$Failed]];
  
  d = ToFileName[{d, "Autoload"}];
  If[FileType[d] === None, CreateDirectory[d]];
  If[FileType[d] =!= Directory, Message[InstallSearchMenus::error]; Return[$Failed]];
  
  d = ToFileName[{d, "NotebookSearchMenus"}];
  If[FileType[d] === None, CreateDirectory[d]];
  If[FileType[d] =!= Directory, Message[InstallSearchMenus::error]; Return[$Failed]];
  
  d = ToFileName[{d, "FrontEnd"}];
  If[FileType[d] === None, CreateDirectory[d]];
  If[FileType[d] =!= Directory, Message[InstallSearchMenus::error]; Return[$Failed]];
  
  file = ToFileName[{d}, "init.m"];
  If[FileType[file] === File, DeleteFile[file]];
  If[FileType[file] =!= None, Message[InstallSearchMenus::error]; Return[$Failed]];
  
  (* The 'With' below prevents these symbols from being created unless absolutely necessary. *)
  With[{
    kernelExecute = Symbol["Global`KernelExecute"],
    menuEvaluator = Symbol["Global`MenuEvaluator"],
    menuKey = Symbol["Global`MenuKey"],
    command = Symbol["Global`Command"],
    option = Symbol["Global`Option"],
    modifiers = Symbol["Global`Modifiers"],
    shift = Symbol["Global`Shift"]    
    },
  Put[Unevaluated @ FrontEnd`AddMenuCommands["RebuildHelpIndex",
  {
  Delimiter,
  Item["Item Name Search...",
    kernelExecute @ ToExpression @ StringJoin[
      "Needs[\"AuthorTools`Experimental`\"];",
      "AuthorTools`Experimental`Private`ItemLookupMenu[];"
    ],
    menuEvaluator -> "Local",
    menuKey["f", modifiers -> {command, option}]
  ],
  Item["Full-Text Search...",
    kernelExecute @ ToExpression @ StringJoin[
      "Needs[\"AuthorTools`Experimental`\"];",
      "AuthorTools`Experimental`NotebookSearch[]"
    ],
    menuEvaluator -> "Local",
    menuKey["F", modifiers -> {command, shift, option}]
  ],
  Item["Rebuild Search Index",
    kernelExecute @ ToExpression @ StringJoin[
      "Needs[\"AuthorTools`Experimental`\"];",
      "AuthorTools`Experimental`RebuildBrowserLookupTable[];"
    ],
    menuEvaluator -> "Local"
  ]
  }], file]
  ];
  
  
  If[FileType[file] =!= File, Message[InstallSearchMenus::error]; Return[$Failed]];

  Message[InstallSearchMenus::restart];
  file
];



UninstallSearchMenus::error = "An error occurred. The menus were not removed.";

UninstallSearchMenus::restart = "The search menu items will be absent the next time you launch the front end.";

UninstallSearchMenus[] := UninstallSearchMenus[$UserBaseDirectory];

UninstallSearchMenus[dir_String] := 
Block[{file},
  file = ToFileName[{dir, "Autoload", "NotebookSearchMenus", "FrontEnd"}, "init.m"];
  If[FileType[file] === File, DeleteFile[file]];
  If[FileType[file] =!= None, Message[UninstallSearchMenus::error]; Return[$Failed]];
  Message[UninstallSearchMenus::restart];
  
]




End[]

EndPackage[]
