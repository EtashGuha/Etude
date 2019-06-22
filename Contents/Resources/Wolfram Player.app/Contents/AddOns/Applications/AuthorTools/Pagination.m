(* :Context: AuthorTools`Pagination` *)

(* :Author: Louis J. D'Andria *)

(* :Summary:
    This package provides the support functions for extracting
    and caching page numbers.
*)

(* :Copyright: *)

(* :Package Version: $Revision: 1.17 $ $Date: 2004/05/03 18:38:51 $ *)

(* :Mathematica Version: 4.2 *)

(* :History:
    
*)

(* :Keywords:
    document, notebook, formatting, pagination 
*)

(* :Discussion:
    
*)

(* :Warning:
    
*)


BeginPackage["AuthorTools`Pagination`", 
  {"AuthorTools`Common`",
   "AuthorTools`MakeProject`"}];


StartingPages::usage = "StartingPages is an option to Paginate which specifies a list {p1, p2, ..., pn} where pk is the starting page number for the kth notebook in the project. Values can be integers, Inherited, \"Next\", \"Even\", or \"Odd\".";

OpenAllCellGroups::usage = "OpenAllCellGroups is an option to Paginate that determines whether to open all the cell groups in each notebook before calculating page breaks.";

PaginationFunction::usage = "PaginationFunction is an option to Paginate that specifies a function to apply to each notebook object after page numbers are calculated but before the notebook is closed.";

PaginationNumerals::usage = "PaginationNumerals is an option to Paginate that specifies what kind of numbering to use for the given notebooks. Possible values are Automatic, \"RomanNumerals\", and \"CapitalRomanNumerals\".";

Paginate::usage = "Paginate[projFile] sets the starting page number for each notebook file in the given project to be the page after the previous notebook finished. It caches the page numbers in each notebooks' tagging rules.";

NotebookPageNumbers::usage = "NotebookPageNumbers[nb] asks the front end to calculate page numbers for nb, and returns a list of integers, one page number for each cell in nb.";

NotebookPaginationCache::usage = "NotebookPaginationCache[nb] reads nb's tagging rules and returns the pagination cache, or None if none exists.";


Begin["`Private`"]


Options[Paginate] = {
  StartingPages -> {1, "Next"},
  OpenAllCellGroups -> False,
  PaginationFunction -> Identity,
  PaginationNumerals -> Automatic
};


Paginate::pages = "StartingPages should be set to a list of length at least 2.";

Paginate[nb_NotebookObject, opts___] :=
Block[{},
  NotebookSaveWarning[nb, Paginate];
  paginate[{NotebookFilePath[nb]}, opts];
] /; !ProjectDialogQ[nb]


Paginate[nb_NotebookObject, opts___]:=
Block[{},
  AuthorTools`MakeProject`Private`ProjectDialogFunction["SaveWarning", nb];
  Paginate[ProjectFileLocation[nb], opts]
]/; ProjectDialogQ[nb]


Paginate[projfile_String, opts___] :=
Block[{pd, files},
  {pd, files} =
    {"Directory", "Files"} /. ProjectInformation[projfile];
  
  paginate[ToFileName[{pd}, #]& /@ files, opts]
] /; FileType[projfile] === File



paginate[lis_List, opts___] :=
Module[{spages, openall, pfunc, nb, pages, result, last, num},
  
  {spages, openall, pfunc, num} = 
    {StartingPages, OpenAllCellGroups, PaginationFunction, PaginationNumerals} /. 
    Flatten[{ opts, Options[Paginate] }];
  
  spages = spages //. Inherited -> "Inherited";
  If[spages === "Inherited", spages = {"Inherited", "Inherited"}];
  
  If[Head[spages] =!= List || Length[spages] < 2,
    Message[Paginate::pages];
    Abort[]
  ];
  
  If[!MemberQ[{"RomanNumerals", "CapitalRomanNumerals"}, num], num = Automatic];
  
  spages = PadRight[spages, Length @ lis, Last @ spages];
  
  result = {};
  next = First[spages];
  
  RememberOpenNotebooks[];
  
  Do[
    nb = NotebookOpen[Part[lis, i]];
    
    If[next === "Inherited",
      next = PrintingStartingPageNumber /.
        AbsoluteOptions[nb, PrintingStartingPageNumber]
    ];
        
    SetOptions[nb, PrintingStartingPageNumber -> next];
    
    If[openall,
      SelectionMove[nb, All, Notebook];
      FrontEndExecute[FrontEndToken[nb, "SelectionOpenAllGroups"]]
    ];
    pages = NotebookPageNumbers[nb];
    AppendTo[result, pages];
    refreshPaginationCache[nb, pages, num];
    last = Last @ pages;
    
    pfunc[nb]; 
    NotebookSave[nb];
    NotebookCloseIfNecessary[nb];
    
    If[i === Length[lis], Continue[]];
    
    next = Switch[ spages[[ i+1 ]],
      "Next", last + 1,
      "Even", last + 2 - Mod[last, 2],
      "Odd",  last + 1 + Mod[last, 2],
      "Inherited", "Inherited",
      _Integer, spages[[ i+1 ]]
    ];
    ,
    {i, 1, Length[lis]}
  ];
  result
] /; Union[FileType /@ lis] === {File};



(*
   Magnification at the notebook level can throw off Show Page Breaks,
   so during the pagination, set the magnification to 100%.
*)


NotebookPageNumbers[nb_NotebookObject]:=
Block[{plis, mag},
  mag = Magnification /. Options[nb, Magnification] /. Magnification -> 1.0;
  SetOptions[nb, ShowPageBreaks->True, Magnification -> 1.0];
  plis = Map[
    Page /. Cases[LayoutInformation /. Last[#], _Rule | _RuleDelayed]&,
    MathLink`CallFrontEnd[FrontEnd`NotebookGetLayoutInformationPacket[nb]]
  ];
  SetOptions[nb, ShowPageBreaks->False, Magnification -> mag];
  plis
]



refreshPaginationCache::usage = "refreshPaginationCache[nb, lis] caches a copy of the given list of page numbers in nb's tagging rules.";

refreshPaginationCache[nb_NotebookObject, lis_List, num_]:=
Module[{tRules},
  tRules = TaggingRules /. Options[nb,TaggingRules];
  tRules = If[tRules===None,{},DeleteCases[tRules,_["PaginationCache",_]]];
  SetOptions[nb, TaggingRules ->
        Join[tRules, {"PaginationCache" -> {Date[],lis, num}}]];
  lis
]



(*
   NotebookPaginationCache reads directly from the file if it's called
   with a full path, using NotebookFileOptions. If it's called with
   a notebook object, then it reads the TaggingRules directly from that
   notebook object using Options in the usual way.
*)

NotebookPaginationCache[nb_NotebookObject]:=
  NotebookPaginationCache[Options[nb, TaggingRules]];

NotebookPaginationCache[nbfile_String] := 
Block[{nb, res},
  If[NotebookCacheValidQ[nbfile],
    NotebookPaginationCache[NotebookFileOptions[nbfile]]
    ,
    RememberOpenNotebooks[];
    nb = NotebookOpen[nbfile];
    res = NotebookPaginationCache[nb];
    NotebookCloseIfNecessary[nb];
    res
  ]
]

NotebookPaginationCache[opts_List]:=
Module[{tRules},
  tRules = TaggingRules /. Flatten[{opts, TaggingRules -> None}];
  If[tRules === None, Return[None]];
  tRules = "PaginationCache" /. tRules;
  If[tRules === "PaginationCache", Return[None]];
  tRules
]



End[]

EndPackage[]
