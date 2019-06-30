(* :Context: AuthorTools`StyleTools` *)

(* :Author: Louis J. D'Andria *)

(* :Summary:
    This package defines a command language for 
    stylesheet handling.
*)

(* :Copyright: *)

(* :Package Version: $Revision: 1.5 $ $Date: 2005/10/10 18:33:39 $ *)

(* :Mathematica Version: 6.0 *)

(* :History:

*)

(* :Keywords:
     
*)

(* :Discussion:

  
  Internally, style names are taken to be of the form StyleData[style] or
  StyleData[style, environment]. Some style names have special meaning to the
  front end, like 'All' (not the string "All", but the symbol) and "Notebook".
  
  Also, it turned out to be much easier to use the stylesheet notebook itself
  as the data format for manipulating stylesheets. So the low level functions
  that manipulate stylesheets do so by taking a stylesheet notebook expression
  as input and outputting an altered expression.
  
  So a common use of some utility functions might be something like this:
  
  s = Stylesheet[nb];
  s = RemoveStyles[s, StyleData["MR", ___]];
  s = RemoveEnvironments[s, "EnhancedPrintout"];
  
  In order to use the changes, you'd then have to "activate" the stylesheet
  by setting placing it in some appropriate location and refering a notebook
  to it.
      
*)

(* :Warning:
    
*)



BeginPackage["AuthorTools`StyleTools`", "AuthorTools`Common`"]


StylesheetQ::usage = "StylesheetQ[nb] returns True if the given notebook file, object, or expression contains any StyleData cells, and False otherwise.";

PrivateStylesheetQ::usage = "PrivateStylesheetQ[nbObj] returns True if the given notebook object has an embedded stylesheet.";

Stylesheet::usage = "Stylesheet[nb] returns the stylesheet notebook expression for the given notebook file or object. If nb itself is a stylesheet, its expression is returned.";

PathToStylesheet::usage = "PathToStylesheet[nbObj] returns the full path to the file being used as the stylesheet for the given notebook object. If the path cannot be determined, $Failed is returned; if the notebook has a private stylesheet, None is returned.";

EmbedStylesheet::usage = "EmbedStylesheet[nb] embeds a private copy of the shared stylesheet used by the notebook file or object. If the notebook already uses a private stylesheet, this has no effect."

InstallStylesheet::usage = "InstallStylesheet[nb, name] saves a copy of the given stylesheet to $UserStylesheetDirectory."

$UserStylesheetDirectory::usage = "$UserStylesheetDirectory gives the location for user defined stylesheets.";

StylesheetDirectory::usage = "StylesheetDirectory is an option to InstallStylesheet that determines where the stylesheet will be installed.";

OverwriteStylesheet::usage = "OverwriteStylesheet is an option to InstallStylesheet that determines whether to automatically overwrite a stylesheet with the same name.";




StylesheetCells::usage = "StylesheetCells[nb, sty] returns the cells from Stylesheet[nb] that define the given style or style name pattern. StylesheetCells[nb] returns all the StyleData cells.";

StylesheetStyles::usage = "StylesheetStyles[nb] returns a list of style names defined in the given stylesheet. StylesheetStyles[nb, sty] returns just those style names matching the given style name pattern";

StylesheetEnvironments::usage = "StylesheetEnvironments[nb] returns a list of style environments defined in the given stylesheet. StylesheetEnvironments[nb, sty] returns just those environment names for styles matching the given style pattern.";

StyleOptions::usage = "StyleOptions[nb, sty] returns the list of options set for the given style in the stylesheet.";

SetStyleOptions::usage = "SetStyleOptions[nb, sty, opts] sets the options for a style in the stylesheet, returning an altered stylesheet.";

ClearStyleOptions::usage = "ClearStyleOptions[nb, sty, opts] removes the setting for the given options from a style in the stylesheet, returning an altered stylesheet.";

AddStyles::usage = "AddStyles[nb, cells] adds the given style prototypes to the stylesheet, returning an altered stylesheet.";

AddStylesToSection::usage = "AddStylesToSection is an option to AddStyles that determines in what section new styles are added.";

RemoveStyles::usage = "RemoveStyles[nb, sty] removes the given styles from the stylesheet, returning an altered stylesheet.";

RemoveEnvironments::usage = "RemoveEnvironments[nb, env] removes the given style environment from the indicated stylesheet.";



StylesheetMenuStyles::usage = "StylesheetMenuStyles[nb] returns the list of style names that appear in the Format > Style menu for the given stylesheet.";



StylesheetSections::usage = "StylesheetSections[nb] returns a list of section headins from the given stylesheet.";

StylesheetSectionCells::usage = "StylesheetSectionCells[nb, sec] returns those style prototypes that appear in the given section of the stylesheet.";

StylesheetSectionStyles::usage = "StylesheetSectionStyles[nb, sec] returns those style names whose prototypes appear in the given section of the stylesheet.";

StyleSection::usage = "StyleSection[nb, sty] returns the section in which the given style name is first defined.";

AddStylesheetSection::usage = "Lou!";
RemoveStylesheetSection::usage = "Lou!";


StylesheetCategories::usage = "StylesheetCategories[nb] returns a list of style categories specified by cell tags in the given stylesheet.";

StylesheetCategoryCells::usage = "StylesheetCategoryCells[nb, cat] returns a list of style prototypes for those cells in the given stylesheet containing the specified cell tag.";

StylesheetCategoryStyles::usage = "StylesheetCategoryStyles[nb, cat] returns a list of style names for those cells in the given stylesheet containing the specified cell tag.";

StyleCategories::usage = "";

AddStyleCategories::usage = "";

RemoveStyleCategories::usage = "";


RelatedStyles::usage = "RelatedStyles[nb, sty] returns a list of styles related to the given style.";

AddRelatedStyles::usage = "";

RemoveRelatedStyles::usage = "";



CellInheritanceList::usage = "CellInheritanceList[nb] returns a list of style prototypes that affect the display of the currently selected cell in the given notebook. CellInheritanceList[nb, opt] returns all parts of the inheritance that contain a setting for the given option.";

NotebookInheritanceList::usage = "NotebookInheritanceList[nb] returns a list of style prototypes that affect the display of the given notebook. NotebookInheritanceList[nb, opt] returns all parts of the inheritance that contain a setting for the given option.";

GlobalInheritanceList::usage = "GlobalInheritanceList[] returns a options set at the global, session, and universal levels. GlobalInheritanceList[nb, opt] returns all parts of the inheritance that contain a setting for the given option.";

StyleInheritanceList::usage = "StyleInheritanceList[nb, sty] returns a list of style prototypes that affect the display of the given style in the given notebook. StyleInheritanceList[nb, sty, opt] returns all parts of the inheritance that contain a setting for the given option.";

StylesUsed::usage = "StylesUsed[expr] returns a list of the style names used by the given notebook or cell expressions.";


Begin["`Private`"]





StylesheetQ[nbExpr_Notebook] :=
  Length[Cases[nbExpr, Cell[_StyleData,___], Infinity, 1]] > 0

StylesheetQ[nb_] := Length[
  NotebookLookup[nb, "CellIndex", Cell[__, "StyleData", ___]] ] > 0



PrivateStylesheetQ[nbObj_NotebookObject] :=
  Not[StringQ @ PathToStylesheet[nbObj]] /; $VersionNumber >= 6

PrivateStylesheetQ[nbObj_NotebookObject] :=
  Head[StyleDefinitions /. Options[nbObj]] === Notebook







Stylesheet[nbExpr_Notebook] :=
  If[StylesheetQ[nbExpr], nbExpr, $Failed]


(*
	Andy would like Stylesheet[nbexpr] to work for all notebook expressions, 
	not just the above
*)


Stylesheet[nbObj_NotebookObject] :=
Block[{sty},
  sty = StyleDefinitions /. Options[nbObj] /. StyleDefinitions -> "Default.nb";
  If[Head[sty] === Notebook, sty, stylesheetcache[nbObj, sty]]
]

stylesheetcache[nbObj_, sty_] :=
  stylesheetcache[nbObj, sty] = Get[PathToStylesheet @ nbObj]




PathToStylesheet::nofile = "No file `1` found.";

PathToStylesheet[nbObj_NotebookObject] :=
Block[{sObj, sPath},
  sObj = First["StyleDefinitions" /. NotebookInformation[nbObj]];
  sPath = NotebookFilePath[sObj];
  If[StringQ @ sPath, sPath, None]
] /; $VersionNumber >= 6



PathToStylesheet[nbObj_NotebookObject] := 
Block[{sty, f},
  sty = StyleDefinitions /. Options[nbObj];
  If[Head[sty] === Notebook, Return[None]];
  
  (* John Fultz promises an api to get the full path. As a placeholder,
  this code looks only in SystemFiles/FrontEnd/Stylesheets *)
  
  If[!StringQ[sty], sty = "Default.nb"];
  f=ToFileName[{$InstallationDirectory, "SystemFiles", "FrontEnd", "Stylesheets"}, sty];
  If[FileType[f] === File,
    f,
    Message[Stylesheet::nofile, f];
    $Failed
  ]
]
  






ExtendStylesheetFunction::usage = "ExtendStylesheetFunction[function, nb, args] is a convenience that defines function[nb, args] when nb is a notebook file or object in terms of function[_Notebook, args].";


ExtendStylesheetFunction[function_, nb_String, args___] :=
Module[{nbExpr, path, result},
  path = nb;
  nbExpr = Get[path];
  result = function[nbExpr, args];
  Export[path, nbExpr];
  result
] /; FileType[nb] === File











(*
   Design point: we refer to styles by the "name": StyleData[sty] or
   StyleData[sty, env].
*)


StylesheetCells[nb_] := StylesheetCells[nb, StyleData[___]];

StylesheetCells[nb_, sty_String] := StylesheetCells[nb, StyleData[___, sty, ___]]

StylesheetCells[nb_, pat_StyleData] := StylesheetCells[nb, Cell[pat, ___]]

StylesheetCells[nb_, pat_] := Cases[Stylesheet[nb], pat, Infinity]





(* UnsortedUnion lifted from the Union further examples *)
UnsortedUnion[x_] := Module[{f}, f[y_] := (f[y] = Sequence[]; y); f /@ x]




StylesheetStyles[args___] := First /@ StylesheetCells[args]

StylesheetStyles[nb_] := First /@ StylesheetCells[nb]

StylesheetStyles[nb_, sty_String] := StylesheetStyles[nb, StyleData[___, sty, ___]]

StylesheetStyles[nb_, pat_StyleData] := Cases[First /@ StylesheetCells[nb], pat]


StylesheetEnvironments[nb_, res___] :=
  Cases[StylesheetStyles[nb, res], StyleData[_,x_] :> x] // UnsortedUnion



StylesheetMenuStyles[nb_] := StylesheetCells[nb,
  Cell[StyleData[s_], opts___ /; !MemberQ[{opts}, StyleMenuListing -> None]] :> StyleData[s]]


styleswithoptions[nb_, {opts___}]:= Cases[StylesheetCells[nb],
  Cell[StyleData[s__], ___, _[Alternatives[opts],_], ___] :> StyleData[s]]
styleswithoptions[nb_, opt_] := styleswithoptions[nb, {opts}]

styleswithoutoptions[nb_, {opts___}] :=
  First /@ DeleteCases[StylesheetCells[nb],
    Cell[StyleData[s__], ___, _[Alternatives[opts],_], ___]]
styleswithoutoptions[nb_, opt_] := styleswithoutoptions[nb, {opts}]




StylesheetSections[nb_] :=
  Cases[Stylesheet[nb], Cell[x_,"Section", ___] :> x, Infinity];


StylesheetSectionCells[nb_, sec_] := 
Block[{data},
  data = Cases[
    Stylesheet[nb],
    Cell[CellGroupData[{Cell[sec, "Section", ___], ___},___]],
    Infinity
  ];
  
  Cases[data, Cell[_StyleData,___], Infinity]
]

StylesheetSectionStyles[nb_, sec_] := First /@ StylesheetSectionCells[nb, sec]



StyleSection[nb_, str_String] := StyleSection[nb, StyleData[str]]

StyleSection[nb_, sty_StyleData] := If[# === {}, None, First[#]]& @
  Select[StylesheetSections[nb], MemberQ[StylesheetSectionStyles[nb, #], sty]&, 1]






$CategoryPrefix = "cat:";
$RelatedPrefix = "see:";
$OrderingPrefix = "ord:";


StylesheetCategories[nb_] :=
  getSpecialTags[StylesheetCells[nb], $CategoryPrefix]


StylesheetCategoryCells[nb_, cat_] :=
With[{t = $CategoryPrefix <> cat},
  Cases[StylesheetCells[nb], Cell[_StyleData, ___, CellTags -> (t | {___, t, ___}), ___]]
]


StylesheetCategoryStyles[nb_, cat_] := First /@ StylesheetCategoryCells[nb, cat]


StyleCategories[nb_, sty_StyleData] :=
  getSpecialTags[StylesheetCells[nb, sty], $CategoryPrefix]


RelatedStyles[nb_, sty_StyleData] :=
  getSpecialTags[StylesheetCells[nb, sty], $RelatedPrefix]


AddStyleCategories[nb_, sty_StyleData, cats__String] :=
  addSpecialTags[nb, sty, {cats}, $CategoryPrefix]

RemoveStyleCategories[nb_, sty_StyleData, cats__String] :=
  removeSpecialTags[nb, sty, {cats}, $CategoryPrefix]

AddRelatedStyles[nb_, sty_StyleData, rel__String] :=
  addSpecialTags[nb, sty, {rel}, $RelatedPrefix]

RemoveRelatedStyles[nb_, sty_StyleData, rel__String] :=
  removeSpecialTags[nb, sty, {rel}, $RelatedPrefix]



getSpecialTags[lis_, prefix_] :=
Block[{tags},
  tags = Cases[lis, Cell[_StyleData, ___, CellTags -> t_, ___] :> t];
  tags = Select[UnsortedUnion @ Flatten @ tags, StringMatchQ[#, prefix <> "*"]&];
  If[tags === {}, {}, StringReplace[tags, prefix -> ""]]
]


addSpecialTags[nb_, sty_, {cats__String}, prefix_] :=
Block[{tags},
  tags = CellTags /. StyleOptions[nb, sty, CellTags] /. CellTags -> {};
  tags = UnsortedUnion @ Flatten @ {tags, (prefix <> #)& /@ {cats}};
  SetStyleOptions[nb, sty, CellTags -> tags]
]

removeSpecialTags[nb_, sty_, {cats__String}, prefix_] :=
Block[{tags},
  tags = CellTags /. StyleOptions[nb, sty, CellTags] /. CellTags -> {};
  tags = DeleteCases[tags, Alternatives @@ Map[(prefix <> #)&, {cats}]];
  If[tags === {},
    ClearStyleOptions[nb, sty, CellTags],
    SetStyleOptions[nb, sty, CellTags -> tags]
  ]
]








EmbedStylesheet[nbObj_NotebookObject] :=
Block[{},
  SetOptions[nbObj, StyleDefinitions -> Stylesheet[nbObj]];
  (* Setting Visible to True because of a bug in SetOptions *)
  SetOptions[nbObj, Visible -> True]
]

EmbedStylesheet[nbFile_String] :=
Block[{nbObj, result},
  RememberOpenNotebooks[];
  nbObj = NotebookOpen[nbFile];
  result = EmbedStylesheet[nbObj];
  NotebookSave[nbObj];
  NotebookCloseIfNecessary[nbObj];
  result
] /; FileType[nbFile] === File





$UserStylesheetDirectory = ToFileName[{$UserBaseDirectory, "SystemFiles", "FrontEnd", "Stylesheets"}];

Options[InstallStylesheet] = {
  StylesheetDirectory :> $UserStylesheetDirectory,
  OverwriteStylesheet -> False  
}


InstallStylesheet::exists = "A stylesheet called `1` already exists in `2`.";
InstallStylesheet::nodir = "The directory `1` does not exist. Using $UserStylesheetDirectory instead.";


InstallStylesheet[nb_Notebook, name_String, opts___] := 
Block[{f, dir, nbObj},
  f = If[StringMatchQ[name, "*.nb"], name, name <> ".nb"];
  
  dir = StylesheetDirectory /. Flatten[{opts, Options[InstallStylesheet]}];
  If[!StringQ[dir] || FileType[dir] =!= Directory,
    Message[InstallStylesheet::nodir, dir];
    dir = $UserStylesheetDirectory
  ];
  
  If[FileType[ToFileName[{dir}, f]] === File &&
    Not @ TrueQ @ 
      (OverwriteStylesheet /. Flatten[{opts, Options[InstallStylesheet]}]),
    Message[InstallStylesheet::exists, name, dir];
    $Failed,
    Export[ToFileName[{dir}, f], Stylesheet[nb]]
  ]
]







StylesUsed[nbObj_NotebookObject] :=
  StylesUsed[NotebookGet @ nbObj] /; MemberQ[Notebooks[], nbObj]

StylesUsed[nbFile_String] :=
  StylesUsed[Get @ nbFile] /; FileType[nbFile] === File

StylesUsed[Notebook[c_,___]] := StylesUsed[c]

StylesUsed[expr_] :=
Block[{fmt, result},
  fmt[x_] := If[# === $Failed, {}, #]& @ cellFormatType[x];
  fmt[] := {};

  result = Union @ Flatten @ Cases[{expr},
    (
      (c:Cell[_, s_String, ___]) |
      StyleBox[_, s_String, ___] |
      ButtonBox[___, ButtonStyle -> s_String]
    ) :> {s, fmt @ c}, Infinity];
  
  StyleData /@ result
]




cellFormatType[Cell[BoxData[FormBox[_,fmt_,___]],___]] := ToString[fmt];
cellFormatType[Cell[_BoxData,___]] := "StandardForm";

(* This isn't quite right -- we need info from the Stylesheet to figure this out *)
cellFormatType[Cell[_, "Input", ___]] := "InputForm";

cellFormatType[Cell[GraphicsData[fmt_,___],___]] := ToString[fmt];
cellFormatType[Cell[_OutputFormData,___]] := "OutputForm";
cellFormaTtype[Cell[_RawData,___]] := "CellExpression";
cellFormatType[Cell[_StyleData,___]] := "StyleData";
cellFormatType[Cell[_TextData,___]] := "TextForm";
cellFormatType[Cell[___]] := "TextForm";
cellFormatType[___] := $Failed;


cellOptions[Cell[_,_String,a___]] := {a};
cellOptions[Cell[_,a_Rule,b___]] := {a,b};
cellOptions[Cell[_,a_RuleDelayed,b___]] := {a,b};
cellOptions[___] := {};



CellInheritanceList[nb_NotebookObject] :=
Block[{info, cell, style, contentData, env, formatType, s},
  info = Developer`CellInformation[nb];
  If[Length[info] =!= 1, Return[$Failed]];
  info = First[info];
  
  {style, contentData} = {"Style", "ContentData"} /. info;
  env = ScreenStyleEnvironment /. Options[nb, ScreenStyleEnvironment];
  cell = NotebookRead[nb, "WrapBoxesWithBoxData" -> True];
  formatType = cellFormatType[cell];
  opts = cellOptions[cell];
  
  s = Stylesheet[nb];
  
  Flatten[{
    If[opts === {}, {},
      Cell[StyleData["CellOptions", "-"], Sequence @@ opts]],
    StylesheetCells[s, StyleData[formatType, env]],
    StylesheetCells[s, StyleData[formatType]],
    StylesheetCells[s, StyleData[style, env]],
    StylesheetCells[s, StyleData[style]],
    StylesheetCells[s, StyleData[All, env]],
    StylesheetCells[s, StyleData[All]], (* is this real? *)
    Cell[StyleData["NotebookOptions", "-"], Sequence @@
      DeleteCases[Options[nb], _[StyleDefinitions,_]] ],
    StylesheetCells[s, StyleData["Notebook", env]],
    StylesheetCells[s, StyleData["Notebook"]],
    GlobalInheritanceList[]
  }]
]


CellInheritanceList[nb_NotebookObject, opt_] :=
  triminheritancelist[CellInheritanceList[nb], opt]



NotebookInheritanceList[nb_NotebookObject] :=
Block[{env, s},
  s = Stylesheet[nb];
  env = ScreenStyleEnvironment /. Options[nb, ScreenStyleEnvironment];
  Flatten[{
    StylesheetCells[s, StyleData[All, env]],
    StylesheetCells[s, StyleData[All]], (* is this real? *)
    Cell[StyleData["NotebookOptions", "-"], Sequence @@
      DeleteCases[Options[nb], _[StyleDefinitions,_]] ],
    StylesheetCells[s, StyleData["Notebook", env]],
    StylesheetCells[s, StyleData["Notebook"]],
    GlobalInheritanceList[]
  }]
]

NotebookInheritanceList[nb_NotebookObject, opt_] :=
  triminheritancelist[NotebookInheritanceList[nb], opt]


GlobalInheritanceList[] :=
{
  Cell[StyleData["$FrontEnd", "-"], Sequence @@
    DeleteCases[Options[$FrontEnd], _[StyleDefinitions,_]] ],
  Cell[StyleData["$FrontEndSession", "-"], Sequence @@
    DeleteCases[Options[$FrontEndSession], _[StyleDefinitions,_]] ],
  Cell[StyleData["Universal", "-"]]
}

GlobalInheritanceList[opt_] :=
DeleteCases[{
    Cell[StyleData["$FrontEnd", "-"], Sequence @@ Options[$FrontEnd, opt] ],
    Cell[StyleData["$FrontEndSession", "-"], Sequence @@ fesessionopt[val] ]
  },
  Cell[_]
]

fesessionopt[opt_] := MathLink`CallFrontEnd[FrontEnd`Options[FrontEnd`$FrontEndSession, opt]]



StyleInheritanceList[nb_, StyleData[style_, y___]] :=
Block[{env, s},
  If[MatchQ[{y}, {_String}], env = y, env = None];
  
  s = Stylesheet[nb];
  
  Flatten[{
    If[env === None, {}, StylesheetCells[s, StyleData[style, env]]],
    StylesheetCells[s, StyleData[style]],
    If[env === None, {}, StylesheetCells[s, StyleData[All, env]]],
    StylesheetCells[s, StyleData[All]], (* is this real? *)
    Cell[StyleData["NotebookOptions", "-"], Sequence @@
      DeleteCases[Options[nb], _[StyleDefinitions,_]] ],
    If[env === None, {}, StylesheetCells[s, StyleData["Notebook", env]]],
    StylesheetCells[s, StyleData["Notebook"]],
    GlobalInheritanceList[]
  }]
]

StyleInheritanceList[nb_, StyleData[x__], opt_] :=
  triminheritancelist[StyleInheritanceList[nb, StyleData[x]], opt]



triminheritancelist[lis_, opt_] :=
Block[{lis2},
  lis2 = Join[Drop[lis, -3], GlobalInheritanceList[opt]];
  Cases[lis2, Cell[s_, ___, r:(_[opt,_]), ___] :> Cell[s,r]]
]




AddStyles::exists = "The style `1` already exists.";

Options[AddStyles] = {AddStylesToSection -> "New Styles"}


(*
   Should AddStyles be able to add styles in a particular category
   too, or just section?
*)

AddStyles[nb_, c:Cell[StyleData[sty_, env___], ___], opts___] := 
Block[{s, sec, pos},
  s = Stylesheet[nb];
  sec = AddStylesToSection /. Flatten[{opts, Options[AddStyles]}];
  
  (* if the style already exists, do nothing *)
  If[MemberQ[StylesheetStyles[s], StyleData[sty, env]],
    Message[AddStyles::exists, StyleData[sty, env]];
    Return[s]
  ];
  
  (* if the style exists in other environments, put this with those *)
  pos = Position[s, Cell[StyleData[sty,___],___]];
  If[pos =!= {},
    Return @ Insert[s, c, MapAt[#+1&, Last @ pos, -1]]
  ];
  
  (* if there's no match, add it in the right section *)
  If[MemberQ[StylesheetSections[s], sec],
    s /. Cell[CellGroupData[{Cell[sec,"Section",a___], b___},open_]] :>
         Cell[CellGroupData[{Cell[sec, "Section", a], b, Cell[StyleData[sty]],c}, open]],
    Insert[s, Cell[CellGroupData[{Cell[sec, "Section"], c}, Closed]], {1,-1}]
  ]
]


AddStyles[nb_, {c__Cell}, opts___] := Fold[AddStyles[##, opts]&, nb, {c}]



RemoveStyles[nb_, str_String] := RemoveStyles[nb, StyleData[str]];

RemoveStyles[nb_, sty_StyleData] := RemoveStyles[nb, Cell[sty, ___]];

RemoveStyles[nb_, c:Cell[_StyleData,___]] := DeleteCases[Stylesheet[nb], c, Infinity]

RemoveStyles[nb_, lis_List] := Fold[RemoveStyles, nb, lis]



RemoveEnvironments[nb_, env_String] := RemoveEnvironments[nb, {env}]

RemoveEnvironments[nb_, {envs__String}] := DeleteCases[
  Stylesheet[nb],
  Cell[StyleData[_, Alternatives @@ {envs}], ___],
  Infinity
]



(* what about adding an environment to all styles? *)






StyleOptions[nb_, str_String, res___] :=
  StyleOptions[nb, StyleData[str], res]

StyleOptions[nb_, sty_StyleData, {opts__}] :=
  Cases[StyleOptions[nb, sty], _[Alternatives[opts],_]]

StyleOptions[nb_, sty_StyleData, opt_] :=
  StyleOptions[nb, sty, {opt}]

StyleOptions[nb_, sty_StyleData] := 
Block[{cells},
  cells = StylesheetCells[nb, sty];
  If[cells === {}, {}, List @@ Rest[First @ cells] ]
]


SetStyleOptions::nosty = "The style `1` does not exist in the given stylesheet; use AddStyles to add the style.";

SetStyleOptions::mult = "The style `1` is defined multiple times in the given stylesheet. Aborting attempt.";

SetStyleOptions[nb_, str_String, res___] :=
  SetStyleOptions[nb, StyleData[str], res]

SetStyleOptions[nb_, sty_StyleData, opts___] :=
Block[{s, pos, cell},

  s = Stylesheet[nb];
  If[!MemberQ[StylesheetStyles[s], sty],
    Message[SetStyleOptions::nosty, sty];
    Return[s]
  ];
  
  pos = Position[s, Cell[sty,___]];
  If[Length[pos] =!= 1,
    Message[SetStyleOptions::mult, sty];
    Return[s]
  ];
  
  cell = s[[ ## ]]& @@ First[pos];
  cell = Fold[resetopt, cell, Flatten[{opts}]];
  ReplacePart[s, cell, pos]
]  


resetopt[Cell[a___, _[opt_, _], b___], r:(_[opt_, _])] := Cell[a, r, b]
resetopt[Cell[a___], r_] := Cell[a, r]



ClearStyleOptions[nb_, sty_StyleData, opts___] :=
If[{opts} === {All},
  Stylesheet[nb] //. Cell[sty,___] :> Sequence[],
  Stylesheet[nb] //.
    Cell[sty,x___, _[Alternatives[opts], _], y___] :> Cell[sty, x, y]
]








(*--------*)
(* IDEAS  *)
(*--------*)


(*

   The following should either apply to stylesheet files or expressions --
   do we need a better way to refer to stylesheet files versus
   embedded stylesheets versus stylesheet expressions?
   
   
   Perhaps the best thing to do is have functions that apply to the
   expressions, which are the easiest things to deal with, and then
   have a utility that can extend an expression-handling function to a
   file-handling function, or even an embedded-style-sheet-handling
   function? We do this elsewhere in AuthorTools to extend single
   notebook functions to project functions...
   
   Note that there would be some functions (like StyleOptions) that
   would only make sense at a higher level and not at the level of
   handling the notebook expression (where the corresponding function
   would be StylesheetCells or the like).
   
   

*)



DuplicateStyle::usage = "DuplicateStyle[nb, sty1, sty2] makes a duplicate of sty1 and names it sty2.";



StyleCategory::usage = "StyleCategory is an option for AddStyles that indicates the string name of the category into which the style should be added. The default value is \"Added Styles\"";


ReorderStylesheet::usage = "ReorderStylesheet[nb, {sty}] moves the given style prototypes in the given stylesheet to the top of the stylesheet, keeping the others in the same relative order.";


MergeStylesheets::usage = "combines two stylesheets, merging duplicate entries";


SplitStylesheet::usage = "split a stylesheet into two or more based on environment or style names";



SetStyleSubOptions::usage = "";
ClearStyleSubOptions::usage = "";


ValidateStylesheet::usage = "Check for duplicate style prototypes, duplicate option settings, style names that are the same as environment names, styles that are defined only in some environment (ie, there's a StyleData[sty, env] without a corresponding StyleData[sty]), styles that are identical but for their style names, ...";

ValidateNotebookStyles::usage = "ValidateNotebookStyles[nb] returns a list of style name strings used in the given notebook but which are not defined by the notebook's stylesheet.";

NotebookStyles::usage = "NotebookStyles[nb] returns a list of style name strings used by the given notebook file, object, or expression.";


AddEnvironment::usage = "AddEnvironment[nb, {env}] adds the given enironment prototypes to the notebook's stylesheet. AddEnviroment[nb, {env}, {sty}] adds the environment only for the indicated styles."

AddSection;
RemoveSection;





End[]

EndPackage[]
