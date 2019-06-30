(* :Context: AuthorTools`ExportNotebook` *)

(* :Author: Robby Villegas
            Paul Hinton
            Louis J. D'Andria *)

(* :Summary:
    This package defines functions for extracting images out
    of notebooks in various formats.
*)

(* :Copyright: *)

(* :Package Version: $Revision: 1.32 $, $Date: 2017/01/11 22:09:43 $ *)

(* :Mathematica Version: 4.2 *)

(* :History:
    This started as a couple packages written by either
    Robby Villegas or Paul Hinton for extracting EPS or
    EPS with preview from notebooks.
    
    Lou D'Andria combined the packages into a coherent
    whole, and added the capability to add image formats
    easily.
    
    Lou D'Andria rearranged the syntax to make tokens a
    more flexible means of input.
    
*)

(* :Keywords:
     
*)

(* :Discussion:
    
*)

(* :Warning:
    
*)

BeginPackage["AuthorTools`ExportNotebook`", "AuthorTools`Common`"]


(*

ExportNotebook[
  nbObj | nbFile
  ,
  
  style_String
  {"CellStyle", All | "All" | style_String}
  {"CellGroup", headstyle_String}
  {"CellTags", tag_String}
  {"ContentData", datatype_String} (*"BoxData", "GraphicsData", ...*)
  ,
  
  ExtractionMethod ->
    "NotebookGet" | "NotebookRead" | "NotebookLookup" | "Get" | Automatic
  ,
  
  ExportDirectory -> Automatic | "path:to:dir"  
]

*)



ExportNotebook::usage = "ExportNotebook[nb, \"style\", \"fmt\"] creates image files of format fmt in the notebook's directory for each cell of the indicated style. ExportNotebook[nb, {elem, data}, \"fmt\"] extracts cells See also: ExtractionMethod.";

ExtractionMethod::usage = "ExtractionMethod is an option for ExportNotebook that determines how the information is read from the notebook. The default setting of Automatic will use \"NotebookGet\" or \"Get\". Settings of \"NotebookGet\" or \"Get\" are oftentimes more time-efficient, and \"NotebookRead\" or \"NotebookLookup\" are more memory-efficient. The memory-efficient methods are not available for all exports.";

ExportDirectory::usage = "ExportDirectory is an option for ExportNotebook that determines the location of the generated files. The default value of Automatic uses the directory containing the given notebook.";

ExportNotebookDriver::usage = "ExportNotebookDriver is an internal function.";

ExportFormat::usage = "ExportFormat is an option that determines the file format for files created by the AuthorTools export palette.";

Begin["`Private`"]


(********* graphics functions ********)

(*
   to add support for a new graphics format, all you
   need to do is add a customDisplay function definition
   that tells how to write a file with the given format.
   If no customDisplay definition exists for the given
   format, $DefaultExportFunction is used.
*)


$DefaultExportFunction = Export;


customDisplayFunction[channel_String, args___] := 
(
  ProgressDialogSetSubcaption[$ProgressDialog, channel];
  customDisplay[channel, args];
  channel
)


customDisplay[
  channel_String, nbexpr:(_Cell | _Notebook | _BoxData), "EPS"] :=
Block[{st},
  st = OpenWrite[channel];
  appendEPS[st, nbexpr];
  Close[st];
  channel
]

customDisplay[
  channel_String, nbexpr:(_Cell | _Notebook | _BoxData), "GIF"] :=
(
  $DefaultExportFunction[channel, nbexpr, "GIF"];
  channel
)

customDisplay[
  channel_String, nbexpr_Cell, fmt:("Notebook" | "Notebooks")] :=
  customDisplay[channel, Notebook[{nbexpr}], fmt];

customDisplay[
  channel_String, nbexpr:{__Cell}, fmt:("Notebook" | "Notebooks")] :=
  customDisplay[channel, Notebook[nbexpr], fmt];


customDisplay[
  channel_String, nbexpr_BoxData, fmt:("Notebook" | "Notebooks")] :=
  customDisplay[channel, Notebook[{Cell[nbexpr, "Input"]}], fmt];

customDisplay[
  channel_String, nbexpr_Notebook, fmt:("Notebook" | "Notebooks")] :=
(
  Export[channel, nbexpr, "NB"];
  channel
)



customDisplay[
  channel_String, nbexpr:(_Cell | _Notebook | _BoxData), x_String] :=
(
  $DefaultExportFunction[channel, nbexpr, x];
  channel
)



(*
   The appendMGF utility function doesn't seem to be being used, but
   I'll leave it in here as is for now.
*)

appendMGF[stream_OutputStream, nbexpr:(_BoxData | _Cell | _Notebook)] :=
Block[{mgf},
  mgf = MathLink`CallFrontEnd[FrontEnd`ConvertToBitmapPacket[nbexpr]];
  If[mgf === $Failed, Return[$Failed]];
  WriteString[stream, First @ mgf];
]


appendEPS[stream_OutputStream, nbexpr:(_BoxData | _Cell | _Notebook)] :=
Block[{ps},
  ps = ConvertToPostScript[nbexpr];
  If[ps === $Failed || Head[ps] === ConvertToPostScript, Return[$Failed]];
  writeHeaderAndEPS[ stream, getEPSandBoundingBox[ps]]
]


EPSDimensions[nbexpr:(_BoxData | _Cell | _Notebook)] :=
Block[{ps},
  ps = ConvertToPostScript[nbexpr];
  If[ps === $Failed || Head[ps] === ConvertToPostScript, Return[$Failed]];
  Rest @ getEPSandBoundingBox @ ps
]

getEPSandBoundingBox[{epsString_String,
  {{xmin_, ymin_}, {xmax_, ymax_}}, baseline_}] :=
  {epsString, {{xmin, ymin}, {xmax, ymax}}, baseline}

getEPSandBoundingBox[{epsString_String,
  baseline_, {{xmin_, ymin_}, {xmax_, ymax_}}}] :=
  {epsString, {{xmin, ymin}, {xmax, ymax}}, baseline}

writeHeaderAndEPS[stream_, 
  {epsString_String, {{xmin_, ymin_}, {xmax_, ymax_}}, baseline_}] :=
(
  WriteString[stream,
    "%!PS-Adobe-2.0 EPSF-2.0\n",
    "%%BoundingBox: ", ToString[xmin], " ", ToString[ymin], " ", 
    ToString[xmax], " ", ToString[ymax], " ", baseline, "\n",
    epsString
  ]
);




(********* end graphics functions ********)



nextFileName[name_, ext_] := 
Module[{temp, c=0},
  While[FileType[temp = ToFileName[{Directory[]},
    name <> "_" <> ToString[++c] <> ext]]===File];
  temp
]




Options[ExportNotebook] = 
{
  ExtractionMethod -> Automatic,
  ExportDirectory -> Automatic
};


ExportNotebook[a_, style_String, b___] :=
  ExportNotebook[a, {"CellStyle", style}, b]

ExportNotebook[a_, {b_, All}, c___] :=
  ExportNotebook[a, {b, "All"}, c]

ExportNotebook[a_, {"CellStyle"}, c___] := ExportNotebook[a,
  {"CellStyle", InputString @ $Resource["Export", "Pick a style"]}, c];

ExportNotebook[a_, {"CellGroup"}, c___] := ExportNotebook[a,
  {"CellGroup", InputString @ $Resource["Export", "Pick a style"]}, c];

ExportNotebook[a_, {"CellTags"}, c___] := ExportNotebook[a,
  {"CellTags", InputString @ $Resource["Export", "Pick a tag"]}, c];

(*
  If someone cancels out of the InputString input of a custom
  CellStyle etc, InputString will return an empty string. In that
  case, simply end quietly.
*)

ExportNotebook[a_, {b_, ""}, c___] := Null


ExportNotebook::noelem = "`1` is not a recognized element specification."
ExportNotebook::nonb = "The specified notebook object, `1`,  does not exist.";
ExportNotebook::nofile = "The specified notebook file, `1`,  does not exist.";
ExportNotebook::nodir = "The specified export directory,`1`, does not exist.";
ExportNotebook::invmeth = "The `1` method cannot handle `2` export. Switching to `3` method.";
ExportNotebook::noexp = "No cells were exported. Please check your notebook against your export criteria and try again.";
ExportNotebook::done = "Export complete. You can find the `1` extracted files in `2`";



(* 
  All of the error checking is done in the following function, so
  that no error checking need be done in the 'exporter' function
  and its subsidiaries.
*)

ExportNotebook[nb_, {elem_String, data_String}, fmt_String, opts___]:=
Module[{method, dir, newmethod, fname},
  {method, dir} = {ExtractionMethod, ExportDirectory} /. 
    Flatten[{opts, Options[ExportNotebook]}];
  
  If[ !MemberQ[{"CellStyle", "CellGroup", "CellTags", "ContentData"}, elem],
    Message[ExportNotebook::noelem, elem];
    Return[$Failed]
  ];
  
  If[ Head[nb] === NotebookObject && !MemberQ[Notebooks[], nb],
    Message[ExportNotebook::nonb, nb];
    Return[$Failed]     
  ];
  
  If[ Head[nb] === String && FileType[nb] =!= File,
    Message[ExportNotebook::nofile, nb];
    Return[$Failed]
  ];
  
  If[method === Automatic,
    If[ Head[nb] === String, method = "Get", method = "NotebookGet"]
  ];
  
  newmethod = method;
  If[ Head[nb] === NotebookObject &&
      MemberQ[{"Get", "NotebookLookup"}, method],
    newmethod = If[method === "Get", "NotebookGet", "NotebookRead"];
    Message[ExportNotebook::invmeth, method, "notebook object", newmethod]
  ];
  method = newmethod;
  
  newmethod = method;
  If[ Head[nb] === String &&
      MemberQ[{"NotebookGet", "NotebookRead"}, method],
    newmethod = If[method === "NotebookGet", "Get", "NotebookLookup"];
    Message[ExportNotebook::invmeth, method, "file", newmethod]
  ];
  method = newmethod;
  
  newmethod = method;
  If[ MemberQ[{"CellGroup", "ContentData"}, elem] &&
      MemberQ[{"NotebookRead", "NotebookLookup"}, method],
    newmethod = If[method === "NotebookRead", "NotebookGet", "Get"];
    Message[ExportNotebook::invmeth, method, elem, newmethod]
  ];
  method = newmethod;
  
  If[dir === Automatic, Switch[Head[nb],
    NotebookObject, dir = DirectoryName[NotebookFilePath[nb]],
    String, dir = DirectoryName[nb]
  ]];
  
  If[FileType[dir] =!= Directory,
    Message[ExportNotebook::nodir, dir];
    Return[$Failed]
  ];
  
  fname = If[ Head[nb] === String,
    StringReplace[nb, {dir -> "", $PathnameSeparator -> "", ".nb" -> ""}],
    StringReplace[NotebookName[nb], {".nb" -> ""}]
  ];
  
  exporter[nb, {elem, data}, fmt, method, dir, fname]
]




(**** Get and NotebookGet methods ****)

cellList[nb_, "NotebookGet"] := FlattenCellGroups @ First @ NotebookGet[nb];
cellList[nb_, "Get"] := FlattenCellGroups @ First @ Get[nb];



exporter[nb_, {"CellStyle", "All"}, fmt_, method:("NotebookGet" | "Get"), dir_, fname_] :=
  exportList[cellList[nb,method], fmt, dir, fname]

exporter[nb_, {"CellStyle", s_}, fmt_, method:("NotebookGet" | "Get"), dir_, fname_] :=
  exportList[Cases[cellList[nb,method], Cell[_, s, ___]],fmt, dir, fname]

exporter[nb_, {"CellTags", s_}, fmt_, method:("NotebookGet" | "Get"), dir_, fname_] :=
  exportList[
    Cases[cellList[nb,method], Cell[___,CellTags -> (s | {___,s,___}),___]],
    fmt, dir, fname
  ]

exporter[nb_, {"ContentData", s_}, fmt_, method:("NotebookGet" | "Get"), dir_, fname_] :=
  exportList[
    Cases[cellList[nb,method], Cell[ToExpression[s][___],___]],
    fmt, dir, fname
  ]

exporter[nb_, {"CellGroup", s_}, fmt_, method:("NotebookGet" | "Get"), dir_, fname_] :=
  exportList[
    Cases[First @ ToExpression[method] @ nb,
      Cell[CellGroupData[{Cell[_,s,___],___},___],___],
      Infinity],
    fmt, dir, fname
  ]



exportList[lis_List, fmt_, dir_, fname_]:=
Module[{ext, result, newlis},
  $ProgressDialog = ProgressDialog[$Resource["Export", "Extracting..."], ""];
  SetDirectory[dir];
  
  ext = Switch[fmt,
    "Notebook", ".nb",
    "Notebooks", ".nb",
    _, "." <> ToLowerCase[fmt]
  ];
  

  (* If you want all the cells in a single notebook, bury the 
     cell list in a sublist, so the whole list of cells is taken
     to be a single entity.
  *)
  newlis = lis;
  If[fmt === "Notebook" && lis =!= {}, newlis = {newlis}];
  
  result = Table[
    customDisplayFunction[
      nextFileName[fname, ext],
      Part[newlis, i],
      fmt],
    {i, 1, Length @ newlis}
  ];
  
  ProgressDialogClose[$ProgressDialog];
  ResetDirectory[];
  
  exportComplete[Length @ result, dir];
  
  result
]
  


(*
   exportComplete only need present a message when the user has
   run the tool through the palette interface. The return value
   from the function is enough for the command-line user - they
   need not see these messages.
*)

exportComplete[0, dir_] :=
If[ButtonNotebook[] =!= $Failed,
  messageDialog[ExportNotebook::noexp]
];


exportComplete[n_, dir_]:=
If[ButtonNotebook[] =!= $Failed,
  messageDialog[ExportNotebook::done, n, dir]
];





(***** NotebookRead method *****)


(*
  The NotebookRead method can handle CellStyle and CellTags
  settings -- anything else would have been routed through
  Get/NotebookGet. These two situations can be easily dealt with
  via NotebookFind, so that's what we do here.
*)

exporter[nb_, {elem_, data_}, fmt_, "NotebookRead", dir_, fname_] :=
Module[{ext, result, optsSetting, tmp, direction},
  $ProgressDialog = ProgressDialog[$Resource["Export", "Extracting..."], ""];
  SetDirectory[dir];
  
  ext = Switch[fmt,
    "Notebook", ".nb",
    "Notebooks", ".nb",
    _, "." <> ToLowerCase[fmt]
  ];
  
  
  optsSetting = FindSettings /. AbsoluteOptions[$FrontEnd, FindSettings];
  SetOptions[$FrontEnd, FindSettings -> {
    "Wraparound" -> False, "IgnoreCase" -> False}];
    
  SelectionMove[nb, Before, Notebook];
  result = {};
  
  
  (*
    If the user wants a single notebook with the results, then
    the first time through the While loop, look for all matches.
    After that, looking for the next match will fail, and you'll
    exit the loop.
  *)
  If[fmt === "Notebook", direction = All, direction = Next];
  
  While[True,
    tmp=NotebookFind[nb, data, direction, ToExpression @ elem];
    If[tmp === $Failed, Break[]];
    tmp = NotebookRead[nb, "WrapBoxesWithBoxData" -> True];
    AppendTo[result,
      customDisplayFunction[ nextFileName[fname, ext], tmp, fmt]];
    SelectionMove[nb, After, Cell];
    direction = Next;
  ];
  
  SetOptions[$FrontEnd, FindSettings -> optsSetting];
  ProgressDialogClose[$ProgressDialog];
  ResetDirectory[];

  exportComplete[Length @ result, dir];
  
  result
]



(***** NotebookLookup method *****)


(*
  Like NotebookRead, the NotebookLookup method can handle
  CellStyle and CellTags settings -- anything else would have
  been routed through Get/NotebookGet.
*)

exporter[nb_, {elem_, data_}, fmt_, "NotebookLookup", dir_, fname_] :=
Module[{ext, result, tmp, cellPattern, cellOutlines},
  $ProgressDialog = ProgressDialog[$Resource["Export", "Extracting..."], ""];
  SetDirectory[dir];
  
  ext = Switch[fmt,
    "Notebook", ".nb",
    "Notebooks", ".nb",
    _, "." <> ToLowerCase[fmt]
  ];
  
  cellPattern = Switch[elem,
    "CellStyle", Cell[___, data, ___],
    "CellTags", Cell[___, CellTags -> (data | {___,data,___}),___]
  ];
  
  (* 
    If you want all the cells in a single notebook, look them up
    all at once with the cell pattern. Otherwise, walk through
    the cell outlines one by one.
  *)
  If[fmt === "Notebook",
    cellOutlines = {cellPattern},
    cellOutlines = NotebookLookup[nb, "CellOutline", cellPattern]
  ];
    
  result = Table[
    tmp = NotebookLookup[nb, "CellExpression", Part[cellOutlines,i]];    
    (* The extra List wrapper is confusing some of the exporters. *)
    If[MatchQ[tmp,{_Cell}], tmp = First @ tmp];
    customDisplayFunction[nextFileName[fname, ext], tmp, fmt],
    {i, 1, Length @ cellOutlines}
  ];
  
  ProgressDialogClose[$ProgressDialog];
  ResetDirectory[];

  exportComplete[Length @ result, dir];
  
  result
]




(* Palette code *)

ExportNotebookDriver[nbObj_, data_, opts___]:=
Block[{fmt},
  fmt = ExportFormat /. Flatten[{opts, Options[ExportNotebookDriver]}];
  ExportNotebook[nbObj, data, fmt, opts]
]

Options[ExportNotebookDriver] = {ExportFormat -> "Notebook"}

optionValues[ExportFormat] =
  Join[{"Notebook", "Notebooks"}, $ExportFormats]

optionValues[ExportDirectory] = {Automatic, "Browse..."}

(* 
   Note that optionValues is only relevant for the palette. Since
   the palette only handles notebook objects and not notebook
   files, we can get by not listing the Get or NotebookLookup
   methods in the palette.
*)

optionValues[ExtractionMethod] =
  {Automatic, "NotebookGet", "NotebookRead"}


End[]

EndPackage[]
