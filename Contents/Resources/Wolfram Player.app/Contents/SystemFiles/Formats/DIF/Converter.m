(* ::Package:: *)

(* ::Subsubtitle:: *)
(*Data Interchange Format (DIF) Converter*)


(* ::Section::Closed:: *)
(*COPYRIGHT*)


(*************************************************************************

                        Mathematica source file

        Copyright 1986 through 2010 by Wolfram Research Inc.

This material contains trade secrets and may be registered with the
U.S. Copyright Office as an unpublished work, pursuant to Title 17,
U.S. Code, Section 408.  Unauthorized copying, adaptation, distribution
or display is prohibited.

*************************************************************************)


(* ::Section::Closed:: *)
(*BEGIN CONVERTER CONTEXT*)


Begin["System`Convert`DIFDump`"];


(* ::Section::Closed:: *)
(*IMPORT*)


(* ::Subsection::Closed:: *)
(*Utilities*)


ReadVector[list_List] :=
Module[
  {data, delim, records,processStrings, processNumbers, numberRegEx},
  
  (* Remove Record Indicators *)
  data = Take[list, {3, Length@list - 1}];
  
  (* Split Data Into Records *)
  delim = {#[[1]], #[[2]] - 1} & /@ 
    Partition[
     Flatten@{Position[data, 
        x_ /; x == "1,0" || StringMatchQ[ToString@x, RegularExpression["0,.+"]]], 
       Length@data + 1}, 2, 1, {1, -1}];
  records = Take[data, #] & /@ delim;

  (*Record Parser Handelers*)
  (*handle the quote wrapers and the quote quote escape for one quote*)
  processStrings =
    StringReplace[(StringJoin@Rest@#), {"\"\"" -> "\"","\""->"",
    {StartOfString~~"\"","\""~~EndOfString}->""}]&;
    
  (*was StringSplit on comma, now strip the marker only*)
  processNumbers = ToExpression@StringDrop[First@#, 2]&;

  (*RegEx replaced with the StringPattern "0,"~~NumberString *)
  numberRegEx = RegularExpression["0,([+-]?\\d+(\\.\\d*)?|[+-]?\\d*\\.(\\d+))"];
  
  (* Return List of Records *)
  Switch[#,
    {marker_/;StringMatchQ[marker, "0,"~~NumberString],"V"}, 
    processNumbers@# (*numbers in the DIF spec do not support,eE*^:/*),
    
    {_,"V"}, StringJoin@StringDrop[First@#, 2](*NumberString match failed as String*), 
    {"0,0","NA"}, None (*value NA support using Symbol None*),
    
    {"0,0","ERROR"},
    Null (*value ERROR support using Symbol Null*),
    
    {"0,1","TRUE"},
    True (*value TRUE support using Symbol True*),
    
    {"0,0","FALSE"},
    False (*value FALSE support using Symbol False*),
    
    {"1,0","\"\""},
    "" (*catching nullsrting *),
    
    {"1,0","\"\"\"\""},
    "\"" (*the quotes around double quote get quoted too*),
    
    {"1,0",__},
    processStrings@#
  ] &/@ records
]


(* ::Subsection:: *)
(*Importers*)


(* ::Subsubsection::Closed:: *)
(*Conditional Raw Importers*)


(* There are no DIF conditional raw importers *)


(* ::Subsubsection::Closed:: *)
(*Default Raw Importer*)


ImportDIF[strm_InputStream, opts___] :=
Module[
  {file, filepattern, sectionmarkers, markerlocations,
  header, data, delimiter, vectors},
  
  (* Split File Between Header and Data *)
  file = Import[strm, "Lines", CharacterEncoding->"ASCII"];
  If[ Length@file < 10,
      Message[Import::fmterr, "DIF"];
      Return[$Failed]
  ];
   
  (* strong pattern test before looking for the "DATA" position *)
  filepattern = Take[file,10];
   
  sectionmarkers =  StringPosition[ filepattern, {
     StartOfString ~~ "TABLE"~~EndOfString, 
     StartOfString ~~ "VECTORS"~~EndOfString, 
     StartOfString ~~ "TUPLES"~~EndOfString, 
     StartOfString ~~ "DATA"~~EndOfString
     }];
                
  markerlocations = sectionmarkers[[{1, 4, 7, 10}, All, 1]];
                
  If[ markerlocations === {{1},{1},{1},{1}},
      delimiter = First @@ Position[file, "DATA"] + 2;
      If[ delimiter == {},
          Message[Import::fmterr, "DIF"];
          Return[$Failed]
      ];
      ,
      Message[Import::fmterr, "DIF"]; 
      Return[$Failed]
  ];
  
  header = Take[file, delimiter];
  data = Drop[file, delimiter];
  
  (* Split Data into Vectors *)
  vectors = Take[data, #] & /@ Partition[Flatten@Position[data, "-1,0"], 2, 1, {1, -1}];
    
  (* Return the Data element *)
  {"Data" -> ReadVector /@ vectors}
]


(* ::Subsubsection::Closed:: *)
(*Post Importers*)


ImportDIFGrid[difDump_List, opts___]:=Grid["Data"/.difDump]


ImportDIFTableView[difDump_List, opts___]:=TableView["Data"/.difDump]


(* ::Section:: *)
(*EXPORT*)


(* ::Subsection::Closed:: *)
(*Utilities*)


RaggedMatrixQ[mat_] := MatchQ[mat, {__List}];


WriteVector[strm_OutputStream, l_List, len_Integer] :=
Module[
  {data, processStrings},
  
  (* Pad List if Necessary *)
  If[Length@l != len, data = PadRight[l, len, ""], data = l];
  
  (* Print Vector Initializer *)
  WriteString[strm, "-1,0\nBOT\n"];


  (**** 
     None supports ERROR, Null supports NA
     True supports TRUE, False supports FALSE
     string "True" "False" stay string values
     string "ERROR" "NA" stay string values
  *****)
  processStrings = StringReplace[
    ToString[#/. {None->"ERROR",Null->"NA",True->"TRUE",False->"FALSE"}], {"\""->"\"\""}]&;

  (* Write Rest of Data *)

  Switch[#,
    _?NumericQ, 
    WriteString[strm, "0,", CForm@#, "\nV\n"],

    Null|None|False, 
    WriteString[strm, "0,0\n",processStrings@#, "\n"],

    True, 
    WriteString[strm, "0,1\n", processStrings@#, "\n"],

    "", 
    WriteString[strm, "1,0\n\"", processStrings@#, "\"\n"],

    _, 
    WriteString[strm, "1,0\n\"", processStrings@#, "\"\n"]
   ] & /@ data;
]



ExportData[strm_OutputStream, data_?RaggedMatrixQ, opts___?OptionQ] :=
Module[
  {numrows, numcols, sig},
  
  (* Find Number of Rows and Cols*)
  numrows = Length@data;
  numcols = Max[Length /@ data];
  sig ="Title" /. Flatten@{opts} /. "Title"-> Automatic /. Automatic->System`ConvertersDump`Utilities`$signature;

  (* Write Header *)
  WriteString[strm, "TABLE\n0,1\n"];
  WriteString[strm, "\""<>sig<>"\"\n"];
  WriteString[strm, "VECTORS\n0,", numrows, "\n\"\"\n"];
  WriteString[strm, "TUPLES\n0,", numcols, "\n\"\"\n"];
  WriteString[strm, "DATA\n0,0\n\"\"\n"];
  
  (* Write Data *)
  WriteVector[strm, #, numcols] & /@ data;
  WriteString[strm, "-1,0\nEOD\n"];
]


ExportData[strm_OutputStream, {}, opts___?OptionQ] := 
ExportData[strm, {{}}, opts]

ExportData[strm_OutputStream, data_List, opts___?OptionQ] := 
ExportData[strm, List /@ data, opts]

ExportData[strm_OutputStream, data_SparseArray, opts___?OptionQ] := 
ExportData[strm, Normal@data, opts]

ExportData[strm_OutputStream, Grid[data_, ___], opts___?OptionQ] := 
ExportData[strm, data, opts]

ExportData[strm_OutputStream, MatrixForm[data_, ___], opts___?OptionQ] := 
ExportData[strm, data, opts]

ExportData[strm_OutputStream, TableForm[data_, ___], opts___?OptionQ] := 
ExportData[strm, data, opts]

ExportData[strm_OutputStream, Row[data_, ___], opts___?OptionQ] := 
ExportData[strm, {data}, opts]

ExportData[strm_OutputStream, Column[data_, ___], opts___?OptionQ] := 
ExportData[strm, List /@ data, opts]

ExportData[strm_OutputStream, ColumnForm[data_, ___], opts___?OptionQ] := 
ExportData[strm, List /@ data, opts]

ExportData[strm_OutputStream, data_, opts___?OptionQ] := 
ExportData[strm, {data}, opts]


(* ::Subsubsection:: *)
(*SingleElementQ and ElementsQ*)


(* These two functions decide which of the overloaded exporters are called by the framework. *)


Attributes[FirstHeld] = {HoldAll}


FirstHeld[_[first_, ___]] := Hold[first]


Attributes[SingleElementQ] = {HoldFirst}


SingleElementQ[expr_] := SingleElementQ[expr, _]


SingleElementQ[expr:(_Rule|_RuleDelayed), elem_] := MatchQ[FirstHeld[expr], Hold[elem]]


SingleElementQ[{expr:(_Rule|_RuleDelayed)}, elem_] := SingleElementQ[expr, elem]


SingleElementQ[expr_, elem_] := False


(* ::Subsection:: *)
(*Exporters*)


ExportDIF[strm_OutputStream, expr_, opts___?OptionQ] :=
Module[{data},
  data = "Data" /. expr;
  If[ Head@data === TableView, data = First@data];
  If[ MatchQ[data, {__TableView}], data = First/@data];
  ExportData[strm, data, opts]
] /; SingleElementQ[expr, "Data"]


ExportDIF[stream_OutputStream, expr_, opts___] :=
(
  Message[Export::noelem, ElementNames@expr, "DIF"]; 
 $Failed
)


(* ::Section::Closed:: *)
(*END CONVERTER CONTEXT*)


End[];
