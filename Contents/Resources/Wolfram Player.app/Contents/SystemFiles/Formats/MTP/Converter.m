(* ::Package:: *)

(* ::Subsubtitle:: *)
(*Minitab Portable Worksheet (MTP) Converter*)


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


System`ConvertersDump`BeginModule["System`Convert`MTPDump`"];


(* ::Section::Closed:: *)
(*IMPORT*)


(* ::Subsection:: *)
(*Utilities*)


toNumeric[x_] :=
 x /. a_String?(StringMatchQ[#,
       RegularExpression[
        "[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eEdD][-+]?\\d+)?"]] &) :>
    ToExpression[StringReplace[a, ("e" | "E" | "d" | "D") -> "*^"]]


trimWhitespace[x_] := 
x /. a_String :> StringReplace[a, 
    {StartOfString~~Whitespace->"", 
    Whitespace~~EndOfString->""}];


linesToData[lines_] :=
Module[
  {header, data, code, varnum, 
  datacount, width, name, takes}, 
  header = First[lines];
  code = ToExpression[StringTake[header, 7]];
  varnum = ToExpression[StringTake[header, {8, 14}]];
  datacount = ToExpression[StringTake[header, {15, 21}]];
  width = ToExpression[StringTake[header, {22, 28}]];
  name = StringReplace[ StringTake[header, {30, 78}], Whitespace ~~ EndOfString -> ""];
  data = Rest[lines];
  If[ MemberQ[{100, 101, 102}, code] && varnum === 0 && datacount === 0 && width===0 && name==="", 
      Return[{code, name->{}}]
  ];
  
  (* Else warn and ignore *)
  If[ code=!=3, Return[{code, name->{}}] ];    
    
  (* Empty Dataset *)
  If[data==={}, Return[{code, name->{}}]];
    
  {
    code, name -> 
    If[ width < 0, (* Strings *)
        width = -width;
        takes = (Plus[{1, width}, #] & /@ (Range[0, Floor[80/(width+1)] - 1]*(width + 1)));
        Flatten[trimWhitespace[Take[Flatten[StringTake[data, takes]], datacount]]], 
          
        (* Else Numeric Data *)
        takes = {{2, 16}, {18, 32}, {34, 48}, {50, 64}, {66, 80}};
        Flatten[toNumeric[(StringTake[data, takes] /. 
        {"              ."->Sequence[],  "              *"->None, "               "->Sequence[]})]]
    ] (* End If *)
  }
]

Options[ImportMTPData] := {"EmptyField" -> ""};
Options[ImportMTP] := {"EmptyField" -> ""};
Options[ImportMTPText] := {"EmptyField" -> ""};


(* ::Subsection:: *)
(*Importers*)


(* ::Subsubsection:: *)
(*Conditional Raw Importers*)


ImportMTPText[file_, opts:OptionsPattern[]] := 
Module[
  {textdata, out}, 
  textdata = Import[file, "Text", opts];
  If[Not@FreeQ[textdata, $Failed], Message[Import::fmterr, "MTP"]; Return[$Failed] ];
  textdata = StringSplit[textdata, StartOfLine ~~ "%"];
  If[Length[textdata]<2, Message[Import::fmterr, "MTP"]; Return[$Failed] ];
  textdata = StringSplit[#, "\n"]& /@ Rest[textdata];
  out = linesToData /@ textdata;
  out = Cases[out, {3, a_}:>a];
  out = DeleteCases[out, _?(FreeQ[#[[2]], _String]&)];
  "TextData" -> (out /. None -> (Quiet@OptionValue["EmptyField"]))
]


ImportMTPData[file_, opts:OptionsPattern[]] :=
Module[
  {textdata, out}, 
  textdata = Import[file, "Text", opts];
  If[Not@FreeQ[textdata, $Failed], Message[Import::fmterr, "MTP"]; Return[$Failed] ];
  textdata = StringSplit[textdata, StartOfLine ~~ "%"];
  If[Length[textdata]<2, Message[Import::fmterr, "MTP"]; Return[$Failed] ];
  textdata = StringSplit[#, "\n"]& /@ Rest[textdata];
  out = linesToData /@ textdata;
  out = Cases[out, {3, a_}:>a[[2]]];
  out = Select[out, FreeQ[#, _String]&];
  out = PadRight[out, Automatic, None];
  out = If[ArrayQ[out, _], Transpose@out, {}];
  "Data" -> Developer`ToPackedArray[out /. None -> (Quiet@OptionValue["EmptyField"])]
]  


(* ::Subsubsection:: *)
(*Default Raw Importer*)


ImportMTP[file_, opts:OptionsPattern[]] :=
Module[
  {textdata, out}, 
  textdata = Import[file, "Text"];
  If[Not@FreeQ[textdata, $Failed], Message[Import::fmterr, "MTP"]; Return[$Failed] ];
  textdata = StringSplit[textdata, StartOfLine ~~ "%"];
  If[Length[textdata]<2, Message[Import::fmterr, "MTP"]; Return[$Failed] ];
  textdata = StringSplit[#, "\n"]& /@ Rest[textdata]; 
  out = linesToData /@ textdata;
  out = Cases[out, {3, a_}:>a];
  out = Select[out, FreeQ[#[[2]], _String]&];
  "LabeledData"->(out /. None -> (Quiet@OptionValue["EmptyField"]))
]
 



(* ::Subsubsection:: *)
(*Post Importer*)


(* There are no registered post-importers for this format *)


(* ::Section::Closed:: *)
(*END CONVERTER CONTEXT*)


System`ConvertersDump`EndModule[];
