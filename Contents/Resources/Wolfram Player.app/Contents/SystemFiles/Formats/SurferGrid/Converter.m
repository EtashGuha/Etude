(* ::Package:: *)

(* ::Subsubtitle:: *)
(*Golden Software Surfer (SurferGrid) Converter*)


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


Begin["System`Convert`SurferDump`"];


(* ::Section::Closed:: *)
(*COMMON UTILITIES*)


(****************************************************
  These are functions used both by import and export
 ****************************************************)


$DefaultElevation = 1.70141*^38;


intReplace = 
Compile[
  {{inmat, _Integer, 2}, {val, _Integer}, {rep, _Integer}}, 
  Module[{mat = inmat}, 
    Do[Do[If[mat[[j, k]] == val, mat[[j, k]] = rep],
       {k, Length[mat[[1]]]}], {j, Length[mat]}
    ];
    mat
  ]
];


realReplace = 
Compile[
  {{inmat, _Real, 2}, {val, _Real}, {rep, _Real}}, 
  Module[
    {mat = inmat}, 
    Do[Do[If[mat[[j, k]] == val, mat[[j, k]] = rep],
       {k, Length[mat[[1]]]}], {j, Length[mat]}
    ];
    mat
  ]
];


replace[data_, missing_, rep_] := 
If[ ArrayQ[data, _, IntegerQ],
    intReplace[data, Round@missing, Round[rep]], 
    realReplace[data, N@missing, N@rep]
];



getMinMax[data_, None] := {Min[data], Max[data]}


getMinMax[data_, missing_] :=
Module[
  {a, b}, 
  a = Min[data];
  b = Max@replace[data, missing, a];
  {a, b}
]


(* ::Section:: *)
(*IMPORT*)


(* ::Subsection:: *)
(*UTILITIES*)


(* ::Subsubsection::Closed:: *)
(*Golden Software ASCII GRID (GSAG)*)


(******************************************************************************)
(************************************ GSAG ************************************)
(******************************************************************************)


Options[myImportGSAG] := {"DefaultElevation" -> Automatic, "DownsamplingFactor"->1, "SpatialRange" -> Automatic, Sequence@@Options[ReliefPlot]}


myImportGSAG[file_, opts : OptionsPattern[]] := 
Module[
  {lines, dimension, range, data, elevationRange, defaultElevation, downsamplingFactor},
  lines = Import[file, "Lines"];
  
  dimension = FromDigits /@ StringSplit[lines[[2]]];
  If[ !MatchQ[dimension, {_Integer, _Integer}], 
      Message[Import::fmterr, "SurferGrid"];
      Return[$Failed]
  ];
  
  range = ExternalService`ParseTable[StringSplit /@ lines[[3 ;; 4]]];
  range = OptionValue["SpatialRange"]/.Automatic->range;
  data = Flatten@ExternalService`ParseTable[StringSplit /@ lines[[6 ;; -1]]];
  
  If[ Length[data] =!= (Times @@ dimension), 
      Message[Import::fmterr, "SurferGrid"];
      Return[$Failed]
  ];
  data = Partition[data, dimension[[1]]];
  
  If[ ArrayQ[data, _Integer | _Real], 
      data = Developer`ToPackedArray[N@data],
      Message[Import::fmterr, "SurferGrid"];
      Return[$Failed]
  ];
  
  data = Clip[data, {-Infinity, $DefaultElevation}];
  elevationRange = getMinMax[data, $DefaultElevation];
  defaultElevation = N@(OptionValue["DefaultElevation"]/.Automatic->$DefaultElevation);

  If[ defaultElevation =!= $DefaultElevation,
     If[ NumberQ[defaultElevation], 
         data = replace[data, $DefaultElevation, defaultElevation], 
         data = data /. $DefaultElevation -> defaultElevation
     ]
  ];
    
  downsamplingFactor = OptionValue["DownsamplingFactor"];
  If[ !IntegerQ[downsamplingFactor],
      Message[Import::erropts, ToString[downsamplingFactor, InputForm],"DownsamplingFactor"];
      Return[$Failed]
  ];
  If[ downsamplingFactor=!=1,
      data = Take[data, {1, -1, downsamplingFactor}, {1, -1, downsamplingFactor}]
  ];
  
  {
   "Data" -> data,
   "ElevationRange" -> elevationRange,
   "RasterSize"->Reverse[Dimensions[data]],
   "SpatialRange" -> range, 
   "SpatialResolution" -> ((#[[2]] - #[[1]]) & /@ range/(dimension - 1))
  }
]


(* ::Subsubsection::Closed:: *)
(*Golden Software Binary Grid (GSBG)*)


(******************************************************************************)
(************************************ GSBG ************************************)
(******************************************************************************)


Options[myImportGSBG] := {"DefaultElevation" -> Automatic, "DownsamplingFactor"->1, "SpatialRange" -> Automatic, Sequence@@Options[ReliefPlot]}


myImportGSBG[str_InputStream, opts : OptionsPattern[]] := 
Module[
  {dimension, range, data, defaultElevation, elevationRange, downsamplingFactor},
  
  dimension = BinaryReadList[str, "Integer16", 2, ByteOrdering->-1];
  
  range = BinaryReadList[str, "Real64", 6, ByteOrdering->-1];
  range = Partition[range[[1 ;; 4]], 2];
  range = OptionValue["SpatialRange"]/.Automatic->range;
  
  data = BinaryReadList[str, "Real32", ByteOrdering->-1];
  If[ Length[data] =!= (Times @@ dimension), 
      Message[Import::fmterr, "SurferGrid"];
      Return[$Failed]
   ];
  data = Partition[data, dimension[[1]]];
  data = Developer`ToPackedArray[N@data];
  
  data = Clip[data, {-Infinity, $DefaultElevation}];
  elevationRange = getMinMax[data, $DefaultElevation];
  defaultElevation = N@(OptionValue["DefaultElevation"]/.Automatic->$DefaultElevation);
  If[ defaultElevation =!= $DefaultElevation,
      If[ NumberQ[defaultElevation], 
          data = replace[data, $DefaultElevation, defaultElevation], 
          data = data /. $DefaultElevation -> defaultElevation
      ]
  ];
    
  downsamplingFactor = OptionValue["DownsamplingFactor"];
  If[ !IntegerQ[downsamplingFactor],
      Message[Import::erropts, ToString[downsamplingFactor, InputForm],"DownsamplingFactor"];
      Return[$Failed]
  ];
  If[ downsamplingFactor=!=1,
      data = Take[data, {1, -1, downsamplingFactor}, {1, -1, downsamplingFactor}]
  ];
  
  {
   "Data" -> data,
   "ElevationRange" -> elevationRange,
   "RasterSize"->Reverse[Dimensions[data]],
   "SpatialRange" -> range, 
   "SpatialResolution" -> ((#[[2]] - #[[1]]) & /@ range/(dimension - 1))
  }
]


(* ::Subsubsection::Closed:: *)
(*Golden Software Surfer 7*)


(******************************************************************************)
(************************************ Surfer7 *********************************)
(******************************************************************************)


ImportSurfer7[elems_, opts___] := 
myImportSurfer7[elems, FilterRules[Flatten@{opts}, Options[myImportSurfer7]]]


Options[myImportSurfer7] := {"DefaultElevation" -> Automatic, "DownsamplingFactor"->1, "SpatialRange" -> Automatic, Sequence@@Options[ReliefPlot]}


myImportSurfer7[str_InputStream, opts : OptionsPattern[]] := 
Module[
  {list, tag, grid, xLL, yLL, xSize, ySize, dimension, range, data,
   defaultElevation, blankvalue, downsamplingFactor},
  If[str === $Failed, Message[Import::fmterr, "SurferGrid"]; Return[$Failed]];
  
  list = {};
  list = Append[list, readBlock[str, 1112691524]];
  While[
    (tag = BinaryRead[str, "Integer32", ByteOrdering->-1]) =!= EndOfFile, 
    list = Append[list, readBlock[str, tag]];
  ];
  
  grid = "44495247" /. list;
  If[grid === {}, Message[Import::fmterr, "SurferGrid"]; Return[$Failed] ];
  
  dimension = grid[[1 ;; 2]];
  
  {xLL, yLL, xSize, ySize} = grid[[3 ;; 6]];
  range = {
    {xLL, xLL + xSize*(dimension[[2]] - 1)},
    {yLL, yLL + ySize*(dimension[[1]] - 1)}
  };
  range = OptionValue["SpatialRange"]/.Automatic->range;
  
  data = "41544144" /. list;
  If[ Length[data] =!= (Times @@ dimension), 
      Message[Import::fmterr, "SurferGrid"];
      Return[$Failed]
  ];
  data = Partition[data, dimension[[2]]];
  data = Developer`ToPackedArray[N@data];
  
  blankvalue = grid[[10]];
  data = Clip[data, {-Infinity, blankvalue}];
  defaultElevation = N@(OptionValue["DefaultElevation"]/.Automatic->$DefaultElevation);
  If[ defaultElevation =!= blankvalue,
      If[ NumberQ[defaultElevation], 
          data = replace[data, blankvalue, defaultElevation], 
          data = data /. blankvalue -> defaultElevation
      ]
  ];
    
  downsamplingFactor = OptionValue["DownsamplingFactor"];
  If[ !IntegerQ[downsamplingFactor],
      Message[Import::erropts, ToString[downsamplingFactor, InputForm],"DownsamplingFactor"];
      Return[$Failed]
  ];
  If[ downsamplingFactor=!=1,
      data = Take[data, {1, -1, downsamplingFactor}, {1, -1, downsamplingFactor}]
  ];
  
  {
   "Data" -> data,
   "ElevationRange" -> grid[[7 ;; 8]],
   "RasterSize"->Reverse[Dimensions[data]],
   "SpatialRange" -> range, "SpatialResolution" -> ((#[[2]] - #[[1]]) & /@ range/(dimension - 1))
  }
]


(*Header section:
IntegerString[1112691524,16]="42525344"
*)
readBlock[str_, 1112691524] := 
Module[
  {len = BinaryRead[str, "Integer32", ByteOrdering->-1]}, 
  "42525344" -> BinaryReadList[str, "Byte", len, ByteOrdering->-1]
];


(*Grid section:
IntegerString[1145655879,16]="44495247"
*)
readBlock[str_, 1145655879] := 
Module[
  {len = BinaryRead[str, "Integer32", ByteOrdering->-1], out = {}},
  If[ len === 72,
      out = BinaryReadList[ str, {"Integer32", "Integer32", "Real64", "Real64", "Real64", "Real64", 
       "Real64", "Real64", "Real64", "Real64"}, 1, ByteOrdering->-1]
  ];
  "44495247" -> If[Length[out] === 1 && Length[out[[1]]] === 10, out[[1]], {}]
];


(*Data section:
IntegerString[1112691524,16]="41544144"
*)
readBlock[str_, 1096040772] := 
Module[
  {len = BinaryRead[str, "Integer32", ByteOrdering->-1]}, 
  "41544144" -> BinaryReadList[str, "Real64", len/8, ByteOrdering->-1]
];


(*Fault Info section:
IntegerString[1112691524,16]="49544c46"
*)
readBlock[str_, 1230261318] := 
Module[
  {len = BinaryRead[str, "Integer32", ByteOrdering->-1]}, 
  "49544c46" -> BinaryReadList[str, "Byte", len, ByteOrdering->-1]
];


(*Other section:
IntegerString[1112691524,16]="42525344"
*)
readBlock[str_, tag___] := 
Module[
  {len = BinaryRead[str, "Integer32", ByteOrdering->-1]}, 
  IntegerString[tag, 16] -> BinaryReadList[str, "Byte", len, ByteOrdering->-1]
];


(* ::Subsection:: *)
(*IMPORTERS*)


(* ::Subsubsection::Closed:: *)
(*Conditional Raw Importers*)


(* There are no registered conditional raw importers for this format *)


(* ::Subsubsection::Closed:: *)
(*Default Raw Importer*)


ImportSurfer[str_, opts___] :=
Module[
  {header},
  If[str === $Failed, Message[Import::fmterr, "SurferGrid"]; Return[$Failed] ];
  
  header = StringJoin@BinaryReadList[str, "Character8", 4, ByteOrdering->-1];
  Switch[header,
    "DSAA",  myImportGSAG[str[[1]], FilterRules[Flatten@{opts}, Options[myImportGSAG]]],
    "DSBB",  myImportGSBG[str, FilterRules[Flatten@{opts}, Options[myImportGSBG]]],
    "DSRB", myImportSurfer7[str, FilterRules[Flatten@{opts}, Options[myImportSurfer7]]],
  _, Message[Import::fmterr, "SurferGrid"]; Return[$Failed]]
]


(* ::Subsubsection:: *)
(*Post Importers*)


getImage[lst_, opts___] := If[ lst===$Failed, $Failed, Image[ Reverse@("Data" /. lst)] ]


getReliefImage[lst_, opts___]:=
Module[
  {defaultElevation, data}, 
  If[lst === $Failed, Return[$Failed]];
  defaultElevation = N@("DefaultElevation" /.Flatten[{opts}] /."DefaultElevation"->Automatic);
  data = ("Data" /. lst);
  If[ Not[NumberQ[defaultElevation]], 
      If[ defaultElevation===Automatic,
          data = replace[data, $DefaultElevation, 0.],
          data = data/.defaultElevation->0.
      ]
  ];
  ReliefImage[Reverse@data, Sequence @@ FilterRules[Flatten@{opts},Options[ReliefImage]]]
]


getGraphics[lst_, opts___] := 
Module[
  {defaultElevation, data, range, gropts},
  If[lst === $Failed, Return[$Failed]];
  defaultElevation = N@("DefaultElevation" /.Flatten[{opts}] /."DefaultElevation"->Automatic);
  data = ("Data" /. lst);
  If[ Not[NumberQ[defaultElevation]], 
      If[ defaultElevation===Automatic,
          data = replace[data, $DefaultElevation, 0.],
          data = data/.defaultElevation->0.
      ]
  ];
  range=("SpatialRange" /. lst);
  gropts = FilterRules[Flatten@{
      opts,
      Frame -> True,
      DataRange -> Reverse[range],
      ColorFunction -> "Topographic",
      FrameTicks -> None, 
      DataReversed -> False
      },
      Options[ReliefPlot]
  ];
  ReliefPlot[data, Sequence @@ gropts]
]



(* ::Section:: *)
(*EXPORT*)


(* ::Subsection:: *)
(*Utilities*)


Attributes[FirstHeld] = {HoldAll}

FirstHeld[_[first_, ___]] := Hold[first]

Attributes[ElementsQ] = {HoldFirst}

ElementsQ[expr_] := ElementsQ[expr, _]

ElementsQ[expr_?Developer`PackedArrayQ, _] := False

ElementsQ[expr:{(_Rule|_RuleDelayed)..}, elem_] :=
Module[{first},
  first = Map[FirstHeld, Unevaluated@expr];
  DeleteCases[first, Hold[elem]]==={}
]

ElementsQ[expr:(_Rule|_RuleDelayed), elem_] :=
  ElementsQ[{expr}, elem]

ElementsQ[expr_, elem_] := False

ElementNames[elem_List] := First/@elem

ElementNames[elem_] := ElementNames[{elem}]


(* ::Subsubsection::Closed:: *)
(*Common export utilities*)


toExponentString[n_] := 
ToString[NumberForm[n, 
   NumberFormat -> (Row[{#1, "E", #3 /. x_ :> NumberForm[
           FromDigits@#3, {3, 0},
           NumberPadding -> {"0", ""}, 
           NumberPoint -> "",
           NumberSigns -> {"-", "+"}, 
           SignPadding -> True
           ]}] &)
]]


toString[n_] := 
Module[
  {s = ToString[n, InputForm]},
  If[ StringCases[s, "*"] =!= {},
      toExponentString[n],
      s
  ]
]


toRangeString[l_] := toString[l[[1]]] <> " " <> toString[l[[2]]]


dataPartition[l_] := 
Append[
  StringJoin[#, " "] & /@ (Riffle[#, " "] & /@ 
  Join[ Partition[l, 10], {l[[-Mod[Length@l, 10] ;; -1]]} ]), ""
]


(* ::Subsubsection::Closed:: *)
(*myExportSurfer[]*)


Options[myExportSurfer] = {"BinaryFormat"->True, "DefaultElevation" -> Automatic, "DownsamplingFactor"->1, "LegacyFormat"->False, "SpatialRange" -> Automatic};


myExportSurfer[filename_, rules_, opts : OptionsPattern[]] := 
Module[
  {legancyFormat=OptionValue["LegacyFormat"], binaryFormat= OptionValue["BinaryFormat"]},
  Switch[ legancyFormat,
    False,
    If[ binaryFormat=!=True,
        Message[Export::erropts, ToString[binaryFormat, InputForm],"BinaryFormat"];
        Return[$Failed]
    ];
    ExportSurfer7[filename, rules, opts],

    True,
    Switch[ binaryFormat,
       True, ExportGSBG[filename, rules, opts],
       False, ExportGSAG[filename, rules, opts],
       _, Message[Export::erropts, ToString[binaryFormat, InputForm],"BinaryFormat"]; Return[$Failed]
    ],
    
    _,
    Message[Export::erropts, ToString[legancyFormat, InputForm],"LegacyFormat"]; Return[$Failed]
  ]
]


(* ::Subsubsection::Closed:: *)
(*ExportGSAG[]*)


Options[ExportGSAG] = {"BinaryFormat"->True, "DefaultElevation" -> Automatic, "DownsamplingFactor"->1, "LegacyFormat"->True, "SpatialRange" -> Automatic};


(* ElementsQ[rules, "Data"|"ElevationRange"|"Graphics"|"RasterSize"|"SpatialRange"|"SpatialResolution"] *)
ExportGSAG[filename_, rules_, opts : OptionsPattern[]] :=
Module[
  {data, range, defaultElevation, dimension, elevationrange, options, downsamplingFactor},
  data = "Data" /. rules;
  defaultElevation = N[OptionValue["DefaultElevation"]/.Automatic->$DefaultElevation];
  If[ !ArrayQ[data, 2, NumericQ]||Not[NumericQ[defaultElevation]], 
      data=data/.{Missing->$DefaultElevation,None->$DefaultElevation,Null->$DefaultElevation, defaultElevation->$DefaultElevation};
      defaultElevation=$DefaultElevation
  ];
  If[ !ArrayQ[data, 2, NumericQ], 
      Message[Export::errelem, "Data", "SurferGrid"];
      Return[$Failed]
  ];

  downsamplingFactor = OptionValue["DownsamplingFactor"];
  If[ !IntegerQ[downsamplingFactor],
      Message[Export::erropts, ToString[downsamplingFactor, InputForm],"DownsamplingFactor"];
      Return[$Failed]
  ];
  If[ downsamplingFactor=!=1,
      data = Take[data, {1, -1, downsamplingFactor}, {1, -1, downsamplingFactor}]
  ];
  
  data = Developer`ToPackedArray[N@data];
  dimension = Dimensions[data];
  
  range = "SpatialRange" /. rules /. ("SpatialRange"->({0, #} & /@dimension));
  range = OptionValue["SpatialRange"]/.Automatic->range;
  If[ !(ArrayQ[range] && Dimensions[range] === {2, 2}), 
      Message[Export::errelem, "SpatialRange", "SurferGrid"];
      Return[$Failed]
  ];
   
  If[ NumericQ[defaultElevation], 
      data = Clip[data, {-Infinity, defaultElevation}]
  ];

  elevationrange = toRangeString[
    "ElevationRange" /. rules /. "ElevationRange" -> ({Min[#], Max[#]}&[DeleteCases[Flatten@data, defaultElevation]])
  ];
  
  data = Map[ ToString[#, InputForm]&, data, {2}];
  data = data /. (ToString[defaultElevation, InputForm] -> "1.70141E+038");

  If[ opts === Null,
      options = {"CharacterEncoding" -> "ASCII"}, 
      If[ ListQ[opts] && FreeQ[opts[[All, 1]], "CharacterEncoding"], 
          options = Append[opts, "CharacterEncoding" -> "ASCII"]
      ]
  ];
  
  Export[
    filename,
    Join[
      {"DSAA"},
      {toRangeString[Reverse@dimension]},
      toRangeString /@ range, {elevationrange},
      Flatten[dataPartition /@ data]
    ],
    "List",
    options
  ]
]/; ElementsQ[rules, "Data"|"ElevationRange"|"Graphics"|"Image"|"RasterSize"|"SpatialRange"|"SpatialResolution"]


ExportGSAG[filename_, rules_, opts___?OptionQ] :=
Module[
  {noelems},
  If[ Head[rules] =!= List,
      Message[Export::type, Head@rules, "SurferGrid"];
      Return[$Failed]
  ];
  If[ MemberQ[MapThread[Or, {MatchQ[#, _ :> _] & /@ rules, MatchQ[#, _ -> _] & /@ rules}], False],
      Message[Export::nodta, Expression, "SurferGrid"];
      Return[$Failed]
  ];
  noelems = DeleteCases[ElementNames[rules], "Data"|"ElevationRange"|"Graphics"|"Image"|"RasterSize"|"SpatialRange"|"SpatialResolution"];
  Message[Export::noelem, noelems, "SurferGrid"];
  Return[$Failed]
]


(* ::Subsubsection::Closed:: *)
(*ExportGSBG[]*)


Options[ExportGSBG] = {"BinaryFormat"->True, "DefaultElevation" -> Automatic, "DownsamplingFactor"->1, "LegacyFormat"->True, "SpatialRange" -> Automatic};


(* ElementsQ[rules, "Data"|"ElevationRange"|"Graphics"|"RasterSize"|"SpatialRange"|"SpatialResolution"] *)
ExportGSBG[file_, rules_, opts : OptionsPattern[]] :=
Module[
  {data, range, defaultElevation, dimension, elevationrange, downsamplingFactor},  
  data = "Data" /. rules;
  If[!ArrayQ[data],Message[Export::errelem,"Data","SurferGrid"];Return[$Failed]];

  downsamplingFactor = OptionValue["DownsamplingFactor"];
  If[ !IntegerQ[downsamplingFactor],
      Message[Export::erropts, ToString[downsamplingFactor, InputForm],"DownsamplingFactor"];
      Return[$Failed]
  ];
  If[ downsamplingFactor=!=1,
      data = Take[data, {1, -1, downsamplingFactor}, {1, -1, downsamplingFactor}]
  ];

  defaultElevation = N[OptionValue["DefaultElevation"]/.Automatic->$DefaultElevation];
  If[ ArrayQ[data,2,NumericQ]&&NumericQ[defaultElevation],
      (* numeric data and defaultElevation*)
      data = Developer`ToPackedArray[N@data];
      data = Clip[data, {-Infinity, defaultElevation}];
      If[ defaultElevation =!= $DefaultElevation,
          data = replace[data, defaultElevation, $DefaultElevation]
      ],

      (* data contains Missing, Null, etc; Or DefaultElevation is not nubmeric *)
      data=data/.{Missing->$DefaultElevation,None->$DefaultElevation,Null->$DefaultElevation,defaultElevation->$DefaultElevation};
      If[ !ArrayQ[data,2,NumericQ],
          Message[Export::errelem,"Data","SurferGrid"];
          Return[$Failed]
      ];
      data = Developer`ToPackedArray[N@data];
      data = Clip[data, {-Infinity, $DefaultElevation}]
  ];

  dimension = Dimensions[data];

  range = "SpatialRange" /. rules /. ("SpatialRange"->({0, #} & /@dimension));
  range = OptionValue["SpatialRange"]/.Automatic->range;
  If[ !(ArrayQ[range] && Dimensions[range] === {2, 2}), 
      Message[Export::errelem, "SpatialRange", "SurferGrid"];
      Return[$Failed]
  ];
  
  elevationrange = 
   "ElevationRange" /. rules /. 
   "ElevationRange" -> ({Min[#], Max[#]} &[DeleteCases[Flatten@data, defaultElevation]]);
  
  BinaryWrite[file, {"D", "S", "B", "B"}, "Character8"]; 
  BinaryWrite[file, Reverse@dimension, "Integer16"]; 
  BinaryWrite[file, Join[Flatten@range, elevationrange], "Real64"];
  BinaryWrite[file, Flatten@data, "Real32"];
  Close[file]
]/; ElementsQ[rules, "Data"|"ElevationRange"|"Graphics"|"Image"|"RasterSize"|"SpatialRange"|"SpatialResolution"]


ExportGSBG[filename_, rules_, opts___?OptionQ] :=
Module[
  {noelems},
  If[ Head[rules] =!= List,
      Message[Export::type, Head@rules, "SurferGrid"];
      Return[$Failed]
  ];
  If[ MemberQ[MapThread[Or, {MatchQ[#, _ :> _] & /@ rules, MatchQ[#, _ -> _] & /@ rules}], False],
      Message[Export::nodta, Expression, "SurferGrid"];
      Return[$Failed]
  ];
  noelems = DeleteCases[ElementNames[rules], "Data"|"ElevationRange"|"Graphics"|"Image"|"RasterSize"|"SpatialRange"|"SpatialResolution"];
  Message[Export::noelem, noelems, "SurferGrid"];
  Return[$Failed]
]


(* ::Subsubsection::Closed:: *)
(*ExportSurfer7[]*)


Options[ExportSurfer7] = {"DefaultElevation" -> Automatic, "DownsamplingFactor"->1, "LegacyFormat"->False, "SpatialRange" -> Automatic};


(*ElementsQ[rules, "Data"|"ElevationRange"|"Graphics"|"RasterSize"|"SpatialRange"|"SpatialResolution"]*)
ExportSurfer7[file_, rules_, opts : OptionsPattern[]] :=
Module[
  {data, elevationrange, spatialrange, spatialresolution, 
   defaultElevation, dimension, downsamplingFactor},
  data = "Data" /. rules;
  If[ !ArrayQ[data],Message[Export::errelem,"Data","SurferGrid"];Return[$Failed] ];
   
  downsamplingFactor = OptionValue["DownsamplingFactor"];
  If[ !IntegerQ[downsamplingFactor],
      Message[Export::erropts, ToString[downsamplingFactor, InputForm],"DownsamplingFactor"];
      Return[$Failed]
  ];
  If[downsamplingFactor=!=1, data = Take[data, {1, -1, downsamplingFactor}, {1, -1, downsamplingFactor}]];
  
  defaultElevation = N[OptionValue["DefaultElevation"]/.Automatic->$DefaultElevation];
  If[ ArrayQ[data,2,NumericQ]&&NumericQ[defaultElevation],
      (* numeric data and defaultElevation*)
      data = Developer`ToPackedArray[N@data];
      data = Clip[data, {-Infinity, defaultElevation}];
      If[ defaultElevation =!= $DefaultElevation,
          data = replace[data, defaultElevation, $DefaultElevation]
      ],
      (* data contains Missing, Null, etc; Or DefaultElevation is not nubmeric *)
      data=data/.{Missing->$DefaultElevation,None->$DefaultElevation,Null->$DefaultElevation,defaultElevation->$DefaultElevation};
      If[!ArrayQ[data,2,NumericQ], Message[Export::errelem,"Data","SurferGrid"]; Return[$Failed] ];
      data = Developer`ToPackedArray[N@data];
      data = Clip[data, {-Infinity, $DefaultElevation}]
  ];

  dimension = Dimensions[data];

  spatialrange = "SpatialRange" /. rules /. ("SpatialRange"->({0, #} & /@dimension));
  spatialrange = OptionValue["SpatialRange"]/.Automatic->spatialrange;
  If[ !(ArrayQ[spatialrange] && Dimensions[spatialrange] === {2, 2}), 
      Message[Export::errelem, "SpatialRange", "SurferGrid"];
      Return[$Failed]
  ];

  elevationrange = 
   "ElevationRange" /. rules /. 
   "ElevationRange" -> ({Min[#], Max[#]} &[DeleteCases[Flatten@data, defaultElevation]]);
  spatialresolution = 
   "SpatialResolution" /. rules /. 
   "SpatialResolution" -> {
                (spatialrange[[1, 2]] - spatialrange[[1, 1]])/(dimension[[2]] - 1),
                (spatialrange[[2, 2]] - spatialrange[[2, 1]])/(dimension[[1]] - 1)
                };
  
  (*Header section: 0x42525344*)
  
  BinaryWrite[file, 1112691524, "Integer32"];
  BinaryWrite[file, 4, "Integer32"];
  BinaryWrite[file, {1, 0, 0, 0}, "Byte"];
  
  (*Grid section: 0x44495247*)
  BinaryWrite[file, 1145655879, "Integer32"];
  BinaryWrite[file, 72, "Integer32"];
  BinaryWrite[file, dimension, "Integer32"];
  BinaryWrite[
    file,
    Join[
      spatialrange[[All, 1]],
      spatialresolution, 
      elevationrange,
      {0., $DefaultElevation}
    ],
    "Real64"
  ];
  
  (*Data section: 0x41544144*)
  BinaryWrite[file, 1096040772, "Integer32"];
  BinaryWrite[file, Length[Flatten@data]*8, "Integer32"];
  BinaryWrite[file, Flatten@data, "Real64"];
  
  Close[file]
]/; ElementsQ[rules, "Data"|"ElevationRange"|"Graphics"|"Image"|"RasterSize"|"SpatialRange"|"SpatialResolution"]


ExportSurfer7[filename_, rules_, opts___?OptionQ] :=
Module[
  {noelems},
  If[ Head[rules] =!= List,
      Message[Export::type, Head@rules, "SurferGrid"];
      Return[$Failed]
  ];
  If[ MemberQ[MapThread[Or, {MatchQ[#, _ :> _] & /@ rules, MatchQ[#, _ -> _] & /@ rules}], False],
      Message[Export::nodta, Expression, "SurferGrid"];
      Return[$Failed]
  ];
  noelems = DeleteCases[ElementNames[rules], "Data"|"ElevationRange"|"Graphics"|"Image"|"RasterSize"|"SpatialRange"|"SpatialResolution"];
  Message[Export::noelem, noelems, "SurferGrid"];
  Return[$Failed]
]


(* ::Subsection::Closed:: *)
(*Exporters*)


ExportSurfer[filename_, rules_, opts___] := myExportSurfer[filename, rules, FilterRules[Flatten@{opts}, Options[myExportSurfer]]]


(* ::Section::Closed:: *)
(*END CONVERTER CONTEXT*)


End[];
