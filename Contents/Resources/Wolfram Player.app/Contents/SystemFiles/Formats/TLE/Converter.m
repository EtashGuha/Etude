(* ::Package:: *)

(* ::Subsubtitle:: *)
(*Two-Line Element (TLE) Converter*)


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


Begin["System`Convert`TLEDump`"];


(* ::Section:: *)
(*IMPORT*)


(* ::Subsection::Closed:: *)
(*UTILITIES*)


ClassifiedRules := {
  "U"->"Unclassified",
  "u"->"Unclassified",
  "C"->"Classified",
  "c"->"Classified",
  "S"->"Secret",
  "s"->"Secret",
  _-> Missing["NotAvailable"]
}   


elems := {
  "Name","SatelliteID","Classification", "InternationalDesignator", "LaunchYear",
  "EpochDate", "MeanMotionChangeRate", "BStar","EphemerisType", "ElementNumber", "Inclination", 
  "AscendingNodeRightAscension", "Eccentricity", "ArgumentPeriapsis", 
  "MeanAnomaly", "MeanMotion", "RevolutionNumber"
};


trim[s_] := StringReplace[s, {
  StartOfLine ~~ Whitespace -> "", 
  Whitespace ~~ EndOfLine -> ""
}]  


isLeapYear[year_] := TrueQ[(Mod[year, 4] == 0 && Mod[year, 100] != 0) || (Mod[year, 400] == 0)]


ToDecimalYears[{year_, day_}] :=
Module[
  {daybase = 365},
  If[ isLeapYear[year],
      daybase = 366
  ];
  year + day/daybase
]
  
  
toFullYear[y_]:= If[ y > 50, 1900 + y, 2000 + y ]


(* http://celestrak.com/columns/v04n03/ *)
EphemerisTypeRules := {
  0 -> "SGP4/SDP4",
  1 -> "SGP",
  2 -> "SGP4",
  3 -> "SDP4",
  4 -> "SGP8",
  5 -> "SDP8",
  _ -> Missing["NotAvailable"]
}


getTLE[lst_List, opts___?OptionQ] :=
Module[
  {name, line1, line2, len1, s, out},
  name = lst[[1]]; 
  {line1, line2} = lst[[2;;3]];
  If[ !StringQ[line1], Return[$Failed] ];
  (* for missing checksum of line1 *)
  len1 = StringLength[line1];
  If[ len1 < 68,
      Return[$Failed],
      If[len1 === 68, line1 = StringJoin[line1, " " (*not real checksum, but take the space*)]]
  ];
  s = StringJoin[line1, line2];
  If[ StringLength[s]<138,
      $Failed,
      out = TLEConversion[s, #] & /@ Rest[elems];
      If[ MemberQ[out, $Failed], $Failed, Prepend[out, name] ]
  ]
]


getTLE[lst___, opts___?OptionQ] := $Failed


TLESplit[lst_] := 
Split[lst, ! StringMatchQ[#1, RegularExpression["2 [0-9]{5}.*"]] &]


(* ::Subsubsection::Closed:: *)
(*TLEConversion[ s_String,  swtich ]*)


TLEConversion[s_String, "SatelliteID"] := 
If[ StringLength[s] > 10, 
    If[ StringMatchQ[#, RegularExpression[" *\\d+"]],
        trim[#],
        $Failed
     ]&[ StringTake[s, {3, 7}] ],
    $Failed
]


TLEConversion[s_String, "Classification"] := 
If[ StringLength[s] > 7, StringTake[s, {8, 8}], $Failed]


TLEConversion[s_String, "InternationalDesignator"] := 
If[ StringLength[s] > 17, StringTrim@StringTake[s, {10, 17}], $Failed]


TLEConversion[s_String, "LaunchYear"] := 
If[ StringLength[s] > 10, 
    If[ StringMatchQ[#, RegularExpression["\\d+"]], 
        toFullYear[ToExpression[#]], $Failed
    ]&[ StringTake[s, {10, 11}] ],
    $Failed
]


TLEConversion[s_String, "LaunchNumber"] := 
If[ StringLength[s] > 13, 
    If[ StringMatchQ[#, RegularExpression["\\d+"]], 
        ToExpression[#],
        $Failed
    ]&[ StringTake[s, {12, 14}] ],
    $Failed
]


TLEConversion[s_String, "LaunchPiece"] := 
If[StringLength[s] > 16, trim[StringTake[s, {15, 17}]], $Failed]


TLEConversion[s_String, "EpochDate"] := 
If[ StringLength[s] > 32, 
    If[ StringMatchQ[ StringTake[#, 5] <> StringDrop[#, 6], RegularExpression["[ \\d]+"]]
        && 
        StringMatchQ[StringTake[#, {6, 6}], "."],
        DateObject[{If[# > 50, 1900 + #, 2000 + #]&[ToExpression[StringTake[#, 2]]],1,ToExpression[StringTake[#, {3, -1}]],0,0,0}],
        $Failed
    ] &[ StringTake[s, {19, 32}] ],
    $Failed
]


TLEConversion[s_String, "MeanMotionChangeRate"] := 
If[ StringLength[s] > 43, 
    If[ StringMatchQ[StringDrop[#, 2], RegularExpression["\\d+ *"]]
        && 
        StringMatchQ[StringTake[#, 2], 
        RegularExpression["[ \\-\\+]\\."]], Quantity[ToExpression[#],"Revolutions"/"Days"/"Days"], 
        $Failed
     ] &[ StringTake[s, {34, 43}] ],
     $Failed
]


TLEConversion[s_String, "BStar"] := 
If[ StringLength[s] > 61, 
    If[ StringMatchQ[StringTake[#, 1], RegularExpression["[\\-\\+ ]"]]
        && 
        StringMatchQ[StringTake[#, {2, 6}], RegularExpression["\\d+ *"]]
        && 
        StringMatchQ[StringTake[#, {7, 8}], RegularExpression["[\\-\\+0]\\d"]], 
        Quantity[N[ToExpression[ StringTake[#, 6]]*(10^ToExpression[StringTake[#, -2]])],1/"EarthEquatorialRadius"],
        $Failed
    ] &[ StringTake[s, {54, 61}] ],
    $Failed
]
   


TLEConversion[s_String, "EphemerisType"] := 
If[ StringLength[s] > 62, 
    If[ StringMatchQ[#, RegularExpression["[0-5]"]], 
        ToExpression[#] /. EphemerisTypeRules,
        $Failed
    ] &[ StringTake[s, {63, 63}] ],
    $Failed
]


TLEConversion[s_String, "ElementNumber"] := 
If[ StringLength[s] > 68, 
    If[ StringMatchQ[#, RegularExpression[" *\\d+ *"]], 
        ToExpression[#],
        $Failed
    ] &[ StringTake[s, {65, 68}] ],
    $Failed
]


TLEConversion[s_String, "Inclination"] := 
If[ StringLength[s] > 75, 
    If[ StringMatchQ[StringTake[#, 3] <> StringDrop[#, 4], RegularExpression[" *\\d+ *"]]
        &&
        StringMatchQ[StringTake[#, {4, 4}], "."],
        Quantity[ToExpression[#],"AngularDegrees"],
        $Failed
    ] &[ StringTake[s, {78, 85}] ],
    $Failed
]


TLEConversion[s_String, "AscendingNodeRightAscension"] := 
If[ StringLength[s] > 94,
    If[ StringMatchQ[StringTake[#, 3] <> StringDrop[#, 4], RegularExpression[" *\\d+ *"]]
        &&
        StringMatchQ[StringTake[#, {4, 4}], "."],
        Quantity[ToExpression[#],"AngularDegrees"],
        $Failed
    ] &[ StringTake[s, {87, 94}] ],
    $Failed
]


TLEConversion[s_String, "Eccentricity"] := 
If[ StringLength[s] > 102, 
    If[ StringMatchQ[StringTake[#, 3] <> StringDrop[#, 4], RegularExpression["\\d+ *"]],
        N[ToExpression[#]/10^7], 
        $Failed
    ] &[ StringTake[s, {96, 102}] ],
    $Failed
]


TLEConversion[s_String, "ArgumentPeriapsis"] := 
If[ StringLength[s] > 111, 
   If[ StringMatchQ[StringTake[#, 3] <> StringDrop[#, 4], RegularExpression[" *\\d+ *"]]
       &&
       StringMatchQ[StringTake[#, {4, 4}], "."],
       Quantity[ToExpression[#],"AngularDegrees"],
       $Failed
    ] &[ StringTake[s, {104, 111}] ],
    $Failed
]


TLEConversion[s_String, "MeanAnomaly"] := 
If[ StringLength[s] > 120, 
    If[ StringMatchQ[StringTake[#, 3] <> StringDrop[#, 4], RegularExpression[" *\\d+ *"]]
        &&
        StringMatchQ[StringTake[#, {4, 4}], "."],
        Quantity[ToExpression[#],"AngularDegrees"],
        $Failed
     ] &[ StringTake[s, {113, 120}] ],
$Failed]


TLEConversion[s_String, "MeanMotion"] := 
If[ StringLength[s] > 132, 
    If[ StringMatchQ[StringTake[#, 2] <> StringDrop[#, 3], RegularExpression[" *\\d+ *"]]
        &&
        StringMatchQ[StringTake[#, {3, 3}], "."],
        Quantity[ToExpression[#],"Revolutions"/"Days"],
        $Failed
    ] &[ StringTake[s, {122, 132}] ],
    $Failed
]


TLEConversion[s_String, "RevolutionNumber"] := 
If[ StringLength[s] >= 137, 
    If[ StringMatchQ[#, RegularExpression[" *\\d+ *"]],
        Quantity[ToExpression[#],"Revolutions"],
        $Failed
    ]&[ StringTake[s, {133, 137}] ],
    $Failed
]


(* ::Subsection::Closed:: *)
(*IMPORTERS*)


(* ::Subsubsection::Closed:: *)
(*Conditional Raw Importers*)


(* There are no registered conditional raw importers for this format *)


(* ::Subsubsection::Closed:: *)
(*Default Raw Importer*)


ImportTLE[file_InputStream, opts___?OptionQ] := 
Module[
  {lines, out}, 
  lines = ReadList[file, String];
  If[ lines === {},
      Message[Import::fmterr, "TLE"];
      Return[$Failed]
  ];
  lines = TLESplit[lines];
  lines = PadLeft[lines, {Length[lines], 3}, None];
  If[ Union[Length/@lines]=!={3},
      Message[Import::fmterr, "TLE"];
      Return[$Failed]
  ];
  out = Quiet[getTLE[#, opts]] & /@ lines;
  If[ MemberQ[out, $Failed],
      Message[Import::fmterr, "TLE"];
      Return[$Failed]
  ];
  {"Data"->out,"LabeledData"->MapThread[Rule, {elems, Transpose[out]}],"Labels" -> elems}
]


(* ::Subsubsection::Closed:: *)
(*Post Importers*)


getSatellite[in_String][rules_, opts___] :=
Module[
  {n=in, labeledData, sid, pos},
  labeledData="LabeledData"/.rules;
  sid ="SatelliteID" /. labeledData;
  If[ ! MemberQ[sid, n], 
      Message[Import::sid, ToString[n, InputForm]];
      Return[$Failed]
  ];
  pos = Position[sid, n][[1, 1]];
  MapThread[Rule, {elems, Transpose[labeledData[[All, 2]]][[pos]] } ]
]


getSatellite[pos_Integer][rules_, opts___] :=
Module[
  {data},
  data="Data"/.rules;
  MapThread[Rule, {elems, data[[pos]]}]
]


(* ::Section::Closed:: *)
(*END CONVERTER CONTEXT*)


End[];  
