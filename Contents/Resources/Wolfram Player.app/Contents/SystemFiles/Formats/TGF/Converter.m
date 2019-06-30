(* ::Package:: *)

(* ::Subsubtitle:: *)
(*Trivial Graph Format (TGF) Converter*)


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
(*FILE LAYOUT*)


(*************************************************************************

Organization of this file:

IMPORT:
  - Utility Functions
  - Registered Importers
    -- Conditional Raw Importers
    -- Default Raw Importer
    -- Post-Importers

EXPORT:
  - Utility Functions
    -- Export Graph Constructor
    -- AdjacencyMatrix to EdgeRules
    -- SingleElementQ and ElementsQ
    -- Validators
  - Overloaded Exporters
    -- ExportTGF[]  --> Called when no explicit export element declared
    -- ExportTGF[] /;ElementsQ[data,"EdgeRules"|...]   --> Called when export via "EdgeRules" element
    -- ExportTGF[] /;ElementsQ[data,"AdjacencyMatrix"|...]  -->  Called when export via "AdjacencyMatrix" element 
    -- ExportTGF[] /;ElementsQ[data,"Graph"]  --> Called when export via "Graph" element

*************************************************************************)


(* ::Section::Closed:: *)
(*BEGIN CONVERTER CONTEXT*)


Begin["System`Convert`TGFDump`"]


(* ::Section::Closed:: *)
(*IMPORT*)


(* ::Subsection::Closed:: *)
(*Utility Functions*)


(* ::Subsubsection::Closed:: *)
(*Mixed graphs and multi graphs switch*)


(* This switch is used remove duplicated edges when importing the "Graph" and "Graphics" elements. *)
$MixedMultiEdges = False;


(* ::Subsubsection::Closed:: *)
(*ReadNodes[] and ReadEdges[]*)


(*************************************************** 
   ReadNodes[] acts on the beginning of the file, processes the file
   until the line containing "#", signaling the end of nodes in
   file.
   ReadNodes[] returns a list of {node_id, label} pairs if successful,
   and $Failed otherwise.
***************************************************)
ReadNodes[strm_]:=
Module[
  {Q, w, line, out, pos, strm2, tempString},
  Q = True;
  out = Last[Reap[While[ Q,
    tempString = Read[strm, "String"];
    If[ tempString === EndOfFile, Return[$Failed, Module] ];
    strm2 = StringToStream[ tempString ];
    pos = SetStreamPosition[strm2, Infinity];
    SetStreamPosition[strm2, 0];
    w = Read[strm2, "Word"];
    If[ w === EndOfFile || StreamPosition[strm2]===EndOfFile, Return[$Failed] ];
    If[ w =!= "#",
        (* the node_id's must be numbers *)
        If[ !StringMatchQ[w, NumberString], Return[$Failed] ];

        (* the rest of the line is the label *)
        If[ StreamPosition[strm2] =!= pos,
            line = StringTrim[Read[strm2, "String"]],
            line = ""
        ];
        Quiet@Close[strm2];
        Sow[{ToExpression[w], line}],

        (* w === "#", we are at the end of the node section *)
        Quiet@Close[strm2];
        Q = False
    ]
    ]]];
    If[ out ==={}, $Failed, First[out] ]
]


(***************************************************
   ReadEdges[] should be used immediately after ReadNodes[] because
   ReadNodes[] places StreamPosition appropriately.
   ReadEdges[] returns a list of {from, to, label} triplets if
   successful, and $Failed otherwise.
***************************************************)
ReadEdges[strm_]:=
Module[
  {pos, w1, w2, l, strm2, temp, endpos, out},
  (* find end of stream *)
  pos = StreamPosition[strm];
  endpos = SetStreamPosition[strm, Infinity];
  SetStreamPosition[strm, pos];
  out = Last[Reap[While[pos =!= endpos,
    strm2 = StringToStream[Read[strm,"String"]];
    pos = SetStreamPosition[strm2, Infinity];
    SetStreamPosition[strm2, 0];
    temp = ReadList[strm2, "Word", 2];
    (* guard against breaks in the stream or unexpected forms of edges *)
    If[ MatchQ[temp, EndOfFile|{}], Break[], {w1, w2}=temp];

    (* the node_id's must be numbers *)
    If[ StringMatchQ[#,NumberString]&/@{w1,w2}=!={True,True},
        Return[$Failed],
        {w1, w2} = ToExpression/@{w1,w2}
    ];

    (* the rest of the line is the label *)
    If[ StreamPosition[strm2] === pos,
        l = "",
        l = StringTrim[Read[strm2, "String"]]
    ];
    Quiet[Close[strm2]];
    Sow[{w1,w2,l}];
    pos = StreamPosition[strm]
  ]]];

  If[out ==={}, $Failed, First[out]]
]


(* ::Subsubsection::Closed:: *)
(*MakeEdge[] and MakeVertex[]*)


(* ::Text:: *)
(*These functions are used to construct Graph[] object.*)


MakeEdge[edge_Rule, attr_List] :=
If[ ("Label" /. attr) === "Label",
    Apply[DirectedEdge, edge],
    Property[DirectedEdge@@edge, {EdgeLabels -> ("Label" /. attr)}]
]


MakeVertex[vertex_, attr_List] :=
If[ ("Label" /. attr) === "Label",
   vertex,
   Property[vertex, {VertexLabels -> ("Label" /. attr)}]
]


(* ::Subsection::Closed:: *)
(*Registered Importers*)


(*************************************************************************
The importers are functions called by the framework when an Import call is made.
There are three types of importers:
   - conditional raw importer
   - default raw importer
   - post-importer
Be sure to read the comment in each type of importer and understand the input and output of these functions.
*************************************************************************)


(* ::Subsubsection::Closed:: *)
(*Conditional Raw Importer*)


(* ::Text:: *)
(*Note the forms of the input and output of the raw-importers.  The raw-importer takes as input an InputStream since RegisterImport has registered the option FunctionChannels -> {"Streams"}.  The raw-importers return a Rule or a List of Rules, with the left hand side being the name of the Element, and the output on the right hand side.*)


(* ::Text:: *)
(*In the TGF format, we have only one raw importer for the elements "Graph" and "VertexCount", specified by the line *)
(*     "Graph"|"VertexCount"  :>  ImportTGFGraph*)
(*in Import.m*)


ImportTGFGraph[strm_, opts___]:=
Module[
  {data, graph, eR, eA, vL, vA},
  data = ImportTGF[strm, opts];
  If[ data === $Failed, Return[$Failed] ];
  {eR, eA, vL, vA} = {"EdgeRules","EdgeAttributes","VertexList","VertexAttributes"}/.data;

  graph = Graph[ MapThread[ MakeVertex,{vL, vA} ], MapThread[MakeEdge, {eR, eA} ], opts ];
  If[ !GraphQ[graph], Return[$Failed] ];

  {
    "Graph" -> graph,
    "VertexCount" -> VertexCount[graph]
  }
]


(* ::Subsubsection::Closed:: *)
(*Default Raw Importer*)


(* ::Text:: *)
(*The default raw importer is called when we import Elements that are not explicitly registered.  (Elements that are explicitly registered require conditional raw importers).  The default raw importer is allowed to return mulitple Elements (in this example, "AdjacencyMatrix", "VertexList", "VertexCount", "VertexAttributes", "EdgeRules", and "EdgeAttributes").  The form of the output is the same as the conditional raw importer, being a Rule or a List of Rules. *)


ImportTGF[strm_, opts___]:=
Module[
  {nodes, edges, nL, nA, eR, eA, nC, adjM},
  nodes = ReadNodes[strm];
  If[ nodes === $Failed, Message[Import::fmterr, "TGF"]; Return[$Failed] ];
  edges = ReadEdges[strm];
  If[ edges === $Failed, Message[Import::fmterr, "TGF"]; Return[$Failed] ];

  nL = First/@ nodes;
  nA = Map[ If[ StringTrim[#] === "", {}, {"Label"->StringTrim[#]}]&, Part[nodes, All, 2] ];
  eR = Map[ #[[1]] -> #[[2]] &, edges];
  eA = Map[ If[ StringTrim[#] === "", {}, {"Label"->StringTrim[#]}]&, Part[edges, All, 3] ];

  (* construct adjacency matrix and ensure that the basis used is the same as the output of the "VertexList" element *)
  nC = Length[nL];
  adjM = eR /. MapThread[Rule,{nL, Range[nC]}];
  adjM = SparseArray[Map[ Rule[Apply[List, First[#]], Last[#]] &, Tally[adjM]], {nC, nC}];

  (* The default raw importer returns a list of rules in the form of Rule[elem_name, elem_value] *)
  {
    "VertexList" -> nL,
    "VertexAttributes" -> nA,
    "EdgeRules" -> eR,
    "EdgeAttributes" -> eA,
    "AdjacencyMatrix" -> adjM
  }
]


(* ::Subsubsection::Closed:: *)
(*Post Importer*)


(* ::Text:: *)
(*Note the forms of the input and output of the post-importer.  The post-importer takes as input the output of the default raw importer (a list of Rule[]'s).  The output is simply the Graphics object, not a Rule["Graphics", ...].*)


ImportTGFGraphics[rules_, opts___]:=
If[ rules =!= $Failed,
    GraphPlot[ "EdgeRules"/.rules, FilterRules[{opts}, Options[GraphPlot]], DirectedEdges -> True ],
    $Failed
]


(* ::Section::Closed:: *)
(*EXPORT*)


(* ::Subsection::Closed:: *)
(*Utility Functions*)


(* ::Subsubsection::Closed:: *)
(*Export Graph Constructor*)


(* ::Text:: *)
(*Given the elements, these functions (1) validate the elements, and (2) manipulate the elements to a form suitable for output to TGF format.*)


ConstructGraphFromEdgeRules[data_]:=
Module[
  {eR, eA, nL, nA},
  {eR, eA, nL, nA} = {"EdgeRules", "EdgeAttributes", "VertexList", "VertexAttributes"} /. data;
  If[ !ValidateElementSet[eR, eA, nL, nA], Return[$Failed] ];

  (* explicitly construct elements not explicitly declared *)
  If[ eA === "EdgeAttributes",   eA = Map[{}&, eR] ];
  If[ nL === "VertexList",       nL = Union[First/@eR, Last/@eR] ];
  If[ nA === "VertexAttributes", nA = Map[{}&, nL] ];

  WriteTGFGraph[eR, eA, nL, nA]
] 


ConstructGraphFromAdjMatrix[data_]:=
Module[
  {eR, eA, adjM, nL, nA},
  {adjM, nL, nA} = {"AdjacencyMatrix", "VertexList", "VertexAttributes"} /. data;
  If[ !ValidateElementSet[adjM, nL, nA], Return[$Failed] ];
  
  If[ nL === "VertexList", nL = Range[First[Dimensions[adjM]]] ];
  eR = AdjToEdgeRules[adjM] /. MapThread[Rule, {Range[First[Dimensions[adjM]]], nL}];

  (* explicitly construct elements not explicitly declared *)
  eA = Map[{}&, eR];
  If[ nA === "VertexAttributes", nA = Map[{}&, nL] ];

  WriteTGFGraph[eR, eA, nL, nA]
] 


ConstructGraphFromGraph[data_]:=
Module[
  {graph, eR, eA, nL, nA},
  graph = "Graph"/.data;
  If[ !GraphQ[graph], Message[Export::errelem, "Graph", "TGF"]; Return[$Failed] ];

  (* TGF only supports label *)
  nL = VertexList[graph];
  eR = EdgeList[graph];
  nA = Map[PropertyValue[{graph, #}, VertexLabels]&, nL]/.{_?PropertyValueQ -> ""};
  eA = Map[PropertyValue[{graph, #}, EdgeLabels]&, eR]/.{_?PropertyValueQ -> ""};
  Clear[graph];

  (* modify to work with WriteTGFGraph[] *)  
  nA = Map[{"Label"-> ToString[#]}&, nA];
  eA = Map[{"Label"-> ToString[#]}&, eA];
  eR = Rule@@@eR;

  WriteTGFGraph[eR, eA, nL, nA]
] 


WriteTGFGraph[eR_, eA_, nL_, nA_]:=
Module[
  {from, to, eLabel, nLabel, nodes, edges},

  nLabel = Map[ Function[x, ToString["Label"/.x] ], nA ] /. {"Label" -> ""};
  nodes = MapThread[ ToString[#1] ~~ " " ~~ #2&, {nL, nLabel} ];
  nodes = Apply[StringJoin, Riffle[nodes, "\n"]];

  from = Map[ ToString[First[#]]&, eR];
  to = Map[ ToString[Last[#]]&, eR];
  eLabel = Map[ Function[x, ToString["Label"/.x] ], eA ] /. {"Label" -> ""};
  edges = MapThread[#1 ~~ " " ~~ #2 ~~ " " ~~ #3 &, {from, to, eLabel} ];
  edges = Apply[StringJoin, Riffle[edges, "\n"]];
 
  nodes ~~ "\n#\n" ~~ edges ~~ "\n"
]


(* ::Subsubsection::Closed:: *)
(*AdjacencyMatrix to EdgeRules*)


(* convert AdjacencyMatrix to EdgeRules *)
AdjToEdgeRules[m_]:= Flatten[Map[convertToList, Most[ArrayRules[Normal[m]]]]]


convertToList[{a_, b_}-> c_Integer]:= Table[a -> b, {c}]


(* ::Subsubsection::Closed:: *)
(*SingleElementQ and ElementsQ*)


(* These two functions decide which of the overloaded exporters are called by the framework. *)


Attributes[FirstHeld] = {HoldAll}


FirstHeld[_[first_, ___]] := Hold[first]


Attributes[SingleElementQ] = {HoldFirst}


SingleElementQ[expr_] := SingleElementQ[expr, _]


SingleElementQ[expr:(_Rule|_RuleDelayed), elem_] := 
  MatchQ[FirstHeld[expr], Hold[elem]]


SingleElementQ[{expr:(_Rule|_RuleDelayed)}, elem_] := 
  SingleElementQ[expr, elem]


SingleElementQ[expr_, elem_] := False


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


(* ::Subsubsection::Closed:: *)
(*Validators*)


PropertyValueQ[x_]:=MatchQ[Head[x],PropertyValue] || SameQ[x, $Failed]


ValidateElementSet[eR_, eA_, nL_, nA_]:=(
  If[ !ValidateEdgeRules[eR],
      Message[Export::errelem, "EdgeRules", "TGF"];
      Return[False] 
  ];

  If[ (eA === "EdgeAttributes") && (nL === "VertexList") && (nA === "VertexAttributes"),
      Return[True]
  ];

  If[ eA =!= "EdgeAttributes",
      If[ !ValidateAttributes[eA],   Message[Export::errelem, "EdgeAttributes", "TGF"]; Return[False] ];
      If[ Length[eA] =!= Length[eR], Message[Export::errelem, "EdgeAttributes", "TGF"]; Return[False] ];
  ];

  If[ !ValidateVertexLabelAndAttributes[nL, nA],
      (* Error message given by ValidateVertexLabelAndAttributes *)
      Return[False]
  ];

  (* we demand that, if "VertexList" is declared, it must contain all the vertices in EdgeRules *)
  If[ nA =!= "VertexAttributes" && Complement[Union[First/@eR, Last/@eR], nL] =!= {}, 
      Message[Export::errelem, "VertexList", "TGF"];
      Return[False]
  ];
  
  (* all tests are passed at this point *)
  Return[True]
) 


ValidateElementSet[adjM_, nL_, nA_]:=(
  If[ !ValidateAdjMatrix[adjM],
      Message[Export::errelem, "AdjacencyMatrix", "TGF"];
      Return[False] 
  ];

  (* return True immediately when nothing to test *)
  If[ nL === "VertexList" && nA === "VertexAttributes",
      Return[True]
  ];

  If[ !ValidateVertexLabelAndAttributes[nL, nA],
      (* Error message given by ValidateVertexLabelAndAttributes *)
      Return[False]
  ];

  (* further check that VertexLabels matches AdjacencyMatrix *)
  If[ nL =!= "VertexList" && Length[nL] =!= First[Dimensions[adjM]],
      Message[Export::errelem, "VertexList", "TGF"];
      Return[False]
  ];
  
  (* all tests are passed at this point *)
  Return[True]
)


(* returns False if there are problems with VertexLabels and VertexAttributes *)
ValidateVertexLabelAndAttributes[nL_, nA_]:=(
  If[ nA =!= "VertexAttributes" && nL === "VertexList",
      Message[Import::general, "The element \"VertexLabels\" must be specified if the element \"VertexAttributes\" is specified."];
      Return[False]
  ];

  If[ (nL =!= "VertexList") && (!MatchQ[nL, {_Integer..}]),
      Message[Export::errelem, "VertexList", "TGF"];
      Return[False]
  ];

  If[ nA =!= "VertexAttributes",
      If[ !ValidateAttributes[nA],   Message[Export::errelem, "VertexAttributes", "TGF"]; Return[False] ];
      If[ Length[nA] =!= Length[nL], Message[Export::errelem, "VertexAttributes", "TGF"]; Return[False] ];
  ];

  (* all tests are passed at this point *)
  Return[True]
)


(* for the TGF, we must have a list of rules, each
   in the form of _Integer -> _Integer *)
ValidateEdgeRules[er_]:=
If[ !MatchQ[er, {__Rule}],
	False,
    MatchQ[Union[First /@ er, Last /@ er], {__Integer}]
]


(* For AdjacencyMatrix, we check two things:
   -- square matrix
   -- entries are positive integers
 *)
ValidateAdjMatrix[AdjM_] :=
MatrixQ[AdjM, (IntegerQ[#] && NonNegative[#])&] && SameQ @@ Dimensions[AdjM]


ValidateAttributes[attr_]:=
Apply[And,  Map[ MatchQ[#, {__Rule} | {}]&, attr] ]


(* ::Subsection::Closed:: *)
(*Overloaded Exporters*)


(********
Because we can register only one TGF-exporter, we overload the TGF exporters
so that we can export to the TGF file via different combinations of elements.
The functions in this seciton are called directly by the Import/Export framework.
 ******)


(* ::Subsubsection::Closed:: *)
(*Export with no explicit element*)


(* ::Text:: *)
(*This function is called when Export[] calls are made without specifying an export element.  For examples,*)
(*     Export[ filename.tgf, {1 -> 2, 2-> 3, 3-> 1}, "TGF"],*)
(*     Export[ filename.tgf, {{0,0}{1,0}}, "TGF"]*)
(* Note that we can export both via "EdgeRules" or "AdjacencyMatrix" element.*)
(* *)
(*In the examples above, this ExportTGF function receives respectively {1 -> 2, 2-> 3, 3-> 1} or  {{0,0}{1,0}} as inputs, and the function calls another version of ExportTGF, based on the type of input.*)


ExportTGF[filename_, data_, opts___]:=
Which[
  GraphQ[data],
  ExportTGF[filename, "Graph" -> data, opts],

  MatchQ[ data, {__Rule}],
  ExportTGF[filename, "EdgeRules" -> data, opts],

  MatrixQ[data],
  ExportTGF[filename, "AdjacencyMatrix" -> data, opts],

  True,
  Message[Export::type, Head[data], "TGF"];
  Return[$Failed]
]


(* ::Subsubsection::Closed:: *)
(*Export from the "EdgeRules" element*)


(* ::Text:: *)
(*This function is called when Export[] calls are made specifying "EdgeRules" and other elements.*)
(**)
(*For examples,*)
(*     Export[ filename.tgf, "EdgeRules -> {1 -> 2, 2-> 3, 3-> 1}, {"TGF", "Rules"}],*)
(*     Export[ filename.tgf, {{0,0}{1,0}}, {"TGF", "EdgeRules"]*)
(* *)
(*In the examples above, this ExportTGF function receives {"EdgeRules" ->  {1 -> 2, 2-> 3, 3-> 1} } as input.*)


ExportTGF[filename_, data_, opts___]:=
Module[
  {stream, out},
  If[!VerifyEdgeRulesData[data], Return[$Failed] ];
  out = ConstructGraphFromEdgeRules[data];
  If[ out =!= $Failed,
      stream = OpenWrite[filename];
      WriteString[stream, out];
      Quiet@Close[stream],
      Return[$Failed]
  ]
]/;ElementsQ[data, "EdgeRules"|"EdgeAttributes"|"VertexList"|"VertexAttributes"]


(* ::Subsubsection::Closed:: *)
(*Export from the "AdjacencyMatrix" element*)


(* ::Text:: *)
(*This function is called when Export[] calls are made specifying "AdjacencyMatrix" and other elements.*)
(**)
(*For examples,*)
(*     Export[ filename.tgf, "AdjacencyMatrix -> {{0,1},{1,1}}, {"TGF", "Rules"}],*)
(*     Export[ filename.tgf, {{0,1},{1,1}}, {"TGF", "AdjacencyMatrix"]*)
(* *)
(*In the examples above, this ExportTGF function receives {"AdjacencyMatrix" -> {{0,1},{1,1}} } as input.*)


ExportTGF[filename_, data_, opts___]:=
Module[
  {stream, out},
  If[ !ValidateAdjMatrix["AdjacencyMatrix"/.data], Return[$Failed] ];
  out = ConstructGraphFromAdjMatrix[data];
  If[ out =!= $Failed,
      stream = OpenWrite[filename];
      WriteString[stream, out];
      Quiet@Close[stream],
      Return[$Failed]
  ]
]/;ElementsQ[data, "AdjacencyMatrix"|"VertexList"|"VertexAttributes"]


(* ::Subsubsection::Closed:: *)
(*Export from the "Graph" element*)


(* ::Text:: *)
(*This function is called when Export[] calls are made specifying "Graph" element.*)
(**)
(*For examples,*)
(*     Export[ filename.tgf, "Graph" -> Graph[...], {"TGF", "Rules"}],*)
(*     Export[ filename.tgf, Graph[...], {"TGF", "AdjacencyMatrix"]*)
(* *)
(*In the examples above, this ExportTGF function receives {"Graph" -> Graph[...] } as input.*)


ExportTGF[filename_, data_, opts___]:=
Module[
  {stream, out},
  out = ConstructGraphFromGraph[data];
  If[ out =!= $Failed,
      stream = OpenWrite[filename];
      WriteString[stream, out];
      Quiet@Close[stream],
      Return[$Failed]
  ]
]/;SingleElementQ[data, "Graph"]


(* ::Section::Closed:: *)
(*END CONVERTER CONTEXT*)


End[]
