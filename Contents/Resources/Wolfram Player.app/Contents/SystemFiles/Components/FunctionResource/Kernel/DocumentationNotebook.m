(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: DocumentationNotebook *)
(* :Context: FunctionResource`DocumentationNotebook` *)
(* :Author: richardh@wolfram.com *)
(* :Date: 2018-11-21 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2018 Wolfram Research *)
(* :Keywords: *)
(* :Discussion: *)


BeginPackage[ "FunctionResource`DocumentationNotebook`" ];

ClearAll @@ Names[ $Context ~~ ___ ];

(* Exported symbols added here with SymbolName::usage *)
GetDocumentationNotebook;
SaveDocumentationNotebook;
ViewDocumentationNotebook;
LocalDocumentationAvailableQ;

Begin[ "`Private`" ];


uuidQ = ResourceSystemClient`Private`uuidQ;
resourceElementDirectory = ResourceSystemClient`Private`resourceElementDirectory;
cloudpath = ResourceSystemClient`Private`cloudpath;
fileExistsQ = ResourceSystemClient`Private`fileExistsQ;


(******************************************************************************)
(* ::Section::Closed:: *)
(*LocalDocumentationAvailableQ*)


LocalDocumentationAvailableQ[ KeyValuePattern[ "Documentation"          -> _ ] ] := True;
LocalDocumentationAvailableQ[ KeyValuePattern[ "DocumentationNotebook"  -> _ ] ] := True;
LocalDocumentationAvailableQ[ KeyValuePattern[ "DefinitionNotebook"     -> _ ] ] := True;

LocalDocumentationAvailableQ[ id_ ] :=
  Quiet @ TrueQ @ Or[
      fileExistsQ @ documentationSaveLocation[ "Local", id ],
      And[ repositoryResourceQ @ id,
           MatchQ[
               Check[
                   ResourceObject[ id ][ "DefinitionNotebookObject" ],
                   $Failed,
                   MessageName[ ResourceObject, "apierr" ]
               ],
               _CloudObject
           ]
      ]
  ];

LocalDocumentationAvailableQ[ ___ ] := False;



(******************************************************************************)
(* ::Section::Closed:: *)
(*GetDocumentationNotebook*)


GetDocumentationNotebook[ args___ ] :=
  getDocumentationNotebook @ args;


(******************************************************************************)
(* ::Section::Closed:: *)
(*SaveDocumentationNotebook*)


SaveDocumentationNotebook[ args___ ] :=
  saveDocumentationNotebook @ args;


(******************************************************************************)
(* ::Section::Closed:: *)
(*ViewDocumentationNotebook*)


ViewDocumentationNotebook[ args___ ] :=
  viewDocumentationNotebook @ args;


(******************************************************************************)
(* ::Section::Closed:: *)
(*Utilities*)



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*repositoryResourceQ*)


repositoryResourceQ[ KeyValuePattern[ "RepositoryLocation" -> URL[ url_ /; MatchQ[ url, $ResourceSystemBase ] ] ] ] :=
  True;

repositoryResourceQ[ _Association ] :=
  False;

repositoryResourceQ[ id_ ] :=
  With[ { info = Quiet @ ResourceObject[ id ][ All ] },
      repositoryResourceQ @ info /; AssociationQ @ info
  ];

repositoryResourceQ[ ___ ] :=
  False;


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*getDocumentationNotebook*)


getDocumentationNotebook[ rf_, loc:_LocalObject|_CloudObject ] :=
  If[ fileExistsQ @ loc,
      Replace[ Import[ loc, "NB" ], Except[ _Notebook ] :> $Failed ],
      createDocumentationNotebook @ rf
  ] // fixCellContextOptions;

getDocumentationNotebook[ rf_, locType:"Local"|"Cloud" ] :=
  getDocumentationNotebook[ rf, documentationSaveLocation[ locType, rf ] ];

getDocumentationNotebook[ rf_ ] :=
  getDocumentationNotebook[ rf, "Local" ];

getDocumentationNotebook[ ___ ] :=
  Missing[ "NotAvailable" ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createAndSaveDocumentationNotebook*)

createAndSaveDocumentationNotebook[ rf_, loc_ ] :=
  Module[ { docNB },
      docNB = createDocumentationNotebook @ rf;
      saveDocumentationNotebook[ rf, loc, docNB ];
      docNB
  ];

createAndSaveDocumentationNotebook[ ___ ] :=
  $Failed;



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*saveDocumentationNotebook*)


saveDocumentationNotebook[ rf_, locType:"Local"|"Cloud", docNB_ ] :=
  saveDocumentationNotebook[
      rf,
      documentationSaveLocation[ locType, rf ],
      docNB
  ];

saveDocumentationNotebook[ rf_, loc:_LocalObject|_CloudObject, docNB_ ] :=
  Export[ loc, docNB, "NB" ];

saveDocumentationNotebook[ rf_, loc_ ] :=
  saveDocumentationNotebook[
      rf,
      loc,
      createDocumentationNotebook @ rf
  ];

saveDocumentationNotebook[ rf_ ] :=
  saveDocumentationNotebook[ rf, "Local" ];

saveDocumentationNotebook[ ___ ] :=
  $Failed;



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*viewDocumentationNotebook*)


viewDocumentationNotebook[ id_, args___ ] :=
  Module[ { uuid, open },
      uuid = ResourceFunction[ id, "UUID" ];
      open = SelectFirst[ Notebooks[ ],
                          CurrentValue[ #, { TaggingRules, "ResourceFunctionID" } ] === uuid &
             ];
      SetSelectedNotebook @ open /; MatchQ[ open, _NotebookObject ]
  ];

viewDocumentationNotebook[ args___ ] :=
  With[ { nb = GetDocumentationNotebook @ args },
      NotebookPut @ nb /; MatchQ[ nb, _Notebook ]
  ];

viewDocumentationNotebook[ ___ ] :=
  Missing[ "NotAvailable" ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*documentationSaveLocation*)


documentationSaveLocation[ "Local", id_String? uuidQ ] :=
  LocalObject @ resourceElementDirectory[ id, "Documentation.nb" ];

documentationSaveLocation[ "Cloud", id_String? uuidQ ] :=
  CloudObject @ cloudpath @ resourceElementDirectory[ id, "Documentation.nb" ];

documentationSaveLocation[ locType_, id_ ] :=
  With[ { uuid = Quiet @ ResourceObject[ id ][ "UUID" ] },
      documentationSaveLocation[ locType, uuid ] /; uuidQ @ uuid
  ];

documentationSaveLocation[ ___ ] :=
  $Failed;



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createDocumentationNotebook*)


createDocumentationNotebook[ args___ ] :=
  With[ { nb = iCreateDocumentationNotebook @ args },
      (createDocumentationNotebook[ args ] = nb) /; MatchQ[ nb, _Notebook ]
  ];

createDocumentationNotebook[ ___ ] :=
  $Failed;


iCreateDocumentationNotebook[ rf_, opts: OptionsPattern[ Notebook ] ] :=

  Module[ { title, info, uuid, cells },

      title = Replace[ ResourceObject[ rf ][ "Name" ],
                       {
                           name_String :> name <> " (Documentation)",
                           ___ :> Automatic
                       }
              ];

      info = getInfo @ rf;
      uuid = Replace[ ResourceFunction[ rf, "UUID" ], Except[ _? uuidQ ] :> Inherited ];

      cells = Flatten[
          {
              createDocumentationAuthorCell[info],
              createDocumentationTitleCell[info],
              createDocumentationDescriptionCell[info],
              createDocumentationUsageCells[info],
              createDocumentationDetailsCells[info],
              createDocumentationExampleCells[info],
              createDocumentationCategoryCells[info],
              createDocumentationKeywordCells[info],
              createDocumentationRelatedSymbolsCells[info],
              createDocumentationSeeAlsoCells[info],
              createDocumentationCitationCells[info],
              createDocumentationRelatedLinkCells[info]
          }
      ];

      Notebook[
          cells,
          StyleDefinitions -> $documentationStylesheet,
          TaggingRules -> {
              "NotebookIndexQ" -> False,
              "ResourceType" -> "Function",
              "ResourceFunctionID" -> uuid
          },
          WindowTitle -> title,
          Saveable -> False,
          opts
      ] /; MatchQ[ cells, { __Cell } ]
  ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*fixCellContextOptions*)

fixCellContextOptions[ nb_ ] :=
  nb /. $cellContextRules;


$cellContextRules :=
  $cellContextRules =
    Dispatch @ Map[
        (sym_Symbol? FunctionResource`Private`symbolQ /; SymbolName[Unevaluated[sym]] === # :> #) &,
        {
            "MenuAnchor",
            "WholeCellGroupOpener",
            "ImageEditMode"
        }
    ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*$documentationStylesheet*)


$documentationStylesheet :=
  $documentationStylesheet = (
      Import @ FileNameJoin @ {
          DirectoryName[ FunctionResource`Private`$NotebookToolsDirectory, 2 ],
          "FrontEnd",
          "StyleSheets",
          "Wolfram",
          "FunctionResourceDocumentationStyles.nb"
      }
  );


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*getInfo*)


getInfo[ id_ ] :=
  With[ { info = Quiet @ findMissingInfo @ ResourceObject[ id ][ All ] },
      Append[ info, "ID" :> id ] /; AssociationQ @ info
  ];

getInfo[ ___ ] :=
  $Failed;



findMissingInfo[ info: KeyValuePattern @ { "Documentation" -> _ } ] :=
  info;

findMissingInfo[ info_Association ] :=
  Module[ { imported, rf },
      imported = importDefinitionNB @ findDefinitionNB @ info;
      rf = FunctionResource`DefinitionNotebook`Private`scrapeResourceFunction @ imported;
      Join[ info, ResourceObject[ rf ][ All ] ]
  ];

findDefinitionNB[ KeyValuePattern @ { "DefinitionNotebook" -> defNB_ } ] :=
  defNB;

findDefinitionNB[ info_ ] :=
  Module[ { loc },
      loc = Quiet @ Check[
          ResourceObject[ info ][ "DefinitionNotebookObject" ],
          Missing[ "NotAvailable" ],
          MessageName[ ResourceObject, "apierr" ]
      ];
      If[ MatchQ[ loc, _CloudObject|_Missing ],
          findDefinitionNB[ info ] = loc,
          loc
      ]
  ];


importDefinitionNB[ loc: _CloudObject|_LocalObject ] :=
  FunctionResource`UpdateDefinitionNotebook @ Import @ loc;

importDefinitionNB[ KeyValuePattern @ { "Data" -> bytes_ByteArray, "Format" -> fmt_ } ] :=
  FunctionResource`UpdateDefinitionNotebook @ ImportByteArray[ bytes, fmt ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createDocumentationAuthorCell*)



createDocumentationAuthorCell[___] := {};



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createDocumentationCategoryCells*)


createDocumentationCategoryCells[KeyValuePattern["Categories" -> categories:{__String}]] :=
  Cell[
      CellGroupData[
          Flatten[
              {
                  Cell["Categories", "MetadataSection"],
                  Map[
                      Function[
                          Cell[
                              BoxData[
                                  StyleBox[
                                      ToBoxes[
                                          Hyperlink[
                                              #1,
                                              StringJoin[
                                                  "https://resources.wolframcloud.com/FunctionRepository/category/",
                                                  #1,
                                                  "/"
                                              ]
                                          ]
                                      ],
                                      "Text"
                                  ]
                              ],
                              "MetadataItem"
                          ]
                      ],
                      categories
                  ]
              }
          ],
          Open
      ]
  ];

createDocumentationCategoryCells[___] := {};



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createDocumentationCitationCells*)


createDocumentationCitationCells[KeyValuePattern["SourceMetadata" -> KeyValuePattern["Citation" -> citation_]]] :=
  Cell[
      CellGroupData[
          Flatten[{Cell["Citation", "MetadataSection"], (Cell[#1, "MetadataItem"] & ) /@ Flatten[{citation}]}],
          Open
      ]
  ];

createDocumentationCitationCells[___] := {};



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createDocumentationDescriptionCell*)


createDocumentationDescriptionCell[KeyValuePattern["Description" -> desc_]] := {Cell[desc, "FunctionDescription"]};

createDocumentationDescriptionCell[___] := {};



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createDocumentationDetailsCells*)


createDocumentationDetailsCells[
    KeyValuePattern["Documentation" -> KeyValuePattern["Notes" -> {notes__Cell}]]
] :=
  Module[
      {details, thumbnail},
      details = With[
          {o1 = $opener1, o2 = $opener2},
          ReplaceRepeated[
              Cell[
                  CellGroupData[
                      {
                          Cell[
                              TextData[
                                  {
                                      Cell[
                                          BoxData[
                                              ToBoxes[
                                                  Dynamic[
                                                      If[
                                                          MatchQ[
                                                              CurrentValue[
                                                                  EvaluationNotebook[],
                                                                  {TaggingRules, "Openers", "PrimaryExamplesSection", "NotesSection"},
                                                                  Closed
                                                              ],
                                                              True | Open
                                                          ],
                                                          o1,
                                                          o2
                                                      ]
                                                  ]
                                              ]
                                          ]
                                      ],
                                      "\[NonBreakingSpace]",
                                      "Details and Options"
                                  }
                              ],
                              "PrimaryExamplesSection",
                              "WholeCellGroupOpener" -> True,
                              ShowGroupOpener -> False,
                              TaggingRules -> {"OpenerID" -> "Details and Options"}
                          ],
                          notes
                      },
                      Dynamic[
                          CurrentValue[
                              EvaluationNotebook[],
                              {TaggingRules, "Openers", "PrimaryExamplesSection", "NotesSection"},
                              Closed
                          ]
                      ]
                  ]
              ],
              (f:Cell | StyleBox)[a___, b:Except["Text"], "InlineFormula", c___] :> f[a, b, "Text", "InlineFormula", c]
          ]
      ];
      thumbnail = thumbnailButtonCell[
          Cell[
              CellGroupData[
                  {Cell["Details and Options", "PrimaryExamplesSection"], notes},
                  Closed
              ]
          ]
      ];
      {details, thumbnail, Cell["", "PageDelimiter",
          Editable -> False,
          CellFrame -> {{0, 0}, {1, 0}},
          ShowCellBracket -> False,
          CellMargins->{{24, 14}, {0, 0}},
          CellFrameMargins->{{0, 0}, {30, 0}},
          "CellElementSpacingsCellMinHeight" -> 0.,
          "CellElementSpacingsClosedCellHeight" -> 0.,
          Evaluatable -> False,
          CellGroupingRules -> {"SectionGrouping", 30},
          CellFrameColor -> GrayLevel[0.792157],
          CellSize -> {Automatic, 0}
      ]}
  ];

createDocumentationDetailsCells[___] := {};



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*thumbnailButtonCell*)


thumbnailButtonCell[
    details: Cell[
        CellGroupData[
            {Cell["Details and Options", "PrimaryExamplesSection", ___], __},
            _
        ]
    ]
] :=
  With[
      {
          thumbnail = Blend[{ImagePad[getThumbnail[details], 10, Padding -> White], White}],
          b = $blueLens,
          g = $grayLens
      },
      With[
          {dims = ImageDimensions[thumbnail]},
          Cell[
              BoxData[
                  ToBoxes[
                      Button[
                          MouseAppearance[
                              Dynamic[
                                  Overlay[
                                      {
                                          thumbnail,
                                          If[
                                              CurrentValue["MouseOver"],
                                              Overlay[
                                                  {ConstantImage[RGBColor[0, 0.666667, 1, 0.06], dims], b},
                                                  Alignment -> {-0.85, Center}
                                              ],
                                              Overlay[
                                                  {ConstantImage[RGBColor[0, 0.666667, 1, 0], dims], g},
                                                  Alignment -> {-0.85, Center}
                                              ]
                                          ]
                                      },
                                      Alignment -> {Left, Center}
                                  ]
                              ],
                              "LinkHand"
                          ],
                          CurrentValue[
                              EvaluationNotebook[],
                              {TaggingRules, "Openers", "PrimaryExamplesSection", "NotesSection"}
                          ] = Open,
                          Appearance -> None
                      ]
                  ]
              ],
              "NotesThumbnails",
              CellOpen -> Dynamic[
                  FEPrivate`Switch[
                      CurrentValue[
                          EvaluationNotebook[],
                          {TaggingRules, "Openers", "PrimaryExamplesSection", "NotesSection"}
                      ],
                      True,
                      False,
                      Open,
                      False,
                      _,
                      True
                  ]
              ]
          ]
      ]
  ];

thumbnailButtonCell[___] := Sequence[];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*getThumbnail*)


getThumbnail[Cell[CellGroupData[{Cell["Details and Options", "PrimaryExamplesSection", ___], cells___}, _]]] :=
  Module[
      {img},
      img = Quiet[ Rasterize[
          Notebook[
              {cells},
              Magnification -> 0.5,
              WindowSize -> {500, 300},
              StyleDefinitions -> $documentationStylesheet
          ],
          ImageResolution -> 36
      ], General::shdw ];
      ImageTake[img, UpTo[150]]
  ];

getThumbnail[___] := Sequence[];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*$blueLens*)


$blueLens :=
  $blueLens = Block[
      { },
      BinaryDeserialize[
          ByteArray[
              CompressedData[
                  "\n1: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\nbowERW4=\n"
              ]
          ]
      ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*$grayLens*)


$grayLens :=
  $grayLens = Block[
      { },
      BinaryDeserialize[
          ByteArray[
              CompressedData[
                  "\n1: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\n89+8jP1babmu+W79Cx9WBaxO2e66\n"
              ]
          ]
      ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createDocumentationExampleCells*)


createDocumentationExampleCells[KeyValuePattern[{"ExampleNotebook" -> nb_, "ID" :> id_}]] := createDocumentationExampleCells[nb, id];

createDocumentationExampleCells[Notebook[{cells___Cell}, ___], id_] :=
  createExampleGroups@{createDocumentationUsageExampleCells[cells], createDocumentationResourceExampleCells[id]};

createDocumentationExampleCells[nb_NotebookObject, id_] := createDocumentationExampleCells[NotebookGet[nb], id];

createDocumentationExampleCells[___] := {};



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createDocumentationResourceExampleCells*)


createDocumentationResourceExampleCells[id_] :=
  Cell[
      CellGroupData[
          {
              Cell["Resource Information", "PrimaryExamplesSection"],
              Cell[
                  TextData[
                      {
                          "Get the underlying ",
                          Cell[
                              BoxData[
                                  TagBox[
                                      ButtonBox[
                                          StyleBox[
                                              "ResourceObject",
                                              "SymbolsRefLink",
                                              ShowStringCharacters -> True,
                                              FontFamily -> "Source Sans Pro"
                                          ],
                                          BaseStyle -> Dynamic[
                                              FEPrivate`If[
                                                  CurrentValue["MouseOver"],
                                                  {"Link", FontColor -> RGBColor[0.854902, 0.396078, 0.145098]},
                                                  {"Link"}
                                              ]
                                          ],
                                          ButtonData -> "paclet:ref/ResourceObject"
                                      ],
                                      MouseAppearanceTag["LinkHand"]
                                  ]
                              ],
                              "Text",
                              "InlineFormula",
                              FontFamily -> "Source Sans Pro"
                          ],
                          ":"
                      }
                  ],
                  "ExampleText"
              ],
              Cell[
                  CellGroupData[
                      {
                          Cell[
                              BoxData[MakeBoxes[ResourceObject[id]]],
                              "Input",
                              "ExampleInput",
                              CellLabel -> "In[1]:="
                          ],
                          Cell[
                              BoxData[ToBoxes[ResourceObject[id]]],
                              "Output",
                              "ExampleOutput",
                              CellLabel -> "Out[1]="
                          ]
                      },
                      Open
                  ]
              ],
              Cell["View the examples:", "ExampleText"],
              Cell[
                  CellGroupData[
                      {
                          Cell[
                              BoxData[MakeBoxes[ResourceObject[id]["ExampleNotebook"]]],
                              "Input",
                              "ExampleInput",
                              CellLabel -> "In[2]:="
                          ],
                          Cell[BoxData[dummyNBObject[1]], "Output", "ExampleOutput", CellLabel -> "Out[2]="]
                      },
                      Open
                  ]
              ],
              Cell["View the definition notebook:", "ExampleText"],
              Cell[
                  CellGroupData[
                      {
                          Cell[
                              BoxData[MakeBoxes[ResourceObject[id]["DefinitionNotebook"]]],
                              "Input",
                              "ExampleInput",
                              CellLabel -> "In[3]:="
                          ],
                          Cell[BoxData[dummyNBObject[2]], "Output", "ExampleOutput", CellLabel -> "Out[3]="]
                      },
                      Open
                  ]
              ],
              Cell["Inspect the full metadata:", "ExampleText"],
              Cell[
                  CellGroupData[
                      {
                          Cell[
                              BoxData[MakeBoxes[Dataset[ResourceObject[id][All]]]],
                              "Input",
                              "ExampleInput",
                              CellLabel -> "In[4]:="
                          ],
                          Cell[
                              BoxData[ToBoxes[Dataset[ResourceObject[id][All]]]],
                              "Output",
                              "ExampleOutput",
                              CellLabel -> "Out[4]="
                          ]
                      },
                      Open
                  ]
              ],
              Sequence @@
                If[ $VersionNumber >= 12
                    ,
                    {
                        Cell["View resource information:", "ExampleText"],
                        Cell[
                            CellGroupData[
                                {
                                    Cell[
                                        BoxData[MakeBoxes[Information[id]]],
                                        "Input",
                                        "ExampleInput",
                                        CellLabel -> "In[5]:="
                                    ],
                                    Cell[
                                        BoxData[ToBoxes[Information[id]]],
                                        "Output",
                                        "ExampleOutput",
                                        CellLabel -> "Out[5]="
                                    ]
                                },
                                Open
                            ]
                        ]
                    }
                    ,
                    { }
                ]
          },
          Closed
      ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*dummyNBObject*)


dummyNBObject[n_] :=
  RowBox[
      {
          "NotebookObject",
          "[",
          PanelBox[
              GridBox[
                  {
                      {
                          StyleBox[
                              DynamicBox[FEPrivate`FrontEndResource["FEBitmaps", "ManipulatePasteIcon"]],
                              DynamicUpdating -> True
                          ],
                          StyleBox[StringJoin["\"Untitled-", ToString[n], "\""], FontColor -> GrayLevel[0.5]]
                      }
                  },
                  GridBoxAlignment -> {"Columns" -> {{Left}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}}
              ],
              FrameMargins -> {{4, 5}, {4, 4}}
          ],
          "]"
      }
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createDocumentationUsageExampleCells*)


createDocumentationUsageExampleCells[cells___] :=
  closeExampleCells[
      ReplaceRepeated[
          ReplaceAll[
              Cell[CellGroupData[{Cell["Examples", "PrimaryExamplesSection"], cells}, Open]],
              {
                  Cell[a__, "Subsection", b___] :> Cell[a, "ExampleSection", b],
                  Cell[a__, "Subsubsection", b___] :> Cell[a, "ExampleSubsection", b],
                  Cell[a__, "Text", b___] :> Cell[a, "ExampleText", b],
                  Cell[a__, "Input", b___] :> Cell[a, "Input", "ExampleInput", b],
                  Cell[a__, "Output", b___] :> Cell[a, "Output", "ExampleOutput", b]
              }
          ],
          {
              (f:Cell | StyleBox)[a___, b:Except["Text"], "InlineFormula", c___] :> f[a, b, "Text", "InlineFormula", c],
              {a___, Cell[_, "ExampleSection", ___], b:Cell[_, "ExampleSection", ___], c___} :> {a, b, c},
              {a___, Cell[_, "ExampleSection", ___]} :> {a}
          }
      ]
  ];


createExampleGroups[ expr_ ] :=
  expr /.
    c: Cell[CellGroupData[{Cell[_, "ExampleSubsubsection", ___ ], ___}, _]] :> createExampleGroup[c] /.
    c: Cell[CellGroupData[{Cell[_, "ExampleSubsection", ___ ], ___}, _]] :> createExampleGroup[c] /.
    c: Cell[CellGroupData[{Cell[_, "ExampleSection", ___ ], ___}, _]] :> createExampleGroup[c] /.
    c: Cell[CellGroupData[{Cell[_, "PrimaryExamplesSection", ___ ], ___}, _]] :> createExampleGroup[c]


$opener1 := $opener1 =
  Style[Graphics[{Thickness[0.18],
      RGBColor[0.8509803921568627, 0.396078431372549, 0],
      Line[{{-1.8, 0.5}, {0, 0}, {1.8, 0.5}}]}, AspectRatio -> 1,
      PlotRange -> {{-3, 4}, {-1, 1}}, ImageSize -> 20],
      Magnification -> 0.68 Inherited];
$opener2 := $opener2 =
  Rotate[Style[
      Graphics[{Thickness[0.18],
          RGBColor[0.8509803921568627, 0.396078431372549, 0],
          Line[{{-1.8, 0.5}, {0, 0}, {1.8, 0.5}}]}, AspectRatio -> 1,
          PlotRange -> {{-3, 4}, {-1, 1}}, ImageSize -> 20],
      Magnification -> 0.68 Inherited], Rational[1, 2] Pi, {-1.65, -1}];


createExampleGroup[
    Cell[
        CellGroupData[
            {
                Cell[
                    name_,
                    sec: Alternatives[
                        "PrimaryExamplesSection",
                        "ExampleSection",
                        "ExampleSubsection",
                        "ExampleSubsubsection"
                    ],
                    ___
                ],
                cells__
            },
            open_
        ]
    ]
] :=
  createExampleGroup[
      Association[
          "Label" -> name,
          "Style" -> sec,
          "Content" -> {cells},
          "Count" -> countExamples[{cells}],
          "Open" -> open
      ]
  ];

createExampleGroup[other_Cell] :=
  other;


createExampleGroup[
    info: KeyValuePattern @ {
        "Label" -> name_,
        "Style" -> style_,
        "Content" -> {content__},
        "Count" -> count_Integer,
        "Open" -> open_
    }
] :=
  With[{id = Hash[info, "Expression", "HexString"], o1 = $opener1, o2 = $opener2},
      Cell[CellGroupData[{
          Cell[TextData[{
              Cell[BoxData[ToBoxes[
                  Dynamic[
                      If[ MatchQ[CurrentValue[EvaluationNotebook[], {TaggingRules, "Openers", style, id}, open], True|Open],
                          o1,
                          o2
                      ]
                  ]
              ]]],
              "\[NonBreakingSpace]", name, "\[NonBreakingSpace]",
              If[ TrueQ @ Positive @ count,
                  Cell["(" <> ToString[count] <> ")", "ExampleCount",
                      FontSize -> 11, FontWeight -> "Plain",
                      FontColor -> GrayLevel[0.564706]],
                  ""
              ]
          }],
              style,
              "WholeCellGroupOpener" -> True,
              ShowGroupOpener -> False,
              TaggingRules -> { "OpenerID" -> id }
          ],
          content
      },
          Dynamic[
              CurrentValue[
                  EvaluationNotebook[], {TaggingRules, "Openers",
                  style, id}, open]]
      ]
      ]
  ];


createExampleGroup[KeyValuePattern["Content" -> {}]] := Sequence[];




countExamples[cells_] :=
  Module[{pageBreaks, subgroups},
      pageBreaks =
        Count[cells,
            Cell[CellGroupData[{Cell[_, "PageBreak"|"ExampleDelimiter", ___], __}, _]]];
      If[pageBreaks === 0,
          subgroups = Total@Cases[cells,
              Cell[
                  CellGroupData[{Cell[_,
                      "ExampleSection" | "ExampleSubsection" |
                        "ExampleSubsubsection", ___], c__}, _]] :> countExamples @ List @ c
          ];
          If[subgroups === 0,
              Count[cells, Cell[_, "Text" | "ExampleText", ___]],
              subgroups
          ]
          ,
          pageBreaks + 1
      ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*closeExampleCells*)


closeExampleCells[nb_] :=
  openFirst[
      ReplaceRepeated[
          nb,
          CellGroupData[{Cell[a___, b:"ExampleSection" | "ExampleSubsection", c___], d___}, Open] :> CellGroupData[{Cell[a, b, c], d}, Closed]
      ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*openFirst*)


openFirst[] := Sequence[];

openFirst[cells__] := (
    Apply[
        Sequence,
        ReplaceAll[
            {cells},
            FirstCase[
                {cells},
                group:CellGroupData[{Cell[a___, b:"ExampleSection" | "ExampleSubsection", c___], d___}, Closed] :> group :> CellGroupData[{Cell[a, b, c], openFirst[d]}, Open],
                { },
                Infinity
            ]
        ]
    ]);



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createDocumentationKeywordCells*)


createDocumentationKeywordCells[KeyValuePattern["Keywords" -> keywords:{__String}]] :=
  Cell[
      CellGroupData[
          Flatten[
              {
                  Cell["Keywords", "MetadataSection"],
                  Map[
                      Function[
                          Cell[
                              BoxData[
                                  StyleBox[
                                      ToBoxes[
                                          Hyperlink[
                                              #1,
                                              StringJoin[
                                                  "https://resources.wolframcloud.com/FunctionRepository/search?i=",
                                                  #1
                                              ]
                                          ]
                                      ],
                                      "Text"
                                  ]
                              ],
                              "MetadataItem"
                          ]
                      ],
                      keywords
                  ]
              }
          ],
          Open
      ]
  ];

createDocumentationKeywordCells[___] := {};

(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createDocumentationRelatedSymbolsCells*)


createDocumentationRelatedSymbolsCells[KeyValuePattern["RelatedSymbols" -> symbs:{__String}]] :=
  Cell[
      CellGroupData[
          Flatten[
              {
                  Cell["Related Symbols", "MetadataSection"],
                  Cell[
                      ReplaceAll[
                          TextData[
                              Riffle[
                                  Map[
                                      Function[
                                          Cell[
                                              BoxData[StyleBox[ToBoxes[relatedSymbolLink[#1]], "Text"]],
                                              "InlineFormula",
                                              FontWeight -> "DemiBold"
                                          ]
                                      ],
                                      symbs
                                  ],
                                  list[
                                      "\[NonBreakingSpace]",
                                      StyleBox[
                                          " \[FilledVerySmallSquare] ",
                                          "InlineSeparator"
                                      ],
                                      "\[NonBreakingSpace]"
                                  ]
                              ]
                          ],
                          list -> Sequence
                      ],
                      "SeeAlso"
                  ]
              }
          ],
          Open
      ]
  ];

createDocumentationRelatedSymbolsCells[___] := {};


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createDocumentationRelatedLinkCells*)


createDocumentationRelatedLinkCells[KeyValuePattern["ExternalLinks" -> links_List]] :=
  Module[
      {interpreted},
      interpreted = interpretLink /@ links;
      Cell[
          CellGroupData[
              Flatten[
                  {
                      Cell["Related Links", "MetadataSection"],
                      (Cell[BoxData[StyleBox[ToBoxes[#1], "Text"]], "MetadataItem"] & ) /@ interpreted
                  }
              ],
              Open
          ]
      ]
  ];

createDocumentationRelatedLinkCells[___] := {};



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*interpretLink*)


interpretLink[url_String] :=
  interpretLink[url] = Replace[
      Interpreter["URL"][url],
      {
          str_String :> Hyperlink[str],
          Failure[_, KeyValuePattern["Input" -> input_]] :> Hyperlink[Style[prettyTooltip[input, Row[{"Error: ", input, " is not a valid URL."}]], "Error"], url]
      }
  ];

interpretLink[Hyperlink[label_, url_String]] :=
  interpretLink[Hyperlink[label, url]] = Replace[
      Interpreter["URL"][url],
      {
          str_String :> Hyperlink[label, str],
          Failure[_, KeyValuePattern["Input" -> input_]] :> Hyperlink[Style[prettyTooltip[label, Row[{"Error: ", input, " is not a valid URL."}]], "Error"], url]
      }
  ];

interpretLink[Hyperlink[url_String]] := interpretLink[url];

interpretLink[Hyperlink[url_String, {url_String, None}]] := interpretLink[url];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createDocumentationSeeAlsoCells*)


createDocumentationSeeAlsoCells[KeyValuePattern["SeeAlso" -> related:{__String}]] :=
  Cell[
      CellGroupData[
          Flatten[
              {
                  Cell["Related Resources", "MetadataSection"],
                  Cell[
                      ReplaceAll[
                          TextData[
                              Riffle[
                                  Map[
                                      Function[
                                          Cell[
                                              BoxData[StyleBox[ToBoxes[seeAlsoLink[#1]], "Text"]],
                                              "InlineFormula",
                                              FontWeight -> "DemiBold"
                                          ]
                                      ],
                                      related
                                  ],
                                  list[
                                      "\[NonBreakingSpace]",
                                      StyleBox[
                                          " \[FilledVerySmallSquare] ",
                                          "InlineSeparator"
                                      ],
                                      "\[NonBreakingSpace]"
                                  ]
                              ]
                          ],
                          list -> Sequence
                      ],
                      "SeeAlso"
                  ]
              }
          ],
          Open
      ]
  ];

createDocumentationSeeAlsoCells[___] := {};



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*relatedSymbolLink*)


relatedSymbolLink[name_String] :=
  Hyperlink[ name, "paclet:ref/" <> name ];

relatedSymbolLink[name_] :=
  Hyperlink[name, None];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*seeAlsoLink*)


seeAlsoLink[name_String] :=
  Module[
      {link},
      link = Check[
          Block[
              {PrintTemporary},
              Quiet[
                  Replace[
                      Check[
                          Hyperlink[name, ResourceObject[name]["DocumentationLink"]],
                          $Failed,
                          MessageName[ResourceAcquire, "apierr"]
                      ],
                      Except[Hyperlink[name, _String | _URL]] :> Hyperlink[
                          Style[
                              prettyTooltip[name, StringJoin["Warning: no resource named \"", name, "\" found."]],
                              "Warning"
                          ],
                          None
                      ]
                  ],
                  MessageName[ResourceAcquire, "apierr"]
              ]
          ],
          $Failed
      ];
      If[
          MatchQ[link, _Hyperlink],
          seeAlsoLink[name] = link,
          Style[
              prettyTooltip[Hyperlink[name, None], StringJoin["Error: couldn't resolve resource from \"", name, "\"."]],
              "Error"
          ]
      ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createDocumentationTitleCell*)


createDocumentationTitleCell[KeyValuePattern[{"Name" -> name_String, "ID" :> rf_, "ContributorInformation" -> KeyValuePattern["ContributedBy" -> author_]}]] :=
  {Cell[BoxData[ToBoxes[clickToCopyTitle[name, ResourceFunction @ rf, author]]], "FunctionTitle"]};

createDocumentationTitleCell[info: KeyValuePattern[{"Name" -> name_String, "ID" :> rf_}]] :=
  {Cell[BoxData[ToBoxes[clickToCopyTitle[name, ResourceFunction @ rf]]], "FunctionTitle"]};

createDocumentationTitleCell[___] := {};



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*clickToCopyTitle*)


clickToCopyTitle[label_, expr_, author_: None] :=
  Module[
      {g1, g2},
      g1 = Grid[
          {
              {
                  $c1,
                  Style[
                      label,
                      FontFamily -> "Source Sans Pro Semibold",
                      FontSize -> 37,
                      FontWeight -> "DemiBold",
                      FontColor -> GrayLevel[0.2]
                  ]
              }
          },
          Alignment -> {Left, Center},
          Frame -> True,
          FrameStyle -> Directive[AbsoluteThickness[1], GrayLevel[254/255]],
          Spacings -> {{2, {}, 2}, {2, {}, 2}}
      ];
      g2 = Grid[
          {
              {
                  $c2,
                  Style[
                      label,
                      FontFamily -> "Source Sans Pro Semibold",
                      FontSize -> 37,
                      FontWeight -> "DemiBold",
                      FontColor -> Blend[{GrayLevel[0.2], White}]
                  ]
              }
          },
          Alignment -> {Left, Center},
          Frame -> True,
          FrameStyle -> Directive[AbsoluteThickness[1], GrayLevel[254/255]],
          Spacings -> {{2, {}, 2}, {2, {}, 2}}
      ];
      Grid[
          {
              {
                  Button[
                      MouseAppearance[
                          Mouseover[
                              g1,
                              Overlay[
                                  {
                                      g2,
                                      Framed[
                                          Style[
                                              Grid[{{$clipboard, "Copy to clipboard"}}, Alignment -> {Center, Center}],
                                              "Text",
                                              FontColor -> RGBColor["#898989"],
                                              FontSize -> 12,
                                              FontWeight -> "Plain"
                                          ],
                                          Background -> RGBColor["#f5f5f5"],
                                          FrameStyle -> RGBColor["#e5e5e5"],
                                          FrameMargins -> 8
                                      ]
                                  },
                                  All,
                                  Alignment -> Center
                              ]
                          ],
                          "LinkHand"
                      ],
                      CopyToClipboard[ExpressionCell[expr, "Input"]],
                      Appearance -> None
                  ],
                  If[ author =!= None,
                      Grid[
                          {
                              {
                                  Style[
                                      "Contributed By:",
                                      "Text",
                                      FontColor -> GrayLevel[0.5],
                                      FontSize -> 10,
                                      ShowStringCharacters -> False
                                  ]
                              },
                              {
                                  Style[
                                      author,
                                      "Text",
                                      FontSlant -> Italic,
                                      FontColor -> GrayLevel[0.25],
                                      FontSize -> 12,
                                      ShowStringCharacters -> False
                                  ]
                              }
                          },
                          Alignment -> {Left, Center},
                          Dividers -> {{1 -> GrayLevel[0.75]}, {1 -> GrayLevel[254/255], 3 -> GrayLevel[254/255]}},
                          Spacings -> {{3, {}}, {1.25, {}, 1.25}}
                      ],
                      Nothing
                  ]
              }
          },
          Alignment -> {Left, Center}
      ]
  ];


$c1 :=
  $c1 = Graphics[{Thickness[0.15], RGBColor["#fc6b34"], Circle[]}, ImageSize -> 20, PlotRange -> {All, {-1.2, 1.4}}];


$c2 :=
  $c2 = Graphics[
      {Thickness[0.15], Blend[{RGBColor["#fc6b34"], White}], Circle[]},
      ImageSize -> 20,
      PlotRange -> {All, {-1.2, 1.4}}
  ];


$clipboard :=
  $clipboard = Block[
      { },
      Show[
          BinaryDeserialize @ ByteArray["OEM6eJxLYylm9cxNTE89JMCsIsPCMApQQGdn539kPNTtA5opAMTn0e3BgkFqBKhgXwMRdsFwAwnmYoQT1G/vCZmF5Kb3MD8SCncc9hWgm4PDPmR3FZBiH5pYAbo+InABIXPx2EdsWsGZZkixjxpg1L6BsY+MdIk1/Q9W+6gFBqt9o/FHHBis9o3GH3FgsNo3Gn/EgVH76GsfrfBA2zdcQTCLU2VJahpTMUtQaU5qMZdzfk5+UXBBYnJqMHOQuxNcgsczryS1KCc1sSwzL72YJaSoNBUAu72+RA=="],
          ImageSize -> {Automatic, 19}
      ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createDocumentationUsageCells*)


createDocumentationUsageCells[KeyValuePattern["Documentation" -> KeyValuePattern["Usage" -> usage_]]] :=
  Map[
      Function[
          Cell[
              CellGroupData[
                  {
                      Cell[Replace[#Usage, BoxData[b_] :> BoxData[StyleBox[b, "Text"]]], "UsageInputs"],
                      Cell[#Description, "UsageDescription"]
                  },
                  Open
              ]
          ]
      ],
      usage
  ];

createDocumentationUsageCells[___] := {};


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*prettyTooltip*)


prettyTooltip[ label_, tooltip_ ] :=
  RawBoxes @
    TemplateBox[ { MakeBoxes @ label, MakeBoxes @ tooltip },
                 "PrettyTooltipTemplate"
    ];



(******************************************************************************)
(* ::Section::Closed:: *)
(*EndPackage*)


End[ ]; (* `Private` *)

EndPackage[ ];
