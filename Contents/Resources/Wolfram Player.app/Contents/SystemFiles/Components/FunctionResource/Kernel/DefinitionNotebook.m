(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: DefinitionNotebook *)
(* :Context: FunctionResource`DefinitionNotebook` *)
(* :Author: richardh@wolfram.com *)
(* :Date: 2018-09-26 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2018 Wolfram Research *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage[ "FunctionResource`DefinitionNotebook`" ];

ClearAll @@ Names[ $Context ~~ ___ ];

(* Exported symbols added here with SymbolName::usage *)

CheckDefinitionNotebook;
$DebugSuggestions = False;
DefinitionTemplateVersion;

Begin[ "`Private`" ];


(******************************************************************************)


CheckDefinitionNotebook // Options = {
    "AutoUpdate" -> True,
    "AutoUpdateOptions" -> { "CreateNewNotebook" -> False },
    "CloseNotebook" -> False
};

CheckDefinitionNotebook[ nbo_NotebookObject, opts: OptionsPattern[ ] ] :=
  Module[ { nb, dc, data, makeButton },
      nb = If[ TrueQ @ OptionValue[ "AutoUpdate" ],
               FunctionResource`UpdateDefinitionNotebook[
                   nbo,
                   Sequence @@ OptionValue[ "AutoUpdateOptions" ]
               ],
               nbo
           ];
      dc = DeleteCases[ CurrentValue[ nb, DockedCells ], Cell[ _, "StripeCell", ___ ] ];
      CurrentValue[ nb, DockedCells ] = dc;
      checkNotebook @ nb;
      If[ TrueQ @ OptionValue[ "CloseNotebook" ], NotebookClose @ nbo; NotebookClose @ nb ];
      data = $lastScrapedHints;
      makeButton = Function @ Append[
          #1,
          "View" ->
            With[ { c = First @ Cells[ nb, CellID -> #CellID ] },
                Button[ "View", moveToCell[ c, All, Cell ] ]
            ]
      ];
      Dataset @ Map[ makeButton, data ]
  ];


CheckDefinitionNotebook[ nb_Notebook, opts: OptionsPattern[ ] ] :=
  CheckDefinitionNotebook[ NotebookPut @ nb, opts ];

CheckDefinitionNotebook[ id_, opts: OptionsPattern[ ] ] :=
  With[ { nb = ResourceFunction[ id, "DefinitionNotebook" ] },
      CheckDefinitionNotebook[ nb, opts ] /; MatchQ[ nb, _NotebookObject ]
  ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*$defaults*)


$defaults := $defaults = <|
    "Name" -> "MyFunction",
    "Description" -> "One-line description explaining the function\[CloseCurlyQuote]s basic purpose",
    "Function" -> "FunctionRepositoryTemporary`MyFunction",
    "Usage" -> {
        Cell[
            CellGroupData[
                {
                    Cell[
                        BoxData[
                            RowBox[{"MyFunction", "[", StyleBox["arg", (FontSlant -> Italic) | "TI"], "]"}]
                        ],
                        "UsageInputs",
                        ___
                    ],
                    Cell[
                        TextData[
                            {
                                "explanation of what use of the argument ",
                                StyleBox["arg", (FontSlant -> Italic) | "TI"],
                                " does."
                            }
                        ],
                        "UsageDescription",
                        ___
                    ]
                },
                _
            ]
        ]
    },
    "Details & Options" -> { Cell[ "Additional information about usage and options.", "Notes", ___ ] },
    "Examples" -> {
        Cell["Text about the example:", "Text", ___],
        Cell[
            CellGroupData[
                {
                    Cell[
                        BoxData[RowBox[{"MyFunction", "[", RowBox[{"x", ",", "y"}], "]"}]],
                        "Input",
                        ___
                    ],
                    Cell[BoxData[RowBox[{"x", " ", "y"}]], "Output", ___]
                },
                _
            ]
        ],
        Cell[
            "Text about additional examples expanding scope (and see other sections for options, applications, etc):",
            "Text",
            ___
        ],
        Cell[
            CellGroupData[
                {
                    Cell[
                        BoxData[RowBox[{"MyFunction", "[", RowBox[{"x", ",", "y", ",", "z"}], "]"}]],
                        "Input",
                        ___
                    ],
                    Cell[
                        BoxData[RowBox[{"x", " ", "y", " ", "z"}]],
                        "Output",
                        ___
                    ]
                },
                _
            ]
        ],
        Cell["", "ResourceHiddenPageBreak", ___ ]
    },
    "ContributorInformation" -> <| "ContributedBy" -> "Author Name" |>,
    "Keywords" -> { "keyword 1" },
    "RelatedSymbols" -> { "SymbolName (documented Wolfram Language symbol)" },
    "SeeAlso" -> { "Resource Name (resources from any Wolfram repository)", "GrayCode (resources from any Wolfram repository)" },
    "ExternalLinks" -> { "Link to other related material" },
    "SourceMetadata" -> { "Source, reference or citation information" },
    "VerificationTests" -> HoldPattern[ VerificationTest[ MyFunction_[ x_, y_ ], Times[ x_, y_ ] ] /; SymbolName @ Unevaluated @ MyFunction === "MyFunction" ]
|>;



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*validate*)


$lastValidations = Internal`Bag[ ];


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*Name*)


validate0[ info: KeyValuePattern @ { "Property" -> "Name", "String" -> _String, "Result" -> renamed_String } ] /; Lookup[ info, "String" ] =!= Lookup[ info, "Result" ] :=
  Module[ { cellID },
      cellID = getFirstCellID @ info;
      collectHint[ cellID, "TitleRenamed" ];
      renamed
  ];


validate0[ info: KeyValuePattern @ { "Property" -> "Name", "String" -> name_String, "Result" -> $Failed } ] :=
  Module[ { cellID },
      cellID = getFirstCellID @ info;
      collectHint[ cellID, "TitleInvalid" ];
      $Failed
  ];


validate0[ info: KeyValuePattern @ { "Property" -> "Name", "Result" -> result: $defaults[ "Name" ] } ] :=
  Module[ { cellID },
      cellID = getFirstCellID @ info;
      collectHint[ cellID, "TitleNotSet" ];
      result
  ];


validate0[ info: KeyValuePattern @ { "Property" -> "Name", "Cells" -> cells_, "String" -> Except[ _? StringQ ], "Result" -> result_ } ] :=
  Module[ { cellID },
      cellID = getFirstCellID @ info;
      collectHint[ cellID, "TitleNotString" ];
      result
  ];


validate0[ info: KeyValuePattern @ { "Property" -> "Name", "String" -> name_String, "Result" -> result_ } ] :=
  Module[ { },
      If[ StringStartsQ[ name, _?LowerCaseQ ],
          collectHint[ getFirstCellID @ info, "TitleNotCapitalized" ]
      ];
      result
  ];



(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*Description*)


validate0[ info: KeyValuePattern @ { "Property" -> "Description", "Result" -> result: $defaults[ "Description" ] } ] :=
  Module[ { cellID },
      cellID = getFirstCellID @ info;
      collectHint[ cellID, "DescriptionNotSet" ];
      result
  ];

validate0[ info: KeyValuePattern @ { "Property" -> "Description", "Result" -> result_String } ] :=
  Module[ { trimmed },
      trimmed = StringTrim[ result, WhitespaceCharacter ];
      If[ trimmed === "",
          collectHint[ getFirstCellID @ info, "DescriptionMissing" ];
          Throw[ Missing[ ], $tag ]
      ];

      If[ StringMatchQ[ trimmed, _? LowerCaseQ ~~ ___ ],
          collectHint[ getFirstCellID @ info, "DescriptionNotCapitalized" ];
      ];

      trimmed
  ] ~Catch~ $tag;

validate0[ info: KeyValuePattern @ { "Property" -> "Description", "Result" -> result: Except[ _String ] } ] :=
  Module[ { },
      collectHint[ getFirstCellID @ info, "DescriptionNotString" ];
      ""
  ];



(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*Function*)


validate0[ info: KeyValuePattern @ { "Property" -> "Function", "SyntaxErrors" -> { Cell[ ___, CellID -> cellID_, ___ ], ___ }, "Result" -> result_ } ] :=
  Module[ {  },
      collectHint[ cellID, "DefinitionSyntax" ];
      result
  ];

validate0[
    info: KeyValuePattern @ { "Property" -> "Function", "UsedCells" -> {Cell[
        BoxData[RowBox[{RowBox[{"MyFunction", "[", "]"}], ":=", "xxxx"}]],
        "Input", ___, CellID -> id_, ___]}, "Result" -> result_ }
] :=
  Module[ {},
      collectHint[ id, "DefinitionNotSet" ];
      result
  ];


(* use target symbol directly instead of trivial proxy symbols *)
validate0[ info: KeyValuePattern @ {
    "Property" -> "Function",
    "Definition" -> Language`DefinitionList[
        HoldForm[ _ ] -> {
            OwnValues -> Alternatives[
                _ :> sym_Symbol? FunctionResource`Private`symbolQ,
                { _ :> sym_Symbol? FunctionResource`Private`symbolQ }
            ],
            Alternatives[
                Messages -> { Verbatim[ HoldPattern ][ MessageName[ _, "shdw" ] ] -> _ },
                PatternSequence[ ]
            ]
        }
    ]
} ] :=
  HoldComplete @ sym;





(* check for global options *)
validate0[ info: KeyValuePattern @ {
    "Property" -> "Function",
    "Definition" -> Language`DefinitionList[
        HoldForm[ _ ] -> {
            ___,
            DefaultValues -> {
                Verbatim[ HoldPattern ][
                    Options[ _ ]
                ] -> {
                    ___,
                    (Rule|RuleDelayed)[
                        sym_Symbol? FunctionResource`Private`symbolQ  /;
                          Context @ Unevaluated @ sym === "FunctionRepositoryTemporary`",
                        _
                    ],
                    ___
                }
            },
            ___
        }
    ],
    "Result" -> result_
} ] :=
  Module[ { cellID },
      cellID = getSectionCellID @ info;
      collectHint[ cellID, "DefinitionGlobalOption" ];
      result
  ];


validate0[ info: KeyValuePattern @ {
    "Property" -> "Function",
    "Metadata" -> KeyValuePattern[ "Name" -> name_ ],
    "Definition" -> def_? FunctionResource`Private`emptyDefinitionsQ,
    "Result" -> result_ }
] :=
  Module[ { cellID },
      cellID = getFirstCellID @ info;
      collectHint[ cellID, "DefinitionMissing", name ];
      result
  ];




(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*Usage*)

validate0[ info: KeyValuePattern @ {
    "Property" -> "Usage",
    "UnusedCellIDs" -> unusedIDs: { ___Integer },
    "Cells" -> cells_,
    "Result" -> result_
} ] :=
  Module[ { unusedID, missingDesc, missingIn, unusable, unformatted, unformattedArgs },
      unusedID = Alternatives @@ unusedIDs;
      missingDesc = Cases[ cells, Cell[ _, "UsageInputs", ___, CellID -> id: unusedID, ___ ] :> id, Infinity ];
      missingIn = Cases[ cells, Cell[ _, "UsageDescription", ___, CellID -> id: unusedID, ___ ] :> id, Infinity ];
      unusable = Complement[ unusedIDs, missingDesc, missingIn ];
      unformatted = findUnformattedUsageInputIDs @ info;
      unformattedArgs = findUnformattedUsageArgumentIDs @ info;
      Do[collectHint[cellID, "UsageMissingDescription"], {cellID, missingDesc}];
      Do[collectHint[cellID, "UsageMissingInput"], {cellID, missingIn}];
      Do[collectHint[cellID, "UnableToUseCell"], {cellID, unusable}];
      Do[collectHint[cellID, "UsageMissingFormatting"], {cellID, unformatted}];
      Do[collectHint[cellID, "DescriptionArgumentFormatting"], {cellID, unformattedArgs}];
      findThreeDotEllipsis @ cells;
      findDefaultUsageCells @ cells;
      findArgumentsThatNeedSubscripts @ cells;
      findDescriptionsThatNeedAPeriod @ cells;
      findUnformattedCode @ cells;
      result
  ];


validate0[ info: KeyValuePattern @ { "Property" -> "Usage", "Result" -> { } } ] :=
  Module[ { cellID },
      cellID = getFirstCellID @ info;
      collectHint[ cellID, "UsageMissing" ];
      { }
  ];


validate0[ info: KeyValuePattern @ {
    "Property" -> "Usage",
    "Metadata" -> KeyValuePattern[ "Name" -> name_ ],
    "Result" -> result_
} ] :=
  With[ { cellID = findCellIDWithWrongUsage @ info },
      (collectHint[ cellID, "UsageMissingSymbol", name ]; result) /; IntegerQ @ cellID
  ];

descriptionEndsInBadCharacterQ[
    Cell[
        caption_String /; StringEndsQ[ StringTrim @ caption, Except[ "." ] ],
        "UsageDescription",
        ___
    ]
] :=
  True;

descriptionEndsInBadCharacterQ[
    Cell[
        TextData @ {
            ___,
            caption_String /; StringEndsQ[ StringTrim @ caption, Except[ "." ] ]
        },
        "UsageDescription",
        ___
    ]
] :=
  True;

descriptionEndsInBadCharacterQ[
    cell: Cell[ TextData @ { ___, Except[ _String ] }, "UsageDescription", ___ ]
] :=
  MatchQ[
      FrontEndExecute @ ExportPacket[ cell, "PlainText" ],
      { str_String, ___ } /; StringEndsQ[ StringTrim @ str, Except[ "." ] ]
  ];

descriptionEndsInBadCharacterQ[ ___ ] :=
  False;


findDescriptionsThatNeedAPeriod[ cells_ ] :=
  Cases[
      cells,
      cell: Cell[ _, "UsageDescription", ___, CellID -> id_, ___ ] /;
        descriptionEndsInBadCharacterQ @ cell :>
          collectHint[ id, "UsageDescriptionNeedsPeriod" ],
      Infinity
  ];


findDefaultUsageCells[ cells: { Alternatives @@ $defaults[ "Usage" ] } ] :=
  collectHint[ getFirstCellID @ cells, "UsageNotSet" ];

findDefaultUsageCells[ cells_ ] :=
  Module[ { default, ids },
      default = Alternatives @@ $defaults[ "Usage" ];

      ids = Flatten @ Cases[
          cells,
          c: default :> Cases[ c, Cell[ ___, CellID -> id_, ___ ] :> id, Infinity ],
          Infinity
      ];

      Do[ collectHint[ id, "IgnoredPlaceholderContent" ], { id, ids } ];
  ];


findArgumentsThatNeedSubscripts[ cells_ ] :=
  Cases[
      cells,
      Cell[ a_ /; ! FreeQ[ a, str_String? maybeNeedsSubscriptQ ], ___, CellID -> id_, ___ ] :>
        collectHint[
            id,
            "MaybeNeedsSubscripts",
            With[ { s = DeleteDuplicates @ Cases[ a, s_String? maybeNeedsSubscriptQ :> s, Infinity ] },
                Sequence @@ { Row[ s, ", " ], s }
            ]
        ],
      Infinity
  ];

maybeNeedsSubscriptQ[ str_String ] :=
  StringMatchQ[ str, _?LowerCaseQ ~~ LetterCharacter ... ~~ DigitCharacter .. ];



findThreeDotEllipsis[ cells_ ] :=
  Module[ { cellIDs },
      cellIDs = Cases[
          cells,
          Cell[ a_, ___, CellID -> id_, ___ ] /; ! FreeQ[ a, "..." | s_String /; StringContainsQ[ s, "..." ] ] :> id,
          Infinity
      ];
      Do[ collectHint[ cellID, "ThreeDotEllipsis" ], { cellID, cellIDs } ]
  ];

findCellIDWithWrongUsage[ KeyValuePattern @ { "Metadata" -> KeyValuePattern[ "Name" -> name_ ], "Cells" -> cells_ } ] :=
  FirstCase[
      cells,
      Cell[ box_, "UsageInputs", ___, CellID -> id_, ___ ] /; FreeQ[ box, name|StringJoin["\"", name, "\""] ] :> id,
      Missing[ ],
      Infinity
  ];


findUnformattedUsageInputIDs[ KeyValuePattern[ "Cells" -> cells_ ] ] :=
  Cases[
      cells,
      Cell[
          BoxData @ RowBox @ row_? probablyNotFormattedQ,
          "UsageInputs",
          ___,
          CellID -> id_,
          ___
      ] :> id,
      Infinity
  ];

probablyNotFormattedQ[ { _, "[", "]" } ] :=
  False;

probablyNotFormattedQ[ { _, "[", "\[Ellipsis]", "]" } ] :=
  False;

probablyNotFormattedQ[ boxes_ ] :=
  FreeQ[
      boxes,
      Alternatives[
          ButtonBox,
          StyleBox[ _, "TI", ___ ],
          s_String /; StringContainsQ[ s, "\*" ~~ ___ ~~ "StyleBox[" ~~ __ ~~ "\"TI\"" ~~ ___ ~~ "]" ]
      ]
  ];


findUnformattedUsageArgumentIDs[ KeyValuePattern[ "Cells" -> cells_ ] ] :=
  Cases[ cells,
      Cell[
          CellGroupData[
              {
                  Cell[usage_BoxData, "UsageInputs", ___],
                  Cell[desc_, "UsageDescription", ___, CellID -> id_, ___]
              },
              _
          ]
      ] /; unformattedArgumentsInDescriptionQ[ usage, desc ] :> id,
      Infinity
  ];


findArguments[ (Cell|BoxData|TextData|RowBox)[ box_, ___ ] ] :=
  findArguments @ box;

findArguments[ list_List ] :=
  DeleteDuplicates @ Cases[ Flatten[ findArguments /@ list ], _String ];

findArguments[ SubscriptBox[
    StyleBox[ a_String, "TI", ___ ],
    StyleBox[ b_String, ___ ]
] ] :=
  StringJoin[ a, b ];

findArguments[ StyleBox[ a_String, "TI" ] ] :=
  a;


unformattedArgumentsInDescriptionQ[ usage_, desc_ ] :=
  Module[ { arguments, isFormatted, isItalics, isPlainText, inPlainText },
      arguments = Alternatives @@ findArguments @ usage;
      isFormatted := !FreeQ[desc, StyleBox[arguments, "TI", ___]];
      isItalics := !FreeQ[desc, StyleBox[arguments, FontSlant -> "Italic"]];
      isPlainText := StringQ[desc] && StringContainsQ[desc, WordBoundary~~arguments~~WordBoundary];
      inPlainText := Not[
          FreeQ[
              desc,
              TextData[
                  {
                      ___,
                      Condition[
                          s_String,
                          StringQ[s] && StringContainsQ[s, WordBoundary~~arguments~~WordBoundary]
                      ],
                      ___
                  }
              ]
          ]
      ];
      TrueQ @ And[
          ! isFormatted,
          isItalics || isPlainText || inPlainText
      ]
  ];



(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*Details & Options*)


validate0[ info: KeyValuePattern @ { "Property" -> "Details & Options", "Result" -> result: $defaults[ "Details & Options" ] } ] :=
  Module[ { cellID },
      cellID = getFirstCellID @ info;
      collectHint[ cellID, "IgnoredPlaceholderContent" ];
      Missing[ ]
  ];

validate0[ info: KeyValuePattern @ { "Property" -> "Details & Options", "Cells" -> cells_, "Result" -> result_ } ] :=
  Module[ { },
      findThreeDotEllipsis @ cells;
      findUnformattedCode @ cells;
      result
  ];


findUnformattedCode[ cells_ ] :=
  Module[ { unformatted },
      unformatted = Cases[
          cells,
          Cell[ a_, ___, CellID -> id_, ___ ] :> { id, syntaxCases @ a },
          Infinity
      ];
      Cases[
          unformatted,
          { id_, { params_, ___ } } :>
            collectHint[ id, "FoundUnformattedCode", params ]
      ]
  ];


$dictionaryNames := $dictionaryNames =
  Alternatives @@ Select[ Names[ "System`*" ], DictionaryWordQ ];

$probablySentenceStarters := $probablySentenceStarters =
  word_String /;
    And[ NameQ @ StringJoin[ "System`", word ],
         MatchQ[ word, $dictionaryNames ]
    ];


$wordCharacter = WordCharacter | "'" | "\[CloseCurlyQuote]";
$wordBoundary = StartOfString | EndOfString | Except[$wordCharacter];


syntaxCases[ string_String ] :=
  Module[ { noBoxes, codeCases, nameCases, sentences },
      noBoxes = StringDelete[ string, "\!\(" ~~ Longest[ __ ] ~~ "\)" ];
      codeCases = StringCases[
          noBoxes,
          WordBoundary ~~ a_?UpperCaseQ ~~ b : WordCharacter ... ~~ "[" ~~ Shortest[ c___ ] ~~ "]" /;
            SyntaxQ @ StringJoin[ a, b, "[", c, "]" ] :> StringJoin[ a, b, "[", c, "]" ]
      ];
      sentences = StringTrim @ StringSplit[ noBoxes, "."|"!"|"?" ];
      nameCases = Select[
          DeleteDuplicates @ Flatten @ Replace[
              StringSplit[ sentences, $wordBoundary ],
              { $probablySentenceStarters, rest___ } :> { rest },
              { 1 }
          ],
          NameQ @ StringJoin[ "System`", #1 ] &
      ];
      Join[ codeCases, nameCases ]
  ];

syntaxCases[ Cell[ box_, ___ ] ] :=
  syntaxCases @ box;

syntaxCases[ (BoxData|TextData)[ stuff_ ] ] :=
  syntaxCases @ stuff;

syntaxCases[ stuff_List ] :=
  Flatten[ syntaxCases /@ stuff ];

syntaxCases[ ___ ] := { };



(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*Examples*)


validate0[ info: KeyValuePattern @ { "Property" -> "Examples", "Cells" -> cells_, "Examples" -> examples_, "Result" -> result_ } ] :=
  Module[ { },
      findThreeDotEllipsis @ cells;
      findTextCellIDsThatEndInBadCharacter @ cells;
      findEmptyExampleSectionCellIDs @ cells;
      findMultipleOutputsFromOneInput @ cells;
      result
  ];



$exampleTextStyles = "Text"|"ExampleText"|"CodeText";

captionEndsInBadCharacterQ[ caption_String ] :=
  StringEndsQ[ StringTrim @ caption, Except[ ":" ] ];

captionEndsInBadCharacterQ[ Cell[ caption_, $exampleTextStyles, ___ ] ] :=
  captionEndsInBadCharacterQ @ caption;

captionEndsInBadCharacterQ[ TextData @ { ___, caption_ } ] :=
  captionEndsInBadCharacterQ @ caption;

captionEndsInBadCharacterQ[ StyleBox[ caption_, ___ ] ] :=
  captionEndsInBadCharacterQ @ caption;

captionEndsInBadCharacterQ[ ___ ] :=
  False;

findTextCellIDsThatEndInBadCharacter[ cells_ ] :=
  Module[ { cellIDs, $tag },
      cellIDs = Flatten @ Last @ Reap[
          cells //. {
              a___,
              cell: Cell[ _, $exampleTextStyles, ___, CellID -> id_, ___ ] /;
                captionEndsInBadCharacterQ @ cell,
              Alternatives[
                  Cell[ _, "Input", ___ ],
                  Cell @ CellGroupData[ { Cell[ _, "Input", ___ ], ___ }, _ ]
              ],
              b___
          } :> (Sow[ id, $tag ]; { a, b }), $tag ];
      Do[ collectHint[ cellID, "ExampleTextLastCharacter" ], { cellID, cellIDs } ];
  ];


$exampleSectionStyles =
  "Subsection" | "Subsubsection" | "Subsubsubsection" |
    "ExampleSection" | "ExampleSubsection" | "ExampleSubsubsection";

findEmptyExampleSectionCellIDs[ cells_ ] :=
  Module[ { cellIDs, $tag },
      cellIDs = Flatten @ Last @ Reap[
          cells //. {
              {
                  a___,
                  cell: Cell[ _, $exampleSectionStyles, ___, CellID -> id_, ___ ],
                  b: Cell[ _, $exampleSectionStyles, ___ ],
                  c___
              } :> (Sow[ id, $tag ]; { a, b, c }),
              {
                  a___,
                  cell: Cell[ _, $exampleSectionStyles, ___, CellID -> id_, ___ ]
              } :> (Sow[ id, $tag ]; { a })
      }, $tag ];
      Do[ collectHint[ cellID, "EmptyExampleSections" ], { cellID, cellIDs } ]
  ];


findMultipleOutputsFromOneInput[ cells_ ] :=
  Cases[
      cells,
      CellGroupData[
          {
              Cell[ BoxData[ _List ], OrderlessPatternSequence[ CellID -> id_, CellLabel -> _String, ___ ] ],
              ___,
              Cell[ _, "Output", ___ ],
              ___,
              Cell[ _, "Output", ___ ],
              ___
          },
          _
      ] :> collectHint[ id, "MultipleOutputs" ],
      Infinity
  ];



(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*ContributorInformation*)


validate0[ info: KeyValuePattern @ {
    "Property" -> "ContributorInformation",
    "Result" -> result: $defaults[ "ContributorInformation" ] | _Missing
} ] :=
  Module[ { cellID },
      cellID = getFirstCellID @ info;
      collectHint[ cellID, "AuthorNotSet" ];
      Missing[ ]
  ];


validate0[ info: KeyValuePattern @ {
    "Property" -> "ContributorInformation",
    "Result" -> result: KeyValuePattern[ "ContributedBy" -> author_ ],
    "Cells" -> cells_
} ] :=
  Module[ { },
      If[ ! StringQ @ author,
          collectHint[ getFirstCellID @ info, "AuthorNotString" ];
          Throw[ Missing[ ], $tag ]
      ];

      result
  ] ~Catch~ $tag;


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*Keywords*)


validate0[ KeyValuePattern @ {
    "Property" -> "Keywords",
    "Cells" -> cells_,
    "Result" -> result: { ___String? (Not @* StringFreeQ[LetterCharacter] ) }
} ] :=
  Module[ { },
      warnIfCommaSeparated @ cells;
      Replace[ DeleteCases[ result, Alternatives @@ $defaults[ "Keywords" ] ], { } -> Missing[ ] ]
  ];


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*RelatedSymbols*)


validate0[ KeyValuePattern @ { "Property" -> "RelatedSymbols", "Result" -> result: { ___String }, "Cells" -> cells_ } ] :=
  Module[ { strings },
      strings = DeleteCases[ result, Alternatives @@ $defaults[ "RelatedSymbols" ] ];

      Cases[ cells,
             Cell[ Except[ str_?documentedSystemNameQ, Alternatives @@ strings ],
                   ___,
                   CellID -> id_,
                   ___
             ] :> collectHint[ id, "NotADocumentedSymbol", str ],
             Infinity
      ];

      Cases[ cells,
             Cell[ Except[ str_?systemNameQ, Alternatives @@ strings ],
                   ___,
                   CellID -> id_,
                   ___
             ] :> collectHint[ id, "NotASystemSymbol", str ],
             Infinity
      ];

      Cases[ cells,
             Cell[ Except[ str_?validSymbolNameQ, Alternatives @@ strings ],
                   ___,
                   CellID -> id_,
                   ___
             ] :> collectHint[ id, "NotAValidSymbolName", str ],
             Infinity
      ];

      Replace[ strings, { } -> Missing[ ] ]
  ];


validSymbolNameQ[ name_String ] :=
  NameQ @ name || TrueQ @ Quiet[ ToExpression[ name, InputForm, FunctionResource`Private`symbolQ ] ];

validSymbolNameQ[ ___ ] :=
  False;


systemNameQ[ name_String ] :=
  Quiet @ TrueQ[ NameQ @ name && Context @ name === "System`" ];

systemNameQ[ ___ ] :=
  False;


documentedSystemNameQ[ (name_)? systemNameQ ] :=
  FileExistsQ @ FileNameJoin @ {
      $InstallationDirectory,
      "Documentation",
      "English",
      "System",
      "ReferencePages",
      "Symbols",
      StringJoin[ Last @ StringSplit[ name, "`" ], ".nb" ]
  };

documentedSystemNameQ[___] :=
  False;


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*SeeAlso*)


validate0[ KeyValuePattern @ {
    "Property" -> "SeeAlso",
    "Result" -> result: { ___String? (Not @* StringFreeQ[LetterCharacter] ) }
} ] := (
    Replace[ DeleteCases[ result, Alternatives @@ $defaults[ "SeeAlso" ] ], { } -> Missing[ ] ]
);


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*ExternalLinks*)


(* TODO: move url validation here? *)
validate0[ KeyValuePattern @ { "Property" -> "ExternalLinks", "Scraped" -> scraped_ } ] :=
  Module[ { validated },
      validated = validateLink /@ DeleteCases[ scraped, { _, Alternatives @@ $defaults[ "ExternalLinks" ] } ];
      Cases[ validated, { id_, Except[ _Hyperlink ] } :> collectHint[ id, "InvalidLink" ] ];
      Replace[ Cases[ validated, { _, link_Hyperlink } :> link ], { } -> Missing[ ] ]
  ];


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*SourceMetadata*)



validate0[ KeyValuePattern @ {
    "Property" -> "SourceMetadata",
    "Result" -> resultData_,
    "Cells" -> cells_
} ] :=
  Module[ { result },
      result = Replace[ resultData, KeyValuePattern[ "Citation" -> c_ ] :> c ];
      findNonStringsInCitation @ cells;
      findMultipleCitationCells @ cells;
      Replace[ ignoreDefaultCitation[ cells, result ],
          {
              { c__String? (Not @* StringFreeQ[LetterCharacter] ) } :> <| "Citation" -> { c } |>,
              { } -> Missing[ ]
          }
      ]
  ];


findNonStringsInCitation[ cells_ ] :=
  Cases[ cells,
         Cell[ Except[ _String ], "Item"|"Text", ___, CellID -> id_, ___ ] :>
           collectHint[ id, "CitationNotString" ],
         Infinity
  ];

ignoreDefaultCitation[ cells_, result_ ] :=
  Module[ { default, ids },
      default = Alternatives @@ $defaults[ "SourceMetadata" ];
      ids = Cases[ cells, Cell[ default, "Item"|"Text", ___, CellID -> id_, ___ ] :> id, Infinity ];
      Do[ collectHint[ id, "IgnoredPlaceholderContent" ], { id, ids } ];
      DeleteCases[ result, default ]
  ];

findMultipleCitationCells[ cells_ ] :=
  Module[ { ids },
      ids = Cases[ cells, Cell[ _String, "Item"|"Text", ___, CellID -> id_, ___ ] :> id, Infinity ];
      If[ Length @ ids > 1,
          Do[ collectHint[ id, "MultipleCitationCells", DeleteCases[ ids, id ] ], { id, ids } ]
      ]
  ];


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*VerificationTests*)


validate0[ KeyValuePattern @ { "Property" -> "VerificationTests", "Result" -> result: HoldComplete[ ___VerificationTest ] } ] :=
  Replace[ DeleteCases[ result, $defaults[ "VerificationTests" ] ], HoldComplete[ ] -> Missing[ ] ];


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*Other*)


validate0[ KeyValuePattern[ "Result" -> result_ ] ] :=
  result;

validate0[ ___ ] :=
  $Failed;



validate[ info_ ] :=
  Module[ { validated },
      findEmptyCells @ info;
      validated = validate0 @ info;
      Internal`StuffBag[ $lastValidations, Append[ info, "Validated" -> validated ] ];
      validated
  ];


validate[ ___ ] :=
  $Failed;



validateLink[ { id_, link_ } ] :=
  { id, validateLink @ link };

validateLink[url_String] :=
  validateLink[url] = Replace[
      Interpreter["URL"][url],
      {
          str_String :> Hyperlink[str],
          _ :> $Failed
      }
  ];

validateLink[Hyperlink[label_, url_String]] :=
  validateLink[Hyperlink[label, url]] = Replace[
      Interpreter["URL"][url],
      {
          str_String :> Hyperlink[label, str],
          _ :> $Failed
      }
  ];

validateLink[Hyperlink[url_String]] := validateLink[url];

validateLink[Hyperlink[url_String, {url_String, None}]] := validateLink[url];


warnIfCommaSeparated[ KeyValuePattern[ "UsedCells" -> cells_ ] ] :=
  warnIfCommaSeparated @ cells;

warnIfCommaSeparated[ KeyValuePattern[ "Cells" -> cells_ ] ] :=
  warnIfCommaSeparated @ cells;

warnIfCommaSeparated[ cells_ ] :=
  Cases[
      cells,
      Cell[ _String? (StringContainsQ[","]), "Item", ___, CellID -> id_, ___ ] :>
        collectHint[ id, "CommaSeparated" ],
      Infinity
  ];





$allowedEmptyStyles = Alternatives[
    "PageBreak",
    "ExampleDelimiter"
];

findEmptyCells[ KeyValuePattern[ "Cells" -> cells_ ] ] :=
  findEmptyCells @ cells;

findEmptyCells[ cells_ ] :=
  Cases[
      cells,
      Cell[
          s_String | BoxData[ s_String ] | TextData @ { s_String } /; StringTrim @ s === "",
          Except[ $allowedEmptyStyles ],
          ___,
          CellID -> id_,
          ___
      ] :> collectHint[ id, "EmptyCell" ],
      Infinity
  ];




(******************************************************************************)
(* ::Subsection::Closed:: *)
(*collectHint*)


collectHint[ cellID_, tag_, params___ ] :=
  Module[ { level },
      level = failureLevel @ tag;
      Internal`StuffBag[ $Hints, <| "CellID" -> cellID, "Level" -> level, "Tag" -> tag, "Parameters" -> Flatten @ { params } |> ];
      If[ level === "Error", Throw[ $Failed, scrapeAll ] ];
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*failureLevel*)


failureLevel[ tag_ ] :=
  If[ KeyExistsQ[ $hiddenStrings[ "Hints", "Suggestions" ], tag ],
      "Suggestion",
      Lookup[ $scrapeConfig, tag, "Warning" ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*$scrapeConfig*)


$scrapeConfig = <|
    "DefinitionSyntax" -> "Error",
    "TitleNotString" -> "Error",
    "TitleInvalid" -> "Error"
|>;


$submitConfig = <|
    "DefinitionSyntax" -> "Error",
    "TitleNotString" -> "Error",
    "TitleInvalid" -> "Error",
    "TitleNotSet" -> "Error",
    "DefinitionMissing" -> "Error",
    "DefinitionNotSet" -> "Error",
    "DefinitionUndefined" -> "Error",
    "DescriptionNotSet" -> "Error",
    "UsageNotSet" -> "Error",
    "UsageMissingSymbol" -> "Error",
    "AuthorNotSet" -> "Error",
    "InvalidLink" -> "Error"
|>;


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*getFirstCellID*)


getFirstCellID[ KeyValuePattern[ "Cells" -> expr: { __ } ] ] :=
  getFirstCellID @ expr;

getFirstCellID[ KeyValuePattern @ { "Property" -> prop_, "Cells" -> { } } ] :=
  getSectionCellID @ prop;

getFirstCellID[ expr_ ] :=
  FirstCase[ expr, Cell[ ___, CellID -> id_, ___ ] :> id, Missing[ ], Infinity ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*getSectionCellID*)


getSectionCellID[ KeyValuePattern[ "Property" -> prop_ ] ] :=
  getSectionCellID @ prop;

getSectionCellID[ prop_String ] :=
  Replace[ CurrentValue[ getCellObject[ $SourceNotebook,
                                        sectionFromProperty @ prop
                         ],
                         CellID
           ],
           Except[ _? IntegerQ ] :> Missing[ ]
  ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*getAllCellIDs*)


getAllCellIDs[ nb_NotebookObject ] :=
  Select[ CurrentValue[ Cells @ nb, CellID ], Positive ];

getAllCellIDs[ expr_ ] :=
  Cases[ expr, Cell[ ___, CellID -> id_? Positive, ___ ] :> id, Infinity ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*sectionFromProperty*)


sectionFromProperty[ "Function"               ] := "Definition";
sectionFromProperty[ "ContributorInformation" ] := "Contributed By";
sectionFromProperty[ "SeeAlso"                ] := "Related Resource Objects";
sectionFromProperty[ "SourceMetadata"         ] := "Source/Reference Citation";
sectionFromProperty[ "ExternalLinks"          ] := "Links";
sectionFromProperty[ "VerificationTests"      ] := "Tests";

sectionFromProperty[ prop_ ] := prop;


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*collectHints*)


collectHints // Attributes = { HoldFirst };

collectHints[ eval_ ] :=
  Block[ { $Hints = Internal`Bag[ ] },
      <|
          "Result" -> eval,
          "Data" -> Internal`BagPart[ $Hints, All ]
      |>
  ];

collectHints[ ___ ] :=
  $Failed;



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*$hiddenStrings*)


$hiddenStrings := $hiddenStrings =
  <|
      "MoreInfoCells" ->
        <|
            "Definition"                -> "Define your function using the name above. All definitions, including dependencies, will be included in the resource function when it is generated. Additional cells can be added and definitions can be given for multiple input cases.\n\nThis section should be evaluated before evaluating creating the Examples section below.",
            "Usage"                     -> "Document every accepted input usage case. Use Enter to create new cases as needed.\n\nEach usage should contain a brief explanation saying what the function does for the given input structure.\n\nSee existing documentation pages for examples.",
            "Details & Options"         -> "Give a detailed explanation of how the function is used. Add multiple cells including tables and hyperlinks as needed. Typical information includes: acceptable inputs, result formats, options specifications, and background information.",
            "Examples"                  -> "Demonstrate how to use the function. Examples should start with the most basic use case. Each example should be described using text cells. Use \"Subsection\" and \"Subsubsection\" cells to group examples as needed.\n\nSee existing documentation pages for examples.",
            "Contributed By"            -> "Name of the person, people or organization that should be publicly credited with contributing the function.",
            "Keywords"                  -> "List relevant terms that should be used to include this resource in search results.",
            "Related Symbols"           -> "List related Wolfram Language symbols. Include up to twenty documented, system-level symbols.",
            "Related Resource Objects"  -> "Names of published resource objects from any Wolfram repository that are related to this resource.",
            "Source/Reference Citation" -> "Citation for original source of the function or its components. For example, original publication of an algorithm or public code repository.",
            "Links"                     -> "URLs or hyperlinks for external information related to the function.",
            "Tests"                     -> "Optional list of tests that can be used to verify that the function is working properly in any environment.\nTests can be specified as Input/Output cell pairs or as literal VerificationTest expressions if you need to specify options.",
            "Submission Notes"          -> "Enter any additional information that you would like to communicate to the reviewer here. This section will not be included in the published resource."
        |>
      ,
      "Tooltips" ->
        <|
            "Check Notebook"        -> "Check notebook for potential errors",
            "ClickToDismissMessage" -> "Click to dismiss this message",
            "ClickToIgnoreMessage"  -> "Don't show again",
            "Editable Version"      -> "Return to editing this notebook",
            "Insert Delimiter"      -> "Insert example delimiter",
            "Literal Input"         -> "Format selection as literal Wolfram Language code",
            "MoreInfoButton"        -> "Click for more information",
            "Open Sample"           -> "View a completed sample definition notebook",
            "Style Guidelines"      -> "View general guidelines for authoring resource functions",
            "Submit to Repository"  -> "Submit your function to the Wolfram Function Repository",
            "Table Functions"       -> "Table Functions",
            "Template Input"        -> "Format selection automatically using appropriate documentation styles",
            "Tools"                 -> "Toggle documentation toolbar",
            "ViewSuggestions"       -> "View suggestions",
            "WarningCountButton"    -> "Potential issues found",
            "Insert2ColTable"       -> "Insert table with two columns",
            "Insert3ColTable"       -> "Insert table with three columns",
            "AddRow"                -> "Add a row to the selected table",
            "SortTable"             -> "Sort the selected table",
            "MergeTables"           -> "Merge selected tables"
        |>
      ,
      "MessageDialogs" ->
        <|
            "CellBracketIsSelected"            -> "A cell bracket is selected.",
            "CreatePreviewFailed"              -> "Couldn't create preview notebook.",
            "CreateResourceFailed"             -> "Couldn't create ResourceFunction.",
            "CursorBetweenCells"               -> "The cursor is between cells.",
            "CursorOutsideTableCell"           -> "The cursor must be inside a table cell.",
            "CursorOutsideTableCellOrBracket"  -> "The cursor must be inside a table cell or selecting the cell bracket of a table cell.",
            "MissingSourceNotebookFile"        -> "Source notebook file `1` not found.",
            "MissingSourceNotebookObject"      -> "Source notebook not found.",
            "MultipleCellsHaveBeenSelected"    -> "Multiple cells have been selected.",
            "NoResourceCreateNotebook"         -> "There is no Resource Create notebook.",
            "NoSelectedCells"                  -> "There are no selected cells.",
            "NotAllSelectedAreTableNotes"      -> "Not all selected cells have style \"TableNotes\".",
            "NotAllSelectedContainTables"      -> "Some of the selected cells do not contain tables.",
            "NotImplemented"                   -> "Coming soon.",
            "NumberOfTableColumns"             -> "The number of columns in the table is < the integer in the argument of FRTableSort.",
            "SelectedIncompatibleColumnCounts" -> "The selected tables must have the same number of columns.",
            "SelectedUnhandled"                -> "One or more \"InlineFormula\" expressions in the selected cell have forms which cannot be handled by this function.",
            "SelectionHasUnhandledForm"        -> "The expression in the selection has a form which cannot be handled by this function.",
            "UnhandledSortStructure"           -> "The structure of the cell is unsuitable for sorting.",
            "UnhandledTableSortColumn"         -> "One or more of the table elements in the column TableSort\nis using to decide how to sort is unsupported by TableSort.",
            "UpdateOldNotebookQ"               -> "This notebook appears to be created from an older version, so some features may not work as expected. Do you wish to try automatically updating this notebook?"
        |>
      ,
      "Hints" ->
        <|
            "Warnings" ->
              <|
                  "TitleMissing"   -> "Missing title",
                  "TitleNotSet"    -> "Name not set",
                  "TitleNotString" -> "The title should be a plain string (avoid using formatting)",
                  "TitleRenamed"   -> "Names are automatically converted to camel-case",
                  "TitleInvalid"   -> "The title is not a valid symbol name",
                  "TitleUnknown"   -> "Unable to determine title"
                  ,
                  "ExamplesMissing" -> "Missing examples section",
                  "ExamplesNotSet"  -> "Examples section not set",
                  "ExamplesUnknown" -> "Unable to get examples section"
                  ,
                  "DefinitionMissing"   -> "No definition found for `1`",
                  "DefinitionNotSet"    -> "Definition not set",
                  "DefinitionSyntax"    -> "Check your syntax and try again",
                  "DefinitionUndefined" -> "Cannot find definition for `1`",
                  "DefinitionUnknown"   -> "Unable to get definition"
                  ,
                  "DescriptionMissing"   -> "Missing description",
                  "DescriptionNotSet"    -> "Description not set",
                  "DescriptionNotString" -> "The description should be a plain string (avoid using formatting)",
                  "DescriptionUnknown"   -> "Unable to get description"
                  ,
                  "UsageMissing"            -> "Missing usage section",
                  "UsageNotSet"             -> "Usage section not set",
                  "UsageMissingSymbol"      -> "Missing usage for `1`",
                  "UsageMissingInput"       -> "Missing input pattern for this description",
                  "UsageMissingDescription" -> "Missing description for this input pattern",
                  "UsageUnknown"            -> "Unable to get usage"
                  ,
                  "AuthorNotSet"    -> "No author information given",
                  "AuthorNotString" -> "The author should be given as a plain string (avoid using formatting)"
                  ,
                  "CitationNotString"     -> "The citation should be given as a plain string (avoid using formatting)",
                  "MultipleCitationCells" -> "There should only be one citation cell"
                  ,
                  "NotASystemSymbol"     -> "No symbol named `1` found in System`",
                  "NotADocumentedSymbol" -> "No documentation found for `1`",
                  "NotAValidSymbolName"  -> "\"`1`\" is not a valid symbol name"
                  ,
                  "InvalidLink" -> "Invalid link"
                  ,
                  "UnableToUseCell" -> "Cell not used"
                  ,
                  "EmptyCell" -> "Cell is empty"
                  ,
                  "CommaSeparated" -> "Each item should be in its own cell"
              |>
            ,
            "Errors" ->
              <|
                  "TitleMissing"   -> "Missing title",
                  "TitleNotSet"    -> "Please give your function a name",
                  "TitleNotString" -> "The title should be a plain string (avoid using formatting)",
                  "TitleRenamed"   -> "Names are automatically converted to camel-case",
                  "TitleInvalid"   -> "The title is not a valid symbol name",
                  "TitleUnknown"   -> "Unable to determine title"
                  ,
                  "ExamplesMissing" -> "Missing examples section",
                  "ExamplesNotSet"  -> "Examples section not set",
                  "ExamplesUnknown" -> "Unable to get examples section"
                  ,
                  "DefinitionMissing"   -> "No definition found for `1`",
                  "DefinitionNotSet"    -> "Definition not set",
                  "DefinitionSyntax"    -> "Check your syntax and try again",
                  "DefinitionUndefined" -> "Cannot find definition for `1`. Check your symbol names and try again",
                  "DefinitionUnknown"   -> "Unable to get definition"
                  ,
                  "DescriptionMissing"   -> "Missing description",
                  "DescriptionNotSet"    -> "Please give your function a description",
                  "DescriptionNotString" -> "The description should be a plain string (avoid using formatting)",
                  "DescriptionUnknown"   -> "Unable to get description"
                  ,
                  "UsageMissing"            -> "Missing usage section",
                  "UsageNotSet"             -> "Usage section not set",
                  "UsageMissingSymbol"      -> "Missing usage for `1`",
                  "UsageMissingInput"       -> "Missing input pattern for this description",
                  "UsageMissingDescription" -> "Missing description for this input pattern",
                  "UsageUnknown"            -> "Unable to get usage"
                  ,
                  "AuthorNotSet" -> "No author information given"
                  ,
                  "NotASystemSymbol"     -> "No symbol named `1` found in System`",
                  "NotADocumentedSymbol" -> "No documentation found for `1`",
                  "NotAValidSymbolName"  -> "\"`1`\" is not a valid symbol name"
                  ,
                  "InvalidLink" -> "Invalid link"
                  ,
                  "UnableToUseCell" -> "Unable to use cell"
                  ,
                  "CommaSeparated" -> "Each item should be in its own cell"
              |>
            ,
            "Suggestions" ->
              <|
                  "TitleNotCapitalized"           -> "Resource function names typically start with a capital letter",
                  "DescriptionNotCapitalized"     -> "The description should usually start with a capital letter",
                  "ExamplesLargeImages"           -> "This cell has very large images. Consider reducing image sizes to improve performance",
                  "ExamplesSaveDefinitions"       -> "It's usually a good idea to use SaveDefinitions -> True with Manipulate",
                  "ExamplesEmptyCells"            -> "Empty example cells should usually be removed",
                  "DeleteUnused"                  -> "This cell can be deleted if not being used",
                  "UsageMissingFormatting"        -> "Arguments are not formatted",
                  "DescriptionArgumentFormatting" -> "Arguments should also be formatted in the description",
                  "EmptyExampleSections"          -> "Empty example sections can be removed",
                  "ExampleTextLastCharacter"      -> "Example captions should usually end in a colon when appearing before input",
                  "ThreeDotEllipsis"              -> "An ellipsis (\[Ellipsis]) should be written as the single character \\[Ellipsis]",
                  "FoundUnformattedCode"          -> "\"`1`\" appears to be unformatted code",
                  "IgnoredPlaceholderContent"     -> "Default placeholder content is not used and can be removed",
                  "MaybeNeedsSubscripts"          -> "`1` should probably be written with subscripts",
                  "UsageDescriptionNeedsPeriod"   -> "Usage descriptions should end in a period (.)",
                  "MultipleOutputs"               -> "Inputs should be separated so they each generate at most a single output",
                  "DefinitionGlobalOption"        -> "Avoid using global symbols for option names (use strings or System` symbols instead)"
              |>
        |>
  |>;


(* ::Subsection::Closed:: *)
(*$defaultActions*)

$defaultActions := {
    <| "Label" -> "Dismiss this message", "Function" -> clearError |>,
    <| "Label" -> "Don't show again", "Function" -> doNotShowAgain |>,
    If[ TrueQ @ $DebugSuggestions,
        <| "Label" -> "Debug the thing", "Function" -> putDebugNotebook |>,
        Nothing
    ]
    ,
    Delimiter
};


(*
Methods for automatically applying suggestions can be declared here.
*)
$customHintActions := $customHintActions = <|
    "TitleNotCapitalized" -> {
        <|
            "Label" -> "Capitalize name",
            "Function" -> capitalizeName
        |>
    }
    ,
    "TitleNotString" -> {
        <|
            "Label" -> "Remove formatting",
            "Function" -> convertToPlainText
        |>
    }
    ,
    "DescriptionNotString" -> {
        <|
            "Label" -> "Remove formatting",
            "Function" -> convertToPlainText
        |>
    }
    ,
    "DescriptionNotCapitalized" -> {
        <|
            "Label" -> "Capitalize description",
            "Function" -> capitalizeName
        |>
    }
    ,
    "UsageMissingFormatting" -> {
        <|
            "Label" -> "Format automatically",
            "Function" -> autoFormatUsage
        |>,
        <|
            "Label" -> "Clear formatting",
            "Function" -> Function[
                SelectionMove[ #ParentCellObject, All, CellContents ];
                FrontEndTokenExecute[ "ClearCellOptions" ];
            ]
        |>
    }
    ,
    "DescriptionArgumentFormatting" -> {
        <|
            "Label" -> "Format arguments automatically",
            "Function" -> autoFormatArguments
        |>
    }
    ,
    "MaybeNeedsSubscripts" -> {
        <|
            "Label" -> "Rewrite with subscripts",
            "Function" -> rewriteWithSubscripts
        |>
    }
    ,
    "UnableToUseCell" -> {
        <|
            "Label" -> "Delete unused cell",
            "Action" :> NotebookDelete @ ParentCell @ EvaluationCell[ ]
        |>
    }
    ,
    "DeleteUnused" -> {
        <|
            "Label" -> "Delete unused cell",
            "Action" :> NotebookDelete @ ParentCell @ EvaluationCell[ ]
        |>
    }
    ,
    "EmptyCell" -> {
        <|
            "Label" -> "Delete cell",
            "Action" :> NotebookDelete @ ParentCell @ EvaluationCell[ ]
        |>
    }
    ,
    "IgnoredPlaceholderContent" -> {
        <|
            "Label" -> "Delete cell",
            "Action" :> NotebookDelete @ ParentCell @ EvaluationCell[ ]
        |>
    }
    ,
    "EmptyExampleSections" -> {
        <|
            "Label" -> "Delete unused cell",
            "Action" :> NotebookDelete @ ParentCell @ EvaluationCell[ ]
        |>
    }
    ,
    "ExampleTextLastCharacter" -> {
        <| "Function" -> addColonToText |>
    }
    ,
    "CommaSeparated" -> {
        <|
            "Label" -> "Split into separate cells",
            "Function" -> splitItemCell
        |>
    }
    ,
    "ThreeDotEllipsis" -> {
        <|
            "Label" -> "Replace \"...\" with \\[Ellipsis] character",
            "Function" -> replaceEllipsis
        |>
    }
    ,
    "FoundUnformattedCode" -> {
        <|
            "Label" -> "Auto format the code",
            "Function" -> autoFormatCode
        |>
    }
    ,
    "CitationNotString" -> {
        <|
            "Label" -> "Remove formatting",
            "Function" -> convertToPlainText
        |>
    }
    ,
    "MultipleCitationCells" -> {
        <|
            "Label" -> "Remove this cell",
            "Action" :> NotebookDelete @ ParentCell @ EvaluationCell[ ]
        |>,
        <|
            "Label" -> "Remove other cells",
            "Function" -> Function @ NotebookDelete @
              Cells[ #NotebookObject,
                     CellID -> Alternatives @@ #MessageParameters
              ]
        |>
    }
    ,
    "UsageDescriptionNeedsPeriod" -> {
        <| "Function" -> addPeriodToText |>
    }
    ,
    "MultipleOutputs" -> {
        <|
            "Label" -> "Split inputs into separate cells",
            "Function" -> splitInputOutputGroup
        |>
    }
|>;


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*Suggestion Helper Functions*)


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*splitInputOutputGroups*)


splitInputOutputGroup[
    info: KeyValuePattern @ {
        "ParentCellObject" -> cell_CellObject,
        "NotebookObject" -> nb_
    }
] :=
  Catch[
      Module[
          {
              firstNum,
              group,
              inputCell,
              inputBoxes,
              inputCellArgs,
              newInputCells,
              outCells,
              newCells
          },
          firstNum = Replace[inputNumber[cell], $Failed :> Throw[$Failed, $tag]];
          SelectionMove[cell, All, CellGroup];
          group = Replace[
              NotebookRead[nb],
              Except[
                  Cell[
                      CellGroupData[
                          {Cell[BoxData[_List], ___, CellLabel -> _String, ___], _Cell, __Cell},
                          _
                      ]
                  ]
              ] :> Throw[$Failed, $tag]
          ];
          inputCell = DeleteCases[
              Replace[group, Cell[CellGroupData[{c_Cell, ___}, ___], ___] :> c],
              TaggingRules | CellLabel | CellID -> _
          ];
          inputBoxes = Replace[
              inputCell,
              {
                  Cell[BoxData[i_List], ___] :> DeleteCases[i, s_String /; StringTrim[s] === ""],
                  ___ :> Throw[$Failed, $tag]
              }
          ];
          inputCellArgs = Replace[inputCell, Cell[_, args___] :> args];
          newInputCells = MapIndexed[
              Function[
                  With[
                      {n = firstNum + First[#2] - 1},
                      Association[
                          "Number" -> n,
                          "Cell" -> Cell[
                              BoxData[#1],
                              inputCellArgs,
                              CellLabel -> StringJoin["In[", ToString[n], "]:="]
                          ]
                      ]
                  ]
              ],
              inputBoxes
          ];
          outCells = Replace[group, Cell[CellGroupData[{_, c___}, ___], ___] :> {c}];
          newCells = Fold[
              Function[
                  Insert[
                      #1,
                      #2["Cell"],
                      FirstPosition[#1, c_Cell /; inputNumber[c] >= #2["Number"], {-1}]
                  ]
              ],
              outCells,
              newInputCells
          ];
          SelectionMove[cell, All, CellGroup];
          NotebookWrite[nb, newCells]
      ],
      $tag
  ];


inputNumber[cell_CellObject] := inputNumber[CurrentValue[cell, CellLabel]];

inputNumber[Cell[___, CellLabel -> label_String, ___]] := inputNumber[label];

inputNumber[label_String] :=
  With[
      {
          num = Replace[
              StringCases[
                  label,
                  StringExpression[
                      StartOfString,
                      "During evaluation of In[" | "In[" | "Out[",
                      d:DigitCharacter..,
                      "]:=" | "]=",
                      EndOfString
                  ] :> d
              ],
              {d_String} :> ToExpression[d]
          ]
      },
      num /; IntegerQ[num]
  ];

inputNumber[___] := $Failed;



(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*rewriteWithSubscripts*)


rewriteWithSubscripts[
    info: KeyValuePattern @ {
        "ParentCellObject" -> cell_CellObject,
        "NotebookObject" -> nb_,
        "MessageParameters" -> params_List
    }
] :=
  rewriteWithSubscripts[ cell, nb, # ] & /@ Cases[ Flatten @ params, _String ];


rewriteWithSubscripts[ cell_, nb_, string_ ] :=
  Module[ { find, ssBoxes },
      SelectionMove[ cell, Before, CellContents ];
      find = NotebookFind[ cell, string ];
      If[ MatchQ[ find, _NotebookSelection ],
          ssBoxes = StringCases[
              string,
              StringExpression[
                  StartOfString,
                  a_? LowerCaseQ,
                  b: LetterCharacter ...,
                  c: DigitCharacter ..,
                  EndOfString
              ] :> SubscriptBox[ StringJoin[ a, b ], c ]
          ];
          Replace[ ssBoxes,
                   { box_SubscriptBox } :> NotebookWrite[ nb, box ]
          ]
      ]
  ];



(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*deleteOtherCells*)

deleteOtherCells[ KeyValuePattern @ {
    "ParentCellObject" -> cell_CellObject,
    "NotebookObject" -> nb_,
    "MessageParameters" -> { ids__Integer }
} ] :=
  NotebookDelete @ Cells[ nb, CellID -> Alternatives @ ids ];


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*convertToPlainText*)

convertToPlainText[ KeyValuePattern @ {
    "ParentCellObject" -> cell_CellObject,
    "NotebookObject" -> nb_
} ] :=
  Module[ { c, s, r },
      c = NotebookRead @ cell;
      s = First @ FrontEndExecute @ ExportPacket[ c, "PlainText" ];
      r = Replace[ c, Cell[ _, args___ ] :> Cell[ s, args ] ];
      NotebookWrite[ cell, r ] /; MatchQ[ r, Cell[ _String, ___ ] ]
  ];


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*autoFormatCode*)


autoFormatCode[ KeyValuePattern @ {
    "ParentCellObject" -> cell_CellObject,
    "NotebookObject" -> nb_,
    "MessageParameters" -> { string_, ___ }
} ] :=
  Module[ { },
      SelectionMove[ cell, Before, CellContents ];
      NotebookFind[ cell, string ];
      FunctionResource`DocuToolsTemplate`FunctionTemplateToggle @ nb
  ];


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*replaceEllipsis*)

replaceEllipsis[ KeyValuePattern[ "ParentCellObject" -> cell_ ] ] :=
  Module[ { content, new },
      content = First @ NotebookRead @ cell;
      new = content /. s_String :> StringReplace[ s, "..." -> "\[Ellipsis]" ];
      SelectionMove[ cell, All, CellContents ];
      NotebookWrite[ Notebooks @ cell, new ]
  ];


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*capitalizeName*)

capitalizeName[ KeyValuePattern[ "ParentCellObject" -> cell_CellObject ] ] :=
  capitalizeName @ cell;

capitalizeName[ cell_CellObject ] :=
  Module[ { content, new },
      content = NotebookRead @ cell;
      new = MapAt[ Capitalize, content, 1 ];
      NotebookWrite[ cell, new ] /; MatchQ[ new, Cell[ _String, ___ ] ]
  ];


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*splitItemCell*)

splitItemCell[ KeyValuePattern[ "ParentCellObject" -> cell_ ] ] :=
  splitItemCell @ cell;

splitItemCell[ cell_CellObject ] :=
  Module[ { content, new },
      content = First @ NotebookRead @ cell;
      new = Cell[ StringTrim @ #, "Item" ] & /@ StringSplit[ content, "," ];
      NotebookWrite[ cell, new ]
  ];


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*addCharacterToEndOfText*)


addColonToText[ info_ ] :=
  addCharacterToEndOfText[ info, ":" ];

addPeriodToText[ info_ ] :=
  addCharacterToEndOfText[ info, "." ];



addCharacterToEndOfText[ KeyValuePattern[ "ParentCellObject" -> cell_ ], char_ ] :=
  addCharacterToEndOfText[ cell, char ];

addCharacterToEndOfText[ cell_CellObject, char_ ] :=
  Module[ { old, new },
      old = NotebookRead @ cell;
      new = addCharacterToEndOfText[ old, char ];
      NotebookWrite[ cell, new ]
  ];

addCharacterToEndOfText[ Cell[ box_, args___ ], char_ ] :=
  Cell[ addCharacterToEndOfText[ box, char ], args ];

addCharacterToEndOfText[ (box: TextData|RowBox)[ { a___, text_String } ], char_ ] :=
  box @ { a, addCharacterToEndOfText[ text, char ] };

addCharacterToEndOfText[ (box: TextData|RowBox)[ { a___, b_ } ], char_ ] :=
  box @ { a, b, char };

addCharacterToEndOfText[ text_String, char_ ] :=
  StringDelete[ text, Except[ "\[Ellipsis]", PunctuationCharacter ] ~~ EndOfString ] <> char;

addCharacterToEndOfText[ BoxData[ boxes_ ], char_ ] :=
  addCharacterToEndOfText[ boxes, char ];


(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*autoFormatArguments*)


autoFormatArguments[ KeyValuePattern[ "ParentCellObject" -> cell_ ] ] :=
  Module[ { desc, usage, args, new },
      desc = NotebookRead[cell];
      usage = NotebookRead[PreviousCell[cell]];
      args = Alternatives @@ findArguments[usage];
      new = ReplaceAll[
          Replace[desc, Cell[a_String, b___] :> Cell[TextData[{a}], b]],
          {
              StyleBox[a:args, FontSlant -> "Italic"] :> StyleBox[a, "TI"],
              TextData[{a___, b:args, c___}] :> TextData[{a, StyleBox[b, "TI"], c}],
              TextData[
                  {a___, b_String /; StringContainsQ[b, WordBoundary~~args~~WordBoundary], c___}
              ] :> TextData[
                  Flatten[
                      {
                          a,
                          StringSplit[
                              b,
                              d:WordBoundary~~e:args~~f:WordBoundary :> {
                                  d,
                                  Cell[
                                      BoxData[StyleBox[e, "TI"]],
                                      "InlineFormula",
                                      FontFamily -> "Source Sans Pro"
                                  ],
                                  f
                              }
                          ],
                          c
                      }
                  ]
              ]
          }
      ];
      NotebookWrite[cell, new]
  ];



(******************************************************************************)
(* ::Subsubsection::Closed:: *)
(*autoFormatUsage*)


autoFormatUsage[ KeyValuePattern[ "ParentCellObject" -> cell_ ] ] :=
  Module[ { },
      SelectionMove[ cell, All, CellContents ];
      FrontEndTokenExecute[ "ClearCellOptions" ];
      FunctionResource`DocuToolsTemplate`FunctionTemplateToggle[ ]
  ];




(* TODO: get rid of this *)
putDebugNotebook[expr_Association] :=
  NotebookPut @ Append[
      ResourceFunction["AssociationNotebook"][expr],
      "ClosingSaveDialog" -> False
  ];

putDebugNotebook[expr_] :=
  NotebookPut[
      Notebook[
          {
              Cell[
                  BoxData[
                      ToBoxes[
                          ResourceFunction["ReadableForm"][
                              Unevaluated[expr],
                              "FormatHeads" -> {CellObject, NotebookObject}
                          ]
                      ]
                  ],
                  "Input"
              ]
          },
          "ClosingSaveDialog" -> False
      ]
  ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*prettyTooltip*)


prettyTooltip[ label_, tooltip_ ] :=
  RawBoxes @
    TemplateBox[ { MakeBoxes @ label, MakeBoxes @ tooltip },
                 "PrettyTooltipTemplate"
    ];




(******************************************************************************)
(* ::Subsection::Closed:: *)
(*getString*)

getString[ "Hints", "Suggestions", keys__ ] :=
  Replace[ $hiddenStrings[ "Hints", "Suggestions", keys ],
      Except[ _String ] :> getString[ "Hints", "Warnings", keys ]
  ];

getString[ "Hints", "Warnings", keys__ ] :=
  Replace[ $hiddenStrings[ "Hints", "Warnings", keys ],
      Except[ _String ] :> getString[ "Hints", "Errors", keys ]
  ];

getString[ keys___ ] :=
  Replace[ $hiddenStrings[ keys ], Except[ _String ] :> "" ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*editFromPreview*)


editFromPreview[ preview_NotebookObject ] :=

  Module[ { edit, file },

      edit = CurrentValue[ preview, { TaggingRules, "Source", "NotebookObject" } ];

      If[ notebookObjectQ @ edit,
          updatePreviewTags[ preview, edit ];
          Throw[ SetSelectedNotebook @ edit, editFromPreview ]
      ];

      file = CurrentValue[ preview, { TaggingRules, "Source", "File" } ];

      If[ StringQ @ file && ! FileExistsQ @ file,
          CurrentValue[ preview, { TaggingRules, "Source", "File" } ] = None;
          MessageDialog @ StringTemplate[ getString[ "MessageDialogs", "MissingSourceNotebookFile" ] ][ file ];
          Throw[ $Failed, editFromPreview ]
      ];

      If[ ! StringQ @ file,
          CurrentValue[ preview, { TaggingRules, "Source", "File" } ] = None;
          MessageDialog @ getString[ "MessageDialogs", "MissingSourceNotebookObject" ];
          Throw[ $Failed, editFromPreview ]
      ];

      edit = NotebookOpen @ file;

      If[ notebookObjectQ @ edit,
          updatePreviewTags[ preview, edit ];
          Throw[ SetSelectedNotebook @ edit, editFromPreview ]
      ];

      MessageDialog @ getString[ "MessageDialogs", "MissingSourceNotebookObject" ];
      $Failed
  ] ~Catch~ editFromPreview;



updatePreviewTags[ preview_, edit_ ] := (
    CurrentValue[ edit, { TaggingRules, "PreviewNotebook" } ] = preview;
    CurrentValue[ preview, { TaggingRules, "Source", "File" } ] = toFileName[Lookup[NotebookInformation[edit], "FileName", None]];
    CurrentValue[ preview, { TaggingRules, "Source", "NotebookObject" } ] = edit;
);


notebookObjectQ[ nb_NotebookObject ] :=
  FailureQ @ NotebookInformation @ nb === False;

notebookObjectQ[ ___ ] :=
  False;


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*getResource*)


getResource[nb_NotebookObject, "Local"] :=
  nb ~checkForUpdates~ Module[
      {rf, lo},
      focusNotebook[nb];
      runWithProgress[
          rf = Check[scrapeResourceFunction[nb], $Failed];
          If[
              MatchQ[rf, _ResourceFunction],
              lo = ResourceSystemClient`Private`defaultLocalDeployLocation["Function",FunctionResource`ResourceFunctionInformation[rf]];
              updateExampleNotebook[ rf, lo ];
              With[
                  {obj = LocalCache[rf, lo], r = rf},
                  ResourceRegister[rf, "Local", "StoreContent" -> False];
                  FunctionResource`DocumentationNotebook`SaveDocumentationNotebook[r, "Local"];
                  FunctionResource`DocumentationNotebook`ViewDocumentationNotebook[r];
                  <|
                      "ResultMessage" -> "Deployed for this computer",
                      "ResultLink" -> prettyTooltip[ hyperlink["\[RightGuillemet]", ButtonFunction :> (FunctionResource`DocumentationNotebook`ViewDocumentationNotebook[r])], "View Documentation" ],
                      "ResourceFunction" :> ResourceFunction[obj],
                      "HiddenContent" -> {
                          {"Local object:", obj},
                          {"Resource object:", clickToCopy[nb, Unevaluated @ ResourceObject[r], Unevaluated @ ResourceObject[ResourceFunction[obj]]]}
                      }
                  |>
              ],
              showError @ nb
          ],
          nb
      ]
  ];

getResource[ nb_NotebookObject, "Cloud" ] :=
  nb ~checkForUpdates~ Module[
      {rf, co},
      focusNotebook[nb];
      runWithProgress[
          rf = Check[scrapeResourceFunction[nb], $Failed];
          If[
              MatchQ[rf, _ResourceFunction],
              ResourceSystemClient`Private`loadDeployResourceShingle[];
              co = DeployedResourceShingle`Private`defaultCloudObjectLocation["Function",ResourceObject[rf]["Name"]];
              updateExampleNotebook[ rf, co ];
              With[
                  {obj = CloudDeploy[rf, co, Permissions -> "Private"], r = rf},
                  SystemOpen @ First @ obj;
                  ResourceRegister[rf, "Cloud", "StoreContent" -> False];
                  <|
                      "ResultMessage" -> "Deployed for this cloud account",
                      "ResultLink" -> Hyperlink["\[RightGuillemet]", First[obj]],
                      "ResourceFunction" :> ResourceFunction[obj],
                      "HiddenContent" -> {
                          {"Cloud object:", obj},
                          {"Visit web page:", Hyperlink[First[obj], First[obj]]},
                          {"Resource object:", clickToCopy[nb, Unevaluated @ ResourceObject[r], Unevaluated @ ResourceObject[ResourceFunction[obj]]]}
                      }
                  |>
              ],
              showError @ nb
          ],
          nb
      ]
  ];


getResource[ nb_NotebookObject, "CloudPublic" ] :=
  nb ~checkForUpdates~ Module[
      {rf, co},
      focusNotebook[nb];
      runWithProgress[
          rf = Check[scrapeResourceFunction[nb], $Failed];
          If[
              MatchQ[rf, _ResourceFunction],
              ResourceSystemClient`Private`loadDeployResourceShingle[];
              co = DeployedResourceShingle`Private`defaultCloudObjectLocation["Function",ResourceObject[rf]["Name"]];
              updateExampleNotebook[ rf, co ];
              With[
                  {obj = CloudDeploy[rf, co, Permissions -> "Public"], r = rf},
                  SystemOpen @ First @ obj;
                  ResourceRegister[rf, "Cloud", "StoreContent" -> False];
                  <|
                      "ResultMessage" -> "Published to the cloud",
                      "ResultLink" -> Hyperlink["\[RightGuillemet]", First[obj]],
                      "ResourceFunction" :> ResourceFunction[obj],
                      "HiddenContent" -> {
                          {"Cloud object:", obj},
                          {"Visit web page:", Hyperlink[First[obj], First[obj]]},
                          {"Resource object:", clickToCopy[nb, Unevaluated @ ResourceObject[r], ResourceObject[r]]}
                      }
                  |>
              ],
              showError @ nb
          ],
          nb
      ]
  ];


getResource[ nb_NotebookObject, "CloudPut" ] :=
  nb ~checkForUpdates~ Module[
      {rf, co},
      focusNotebook[nb];
      runWithProgress[
          rf = Check[scrapeResourceFunction[nb], $Failed];
          If[
              MatchQ[rf, _ResourceFunction],
              ResourceSystemClient`Private`loadDeployResourceShingle[];
              co = DeployedResourceShingle`Private`defaultCloudObjectLocation["Function",ResourceObject[rf]["Name"]];
              (*updateExampleNotebook[ rf, co ];*)
              With[
                  {obj = CloudPut[ResourceObject[KeyDrop[ResourceObject[FunctionResource`Private`inlineDefinitions @ rf][All], {"ExampleNotebook", "Documentation"}]], co, Permissions -> "Private"], r = rf},
                  ResourceRegister[rf, "Cloud", "StoreContent" -> False];
                  <|
                      "ResultMessage" -> "Deployed for this cloud account",
                      "ResourceFunction" :> ResourceFunction[obj],
                      "HiddenContent" -> {
                          {"Cloud object:", obj},
                          {"Resource object:", Defer @ ResourceObject @ obj}
                      }
                  |>
              ],
              showError @ nb
          ],
          nb
      ]
  ];

getResource[ nb_NotebookObject, "CloudPutPublic" ] :=
  nb ~checkForUpdates~ Module[
      {rf, co},
      focusNotebook[nb];
      runWithProgress[
          rf = Check[scrapeResourceFunction[nb], $Failed];
          If[
              MatchQ[rf, _ResourceFunction],
              ResourceSystemClient`Private`loadDeployResourceShingle[];
              co = DeployedResourceShingle`Private`defaultCloudObjectLocation["Function",ResourceObject[rf]["Name"]];
              (*updateExampleNotebook[ rf, co ];*)
              With[
                  {obj = CloudPut[ResourceObject[KeyDrop[ResourceObject[FunctionResource`Private`inlineDefinitions @ rf][All], {"ExampleNotebook", "Documentation"}]], co, Permissions -> "Public"], r = rf},
                  ResourceRegister[rf, "Cloud", "StoreContent" -> False];
                  <|
                      "ResultMessage" -> "Published to the cloud",
                      "ResourceFunction" :> ResourceFunction[obj],
                      "HiddenContent" -> {
                          {"Cloud object:", obj},
                          {"Resource object:", Defer @ ResourceObject @ obj}
                      }
                  |>
              ],
              showError @ nb
          ],
          nb
      ]
  ];

getResource[nb_NotebookObject, "KernelSession"] :=
  nb ~checkForUpdates~ Module[
      {rf},
      runWithProgress[
          rf = Check[scrapeResourceFunction[nb], $Failed];
          If[
              MatchQ[rf, _ResourceFunction],
              With[
                  {r = rf},
                  <|
                      "ResultMessage" -> "Deployed for this session",
                      "ResourceFunction" -> r,
                      "HiddenContent" -> {
                          {"Resource object:", Defer[ResourceObject[r]]}
                      }
                  |>
              ],
              showError @ nb
          ],
          nb
      ]
  ];


$toolsCell =
  Cell[ BoxData @ TemplateBox[ { }, "ToolsGridTemplate" ],
      "DockedCell",
      TaggingRules -> { "Tools" -> True },
      CellFrameMargins -> 0,
      CellFrame -> 0,
      CellMargins -> { { -10, -10 }, { 0, 0 } },
      Background -> RGBColor[ "#eb571b" ]
  ];

getResource[ nb_NotebookObject, "Tools" ] :=
  nb ~checkForUpdates~ Module[{dockedcells = CurrentValue[nb, DockedCells]},
  	CurrentValue[nb, DockedCells] =
     If[ Cases[dockedcells, Cell[__, TaggingRules -> {"Tools" -> True}, ___]] === {},
         Insert[dockedcells, $toolsCell, 2],
         DeleteCases[dockedcells, Cell[__, TaggingRules -> {"Tools" -> True}, ___]]
     ]
  ];



getResource[nb_NotebookObject, "Preview"] :=
  nb ~checkForUpdates~ Block[ { $previewing = True }, Module[
      {rf},
      runWithProgress[
          rf = Check[scrapeResourceFunction[nb], $Failed];
          If[
              MatchQ[rf, _ResourceFunction],
              viewPreviewNotebook[nb, rf];
              With[
                  {r = rf},
                  <|
                      "ResultMessage" -> "Deployed for this session",
                      "ResultLink" -> prettyTooltip[ hyperlink["\[RightGuillemet]", ButtonFunction :> (viewPreviewNotebook[nb, r])], "View Documentation" ],
                      "ResourceFunction" -> r,
                      "HiddenContent" -> {
                          {"Resource object:", Defer[ResourceObject[r]]}
                      }
                  |>
              ],
              showError @ nb
          ],
          nb
      ]
  ] ];


getResource[nb_NotebookObject, "Debug"] :=
  nb ~checkForUpdates~ Module[
      { },
      SetSelectedNotebook[nb];
      setStatusMessage[nb, ProgressIndicator[Appearance -> "Necklace"]];
      appendStripe[
          nb,
          stripeGrid[
              {
                  {
                      Style[Row[{"This is a test message "}], "Text"],
                      Row[{Style["Resource function: ", "Text"], clickToCopy[nb, "NothingHere"]}]
                  }
              }
          ]
      ];
      resetStatusMessage[nb]
  ];




refreshDockedCells[ nb_ ] /; $OperatingSystem === "MacOSX" && $VersionNumber >= 12 :=
  CurrentValue[ nb, DockedCells ] = Identity @ CurrentValue[ nb, DockedCells ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*checkNotebook*)



checkNotebook[nb_NotebookObject] :=
  nb ~checkForUpdates~ Module[
      {rf},
      runWithProgress[
          rf = Check[scrapeResourceFunction[nb, False], $Failed];
          If[
              MatchQ[rf, _ResourceFunction],
              With[ { r = rf },
                  NotebookClose @ ResourceFunction[ r, "ExampleNotebookObject" ];
                  Clear @ r;
                  <|
                      "ResultMessage" -> "Check complete",
                      "VisibleContent" -> "",
                      "HiddenContent" -> None
                  |>
              ],
              showError @ nb
          ],
          nb
      ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*focusNotebook*)


focusNotebook[ nb_NotebookObject ] := (
    SetSelectedNotebook @ nb;
    (*SelectionMove[ nb, Before, Notebook, AutoScroll -> True ]*)
);



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*appendStripe*)


appendStripe[nb_, content_Association] :=
  Module[
      {str, id},
      {str, {{id}}} = Reap[stripe[Append[content, "NotebookObject" -> nb]], $stripeID];
      CurrentValue[nb, DockedCells] = Append[
          Flatten[{CurrentValue[nb, DockedCells]}] /. DynamicModuleBox[___, BoxID -> "WarningCountButton", ___] -> "",
          stripeCell @ str
      ];
      id
  ];


appendStripe[nb_, content_] :=
  Module[
      {str, id},
      {str, {{id}}} = Reap[stripe[nb, content], $stripeID];
      CurrentValue[nb, DockedCells] = Append[
          Flatten[{CurrentValue[nb, DockedCells]}] /. DynamicModuleBox[___, BoxID -> "WarningCountButton", ___] -> "",
          stripeCell @ str
      ];
      id
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*stripeCell*)


stripeCell[ str_ ] :=
  Internal`InheritedBlock[ { CloudObject },
      GeneralUtilities`BlockProtected[ { CloudObject },
          FormatValues[ CloudObject ] = { };
      ];
      Cell[ BoxData @ ToBoxes @ Style[ str, "Text", FontSize -> 12, FontColor -> GrayLevel[ 0.25 ] ],
            "StripeCell",
            CellFrameMargins -> { { 25, 5 }, { 2, 2 } },
            CellFrame -> { { 0, 0 }, { 1, 0 } },
            CellFrameColor -> GrayLevel[ 0.75 ]
      ]
  ];




(******************************************************************************)
(* ::Subsection::Closed:: *)
(*stripe*)


stripe[nb_, info_Association] :=
  stripe[Append[info, "NotebookObject" -> nb]];

stripe[nb_, expr_] :=
  Grid[
      {{expr, Style[TextString[Now], FontSlant -> Italic, FontColor -> GrayLevel[0.5], FontSize -> Inherited * 0.9], xButton[nb]}},
      Alignment -> {{Left, Right, Right}, Center},
      ItemSize -> {{Fit, Automatic, Automatic}, Automatic}
  ];


stripe[ info: KeyValuePattern @ { "NotebookObject" -> nb_, "HiddenContent" -> content: Except[_$formatted|None] } ] :=
  stripe @ Append[ info, "HiddenContent" -> stripeHiddenContent[ nb, content ] ];


stripe[ info: KeyValuePattern @ {
    "HiddenContent" -> $formatted[hiddenContent_]
} ] :=
  Module[ { visible },
      visible = stripe @ Append[ info, "HiddenContent" -> None ];
      OpenerView[ {
          visible,
          Grid[
              {{"", hiddenContent}},
              ItemSize -> { { Automatic, Fit }, Automatic },
              Alignment -> Left,
              Dividers -> {None, {-1 -> GrayLevel[239/255]}},
              Spacings -> {{1, {}, 1}, {1, {}, 2}}
          ]
      } ]
  ];

stripe[ info_Association ] /; ! KeyExistsQ[ info, "ResultLink" ] :=
  stripe @ Append[ info, "ResultLink" -> "" ];

stripe[ info: KeyValuePattern @ {
    "NotebookObject" -> nb_,
    "ResultMessage" -> resultMessage_,
    "ResultLink" -> resultLink_,
    "VisibleContent" -> visibleContent_,
    "HiddenContent" -> None
} ] :=
  Module[ { time, closeButton },
      time = Style[TextString[Now], FontSlant -> Italic, FontColor -> GrayLevel[0.5], FontSize -> Inherited * 0.9];
      closeButton = xButton[nb];
      Grid[{{
          stripeMessage[resultMessage, resultLink],
          visibleContent,
          warningCountButton[nb],
          time,
          closeButton
      }},
          ItemSize -> {{Scaled[.35], Scaled[.35], Fit, Automatic Automatic}, Automatic},
          Alignment -> {{Left, Left, Center, Right, Right}, Center}
      ]
  ];

stripe[ info: KeyValuePattern @ { "NotebookObject" -> nb_, "ResourceFunction" -> rf_ } ] :=
  stripe @ KeyDrop[ Append[ info, "VisibleContent" -> stripeResourceFunction[nb, rf]],
                    "ResourceFunction"
           ];

stripe[ info: KeyValuePattern @ { "NotebookObject" -> nb_, "ResourceFunction" :> rf_ } ] :=
  stripe @ KeyDrop[ Append[ info, "VisibleContent" -> stripeResourceFunction[nb, rf, Unevaluated[rf]]],
                    "ResourceFunction"
           ];


stripeMessage[message_] :=
  stripeMessage[message, ""];

stripeMessage[message_, link_] :=
  Grid[{{message, link}}, Spacings -> 0.25];

stripeResourceFunction[nb_, show_] :=
  stripeResourceFunction[nb, Unevaluated@show, Unevaluated@show];

stripeResourceFunction[nb_, show_, copy_] :=
  Grid[{{"ResourceFunction:", clickToCopy[nb, show, Unevaluated[copy]]}}];

stripeHiddenContent[nb_, content: {__List}] :=
  $formatted @
      Style[
          Grid[
              formatHiddenRow[nb] /@ content,
              Alignment -> Left,
              ItemSize -> {{Automatic, Automatic, Fit}, Automatic}
          ],
          FontColor -> GrayLevel[0.5],
          FontSize -> 12
      ];

formatHiddenRow[ nb_ ][ { a_, b_RawBoxes } ] :=
  { "\[FilledVerySmallSquare]", a, b };

formatHiddenRow[ nb_ ][ { a_, (b:Button|Hyperlink)[ expr_, args___ ] } ] :=
  { "\[FilledVerySmallSquare]", a, b[ short @ expr, args ] };

formatHiddenRow[ nb_ ][ { a_, Defer[ expr_ ] } ] :=
  { "\[FilledVerySmallSquare]", a, clickToCopy[ nb, short @ expr, Unevaluated @ expr ] };

formatHiddenRow[ nb_ ][ { a_, expr_ } ] :=
  { "\[FilledVerySmallSquare]", a, clickToCopy[ nb, short @ expr, expr ] };



short // Attributes = { HoldFirst };

short[ LocalObject[ string_String ] ] :=
  With[ { shortString = StringReplace[ string, $LocalBase -> "file://\[Ellipsis]" ] },
      RawBoxes @ MakeBoxes @ LocalObject @ shortString
  ];

short[ CloudObject[ string_String ] ] :=
  With[ { shortString = shortURL @ string },
      RawBoxes @ RowBox @ { "CloudObject", "[", ToBoxes @ Hyperlink[ shortString, string ], "]" }
  ];

short[ other_ ] :=
  RawBoxes @ MakeBoxes @ Short[ other, .75 ];



shortURL[ url_String ] :=
  StringReplace[ url,
                 URLBuild @ { $CloudBase, "objects" } ->
                   URLParse[ $CloudBase, "Scheme" ] <> "://\[Ellipsis]"
  ];



warningCountButton[ nb_ ] :=
  With[{warningCells = Cases[
      CurrentValue[nb, {TaggingRules, "AttachedHints"}],
      cell_CellObject /;
        MatchQ[CurrentValue[cell, CellStyle], {___,
            "WarningText", ___}] :> ParentCell[cell]
  ]},
      If[ TrueQ[Length[warningCells] >= 1],
          DynamicModule[ { idx = 0, len = Length @ warningCells },
              Button[
                  MouseAppearance[
                      prettyTooltip[
                          Grid[{{Show[$warningIcon, ImageSize -> 11],
                              Style[len, FontColor -> GrayLevel[0.5], FontSize -> 9]}},
                              Alignment -> {Left, Center}, Spacings -> 0.25],
                          getString["Tooltips", "WarningCountButton"]
                      ],
                      "LinkHand"
                  ],
                  moveToCell[ warningCells[[ idx = Mod[ idx + 1, len, 1 ] ]], All, CellContents ],
                  Appearance -> None
              ],
              BoxID -> "WarningCountButton"
          ],
          ""
      ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*xButton*)

xButton[_] :=
  With[
      {id = Sow[CreateUUID[], $stripeID], label = xButtonLabel},
      MouseAppearance[
          Button[
              label,
              Symbol["System`ResourceFunction"]; deleteMe[id],
              Appearance -> None,
              BoxID -> id
          ],
          "LinkHand"
      ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*deleteMe*)


deleteMe[id_] :=
  CurrentValue[EvaluationNotebook[], DockedCells] =
    DeleteCases[
        Flatten[{CurrentValue[EvaluationNotebook[], DockedCells]}],
        c_Cell /;  !FreeQ[c, id]
    ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*xButtonLabel*)

range = 0.7;
size = 18;

x :=
  x = Sequence[
      Polygon[
          {
              {-0.282842712474619, -0.42426406871192845},
              {-0.42426406871192845, -0.282842712474619},
              {0.282842712474619, 0.42426406871192845},
              {0.42426406871192845, 0.282842712474619}
          }
      ],
      Polygon[
          {
              {0.42426406871192845, -0.282842712474619},
              {0.282842712474619, -0.42426406871192845},
              {-0.42426406871192845, 0.282842712474619},
              {-0.282842712474619, 0.42426406871192845}
          }
      ]
  ];

xButtonDefault :=
  xButtonDefault = Graphics[
      {GrayLevel[192/255], x},
      ImageSize -> size,
      PlotRangePadding -> 0,
      PlotRange -> range,
      Background -> None
  ];

xButtonHover :=
  xButtonHover = Graphics[
      {GrayLevel[128/255], x},
      ImageSize -> size,
      PlotRangePadding -> 0,
      PlotRange -> range,
      Background -> None
  ];

xButtonLabel := xButtonLabel =
  MouseAppearance[Mouseover[xButtonDefault, xButtonHover], "LinkHand"];




(******************************************************************************)
(* ::Subsection::Closed:: *)
(*$now*)


$now :=
  Module[
      {label},
      label = Style[TextString[Now], "Text", FontSlant -> Italic, FontColor -> GrayLevel[0.5], FontSize -> Inherited*0.8];
      RawBoxes[AdjustmentBox[ToBoxes[label], BoxMargins -> {{0., 0.}, {0., 1.}}]]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*clickToCopy*)


clickToCopy[nb_, expr_] := clickToCopy[nb, Unevaluated[expr], Unevaluated[expr]];

(*clickToCopy[nb_, expr_, copy_] :=
  With[ { boxes = RawBoxes @ MakeBoxes[ copy, StandardForm ] },
      Button[
          MouseAppearance[Tooltip[Defer[expr], "Click to copy to the clipboard"], "LinkHand"],
          CopyToClipboard[boxes];
          setStatusMessage[nb, "Copied to clipboard", 5],
          Appearance -> None
      ]
  ];*)

clickToCopy[nb_, expr_, copy_] :=
  With[ { boxes = RawBoxes @ MakeBoxes[ copy, StandardForm ] },
      RawBoxes @
        TemplateBox[
            {
                ToBoxes @ Defer @ expr,
                boxes
            },
            "ClickToCopyTemplate"
        ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*setStatusMessage*)


setStatusMessage[nb_NotebookObject, message_, time_] :=
  Quiet[
      resetStatusMessage[nb];
      CurrentValue[nb, {TaggingRules, "StatusMessage"}] = message;
      CurrentValue[nb, {TaggingRules, "StatusMessageTask"}] =
        RunScheduledTask[
            resetStatusMessage[nb];
            RemoveScheduledTask[$ScheduledTask];
            CurrentValue[nb, {TaggingRules, "StatusMessageTask"}] = Inherited,
            {time}
        ]
  ];

setStatusMessage[nb_NotebookObject, message_] :=
  Quiet[
      resetStatusMessage[nb];
      CurrentValue[nb, {TaggingRules, "StatusMessage"}] = message;
      CurrentValue[nb, {TaggingRules, "StatusMessageTask"}] = Inherited;
      Null
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*resetStatusMessage*)


resetStatusMessage[nb_NotebookObject] :=
  Quiet[
      CurrentValue[nb, {TaggingRules, "StatusMessage"}] = "";
      RemoveScheduledTask[CurrentValue[nb, {TaggingRules, "StatusMessageTask"}]];
      Null
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*runWithProgress*)


runWithProgress // Attributes = {HoldFirst};

runWithProgress[eval_, nb_] :=
  runWithProgress[eval, nb, stripeGrid[{{ProgressIndicator[Appearance -> "Percolate"]}}], Identity];

runWithProgress[eval_, nb_, temp_] := runWithProgress[eval, nb, temp, Identity];

runWithProgress[eval_, nb_, temp_, displayResult_] :=
  catchErrors @ Module[
      {tempID, result},
      SetSelectedNotebook @ nb;
      tempID = appendStripe[nb, temp];
      result = showProgress[eval];
      replaceStripe[nb, tempID -> displayResult[result]]
  ];



$caughtError = "";


$resourceSubmitMessageNames :=
  $resourceSubmitMessageNames =
    Internal`InheritedBlock[ { ResourceSubmit },
        ClearAttributes[ ResourceSubmit, ReadProtected ];
        Apply[ HoldComplete,
               Union @ Cases[ Messages @ ResourceSubmit,
                              msg: HoldPattern @ MessageName[ ResourceSubmit, _String ] :> HoldComplete @ msg,
                              Infinity,
                              Heads -> True
                       ]
        ] // Flatten
    ];


$otherIgnoredMessageNames =
  HoldComplete[
      CloudConnect::clver
  ];


$ignoredMessages :=
  $ignoredMessages =
    Join[ $resourceSubmitMessageNames, $otherIgnoredMessageNames ];


quietResourceSubmit // Attributes = { HoldFirst };
quietResourceSubmit[ eval_ ] :=
  Replace[ $ignoredMessages,
           {
               HoldComplete[ msgs___MessageName ] :> Quiet[ eval, { msgs } ],
               ___ :> eval
           }
  ];


messageHandler[ Hold[ Message[ m: MessageName[ ResourceSubmit, _ ], args___ ], _ ] ] :=
  Module[ { stringArgs },
      stringArgs = Cases[ HoldComplete @ args, arg_ :> ToString @ Unevaluated @ arg ];
      $caughtError =
        Grid[
            {{ RawBoxes @ MakeBoxes @ m, ": ", StringTemplate[ m ] @@ stringArgs }},
            Spacings -> 0,
            Alignment -> { Left, Center }
        ]
  ];



catchErrors // Attributes = { HoldFirst };
catchErrors[ eval_ ] :=
  (
      $caughtError = "";
      Internal`HandlerBlock[ { "Message", messageHandler },
          quietResourceSubmit @ eval
      ]
  );


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*replaceStripe*)


replaceStripe[nb_, old_ -> new_] :=
  Module[
      {str, id, cells},
      {str, {{id}}} = Reap[stripe[nb, new], $stripeID];
      refreshDockedCells[nb];
      cells = Flatten[{CurrentValue[nb, DockedCells]}];
      If[
          FreeQ[cells, old],
          appendStripe[nb, new],
          CurrentValue[nb, DockedCells] = Replace[
              cells,
              c_Cell /;  !FreeQ[c, old] :> stripeCell @ str,
              {1}
          ]
      ];
      refreshDockedCells[nb];
      id
  ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*showProgress*)


showProgress // Attributes = {HoldFirst};

showProgress[eval_, nb_] :=
  Module[
      {result},
      setStatusMessage[nb, ProgressIndicator[Appearance -> "Necklace"]];
      Block[{showProgress = #1 & }, result = scrollIfMessage[ nb, eval ]];
      If[
          CurrentValue[nb, {TaggingRules, "StatusMessage"}] === ProgressIndicator[Appearance -> "Necklace"],
          resetStatusMessage[nb]
      ];
      result
  ];

showProgress[eval_] := showProgress[eval, EvaluationNotebook[]];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*stripeGrid*)


stripeGrid[rows_] :=
  Grid[rows, ItemSize -> {{Fit, Scaled[0.6]}, Automatic}, Alignment -> Left, Spacings -> {10, Automatic}];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*hyperlink*)


hyperlink[label_, url_String] := Hyperlink[label, url];
hyperlink[label_, ButtonFunction :> action_, opts___] :=
  With[{boxes = ToBoxes@label},
      RawBoxes@
        ButtonBox[
            TagBox[TemplateBox[{boxes, StyleBox[boxes, "HyperlinkActive"],
                BaseStyle -> "Hyperlink"}, "MouseoverTemplate"],
                MouseAppearanceTag["LinkHand"]], opts, BaseStyle -> "Hyperlink",
            ButtonFunction :> action, Evaluator -> Automatic,
            Method -> "Queued"]
  ];




(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeResourceFunction*)


scrapeResourceFunction[ nb_Notebook, register_: True ] :=
  Module[ { scraped },
      ClearAll @ $lastValidations;
      $lastValidations = Internal`Bag[ ];
      scraped = collectHints @ Catch[
          Module[ { info, reg, rf, uuid },
              info = Association["ResourceType" -> "Function"];
              info["Name"] = scrapeName[nb, info];
              info["ShortName"] = info["Name"];
              info["Description"] = scrapeDescription[nb, info];
              info["Documentation"] = DeleteMissing @ Association["Usage" -> scrapeUsage[nb, info], "Notes" -> scrapeNotes[nb, info]];
              info["ExampleNotebook"] = scrapeExampleNotebook[nb, info];
              info["ContributorInformation"] = scrapeAuthor[nb, info];
              info["Categories"] = scrapeCategories[nb, info];
              info["Keywords"] = scrapeKeywords[nb, info];
              info["SeeAlso"] = scrapeSeeAlso[nb, info];
              info["RelatedSymbols"] = scrapeRelatedSymbols[nb, info];
              info["SourceMetadata"] = scrapeSourceMetadata[nb, info];
              info["ExternalLinks"] = scrapeExternalLinks[nb, info];
              info["VerificationTests"] = scrapeVerificationTests[nb, info];
              info["Function"] = scrapeFunction[nb, info];
              info["SymbolName"] = FunctionResource`Private`fullSymbolName @@ info["Function"];
              info["DefinitionNotebook"] = scrapeDefinitionNotebook[nb, info];
              reg = If[ TrueQ @ register, ResourceRegister, #1 & ][
                  ResourceObject[DeleteMissing[info]],
                  "KernelSession"
              ];
              rf = ResourceFunction[reg];
              uuid = ResourceFunction[ rf, "UUID" ];
              DefinitionTemplateVersion[ uuid ] = $templateVersion;
              rf
          ],
          scrapeAll
      ];
      $lastScrapedHints = scraped[ "Data" ];
      scraped[ "Result" ]
  ];



$templateVersion :=
  If[ CurrentValue[ $SourceNotebook, { TaggingRules, "ResourceType" } ] === "Function"
      ,
      Replace[ CurrentValue[ $SourceNotebook, { TaggingRules, "TemplateVersion" } ],
               Except[ _String? StringQ ] :> Missing[ "Unknown" ]
      ]
      ,
      Missing[ "LegacyNotebook" ]
  ];



$defaultScrapedNotebookOptions :=
  Sequence[
      StyleDefinitions -> $stylesheet,
      TaggingRules -> {
          "ResourceType" -> "Function",
          "ResourceCreateNotebook" -> True
      },
      CreateCellID -> True
  ];


scrapeResourceFunction[ nbo_NotebookObject, register_: True ] :=
  Block[ { $SourceNotebook = nbo }, Module[ { cells, nb, scraped, attached },
      NotebookDelete /@ CurrentValue[ nbo, { TaggingRules, "AttachedHints" } ];
      fixMissingCellIDs @ nbo;
      cells = First @ NotebookGet @ nbo;
      nb = Notebook[ Flatten @ { cells }, $defaultScrapedNotebookOptions ];
      scraped = scrapeResourceFunction[ nb, register ];

      attached =
        Cases[ $lastScrapedHints,
               KeyValuePattern @ {
                   "CellID" -> id_,
                   "Level" -> level_,
                   "Tag" -> tag_,
                   "Parameters" -> {params___}
               } :> setHint[ level, nbo, id, tag, params ]
        ];

      CurrentValue[ nbo, { TaggingRules, "AttachedHints" } ] = Cases[ attached, _CellObject ];
      scraped
  ] ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*fixMissingCellIDs*)


findCellsWithMissingCellIDs[nbo_NotebookObject] :=
  Cases[Cells[nbo, CellID -> 0], _CellObject];

updateCellID[cell_CellObject] :=
  Module[{nbo, content, updated},
      nbo = Notebooks[cell];
      content = NotebookRead[cell];
      NotebookWrite[cell, content, All, AutoScroll -> False];
      updated = Replace[Flatten@SelectedCells[nbo], {c_CellObject} :> c];
      <|"Updated" -> updated, "ID" -> CurrentValue[updated, CellID]|>
  ];

fixMissingCellIDs[nbo_NotebookObject] :=
  Module[{cellsWithMissingIDs},
      cellsWithMissingIDs = findCellsWithMissingCellIDs[nbo];
      Table[updateCellID[cell], {cell, cellsWithMissingIDs}]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeDefinitionNotebook*)


scrapeDefinitionNotebook[ nb_, ___ ] /; TrueQ @ $submittingQ || TrueQ @ $previewing :=
  Replace[
      ResourceSystemClient`Private`includeDefinitionNotebook[ "Function", nb ],
      a: KeyValuePattern[ "Format" -> Sequence[ fmt_, ___ ] ] :> Append[ a, "Format" -> fmt ]
  ];

scrapeDefinitionNotebook[ ___ ] :=
  Missing[ "NotAvailable" ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeAuthor*)


scrapeAuthor[nb_NotebookObject, ___] :=
  FirstCase[
      DeleteCases[scrapeSection[nb, "Contributed By"], Cell["Author Name", "Text", ___]],
      Cell[author_String, "Text", ___] :> Association["ContributedBy" -> author]
  ];


scrapeAuthor[ nb_Notebook, info_ ] :=
  Module[ { cells, author, contributedBy },

      cells = scrapeSection[ nb, "Contributed By" ];
      author = FirstCase[ cells, Cell[ a_String, "Text", ___ ] :> a, Missing[ "NotFound" ], Infinity ];
      contributedBy = If[ StringQ @ author, <| "ContributedBy" -> author |>, Missing[ "NotFound" ] ];

      validate @ <|
          "Metadata" -> info,
          "Property" -> "ContributorInformation",
          "Cells" -> Flatten @ List @ cells,
          "Author" -> author,
          "Result" -> contributedBy
      |>
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeSection*)


scrapeSection[ nb_, section_String ] := (
    clearError[ nb, section ];
    Replace[
        getCellGroupByTag[ nb, section ],
        {
            Cell[ CellGroupData[ { header_ /; !FreeQ[ header, section ], cells___Cell }, ___ ] ] :> { cells },
            ___ :> { }
        }
    ]
);



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*clearError*)


clearError[nb_, section_] := clearAttachedInlineCell[nb, section];

clearError[KeyValuePattern[{"NotebookObject" -> nb_, "ID" -> id_}]] :=
  clearError[nb, id];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*clearAttachedInlineCell*)


clearAttachedInlineCell[ nb_, id_ ] :=
  Module[ { target },
      target = Replace[ getCellObject[ nb, id ], Except[ _CellObject ] :> Throw[ $Failed, clearAttachedInlineCell ] ];
      NotebookDelete @ CurrentValue[ target, { TaggingRules, "AttachedInlineCell" } ];
      CurrentValue[ target, { TaggingRules, "AttachedInlineCell" } ] = Inherited;
  ] ~Catch~ clearAttachedInlineCell;


clearAttachedInlineCell[ ___ ] :=
  $Failed;



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*getCellGroupByTag*)


getCellGroupByTag[nb_NotebookObject, tag_String] :=
  Replace[
      NotebookFind[nb, tag, All, CellTags, AutoScroll -> False],
      {
          _NotebookSelection :> (
              SelectionMove[nb, All, CellGroup, AutoScroll -> False];
              NotebookRead[nb]
          ),
          $Failed :> getCellGroupByName[nb, tag]
      }
  ];


getCellGroupByTag[ nb_Notebook, tag_String ] :=
  FirstCase[
      nb,
      Cell @ CellGroupData[ { Cell[ ___, CellTags -> { ___, tag, ___ }|tag, ___ ], ___ }, ___ ],
      getCellGroupByName[ nb, tag ],
      Infinity
  ];


getCellGroupByName[nb_NotebookObject, name_String] :=
  Module[
      {cell},
      cell = SelectFirst[
          Cells[nb, CellStyle -> {"Section", "Subsection"}],
          Function[
              MatchQ[
                  NotebookRead[#1],
                  Cell[name | TextData[{name, ___}], "Section" | "Subsection", ___]
              ]
          ]
      ];
      SelectionMove[cell, All, CellGroup, AutoScroll -> False];
      NotebookRead[nb]
  ];


getCellGroupByName[nb_Notebook, name_String] :=
  FirstCase[
      nb,
      Cell[
          CellGroupData[
              {
                  sec:Cell[name | TextData[{name, ___}], "Section" | "Subsection", ___],
                  cells:Longest[(Cell[_] | Cell[_, Except["Section" | "Subsection"], ___])...],
                  ___
              },
              _
          ]
      ] :> Cell[CellGroupData[{sec, cells}, Open]],
      Missing["NotFound"],
      Infinity
  ];



availableSectionNames[ nb_NotebookObject ] :=
  availableSectionNames @ NotebookGet @ nb;


availableSectionNames[ nb_Notebook ] :=
  GeneralUtilities`DeepUniqueCases[
      nb /.
        Cell[CellGroupData[{c:Cell["Examples"|TextData[{"Examples", ___}], "Section", ___], ___}, _]] :> c,
      Cell[
          (name_String) | TextData[{name_String, ___}],
          "Section" | "Subsection",
          ___
      ] :> name
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeCategories*)


scrapeCategories[nb_NotebookObject, ___] :=
  Module[
      {scraped, checkboxes, strings},
      scraped = scrapeSection[nb, "Categories"];
      checkboxes = Cases[scraped, CheckboxBox[checked_String, {False, checked_String}] :> checked, Infinity];
      strings = Cases[scraped, Cell[category_String, "Text" | "Item", ___] :> category, Infinity];
      Replace[Union[checkboxes, strings], Except[{__String}] :> Missing["NotFound"]]
  ];


scrapeCategories[ nb_Notebook, info_ ] :=
  Module[ { cells, checkboxes, strings, categories },

      cells = scrapeSection[ nb, "Categories" ];
      checkboxes = Cases[ cells, CheckboxBox[ checked_String, { False, checked_String } ] :> checked, Infinity ];
      strings = Cases[ cells, Cell[ category_String, "Text"|"Item", ___ ] :> category, Infinity ];
      categories = Replace[ Union[ checkboxes, strings ], Except @ { __String } :> Missing[ "NotFound" ] ];

      validate @ <|
          "Metadata" -> info,
          "Property" -> "Categories",
          "Cells" -> Flatten @ List @ cells,
          "Result" -> categories
      |>
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeDescription*)


scrapeDescription[nb_NotebookObject, ___] :=
  Module[
      {descString},
      clearError[nb, "Description"];
      SelectionMove[nb, Before, Notebook, AutoScroll -> False];
      NotebookFind[nb, "Title", Next, CellStyle, AutoScroll -> False];
      SelectionMove[nb, All, CellGroup, AutoScroll -> False];
      descString = FirstCase[
          NotebookRead[nb],
          Cell[description_String, "Text", ___] :> description,
          setWarning[nb, "Description", "DescriptionMissing"];
          Missing[],
          Infinity
      ];
      If[
          descString === "One-line description explaining the function\[CloseCurlyQuote]s basic purpose.",
          setWarning[nb, "Description", "DescriptionNotSet"]
      ];
      descString
  ];


scrapeDescription[ nb_Notebook, info_ ] :=
  Module[ { firstCell, descString },

      firstCell =
        FirstCase[
            nb,
            Cell @ CellGroupData[ { Cell[ _, "Title", ___ ], cell_Cell, ___ }, ___ ] :> cell,
            FirstCase[
                nb,
                { ___, Cell[ _, "Title", ___ ], cell_Cell, ___ } :> cell,
                Missing[ "NotFound" ],
                Infinity
            ],
            Infinity
        ];

      descString =
        Replace[
            firstCell,
            {
                Cell[ desc_String, "Text", ___ ] :> desc,
                Cell[ b_, Except[ "Text" ], ___ ] :>
                  FirstCase[
                      b,
                      Cell[ desc_String, "Text", ___ ] :> desc,
                      Missing[ "NotFound" ],
                      Infinity
                  ]
            }
        ];

      validate @ <|
          "Metadata" -> info,
          "Property" -> "Description",
          "Cells" -> Flatten @ List @ firstCell,
          "Result" -> descString
      |>
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*setWarning*)


$useFancyWarnings = False;


(*setWarning[nb_, section_, error_String, params___] :=
  With[ { string = StringTemplate[ getString[ "Hints", "Warnings", error ] ][ params ] },
      If[ MatchQ[ string, Except[ "", _String ] ],
          attachInlineCell[nb, section, warningCell[section, string, True]],
          attachInlineCell[nb, section, warningCell[section, error, True]]
      ]
  ];*)

setWarning[nb_, section_, error_String, params___] :=
  setSuggestion @ <|
      "NotebookObject" -> nb,
      "ID" -> section,
      "MessageTag" -> error,
      "MessageParameters" -> { params },
      "Icon" -> $warningIcon,
      "FontColor" -> RGBColor[0.921569, 0.341176, 0.105882, 1.],
      "FontWeight" -> "SemiBold",
      "FontFamily" -> "Source Sans Pro SemiBold"
  |>;


setWarning[nb_, section_, error_String, params___] /; TrueQ @ $useFancyWarnings :=
  With[ { string = StringTemplate[ getString[ "Hints", "Warnings", error ] ][ params ] },
      attachFancyWarningCell[ nb, section, string ] /;
        MatchQ[ string, Except[ "", _String ] ]
  ];


attachFancyWarningCell[nb_, id_, warning_] :=
  Catch[
      Module[
          {target, cell, margin, attached},
          target = Replace[getCellObject[nb, id], Except[_CellObject] :> Throw[$Failed, attachWarningCell]];
          NotebookDelete[CurrentValue[target, {TaggingRules, "AttachedInlineCell"}]];
          cell = With[
              {p = target},
              Cell[
                  BoxData[
                      ToBoxes[
                          DynamicModule[
                              {open = False},
                              PaneSelector[
                                  {
                                      True -> Panel[
                                          Grid[
                                              {
                                                  {
                                                      $warningIcon,
                                                      Item[
                                                          Style["Warning", Bold, FontSize -> Inherited*1.2],
                                                          Alignment -> {Left, Center}
                                                      ],
                                                      Button[
                                                          $closeButton,
                                                          NotebookDelete[EvaluationCell[]];
                                                          Null,
                                                          Appearance -> None
                                                      ]
                                                  },
                                                  {
                                                      "",
                                                      Item[Style[warning, LineIndent -> 0], ItemSize -> Automatic],
                                                      Button[$minimizeButton, open = False; , Appearance -> None]
                                                  }
                                              },
                                              Alignment -> {Left, Top}
                                          ]
                                      ],
                                      False -> MouseAppearance[
                                          Button[$warningIcon, open = True; , Appearance -> None],
                                          "LinkHand"
                                      ]
                                  },
                                  Dynamic[open],
                                  ImageSize -> Automatic
                              ]
                          ]
                      ]
                  ],
                  "Text",
                  FontColor -> Black
              ]
          ];
          margin = Replace[AbsoluteCurrentValue[target, CellMargins], {{{_, m_Integer}, _} :> m, ___ :> 0}];
          attached = MathLink`CallFrontEnd[
              FrontEnd`AttachCell[
                  target,
                  cell,
                  {Offset[{margin, 0}, 0], {Right, Top}},
                  {Right, Top},
                  "ClosingActions" -> {"ParentChanged", "EvaluatorQuit"}
              ]
          ];
          CurrentValue[target, {TaggingRules, "AttachedInlineCell"}] = attached
      ],
      attachWarningCell
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*setSuggestion*)


(*setSuggestion[nb_, section_, error_String, {params___}] :=
  With[ { string = StringTemplate[ getString[ "Hints", "Suggestions", error ] ][ params ] },
      If[ MatchQ[ string, Except[ "", _String ] ],
          attachInlineCell[nb, section, suggestionCell[section, string, True]],
          attachInlineCell[nb, section, suggestionCell[section, error, True]]
      ]
  ];*)

setSuggestion[nb_, section_, error_String, params___] :=
  setSuggestion @ <|
      "NotebookObject" -> nb,
      "ID" -> section,
      "MessageTag" -> error,
      "MessageParameters" -> { params }
  |>;

setSuggestion[ info: KeyValuePattern @ {
    "NotebookObject" -> nb_,
    "ParentCellObject" -> cell_,
    "Message" -> message_,
    "MessageTag" -> tag_
} ] :=
  Module[ { cellID },
      cellID = ToString @ CurrentValue[ cell, CellID ];
      If[ ! TrueQ @ CurrentValue[ nb, { TaggingRules, "IgnoredMessages", tag, cellID } ],
          attachInlineCell[
              nb,
              cell,
              suggestionCell[ cell, message, info ]
          ]
      ]
  ];


setSuggestion[info: KeyValuePattern[{
    "NotebookObject" -> nb_,
    "ParentCellObject" -> cellObject_,
    "MessageTag" -> tag_,
    "MessageParameters" -> { params___ }
}]] :=
  Module[ { string, message },
      string = StringTemplate[ getString[ "Hints", "Suggestions", tag ] ][ params ];
      message = If[ MatchQ[ string, Except[ "", _String ] ], string, tag ];
      setSuggestion @ Append[ info, "Message" -> message ]
  ];


setSuggestion[info: KeyValuePattern[{
    "MessageTag" -> tag_
}]] /; ! KeyExistsQ[ info, "MessageParameters" ] :=
  setSuggestion @ Append[ info, "MessageParameters" -> { } ];


setSuggestion[info: KeyValuePattern[{
    "NotebookObject" -> nb_,
    "ID" -> _
}]] /; ! KeyExistsQ[ info, "ParentCellObject" ] :=
  Module[ { cellObject },
      cellObject = getCellObject[ nb, Lookup[info, "ID"] ];
      setSuggestion @ Append[ info, "ParentCellObject" -> cellObject ] /; cellObjectQ @ cellObject
  ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*Hint cell elements*)

$warningIcon :=
  Graphics[
      {
          Thickness[0.0833],
          Style[
              {
                  FilledCurve[
                      {{{0, 2, 0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}}},
                      {
                          {
                              {9.285, 0.75},
                              {2.715, 0.75},
                              {1.215, 0.75},
                              {0.27, 2.57},
                              {1.004, 4.043},
                              {3.918, 9.879},
                              {4.703, 11.707},
                              {7.297, 11.707},
                              {8.082, 9.879},
                              {10.996, 4.043},
                              {11.73, 2.57},
                              {10.785, 0.75},
                              {9.285, 0.75}
                          }
                      }
                  ]
              },
              FaceForm[RGBColor[0.921569, 0.341176, 0.105882, 1.]]
          ],
          Style[
              {
                  FilledCurve[
                      {{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}},
                      {{{5.25, 9.}, {6.75, 9.}, {6.75, 4.5}, {5.25, 4.5}}, {{5.25, 3.75}, {6.75, 3.75}, {6.75, 2.25}, {5.25, 2.25}}}
                  ]
              },
              FaceForm[RGBColor[1., 1., 1., 1.]]
          ]
      },
      ImageSize -> {16., 16.},
      PlotRange -> {{0., 12.}, {0., 12.}},
      AspectRatio -> Automatic
  ];

(* Placeholder icon *)
$suggestionIcon :=
  Style[
      "\[LightBulb]",
      FontColor -> RGBColor["#fc6b34"],
      FontWeight -> Bold
  ];

$errorIcon :=
  Graphics[
      {
          Thickness[0.0784313725490196],
          Style[
              {
                  FilledCurve[
                      {{{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}}},
                      {
                          {
                              {11.625, 6.},
                              {11.625, 3.1},
                              {9.273, 0.75},
                              {6.375, 0.75},
                              {3.477, 0.75},
                              {1.125, 3.1},
                              {1.125, 6.},
                              {1.125, 8.898},
                              {3.477, 11.25},
                              {6.375, 11.25},
                              {9.273, 11.25},
                              {11.625, 8.898},
                              {11.625, 6.}
                          }
                      }
                  ]
              },
              FaceForm[RGBColor[0.866667, 0.0666667, 0., 1.]]
          ],
          Style[
              {
                  FilledCurve[
                      {{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}, {{0, 2, 0}, {0, 1, 0}, {0, 1, 0}}},
                      {
                          {{5.625, 9.375}, {7.125, 9.375}, {7.125, 4.125}, {5.625, 4.125}},
                          {{5.625, 3.375}, {7.125, 3.375}, {7.125, 1.875}, {5.625, 1.875}}
                      }
                  ]
              },
              FaceForm[RGBColor[1., 1., 1., 1.]]
          ]
      },
      ImageSize -> {18., 16.65},
      PlotRange -> {{0., 12.75}, {0., 12.}},
      AspectRatio -> Automatic
  ];


$minimizeButton :=
  $minimizeButton = MouseAppearance[
      Mouseover[
          Show[
              Import[
                  FileNameJoin[
                      {
                          $InstallationDirectory,
                          "SystemFiles",
                          "FrontEnd",
                          "SystemResources",
                          "Bitmaps",
                          "Toolbars",
                          "DocCenter",
                          "ForwardIcon@144dpi.png"
                      }
                  ]
              ],
              ImageSize -> 18
          ],
          Show[
              Import[
                  FileNameJoin[
                      {
                          $InstallationDirectory,
                          "SystemFiles",
                          "FrontEnd",
                          "SystemResources",
                          "Bitmaps",
                          "Toolbars",
                          "DocCenter",
                          "ForwardIconHot@144dpi.png"
                      }
                  ]
              ],
              ImageSize -> 18
          ]
      ],
      "LinkHand"
  ];

$closeButton :=
  $closeButton = MouseAppearance[
      Mouseover[
          Show[
              Import[
                  FileNameJoin[
                      {
                          $InstallationDirectory,
                          "SystemFiles",
                          "FrontEnd",
                          "SystemResources",
                          "Bitmaps",
                          "Typeset",
                          "Message",
                          "close@144dpi.png"
                      }
                  ]
              ],
              ImageSize -> 18
          ],
          Show[
              Import[
                  FileNameJoin[
                      {
                          $InstallationDirectory,
                          "SystemFiles",
                          "FrontEnd",
                          "SystemResources",
                          "Bitmaps",
                          "Typeset",
                          "Message",
                          "close-hover@144dpi.png"
                      }
                  ]
              ],
              ImageSize -> 18
          ]
      ],
      "LinkHand"
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*setHint*)


setHint[ "Warning", args___ ] := setWarning @ args;
setHint[ "Error", args___ ] := setError @ args;
setHint[ "Suggestion", args___ ] := setSuggestion @ args; (* not yet implemented *)



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*attachInlineCell*)


attachInlineCell[ nb_, id_, cell_Cell, scroll_: False, closing_: { "ParentChanged", "EvaluatorQuit" } ] :=
  Module[ { target, targetMargins, margins, new, attached },
      target = Replace[ getCellObject[ nb, id ], Except[ _CellObject ] :> Throw[ $Failed, attachInlineCell ] ];
      NotebookDelete @ CurrentValue[ target, { TaggingRules, "AttachedInlineCell" } ];
      targetMargins = AbsoluteCurrentValue[ target, CellMargins ];
      margins = Replace[ targetMargins, { { l_, r_}, _ } :> { { l, r }, { Inherited, Inherited } } ];
      new = Append[ cell, CellMargins -> margins ];
      attached = MathLink`CallFrontEnd @ FrontEnd`AttachCell[ target, new, "Inline", "ClosingActions" -> closing ];
      If[ TrueQ @ scroll, moveToCell @ target ];
      CurrentValue[ target, { TaggingRules, "AttachedInlineCell" } ] = attached
  ] ~Catch~ attachInlineCell;


moveToCell[ target_, dir_: All, unit_: Cell ] :=
  Module[ { nb },
      nb = Notebooks @ target;
      SelectionMove[ target, All, Cell ];
      FixedPoint[
          (
              SelectionMove[ nb, All, CellGroup, AutoScroll -> False ];
              If[ MatchQ[ NotebookRead @ nb, Cell[ CellGroupData[ _, Closed ], ___ ] ],
                  FrontEndTokenExecute[ nb, "OpenCloseGroup" ]
              ];
              NotebookRead @ nb
          ) &,
          Null
      ];
      SelectionMove[ target, dir, unit ];
  ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*cellObjectQ*)


cellObjectQ[ cell_CellObject ] :=
  ! FailureQ @ CurrentValue[ cell, CellStyle ];

cellObjectQ[ ___ ] :=
  False;


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*getCellObject*)


getCellObject[ _, cell_CellObject? cellObjectQ, ___ ] :=
  cell;

getCellObject[ nb_NotebookObject, id_Integer, opts: OptionsPattern[ Cells ] ] :=
  Replace[
      Cells[ nb, CellID -> id, opts ],
      {
          { cell_CellObject, ___ } :> cell,
          ___ :> $Failed
      }
  ];

getCellObject[ nb_NotebookObject, tag_String, opts: OptionsPattern[ Cells ] ] :=
  Replace[
      Cells[ nb, CellTags -> tag, opts ],
      {
          { cell_CellObject, ___ } :> cell,
          ___ :> getCellByContent[ nb, tag, opts ]
      }
  ];

getCellObject[ ___ ] :=
  $Failed;



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*getCellExpression*)


getCellExpression[ nb_Notebook, id_Integer, ___ ] :=
  FirstCase[
      nb,
      Cell[ ___, CellID -> id, ___ ],
      $Failed,
      Infinity
  ];

getCellExpression[ nb_Notebook, tag_String, ___ ] :=
  FirstCase[
      nb,
      Cell[ ___, CellTags -> { ___, tag, ___ }|tag, ___ ],
      $Failed,
      Infinity
  ];

getCellExpression[ nb_NotebookObject, args___ ] :=
  With[ { cell = getCellObject[ nb, args ] },
      NotebookRead @ cell /; MatchQ[ cell, _CellObject ]
  ];

getCellExpression[ ___ ] :=
  $Failed;



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*warningCell*)


warningCell[section_, content_, open_:False] :=
  Cell[
      BoxData[ToBoxes[prettyTooltip[warningGrid[content], getString["Tooltips", "ClickToDismissMessage"]]]],
      "WarningText",
      Deletable -> True,
      CellTags -> {createErrorTag[section]},
      CellOpen -> TrueQ[open],
      CellEventActions -> {"MouseClicked" :> clearError[SelectedNotebook[], section]}
  ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*suggestionCell*)


suggestionCell[section_, content_, info_] :=
  Cell[
      BoxData[ToBoxes[suggestionGrid[content, info]]],
      "WarningText",
      Deletable -> True,
      CellTags -> {createErrorTag[section]}
  ];




(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createErrorTag*)


createErrorTag[section_String] := StringJoin["SectionError", toCamelCase[section]];

createErrorTag[ id_Integer ] := "CellID$" <> ToString @ id;



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*toCamelCase*)


toCamelCase[str_String] :=
  Module[{split}, split = StringSplit[str, Except[WordCharacter]]; StringJoin[Capitalize /@ split]];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*warningGrid*)


warningGrid[ label_ ] :=
  Deploy[
      Grid[
          {
              {
                  $warningIcon,
                  Style[
                      label,
                      FontFamily -> "Source Sans Pro SemiBold",
                      FontWeight -> "SemiBold",
                      FontColor -> RGBColor[0.921569, 0.341176, 0.105882, 1.]
                  ]
              }
          },
          Alignment -> {Left, Bottom},
          ItemSize -> {{All, All}, All},
          Spacings -> {{1 -> 0, 2 -> 0.25}, 1 -> 1.25},
          Frame -> All,
          FrameStyle -> GrayLevel[254/255]
      ]
  ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*suggestionGrid*)




suggestionGrid[ label_, info_ ] :=
  Module[ { icon, customActions, actions, menu, button, gOpts, grid1, grid2 },

      icon = Lookup[ info, "Icon", $suggestionIcon ];

      customActions = Join[
          Lookup[ $customHintActions, info["MessageTag"], { } ],
          Lookup[ info, "CustomActions", { } ]
      ];

      actions = makeActions[info] @ Flatten @ {
          $defaultActions,
          customActions
      };

      menu = prettyTooltip[
          ActionMenu[
              $minimizeButton,
              actions,
              Appearance -> None
          ],
          getString[ "Tooltips", "ViewSuggestions" ]
      ];

      button = Button[
          prettyTooltip[
              Style[
                  label,
                  "WarningText",
                  FontFamily -> Lookup[ info, "FontFamily", "Source Sans Pro" ],
                  FontWeight -> Lookup[ info, "FontWeight", Plain ],
                  FontColor -> Lookup[ info, "FontColor", GrayLevel[ 0.5 ] ]
              ],
              getString[ "Tooltips", "ClickToDismissMessage" ]
          ],
          clearError @ info,
          Appearance -> None
      ];

      gOpts = Sequence[
          Alignment -> { Left, Center },
          ItemSize -> { { All, All }, All },
          Spacings -> { { 1 -> 0, 2 -> 0.25, 3 -> 0.25 }, 1 -> 1.25 },
          Frame -> All,
          FrameStyle -> GrayLevel[ 254/255 ]
      ];

      grid1 = Grid[ { { icon, button, Invisible @ menu } }, gOpts ];
      grid2 = Grid[ { { icon, button,             menu } }, gOpts ];

      Deploy @ Mouseover[ grid1, grid2 ]
  ];


makeActions[info_][actions_List] :=
  FixedPoint[
      Replace @ {
          { a___, Delimiter } :> { a },
          { Delimiter, a___ } :> { a },
          { a___, Delimiter, Delimiter, b___ } :> { a, Delimiter, b }
      },
      makeActions[info] /@ actions
  ];

makeActions[ _ ][ Delimiter ] :=
  Delimiter;

makeActions[ _ ][ (Rule|RuleDelayed)[ a_, b_ ] ] :=
  a :> b;

makeActions[ info_ ][ KeyValuePattern @ { "Label" -> label_, (Rule|RuleDelayed)[ "Function", func_ ] } ] :=
  makeActions[ info ][ <| "Label" -> label, "Action" :> func @ info |> ];

makeActions[ info_ ][ KeyValuePattern @ { "Label" -> label_, (Rule|RuleDelayed)[ "Action", action_ ] } ] :=
  label :> (action; NotebookDelete @ EvaluationCell[ ]);

makeActions[ info_ ][ action_Association /; ! KeyExistsQ[ action, "Label" ] ] :=
  makeActions[ info ] @ Append[ action, "Label" -> "Fix this automatically" ];

makeActions[ ___ ][ ___ ] :=
  { };



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*doNotShowAgain*)


(*doNotShowAgain[ info: KeyValuePattern @ {
    "ParentCellObject" -> cell_CellObject,
    "NotebookObject" -> nb_NotebookObject,
    "MessageTag" -> tag_
} ] :=
  Module[ { cellID },
      cellID = ToString @ CurrentValue[ cell, CellID ];
      CurrentValue[ nb, { TaggingRules, "IgnoredMessages", tag, cellID } ] = True;
      clearError @ info
  ];*)

doNotShowAgain[ info: KeyValuePattern @ {
    "ParentCellObject" -> cell_CellObject,
    "MessageTag" -> tag_,
    "MessageParameters" -> { params___ }
} ] :=
  With[ { id = CurrentValue[ cell, CellID ] },
      collectHint[ id, tag, params ] = Null;;
      clearError @ info
  ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeExampleNotebook*)


scrapeExampleNotebook[nb_NotebookObject, info:KeyValuePattern[{"Name" -> name_String}]] :=
  Module[
      {examples, updated},
      examples = Replace[
          scrapeSection[nb, "Examples"],
          {{} :> Throw[setError[nb, "Examples", "ExamplesMissing"], scrapeAll]}
      ];
      updated = FunctionResource`Private`updateInputOutputCells[examples, name, info];
      NotebookPut[removeEmptySections @ Notebook[updated, Visible -> False]]
  ];


scrapeExampleNotebook[ nb_Notebook, info: KeyValuePattern[ "Name" -> name_String ] ] :=
  Module[ { cells, examples, exampleNB },

      cells = DeleteCases[
          scrapeSection[ nb, "Examples" ],
          Alternatives @@ $defaults[ "Examples" ],
          Infinity
      ] //. Cell[ CellGroupData[ { }, _ ] ] :> Sequence[ ];

      examples = FunctionResource`Private`updateInputOutputCells[ cells, name, info ];
      exampleNB = removeEmptySections @ Notebook[ examples, Visible -> False ];

      validate @ <|
          "Metadata" -> info,
          "Property" -> "Examples",
          "Cells" -> Flatten @ List @ cells,
          "Examples" -> examples,
          "ExampleNotebook" -> exampleNB,
          "Result" -> NotebookPut @ exampleNB
      |>
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*removeEmptySections*)


removeEmptySections[ expr_ ] :=
  expr //. {
      {
          a___,
          Cell[ __, c: "Section"|"Subsection"|"Subsubsection", ___ ],
          b: Cell[ __, c_, ___ ],
          d___
      } :> { a, b, d }
      ,
      { a___, Cell[ __, "Section"|"Subsection"|"Subsubsection", ___ ] } :> { a }
      ,
      {
          a___,
          Cell[ __, c: "Section"|"Subsection"|"Subsubsection", ___ ],
          b: Cell @ CellGroupData[ { Cell[ __, c_, ___ ], ___ }, _ ],
          d___
      } :> { a, b, d }

  };



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*setError*)


setError[nb_, section_, error_String, params___] :=
  With[ { string = StringTemplate[ getString[ "Hints", "Errors", error ] ][ params ] },
      $caughtError = string;
      attachInlineCell[nb, section, errorCell[section, string, True], True] /;
        MatchQ[ string, Except[ "", _String ] ]
  ];



showError[ nb_, label_: "" ] :=
  Module[ { errorCell },
      errorCell = FirstCase[CurrentValue[nb, {TaggingRules, "AttachedHints"}], cell_CellObject /; MatchQ[CurrentValue[cell, CellStyle], {___, "ErrorText", ___}]];
      If[ cellObjectQ @ errorCell,
          With[ { anchor = ParentCell @ errorCell },
              Grid[{{ Show[$errorIcon, ImageSize -> 12], label, $caughtError, hyperlink["\[RightGuillemet]", ButtonFunction :> moveToCell @ anchor ] }}, Alignment -> { Left, Center }, Spacings -> {{2 -> 0.1}, Automatic}]
          ],
          Grid[{{ Show[$errorIcon, ImageSize -> 12], label, $caughtError }}, Alignment -> { Left, Center }, Spacings -> {{2 -> 0.1}, Automatic} ]
      ]
  ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*errorCell*)


errorCell[section_, content_, open_:False] :=
  Cell[
      BoxData[ToBoxes[prettyTooltip[errorGrid[content], getString["Tooltips", "ClickToDismissMessage"]]]],
      "ErrorText",
      Deletable -> True,
      CellTags -> {createErrorTag[section]},
      CellOpen -> TrueQ[open],
      CellEventActions -> {"MouseClicked" :> clearError[SelectedNotebook[], section]}
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*errorGrid*)


errorGrid[ label_ ] :=
  Deploy[
      Grid[
          {
              {
                  $errorIcon,
                  Style[
                      label,
                      FontFamily -> "Source Sans Pro SemiBold",
                      FontWeight -> "SemiBold",
                      FontColor -> RGBColor[0.866667, 0.0666667, 0., 1.]
                  ]
              }
          },
          Alignment -> {Left, Bottom},
          ItemSize -> {{All, All}, All},
          Spacings -> {{1 -> 0, 2 -> 0.25}, 1 -> 1.25},
          Frame -> All,
          FrameStyle -> GrayLevel[254/255]
      ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*updateExampleNotebook*)


updateExampleNotebook[ rf_, handle_ ] :=
  Module[ { ro, info, id, oldExampleNB, examples, updated, newExampleNB },
      ro = ResourceObject @ rf;
      info = ro[ All ];
      id = Lookup[ info, "UUID" ];
      oldExampleNB = Lookup[ info, "ExampleNotebook", ResourceSystemClient`CreateExampleNotebook @ ro ];
      examples = Replace[ oldExampleNB, nb_NotebookObject :> NotebookGet @ nb ];
      NotebookClose @ oldExampleNB;
      updated = FunctionResource`Private`updateInputOutputCells[ examples, handle, info ];
      newExampleNB = NotebookPut @ DeleteDuplicates @ Append[ updated, Visible -> False ];
      ResourceSystemClient`Private`setResourceInfo[ id, <| "ExampleNotebook" -> newExampleNB |> ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeExternalLinks*)


scrapeExternalLinks[nb_NotebookObject, ___] :=
  Module[
      {scraped, strings, hyperlinks},
      scraped = scrapeSection[nb, "Links"];
      strings = DeleteCases[
          Cases[scraped, Cell[url_String, "Item" | "Text", ___] :> url, Infinity],
          "Link to other related material"
      ];
      hyperlinks = Join[
          strings,
          Cases[
              scraped,
              ButtonBox[label_String, ___, ButtonData -> {URL[url_String], None}, ___] :> Hyperlink[StringTrim[label, "\""], url],
              Infinity
          ],
          Cases[
              scraped,
              ButtonBox[label_String, ___, ButtonData -> {url_String, None}, ___] :> Hyperlink[StringTrim[label, "\""], url],
              Infinity
          ],
          Cases[
              scraped,
              TemplateBox[{label_String, url_String}, "HyperlinkURL"] :> Hyperlink[StringTrim[label, "\""], url],
              Infinity
          ]
      ];
      Replace[hyperlinks, {{s:(_Hyperlink | _String)..} :> {s}, ___ :> Missing["NotFound"]}]
  ];


scrapeExternalLinks[ nb_Notebook, info_ ] :=
  Module[ { cells, scraped },

      cells = Cases[ scrapeSection[ nb, "Links" ],
          Cell[
              Except[ Alternatives @@ $defaults[ "ExternalLinks" ] ],
              "Item"|"Subitem"|"Text",
              ___
          ],
          Infinity
      ];

      scraped = scrapeLink /@ cells;

      validate @ <|
          "Metadata" -> info,
          "Property" -> "ExternalLinks",
          "Cells" -> Flatten @ List @ cells,
          "Scraped" -> scraped,
          "Result" -> scraped[[All, 2]]
      |>
  ];


scrapeLink[
    Cell[url_String, "Item" | "Text", ___, CellID -> id_, ___]] :=
  Sequence @@ Thread @ {id, StringSplit[ url, WhitespaceCharacter ]};

scrapeLink[Cell[b_, ___, CellID -> id_, ___]] :=
  Sequence @@ Thread @ {
      id,
      Replace[ Cases[b,
          ButtonBox[label_String, ___,
              ButtonData -> {URL[url_String] | url_String, None}, ___] |
            TemplateBox[{label_String, url_String}, "HyperlinkURL"] :>
            Hyperlink[StringTrim[label, "\""], url], Infinity], {} :> {$Failed} ]
  };

scrapeLink[___] := Sequence[];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeFunction*)


scrapeFunction[nb_NotebookObject, info:KeyValuePattern[{"Name" -> name_}]] :=
  Module[
      {context, fullName, defCells},
      context = FunctionResource`$ResourceFunctionTempContext;
      ClearAll @@ {StringJoin[context, "*"]};
      fullName = StringJoin[context, name];
      defCells = scrapeDefinitionCells[nb, info];
      Quiet[
          Block[
              {$Context = context, $ContextPath = {"System`", context}},
              Cases[
                  defCells,
                  Cell[BoxData[b_], ___] :> Quiet[
                      Check[
                          ToExpression[b, StandardForm],
                          Throw[setError[nb, "Definition", "DefinitionSyntax"], scrapeAll],
                          MessageName[ToExpression, "esntx"]
                      ],
                      MessageName[ToExpression, "esntx"]
                  ]
              ]
          ],
          MessageName[General, "shdw"]
      ];
      Replace[
          ToExpression[fullName, InputForm, FunctionResource`Private`minimalDefinition],
          {
              Language`DefinitionList[] :> Throw[setError[nb, "Definition", "DefinitionMissing", name], scrapeAll],
              Language`DefinitionList[HoldForm[f_] -> _, ___] :> HoldComplete[f],
              ___ :> Throw[
                  setError[
                      nb,
                      "Definition",
                      "DefinitionUndefined",
                      name
                  ],
                  scrapeAll
              ]
          }
      ]
  ];


scrapeFunction[ nb_Notebook, info: KeyValuePattern[ "Name" -> name_String ] ] :=
  Module[ { context, fullName, cells, used, syntaxErrorQ, syntaxBag, syntaxErrors, function, definition },

      context = FunctionResource`$ResourceFunctionTempContext;
      ClearAll @@ {StringJoin[ context, "*" ]};
      fullName = StringJoin[ context, name ];
      cells = scrapeDefinitionCells[ nb, info ];
      syntaxBag := syntaxBag = Internal`Bag[ ];

      used = Quiet[
          Block[
              { $Context = context, $ContextPath = { "System`", context } },
              Cases[
                  cells,
                  cell: Cell[ BoxData[ b_ ], "Input"|"Code", ___ ] :> (Quiet[
                      Check[
                          ToExpression[ b, StandardForm ],
                          syntaxErrorQ = True;
                          Internal`StuffBag[ syntaxBag, cell ],
                          ToExpression::esntx
                      ],
                      ToExpression::esntx
                  ]; cell),
                  Infinity
              ]
          ]
      ];

      syntaxErrors = If[ TrueQ @ syntaxErrorQ, Internal`BagPart[ syntaxBag, All ], { } ];
      function = ToExpression[ fullName, InputForm, HoldComplete ];
      definition = FunctionResource`Private`minimalDefinition @@ function;

      validate @ <|
          "Metadata" -> info,
          "Property" -> "Function",
          "Cells" -> Flatten @ List @ cells,
          "UsedCells" -> used,
          "FullName" -> fullName,
          "SyntaxErrors" -> syntaxErrors,
          "Definition" -> definition,
          "Result" -> function
      |>
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeDefinitionCells*)


scrapeDefinitionCells[nb_NotebookObject, KeyValuePattern[{"Name" -> name_}]] :=
  Module[
      {content, nodefault, cells},
      content = scrapeSection[nb, "Definition"];
      If[
          MatchQ[content, {Cell[BoxData[RowBox[{RowBox[{"MyFunction", "[", "]"}], ":=", "xxxx"}]], "Input", ___]}],
          setWarning[nb, "Definition", "DefinitionNotSet"]
      ];
      nodefault = If[
          name === "MyFunction",
          content,
          DeleteCases[content, Cell[BoxData[RowBox[{RowBox[{"MyFunction", "[", "]"}], ":=", "xxxx"}]], "Input", ___]]
      ];
      cells = Cases[nodefault, Cell[___, "Input" | "Code", ___], Infinity];
      If[
          cells === {} && name =!= "MyFunction",
          Throw[setError[nb, "Definition", "DefinitionMissing", name], scrapeAll]
      ];
      cells
  ];


scrapeDefinitionCells[nb_Notebook, KeyValuePattern[ "Name" -> name_ ] ] :=
  scrapeSection[ nb, "Definition" ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeKeywords*)


scrapeKeywords[nb_NotebookObject, ___] :=
  Module[
      {scraped, strings},
      scraped = scrapeSection[nb, "Keywords"];
      strings = DeleteCases[Cases[scraped, Cell[kw_String, "Item" | "Text", ___] :> kw, Infinity], "keyword 1"];
      Replace[strings, {} :> Missing["NotFound"]]
  ];


scrapeKeywords[ nb_Notebook, info_ ] :=
  Module[ { cells, strings, keywords },

      cells = scrapeSection[ nb, "Keywords" ];
      strings = Cases[ cells, Cell[ kw_String, "Item"|"Text", ___ ] :> kw, Infinity ];
      keywords = Replace[ strings, { } :> Missing[ "NotFound" ] ];

      validate @ <|
          "Metadata" -> info,
          "Property" -> "Keywords",
          "Cells" -> Flatten @ List @ cells,
          "Result" -> keywords
      |>
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeName*)


scrapeName[nb_NotebookObject, ___] :=
  Module[
      {sel, titleCell, titleString, name},
      clearError[nb, "Title"];
      SelectionMove[nb, Before, Notebook, AutoScroll -> False];
      sel = NotebookFind[nb, "Title", Next, CellStyle, AutoScroll -> False];
      Replace[sel, Except[_NotebookSelection] :> Throw[setError[nb, "Title", "TitleMissing"], scrapeAll]];
      titleCell = NotebookRead[nb];
      titleString = Replace[titleCell, Cell[title_String, "Title", ___] :> title];
      If[ !StringQ[titleString], Throw[setError[nb, "Title", "TitleNotString"], scrapeAll]];
      name = Quiet[FunctionResource`Private`makeShortName[titleString]];
      If[name === "MyFunction", setWarning[nb, "Title", "TitleNotSet"]];
      Replace[
          name,
          {Except[_?StringQ] :> Throw[setError[nb, "Title", "TitleInvalid"], scrapeAll]}
      ]
  ];


scrapeName[ nb_Notebook, info_ ] :=
  Module[ { titleCell, titleString, name },
      titleCell = FirstCase[ nb, Cell[ _, "Title", ___ ], $Failed, Infinity ];
      titleString = Replace[ titleCell, Cell[ title_String, "Title", ___ ] :> title ];
      name = Quiet @ FunctionResource`Private`makeShortName @ titleString;
      validate @ <|
          "Metadata" -> info,
          "Property" -> "Name",
          "Cells" -> Flatten @ List @ titleCell,
          "String" -> titleString,
          "Result" -> name
      |>
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeNotes*)


scrapeNotes[nb_NotebookObject, KeyValuePattern["Name" -> name_String]] :=
  ReplaceAll[
      DeleteCases[
          Cases[
              scrapeSection[nb, "Details & Options"],
              note:Cell[_, "Notes" | "TableNotes" | "Item", ___] :> note,
              Infinity
          ],
          Cell["Additional information about usage and options.", "Notes", ___]
      ],
      getReplacementRule[name]
  ];



scrapeNotes[ nb_Notebook, info: KeyValuePattern[ "Name" -> name_String ] ] :=
  Module[ { cells, noteData, notes, withTables },

      cells =
        NotebookImport[
            Notebook @ Flatten @ scrapeSection[ nb, "Details & Options" ],
            "Text"|"Notes"|"TableNotes"|"Item" -> "Cell"
        ];

      noteData = Cases[ cells,
                        note_Cell :> Replace[ convertToTI @ note,
                                              "Item" -> "Notes",
                                              { 1 }
                                     ]
                 ];

      notes = noteData /. getReplacementRule @ name;
      withTables = convertNotesTables @ notes;

      validate @ <|
          "Metadata" -> info,
          "Property" -> "Details & Options",
          "Cells" -> Flatten @ List @ cells,
          "NoteData" -> noteData,
          "Result" -> withTables
      |>
  ];


convertNotesTables[ expr_ ] :=
  expr /.
    Cell[ TextData @ Cell @ BoxData @ grid_GridBox,
          "Notes"|"TableNotes"|"Item"|"Subitem"|"Text",
          opts___
    ] :>
      Cell[ BoxData @ grid, "TableNotes", opts ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*getReplacementRule*)


getReplacementRule[name_String] :=
  Module[
      {patt, headBox, nameBox},
      patt = name | "$$Function";
      headBox = StyleBox["ResourceFunction", "ResourceFunctionSymbol"];
      nameBox = StyleBox[StringJoin["\"", name, "\""], "ResourceFunctionName"];
      patt -> StyleBox[RowBox[{headBox, "[", nameBox, "]"}], "ResourceFunctionHandle"]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeSeeAlso*)


scrapeSeeAlso[nb_NotebookObject, ___] :=
  Module[
      {scraped, strings},
      scraped = scrapeSection[nb, "Related Resource Objects"];
      strings = DeleteCases[
          Cases[scraped, Cell[kw_String, "Item" | "Text", ___] :> kw, Infinity],
          "Resource Name (from any repository)"|"GrayCode (resources from any Wolfram repository)"
      ];
      Replace[strings, {} :> Missing["NotFound"]]
  ];


scrapeSeeAlso[ nb_Notebook, info_ ] :=
  Module[ { cells, strings, related },

      cells = scrapeSection[ nb, "Related Resource Objects" ];
      strings = Cases[ cells, Cell[ kw_String, "Item"|"Text", ___ ] :> kw, Infinity ];
      related = Replace[ strings, { } :> Missing[ "NotFound" ] ];

      validate @ <|
          "Metadata" -> info,
          "Property" -> "SeeAlso",
          "Cells" -> Flatten @ List @ cells,
          "Result" -> related
      |>
  ];




(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeRelatedSymbols*)


scrapeRelatedSymbols[nb_NotebookObject, ___] :=
  Module[
      {scraped, strings},
      scraped = scrapeSection[nb, "Related Symbols"];
      strings = DeleteCases[
          Cases[scraped, Cell[kw_String, "Item" | "Text", ___] :> kw, Infinity],
          "SymbolName (documented Wolfram Language symbol)"
      ];
      Replace[strings, {} :> Missing["NotFound"]]
  ];


scrapeRelatedSymbols[ nb_Notebook, info_ ] :=
  Module[ { cells, strings, related },

      cells = scrapeSection[ nb, "Related Symbols" ];
      strings = Cases[ cells, Cell[ kw_String, "Item"|"Text", ___ ] :> kw, Infinity ];
      related = Replace[ strings, { } :> Missing[ "NotFound" ] ];

      validate @ <|
          "Metadata" -> info,
          "Property" -> "RelatedSymbols",
          "Cells" -> Flatten @ List @ cells,
          "Result" -> related
      |>
  ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeSourceMetadata*)


scrapeSourceMetadata[nb_NotebookObject, ___] :=
  Module[
      {scraped, strings},
      scraped = scrapeSection[nb, "Source/Reference Citation"];
      strings = DeleteCases[
          Cases[scraped, Cell[kw_String, "Item" | "Text", ___] :> kw, Infinity],
          "Source, reference or citation information"
      ];
      Replace[strings, {{s__String} :> Association["Citation" -> {s}], ___ :> Missing["NotFound"]}]
  ];


scrapeSourceMetadata[ nb_Notebook, info_ ] :=
  Module[ { cells, strings, citation },

      cells = scrapeSection[ nb, "Source/Reference Citation" ];
      strings = Cases[ cells, Cell[ kw_String, "Item"|"Text", ___ ] :> kw, Infinity ];

      citation =
        Replace[
            strings,
            {
                { s__String } :> <| "Citation" -> { s } |>,
                ___ :> Missing[ "NotFound" ]
            }
        ];

      validate @ <|
          "Metadata" -> info,
          "Property" -> "SourceMetadata",
          "Cells" -> Flatten @ List @ cells,
          "Result" -> citation
      |>
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeUsage*)


scrapeUsage[nb_NotebookObject, KeyValuePattern["Name" -> name_String]] :=
  Module[
      {cells, usageData, usage},
      cells = scrapeSection[nb, "Usage"];
      usageData = Cases[
          cells,
          {Cell[usage_, "UsageInputs", ___], Cell[desc_, "UsageDescription", ___]} :> convertToTI[Association["Usage" -> usage, "Description" -> desc]],
          Infinity
      ];
      usage = If[name === "MyFunction", usageData, DeleteCases[usageData, $defaultUsage]] /. getReplacementRule[name];
      Replace[
          usage,
          {
              { } :> setWarning[nb, "Usage", "UsageMissing"],
              e_ /; FreeQ[e, name | StringJoin["\"", name, "\""]] :> setWarning[nb, "Usage", "UsageMissingSymbol", name]
          }
      ];
      usage
  ];


scrapeUsage[ nb_Notebook, info: KeyValuePattern[ "Name" -> name_String ] ] :=
  Module[ { cells, filtered, regrouped, usageData, usedCellIDs, usage, $tag, unusedCellIDs },

      cells = scrapeSection[ nb, "Usage" ];
      filtered = NotebookImport[Notebook[cells], "UsageInputs"|"UsageDescription" -> "Cell"];
      regrouped = FixedPoint[
          Replace[
              {
                  a___,
                  b:Cell[___, "UsageInputs", ___],
                  c:Cell[___, "UsageDescription", ___],
                  d___
              } :> {a, {b, c}, d}
          ],
          filtered,
          100
      ];

      {usageData, usedCellIDs} = Reap[
          Cases[
              regrouped,
              c:{Cell[usage_, "UsageInputs", ___], Cell[desc_, "UsageDescription", ___]} :> (
                  Sow[getAllCellIDs[c], $tag];
                  convertToTI[Association["Usage" -> usage, "Description" -> desc]]
              ),
              Infinity
          ],
          $tag
      ];

      unusedCellIDs = Complement[getAllCellIDs[cells], Flatten[usedCellIDs]];

      usage = usageData /. getReplacementRule @ name;

      validate @ <|
          "Metadata" -> info,
          "Property" -> "Usage",
          "Cells" -> Flatten @ List @ cells,
          "UnusedCellIDs" -> unusedCellIDs,
          "UsageData" -> usageData,
          "Result" -> usage
      |>
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*convertToTI*)


convertToTI := Identity;



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*$defaultUsage*)


$defaultUsage :=
  Alternatives[
      KeyValuePattern[
          {
              "Usage" -> BoxData[RowBox[{"MyFunction", "[", StyleBox["arg", "TI"], "]"}]],
              "Description" -> TextData[
                  {
                      "explanation of what use of the argument ",
                      Cell[BoxData[StyleBox["arg", "TI"]], "InlineFormula"],
                      " does."
                  }
              ]
          }
      ],
      KeyValuePattern[
          {
              "Usage" -> BoxData[
                  RowBox[
                      {
                          "MyFunction",
                          "[",
                          RowBox[
                              {
                                  SubscriptBox[StyleBox["arg", "TI"], StyleBox["1", "TR"]],
                                  ",",
                                  SubscriptBox[StyleBox["arg", "TI"], StyleBox["2", "TR"]]
                              }
                          ],
                          "]"
                      }
                  ]
              ],
              "Description" -> TextData[
                  {
                      "explanation of what use of the arguments ",
                      Cell[BoxData[SubscriptBox[StyleBox["arg", "TI"], StyleBox["1", "TR"]]], "InlineFormula"],
                      " and ",
                      Cell[BoxData[SubscriptBox[StyleBox["arg", "TI"], StyleBox["2", "TR"]]], "InlineFormula"],
                      " does."
                  }
              ]
          }
      ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrapeVerificationTests*)


scrapeVerificationTests[nb_NotebookObject, ___] :=
  Module[
      {cells, testData, tests},
      cells = DeleteCases[
          scrapeSection[nb, "Tests"],
          Cell[
              CellGroupData[
                  {
                      Cell[
                          BoxData[RowBox[{"MyFunction", "[", RowBox[{"x", ",", "y"}], "]"}]],
                          "Input",
                          CellLabel -> "In[3]:="
                      ],
                      Cell[BoxData[RowBox[{"x", " ", "y"}]], "Output", CellLabel -> "Out[3]="]
                  },
                  Open
              ]
          ]
      ];
      testData = createTestData[cells];
      tests = Cases[createTests[testData], HoldComplete[_VerificationTest]];
      Replace[Flatten[HoldComplete @@ tests], HoldComplete[] -> Missing[]]
  ];


scrapeVerificationTests[ nb_Notebook, info: KeyValuePattern[ "Name" -> name_String ] ] :=
  Module[ { cells, testData, tests, result },

      cells = scrapeSection[ nb, "Tests" ];
      testData = createTestData @ cells;
      tests = Cases[ createTests @ testData, HoldComplete[ _VerificationTest ] ];
      result = Replace[ Flatten[ HoldComplete @@ tests ], HoldComplete[ ] -> Missing[ ] ];

      validate @ <|
          "Metadata" -> info,
          "Property" -> "VerificationTests",
          "Cells" -> Flatten @ List @ cells,
          "Result" -> result
      |>
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createTestData*)


createTestData[Cell[CellGroupData[{Cell[BoxData[input_], "Input", ___], Cell[BoxData[output_], "Output", ___]}, _]]] :=
  Check[
      Association[
          "Input" -> ToExpression[input, StandardForm, HoldComplete],
          "Output" -> ToExpression[output, StandardForm, HoldComplete]
      ],
      $Failed
  ];

createTestData[Cell[BoxData[input_], "Input", ___]] :=
  Check[Association["Input" -> ToExpression[input, StandardForm, HoldComplete], "Output" -> None], $Failed];

createTestData[Cell[CellGroupData[cells_List, _]]] := createTestData[cells];

createTestData[cells_List] := Flatten[createTestData /@ cells];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*createTests*)


createTests[KeyValuePattern[{"Input" -> HoldComplete[test_VerificationTest]}]] := HoldComplete[test];

createTests[KeyValuePattern[{"Input" -> in_ /;  !FreeQ[HoldComplete[in], _VerificationTest]}]] :=
  Cases[in, v_VerificationTest :> HoldComplete[v], Infinity];

createTests[KeyValuePattern[{"Input" -> HoldComplete[in_], "Output" -> HoldComplete[expected_]}]] :=
  HoldComplete[VerificationTest[in, expected]];

createTests[data_List] := Flatten[createTests /@ data];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*stripeSubGrid*)


stripeSubGrid[rows_] :=
  Grid[
      rows,
      Alignment -> Left,
      Dividers -> {False, {1 -> GrayLevel[0.75]}},
      Spacings -> {Automatic, 1 -> 1.5},
      FrameStyle -> Thickness[2],
      ItemSize -> Fit
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*submitRepository*)


submitRepository[nb_NotebookObject]:=
  nb ~checkForUpdates~ Block[ { $submittingQ = True },
      Module[
          {rf, uuid, so},
          SetSelectedNotebook[nb];
          runWithProgress[
              rf = Check[ Block[ { $scrapeConfig = $submitConfig }, scrapeResourceFunction @ nb ], $Failed];
              If[
                  MatchQ[rf, _ResourceFunction],
                  uuid = ResourceObject[ rf ][ "UUID" ];
                  updateExampleNotebook[ rf, uuid ];
                  so = ResourceSubmit[ResourceObject[rf]];
                  If[ MatchQ[ so, _Success|_ResourceSubmissionObject ],
                      With[ { s = so, id = so[ "SubmissionID" ] },
                          <|
                              "ResultMessage" -> "Your resource has been submitted for review.",
                              "VisibleContent" -> Grid[{{"Submission ID:", clickToCopy[nb, id]}}],
                              "HiddenContent" -> {
                                  {"Submission result:", so}
                              }
                          |>
                      ],
                      $failedExpr = so;
                      showError[nb]
                  ],
                  $failedExpr = rf;
                  showError[nb]
              ],
              nb
          ]
      ]
  ];


submissionResultMessage[so_Success] :=
  StringTemplate[so["MessageTemplate"]][so["MessageParameters"]];

submissionResultMessage[so_ResourceSubmissionObject] :=
  StringTemplate["Your resource has been submitted for review. Your submission id is `1`."][so["SubmissionID"]];

submissionResultMessage[___] :=
  "";




(******************************************************************************)
(* ::Subsection::Closed:: *)
(*scrollIfMessage*)



getMessageCells[ nb_ ] :=
  Cells[ nb, CellStyle -> "Message"|"MSG" ];

scrollAfterLast[ { ___, cell_CellObject } ] :=
  SelectionMove[ cell, After, Cell ];

scrollBeforeFirst[ { cell_CellObject, ___ } ] :=
  SelectionMove[ cell, Before, Cell ];


scrollIfMessage // Attributes = { HoldRest };
scrollIfMessage[ nb_, eval_ ] :=
  Module[ { existing },
      existing = getMessageCells @ nb;
      Check[ eval
          ,
          scrollAfterLast @ Complement[ getMessageCells @ nb, existing ];
          $Failed
      ]
  ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*viewExampleNotebook*)


viewExampleNotebook[] := ButtonNotebook[ ] ~checkForUpdates~
  Module[ { nb, cells, opts },
      nb = Import @ FileNameJoin @ {
          DirectoryName @ FunctionResource`Private`$NotebookToolsDirectory,
          "Templates",
          "ExampleDefinitionNotebook.nb"
      };

      cells = First @ nb;

      opts = List @@ DeleteCases[ Rest @ nb, WindowMargins|WindowTitle -> _ ];

      NotebookPut[
          Notebook[
              cells,
              WindowTitle -> "Sample ResourceFunction Definition Notebook",
              WindowMargins -> offsetMargins[ButtonNotebook[]],
              opts
          ]
      ]
  ];


viewStyleGuidelines[ ] :=
  checkForUpdates[
      ButtonNotebook[ ]
      ,
      SystemOpen @ $styleGuidelinesURL
  ];


$defaultStyleGuidelinesURL =
  "https://resources.wolframcloud.com/FunctionRepository/style-guidelines";

$styleGuidelinesURL :=
  Replace[
      URLParse[$ResourceSystemBase],
      {
          p: KeyValuePattern["Path" -> {___, "resourcesystem", "api", _}] :>
            $defaultStyleGuidelinesURL
          ,
          p: KeyValuePattern["Path" -> {base___, "api", _}] :> URLBuild[
              Append[
                  p,
                  "Path" -> {base, "published", "FunctionRepository", "style-guidelines"}
              ]
          ]
          ,
          ___ :> $defaultStyleGuidelinesURL
      }
  ];


checkForUpdates // Attributes = { HoldRest };

checkForUpdates[ nb_? FunctionResource`Private`notebookOutDatedQ, eval_ ] :=
  Module[ { updated },
      updated = MatchQ[ askToUpdateOnce @ nb, _NotebookObject ];
      If[ TrueQ @ updated,
          askToUpdateOnce[ nb ] = Null,
          eval
      ]
  ];


checkForUpdates[ _, eval_ ] :=
  eval;


checkForUpdates[ nb_ ] :=
  checkForUpdates[ nb, Null ];


askToUpdateOnce[ nb_ ] := askToUpdateOnce[ nb ] =
  If[ ChoiceDialog[ getString[ "MessageDialogs", "UpdateOldNotebookQ" ] ],
      FunctionResource`UpdateDefinitionNotebook[ nb, "CreateNewNotebook" -> False ]
  ];


(******************************************************************************)
(* ::Subsection::Closed:: *)
(*offsetMargins*)


offsetMargins[nb_] :=
  Module[
      {os, shifted, margins, new},
      os = 25;
      shifted = AbsoluteCurrentValue[nb, WindowMargins] + {{os, -os}, {-os, os}};
      margins = shifted /. HoldPattern[___ + Automatic + ___] :> Automatic;
      new = Replace[margins, {{left_, _}, {_, top_}} :> {{left, Automatic}, {Automatic, top}}];
      Replace[new, Except[{{_Integer, Automatic}, {Automatic, _Integer}}] -> Automatic]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*viewPreviewNotebook*)


viewPreviewNotebook[nb_NotebookObject, rf_ResourceFunction] :=
  Catch[
      Module[
          {new},
          new = Replace[
              getPreviewNotebook[rf, nb],
              Except[_Notebook] :> Throw[MessageDialog[getString["MessageDialogs", "CreatePreviewFailed"]], viewPreviewNotebook]
          ];
          NotebookClose[CurrentValue[nb, {TaggingRules, "PreviewNotebook"}]];
          $resultsNB = CurrentValue[nb, {TaggingRules, "PreviewNotebook"}] = NotebookPut[new]
      ],
      viewPreviewNotebook
  ];

viewPreviewNotebook[nb_NotebookObject, ___] :=
  Catch[
      Module[
          {rf, new},
          rf = Replace[
              scrapeResourceFunction[nb],
              Except[_ResourceFunction] :> Throw[MessageDialog[getString["MessageDialogs", "CreateResourceFailed"]], viewPreviewNotebook]
          ];
          new = Replace[
              getPreviewNotebook[rf, nb],
              Except[_Notebook] :> Throw[MessageDialog[getString["MessageDialogs", "CreatePreviewFailed"]], viewPreviewNotebook]
          ];
          NotebookClose[CurrentValue[nb, {TaggingRules, "PreviewNotebook"}]];
          $resultsNB = CurrentValue[nb, {TaggingRules, "PreviewNotebook"}] = NotebookPut[new]
      ],
      viewPreviewNotebook
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*getPreviewNotebook*)


getPreviewNotebook[rf_, nb_NotebookObject] :=
  Module[
      {name, cells, settings},
      name = Replace[ResourceObject[rf]["Name"], Except[_?StringQ] :> "ResourceFunction"];
      cells = First @ FunctionResource`DocumentationNotebook`GetDocumentationNotebook @ rf;
      settings = getWindowSettings[CurrentValue[nb, {TaggingRules, "PreviewNotebook"}]];
      Notebook[
          cells,
          StyleDefinitions -> $previewStylesheet,
          TaggingRules -> {
              "Source" -> {"NotebookObject" -> nb, "File" -> toFileName[Lookup[NotebookInformation[nb], "FileName", None]]},
              "NotebookIndexQ" -> False,
              "ResourceType" -> "Function"
          },
          Editable -> False,
          WindowTitle -> StringJoin[name, " (Preview)"],
          WindowMargins -> offsetMargins[nb],
          NotebookDynamicExpression -> Dynamic[
              Symbol[ "System`ResourceFunction" ],
              SynchronousUpdating -> False
          ],
          Sequence @@ settings
      ]
  ];



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*$previewStylesheet*)


$previewStylesheet :=
  $previewStylesheet = (
      Import @ FileNameJoin @ {
          DirectoryName[ FunctionResource`Private`$NotebookToolsDirectory, 2 ],
          "FrontEnd",
          "StyleSheets",
          "Wolfram",
          "FunctionResourcePreviewStyles.nb"
      }
  );



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*getWindowSettings*)


getWindowSettings[nb_NotebookObject] :=
  Replace[
      Quiet[Options[nb, {WindowSize, WindowMargins}]],
      Except[KeyValuePattern[{WindowSize -> _, WindowMargins -> _}]] :> { }
  ];

getWindowSettings[___] := {};



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*toFileName*)


toFileName[x_] := With[{file = Quiet[ToFileName[x]]}, file /; StringQ[file]];

toFileName[___] := None;



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*$stylesheet*)


$stylesheet :=
  $stylesheet = (
      Import @ FileNameJoin @ {
          DirectoryName[ FunctionResource`Private`$NotebookToolsDirectory, 2 ],
          "FrontEnd",
          "StyleSheets",
          "Wolfram",
          "FunctionResourceDefinitionStyles.nb"
      }
  );



(******************************************************************************)
(* ::Subsection::Closed:: *)
(*updateDefinitionNotebook*)


$updateScrapeSections = {
    "Definition",
    "Usage",
    "Details & Options",
    "Examples",
    "Contributed By",
    "Keywords",
    "Related Symbols",
    "Related Resource Objects",
    "Source/Reference Citation",
    "Links",
    "Tests",
    "Notes",
    "Author Notes",
    "Submission Notes"
};


updateDefinitionNotebook[ nb_Notebook, config_ ] :=

  Module[ { templateFile, template, scrapeSections, nbData, new  },

      templateFile = Lookup[ config, "TemplateFile", FunctionResource`Private`$CreateFunctionResourceBlank ];
      template = Get @ templateFile;

      scrapeSections = Complement[
          Replace[ Lookup[ config, "ScrapeSections", $updateScrapeSections ],
                   Automatic :> Union[ availableSlots @ template, $updateScrapeSections ]
          ],
          Lookup[ config, "DropSections", { } ],
          If[ TrueQ @ Lookup[ config, "IncludeNotes", True ], { }, { "Notes", "SubmissionNotes" } ]
      ];

      nbData = Join[
          <|
              "Name" -> getCellExpression[ nb, "Title" ],
              "Description" -> getCellExpression[ nb, "Description" ]
          |>,
          AssociationMap[
              scrapeSection[ nb, #1 ] & ,
              scrapeSections
          ]
      ];

      nbData[ "Submission Notes" ] = Flatten @ DeleteMissing @
        { nbData[ "Notes" ], nbData[ "Submission Notes" ] };

      If[ config[ "RemoveEmptyExampleSections" ],
          nbData[ "Examples" ] = removeEmptySections @
            DeleteCases[ nbData[ "Examples" ],
                         Cell[ "", "ResourceHiddenPageBreak", ___ ],
                         Infinity
            ]
      ];

      If[ config[ "UpdateRefLinkButtons" ],
          nbData = updateRefLinkButtons @ nbData
      ];

      If[ config[ "FixInlineFormulaFonts" ],
          nbData = fixInlineFormulaFonts @ nbData
      ];

      If[ config[ "ConvertPastedExampleText" ],
          nbData = convertPastedExampleText @ nbData
      ];

      If[ config[ "ConvertLegacyExampleGroups" ],
          nbData = convertLegacyExampleGroups @ nbData
      ];

      If[ config[ "ConvertStandaloneImageCells" ],
          nbData = convertStandaloneImageCells @ nbData
      ];

      If[ config[ "ConvertTableNotes" ],
          nbData = convertTableNotes @ nbData
      ];

      If[ config[ "UseDefaultsIfMissing" ],
          nbData = DeleteCases[ nbData, _Missing | { ""... } ]
      ];

      new = TemplateApply[ template, nbData ];

      new = openGroupsByName[  new, Lookup[ config, "OpenSections" , { } ] ];
      new = closeGroupsByName[ new, Lookup[ config, "CloseSections", { } ] ];

      If[ config[ "AppendUnhandledCells" ],
          new = appendUnhandledCells[nb, template, new]
      ];

      FunctionResource`Private`prepareFunctionResourceCreateNotebook @ new
  ];


updateDefinitionNotebook[nb_NotebookObject, config_] :=
  Module[ { new },
      new = NotebookPut[
          Append[
              DeleteCases[
                  updateDefinitionNotebook[NotebookGet[nb], config],
                  WindowMargins -> _
              ],
              WindowMargins -> offsetMargins[nb]
          ],
          If[ TrueQ @ Lookup[ config, "CreateNewNotebook", True ],
              Sequence @@ { },
              nb
          ]
      ];
      SelectionMove[ new, Before, Notebook, AutoScroll -> True ];
      new
  ];


updateRefLinkButtons[ nb_ ] :=
  nb /. ButtonBox[ name_String, BaseStyle -> "Link" ] :>
    With[ { box = FunctionResource`DocuToolsTemplate`FunctionLinkButton[ name, Null ] },
        box /; True
    ];


fixInlineFormulaFonts[ nb_ ] :=
  nb /.
    Cell[ a__, "InlineFormula", b___ ] /; FreeQ[ { b }, FontFamily -> _, { 1 } ] :>
      Cell[ a, "InlineFormula", FontFamily -> "Source Sans Pro", b ];


convertPastedExampleText[ nb_ ] :=
  nb /.
    cell: Cell[ __, "ExampleText", ___ ] :>
      Replace[ cell, "ExampleText" -> "Text", { 1 } ];


convertLegacyExampleGroups[ nb_ ] :=
  nb /.
    {
        a___,
        Cell["More Examples", "Subsection", ___],
        b___
    } :> Join[
        {a},
        {b} /. {
            Cell[c___, "Subsubsection", d___] :> Cell[c, "Subsection", d],
            Cell[c___, "Subsubsubsection", d___] :>
              Cell[c, "Subsubsection", d]
        }
    ];


convertStandaloneImageCells[ nb_ ] :=
  nb /.
    Cell[a: BoxData[
        GraphicsBox[TagBox[_RasterBox, _BoxForm`ImageTag, ___], ___]],
        "Input", b : Except[CellLabel -> _] ...] :> Cell[a, "Print", b];

convertTableNotes[ nb_ ] :=
  nb /.
    Cell[TextData[Cell[BoxData[a_GridBox]]], "Notes", b___] :>
      Cell[BoxData[a], "TableNotes", b];


appendUnhandledCells[nb_, template_, new_] :=
  Module[
      {cells, usedCellIDs, ignore, slots, unusedCells},
      cells = NotebookImport[nb, _ -> "Cell"];
      usedCellIDs = GeneralUtilities`DeepUniqueCases[new, HoldPattern[CellID -> id_] :> id];
      ignore = Alternatives[
          Cell["Documentation", "Section", ___, CellTags -> "Documentation", ___],
          Cell[
              "Source & Additional Information",
              "Section",
              ___,
              CellTags -> "Source & Additional Information",
              ___
          ],
          Cell["Unused Cells", "Section", ___, CellTags -> "Unused Cells", ___],
          Cell["Basic Examples"|"Scope"|"Options"|"Applications"|"Properties and Relations"|"Possible Issues"|"Neat Examples", "Subsection", ___]
      ];
      slots = availableSlots @ template;
      unusedCells = Replace[
          Cases[
              DeleteCases[cells, ignore],
              Cell[
                  Except[(key_String) | TextData[{key_String, ___}] /; MemberQ[slots, key]],
                  ___,
                  CellID -> id_ /;  !MemberQ[usedCellIDs, id],
                  ___
              ]
          ],
          {a__} :> {
              Cell[
                  CellGroupData[
                      {
                          Cell[
                              "Unused Cells",
                              "Section",
                              FontColor -> GrayLevel[0.65],
                              CellGroupingRules -> {TitleGrouping, -10000},
                              CellTags -> "Unused Cells"
                          ],
                          a
                      },
                      Closed
                  ]
              ]
          }
      ];
      Replace[new, Notebook[{a___}, b___] :> Notebook[Flatten[{a, unusedCells}], b]]
  ];


availableSlots[ template_ ] :=
  DeleteCases[GeneralUtilities`DeepUniqueCases[
      template,
      TemplateSlot[name:Except["$$Extra", _String], ___] :> name
  ], "Name"|"Description"];



closeGroupsByName[ nb_, name_ ] := openCloseGroupsByName[ nb, name, Closed ];
openGroupsByName[  nb_, name_ ] := openCloseGroupsByName[ nb, name, Open   ];


openCloseGroupsByName[ nb_, names: { __String }, open_ ] :=
  openCloseGroupsByName[ nb, Alternatives @@ names, open ];

openCloseGroupsByName[ nb_, name_String, open_ ] :=
  openCloseGroupsByName[ nb, Alternatives @ name, open ];

openCloseGroupsByName[ nb_, name_Alternatives, open_ ] :=
  nb /.
    CellGroupData[ cells: { Cell[ name | TextData[ { name, ___ } ], "Section"|"Subsection", ___ ], ___ }, _ ] :>
      CellGroupData[ cells, open ];

openCloseGroupsByName[ nb_, ___ ] :=
  nb;



(* configure for download version *)
processConfig[ a: KeyValuePattern[ "Cleanup" -> True ] ] :=
  processConfig @ Join[
      a,
      <|
          "IncludeNotes"                -> False,
          "UpdateRefLinkButtons"        -> True,
          "RemoveEmptyExampleSections"  -> True,
          "FixInlineFormulaFonts"       -> True,
          "ConvertPastedExampleText"    -> True,
          "AppendUnhandledCells"        -> False
      |>
  ];

processConfig[ a: KeyValuePattern[ "TemplateFile" -> Automatic ] ] :=
  Append[ a, "TemplateFile" -> FunctionResource`Private`$CreateFunctionResourceBlank ];

processConfig[ a_Association ] :=
  a;


FunctionResource`UpdateDefinitionNotebook // Options = {
    "IncludeNotes"                -> True,
    "CreateNewNotebook"           -> True,
    "Cleanup"                     -> False,
    "UpdateRefLinkButtons"        -> True,
    "RemoveEmptyExampleSections"  -> True,
    "FixInlineFormulaFonts"       -> True,
    "ConvertPastedExampleText"    -> True,
    "AppendUnhandledCells"        -> False,
    "ConvertLegacyExampleGroups"  -> True,
    "ConvertStandaloneImageCells" -> True,
    "ConvertTableNotes"           -> True,
    "UseDefaultsIfSectionMissing" -> True,
    "TemplateFile"                -> Automatic,
    "ScrapeSections"              -> Automatic,
    "DropSections"                -> { },
    "CloseSections"               -> { },
    "OpenSections"                -> { }
};


FunctionResource`UpdateDefinitionNotebook[ nb_, opts: OptionsPattern[ ] ] :=
  updateDefinitionNotebook[
      nb,
      Join[ Association @ Options @ FunctionResource`UpdateDefinitionNotebook,
            Association @ opts
      ] // processConfig
  ];



(******************************************************************************)
(* ::Section::Closed:: *)
(*EndPackage*)


End[ ]; (* `Private` *)

EndPackage[ ];
