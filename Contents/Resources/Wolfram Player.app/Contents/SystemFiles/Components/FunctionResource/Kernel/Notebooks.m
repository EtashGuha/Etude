(* Wolfram Language Package *)


BeginPackage["FunctionResource`"]

Begin["`Private`"] (* Begin Private Context *)


$currentTemplateVersion :=
  $currentTemplateVersion =
    getTemplateVersion @ $CreateFunctionResourceBlank;


ResourceSystemClient`Private`includeDefinitionNotebookQ[ "Function" ] = True;


$NotebookToolsDirectory=FileNameJoin[{$frDirectory,"Templates"}];

$CreateFunctionResourceBlank=FileNameJoin[{$NotebookToolsDirectory,"FunctionResourceDefinition.nb"}];

ResourceSystemClient`Private`createResourceNotebook[$FunctionResourceTypes,rest___]:=newFunctionResourceDefinitionNotebook[]

newFunctionResourceDefinitionNotebook[]:=With[{nbo=newfunctionResourceDefinitionNotebook[]},
	If[Head[nbo]===NotebookObject,    
		SetOptions[nbo,{Visible->True}];  
		SetSelectedNotebook[nbo];
		nbo
		,
		$Failed
	]
]


newfunctionResourceDefinitionNotebook[ ] :=
  With[ { nb = TemplateApply[ Get @ $CreateFunctionResourceBlank, <| |> ] },
      NotebookPut @ prepareFunctionResourceCreateNotebook @ nb
  ];


prepareFunctionResourceCreateNotebook[ nb_ ] :=
  Module[ { uuid, versioned },
      uuid = CreateUUID[ ];
      versioned = setTemplateVersion @ nb;
      Replace[ versioned,
               {
                   ResourceSystemClient`Private`$temporaryuuid :> uuid,
                   "ResourceSystemClient`Private`$temporaryuuid" :> uuid
               },
               Infinity
      ]
  ];



viewExampleDefinitionNotebook[ ] :=
  NotebookPut @ prepareFunctionResourceCreateNotebook @
    Import[ FileNameJoin @ { $NotebookToolsDirectory,
                             "ExampleDefinitionNotebook.nb" },
            "NB"
    ];


(******************************************************************************)


setTemplateVersion[ nb_Notebook /; getTemplateVersion @ nb =!= Inherited ] :=
  nb;

setTemplateVersion[ Notebook[ cells_, a___, TaggingRules -> { b___ }, c___ ] ] :=
  Notebook[
      cells,
      a,
      TaggingRules -> Append[ DeleteCases[ { b }, "TemplateVersion" -> _ ],
                              "TemplateVersion" -> $currentTemplateVersion
                      ],
      c
  ];

setTemplateVersion[ Notebook[ args: Except[ TaggingRules -> _List ] ... ] ] :=
  Notebook[ args, TaggingRules -> { "TemplateVersion" -> $currentTemplateVersion } ];

setTemplateVersion[ nb_NotebookObject ] :=
  CurrentValue[ nb, { TaggingRules, "TemplateVersion" } ] = $currentTemplateVersion;

setTemplateVersion[ other___ ] :=
  other;



getTemplateVersion[ nb: Notebook[ ___, TaggingRules -> { ___, "TemplateVersion" -> v_, ___ }, ___ ] ] :=
  v;

getTemplateVersion[ nb_NotebookObject ] :=
  CurrentValue[ nb, { TaggingRules, "TemplateVersion" } ];

getTemplateVersion[ file_String? FileExistsQ ] :=
  getTemplateVersion @ Get @ file;

getTemplateVersion[ other___ ] :=
  Inherited;


notebookOutDatedQ[ nb_ ] :=
  TrueQ @ versionGreater[
      $currentTemplateVersion,
      Replace[ getTemplateVersion @ nb, Inherited -> "0.0.0" ]
  ];


versionGreater[ v1_String, v2_String ] :=
  ! OrderedQ @ {
      PadRight[ ToExpression @ StringSplit[ v1, "." ], 5 ],
      PadRight[ ToExpression @ StringSplit[ v2, "." ], 5 ]
  };


(******************************************************************************)


cellPattern // ClearAll;
cellPattern[ s_String ] :=
  Alternatives[
      Cell[ _, s, ___ ],
      Cell[ ___, CellTags -> s, ___ ],
      Cell[ ___, CellTags -> { ___, s, ___ }, ___ ]
  ];


getCellsBetween // ClearAll;

(* Extract all the cells from a notebook that appear between the patterns c1 and c2 *)
getCellsBetween[ nb_, { c1 : Except[ _String ], c2 : Except[ _String ] } ] :=
  Replace[ NotebookImport[ nb, _ -> "Cell" ],
           {
               { ___, c1, initCells___, c2, ___ } :> { initCells },
               ___ :> { }
           }
  ];

(* Extract the cells that appear between cell styles s1 and s2 *)
getCellsBetween[ nb_, { s1_String, s2_String } ] :=
  getCellsBetween[ nb, { cellPattern @ s1, cellPattern @ s2 } ];

(* Only extract the cells between s1 and s2 that match patt *)
getCellsBetween[ nb_, { s1_, patt : Except[ _String ], s2_ } ] :=
  Cases[ getCellsBetween[ nb, { s1, s2 } ], patt | Cell[ _, patt, ___ ] ];

(* Only extract the cells between s1 and s2 that have the indicated style *)
getCellsBetween[ nb_, { s1_, style_String, s2_ } ] :=
  getCellsBetween[ nb, { s1, Cell[ _, style, ___ ], s2 } ];

(* Fail otherwise *)
getCellsBetween[ ___ ] := $Failed;


(******************************************************************************)
(* Get initialization content *)


getInitializationCells // ClearAll;

getInitializationCells[ id_, nb_ ] :=
  DeleteCases[
      Replace[
          getCellsBetween[ nb, { "MoreInfoCode", "Input"|"Code", "FunctionSection" } ],
          (* for backwards compatibility *)
          { } :> getInitializationCells2[ id, nb ]
      ],
      Cell[ ___, "ResourceExampleText", ___ ]
  ];

getInitializationCells2[ id_, nb_ ] :=
  Replace[
      getCellsBetween[ nb, { "MoreInfoDependencies", "Input"|"Code", "FunctionSection" } ],
      (* for backwards compatibility *)
      { } :> getInitializationCells1[ id, nb ]
  ];

getInitializationCells1[ _, nb_ ] :=
  getCellsBetween[ nb, { "FunctionResourceInitializationSection", "Input"|"Code", "FunctionResourceFunctionSection" } ];

getInitializationCells[ ___ ] :=
  $Failed;


getInitializationBoxes // ClearAll;
getInitializationBoxes[ id_, cells : { ___Cell } ] := Cases[ cells, Cell[ b_BoxData, ___ ] :> b ];
getInitializationBoxes[ id_, nb_ ] := getInitializationBoxes[ id, getInitializationCells[ id, nb ] ];
getInitializationBoxes[ ___ ] := $Failed;


getInitializationExpressions // ClearAll;

getInitializationExpressions[ id_, boxes : { ___BoxData }, context_ ] :=
  Block[ { $Context = context, $ContextPath = { "System`", context } },
      ToExpression[ boxes, StandardForm, HoldComplete ] //.
        {
            HoldComplete[ a___, Null, b___ ] :> HoldComplete[ a, b ],
            HoldComplete[ a___, CompoundExpression[ b__, Null ], CompoundExpression[ c__, Null ], d___ ] :>
              HoldComplete[ a, CompoundExpression[ b, c, Null ], d ]
        }
  ];

getInitializationExpressions[ id_, nb_ ] :=
  getInitializationExpressions[ id, nb, getBaseContext @ id ];

getInitializationExpressions[ id_, nb_, context_ ] :=
  getInitializationExpressions[ id, getInitializationBoxes[ id, nb ], context ];

getInitializationExpressions[ ___ ] := $Failed;


(******************************************************************************)
(* Get function content *)


getFunctionCells // ClearAll;
getFunctionCells[ _, nb_ ] := NotebookImport[ nb, "FunctionResourceContentInput" -> "Cell" ];
getFunctionCells[ ___ ] := $Failed;


$defaultBox = BoxData @ RowBox @ {
  RowBox @ { "$$Function", "[", TagBox[ _, "Placeholder" ], "]" }, ":=", TagBox[ _, "Placeholder" ]
};


getFunctionBoxes // ClearAll;
(* TODO: throw an error or make this pattern smarter *)
getFunctionBoxes[ id_, cells : { ___Cell } ] := Cases[ cells, Cell[ b_?(FreeQ[#,TagBox[_, "Placeholder"]]&), ___ ] :> b ];
getFunctionBoxes[ id_, nb_ ] := getFunctionBoxes[ id, getFunctionCells[ id, nb ] ];
getFunctionBoxes[ ___ ] := $Failed;


getFunctionExpressions // ClearAll;

getFunctionExpressions[ id_, boxes : { ___BoxData }, context_ ] :=
  Block[ { $Context = context, $ContextPath = { "System`", context } },
      ToExpression[ boxes, StandardForm, HoldComplete ] //.
        {
            HoldComplete[ a___, Null, b___ ] :> HoldComplete[ a, b ],
            HoldComplete[ a___, CompoundExpression[ b__, Null ], CompoundExpression[ c__, Null ], d___ ] :>
              HoldComplete[ a, CompoundExpression[ b, c, Null ], d ]
        }
  ];

getFunctionExpressions[ id_, nb_ ] :=
  getFunctionExpressions[ id, nb, getBaseContext @ id ];

getFunctionExpressions[ id_, nb_, context_ ] :=
  getFunctionExpressions[ id, getFunctionBoxes[ id, nb ], context ];

getFunctionExpressions[ ___ ] := $Failed;



makeNameFromCell // ClearAll;

makeNameFromCell[ Cell[ BoxData[ b_ ], ___ ] ] :=
  makeNameFromCell @ Cell[ b, ___ ];

(* if we have to make a name from the ResourceName, we'll need to format it like a symbol *)
makeNameFromCell[ Cell[ s_String, ___, CellTags -> "ResourceName", ___ ] ] :=
  makeNameFromCell @ makeShortName @ s;

(* ignore leading/trailing whitespace when looking for function name  *)
makeNameFromCell[ Cell[ s_String, ___ ] ] :=
  makeNameFromCell @ StringTrim @ s;

makeNameFromCell[ name_String ] :=

  Quiet @ Module[ { expr, symName },

      expr = Check[ noContext @ ToExpression[ name, InputForm, HoldComplete ],
                    Throw[ $failed, makeNameFromCell ]
             ];

      (* make sure that we actually have a symbol and not some other expression *)
      symName = Replace[ expr,
                         {
                             HoldComplete[ sym_Symbol? symbolQ ] :> SymbolName @ Unevaluated @ sym,
                             ___ :> Throw[ $failed, makeNameFromCell ]
                         }
                ];

      (* this prevents things like "Null" from being used, since contexts are checked *)
      If[ invalidResourceFunctionNameQ @ symName,
          Throw[ $failed, makeNameFromCell ],
          symName
      ]

  ] ~Catch~ makeNameFromCell;

makeNameFromCell[ ___ ] := $failed;



scrapeFunctionName // ClearAll;

scrapeFunctionName[ id_, nb_ ] :=

  Module[ { cells, fname, name },

      cells = NotebookImport[ nb, "ResourceTextInput" -> "Cell" ];

      (* try to get a function name from the FunctionName cell (it might be empty) *)
      fname = FirstCase[ cells,
                         cell : Cell[ ___, CellTags -> "FunctionName", ___ ] :> makeNameFromCell @ cell,
                         $Failed
              ];

      (* if no FunctionName was supplied, try to make one from the ResourceName *)
      name = If[ fname === "Null" || FailureQ @ fname,
                 FirstCase[ cells,
                            cell : Cell[ ___, CellTags -> "ResourceName", ___ ] :> makeNameFromCell @ cell,
                            "$$Function"
                 ],
                 fname
             ];

      getBaseContext @ id <> name
  ];


getFunctionString // ClearAll;
getFunctionString[ id_, ___ ] :=
  Module[ { fName, def },
      fName = getBaseContext @ id <> "$$Function";
      def = minimalDefinition @@ { fName };
      Replace[ def,
               {
                   (* if $$Function was defined as "$$Function = symbol", use "symbol" *)
                   Language`DefinitionList[ HoldForm[ f_ ] -> { OwnValues -> Verbatim[ HoldPattern ][ f_ ] :> g_Symbol? symbolQ } ] :> fullSymbolName @ g,
                   (* if $$Function was defined as "$$Function := symbol", use "symbol" *)
                   Language`DefinitionList[ HoldForm[ f_ ] -> { OwnValues -> { Verbatim[ HoldPattern ][ f_ ] :> g_Symbol? symbolQ } } ] :> fullSymbolName @ g,
                   (* in any other case, just use $$Function *)
                   ___ :> fName
               }
      ]
  ];


(******************************************************************************)


getReplacementNames // ClearAll;

getReplacementNames[ id_, nb_ ] :=

  Module[ { nameList },

      nameList = {
          "$$Function",
          Last @ StringSplit[ scrapeFunctionName[ id, nb ], "`" ],
          Last @ StringSplit[ getFunctionString @ id, "`" ]
      };

      Alternatives @@ Union @ Cases[ nameList, _? StringQ ]
  ];


(******************************************************************************)


getReplacementRule // ClearAll;

getReplacementRule[ id_, name_String, style_ ] :=
  Module[ { patt, headBox, nameBox },
      patt = name | "$$Function";
      headBox = StyleBox[ "ResourceFunction", "ResourceFunctionSymbol" ];
      nameBox = StyleBox[ "\"" <> name <> "\"", "ResourceFunctionName" ];
      patt -> StyleBox[ RowBox[ { headBox, "[", nameBox, "]" } ], "ResourceFunctionHandle" ]
  ];

getReplacementRule[ id_, nb_Notebook, "Input" | "Usage" ] :=
  Module[ { patt, name, headBox, nameBox },
      patt = getReplacementNames[ id, nb ];
      name = Last @ StringSplit[ scrapeFunctionName[ id, nb ], "`" ];
      headBox = StyleBox[ "ResourceFunction", "ResourceFunctionSymbol" ];
      nameBox = StyleBox[ "\"" <> name <> "\"", "ResourceFunctionName" ];
      patt -> StyleBox[ RowBox[ { headBox, "[", nameBox, "]" } ], "ResourceFunctionHandle" ]
  ];

getReplacementRule[ id_, nb_NotebookObject, style_ ] :=
  getReplacementRule[ id, NotebookGet @ nb, style ];


(******************************************************************************)


getFunctionExpression // ClearAll;

getFunctionExpression[ id_, held : { ___HoldComplete } ] :=

  Module[ { evaluated, name, definedQ },

      evaluated = ReleaseHold @ held;
      name = getFunctionString @ id;
      definedQ = ToExpression[ name, InputForm, minimalDefinition ] =!= Language`DefinitionList[ ];

      Replace[ ReleaseHold @ held,
               {
                   (* this enables arbitrary expressions (pure functions, etc) to be entered into the definition cell *)
                   { ___, expr : Except[ Null ] } /; ! definedQ :> HoldComplete @ expr,

                   (* if defined with SetDelayed etc, there won't be an output, so return the defined symbol *)
                   ___ /; definedQ :> ToExpression[ name, InputForm, HoldComplete ],

                   (* there may be more ways to detect defined functions here which could be added later *)
                   ___ :> $failed
               }
      ]
  ];

getFunctionExpression[ id_, args___ ] :=
  getFunctionExpression[ id, getFunctionExpressions[ id, args ] ];

getFunctionExpression[ ___ ] := $Failed;


initializeNotebookFunction // ClearAll;

initializeNotebookFunction[ id_, nb_ ] :=

  Module[ { init, templateSymbol, declaredSymbol, function, result },

      (* define auxiliary functions *)
      init = getInitializationExpressions[ id, nb ];
      ReleaseHold @ init;

      (* if the user didn't define $$Function, but used the symbol indicated by function name, we'll need to use that *)
      templateSymbol = ToExpression[ getBaseContext @ id <> "$$Function", InputForm, HoldComplete ];
      declaredSymbol = ToExpression[ scrapeFunctionName[ id, nb ], InputForm, HoldComplete ];

      function = Replace[ getFunctionExpression[ id, nb ],
                          _? FailureQ :> (
                              (* no definition for $$Function found, so set it to their given name and try again *)
                              SetDelayed @@ Flatten[ HoldComplete @@ { templateSymbol, declaredSymbol } ];
                              getFunctionExpression[ id, nb ]
                          )
                 ];

      result = ReleaseHold @ function;

      result

      (* TODO: clean up these shdw message assignments on created symbols *)
  ] ~Quiet~ General::shdw;


scrapeFunction // ClearAll;
scrapeFunction[ id_, nb_ ] :=
  With[ { expr = initializeNotebookFunction[ id, nb ], name = getFunctionString @ id },
      <|
          "ShortName" -> Last @ StringSplit[ scrapeFunctionName[ id, nb ], "`" ],
          "Function" -> HoldComplete @ expr,
          "UUID" -> id,
          "SymbolName" -> name
      |>
  ];


(******************************************************************************)


updateInputOutputCells // ClearAll;


updateInputOutputCells[exampleCells_, id_, KeyValuePattern["Name" -> name_]] :=
  Module[
      {functionInBoxes, functionOutBoxes, $$functionPatt},
      ResourceFunction;
      functionInBoxes = MakeBoxes[ResourceFunction[id]];
      functionOutBoxes = insertBoxID[ResourceFunction[id], name];
      $$functionPatt = InterpretationBox[ ___, BoxID -> name, ___ ]|"$$Function"|name;
      ReplaceAll[
          exampleCells,
          {
              Cell[BoxData[$$functionPatt], "Input", opts___] :> Cell[BoxData[functionInBoxes], "Input", opts],
              Cell[in_, style:"InlineFormula", opts___] :> Cell[
                  ReplaceAll[
                      in,
                      {
                          ButtonBox[$$functionPatt, BaseStyle -> "Link"] | $$functionPatt -> StyleBox[name, "InlineResourceFunction"]
                      }
                  ],
                  style,
                  opts
              ],
              Cell[in_, style:"VerificationTest", opts___] :> Cell[in /. {$$functionPatt -> functionInBoxes}, style, opts],
              Cell[in_, style:"Input", opts___] :> Cell[in /. {$$functionPatt -> functionOutBoxes}, style, opts],
              Cell[in_, style:"Output" | "ExpectedOutput", opts___] :> Cell[in /. {$$functionPatt -> functionOutBoxes}, style, opts]
          }
      ]
  ];

updateInputOutputCells[ ___ ] := $failed;



insertBoxID // Attributes = { HoldFirst };

insertBoxID[ expr_, name_ ] :=
  insertBoxID[ expr, name, makeResourceFunctionBoxes @ name ];

insertBoxID[ expr_, name_, InterpretationBox[ a_, b_, c___, BoxID -> _, d___ ] ] :=
  insertBoxID[ expr, name, DeleteCases[ InterpretationBox[ a, b, c, d ], BoxID -> _ ] ];

insertBoxID[ expr_, name_, InterpretationBox[ a_, b_, c___ ] ] :=
  InterpretationBox[ a, expr, BoxID -> name, c ];

insertBoxID[ expr_, name_, box_ ] :=
  InterpretationBox[ box, expr, BoxID -> name ];

insertBoxID[ ___ ] :=
  $failed;


(******************************************************************************)


removeMissingExampleSections // ClearAll;
removeMissingExampleSections[ examples_ ] :=
  examples //. {
      a___,
      Cell[ _, style : "Subsection" | "Subsubsection", ___ ],
      (* no content here, next section follows immediately *)
      next : Cell @ CellGroupData[ { Cell[ _, style_, ___ ], ___ }, _ ],
      b___
  } :>
    { a, next, b }


(******************************************************************************)


removeTemporaryCells // ClearAll;
removeTemporaryCells[ examples_ ] :=
  DeleteCases[ examples,
               Cell[ ___, "CommentCellPleaseIgnore", ___ ],
               Infinity
  ];


(******************************************************************************)


ResourceSystemClient`Private`replaceScrapedNotebookExampleSymbols[ $FunctionResourceTypes, examples_, id_, ro_ ] :=
  Module[ { shortName, symbolName, functionString, synonyms, updated, cleaned },
      shortName = resourceObjectProperty[ ro, "ShortName" ];
      symbolName = Last @ StringSplit[ resourceObjectProperty[ ro, "SymbolName" ], "`" ];
      functionString = getFunctionString @ id;
      synonyms = Union @ Cases[ { shortName, symbolName, functionString }, _? StringQ ];
      updated = updateInputOutputCells[ examples, id, resourceObjectProperty[ ro, All ] ];
      cleaned = removeMissingExampleSections @ updated;
      removeTemporaryCells @ cleaned
  ];


(******************************************************************************)


ResourceSystemClient`Private`repositoryCreateResourceFromNotebook[
    t: $FunctionResourceTypes,
    nb_
] /; ! TrueQ @ $stopOnFail :=
  Block[ { $stopOnFail = True },
      failOnMessage @
        ResourceSystemClient`Private`repositoryCreateResourceFromNotebook[ t, nb ]
  ];


(******************************************************************************)


ResourceSystemClient`Private`scrapeDefinitonNotebookContent[$FunctionResourceTypes, id_, nb_] :=
  With[{def = scrapeFunctionDefinitonNotebookContent[id, nb]},
      Association["ResourceType" -> "Function", def]
  ]

scrapeFunctionDefinitonNotebookContent[ id_, nb_ ] :=
  Module[ { functionInfo, verificationTests },
      functionInfo = scrapeFunction[ id, nb ];
      verificationTests = scrapeVerificationTests[ id, nb ];
      <|
          functionInfo,
          "VerificationTests" -> verificationTests
      |>
  ] // withContext;


usageMessageCell[]:=Cell[CellGroupData[{
	Cell[
    BoxData[RowBox[{TagBox[FrameBox["\"name\""], "Placeholder"], "[", 
       TagBox[FrameBox["\"inputs\""], "Placeholder"], "]"}]], 
    "UsageInputs",CellTags->"ResourceUsageInputs"], 
    Cell["", "UsageDescription",CellTags->"ResourceUsageDescription"]}, Open]]

insertUsageMessage[]:=(
	NotebookLocate["ResourceUsageDescription"];
	SelectionMove[EvaluationNotebook[], After, Cell];
	NotebookWrite[EvaluationNotebook[],usageMessageCell[]];
	)

replaceFunctionNotebookExampleInput[input_,str_,expr_]:=With[{objectblob=RowBox[{"ResourceObject", "[", ToString[str, InputForm], "]"}],
	functionblob=ToBoxes[ResourceFunction[str]]},
	replacePriority2[replacePriority1[input,str],objectblob,functionblob]
]

replaceFunctionNotebookExampleOutput[input_,str_,expr_]:=With[{objectblob=ToBoxes[expr],
	functionblob=ToBoxes[ResourceFunction[str]]},
	replacePriority2[input,objectblob,functionblob]
]

replacePriority1[group_,str_]:=Replace[
  group, {
   BoxData["$$Function"] -> BoxData[RowBox[{"ResourceFunction", "[", ToString[str, InputForm], "]"}]]
   }, {0,50}]

replacePriority2[group_,objectblob_,functionblob_]:=Replace[group, {"$$Object" -> objectblob,"$$Function"->functionblob}, 50]


ResourceSystemClient`Private`scrapeResourceTypeProperties[$FunctionResourceTypes, id_, nb_]:=With[
	{usageCells=scrapeUsageCells[id,nb],noteCells=scrapeNoteCells[id,nb],
		(*tests=scrapeVerificationTests[id,nb],*)wlVersion=scrapeWLVersion[id,nb]},
	DeleteMissing@Association[
		"Documentation"->Association[
			"Usage"->usageCells,
			"Notes"->noteCells
		],
		(*"VerificationTests"->tests,*)
		"WolframLanguageVersionRequired"->wlVersion
	]
]


scrapeUsageCells // ClearAll;

scrapeUsageCells[ id_, nb_NotebookObject ] :=
  scrapeUsageCells[ id, NotebookGet @ nb ];

scrapeUsageCells[ id_, nb_Notebook ] :=

  Module[ { rule },

      rule = getReplacementRule[ id, nb, "Usage" ];

      Cases[ nb,
             CellGroupData[ {
                 Cell[ usage_ /; FreeQ[ usage, TagBox[ _, "Placeholder" ] ], "UsageInputs", ___ ],
                 Cell[ desc_, "UsageDescription", ___ ],
                 ___
             }, ___ ] :>
               <|
                   "Usage" -> usage /. rule,
                   "Description" -> desc
               |>,
             Infinity
      ]
  ];

scrapeNoteCells[ _, nb_ ] := DeleteCases[ NotebookImport[ nb, "Notes"|"TableNotes" -> "Cell" ], Cell[ "" | BoxData[ "" ], ___ ] ];
scrapeWLVersion[_,nb_]:=ResourceSystemClient`Private`scrapeonetextCell[ResourceSystemClient`Private`findCellTags[nb, "ResourceWLVersion"]]


scrapeVerificationTests // ClearAll;
scrapeVerificationTests[ id_, nb_NotebookObject ] := scrapeVerificationTests[ id, NotebookGet @ nb ];
scrapeVerificationTests[ id_, nb_Notebook ] :=

  Module[ { testCellGroups, replaceNames, functionNamePatt, testCells, tempTestNB, stream, testExpressions, held },

      Needs[ "MUnit`" ];

      testCellGroups = Cases[ nb, Cell[ CellGroupData[ { Cell[_, "VerificationTest", ___ ], ___ }, ___ ] ], Infinity ];

      replaceNames = { "$$Function", scrapeFunctionName[ id, nb ], Last @ StringSplit[ getFunctionString @ id, "`" ] };

      functionNamePatt = Alternatives @@ Union @ Cases[ replaceNames, _? StringQ ];

      testCells = DeleteCases[ testCellGroups,
                               Cell[ _, Except[ "VerificationTest" | "ExpectedOutput" | "ExpectedMessage" | "TestOptions" ], ___ ],
                               4
                  ] /. functionNamePatt -> getFunctionString @ id;

      tempTestNB = NotebookPut @ Notebook[ testCells, Visible -> False ];
      stream = StringToStream @ MUnit`NotebookToTests @ tempTestNB;

      testExpressions = Cases[ ReadList[ stream, HoldComplete @ Expression ],
                               HoldComplete[ _VerificationTest ]
                        ];

      Close @ stream;
      NotebookClose @ tempTestNB;

      held = DeleteCases[ Flatten[ HoldComplete @@ testExpressions ],
                          HoldPattern @ VerificationTest[ _[ Placeholder[ "arguments" ] ],
                                                          Placeholder[ "expected output" ]
                                        ]
             ];

      (* make sure verification testing won't create symbols in Global` later *)
      replaceContext[ held, $Context -> getBaseContext @ id ]
  ];



(******************************************************************************)
(* Generate example notebooks with usage and details                          *)
(******************************************************************************)


makeRow // ClearAll;
makeRow = {
    "",
    Cell[ TextData @ { Cell[ #Usage, "UsageInputs", Background -> None ],
                       "\n", "  ",
                       Cell[ #Description, "UsageDescription", Background -> None ]
                     },
          "FunctionIntro"
    ]
} &;



wrapCharacters // ClearAll;
wrapCharacters[ expr_, chars_ ] :=
  Module[ { rules },
      rules = # -> StyleBox[ #, "Char" <> CharacterName @ # ] & /@ chars;
      expr /. rules
  ];


(******************************************************************************)


createShingleNB // ClearAll;

createShingleNB[ (ResourceFunction|ResourceObject)[ info_, ___ ] ] :=
  createShingleNB @ info;

createShingleNB[ info: KeyValuePattern[ "ExampleNotebook" -> nb_ ] ] :=
  createShingleNB[ info, nb ];

createShingleNB[ info_, nb_NotebookObject ]:=
  createShingleNB[ info, NotebookGet @ nb ];

createShingleNB[ info_, Notebook[ cells_, opts___ ] ]:=
  Notebook[
      Flatten @ {
          createShingleUsage @ info,
          createShingleDetails @ info,
          Cell[ "Examples", "Section" ],
          cells
      },
      opts
  ];


createShingleUsage // ClearAll;
createShingleUsage[ KeyValuePattern @ { "Name" -> name_, "Documentation" -> KeyValuePattern[ "Usage" -> usage_ ] } ] :=
  wrapCharacters[ Cell[ BoxData @ GridBox[ makeRow /@ usage, GridBoxAlignment -> { "Columns" -> { { Left } } } ],
                        "FunctionIntroWrap"
                  ],
                  { "[", "]", "{", "}", ",", "\[Ellipsis]" }
  ];
createShingleUsage[ ___ ] := { };



createShingleDetails // ClearAll;
createShingleDetails[ KeyValuePattern @ { "Name" -> name_, "Documentation" -> KeyValuePattern[ "Notes" -> notes_ ] } ] :=
  If[ notes === { },
      { },
      Cell[ CellGroupData[ Flatten @ {
                  Cell[ "Details and Options", "Section" ],
                  wrapCharacters[ formatTableCells @ notes,
                                  { "[", "]", "{", "}", ",", "\[Ellipsis]" }
                  ]
                }, Open ],
            "DetailsAndOptions"
      ]
  ];
createShingleDetails[ ___ ] := { };


(******************************************************************************)


createDocumentationNB // ClearAll;

createDocumentationNB[ (ResourceFunction|ResourceObject)[ info_, ___ ] ] :=
  createDocumentationNB @ info;

createDocumentationNB[ info: KeyValuePattern[ "ExampleNotebook" -> nb_ ] ] :=
  createDocumentationNB[ info, nb ];

createDocumentationNB[ info_, nb_NotebookObject ]:=If[
	ResourceSystemClient`Private`notebookObjectQ[nb],
	createDocumentationNB[ info, NotebookGet @ nb ],
	createDocumentationNB[ info, ResourceSystemClient`Private`openLocalExampleNotebook[info["UUID"] ] ]
]
  createDocumentationNB[ info, NotebookGet @ nb ];

createDocumentationNB[ info_, nb_Notebook ] :=
  Notebook[
      Flatten @ {
      	(*
          createNBTitle @ info,
          createNBUsage @ info,
          createNBDetails @ info,
          Cell[ "Examples", "Section" ],
          *)
          wrapCharacters[ First @ nb, { "[", "]", "{", "}", ",", "\[Ellipsis]" } ]
      },
      Sequence @@ DeleteCases[ Rest @ nb, Visible -> False ]
  ];

createDocumentationNB[ info_, nb:(_CloudObject|_LocalObject) ] :=
  Import[ nb, "NB" ];


createDocumentationNB[ info_, _ ] :=$Failed

createNBTitle // ClearAll;
createNBTitle[ KeyValuePattern[ "Name" -> name_] ] :=
  Cell[ name,
        FontColor -> GrayLevel[ 0.2 ],
        CellMargins -> { { 24, 22 }, { 15, 40 } },
        FontFamily -> "Source Sans Pro Semibold",
        FontSize -> 37,
        FontWeight -> "DemiBold"
  ];
createNBTitle[ ___ ] := { };


createNBUsage // ClearAll;
createNBUsage[ KeyValuePattern @ { "Name" -> name_, "Documentation" -> KeyValuePattern[ "Usage" -> usage_ ] } ] :=
  Cell[ BoxData @ GridBox[ makeRow /@ usage, GridBoxAlignment -> { "Columns" -> { { Left } } } ],
        "FunctionIntroWrap"
  ];
createNBUsage[ ___ ] := { };



createNBDetails // ClearAll;
createNBDetails[ KeyValuePattern @ { "Name" -> name_, "Documentation" -> KeyValuePattern[ "Notes" -> notes_ ] } ] :=
  If[ notes === { },
      { },
      Cell @ CellGroupData[ Flatten @ {
          Cell[ "Details and Options", "Section", ShowGroupOpener -> True, "WholeCellGroupOpener" -> True ],
          formatTableCells @ notes
      }, Closed ]
  ];
createNBDetails[ ___ ] := { };


formatTableCells // ClearAll;
(* TODO: find and format cells containing tables for details & options and update stylesheet *)
formatTableCells[ content_ ] := content;



(******************************************************************************)
(* Definition overrides                                                       *)
(******************************************************************************)

ResourceSystemClient`Private`customExampleNotebook[
    id_,
    info: KeyValuePattern[ "ResourceType" -> "Function" ]
] /; ! TrueQ @ $exampleNBOverride :=
  Block[ { $exampleNBOverride = True, nb },
  	nb=createDocumentationNB @ info;
     If[Head[nb]===Notebook,
     	NotebookPut @ nb
     	,
     	$Failed
     ]
  ] /; KeyExistsQ[ info, "ExampleNotebook" ];


(******************************************************************************)


DeployedResourceShingle`Private`exampleSection[ "Function", id_, target_, info_, nb_ ] :=
 (
      DeployedResourceShingle`Private`loadTransmogrify @ All;
     
      DeployedResourceShingle`Private`shingleexampleSection[ "Function", id, target, info, 
      	updateInputOutputCells[createShingleNB[ info, nb ], target, info] ]
 );


(******************************************************************************)


(* Use custom transformation rules and XML template for shingle pages *)


(* rules are included with ResourceSystemClient paclet *)
DeployedResourceShingle`Private`transmogrifyRules[ "Function" ] :=
  DeployedResourceShingle`Private`transmogrifyRules[ "Documentation" ];


(* shingle template is included in this paclet *)
DeployedResourceShingle`Private`templateFile[ "Function" ] :=
  FileNameJoin @ {
      $packageRoot,
      "WebpageDeployment",
      "WebResources",
      "functionshingle.xml"
  };


(******************************************************************************)



End[ ]; (* End Private Context *)

EndPackage[ ];
