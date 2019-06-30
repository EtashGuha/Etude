(* Wolfram Language Package *)


(Unprotect[#]; Clear[#]) & /@ {
    "System`DefineResourceFunction",
    "System`ResourceFunction"
};

System`DefineResourceFunction;


BeginPackage["FunctionResource`"];

Begin["`Private`"];


(******************************************************************************)


$stackLength := If[ FreeQ[ Stack[ ], StackComplete ], 12, 16 ];
$definitionDataPrefix = "ResourceFunctions/DefinitionData/";


(******************************************************************************)


definitionDataPathString // ClearAll;
definitionDataPathString[ uuid_String ] :=
  StringReplace[ StringRiffle[ Flatten @ {
                                 $definitionDataPrefix,
                                 StringPartition[ StringTake[ uuid, 4 ], 2 ],
                                 uuid
                               },
                               "/"
                 ],
                 "//" -> "/"
  ];


(******************************************************************************)


ResourceFunctionLoadedQ // ClearAll;

ResourceFunctionLoadedQ[ KeyValuePattern[ "UUID" -> uuid_ ] ] :=
  ResourceFunctionLoadedQ @ uuid;

ResourceFunctionLoadedQ[ (ResourceFunction|ResourceObject)[ info_, ___ ] ] :=
  ResourceFunctionLoadedQ @ info;

ResourceFunctionLoadedQ[ _Association ] := False;
ResourceFunctionLoadedQ[ _String      ] := False;
ResourceFunctionLoadedQ[ ___          ] := $failed;


(******************************************************************************)


ResourceFunction::remdf = "The definition for `1` is not available.";
ResourceFunction::lfail = "Failed to load resource function from `1`.";
ResourceFunction::sfail = "Failed to save resource function to `1`.";
ResourceFunction::ffail = "Failed to find resource function `1`.";
ResourceFunction::fdefn = "Failed to create definition.";
ResourceFunction::invld = "`1` contains insufficient information to create a ResourceFunction.";
ResourceFunction::invpl = "The specified PersistenceLocation is no longer available.";
ResourceFunction::invfl = "The specified definition location `1` does not exist.";

ResourceFunction::symname = "The string \"`1`\" cannot be used for a symbol name. " <>
                                   "A symbol name must start with a letter followed by letters and numbers.";


(******************************************************************************)


resourceFunctionStaticProperty // ClearAll;
resourceFunctionStaticProperty // Attributes = { HoldAllComplete };

resourceFunctionStaticProperty[ KeyValuePattern[ key_ -> value_ ], key_ ] := value;
resourceFunctionStaticProperty[ Association[ ___, key_ -> value_, ___ ], key_ ] := value;
resourceFunctionStaticProperty[ (ResourceFunction | ResourceObject)[ info_, ___ ], key_ ] := resourceFunctionStaticProperty[ info, key ];
resourceFunctionStaticProperty[ id_, arg___ ] := With[ { ro = ResourceObject @ id }, resourceFunctionStaticProperty[ ro, arg ] /; HoldComplete @ id =!= HoldComplete @ ro ];
resourceFunctionStaticProperty[ ___ ] := Missing[ "NotAvailable" ];


(******************************************************************************)


definitionDataSaveLocation // ClearAll;


definitionDataSaveLocation[ "Local", id_String ? uuidQ ] :=
  LocalObject @ resourceElementDirectory[ id, "DefinitionData" ];

definitionDataSaveLocation[ "Cloud", id_String ? uuidQ ] :=
  CloudObject @ cloudpath @ resourceElementDirectory[ id, "DefinitionData" ];


definitionDataSaveLocation[ locType_, KeyValuePattern[ "UUID" -> id_ ] ] :=
  definitionDataSaveLocation[ locType, id ];

definitionDataSaveLocation[ locType_, (ResourceFunction|ResourceObject)[ info_, ___ ] ] :=
  definitionDataSaveLocation[ locType, info ];

definitionDataSaveLocation[ ___ ] := $failed;



(******************************************************************************)


msgLoadFail // ClearAll;

msgLoadFail[ KeyValuePattern @ { "FunctionLocation" -> "Inline", "DefinitionData" -> _ }, ___  ] :=
  Message[ ResourceFunction::lfail, "the definition data" ];

msgLoadFail[ KeyValuePattern[ "FunctionLocation" -> loc: Except[ None|"Inline" ] ], ___ ] :=
  Message[ ResourceFunction::lfail, loc ];

msgLoadFail[ KeyValuePattern[ "ShortName" -> name_ ], ___ ] :=
  Message[ ResourceFunction::remdf, name ];

msgLoadFail[ ___ ] :=
  Message[ ResourceFunction::lfail, "the given arguments" ];




iImportDefinitionData // ClearAll;

iImportDefinitionData[ _, loc : _LocalObject | _CloudObject | _String ] :=
  Quiet @ ByteArray @ Import[ loc, "Binary" ];

iImportDefinitionData[ id_String, loc_PersistenceLocation ] :=
  PersistentValue[ definitionDataPathString @ id, loc ];

iImportDefinitionData[ arg___ ] := (
    msgLoadFail @ arg;
    Throw[ $failed, importDefinitionData ]
);



importDefinitionData // ClearAll;

importDefinitionData[ id_, $Failed ] := $Failed;

importDefinitionData[ KeyValuePattern @ { "FunctionLocation" -> "Inline", "DefinitionData" -> def_ } ] :=
  def;

importDefinitionData[ info : KeyValuePattern @ { "FunctionLocation" -> None, "SymbolName" -> name_ } ] :=
  ToExpression[ name,
                InputForm,
                Function[ s,
                          serializeWithDefinitions @ Unevaluated @ s,
                          HoldAllComplete
                ]
  ];

importDefinitionData[ info:KeyValuePattern @ { "UUID" -> id_ },loc:HoldPattern[_CloudObject] ] :=
  With[{def=Quiet @ ByteArray @ Import[ loc, "Binary" ], local=definitionDataSaveLocation[ "Local", id ]},
  	Check[ exportDefinitionData[ id, local, def ];
  			updateInfo[ <| info,"FunctionLocation"-> local|>],
	        Throw[ $failed, importDefinitionData ]
	 ];
	 def
  ]/;MemberQ[ResourceSystemClient`Private`$localResources,id]
  
  
importDefinitionData[ KeyValuePattern @ { "UUID" -> id_, "FunctionLocation" -> loc_ } ] :=
  importDefinitionData[ id, loc ];

importDefinitionData[ (ResourceFunction|ResourceObject)[ info_, ___ ] ] :=
  importDefinitionData @ info;

importDefinitionData[ id_, loc_ ] :=
  Catch[ Replace[ iImportDefinitionData[ id, loc ],
                  Except[ _ByteArray ? ByteArrayQ ] :> (
                      Message[ ResourceFunction::lfail, loc ];
                      $failed
                  )
         ],
         importDefinitionData
  ];

importDefinitionData[ arg___ ] := (
    Message[ ResourceFunction::lfail, arg ];
    $failed
);


(******************************************************************************)


iExportDefinitionData // ClearAll;

iExportDefinitionData[ id_, loc: _LocalObject | _CloudObject | _String, bytes_ByteArray ] :=
  Module[ { old },
      old = Quiet @ Import[ loc, "WXF" ];
      If[ MatchQ[ old, KeyValuePattern[ "UUID" -> id ] ]
          ,
          Replace[ old,
              KeyValuePattern @ { "UUID" -> uuid_, "Definition" -> def_ } :> (
                  Quiet[ Language`ExtendedFullDefinition[ ] = def ];
                  ResourceFunctionLoadedQ[ uuid ] = True;
              )
          ];
          loc
          ,
          Replace[ Export[ loc, bytes, "Binary" ],
                   Except[ loc ] :> (
                       Message[ ResourceFunction::sfail, loc ];
                       Throw[ $failed, exportDefinitionData ]
                   )
          ]
      ]
  ];

iExportDefinitionData[ KeyValuePattern[ "UUID" -> uuid_ ], loc_, bytes_ByteArray ] :=
  iExportDefinitionData[ uuid, loc, bytes ];

iExportDefinitionData[ id_String, loc_PersistenceLocation, bytes_ByteArray ] :=
  PersistentValue[ definitionDataPathString @ id, loc ] = bytes;

iExportDefinitionData[ _, "Inline", bytes_ByteArray ] :=
  bytes;



exportDefinitionData // ClearAll;

exportDefinitionData[ id_, loc_, bytes_ ] :=
  Catch[ Replace[ Check[ iExportDefinitionData[ id, loc, bytes ], $Failed ],
                  $Failed :> (
                      Message[ ResourceFunction::sfail, loc ];
                      $failed
                  )
         ],
         exportDefinitionData
  ];

exportDefinitionData[ arg___ ] := (
    Message[ ResourceFunction::sfail, arg ];
    $failed
);


(******************************************************************************)


loadResourceFunction // ClearAll;


loadResourceFunction[ _? ResourceFunctionLoadedQ ] := Null;


loadResourceFunction[ (ResourceFunction|ResourceObject)[ info_, ___ ] ] :=
  loadResourceFunction @ info;


loadResourceFunction[ info : KeyValuePattern @ { "FunctionLocation" -> "Inline", "DefinitionData" -> bytes_ByteArray } ] :=
  loadResourceFunction[ info, bytes ];


loadResourceFunction[ info : KeyValuePattern @ { "UUID" -> uuid_, "FunctionLocation" -> loc_ } ] :=
  loadResourceFunction[ info,
                        Check[ importDefinitionData[ info,
                                                     standardizeFunctionLocation @ loc
                               ],
                               Throw[ $Failed, loadResourceFunction ]
                        ]
  ] ~Catch~ loadResourceFunction;


$reacquired=False;
loadResourceFunction[ info:KeyValuePattern[{ "UUID" -> uuid_ }] ] :=
  With[{resp=ResourceSystemClient`Private`resourceacquire[uuid, 
		<|System`ResourceSystemBase->ResourceSystemClient`Private`resourcerepositoryBase[info]|>]},
  	If[Head[resp]===ResourceObject,
  		Block[{$reqacquired=True},
  			loadResourceFunction[ resp[All]]
  		],
  		Throw[ $Failed, loadResourceFunction ]
  		
  	]
  ] ~Catch~ loadResourceFunction /;!KeyExistsQ[info,"FunctionLocation"]&&!TrueQ[$reacquired]
  
loadResourceFunction[ KeyValuePattern @ { "UUID" -> uuid_, "FunctionLocation" -> loc_ }, bytes_ByteArray ] :=
  Catch[ Check[ deserializeWithDefinitions @ bytes,
                Message[ ResourceFunction::lfail, loc ];
                Throw[ $Failed, loadResourceFunction ]
         ];
         ResourceFunctionLoadedQ @ uuid = True;
         ,
         loadResourceFunction
  ];


loadResourceFunction[ id_String ] :=
  Catch[ loadResourceFunction @ Check[ ResourceObject @ id,
                                       Message[ ResourceFunction::ffail, id ];
                                       Throw[ $Failed, loadResourceFunction ]
                                ],
         loadResourceFunction
  ];


loadResourceFunction[ args___ ] := (
  Message[ ResourceFunction::ffail, HoldForm @ args ];
  $failed
);


(******************************************************************************)


saveResourceFunction // ClearAll;

(* at this point the function has already resolved to its internal symbol value, so there's nothing to save *)
saveResourceFunction[ rf_, ___ ] /; TrueQ @ $applyingSubValuesQ := rf;

(* definitions are already stored in the resource function, so nothing to do here *)
saveResourceFunction[ rf : HoldPattern @ ResourceFunction @ ResourceObject[ KeyValuePattern @ {
    "FunctionLocation" -> "Inline",
    "DefinitionData" -> _ByteArray
}, ___ ] ] := rf;

(* avoid potential undesired recursion *)
saveResourceFunction[ rf_, ___ ] /; TrueQ @ $savingInProgress := rf;


saveResourceFunction[ rf_ResourceFunction ] /; ! TrueQ @ ResourceFunctionLoadedQ @ rf :=
  Module[ { },
      loadResourceFunction @ rf;
      If[ ResourceFunctionLoadedQ @ rf,
          saveResourceFunction @ rf,
          $failed
      ]
  ];

saveResourceFunction[ rf_ResourceFunction ] := Block[ { $savingInProgress = True },
  Catch[
      Module[ { symbolName, heldSymbol, bytes, uuid, location, exported, newInfo },

          symbolName = resourceFunctionStaticProperty[ rf, "SymbolName"       ];
          uuid       = resourceFunctionStaticProperty[ rf, "UUID"             ];
          location   = resourceFunctionStaticProperty[ rf, "FunctionLocation" ];

          heldSymbol = ToExpression[ symbolName, InputForm, HoldComplete ];
          bytes      = Replace[ heldSymbol, HoldComplete[ s_ ] :> serializeWithDefinitions @ Unevaluated @ s ];

          exported = Check[ exportDefinitionData[ uuid, location, bytes ],
                            Throw[ $failed, saveResourceFunction ]
                     ];

          newInfo =
            Replace[ location,
                {
                    "Inline" :> If[ ByteArrayQ @ exported,
                                    insertResourceFunctionInfo[ rf, "DefinitionData" -> exported ],
                                    Throw[ $failed, saveResourceFunction ]
                                ]
                    ,
                    ___ :> dropResourceFunctionInfo[ rf, "DefinitionData" ]
                }
            ];

          (* make sure resource info will be up to date *)
          updateInfo @ newInfo
      ],
      saveResourceFunction
  ] ];


saveResourceFunction[ id_ ] :=
  With[ { rf = ResourceFunction @ id },
      saveResourceFunction @ rf /; standardFunctionInfoQ @ rf
  ];


saveResourceFunction[ rf_, loc_ ] := (
    loadResourceFunction @ rf;
    saveResourceFunction @ insertResourceFunctionInfo[ rf, "FunctionLocation" -> standardizeFunctionLocation @ loc ]
);


saveResourceFunction[ arg___ ] := arg;


(******************************************************************************)


inlineDefinitions // ClearAll;
inlineDefinitions[ rf_ ] := saveResourceFunction[ rf, "Inline" ];


(******************************************************************************)


updateInfo // ClearAll;

updateInfo[ info : KeyValuePattern[ "UUID" -> uuid_ ] ] :=

  Module[ { newInfo, cached, joined },

      If[ ! MemberQ[ ResourceSystemClient`Private`$loadedResources, uuid ],
          insertResourceFunctionInfo[ info, "Autoload" -> True ]
      ];

      newInfo = Join[ resourceObjectProperty[ info, All ], info ];

      cached = Replace[ ResourceSystemClient`Private`resourceInfo @ uuid,
                        Except[ _Association ] :> <| |>
               ];

      joined = Replace[ Join[ cached, newInfo ],
                        Except[ _Association ] :> Throw[ $failed, updateInfo ]
               ];

      ResourceSystemClient`Private`setResourceInfo[ uuid, joined ];

      joined
  ] ~Catch~ updateInfo;


updateInfo[ (r : ResourceFunction|ResourceObject)[ info_, a___ ] ] :=
  r[ updateInfo @ info, a ];


(******************************************************************************)


insertResourceFunctionInfo // ClearAll;

insertResourceFunctionInfo[ info_Association, (rule : Rule|RuleDelayed)[ key_, value_ ] ] :=
  ResourceFunction @ ResourceObject @
    If[ KeyExistsQ[ info, key ],
        Insert[ info, rule[ key, value ], Key @ key ],
        Append[ info, rule[ key, value ] ]
    ];

insertResourceFunctionInfo[ (ResourceFunction|ResourceObject)[ info_, ___ ], new_ ] :=
  insertResourceFunctionInfo[ info, new ];

insertResourceFunctionInfo[ ___ ] :=
  $failed;


(******************************************************************************)


dropResourceFunctionInfo // ClearAll;

dropResourceFunctionInfo[ info_Association, key_ ] :=
  ResourceFunction @ ResourceObject @ KeyDrop[ info, key ];

dropResourceFunctionInfo[ (ResourceFunction|ResourceObject)[ info_, ___ ], key_ ] :=
  dropResourceFunctionInfo[ info, key ];

dropResourceFunctionInfo[ ___ ] :=
  $failed;


(******************************************************************************)
(* Argument Evaluation                                                        *)
(******************************************************************************)

HoldPattern @ ResourceFunction[ ResourceObject[ info : KeyValuePattern @ {
    "UUID" -> _,
    "SymbolName" -> _
}, ___ ] ] :=

  Block[ { $applyingSubValuesQ = False },

      $applyingSubValuesQ =
        TrueQ @ StackInhibit @
          With[ { stack = Take[ Reverse @ Stack[ _ ], UpTo @ $stackLength ] },
              MatchQ[ stack, { ___, HoldForm[ ResourceFunction[ _ ][ ___ ] ], ___ } ]
          ];

      Catch[
          Module[ { uuid, symName },
              uuid    = Lookup[ info, "UUID"      , Throw[ $failed, ResourceFunction ] ];
              symName = Lookup[ info, "SymbolName", Throw[ $failed, ResourceFunction ] ];

              If[ ! TrueQ @ ResourceFunctionLoadedQ @ uuid,
                  Replace[ loadResourceFunction @ info,
                           _? FailureQ :> Throw[ $failed, ResourceFunction ]
                  ]
              ];

              Symbol @ symName
          ],
          ResourceFunction
      ] /; $applyingSubValuesQ
  ];


HoldPattern @ ResourceFunction[ ro : ResourceObject[ KeyValuePattern @ {
    "UUID" -> uuid_,
    "SymbolName" -> name_
}, ___ ] ][ args___ ] :=
  If[ TrueQ @ ResourceFunctionLoadedQ @ uuid
      ,
      Symbol[ name ][ args ]
      ,
      Replace[ loadResourceFunction @ ro,
               _? FailureQ :> Throw[ $failed, ResourceFunction ]
      ];
      Symbol[ name ][ args ]
  ] ~Catch~ ResourceFunction;



(******************************************************************************)
(* Obtain Resource From Other Arguments                                       *)
(******************************************************************************)

HoldPattern[ ResourceFunction[ id_String ] ] :=
  failOnMessage @ ResourceFunction @ ResourceObject[ id ];


HoldPattern[ ResourceFunction[ info_Association ] ] :=
  failOnMessage @ ResourceFunction @ ResourceObject @ info;


HoldPattern[ ResourceFunction[ obj : (_LocalObject | _CloudObject) ] ] :=
  failOnMessage @ ResourceFunction @ ResourceObject @ obj;


HoldPattern[ ResourceFunction[ nb_NotebookObject ] ] :=
  failOnMessage @ Quiet[ Check[ ResourceFunction @ ResourceObject @ nb,
                                (* workaround for an error that only appears when running verification tests *)
                                ResourceFunction @ ResourceObject @ nb,
                                ResourceObject::noas
                         ],
                         ResourceObject::noas
                  ];


(*HoldPattern[
    ResourceFunction @ ResourceSubmissionObject @
      KeyValuePattern[ "UUID" -> uuid_? ResourceFunctionLoadedQ ]
] :=
  ResourceFunction @ uuid;


HoldPattern[
  ResourceFunction[
      rso: ResourceSubmissionObject @
        KeyValuePattern @ {
            "ResourceType" -> "Function",
            "Download" -> KeyValuePattern[ "Automatic"|Automatic -> loc_ ]
        }
  ]
] :=
  failOnMessage @ inlineDefinitions @
    Insert[ rso @ All,
            "FunctionLocation" -> loc,
            Key @ "FunctionLocation"
    ];*)


ResourceFunction[ $Failed ] := $failed;



(******************************************************************************)
(* Resource Function Properties                                               *)
(******************************************************************************)


resourceObjectProperty // ClearAll;

resourceObjectProperty[ HoldPattern @ ResourceObject[ info_, ___ ], property_ ] :=
  Quiet[ Check[ ResourceObject[ info ][ property ]
                ,
                ResourceObject[ Append[ info, "Autoload" -> True ] ][ property ]
                ,
                { ResourceObject::notf, KeySortBy::invrl }
         ],
         { ResourceObject::notf, KeySortBy::invrl }
  ];

resourceObjectProperty[ ro_, property_ ] :=
  resourceObjectProperty[ ResourceObject @ ro, property ];

resourceObjectProperty[ ___ ] := $failed;


(******************************************************************************)


loadedResourceObjectProperty[ ro_, property_ ] := (
    loadResourceFunction @ ro;
    resourceObjectProperty[ ro, property ]
);

loadedResourceObjectProperty[ ___ ] :=
  $failed;


(******************************************************************************)


$reservedProperties = {
    "DocumentationNotebook",
    "ResourceObject",
    "Symbol",
    "Function",
    "Definition",
    "Usage",
    "VerificationTests",
    "TestReport"
};


ResourceSystemClient`Private`repositoryReservedProperties[ $FunctionResourceTypes ] =
  $reservedProperties;


ResourceSystemClient`Private`repositoryResourceMetadataLookup[
    $FunctionResourceTypes,
    id_,
    _,
    prop: Alternatives @@ $reservedProperties,
    ___
] :=
  ResourceFunction[ id, prop ];


(******************************************************************************)


(* avoid returning the notebook object *)
HoldPattern[ ResourceFunction[ rf_, "ExampleNotebook" ] ] :=
  resourceObjectProperty[ rf, "ExampleNotebook" ];


HoldPattern[ ResourceFunction[ rf_, "DocumentationNotebook" ] ] :=
  FunctionResource`DocumentationNotebook`ViewDocumentationNotebook @
    ResourceFunction @ rf;


HoldPattern[ ResourceFunction[ rf_, "ResourceObject" ] ] :=
  ResourceObject @ rf;


HoldPattern[ ResourceFunction[ rf_, "Symbol" ] ] :=
  Replace[ loadedResourceObjectProperty[ rf, "SymbolName" ],
           {
               name_String :> ToExpression[ name, InputForm, HoldForm ],
               ___ :> Missing[ "NotAvailable" ]
           }
  ];


(* TODO: since "Function" can be reconstructed from "SymbolName", it should be dropped from the metadata *)
HoldPattern[ ResourceFunction[ rf_, "Function" ] ] :=
  Replace[ loadedResourceObjectProperty[ rf, "SymbolName" ],
           {
               name_String :> Symbol @ name,
               ___ :> Missing[ "NotAvailable" ]
           }
  ];


HoldPattern[ ResourceFunction[ rf_, "DefinitionList" ] ] :=
  Replace[ loadedResourceObjectProperty[ rf, "SymbolName" ],
           {
               name_String :> minimalFullDefinition @ name,
               ___ :> Missing[ "NotAvailable" ]
           }
  ];


HoldPattern[ ResourceFunction[ rf_, "DefinitionData" ] ] :=
  Replace[ loadResourceFunction @ rf;
           importDefinitionData @ ResourceFunction @ rf,
           {
               data_ByteArray ? ByteArrayQ :> data,
               ___ :> Missing[ "NotAvailable" ]
           }
  ];


$symbolProperties = {
    "Attributes",
    "Context",
    "DefaultValues",
    "Definition",
    "DownValues",
    "FormatValues",
    "FullDefinition",
    "Messages",
    "NValues",
    "Options",
    "OwnValues",
    "SubValues",
    "UpValues"
};


symbolPropertyQ[ Alternatives @@ $symbolProperties ] := True;
symbolPropertyQ[ sym_Symbol? symbolQ ] := symbolPropertyQ @ SymbolName @ sym;
symbolPropertyQ[ ___ ] := False;

toSymbolProperty[ name_String ] := Symbol @ name;
toSymbolProperty[ sym_Symbol ] := sym;
toSymbolProperty[ ___ ] := $Failed &;


HoldPattern[ ResourceFunction[ rf_, s_? symbolPropertyQ ] ] :=
  Module[ { name, held, prop },
      name = loadedResourceObjectProperty[ rf, "SymbolName" ];
      held = ToExpression[ name, InputForm, HoldComplete ];
      prop = toSymbolProperty @ s;
      Replace[ held,
               {
                   HoldComplete[ sym_? symbolQ ] :> prop @ sym,
                   ___ :> Missing[ "NotAvailable" ]
               }
      ]
  ];


HoldPattern[ ResourceFunction[ rf_, "Usage" ] ] /; $VersionNumber < 12 :=
  Information[ ResourceFunction @ rf, LongForm -> False ];

HoldPattern[ ResourceFunction[ rf_, "Usage" ] ] :=
  ResourceFunctionInformation[ rf, "Usage" ];


HoldPattern[ ResourceFunction[ ResourceObject[ info_, ___ ], "VerificationTests" ] ] :=
  Lookup[ info, "VerificationTests", Missing[ "NotAvailable" ] ];


HoldPattern[ ResourceFunction[ ResourceObject[ info_, ___ ], "TestReport" ] ] :=
  Catch[
      Module[ { tests, report },

          loadResourceFunction @ info;

          tests = Lookup[ info,
                          "VerificationTests",
                          Throw[ Missing @ "NotAvailable", $testReportTag ]
                  ];

          report = Replace[ tests, HoldComplete[ vt___ ] :> TestReport @ { vt } ];

          Insert[ report,
                  "Title" -> Lookup[ info, "Name", "Automatic" ],
                  { 1, Key @ "Title" }
          ]
      ],
      $testReportTag
  ];


HoldPattern[ ResourceFunction[ id : Except[ _ResourceObject ], property_ ] ] :=
  ResourceFunction[ ResourceObject @ id, property ];


HoldPattern[ ResourceFunction[ rf_, property_ ] ] :=
  resourceObjectProperty[ rf, property ];


HoldPattern[ ResourceFunction[ rf_ResourceFunction ] ] :=
  rf;


(******************************************************************************)
(* UpValues for ResourceFunction                                              *)
(******************************************************************************)


ResourceFunction /:
  HoldPattern[ ResourceRemove[ rf_ResourceFunction ] ] :=
    ClearAll @ rf;


$allSymbolFunctions := {
    Attributes,
    ClearAttributes,
    Context,
    DefaultValues,
    Definition,
    DownValues,
    GeneralUtilities`Definitions,
    GeneralUtilities`PrintDefinitions,
    GeneralUtilities`PrintDefinitionsLocal,
    FormatValues,
    FullDefinition,
    If[ $VersionNumber < 12, Information, Nothing ],
    Language`ExtendedDefinition,
    Language`ExtendedFullDefinition,
    Messages,
    NValues,
    Options,
    OwnValues,
    Protect,
    SetAttributes,
    SetOptions,
    SubValues,
    SymbolName,
    Unprotect,
    UpValues
};


$symbolFunctions := $symbolFunctions =
  Select[ $allSymbolFunctions,
          ! FreeQ[ Attributes @ #,
                   HoldAll | HoldAllComplete | HoldFirst | HoldRest
            ] &
  ];


$symbolFunctionsUnevaluated := $symbolFunctionsUnevaluated =
  Complement[ $allSymbolFunctions, $symbolFunctions ];


With[ { sf = Alternatives @@ $symbolFunctions },
    ResourceFunction /:
      HoldPattern[ (f : sf)[ rf_ResourceFunction, a___ ] ] :=
      Replace[ resourceFunctionStaticProperty[ rf, "SymbolName" ],
          {
              name_String :> (
                  loadResourceFunction @ rf;
                  ToExpression[ name,
                                InputForm,
                                Function[ s, f[ s, a ], { HoldAllComplete } ]
                  ]
              ),
              ___ :> Missing[ "NotAvailable" ]
          }
      ] // failOnMessage;
];


With[ { sf = Alternatives @@ $symbolFunctionsUnevaluated },
    ResourceFunction /:
      HoldPattern[ (f : sf)[ rf_ResourceFunction, a___ ] ] :=
      Replace[ resourceFunctionStaticProperty[ rf, "SymbolName" ],
          {
              name_String :> (
                  loadResourceFunction @ rf;
                  ToExpression[ name,
                                InputForm,
                                Function[ s, f[ Unevaluated @ s, a ], { HoldAllComplete } ]
                  ]
              ),
              ___ :> Missing[ "NotAvailable" ]
          }
      ] // failOnMessage;
];



(******************************************************************************)
(* Information                                                                *)
(******************************************************************************)

ResourceFunction /:
  HoldPattern[ Information`GetInformation[ rf_ResourceFunction ] ] :=
    ResourceFunctionInformationData @ rf;


ResourceFunction /:
  HoldPattern[ Information`GetInformationSubset[ rf_ResourceFunction, props_List ] ] :=
    ResourceFunctionInformationData[ rf, props ];


ResourceFunction /:
  Information`OpenerViewQ[ ResourceFunction,
                           Alternatives[
                               "Attributes",
                               "ExternalLinks",
                               "Keywords",
                               "Options",
                               "ResourceLocations",
                               "SeeAlso",
                               "Documentation"
                           ]
  ] :=
    True;


ResourceFunction /:
  HoldPattern[ Clear[ rf_ResourceFunction ] ] :=
    clearResourceFunction @ rf;

ResourceFunction /:
  HoldPattern[ ClearAll[ rf : ResourceFunction[ ro_ResourceObject ] ] ] := (
    clearResourceFunction @ rf;
    ResourceRemove @ ro
);



(******************************************************************************)
(* Box Formatting                                                             *)
(******************************************************************************)


$rfBoxColor = RGBColor[ "#f7f7f7" ];
$rfFrmColor = RGBColor[ "#dcdcdc" ];
$rfTxtColor = RGBColor[ "#474747" ];
$rfIcnColor = RGBColor[ "#fc6640" ];

$icon :=
  Style[ "\[EmptyCircle]",
         FontSize -> Inherited * 1.2,
         ShowStringCharacters -> False,
         FontColor -> $rfIcnColor,
         FontWeight -> Dynamic @ FEPrivate`If[ CurrentValue @ Evaluatable, Bold, Plain ]
  ];

boxLabel[ name_String ] :=
  Style[ name,
         FontColor            -> $rfTxtColor,
         ShowAutoStyles       -> False,
         ShowStringCharacters -> False,
         FontFamily           -> "Roboto",
         FontSize -> Inherited * 0.9
         ,
         FontWeight -> Dynamic @
           FEPrivate`If[ CurrentValue @ Evaluatable,
                         "DemiBold",
                         Plain
           ]
  ];


frame[name_] :=
  Framed[ Grid[ { { $icon, boxLabel @ name } },
                Alignment -> { Left, Center },
                Spacings -> 0.25
          ],
          Background        -> $rfBoxColor,
          ContentPadding    -> False,
          FrameMargins      -> { { 3, 4 }, { 2, 2 } },
          FrameStyle        -> Directive[ Thickness @ 1, $rfFrmColor ],
          RoundingRadius    -> 3,
          StripOnInput      -> False,
          Selectable        -> False
  ];


makeResourceFunctionBoxes // Attributes = { HoldAllComplete };

makeResourceFunctionBoxes[
    rf: ResourceFunction @ ResourceObject[ KeyValuePattern[ "ShortName" -> name_ ], ___ ],
    fmt_: StandardForm
] := makeResourceFunctionBoxes[ rf, fmt ] =
  With[ { label = makeShortName @ name },
      Append[ ToBoxes[ Interpretation[ frame @ label, rf ], fmt ],
              Selectable -> False
      ]
  ];

makeResourceFunctionBoxes[ name_String, fmt_: StandardForm ] :=
  makeResourceFunctionBoxes[ name, fmt ] =
    Append[ ToBoxes[ Interpretation[ frame @ name, ResourceFunction @ name ], fmt ],
            Selectable -> False
    ];

ResourceFunction /:
  MakeBoxes[ rf: ResourceFunction @ ResourceObject[ KeyValuePattern[ "ShortName" -> name_ ], ___ ], fmt_ ] :=
    makeResourceFunctionBoxes[ rf, fmt ];



(******************************************************************************)
(* Deployment                                                                 *)
(******************************************************************************)


ResourceFunction /:
  HoldPattern[ LocalCache[ rf_ResourceFunction, args___ ] ] :=
    LocalCache[ ResourceObject @ rf, args ];

ResourceFunction /:
  HoldPattern @ CloudDeploy[ rf_ResourceFunction, args___ ] :=
    CloudDeploy[ ResourceObject @ rf, args ];


(******************************************************************************)
(* Resource Management                                                        *)
(******************************************************************************)


(* Applying ResourceObject to a ResourceFunction will return its ResourceObject *)
ResourceFunction /:
  HoldPattern[ ResourceObject @ ResourceFunction[ ro_ResourceObject ] ] :=
    ro;


ResourceFunction /:
  HoldPattern @ ResourceRegister[ rf_ResourceFunction, args___ ] :=
    ResourceFunction @ ResourceRegister[ ResourceObject @ rf, args ];


(* Run the verification tests and return a test report *)
ResourceFunction /:
  HoldPattern @ TestReport[ rf_ResourceFunction, args___ ] :=
    ResourceFunctionInformation[ rf, "TestReport" ];


(* DeleteObject will clear kernel definitions as well as delete the resource object *)
ResourceFunction /:
  HoldPattern @ DeleteObject[ rf_ResourceFunction, args___ ] := (
    clearResourceFunction @ rf;
    DeleteObject[ ResourceObject @ rf, args ]
);


(******************************************************************************)


DefineResourceFunction // ClearAll;
DefineResourceFunction // Attributes = { };
DefineResourceFunction // Options = {
    "ContextPreserved" -> False,
    "FunctionLocation" -> Automatic,
    "PersistenceLocations" -> { "KernelSession" }
};

DefineResourceFunction[ args__ ] :=
  failOnMessage @ registerNewFunction @ withContext @ saveResourceFunction @ iDefineResourceFunction @ args;

DefineResourceFunction[ ] := iDefineResourceFunction[ ];



iDefineResourceFunction // ClearAll;
iDefineResourceFunction // Attributes = { HoldAllComplete };


iDefineResourceFunction[ f_Symbol? symbolQ, name_String, opts : OptionsPattern[ DefineResourceFunction ] ] :=
  ResourceFunction @ ResourceObject @ <|
      "Name" -> name,
      "ShortName" -> makeShortName @ name,
      "SymbolName" -> fullSymbolName @ f,
      "ResourceType" -> "Function",
      opts
  |>;


iDefineResourceFunction[ f_, name_String, opts : OptionsPattern[ DefineResourceFunction ] ] :=
  ResourceFunction @ ResourceObject @ <|
      "Name" -> name,
      "ShortName" -> makeShortName @ name,
      "Function" -> HoldComplete @ f,
      "ResourceType" -> "Function",
      opts
  |>;


iDefineResourceFunction[ info_Association, opts : OptionsPattern[ DefineResourceFunction ] ] :=
  ResourceFunction @ ResourceObject @ Join[ <| opts |>, info ];


iDefineResourceFunction[ HoldPattern @ ResourceObject[ info_Association, ___ ], opts : OptionsPattern[ DefineResourceFunction ] ] :=
  iDefineResourceFunction[ info, opts ];


iDefineResourceFunction[ f_Symbol? symbolQ, opts : OptionsPattern[ DefineResourceFunction ] ] :=
  With[ { name = SymbolName @ Unevaluated @ f },
      iDefineResourceFunction[ f, name, opts ]
  ];


iDefineResourceFunction[ f_, opts : OptionsPattern[ DefineResourceFunction ] ] :=
  With[ { name = untitledName[ ] },
      iDefineResourceFunction[ f, name, opts ]
  ];


iDefineResourceFunction[ ] :=
  newfunctionResourceDefinitionNotebook[ ];


iDefineResourceFunction[ ___ ] :=
  $failed;



$untitledResourceNameCounter = 1;
$untitledResourceNamePrefix  = "Untitled";

nextUntitledName[ ] :=
  Module[ { n },
      If[ ! And[ IntegerQ @ $untitledResourceNameCounter,
                 Positive @ $untitledResourceNameCounter
            ],
          $untitledResourceNameCounter = 1;
      ];
      n = ToString @ $untitledResourceNameCounter++;
      $untitledResourceNamePrefix <> n
  ];

usedNameQ[ name_String ] :=
  StringQ @
    ResourceSystemClient`Private`checkRegistryLookup[ "Function", name ];

untitledName[ ] :=
  Module[ { name },
      TimeConstrained[
          While @ usedNameQ[ name = nextUntitledName[ ] ],
          0.25
      ];
      name /; StringQ @ name
  ];

untitledName[ ] :=
  "Untitled";



registerNewFunction[ rf : HoldPattern @ ResourceFunction @ ResourceObject[ KeyValuePattern @ {
    "PersistenceLocations" -> pl_}, ___ ] ]:=registerNewFunction[pl,rf]

registerNewFunction[ rf_ResourceFunction ]:=registerNewFunction[OptionValue[DefineResourceFunction,"PersistenceLocations"],rf]
    
registerNewFunction[pl_String,rf_]:=registerNewFunction[{pl},rf]
registerNewFunction[l_List,rf_]:=(ResourceRegister[rf,l];rf)

registerNewFunction[expr_]:=expr
registerNewFunction[__,expr_]:=expr

(******************************************************************************)



ResourceSystemClient`resourceTypeFunction[$FunctionResourceTypes,args__]:=functionResourceFunction[args]

(* DO the magic, return ResourceFunction[ResourceObject] and adds the uuid to $loadedResourceFunctions *)
functionResourceFunction[id_String,info_Association]:=Block[{$applyingSubValuesQ},
	 $applyingSubValuesQ =
	  TrueQ@StackInhibit@
	    With[{stack = Take[Reverse@Stack[_], UpTo[$stackLength]]},
	    
	     MatchQ[stack, {___, HoldForm[ResourceFunction[_][___]]}]
	     ];
	 ReleaseHold[getResourceFunction[id, info]] /; $applyingSubValuesQ
 ]

(* retrieve resource function content *)
functionResourceFunction[id_String,info_Association, rest__]:={}

functionResourceFunction[___]:=$Failed

getResourceFunction[id_, info_]:=importResourceFunction[id,info,info["FunctionLocation"]]/;KeyExistsQ[info,"FunctionLocation"]
getResourceFunction[id_, info_]:=info["Function"]/;KeyExistsQ[info,"Function"]
getResourceFunction[id_, info_]:=Symbol@info["SymbolName"]/;KeyExistsQ[info,"SymbolName"]

getResourceFunction[___]:=$Failed

importResourceFunction[_,_,file_LocalObject]:=BinaryDeserialize[ByteArray[Import[file, "Binary"]]]
importResourceFunction[id_,info_,file_CloudObject]:=With[{
	res=ResourceSystemClient`Private`repositoryresourcedownload["Function",id,info]},
	If[KeyExistsQ[res,"FunctionLocation"],
		If[Head[res]["FunctionLocation"]=!=CloudObject,
			importResourceFunction[id,info,res["FunctionLocation"]]
			,
			Throw[$Failed]
		],
		Throw[$Failed]
	]	
]

importResourceFunction[file_String]:=Import[file]/;FileExistsQ[file]

importResourceFunction[___]:=$Failed (* TODO Message *)

End[] 

EndPackage[]

Protect /@ {
    "System`ResourceFunction",
    "System`DefineResourceFunction"
};
