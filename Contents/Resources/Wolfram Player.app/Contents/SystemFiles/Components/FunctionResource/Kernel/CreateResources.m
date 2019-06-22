(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["FunctionResource`"]

Begin["`Private`"] (* Begin Private Context *)


(******************************************************************************)


$defaultFunctionLocation = None;


(******************************************************************************)


$standardizeParameterFunctions // ClearAll;
$standardizeParameterFunctions := {
    standardizeFunctionLocation,
    standardizeShortName
};


standardizeParameters // ClearAll;
standardizeParameters[ args___ ] :=
  Catch[ (RightComposition @@ $standardizeParameterFunctions)[ args ],
         standardizeParameters
  ];


(******************************************************************************)


$persistenceLocations = Alternatives[
    "KernelSession",
    "FrontEndSession",
    "Notebook",
    "Local",
    "LocalShared",
    "Cloud",
    "Installation"
];


standardizeFunctionLocation // ClearAll;

standardizeFunctionLocation[ loc : $persistenceLocations ] :=
  PersistenceLocation @ loc;

standardizeFunctionLocation[ HoldPattern @ PersistenceLocation[ "NullLocation", ___ ] ] :=
  (
      Message[ System`ResourceFunction::invpl ];
      $failed
  );

standardizeFunctionLocation[ "Inline" ] := "Inline";

standardizeFunctionLocation[ file : _LocalObject|_CloudObject|_File|_String ] := file;

standardizeFunctionLocation[ info : KeyValuePattern[ "FunctionLocation" -> loc_ ] ] :=
  Insert[ info, "FunctionLocation" -> standardizeFunctionLocation @ loc, Key @ "FunctionLocation" ];

standardizeFunctionLocation[ (r : ResourceFunction|ResourceObject)[ info_, ___ ] ] :=
  r @ standardizeFunctionLocation @ info;

standardizeFunctionLocation[ arg___ ] := arg;



(******************************************************************************)



standardizeShortName // ClearAll;

standardizeShortName[ name_String ] :=
  makeShortName @ name;

standardizeShortName[ info : KeyValuePattern[ "Name" -> name_ ] /; ! KeyExistsQ[ info, "ShortName" ] ] :=
  With[ { shortName = standardizeShortName @ name },
      Append[ Insert[ info, "Name" -> shortName, Key @ "Name" ],
              "ShortName" -> shortName
      ]
  ];

standardizeShortName[ info : KeyValuePattern[ "ShortName" -> short_ ] ] :=
  With[ { shortName = standardizeShortName @ short },
      Insert[ Insert[ info, "ShortName" -> shortName, Key @ "ShortName" ],
              "Name" -> shortName,
              Key @ "Name"
      ]
  ];

standardizeShortName[ (r : ResourceFunction|ResourceObject)[ info_, ___ ] ] :=
  r @ standardizeShortName @ info;

standardizeShortName[ arg___ ] := arg;



(******************************************************************************)


(* when a resource is created, standardize the metadata in memory *)
ResourceSystemClient`Private`repositorystandardizeContentMetadata[ $FunctionResourceTypes, id_, info_ ] :=
  standardizeFunctionResource[ id,
                               Check[ standardizeParameters @ info,
                                      Throw[ $Failed, repositorystandardizeContentMetadata ]
                               ]
  ] ~Catch~ repositorystandardizeContentMetadata // withContext // failOnMessage // KeyDrop[ "Function" ];


resourceFunctionReadyQ // ClearAll;
resourceFunctionReadyQ[ info_Association ] := ResourceFunctionLoadedQ @ info && standardFunctionInfoQ @ info;
resourceFunctionReadyQ[ (ResourceFunction|ResourceObject)[ info_, ___ ] ] := resourceFunctionReadyQ @ info;


(* prevent invalid names *)
standardizeFunctionResource[ _, _? invalidResourceFunctionNameQ ] :=
  $failed;


(* everything is standardized and good to go *)
standardizeFunctionResource[ _, info_? resourceFunctionReadyQ ] :=
  ResourceFunctionInformation @ saveResourceFunction @ info;


(* insert given UUID if missing from info *)
standardizeFunctionResource[ id_String ? uuidQ, info_Association /; ! KeyExistsQ[ info, "UUID" ] ] :=
  standardizeFunctionResource[ id, Append[ info, "UUID" -> id ] ];


(* if there's no UUID, create one and insert it *)
standardizeFunctionResource[ _, info_Association /; ! KeyExistsQ[ info, "UUID" ] ] :=
  With[ { uuid = getUUID @ info },
      standardizeFunctionResource[ uuid, Append[ info, "UUID" -> uuid ] ]
  ];


(* make sure that "FunctionLocation" is set *)
standardizeFunctionResource[ uuid_, info_Association /; ! KeyExistsQ[ info, "FunctionLocation" ] ] :=
  standardizeFunctionResource[ uuid, Append[ info, "FunctionLocation" -> Automatic ] ];


(* convert "FunctionLocation" of Automatic to default value *)
standardizeFunctionResource[ uuid_, info : KeyValuePattern[ "FunctionLocation" -> Automatic ] ] :=
  standardizeFunctionResource[ uuid, Append[ info, "FunctionLocation" -> $defaultFunctionLocation ] ];


(* insert definition if "FunctionLocation" is set to "Inline" *)
standardizeFunctionResource[
    uuid_,
    info : KeyValuePattern[ "FunctionLocation" -> "Inline" ] /;
      ! KeyExistsQ[ info, "DefinitionData" ] && ResourceFunctionLoadedQ @ info
] :=
  standardizeFunctionResource[ uuid, insertDefinitionData @ info ];


(* drop definition data if function location is None *)
standardizeFunctionResource[
    uuid_,
    info : KeyValuePattern[ { "FunctionLocation" -> None, "DefinitionData" -> _ } ] /; ResourceFunctionLoadedQ @ info
] :=
  standardizeFunctionResource[ uuid, KeyDrop[ info, "DefinitionData" ] ];


(* load function into private context if info is complete but function hasn't been loaded yet *)
standardizeFunctionResource[ id_, info : KeyValuePattern @ {
    "UUID" -> uuid_,
    "SymbolName" -> name_String
} ] /; standardFunctionInfoQ @ info && ! TrueQ @ ResourceFunctionLoadedQ @ info :=

  Catch[ Module[ { inContextDef, newSymbol, newName, newInfo },

      inContextDef = getForkedDefinition @ info;

      newSymbol =
        Replace[ inContextDef,
                 {
                     def : Language`DefinitionList[ HoldForm[ sym_ ] -> _, ___ ] :> (
                         Quiet[ Language`ExtendedFullDefinition[ ] = def ];
                         ResourceFunctionLoadedQ[ uuid ] = True;
                         HoldComplete @ sym
                     )
                     ,
                     ___ :> Throw[ $failed, standardizeFunctionResource ]
                 }
        ];

      newName = fullSymbolName @@ newSymbol;

      newInfo = Join[ info,
                      <|
                          "SymbolName" -> newName,
                          "Function" -> ToExpression[ newName, InputForm, HoldComplete ]
                      |>
                ];

      If[ KeySort @ info === KeySort @ newInfo,
          info,
          standardizeFunctionResource[ id, newInfo ]
      ]

  ], standardizeFunctionResource ];


(* convert "Symbol" value into a string and held function *)
standardizeFunctionResource[ id_, info : KeyValuePattern[ (Rule | RuleDelayed)[ "Symbol", symbol_Symbol? symbolQ ] ] ] :=
  With[ { symName = fullSymbolName @ symbol },
      standardizeFunctionResource[ id,
                                   KeyDrop[ Join[ info,
                                                  <|
                                                      "SymbolName" -> symName,
                                                      "Function" -> HoldComplete @ symbol
                                                  |>
                                            ],
                                            "Symbol"
                                   ]
      ]
  ];


(* if a symbol has been given for "Function", convert to correct form for "Function" and include symbol string *)
standardizeFunctionResource[ id_, info : KeyValuePattern[ (Rule | RuleDelayed)[ "Function", symbol_Symbol? symbolQ ] ] ] :=
  With[ { symName = fullSymbolName @ symbol },
      standardizeFunctionResource[ id,
                                   KeyDrop[ Join[ info,
                                                  <|
                                                      "SymbolName" -> symName,
                                                      "Function" -> HoldComplete @ symbol
                                                  |>
                                            ],
                                            "Symbol"
                                   ]
      ]
  ];


(* if some other expression has been given for "Function", create a symbol for it *)
standardizeFunctionResource[ id_, info : KeyValuePattern[ "Function" -> HoldComplete[ f : Except[ _Symbol? symbolQ ] ] ] ] :=
  With[ { symbol = createSymbolName[ id, info ] },
      SetDelayed @@ HoldComplete[ symbol, f ];
      standardizeFunctionResource[ id, Append[ info, "Function" :> symbol ] ]
  ];


(* put "Function" into HoldComplete form *)
standardizeFunctionResource[ id_, info : KeyValuePattern[ "Function" -> f : Except[ _HoldComplete ] ] ] :=
  standardizeFunctionResource[ id, Append[ info, "Function" -> HoldComplete @ f ] ];


(* if given a symbol string, but missing the function, create it from the string *)
standardizeFunctionResource[ id_, info : KeyValuePattern[ "SymbolName" -> name_ ] /; ! KeyExistsQ[ info, "Function" ] ] :=
  standardizeFunctionResource[ id, Append[ info, "Function" -> ToExpression[ name, InputForm, HoldComplete ] ] ];


(* make sure ShortName and Name are the same (for backwards compatibility) *)
standardizeFunctionResource[ id_, info: KeyValuePattern @ {
    "Name" -> name_,
    "ShortName" -> shortName_
} ] :=
  With[ { diffQ = name =!= shortName },
      standardizeFunctionResource[
          id,
          Append[ info, "Name" -> shortName ]
      ] /; diffQ
  ];


(* catch all to failure *)
standardizeFunctionResource[ _, info_ ] := (
  Message[ ResourceFunction::invld, info ];
  $failed
);

standardizeFunctionResource[ ___ ] := $failed;



standardFunctionInfoQ // ClearAll;

standardFunctionInfoQ[ KeyValuePattern @ {
    "Name" -> _String,
    "ResourceType" -> $FunctionResourceTypes,
    "SymbolName" -> _String,
    "UUID" -> _String,
    "FunctionLocation" -> Except[ Automatic ]
} ] := True;

standardFunctionInfoQ[ (ResourceFunction|ResourceObject)[ info_, ___ ] ] := standardFunctionInfoQ @ info;
standardFunctionInfoQ[ _Association ] := False;
standardFunctionInfoQ[ ___ ] := False;






(* how to cache user-created resources locally on disk *)
ResourceSystemClient`Private`repositorysaveresourceobject[ $FunctionResourceTypes, info_ ] :=
  cacheresourceinfo @
    KeyDrop[ ResourceFunctionInformation @
               saveResourceFunction[ info, definitionDataSaveLocation[ "Local", info ] ],
             "DefinitionData"
    ];


ResourceSystemClient`Private`resourcelocalDeploy[
    ro: ResourceObject[ KeyValuePattern[ "ResourceType" -> "Function" ], ___ ],
    location_
] :=
  Module[ { saved, res },
      saved = ResourceSystemClient`Private`saveResourceObject @ ro;
      If[ ! FailureQ @ saved
          ,
          ResourceRegister[ saved, "Local" ];
          res = Switch[ location,
                None, Null,
                _File, Put[ saved, location[[1]] ],
                _String | _LocalObject, Put[ saved, location ],
                _, (Message[ ResourceRegister::invloc, location ]; $Failed)
            ];

          If[ ! FailureQ @ res,
              If[ location =!= None, location ],
              $Failed
          ]
      ]
  ];





(* how to deploy user-created resources to cloud *)
ResourceSystemClient`Private`repositoryclouddeployResourceContent[
    $FunctionResourceTypes,
    _,
    info: KeyValuePattern[ "UUID" -> id_ ],
    args___
] :=
  Module[ { saved, savedObj, opts, permissions, rdir, loc, add },
      saved = saveResourceFunction[ info, definitionDataSaveLocation[ "Cloud", info ] ];
      savedObj = resourceFunctionStaticProperty[ #, "FunctionLocation" ] & @ saved;
      opts = FilterRules[ Cases[ { args }, _Rule|_RuleDelayed ], Options @ CloudDeploy ];
      permissions = OptionValue[ CloudDeploy, { opts }, Permissions ];
      SetPermissions[ savedObj, Replace[ permissions, Automatic :> $Permissions ] ];
      rdir = ResourceSystemClient`Private`resourceDirectory @ id;
      loc = CloudObject @ ResourceSystemClient`Private`cloudpath @ rdir;
      add = <| "FunctionLocation" -> savedObj, "ResourceLocations" -> { loc } |>;
      KeyDrop[ Join[ info, add ], "DefinitionData" ]
  ];


End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];
