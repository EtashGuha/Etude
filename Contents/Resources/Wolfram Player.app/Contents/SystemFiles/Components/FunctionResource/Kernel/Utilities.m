(* Wolfram Language Package *)

BeginPackage[ "FunctionResource`" ];


$ResourceFunctionRootContext;
$ResourceFunctionTempContext;
$FunctionResourceTypes;
ResourceFunctionLoadedQ;
$ResourceFunctionLoadStatus;


Begin[ "`Private`" ];


(******************************************************************************)
(* Config                                                                     *)
(******************************************************************************)


$debug = False;


$FunctionResourceTypes = "Function" | "FunctionResource";

$ResourceFunctionRootContext = "FunctionRepository`";
$ResourceFunctionTempContext = "FunctionRepositoryTemporary`";

ResourceSystemClient`Private`usableresourceinfoKeys[ $FunctionResourceTypes ] =
  {
      "SymbolName",
      "Function",
      "FunctionLocation",
      "ContextPreserved",
      "DefinitionData",
      "VerificationTests",
      "ShortName",
      "DocumentationNotebook"
  };

ResourceSystemClient`Private`addToResourceTypes["Function"];

ResourceSystemClient`Private`sortingProperties["Function"]={"Categories"};


(******************************************************************************)

$packageRoot = DirectoryName @ $InputFileName;


$thisPacletVersion :=
  $thisPacletVersion = Quiet @
    Replace[ Get @ FileNameJoin @ { DirectoryName @ $packageRoot, "PacletInfo.m" },
             {
                 _[ ___, Version -> ver_, ___ ] :> ver,
                 ___ :> Missing[ "NotAvailable" ]
             }
    ];

(******************************************************************************)


System`ResourceFunction::rfunknown = "An unknown error occurred.";

$failed :=
  StackInhibit @ If[ TrueQ @ $debug
                     ,
                     Message[ System`ResourceFunction::rfunknown ];
                     $resourceFunctionStack = Stack[ _ ];
                     If[ TrueQ @ $abortOnFailure, Abort[ ], $Failed ]
                     ,
                     $Failed
                 ];


failWithExpr[ expr_ ] :=
  (
      $resourceFunctionLastFailed = HoldForm @ expr;
      $failed
  );


(******************************************************************************)


msgThrow // ClearAll;
msgThrow // Attributes = { HoldAllComplete };

msgThrow[ default_, listen_, Hold[ m : Message[ msg_, ___ ], False ] ] /;
  MatchQ[ Unevaluated @ msg, listen ] := (
    $failedMessage = HoldComplete @ msg;
    Throw[ Unevaluated[ m; default ],
           failOnMessage
    ]
);

msgThrow[ default_, _, Hold[ msg_, True ] ] := (
    $failedMessage = HoldComplete @ msg;
    Throw[ default, failOnMessage ]
);

$defaultQuiet := HoldComplete[
    CloudConnect::clver,
    SessionSubmit::timc,
    ResourceObject::updav,
    ResourceObject::updavb
];


failOnMessage // ClearAll;
failOnMessage // Attributes = { HoldAllComplete };

failOnMessage[ eval_ ] :=
  failOnMessage[ eval, $failed ];

failOnMessage[ eval_, default_ ] :=
  Replace[ $defaultQuiet,
      {
          HoldComplete[ quiet___ ] :> failOnMessage[ eval, default, { quiet } ],
          ___ :> failOnMessage[ eval, default, None ]
      }
  ];

failOnMessage[ eval_, default_, quiet_ ] :=
  failOnMessage[ eval, default, quiet, None ];

failOnMessage[ eval_, default_, quiet_, All ] :=
  failOnMessage[ eval, default, quiet, _ ];

failOnMessage[ eval_, default_, quiet_, { listen___ } ] :=
  failOnMessage[ eval, default, quiet, Alternatives @ listen ];

failOnMessage[ eval_, default_, quiet_, listen_ ] :=
  If[ FreeQ[ Internal`Handlers[ "Message" ], _msgThrow ],
      Internal`HandlerBlock[ { "Message", msgThrow[ default, listen, ## ] & },
                             Quiet[ eval, quiet ]
      ],
      eval
  ] ~Catch~ failOnMessage;


failOnMessage[ ___ ] := $failed;



(******************************************************************************)


$ResourceFunctionLoadStatus // ClearAll;
$ResourceFunctionLoadStatus :=
KeySort @ Association @
  Cases[ DownValues @ ResourceFunctionLoadedQ,
         HoldPattern[ Verbatim[ HoldPattern ][ ResourceFunctionLoadedQ[ uuid_String ] ] :> q_ ] :> uuid -> q
  ];



(******************************************************************************)
(* Definition Utilities                                                       *)
(******************************************************************************)


contextQ // ClearAll;
contextQ[ str_String ] := StringMatchQ[ str, __ ~~ "`" ];
contextQ[ ___ ] := False;


(******************************************************************************)


subContexts // ClearAll;

subContexts[ context_String ? contextQ ] :=
  SortBy[ Contexts[ context <> "*" ],
          Length @ StringSplit[ #, "`" ] &
  ];


(******************************************************************************)


$TemporaryContext = "FunctionResource`Temp`";


noContext // ClearAll;
noContext // Attributes = { HoldAllComplete };
noContext[ arg___ ] := arg;


withContext // ClearAll;
withContext // Attributes = { HoldAllComplete };

withContext[ eval_ ] :=

  Module[ { context, nc, result },

      context = $TemporaryContext;

      nc =
        With[ { oc = $Context, ocp = $ContextPath },
            Function[ Null,
                Block[ { $Context = oc, $ContextPath = ocp },
                    ##
                ],
                { HoldAllComplete }
            ]
        ];

      result = Quiet[ Block[ { $Context = context, $ContextPath = { context, "System`" }, withContext = # &, noContext = nc },
                             eval
                      ],
                      General::shdw
               ];

      Remove /@ (StringJoin[ #, "*" ] & /@ subContexts @ context);

      result
  ];


(******************************************************************************)


$excludedContexts =
  Cases[ Lookup[ Options @ Language`ExtendedFullDefinition,
                 "ExcludedContexts",
                 { }
         ],
         ctx_String ? StringQ :> ctx <> "`"
  ] ~Union~ { "GeneralUtilities`", "FunctionResource`" };


excludedContextQ :=
  StringMatchQ[ Alternatives @@ $excludedContexts ~~ ___ ];


inBaseContextQ // ClearAll;
inBaseContextQ // Attributes = { };

inBaseContextQ[ ctx_String ? contextQ ] :=
  TrueQ @ StringMatchQ[ ctx, $ResourceFunctionRootContext ~~ __ ];

inBaseContextQ[ name_String ] :=
  inBaseContextQ @ safeContext @ name;

inBaseContextQ[ KeyValuePattern[ "ContextPreserved" -> True ] ] :=
  True;

inBaseContextQ[ KeyValuePattern @ { "SymbolName" -> name_ } ] :=
  inBaseContextQ @ name;

inBaseContextQ[ ___ ] :=
  False;



getBaseContext // ClearAll;

getBaseContext[ uuid_String ? uuidQ ] :=
  StringJoin[ $ResourceFunctionRootContext,
              "$", StringDelete[ uuid, "-" ], "`"
  ];

getBaseContext[ info : KeyValuePattern @ { "UUID" -> uuid_String ? uuidQ } ] :=
  If[ inBaseContextQ @ info,
      "",
      getBaseContext @ uuid
  ];

getBaseContext[ HoldPattern[ context_String /; StringMatchQ[ context, $ResourceFunctionRootContext ~~ __ ~~ "`" ] ] ] :=
  context;

getBaseContext[ name_String ] :=
  getBaseContext @ safeContext @ name;

getBaseContext[ sym_Symbol? symbolQ ] :=
  getBaseContext @ safeContext @ sym;

getBaseContext[ (Hold | HoldComplete | HoldForm | HoldPattern | Unevaluated)[ symbol_Symbol? symbolQ ] ] :=
  getBaseContext @ safeContext @ symbol;

getBaseContext[ context_String /; StringMatchQ[ context, __ ~~ "`" ] ] :=
  getBaseContext @ CreateUUID[ ];

getBaseContext[ ___ ] :=
  $failed;



emptyDefinitionsQ[ Language`DefinitionList[ ] ] :=
  True;

emptyDefinitionsQ[ Language`DefinitionList[ _ -> { } ] ] :=
  True;

emptyDefinitionsQ[ Language`DefinitionList[ HoldForm[ $ExpressionPlaceholder ] -> { OwnValues -> { _ :> s_Symbol? symbolQ } } ] ] :=
  TrueQ @ inBaseContextQ @ fullSymbolName @ s;

emptyDefinitionsQ[ Language`DefinitionList[ _ -> { __ }, ___ ] ] :=
  False;

emptyDefinitionsQ[ ___ ] := $failed;


getForkedDefinition // ClearAll;
getForkedDefinition // Attributes = { HoldAllComplete };


getForkedDefinition[ KeyValuePattern @ { "DefinitionData" -> data_ByteArray } ] :=
  With[ { def = Lookup[ BinaryDeserialize @ data, "Definition" ] },
      def /; ! emptyDefinitionsQ @ def
  ];


(* check to see if the definition is already created, so we don't overwrite it *)
getForkedDefinition[ info: KeyValuePattern @ { "FunctionLocation" -> _ } ] :=
  Module[ { importTest },
      importTest = Quiet @ Lookup[ BinaryDeserialize @ importDefinitionData @ info, "Definition" ];
      importTest /; MatchQ[ importTest, Language`DefinitionList[ HoldForm[ s_ ] -> _ ] /; inBaseContextQ @ fullSymbolName @ s ]
  ];


getForkedDefinition[ symbol_Symbol? symbolQ ] :=
  minimalFullDefinition @ symbol;


getForkedDefinition[ info : KeyValuePattern[ "SymbolName" -> name_String ] ? inBaseContextQ ] :=
  With[ { def = Replace[ minimalFullDefinition @ name,
                         (* if undefined, load resource function and try again *)
                         _? emptyDefinitionsQ :> (loadResourceFunction @ info;
                                                  minimalFullDefinition @ name)
                ] },
      (* return only if definition found *)
      def /; ! emptyDefinitionsQ @ def
  ];


getForkedDefinition[ info_Association /; ! AssociationQ @ Unevaluated @ info ] :=
  With[ { init = info },
      getForkedDefinition @ init /; AssociationQ @ init
  ];


getForkedDefinition[ f_, baseContext : Except[ _String ] ] :=
  With[ { bc = baseContext },
      getForkedDefinition[ f, bc ] /; StringQ @ bc
  ];


getForkedDefinition[ name_String, baseContext_String ] :=
  Module[ { localDef, deps, withInits, names, rules },
      localDef = minimalFullDefinition @ name;
      deps = dependentSymbols @@ { localDef };
      names = Select[ fullSymbolName /@ deps, Not @* inBaseContextQ ];
      rules = makeSymbolRule[ baseContext ] /@ names;
      (* TODO: repack expressions to restore atomic state where needed *)
      With[ { forked = unpackDefinition[ localDef ] //. rules },
          cleanupTemporarySymbols[ ];
          forked
      ]
  ];


getForkedDefinition[ expr : Except[ _String ], baseContext_String ] :=
  Block[ { $PlaceholderSymbol },
      $PlaceholderSymbol := expr;
      With[ { name = fullSymbolName @ $PlaceholderSymbol },
          getForkedDefinition[ name, baseContext ]
      ]
  ];


getForkedDefinition[ info : KeyValuePattern @ { "SymbolName" -> name_ } ] :=
  With[ { baseContext = getBaseContext @ info },
      setUsageString @ info;
      getForkedDefinition[ name, baseContext ]
  ];


getForkedDefinition[ HoldPattern @ ResourceObject[ info_, ___ ] ] :=
  getForkedDefinition @ info;


getForkedDefinition[ HoldPattern @ ResourceFunction @ ResourceObject[ info_, ___ ] ] :=
  getForkedDefinition @ info;


getForkedDefinition[ ___ ] := (
    Message[ System`ResourceFunction::fdefn ];
    $failed
);


(******************************************************************************)


setUsageString[ info: KeyValuePattern[ "SymbolName" -> name_ ] ] :=
  Module[ { string, heldSym },
      string = Quiet @ usageString @ info;
      If[ StringQ @ string,
          heldSym = ToExpression[ name, InputForm, HoldComplete ];
          Replace[ heldSym,
                   HoldComplete[ s_Symbol? symbolQ ] :> (s::usage = string)
          ]
      ]
  ];


(******************************************************************************)


getUUID // ClearAll;
getUUID[ uuid_String ? uuidQ ] := uuid;
getUUID[ KeyValuePattern @ { "UUID" -> uuid_ } ] := getUUID @ uuid;
getUUID[ HoldPattern @ ResourceObject[ info_, ___ ] ] := getUUID @ info;
getUUID[ HoldPattern @ ResourceFunction[ ro_ ] ] := getUUID @ ro;
getUUID[ s_Symbol? symbolQ ] := getUUID @ fullSymbolName @ s;
getUUID[ a_Association /; ! AssociationQ @ a ] := With[ { a$ = a }, getUUID @ a$ /; AssociationQ @ a$ ];

getUUID[ HoldPattern[ s_String /; StringMatchQ[ s, $ResourceFunctionRootContext ~~ __ ~~ "`" ~~ ___ ] ] ] :=
  With[ { uuid = StringReplace[ s,
                 StringExpression[ $ResourceFunctionRootContext,
                                   "$",
                                   c : Longest[ Except[ "`" ] .. ],
                                   "`",
                                   ___
                 ] :> StringInsert[ c, "-", { 9, 13, 17, 21 } ]
  ] },
      uuid /; uuidQ @ uuid
  ];

getUUID[ ___ ] := CreateUUID[ ];


(******************************************************************************)


serializeWithContext // ClearAll;
serializeWithContext // Attributes = { };

serializeWithContext[ expr_, args___ ] :=
  withContext @ BinarySerialize[ Unevaluated @ expr, args ];


serializeWithDefinitions // ClearAll;
serializeWithDefinitions // Attributes = { };

serializeWithDefinitions[ info : KeyValuePattern[ "Expression" -> HoldComplete[ expr_ ] ] ] :=
  Module[ { definition, uuid },

      definition = minimalFullDefinition @ expr;
      uuid = getUUID @ info;

      serializeWithContext[ serializedMetadata[ expr, definition, uuid, info ],
                            PerformanceGoal -> "Size"
      ]
  ];

serializeWithDefinitions[ expr_, uuid_ ] :=
  serializeWithDefinitions @ <| "Expression" -> HoldComplete @ expr, "UUID" -> uuid |>;

serializeWithDefinitions[ expr_ ] :=
  With[ { uuid = getUUID @ Unevaluated @ expr },
      serializeWithDefinitions @ <| "UUID" -> uuid, "Expression" -> HoldComplete @ expr |>
  ];



serializedMetadata // Attributes = { HoldFirst };

serializedMetadata[ expr_, definition_, uuid_, info_ ] :=
  <|
      "UUID"                -> uuid,
      "Expression"          -> HoldComplete @ expr,
      "Definition"          -> definition,
      "Timestamp"           -> AbsoluteTime[ TimeZone -> 0 ],
      "TemplateVersion"     -> templateVersion @ uuid,
      "PacletVersion"       -> $thisPacletVersion,
      "SystemInformation"   -> GeneralUtilities`ToAssociations @ SystemInformation[ "Small" ]
  |>;


templateVersion[ uuid_ ] :=
  Replace[ FunctionResource`DefinitionNotebook`DefinitionTemplateVersion @ uuid,
           Except[ _String?StringQ | _Missing ] :> Missing[ "NotAvailable" ]
  ];


deserializeWithDefinitions // ClearAll;
deserializeWithDefinitions // Attributes = { };

deserializeWithDefinitions[ bytes_ByteArray ] :=
  Replace[ BinaryDeserialize @ bytes,
           KeyValuePattern @ { "UUID" -> uuid_, "Expression" -> HoldComplete[ expr_ ], "Definition" -> def_ } :> (
               Quiet[ Language`ExtendedFullDefinition[ ] = def ];
               ResourceFunctionLoadedQ[ uuid ] = True;
               expr
           )
  ];

deserializeWithDefinitions[ bytes : { ___Integer } ] :=
  deserializeWithDefinitions @ ByteArray @ bytes;

deserializeWithDefinitions[ loc : _LocalObject | _CloudObject ] :=
  deserializeWithDefinitions @ Import[ loc, "Binary" ];


(******************************************************************************)


$symbolInitConfig = {
    <| "Head" -> Association,   "InitializedQ" -> AssociationQ          |>,
    <| "Head" -> ByteArray,     "InitializedQ" -> ByteArrayQ            |>,
    <| "Head" -> Graph,         "InitializedQ" -> GraphQ                |>,
    <| "Head" -> Image,         "InitializedQ" -> ImageQ                |>,
    <| "Head" -> RawArray,      "InitializedQ" -> Developer`RawArrayQ   |>,
    <| "Head" -> Symbol,        "InitializedQ" -> symbolQ               |>,
    <| "Head" -> Dispatch,      "InitializedQ" -> DispatchQ             |>
};



$initialized   // Attributes = { HoldAllComplete };
$uninitialized // Attributes = { HoldAllComplete };



reinitializeExpression // ClearAll;

reinitializeExpression[ $initialized[ head_, args___ ] ] /; FreeQ[ HoldComplete @ args, $initialized ] :=

  Module[ { symbols },

      symbols = Flatten[
          HoldComplete @@ DeleteDuplicates @
            Cases[ HoldComplete @ args,
                   s_? unlockedSymbolQ :> HoldComplete @ s,
                   Infinity,
                   Heads -> True
            ]
      ];

      Replace[ symbols,
               HoldComplete[ syms___ ] :>
                 Block[ { syms },
                     ReleaseHold[ HoldComplete @ head @ args /. Sequence -> $sequence ] /. $sequence -> Sequence
                 ]
      ]
  ];



makeRules // ClearAll;

makeRules[ KeyValuePattern @ { "Head" -> head_, "InitializedQ" -> validQ_ } ] :=
  Sequence[
      expr : HoldPattern @ head[ args___ ] /; validQ @ Unevaluated @ expr :>
        RuleCondition @ $initialized[ head, args ]
      ,
      expr : HoldPattern @ head[ args___ ] /; ! validQ @ Unevaluated @ expr :>
        RuleCondition @ $uninitialized[ head, args ]
  ];


$extraRules // ClearAll;

$extraRules :=
  $extraRules = {
      b_ByteArray /; ByteArrayQ @ Unevaluated @ b :>
        With[ { n = Normal @ b },
              $initialized[ ByteArray, n ] /; True
        ],
      d_Dispatch /; DispatchQ @ Unevaluated @ d :>
        With[ { n = Normal @ d },
            $initialized[ Dispatch, n ] /; True
        ]
  };



$initializationRules // ClearAll;

$initializationRules :=
  $initializationRules =
    Join[ makeRules /@ $symbolInitConfig, $extraRules ];



markInitialization // ClearAll;

markInitialization[ expr_ ] :=
  expr //. $initializationRules;


(******************************************************************************)


$failureContext = "$Failed`";


(******************************************************************************)



unpackDefinition =
  ToExpression[ ToString @ FullForm @ #, InputForm ] &;



dependentSymbols // ClearAll;
dependentSymbols // Attributes = { HoldAllComplete };


dependentSymbols[ str_String ] :=
  ToExpression[ str, InputForm, dependentSymbols ];


dependentSymbols[ expr_ ] :=
  Select[ DeleteDuplicates @ Cases[ unpackDefinition @ HoldComplete @ expr,
                                    (s_Symbol ? symbolQ) :> HoldComplete @ s,
                                    Infinity,
                                    Heads -> True
                             ],
          ! TrueQ @ excludedContextQ @ safeContext @ # &
  ];


dependentSymbols[ symbol_ ? symbolQ, All ] :=
  Cases[ minimalFullDefinition @ symbol,
         HoldPattern[ HoldForm[ s_ ] -> _ ] :> HoldComplete @ s
  ];


dependentSymbols[ name_String ? NameQ, All ] :=
  ToExpression[ name,
                InputForm,
                Function[ symbol, dependentSymbols[ symbol, All ], HoldAllComplete ]
  ];


dependentSymbols[ expr_, All ] :=
  Block[ { $ExpressionPlaceholder },
      $ExpressionPlaceholder := expr;
      DeleteCases[ dependentSymbols[ $ExpressionPlaceholder, All ],
                   HoldComplete @ $ExpressionPlaceholder
      ]
  ];



(******************************************************************************)


readProtectedQ // ClearAll;
readProtectedQ // Attributes = { HoldAllComplete };
readProtectedQ[ symbol_Symbol? symbolQ ] := readProtectedQ @@ { fullSymbolName @ symbol };
readProtectedQ[ name_String ] := MemberQ[ Attributes @ name, ReadProtected ];


(******************************************************************************)


$emptyDefPattern = Alternatives @@
  Block[ { x },
         Append[ Language`ExtendedDefinition[ x ][[ 1, 2 ]],
                 Attributes -> { Temporary }
         ]
  ];


minimalDefinition // ClearAll;
minimalDefinition // Attributes = { HoldAllComplete };


minimalDefinition[ HoldComplete[ symbol_Symbol? symbolQ ] ] :=
  minimalDefinition @ symbol;

(* need to clear read protected attribute but make sure it still shows up in the definition *)
minimalDefinition[ ( symbol_Symbol? symbolQ)? readProtectedQ ] :=
  With[ { attributes = Attributes @ symbol },
      Internal`InheritedBlock[ { symbol },
          ClearAttributes[ symbol, { Protected, ReadProtected } ];
          Insert[ DeleteCases[ minimalDefinition @ symbol,
                               HoldPattern[ Attributes -> _ ],
                               { 3 }
                  ],
                  Attributes -> attributes,
                  { 1, 2, -1 }
          ]
      ]
  ];

minimalDefinition[ symbol_Symbol? symbolQ ] :=
  Replace[ DeleteCases[ Language`ExtendedDefinition @ symbol,
                        $emptyDefPattern,
                        3
           ],
           Language`DefinitionList[ _ -> { } ] -> Language`DefinitionList[ ]
  ];


minimalDefinition[ name_String ] :=
  ToExpression[ name, InputForm, minimalDefinition ];

minimalDefinition[ ___ ] :=
  $failed;



minimalFullDefinition // ClearAll;
minimalFullDefinition // Attributes = { HoldAllComplete };

minimalFullDefinition[ symbol_Symbol? symbolQ, maxCount_ : Infinity ] :=

  Catch[ Module[ { $count, $definitions, unseenQ, getNext },

      $count = 0;
      $definitions = <| |>;
      unseenQ[ ___ ] := True;

      getNext[ HoldComplete[ sym_ ] ? unseenQ ] :=
        With[ { def = minimalDefinition @ sym },
            If[ ++$count > maxCount, Throw[ $failed, minimalFullDefinition ] ];
            unseenQ[ HoldComplete @ sym ] = False;
            $definitions[ HoldComplete @ sym ] = def;

            getNext /@ dependentSymbols @ def;
        ];

      getNext @ HoldComplete @ symbol;

      Replace[ Flatten[ Language`DefinitionList @@ Values @ $definitions ],
               Language`DefinitionList[ ] :>
                 Language`DefinitionList[ HoldForm[ symbol ] -> { } ]
      ]

  ], minimalFullDefinition ];


minimalFullDefinition[ name_String ? NameQ, maxCount_ : Infinity ] :=
  ToExpression[ name,
                InputForm,
                Function[ symbol, minimalFullDefinition[ symbol, maxCount ], HoldAllComplete ]
  ];


minimalFullDefinition[ expr_, maxCount_ : Infinity ] :=
  Block[ { $ExpressionPlaceholder },
      $ExpressionPlaceholder := expr;
      minimalFullDefinition[ $ExpressionPlaceholder, maxCount ]
  ];


minimalFullDefinition[ ___ ] :=
  $failed;


(******************************************************************************)


replaceContext // Attributes = { };
replaceContext // Options =    { };


(* No replacement for same context *)
replaceContext[ expression_, ctx_ -> ctx_ ] :=
  expression;


replaceContext[ expression_, oldContext_String -> newContext_String ] :=

  Module[ { heldSymbol, withHeldSymbols },

      heldSymbol // Attributes = { HoldAllComplete };

      withHeldSymbols = HoldComplete @ expression //.
        s_Symbol? symbolQ /; safeContext @ s === oldContext :>
          withHolding[
              {
                  newSymString = newContext <> SymbolName @ Unevaluated @ s,
                  newSymbol = ToExpression[ newSymString,
                                            StandardForm,
                                            heldSymbol
                              ]
              },
              newSymbol
          ];

      ReleaseHold[ withHeldSymbols //. heldSymbol[ s_ ] :> s ]
  ];


(******************************************************************************)


fullSymbolName // ClearAll;
fullSymbolName // Attributes = { HoldAllComplete };

fullSymbolName[ symbol_Symbol? symbolQ ] :=
  StringReplace[ ToString[ HoldComplete @ symbol, InputForm ],
                 StringExpression[ StartOfString,
                                   "System`HoldComplete[" | "HoldComplete[",
                                   name__,
                                   "]",
                                   EndOfString
                 ] :> Context @ name <> Last @ StringSplit[ name, "`" ]
  ];

fullSymbolName[ (Hold | HoldComplete | HoldForm | HoldPattern | Unevaluated)[ symbol_Symbol? symbolQ ] ] :=
  fullSymbolName @ symbol;

fullSymbolName[ name_String ? NameQ ] :=
  ToExpression[ name, InputForm, fullSymbolName ];

fullSymbolName[ ___ ] := $failed;


makeSymbolRule // ClearAll;

makeSymbolRule[ _ ][ _String? linkSymbolQ ] :=
  Sequence[ ];

makeSymbolRule[ baseContext_String ][ name_String ] :=
  makeNewRule[ name, makeNewSymbol[ baseContext, name ] ];

makeNewSymbol[ base_, name_ ] :=
  ToExpression[ StringDelete[ base <> name, $ResourceFunctionTempContext ],
                InputForm,
                HoldComplete
  ];

makeNewRule[ name_, HoldComplete[ new_ ] ] :=
  With[ { old = ToExpression[ name, InputForm, HoldPattern ] },
      old :> new
  ];

(******************************************************************************)



linkSymbolQ // Attributes = { HoldAllComplete };

linkSymbolQ[ name_String ] :=
  ToExpression[ name, InputForm, linkSymbolQ ];

linkSymbolQ[ sym_Symbol? symbolQ ] :=
  linkFunctionQ @ sym || linkObjectQ @ sym;

linkSymbolQ[ ___ ] :=
  False;



linkObjectQ // Attributes = { HoldAllComplete };

linkObjectQ[ name_String ] :=
  ToExpression[ name, InputForm, linkObjectQ ];

linkObjectQ[ sym_Symbol? symbolQ ] :=
  MemberQ[ OwnValues @ sym,
      _ :> _Symbol? (contextStartsQ[ "JLink`Objects`"|"NETLink`Objects`" ])
  ];

linkObjectQ[ ___ ] :=
  False;



linkFunctionQ // Attributes = { HoldAllComplete };

linkFunctionQ[ name_String ] :=
  ToExpression[ name, InputForm, linkFunctionQ ];

linkFunctionQ[ sym_Symbol? symbolQ ] :=
  MemberQ[ DownValues @ sym,
           _ :> Alternatives[ _JLink`CallJava`Private`callJava,
                              _NETLink`CallNET`Private`netStaticMethod
                ]
  ];

linkFunctionQ[ ___ ] :=
  False;


(******************************************************************************)


contextStartsQ // Attributes = { HoldAllComplete };

contextStartsQ[ sym_Symbol? symbolQ, ctx_ ] :=
  TrueQ @ StringStartsQ[ Context @ Unevaluated @ sym, ctx ];

contextStartsQ[ name_String, ctx_ ] :=
  ToExpression[ name, InputForm, contextStartsQ @ ctx ];

contextStartsQ[ ctx_ ] :=
  Function[ sym, contextStartsQ[ sym, ctx ], { HoldAllComplete } ];

contextStartsQ[ ___ ] :=
  False;


(******************************************************************************)


safeContext // Attributes = { HoldAllComplete };
safeContext // Options    = { };


safeContext[ symbol_Symbol? symbolQ ] :=
  With[ { name = fullSymbolName @ symbol },
      Context @ name
  ];


safeContext[ (Hold | HoldComplete | HoldForm | HoldPattern | Unevaluated)[ symbol_Symbol? symbolQ ] ] :=
  safeContext @ symbol;


safeContext[ string_String ? NameQ ] :=
  ToExpression[ string, InputForm, safeContext ];


safeContext[ string_String /; StringContainsQ[ string, "\\[" ] ] :=
  Replace[ ToExpression[ StringJoin[ "\"", string, "\"" ],
                         InputForm,
                         HoldComplete
           ],
           HoldComplete[ s_ ] :> safeContext @ s
  ];


safeContext[ string_String /; StringContainsQ[ string, "`" ] ] :=
  StringRiffle[ Most @ StringSplit[ string, "`" ], "`" ] <> "`";


safeContext[ _String ] :=
  $Context;


safeContext[ a___ ] :=
  (
      Message[ safeContext::ssle,
               HoldForm @ safeContext @ a,
               1
      ];

      $failureContext
  );


(******************************************************************************)


safeOptions // Attributes = { HoldAllComplete };

safeOptions[ name_String? NameQ ] :=
  ToExpression[ name, InputForm, safeOptions ];

safeOptions[ symbol_ ] :=
  Options @ Unevaluated @ symbol;

safeOptions[ ___ ] :=
  { };


(******************************************************************************)


symbolQ // Attributes = { HoldAllComplete };
symbolQ // Options    = { };


symbolQ[ s_Symbol ] := Depth @ HoldComplete @ s === 2;
symbolQ[ ___ ] := False;


(******************************************************************************)


tempHold // Attributes = { HoldAllComplete };
tempHold // Options    = { };


tempHold[ th_tempHold ] := th;


(******************************************************************************)


(* Trott-Strzebonski in-place evaluation
   http://library.wolfram.com/infocenter/Conferences/377/ *)
trEval // Attributes = { };
trEval // Options    = { };


trEval /: HoldPattern[ swap_ :> trEval @ eval_ ] :=
  swap :> With[ { eval$ = eval }, eval$ /; True ];



(******************************************************************************)


unlockedSymbolQ // Attributes = { HoldAllComplete };
unlockedSymbolQ // Options    = { };


unlockedSymbolQ[ s_Symbol? symbolQ ] := FreeQ[ Attributes @ Unevaluated @ s, Locked ];
unlockedSymbolQ[ ___ ] := False;


(******************************************************************************)


withHolding // Attributes = { HoldAllComplete };
withHolding // Options    = { };


withHolding /:
  patt_ :> withHolding[ { sets___Set, set_Set }, expr_ ] :=
    patt :> withHolding[ { sets }, With[ { set }, expr /; True ] ];


withHolding /:
  patt_ :> withHolding[ { }, expr_ ] :=
    patt :> expr;


(******************************************************************************)


symbolHash // Attributes = { HoldAllComplete };
symbolHash // Options    = { };

symbolHash[ sym : (_Symbol? symbolQ | _String) ] :=
  Block[ { $symbol },
      With[ { def = Language`ExtendedFullDefinition @ sym },
          Hash[ Sort @ def /.
                  MapIndexed[ # -> $symbol @ First @ #2 &,
                              Cases[ def, HoldPattern[ HoldForm[ s_ ] -> _ ] :> HoldPattern @ s ]
                  ]
          ]
      ]
  ];


(******************************************************************************)


compressToBytes // ClearAll;
compressToBytes[ expr_ ] :=
  withContext @ GeneralUtilities`CompressToByteArray @ HoldComplete @ expr;


uncompressBytes // ClearAll;
uncompressBytes[ bytes_ByteArray, wrapper_: Identity ] :=
  wrapper @@ GeneralUtilities`UncompressFromByteArray @ bytes;


(******************************************************************************)


equiv // ClearAll;

equiv[ "UUID", uuid_ ] := uuidQ @ uuid;
equiv[ "ResourceLocations", loc_ ] := Sort[ Head /@ loc ];
equiv[ "ExampleNotebook", nb_ ] := Head @ nb;

equiv[ "Function", f_ ] :=
  f /. s_Symbol? symbolQ /; inBaseContextQ @ safeContext @ s :>
    RuleCondition @ equiv[ "SymbolName", fullSymbolName @ s ];

equiv[ "SymbolName", name_ ] :=
  StringTrim[
      StringRiffle[
          Rest @ StringSplit[
              StringDelete[ name, $ResourceFunctionRootContext ],
              "`"
          ],
          "`"
      ],
      DigitCharacter ..
  ];

equiv[ "VerificationTests", tests_ ] :=
  equiv[ "Function", tests ];

equiv[ "Definition", f_ ] :=
  DeleteCases[
      DeleteCases[
          DeleteCases[ equiv[ "Function", f ],
              HoldPattern[ Verbatim[ HoldPattern ][ MessageName[ _, "shdw" ] ] -> _ ],
              Infinity
          ],
          Messages -> { },
          Infinity
      ],
      HoldForm[ _String ] -> { }
  ];

equiv[ _, e_ ] := e;


$noCompareInfoKeys // ClearAll;
$noCompareInfoKeys = {
    "DefinitionData",
    "FunctionLocation"
};

$compareInfoKeys // ClearAll;
$compareInfoKeys =
  Complement[ ResourceSystemClient`Private`usableResourceInfoKeys[ "Function" ],
              $noCompareInfoKeys
  ];

equivalentInfoQ // ClearAll;

equivalentInfoQ[ info1_Association, info2_Association ] :=
  SameQ[ KeyValueMap[ equiv, info1[[ $compareInfoKeys ]] ],
         KeyValueMap[ equiv, info2[[ $compareInfoKeys ]] ]
  ];

equivalentInfoQ[ (ResourceFunction|ResourceObject)[ info1_, ___ ], info2_ ] :=
  equivalentInfoQ[ info1, info2 ];

equivalentInfoQ[ info1_, (ResourceFunction|ResourceObject)[ info2_, ___ ] ] :=
  equivalentInfoQ[ info1, info2 ];

equivalentInfoQ[ ___ ] := False;


equivalentDefinitionsQ // ClearAll;
equivalentDefinitionsQ[ rf1_, rf2_ ] :=
  Module[ { data1, data2 },
      data1 = ResourceFunctionInformation[ rf1, "DefinitionList" ];
      data2 = ResourceFunctionInformation[ rf2, "DefinitionList" ];
      equiv[ "Definition", data1 ] === equiv[ "Definition", data2 ]
  ];


equivalentResourceFunctionsQ // ClearAll;
equivalentResourceFunctionsQ[ rf1_, rf2_ ] :=
  And @@ Map[ #[ rf1, rf2 ] &, { equivalentInfoQ, equivalentDefinitionsQ } ];


(******************************************************************************)


createSymbolName // ClearAll;

createSymbolName[ _, KeyValuePattern[ "ShortName" -> name_ ], ___ ] :=
  With[ { temp = StringJoin[ $ResourceFunctionTempContext, name ] },
      Symbol @ Quiet @ Check[ Unprotect @ temp;
                              ClearAll @ temp;
                              temp
                              ,
                              fullSymbolName @@ { Unique @ temp }
                       ]
  ];

createSymbolName[ ___ ] :=
  $failed;



cleanupTemporarySymbols[ ] :=
  With[ { pattern = $ResourceFunctionTempContext <> "*" },
      <|
          "Unprotected" -> Check[ Unprotect @ pattern, $Failed ],
          "Removed"     -> Check[ Remove @ pattern   , $Failed ]
      |>
  ] // Quiet;



forkSymbolDefinition[ symb_, usersymb_ ] :=
  Set @@ HoldComplete[ symb, usersymb ](* TODO *)


(******************************************************************************)


clearResourceFunction // ClearAll;

clearResourceFunction[ rf_ResourceFunction ] :=
  failOnMessage @ Module[ { symName, heldSym, ctx, uuid },
      symName = resourceFunctionStaticProperty[ rf, "SymbolName" ];
      heldSym = ToExpression[ symName, InputForm, HoldComplete ];
      ctx = getBaseContext @ heldSym;
      Scan[ Unprotect, Names[ ctx ~~ __ ] ] // Quiet;
      Scan[ Remove, Names[ ctx ~~ __ ] ] // Quiet;
      uuid = resourceFunctionStaticProperty[ rf, "UUID" ];
      ResourceFunctionLoadedQ[ uuid ] = False;
  ];


(* This will turn on some optional formatting for long symbols belonging to resource functions *)
enableResourceFunctionSymbolFormatting // ClearAll;

enableResourceFunctionSymbolFormatting[ ] :=

  Module[ { $localSymbolHashMod, $localSymbolColorFunction, rfSymbolQ },

      $localSymbolHashMod = 64;
      $localSymbolColorFunction = ColorData @ 97;

      rfSymbolQ = Function[ symbol,
                            StringMatchQ[ Context @ Unevaluated @ symbol,
                                          $ResourceFunctionRootContext ~~ __
                            ],
                            { HoldAllComplete }
                  ];

      MakeBoxes[ s_Symbol ? rfSymbolQ, StandardForm ] :=

        Module[ { string, symHash, color, simpleName },

            string = ToString @ Unevaluated @ s;
            symHash = Mod[ Hash @ Context @ Unevaluated @ s, $localSymbolHashMod ];
            color = $localSymbolColorFunction @ symHash;
            simpleName = SymbolName @ Unevaluated @ s;

            With[ { str = string, n = simpleName, c = color },
                InterpretationBox[ TagBox[ TooltipBox[ StyleBox[ n, c, Italic, Bold ],
                                                       str
                                           ],
                                           #1 &
                                   ],
                                   s
                ]
            ]
        ]
  ];


(******************************************************************************)


End[] 

EndPackage[]
