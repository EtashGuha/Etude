(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["FunctionResource`"]

Begin["`Private`"] (* Begin Private Context *)

checkPaclet[name_]:=Block[{paclets},
	paclets=PacletManager`PacletFind[name];
	If[Length[paclets]>0,
		True
		,
		paclets=PacletManager`PacletFindRemote[name];
		If[Length[paclets]>0,
			True
			,
			False
		]
	]
]

If[TrueQ[checkPaclet["ResourceSystemClient"]],
	Needs["ResourceSystemClient`"]
	,
	Message[ResourceData::norsys, "FunctionResource"]
]




(******************************************************************************)
(* Load files only when needed                                                *)
(******************************************************************************)


$autoloadingSymbols = HoldComplete[
    { FunctionResource`UpdateDefinitionNotebook, "FunctionResource`DefinitionNotebook`" }
    ,
    { FunctionTemplateToggle`DT`FunctionTemplateLiteralInput, "FunctionResource`DocuToolsTemplate`" },
    { FunctionTemplateToggle`DT`FunctionTemplateToggle, "FunctionResource`DocuToolsTemplate`" },
    FunctionResource`DocuToolsTemplate`FunctionTemplateLiteralInput,
    FunctionResource`DocuToolsTemplate`FunctionTemplateToggle,
    FunctionResource`DocuToolsTemplate`TableInsert,
    FunctionResource`DocuToolsTemplate`TableMerge,
    FunctionResource`DocuToolsTemplate`TableSort,
    FunctionResource`DocuToolsTemplate`DocDelimiter,
    FunctionResource`DocuToolsTemplate`FunctionLinkButton
    ,
    FunctionResource`DocumentationNotebook`GetDocumentationNotebook,
    FunctionResource`DocumentationNotebook`SaveDocumentationNotebook,
    FunctionResource`DocumentationNotebook`ViewDocumentationNotebook,
    FunctionResource`DocumentationNotebook`LocalDocumentationAvailableQ
    ,
    FunctionResource`DefinitionNotebook`CheckDefinitionNotebook,
    FunctionResource`DefinitionNotebook`Private`showProgress,
    FunctionResource`DefinitionNotebook`Private`viewExampleNotebook,
    FunctionResource`DefinitionNotebook`Private`viewStyleGuidelines,
    FunctionResource`DefinitionNotebook`Private`getResource,
    FunctionResource`DefinitionNotebook`Private`submitRepository,
    FunctionResource`DefinitionNotebook`Private`scrapeResourceFunction,
    FunctionResource`DefinitionNotebook`Private`deleteMe
    ,
    FunctionResource`Autocomplete`InitializeAutocomplete
];


setSymbolAutoLoad // Attributes = { HoldFirst };

setSymbolAutoLoad[ symbol_Symbol, context_String ] :=
  If[ ! TrueQ @ GeneralUtilities`HasDefinitionsQ @ symbol,
      symbol := (
          ClearAll @ symbol;
          PreemptProtect @ Block[ { $ContextPath }, Needs @ context ];
          symbol
      )
  ];

setSymbolAutoLoad[ { symbol_, context_ } ] :=
  setSymbolAutoLoad[ symbol, context ];

setSymbolAutoLoad[ symbol_Symbol ] :=
  setSymbolAutoLoad[
      symbol,
      StringReplace[ Context @ Unevaluated @ symbol,
                     "`Private`"~~EndOfString :> "`"
      ]
  ];



Cases[ $autoloadingSymbols, s_ :> setSymbolAutoLoad @ s ];



(******************************************************************************)


End[];

EndPackage[]

SetAttributes[{},
   {ReadProtected, Protected}
];