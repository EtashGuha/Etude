
FunctionResource`$ResourceFunctionRootContext;
FunctionResource`$FunctionResourceTypes;
FunctionResource`ResourceFunctionLoadedQ;
FunctionResource`ResourceFunctionInformation;
FunctionResource`ResourceFunctionInformationData;
FunctionResource`UpdateDefinitionNotebook;

Get["FunctionResource`Dependencies`"]
Get["FunctionResource`CommonTools`"]
Get["FunctionResource`Copy`"]
Get["FunctionResource`Utilities`"]
Get["FunctionResource`CreateResources`"]
Get["FunctionResource`Metadata`"]
Get["FunctionResource`Objects`"]
Get["FunctionResource`UpValues`"]
Get["FunctionResource`Notebooks`"]
Get["FunctionResource`Function`"]
Get["FunctionResource`Submit`"]
Get["FunctionResource`Information`"]


(******************************************************************************)
(* ::Section::Closed:: *)
(*Extra Initialization*)


SetOptions[ Language`ExtendedFullDefinition,
    "ExcludedContexts" ->
      Join[ OptionValue[ Language`ExtendedFullDefinition, "ExcludedContexts" ],
            { "ResourceSystemClient", "FunctionResource" }
      ]
];

(*Autocomplete*)
If[ TrueQ @ And[ $EvaluationEnvironment === "Session",
                 ! $CloudEvaluation,
                 $Notebooks
            ],
    FunctionResource`Autocomplete`InitializeAutocomplete[ ]
];


(*Fix FE Symbols that have no context*)
Scan[ If[ ! NameQ[ # ], Symbol[ "System`" <> # ] ] &,
    {
        "MenuAnchor",
        "WholeCellGroupOpener",
        "ImageEditMode"
    }
];
