
BeginPackage["TypeFramework`Inference`TypeInferenceState`"]

TypeInferenceStateQ
CreateTypeInferenceState

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`Inference`ConstraintSolveState`"]
Needs["TypeFramework`ConstraintObjects`GeneralizeConstraint`"];
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



$DebugMode := TypeFramework`Utilities`Error`Private`$DebugMode

generalizeWithPredicates := generalizeWithPredicates = 
    TypeFramework`ConstraintObjects`GeneralizeConstraint`Private`generalizeWithPredicates;


RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeInferenceStateClass = DeclareClass[
	TypeInferenceState,
	<|
        "initialize" -> Function[{},
            Self["setProperties", CreateReference[<||>]];
        ],
		"generate" -> Function[ {prog}, generate[ Self, prog]],
		"infer" -> (infer[Self, ##]&),
		"addAssumption" -> (addAssumption[Self, ##]&),
		"hasAssumption" -> (hasAssumption[Self, ##]&),
		"lookupAssumptions" -> (lookupAssumptions[Self, ##]&),
		"dropAssumption" -> (dropAssumption[Self, ##]&),
        "dropAssumptions" -> (dropAssumptions[Self, ##]&),
        "mergeAssumptions" -> (mergeAssumptions[Self, ##]&),
		"processAssumptions" -> (processAssumptions[Self, ##]&),
		"cloneMonomorphicSet" -> (cloneMonomorphicSet[Self, ##]&),
        "cloneAssumptions" -> (cloneAssumptions[Self, ##]&),
        "appendSuccessConstraint" -> (appendSuccessConstraint[Self, ##]&),
        "appendFailureConstraint" -> (appendFailureConstraint[Self, ##]&),
		"appendEqualConstraint" -> (appendEqualConstraint[Self, ##]&),
        "appendProveConstraint" -> (appendProveConstraint[Self, ##]&),
        "appendAssumeConstraint" -> Function[{t1}, appendAssumeConstraint[Self, t1]],
		"appendLookupConstraint" -> (appendLookupConstraint[Self, ##]&),
        "appendAlternativeConstraint" -> Function[{t1, initial, t2, m}, appendAlternativeConstraint[Self, t1, initial, t2, m]],
        "appendGeneralizeConstraint" -> Function[{t1, t2, m}, appendGeneralizeConstraint[Self, t1, t2, m]],
		"appendInstantiateConstraint" -> Function[{t1, t2}, appendInstantiateConstraint[Self, t1, t2]],
        "appendImplicitInstanceConstraint" -> Function[{t1, t2, m}, appendImplicitInstanceConstraint[Self, t1, t2, m]],
        "prependSuccessConstraint" -> (prependSuccessConstraint[Self, ##]&),
        "prependFailureConstraint" -> (prependFailureConstraint[Self, ##]&),
        "prependEqualConstraint" -> (prependEqualConstraint[Self, ##]&),
        "prependProveConstraint" -> (prependProveConstraint[Self, ##]&),
        "prependAssumeConstraint" -> Function[{t1}, prependAssumeConstraint[Self, t1]],
        "prependLookupConstraint" -> (prependLookupConstraint[Self, ##]&),
        "prependAlternativeConstraint" -> Function[{t1, initial, t2, m}, prependAlternativeConstraint[Self, t1, initial, t2, m]],
        "prependGeneralizeConstraint" -> Function[{t1, t2, m}, prependGeneralizeConstraint[Self, t1, t2, m]],
        "prependInstantiateConstraint" -> Function[{t1, t2}, prependInstantiateConstraint[Self, t1, t2]],
        "prependImplicitInstanceConstraint" -> Function[{t1, t2, m}, prependImplicitInstanceConstraint[Self, t1, t2, m]],
        "pushScoped" -> Function[{vars}, pushScoped[Self, vars]],
        "popScoped" -> Function[{scope}, popScoped[Self, scope]],
       	"isScoped" -> Function[{var}, isScoped[Self, var]],
		"resolveType" -> Function[{ty}, resolveType[Self, ty]],
		"format" -> (format[Self,##]&),
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"typeEnvironment",
		"generateFunction",
		"constraints",
		"assumptions",
		"monomorphicSet",
		"scoped",
        "properties"
    },
	Predicate -> TypeInferenceStateQ,
    Extends -> {
        ClassPropertiesTrait
    }
]
]]



CreateTypeInferenceState[env_?TypeEnvironmentQ, generateFunction_, opts:OptionsPattern[]] :=
    CreateTypeInferenceState[env, generateFunction, <| opts |>];
CreateTypeInferenceState[env_?TypeEnvironmentQ, generateFunction_, opts_?AssociationQ] :=
	With[{
	   obj = CreateObject[TypeInferenceState, <|
	        "typeEnvironment" -> env,
	        "generateFunction" -> generateFunction,
	        "constraints" -> CreateConstraintSolveState[env],
	        "assumptions" -> CreateReference[<||>],
	        "monomorphicSet" -> CreateReference[{}],
	        "scoped" -> CreateReference[<||>]
	   |>]
	},
	   obj["setProperties", CreateReference[opts]];
	   obj
	];

CreateTypeInferenceState[args___] :=
    ThrowException[{"Unrecognized call to CreateTypeInferenceState", {args}}]



generate[ self_, prog_] :=
	Module[ {ty},
		ty = self["generateFunction"][self, prog];
		processAssumptions[self];
		ty
	]


processAssumptions[self_] :=
    Module[ {assumptions},
        assumptions = self["assumptions"];
		Scan[
			Function[ {varName},
				If[ self["typeEnvironment"]["functionTypeLookup"]["hasTypes", varName],
					Map[
						Function[{tyVar},
							(* Not sure why I have to add ToString -- surely varName is a String already? But it seems
							 * to be a Symbol. *)
							self["appendLookupConstraint", varName, tyVar, {}, <| "source" -> (ToString[varName] -> {tyVar}) |>]
						],
						assumptions["lookup", varName]
					];
					self["dropAssumption", varName];
				]
			]
			,
			assumptions["keys"]
        ];
    ]



(*
  This is the only place in the Inference mechanism that hardcodes a type.
*)
inferError[ self_, a_, b_] :=
	self["resolveType", TypeSpecifier["Error"]]

infer[ self_, prog_] :=
    infer[self, prog, <| |>];
infer[ self_, prog_, opts_?AssociationQ] :=
	Module[ {ty, subs},
		CatchTypeFailure[
			ty = self["generate", prog];
			If[ Length[self["assumptions"]["get"]] =!= 0,
				Throw[TypeInferenceException[self["assumptions"]["get"]], TypeError]
			];
			{subs, ty} = self["constraints"]["solve", ty, opts];
			If[Length[ty["free"]] === 0,
			     ty,
			     generalizeWithPredicates[
			         <|
			             "ConstraintSolveState" -> self["constraints"],
                         "TypeEnvironment" -> self["typeEnvironment"]
			         |>,
			         self["monomorphicSet"]["get"],
			         ty
			    ]
			]
			,
			TypeError,
			(
			    If[$DebugMode,
				     Print[#1]
				];
				inferError[self, ##]
			)&
		]
	];

cloneMonomorphicSet[self_] :=
	self["monomorphicSet"]["clone"]
	
	
cloneAssumptions[self_] :=
    self["assumptions"]["clone"]

addAssumption[self_, key_ -> val_, props_:<||>] :=
	addAssumption[self, key, val, props]


addAssumption[self_, key_, val:List[tyVar_], props_:<||>] :=
	Module[{ent},
		tyVar["joinProperties", props];
		ent = self["assumptions"]["lookup", key, {}];
		ent = Flatten[{ent, val}];
		self["assumptions"]["associateTo", key -> ent];
		self
	]

lookupAssumptions[ self_, key_, default_:{}] :=
	self["assumptions"]["lookup", key, default]

hasAssumption[self_, key_] :=
	self["assumptions"]["keyExistsQ", key]

mergeAssumptions[self_, other_?ReferenceQ] := (
    self["setAssumptions",
        CreateReference[
            Merge[{self["assumptions"]["get"], other["get"]}, Flatten]
        ]
    ];
    self
);
    
mergeAssumptions[self_, other_?AssociationQ] :=
    mergeAssumptions[self, CreateReference[other]]

dropAssumption[self_, key_] := 
    If[ListQ[key],
        dropAssumptions[self, key],
        dropAssumptions[self, {key}]
    ];

dropAssumptions[self_, keys_?ListQ] := (
    self["assumptions"]["keyDropFrom", keys];
    self
);
    

pushScoped[self_, vars_List] :=
	Module[{scoped = self["scoped"], oldScoped},
		oldScoped = scoped["clone"];
		Scan[
			scoped["associateTo", # -> True]&,
			vars
		];
		oldScoped
	]

popScoped[self_, scope_] :=
	self["setScoped", scope]

isScoped[self_, var_] :=
	self["scoped"]["keyExistsQ", var]

appendSuccessConstraint[self_, props_:<||>] :=
    self["constraints"]["appendSuccessConstraint", props];
appendSuccessConstraint[args___] :=
    ThrowException[{"Unknown arguments to appendSuccessConstraint", {args}}]

appendFailureConstraint[self_, props_:<||>] :=
    self["constraints"]["appendFailureConstraint", props];
appendFailureConstraint[args___] :=
    ThrowException[{"Unknown arguments to appendFailureConstraint", {args}}]

appendEqualConstraint[self_, t1_Type, t2_, props_:<||>] :=
	appendEqualConstraint[self, self["resolveType", t1], t2, props]
appendEqualConstraint[self_, t1_, t2_Type, props_:<||>] :=
	appendEqualConstraint[self, t1, self["resolveType", t2], props]
appendEqualConstraint[self_, t1_?TypeObjectQ, t2_?TypeObjectQ, props_:<||>] :=
	self["constraints"]["appendEqualConstraint", t1, t2, props];
appendEqualConstraint[args___] :=
	ThrowException[{"Unknown arguments to appendEqualConstraint", {args}}]	
	
appendProveConstraint[self_, t1_Type, props_:<||>] :=
	appendProveConstraint[self, self["resolveType", t1], props]
appendProveConstraint[self_, t1_?TypeObjectQ, props_:<||>] := 
    self["constraints"]["appendProveConstraint", t1, props];
appendProveConstraint[args___] :=
	ThrowException[{"Unknown arguments to appendProveConstraint", {args}}]	
	
appendAssumeConstraint[self_, t1_Type, props_:<||>] :=
	appendAssumeConstraint[self, self["resolveType", t1], props]	
appendAssumeConstraint[self_, t1_?TypeObjectQ, props_:<||>] :=
    self["constraints"]["appendAssumeConstraint", t1, props];  
appendAssumeConstraint[args___] :=
	ThrowException[{"Unknown arguments to appendAssumeConstraint", {args}}]	

appendLookupConstraint[self_, name_, t1_Type, m_, props_:<||>] :=
	appendLookupConstraint[self, name, t1, self["resolveType", t1], m, props]	
appendLookupConstraint[self_, name_, ty_?TypeObjectQ, m_, props_:<||>] :=
	self["constraints"]["appendLookupConstraint", name, ty, m, props];
appendLookupConstraint[args___] :=
	ThrowException[{"Unknown arguments to appendLookupConstraint", {args}}]	
	
appendAlternativeConstraint[self_, t1_Type, initial_, alts_, m_, props_:<||>] :=
	appendAlternativeConstraint[self, self["resolveType", t1], initial, alts, m, props]
appendAlternativeConstraint[self_, ty_?TypeObjectQ, initial_, alts_, m_, props_:<||>] :=
    self["constraints"]["appendAlternativeConstraint", ty, initial, alts, m, props];
appendAlternativeConstraint[args___] :=
    ThrowException[{"Unknown arguments to appendAlternativeConstraint", {args}}] 
    
appendGeneralizeConstraint[self_, sigma_Type, tau_, m_, props_:<||>] :=
	appendGeneralizeConstraint[self, self["resolveType", sigma], tau, m, props]
appendGeneralizeConstraint[self_, sigma_, tau_Type, m_, props_:<||>] :=
	appendGeneralizeConstraint[self, sigma, self["resolveType", tau], m, props]
appendGeneralizeConstraint[self_, sigma_?TypeObjectQ, tau_?TypeObjectQ, m_?ListQ, props_:<||>] :=
    self["constraints"]["appendGeneralizeConstraint", sigma, tau, m, props];
appendGeneralizeConstraint[args___] :=
    ThrowException[{"Unknown arguments to appendGeneralizeConstraint", {args}}]    
    
appendInstantiateConstraint[self_, tau_Type, sigma_, props_:<||>] :=
    appendInstantiateConstraint[self, self["resolveType", tau], sigma, props];
appendInstantiateConstraint[self_, tau_, sigma_Type, props_:<||>] :=
    appendInstantiateConstraint[self, tau, self["resolveType", sigma], props];
appendInstantiateConstraint[self_, tau_, sigma_, props_:<||>] :=
    self["constraints"]["appendInstantiateConstraint", tau, sigma, props];
appendInstantiateConstraint[args___] :=
    ThrowException[{"Unknown arguments to appendInstantiateConstraint", {args}}]  

appendImplicitInstanceConstraint[self_, t1_Type, t2_, props_:<||>] :=
    appendImplicitInstanceConstraint[self, self["resolveType", t1], t2, props];
appendImplicitInstanceConstraint[self_, t1_, t2_Type, props_:<||>] :=
    appendImplicitInstanceConstraint[self, t1, self["resolveType", t2], props];
appendImplicitInstanceConstraint[self_, t1_?TypeObjectQ, t2_?TypeObjectQ, m_, props_:<||>] :=
    self["constraints"]["appendImplicitInstanceConstraint", t1, t2, m, props];
appendImplicitInstanceConstraint[args___] :=
    ThrowException[{"Unknown arguments to appendImplicitInstanceConstraint", {args}}]


prependSuccessConstraint[self_, props_:<||>] :=
    self["constraints"]["prependSuccessConstraint", props];

prependFailureConstraint[self_, props_:<||>] :=
    self["constraints"]["prependFailureConstraint", props];

prependEqualConstraint[self_, t1_?TypeObjectQ, t2_?TypeObjectQ, props_:<||>] :=
	self["constraints"]["prependEqualConstraint", t1, t2, props];
    
prependProveConstraint[self_, t1_?TypeObjectQ, props_:<||>] := 
    self["constraints"]["prependProveConstraint", t1, props];

prependAssumeConstraint[self_, t1_?TypeObjectQ] :=
    self["constraints"]["prependAssumeConstraint", t1];
    
prependEqualConstraint[args___] :=
    ThrowException[{"Unknown arguments to prependEqualConstraint", {args}}]  

prependLookupConstraint[self_, name_, ty_?TypeObjectQ, m_, props_:<||>] :=
	self["constraints"]["prependLookupConstraint", name, ty, m, props];

prependLookupConstraint[args___] :=
    ThrowException[{"Unknown arguments to prependLookupConstraint", {args}}] 
    
prependAlternativeConstraint[self_, ty_, initial_, alts_, m_] :=
    self["constraints"]["prependAlternativeConstraint", ty, initial, alts, m];
    
prependAlternativeConstraint[args___] :=
    ThrowException[{"Unknown arguments to prependAlternativeConstraint", {args}}] 
    

prependGeneralizeConstraint[self_, sigma_?TypeObjectQ, tau_?TypeObjectQ, m_?ListQ] :=
    self["constraints"]["prependGeneralizeConstraint", sigma, tau, m];

prependGeneralizeConstraint[args___] :=
    ThrowException[{"Unknown arguments to prependGeneralizeConstraint", {args}}]    
    
prependInstantiateConstraint[self_, tau_, sigma_] :=
    self["constraints"]["prependInstantiateConstraint", tau, sigma];

prependInstantiateConstraint[args___] :=
    ThrowException[{"Unknown arguments to prependInstantiateConstraint", {args}}]  

prependImplicitInstanceConstraint[self_, t1_?TypeObjectQ, t2_?TypeObjectQ, m_] :=
    self["constraints"]["prependImplicitInstanceConstraint", t1, t2, m];
    
prependImplicitInstanceConstraint[args___] :=
    ThrowException[{"Unknown arguments to prependImplicitInstanceConstraint", {args}}]

resolveType[self_, ty_] :=
	self["typeEnvironment"]["resolve", ty]

(**************************************************)

icon := icon = Graphics[Text[
	Style["INF\nST",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[self_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"TypeInferenceState",
		self,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["constraints: ", {90, Automatic}], self["constraints"]["length"]}],
			BoxForm`SummaryItem[{Pane["assumptions: ", {90, Automatic}], Length[self["assumptions"]["keys"]]}]
  		},
  		{
  			
  		}, 
  		fmt
  	]


toString[env_] := "TypeInferenceState[<>]"


format[self_, shortQ_:True] :=
    RawBoxes[BoxForm`ArrangeSummaryBox[
        "TypeInferenceState",
        self,
        None,
        {
            BoxForm`MakeSummaryItem[{Pane["constraints: ",    
            	{90, Automatic}], Column@self["constraints"]["format", shortQ]}, StandardForm],
            BoxForm`MakeSummaryItem[{Pane["assumptions: ",    
            	{90, Automatic}], Map[ Map[#["format", shortQ]&, #]&, self["assumptions"]["get"]]}, StandardForm],
            BoxForm`MakeSummaryItem[{Pane["monomorphicSet: ", {90, Automatic}], self["monomorphicSet"]["get"]}, StandardForm]
        },
        {
        },
        StandardForm
    ]]




End[]

EndPackage[]

