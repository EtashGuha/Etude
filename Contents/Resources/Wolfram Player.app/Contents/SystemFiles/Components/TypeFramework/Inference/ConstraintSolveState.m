
BeginPackage["TypeFramework`Inference`ConstraintSolveState`"]

ConstraintSolveStateQ
CreateConstraintSolveState

Begin["`Private`"]


Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`ConstraintObjects`AlternativeConstraint`"]
Needs["TypeFramework`ConstraintObjects`EqualConstraint`"]
Needs["TypeFramework`ConstraintObjects`GeneralizeConstraint`"]
Needs["TypeFramework`ConstraintObjects`InstantiateConstraint`"]
Needs["TypeFramework`ConstraintObjects`LookupConstraint`"]
Needs["TypeFramework`ConstraintObjects`ProveConstraint`"]
Needs["TypeFramework`ConstraintObjects`AssumeConstraint`"]
Needs["TypeFramework`ConstraintObjects`SuccessConstraint`"]
Needs["TypeFramework`ConstraintObjects`FailureConstraint`"]
Needs["TypeFramework`Inference`ConstraintData`"]
Needs["TypeFramework`Inference`Substitution`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`TypeObjects`TypePredicate`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
ConstraintSolveStateClass = DeclareClass[
	ConstraintSolveState,
	<|
		"solve" -> (solve[Self, ##]&),
		"length" -> Function[{ }, Self["constraints"]["length"]],
		"appendSuccessConstraint" -> (appendSuccessConstraint[Self, ##]&),
        "appendFailureConstraint" -> (appendFailureConstraint[Self, ##]&),
        "appendEqualConstraint" -> (appendEqualConstraint[Self, ##]&),
		"appendLookupConstraint" -> (appendLookupConstraint[Self, ##]&),
        "appendAlternativeConstraint" -> Function[{t1, initial, t2, m}, appendAlternativeConstraint[Self, t1, initial, t2, m]],
        "appendAssumeConstraint" -> Function[{t1}, appendAssumeConstraint[Self, t1]],
        "appendProveConstraint" -> (appendProveConstraint[Self, ##]&),
        "appendGeneralizeConstraint" -> Function[{t1, t2, m}, appendGeneralizeConstraint[Self, t1, t2, m]],
        "appendInstantiateConstraint" -> Function[{t1, t2}, appendInstantiateConstraint[Self, t1, t2]],
        "appendImplicitInstanceConstraint" -> Function[{t1, t2, m}, appendImplicitInstanceConstraint[Self, t1, t2, m]],
        "prependSuccessConstraint" -> (prependSuccessConstraint[Self, ##]&),
        "prependFailureConstraint" -> (prependFailureConstraint[Self, ##]&),
        "prependEqualConstraint" -> (prependEqualConstraint[Self, ##]&),
        "prependLookupConstraint" -> (prependLookupConstraint[Self, ##]&),
        "prependAlternativeConstraint" -> Function[{t1, initial, t2, m}, prependAlternativeConstraint[Self, t1, initial, t2, m]],
        "prependAssumeConstraint" -> Function[{t1}, prependAssumeConstraint[Self, t1]],
        "prependProveConstraint" -> (prependProveConstraint[Self, ##]&),
        "prependGeneralizeConstraint" -> Function[{t1, t2, m}, prependGeneralizeConstraint[Self, t1, t2, m]],
        "prependInstantiateConstraint" -> Function[{t1, t2}, prependInstantiateConstraint[Self, t1, t2]],
        "prependImplicitInstanceConstraint" -> Function[{t1, t2, m}, prependImplicitInstanceConstraint[Self, t1, t2, m]],
		"appendConstraint" -> Function[{con}, appendConstraint[Self, con]],
        "prependConstraint" -> Function[{con}, prependConstraint[Self, con]],
		"addScheme" -> (addScheme[Self, ##]&),
        "lookupScheme" -> (lookupScheme[Self, ##]&),
        "hasScheme" -> (hasScheme[Self, ##]&),
        "contextReduction" -> (contextReduction[Self, ##]&),
        "createData" -> (createData[Self]&),
		"format" -> (format[Self,##]&),
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"options",
		"typeEnvironment",
		"schemes", (* Mapping scheme variables to schemes *)
		"predicatesToProve",
		"assumedPredicates",
		"substitution",
		"constraints",
		"constraintData" -> Null
	},
	Predicate -> ConstraintSolveStateQ
]
]]



CreateConstraintSolveState[env_] :=
	CreateObject[ConstraintSolveState, <|
		"typeEnvironment" -> env,
        "schemes" -> CreateReference[<||>],
		"constraints" -> CreateReference[{}],
		"predicatesToProve" -> CreateReference[{}],
        "assumedPredicates" -> CreateReference[{}],
        "substitution" -> CreateTypeSubstitution["TypeEnvironment" -> env]
	|>];


addScheme[self_, key_?TypeVariableQ -> val_] :=
    addScheme[self, key["id"] -> val];
addScheme[self_, key_?IntegerQ -> val_] :=
    self["schemes"]["associateTo", key -> val];
lookupScheme[self_, key_?TypeVariableQ] :=
    lookupScheme[self, key["id"]];
lookupScheme[self_, key_?IntegerQ] :=
    If[self["hasScheme", key],
        self["schemes"]["lookup", key],
        ThrowException[{"cannot find scheme `1` in constraint solve state", key}]
    ];
hasScheme[self_, key_?TypeVariableQ] :=
    hasScheme[self, key["id"]];
hasScheme[self_, key_?IntegerQ] :=
    self["schemes"]["keyExistsQ", key];


tv[e_] :=
    e["free"]

tv[lst_?ListQ] :=
    Join @@ Map[tv, lst] 
    
tvId[e_] :=
    Keys[tv[e]]

ambiguities[self_, tyVars_, preds_] :=
    With[{
        vs = Complement[tv[preds], tyVars, SameTest ->sameQ]
    },
        Association@Table[
            v -> Select[preds, MemberQ[tvId[#], v["id"]]&],
            {v, vs}
        ]
    ];


(*  Rewrite and break ps into a pair (ds, rs) of deferred predicates ds and "retained" predicates rs *)
splitPredicates[self_, fixedTyVars_, quantTyVars_, reducedPreds_?ListQ] :=
    With[{
        partition = GroupBy[reducedPreds, testFixedPredicate[self, fixedTyVars, #]&]  
    },
        <|
            "Deferred" -> partition["True"],
            "Retained" -> partition["False"]
        |>
    ]
    
(* Only include predicates that might be of use, e.g. ones
 * which have nonfixed variables 
 *)
testFixedPredicate[self_, fixedTyVars_, pred_] :=
    SubsetQ[#["id"]& /@ fixedTyVars, Keys[pred["free"]]]
        

contextReduction[self_] :=
    contextReduction[self, self["predicatesToProve"]["get"]];
contextReduction[self_, preds:{___?TypePredicateQ}] :=
    With[{
        tenv = self["typeEnvironment"]
    },
    With[{
        abstractTenv = tenv["abstracttypes"]
    },
    With[{
        reducedPreds = abstractTenv["contextReduce", preds]
    },
        self["predicatesToProve"]["set", reducedPreds] 
    ]]];

contextReduction[args___] :=
    ThrowException[{"Unrecognized call to contextReduction", {args}}]


resolvePredicateDefaults[self_] :=
    With[{
        preds = resolvePredicateDefaults[self, self["predicatesToProve"]["get"]]
    },
        self["predicatesToProve"]["set", preds]
    ];
resolvePredicateDefaults[self_, preds_] :=
    With[{
        tenv = self["typeEnvironment"]
    },
    With[{
        abstractTenv = tenv["abstracttypes"]
    },
	    Table[
	        If[abstractTenv["hasDefault", pred["underlyingAbstractType"]],
	            With[{
	                default = abstractTenv["getDefault", pred["underlyingAbstractType"]]
	            },
	                If[ListQ[default],
	                    MapThread[
	                        self["appendEqualConstraint", #1, #2, <| "source" -> pred |>]&,
	                        {pred["types"], default}
	                    ],
	                    self["appendEqualConstraint", #, default, <| "source" -> pred |>]& /@ pred["types"]
	                ]
	            ];
	            Nothing,
	            pred
	        ],
	        {pred, preds}
	    ]
    ]];

appendConstraint[self_, con_] := (
	self["constraints"]["appendTo", con];
	con
);

appendSuccessConstraint[self_, props_:<||>] :=
    With[{
        constraint = CreateSuccessConstraint[]
    },
        constraint["setProperties", CreateReference[props]];
        self["appendConstraint", constraint]
    ]
    
appendFailureConstraint[self_, props_:<||>] :=
    With[{
        constraint = CreateFailureConstraint[]
    },
        constraint["setProperties", CreateReference[props]];
        self["appendConstraint", constraint]
    ]
	
appendEqualConstraint[self_, t1_, t2_, props_:<||>] :=
	With[{
		constraint = CreateEqualConstraint[t1, t2]
	},
		constraint["setProperties", CreateReference[props]];
		self["appendConstraint", constraint]	
	]

appendLookupConstraint[self_, t1_, t2_, m_, props_:<||>] :=
	With[{
		constraint = CreateLookupConstraint[t1, t2, m]
	},
		constraint["setProperties", CreateReference[props]];
		self["appendConstraint", constraint]
	]
	
appendAlternativeConstraint[self_, ty_, initial_, alts_, m_] :=
    self["appendConstraint", CreateAlternativeConstraint[ty, initial, alts, m]]

appendProveConstraint[self_, t1_?TypePredicateQ, props_:<||>] :=
	With[{
		constraint = CreateProveConstraint[t1]
	},
		constraint["setProperties", CreateReference[props]];
		self["appendConstraint", constraint]
	]

appendAssumeConstraint[self_, t1_?TypePredicateQ] :=
    self["appendConstraint", CreateAssumeConstraint[t1]]
   
appendGeneralizeConstraint[self_, sigma_?TypeObjectQ, tau_?TypeObjectQ, m_?ListQ] :=
    self["appendConstraint", CreateGeneralizeConstraint[sigma, tau, m]]

appendInstantiateConstraint[self_, tau_?TypeObjectQ, rho_?TypeObjectQ] :=
    With[{},
        If[!TypeVariableQ[rho] && !TypeForAllQ[rho],
            Print["WARNING:: Expecting a rho type (for all) as the rhs for the instantiate constraint"]
        ];
        self["appendConstraint", CreateInstantiateConstraint[tau, rho]]
    ]

(* syntax sugar:: t1 <=_M t2 *) 
appendImplicitInstanceConstraint[self_, t1_?TypeObjectQ, t2_?TypeObjectQ, m_?ReferenceQ] :=
    appendImplicitInstanceConstraint[self, t1, t2, m["get"]]
    
appendImplicitInstanceConstraint[self_, t1_?TypeObjectQ, t2_?TypeObjectQ, m_?ListQ] :=
    With[{
        theta = CreateTypeVariable["theta"]
    },
        {appendGeneralizeConstraint[self, theta, t2, m],
         appendInstantiateConstraint[self, t1, theta]}
    ];
    
appendImplicitInstanceConstraint[args___] :=
    ThrowException[{"Unknown arguments to appendImplicitInstanceConstraint", {args}}]


prependConstraint[self_, con_] := (
    self["constraints"]["prependTo", con];
    con
);

prependSuccessConstraint[self_, props_:<||>] :=
    With[{
        constraint = CreateSuccessConstraint[]
    },
        constraint["setProperties", CreateReference[props]];
        self["prependConstraint", constraint]
    ]
    
prependFailureConstraint[self_, props_:<||>] :=
    With[{
        constraint = CreateFailureConstraint[]
    },
        constraint["setProperties", CreateReference[props]];
        self["prependConstraint", constraint]
    ]
    
prependEqualConstraint[self_, t1_, t2_, props_:<||>] :=
    With[{
        constraint = CreateEqualConstraint[t1, t2]
    },
        constraint["setProperties", CreateReference[props]];
        self["prependConstraint", constraint]    
    ]

prependAlternativeConstraint[self_, ty_, initial_, alts_, m_] :=
    self["prependConstraint", CreateAlternativeConstraint[ty, initial, alts, m]]

prependLookupConstraint[self_, t1_, t2_, m_, props_:<||>] :=
	With[{
		constraint = CreateLookupConstraint[t1, t2, m]
	},
		constraint["setProperties", CreateReference[props]];
		self["prependConstraint", constraint]
	]
	
prependProveConstraint[self_, t1_?TypePredicateQ, props_:<||>] :=
	With[{
		constraint = CreateProveConstraint[t1]
	},
		constraint["setProperties", CreateReference[props]];
		self["prependConstraint", constraint]
	]
    
prependAssumeConstraint[self_, t1_?TypePredicateQ] :=
    self["prependConstraint", CreateAssumeConstraint[t1]]
   
prependGeneralizeConstraint[self_, sigma_?TypeObjectQ, tau_?TypeObjectQ, m_?ListQ] :=
    self["prependConstraint", CreateGeneralizeConstraint[sigma, tau, m]]

prependInstantiateConstraint[self_, tau_?TypeObjectQ, rho_?TypeObjectQ] :=
    With[{},
        If[!TypeVariableQ[rho] && !TypeForAllQ[rho],
            Print["WARNING:: Expecting a rho type (for all) as the rhs for the instantiate constraint"]
        ];
        self["prependConstraint", CreateInstantiateConstraint[tau, rho]]
    ]
    
(* syntax sugar:: t1 <=_M t2 *) 
prependImplicitInstanceConstraint[self_, t1_?TypeObjectQ, t2_?TypeObjectQ, m_?ReferenceQ] :=
    prependImplicitInstanceConstraint[self, t1, t2, m["get"]];

prependImplicitInstanceConstraint[self_, t1_?TypeObjectQ, t2_?TypeObjectQ, m_?ListQ] :=
    With[{
        theta = CreateTypeVariable["theta"]
    },
        {prependInstantiateConstraint[self, t1, theta],
         prependGeneralizeConstraint[self, theta, t2, m]}
    ];
    
prependImplicitInstanceConstraint[args___] :=
    ThrowException[{"Unknown arguments to prependImplicitInstanceConstraint", {args}}]

(*
    Constraint Solving
*)

solve[self_] :=
    solve[self, None, <||>];
solve[self_, opts_?AssociationQ] :=
    solve[self, None, opts];
    
solve[self_, ty_] :=
    solve[self, ty, <||>];
solve[self_, ty_, opts_?AssociationQ] :=
	Module[{},
		self["setOptions", opts];
		solve0[self, ty]
	];

solve0[self_, ty0_:None] :=
    Module[{next, ty = ty0, sub = Null, rest},
    	createData[ self];
        While[!self["constraintData"]["emptyQ"],
        	next = self["constraintData"]["nextToSolve"];
	        (*next = nextSolvable[self, constraints];*)
	        If[ next === Null,
	            (*If[self["predicatesToProve"]["length"] > 0,
	               resolvePredicateDefaults[self];
	               Continue[];
	            ];*)
	        	
	        	If[ !self["constraintData"]["emptyQ"],
	        		rest = self["constraintData"]["getUnsolved"];
					If[ Lookup[self["options"], "MonitorSolve", False],
						Print[ConstraintSolveForm[ <|"name" -> "Unsolved", "rest" -> rest|>]]
					];
	            	$UnsolvedConstraints = rest
	            ];
	            Break[]
	            , (* Else *)
	            sub = solveOne[self, next];
	            If[ty =!= None,
	               ty = sub["apply", ty];
	            ];
	        ]
        ];
        (*
         Any unsolved constraints should cause an exception here.
        *)
        If[ !self["constraintData"]["emptyQ"] ,
        	TypeFailure["UnresolvedConstraints", "Unsolved lookup constraint ", self["constraintData"]["getUnsolved"]]
        ];
        If[sub === Null,
        	sub = CreateTypeSubstitution["TypeEnvironment" -> self["typeEnvironment"]]];
        If[ty === None,
            sub,
            {sub, ty}
        ]
    ];




(*
  There was a recoverable error.
  If this is not the last constraint and it hasn't been tried before, 
  then put this one at the back and try again.  Note that AlternativeConstraint
  solving also adds an error.  We used the fixedError counter,  this never gets
  incremented past 1 for AlternativeConstraint.
  
  We get the original constraint to insert back in,  this might be done better 
  in other insertConstraint cases.
*)
alternativeConstraintError[self_, c_, err_] :=
	Module[ {errCnt = c["getProperty", "fixedError", 0], cOrig},
		If[ !errCnt > 1 && !self["constraintData"]["emptyQ"],
			c["setProperty", "fixedError" -> errCnt+1];
			self["constraints"]["appendTo", c];
			cOrig = self["constraintData"]["getOriginal", c];
			insertConstraint[self, cOrig];
			,
			TypeFailure["AlternativeConstraints", err]
		];
		CreateTypeSubstitution[
		  "TypeEnvironment" -> self["typeEnvironment"]
		]
	]

solveOne[self_, c_] :=
	Module[ {},
		CatchTypeFailure[
			solveOne0[self, c],
			"AlternativeConstraintOverlap" | "AlternativeConstraintNotFound",
			alternativeConstraintError[self, c, #]&
		]
	]


solveOne0[self_, c_] :=
	Module[{sub = Null, newCons = {}},
		(*
		 Don't solve if this is an equal constraint with lhs and rhs the same.
		*)
		If[ !(EqualConstraintQ[c] && c["lhs"]["id"] === c["rhs"]["id"]),
			{sub, newCons} = c["solve", self]];
        If[ sub === Null || sub["isEmpty"], (* optimization *)
        	self["constraintData"]["deleteLink", c];
        	insertConstraints[self, newCons];
        	,
	        self["substitution"]["composeIterated", sub];
	        self["constraintData"]["applySubstitution", c, self["substitution"]];
        	self["constraintData"]["deleteLink", c];
        	insertConstraints[self, newCons];
	        self["substitution"]["apply", self["predicatesToProve"]];
	        self["substitution"]["apply", self["assumedPredicates"]];
        ];
        If[Lookup[self["options"], "MonitorSolve", False],
            Print[c["monitorForm", sub, self["constraintData"]["getUnsolved"]]]
        ];
        self["substitution"]
    ] 


insertConstraint[ self_, cons_] :=
	If[ !(EqualConstraintQ[cons] && cons["lhs"]["id"] === cons["rhs"]["id"]),
		If[ cons["hasProperty", "fixedError"],
				self["constraintData"]["insertBack", cons],
				self["constraintData"]["insert", cons]]]
			
insertConstraints[ self_, consList_] :=
	Scan[insertConstraint[self, #]&, consList]

sameQ[a_, b_] := a["sameQ", b];



createData[ self_] :=
	self["setConstraintData", CreateConstraintData[ self["constraints"]["get"]]]




(**************************************************)

icon := Graphics[Text[
	Style["CON\nST",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[cset_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"ConstraintSolveState",
		cset,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["length: ", {90, Automatic}], cset["constraints"]["length"]}]
  		},
  		{
  		    Framed[
	  		    Column[
                    cset["constraints"]["toList"]
	  		    ],
	  		    Background -> White
	  		]
  		},
  		fmt
  	]


toString[env_] := "ConstraintSolveState[<>]"



format[self_, shortQ_:True] :=
	Map[ #["format", shortQ]&, self["constraints"]["get"]]




End[]

EndPackage[]

