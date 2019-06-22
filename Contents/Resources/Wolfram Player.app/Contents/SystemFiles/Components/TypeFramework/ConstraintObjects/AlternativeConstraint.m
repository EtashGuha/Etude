
BeginPackage["TypeFramework`ConstraintObjects`AlternativeConstraint`"]

CreateAlternativeConstraint
AlternativeConstraintQ

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`"]
Needs["TypeFramework`Inference`Unify`"]
Needs["TypeFramework`ConstraintObjects`ConstraintBase`"]
Needs["TypeFramework`ConstraintObjects`GeneralizeConstraint`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["TypeFramework`TypeObjects`TypeQualified`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`TypeObjects`TypeRecurse`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`Inference`Substitution`"]
Needs["TypeFramework`Utilities`TypeOrder`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]


generalizeWithPredicates = TypeFramework`ConstraintObjects`GeneralizeConstraint`Private`generalizeWithPredicates

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
AlternativeConstraintClass = DeclareClass[
    AlternativeConstraint,
    <|
        "active" -> (active[Self]&),
        "solve" -> (solve[Self, ##]&),
        "computeFree" -> Function[{}, computeFree[Self]],
        "monitorForm" -> (monitorForm[Self, ##]&),
        "unresolve" -> (unresolve[Self]&),
        "format" -> (format[Self,##]&),
        "toString" -> Function[{}, toString[Self]],
        "toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
    |>,
    {
    	"id",
        "monomorphic",
        "initial",
        "type",
        "alternatives"
    },
    Predicate -> AlternativeConstraintQ,
    Extends -> BaseConstraintClass
];
RegisterConstraint[ AlternativeConstraint];
]]

CreateAlternativeConstraint[type_, initial_, alternatives_?ListQ] :=
    CreateAlternativeConstraint[type, initial, alternatives, {}]
CreateAlternativeConstraint[type_, initial_, alternatives_, m_?ListQ] :=
    CreateObject[AlternativeConstraint, <|
    	"id" -> GetNextConstraintId[],
        "monomorphic" -> m,
        "type" -> type,
        "initial" -> initial,
        "alternatives" -> alternatives
    |>]


computeFree[self_] :=
	self["type"]["free"]


active[self_] :=
    Values[
        Join @@ Map[#["free"]&, Append[self["alternatives"], self["type"]]]
    ]


sameQ[a_, b_] :=
    a["sameQ", b];


predicateForm[st_, m_, ty_] := ty;
predicateForm[st_, m_, ty_?TypeArrowQ] :=
    generalizeWithPredicates[
        <|
            "ConstraintSolveState" -> st,
            "TypeEnvironment" -> st["typeEnvironment"]
        |>,
        m,
        ty
    ];


instantiateIfPossible[st_, t_?TypeForAllQ] := 
	With[ {
		nt = t["instantiate", "TypeEnvironment" -> st["typeEnvironment"]]
	},
		nt["cloneProperties", t];
		nt
	]
	
instantiateIfPossible[_, t_] := t;

getType[ ty_?TypeQualifiedQ] :=
    ty["type"]

getType[ ty_] :=
    ty

getPredicates[ ty_?TypeQualifiedQ] :=
    ty["predicates"]

getPredicates[ ty_] :=
    {}


isCandidateWork[st_, ty1In_, ty2In_] :=
    Module[{ty1, ty2, preds1, preds2, preds, subs, tyRes},
        ty1 = getType[ty1In];
        ty2 = getType[ty2In];
        subs = TypeUnify[
                <| "ConstraintSolveState" -> st |>,
                ty1,
                ty2,
                "TypeEnvironment" -> st["typeEnvironment"]
        ];

        preds1 = getPredicates[ty1In];
        preds2 = getPredicates[ty2In];
        preds = subs["apply", Join[preds1, preds2]];
        preds = st["typeEnvironment"]["abstracttypes"]["contextReduce", preds];

        ty2 = subs["apply", ty2];

        tyRes = CreateTypeQualified[preds, ty2];
        tyRes["cloneProperties", ty2In];
        <| "type" -> tyRes, "substitutions" -> subs|>
    ]


isCandidateWorkRecurse[st_, ty1In_, ty2In_, recurseIn_, symbol_] :=
    Module[{ty1, ty2, preds1, preds2, preds, subs, recurse, tyRes},
        ty1 = getType[ty1In];
        ty2 = getType[ty2In];
        subs = TypeUnify[
                <| "ConstraintSolveState" -> st |>,
                ty1,
                ty2,
                "TypeEnvironment" -> st["typeEnvironment"]
        ];

        preds1 = getPredicates[ty1In];
        preds2 = getPredicates[ty2In];
        preds = subs["apply", Join[preds1, preds2]];
        preds = st["typeEnvironment"]["abstracttypes"]["contextReduce", preds];

        ty2 = subs["apply", ty2];
        recurse = subs["apply", recurseIn];
        tyRes = CreateTypeQualified[preds, ty2];
        tyRes["cloneProperties", ty2In];
        <| "type" -> tyRes, "recurse" -> recurse, "symbol" -> symbol|>
    ]


isCandidateQ[st_, ty_, candidate0_?TypeRecurseQ] :=
	Module[ {tmp, type, recurse},
		tmp = candidate0["instantiate", st["typeEnvironment"]];
		type = tmp["type"];
		type["cloneProperties", candidate0];
		recurse = tmp["recurse"];
        CatchTypeFailure[
        	Append[ isCandidateWorkRecurse[ st, ty, type, recurse, candidate0["symbol"]],  {"original" -> candidate0, "monomorphic" -> tmp["monomorphic"]}],
            _,
            Missing[candidate0]&
        ]
	]


(*
 Handle an error from trying the branches of the Alternatives.  The general case is 
 that Missing is returned, which causes this branch not to be considered.  However, 
 if an OccursCheck failure is found, then we reject the entire alternatives.  Solving 
 another constraint might make it easier to resolve this.   
 
 An example that behaved badly here was
 
 Function[Typed[arg, TypeSpecifier["PackedArray"]["Real64", 1]], 
                             Fold[If[#1 > 0.0, #1 + #2, #1] &, 0.0, arg]]
                             
 which lead to an Occurs failure which led to the Alternatives choosing a bad branch.
*)

handleError[ st_, ty_, candidate_, ex:TypeError[Failure["TypeUnificationOccursCheck", __]] ] :=
	ex

handleError[ st_, ty_, candidate_, ex_] :=
	Missing[candidate]

isCandidateQ[st_, ty_, candidate0_] :=
    With[{
        candidate = instantiateIfPossible[st, candidate0]
    },
        CatchTypeFailure[
            Append[ isCandidateWork[st, ty, candidate], "original" -> candidate0],
            _,
            handleError[st, ty, candidate0, #]&
        ]
    ];

solve[self_, st_] :=
    Module[{ty, alternatives, predTy, candidates, newSubs, newCons, partialSubs, hasError},
        ty = self["type"];
        alternatives = self["alternatives"];

        predTy = instantiateIfPossible[st, predicateForm[st, self["monomorphic"], ty]];
        candidates = DeleteMissing[isCandidateQ[st, predTy, #]& /@ alternatives];
        (*
         If hasError is true, we reject all the branches of the Alternatives.
        *)
        hasError = MemberQ[candidates,_TypeError];
        Which[
            Length[candidates] === 0 || hasError,
                makeFailure[
                	self,
                    "AlternativeConstraintNotFound",
                    "no satisfiable candidate of the type `1` was found in the alternative list `2`",
                    {ty, alternatives}
                ],
            Length[candidates] === 1,
                newCons = processSingleCandidate[self, st, ty, First[ candidates]];
                newSubs = CreateTypeSubstitution["TypeEnvironment" -> st["typeEnvironment"]];
            ,
            True, (* More than one possible match, try to get the most specific *)
                With[{
                    tys = Lookup[#, "type"]& /@ candidates,
                    origTys = Lookup[#, "original"]& /@ candidates
                },
                With[{ (**
                         * get the ordering of the types . Ordering will raise an error if an Indeterminate case
                         * (aka a case where there is no clear less than or greater than type)  is encountered
                         *)
                    order1 = CatchTypeFailure[TypeOrdering[st["typeEnvironment"], tys], _, {}&]
                },
                With[ {
                	order = If[ order1 === {}, CatchTypeFailure[TypeOrdering[st["typeEnvironment"], origTys], _, {}&], order1]
                },
                    If[order =!= {},
                    	newSubs = CreateTypeSubstitution["TypeEnvironment" -> st["typeEnvironment"]];
                    	newCons = processSingleCandidate[self, st, ty, First[ Part[candidates, order]]];
                        ,
                       
                       	partialSubs = getPartialSubstitution[ self, st, candidates, predTy, ty]; 
                       	If[ partialSubs =!= Null,
             				newSubs = partialSubs;
             				(*
             				  It's possible that applying the subs here is not a good idea
             				  and that it should be done later to make sure that the redirects
             				  get fixed up.
             				*)
             				newCons = {newSubs["apply", self]};
             				,
                                 
	                        (* Return a type failure, since we cannot determin the order of multiple definitions *)
		                    makeFailure[
		                    	self,
			                    "AlternativeConstraintOverlap",
			                    "multiple satisfiable instances of the type `1` found in the list of alternatives `2`",
			                    {ty, alternatives}
			                ]]
                    ]
                ]]]

        ];
        {newSubs, newCons}
    ]


(*
  candidates are all feasible choices each of which could unify with the test type, but there is insufficient information to pick one over the others.
  Despite this there might be some information in each unification that can be used.  The results are stored in the "substitutions" field of each candiate.
  Go through and find all those that have RHSs that are equal and free of any variables.
  These substitutions need to be applied to predTy which must then be unified with ty,  anything that comes out can be fed 
  back to the solver to be used.
  
  If the constraint already has a fixedError property then don't do this.  This meshes with the error recovery mechanism 
  in ConstraintSolveState.
  
  An example of this is 
  	Function[{Typed[arg1, "NumberArray"["Real64", 1]]}, Map[# + 1.2 &, arg1]]
  
  This generates constraints
  	Lookup[Map,v4708]
	v4710 = v4707
	v4708 = (v4709,N[R,1:I])->v4710
	v4706 = v4709
	v4705 = (N[R,1:I])->v4707
	Lookup[Plus,v4715]
	v4717 = v4714
	v4715 = (v4716,R)->v4717
	v4706 = (v4713)->v4714
	v4713 = v4716
	
	another example is 
	Function[{Typed[x, "NumberArray"["Complex"["Real64"], 1]]}, Map[Re, x]]
	
	this needs info from the Map constraint to help solve the Re constraint

*)

getPartialSubstitution[ self_, st_, candidates_, predTy_, ty_] :=
	Module[ {subsList, firstSub, restSubs, predTySubs, subNew},
		If[ self["getProperty", "fixedError", 0] > 0,
			Return[ Null]];
		subsList = Map[Lookup[#, "substitutions", CreateTypeSubstitution["TypeEnvironment" -> st["typeEnvironment"]]]&, candidates];
		(*
		 Now go through each element of substitutions,  we should return any that are all consistent.
		 Consistent means that the LHS are the same for all substitutions and the RHS are equal up to variables.
		*) 
		firstSub = First[subsList];
		restSubs = Rest[subsList];
		
		Scan[ mergePartials[st, firstSub, #]&, restSubs];
		If[ firstSub["isEmpty"],
			Return[Null]];
				
		predTySubs = stripQualified[firstSub["apply", predTy]];
		subNew = TypeUnifyCatch[
                <| "ConstraintSolveState" -> st |>,
                predTySubs,
                ty,
                "TypeEnvironment" -> st["typeEnvironment"]
        ];
        If[ subNew === Null,
        	Return[Null]];
        self["setProperty", "fixedError" -> 1];
        subNew
	]

getType[subs_, var_, def_] :=
	Module[ {},
		1
	]

mergePartials[ st_, subs_, oSubs_] :=
	Scan[ 
		Module[{ty = subs["lookupType", #, Null], oTy = oSubs["lookupType", #, Null], var, nTy},
				nTy = 
					If[ oTy === Null,
						Null
						,
						FindGeneralUnifierCatch[
                			<| "ConstraintSolveState" -> st |>,
                			ty,
                			oTy,
                			"TypeEnvironment" -> st["typeEnvironment"]
        				]];
        		var = subs["varMap"]["lookup", #];
        		If[ nTy === Null || TypeVariableQ[nTy],
        			subs["drop", var];
        			,
					subs["add", var, nTy]];
			]&, subs["varMap"]["keys"]]


stripQualified[ ty_?TypeQualifiedQ] :=
	If[ ty["predicates"] === {}, ty["type"], ty]
	
stripQualified[ ty_] :=
	ty
	

(*
  We have a viable candidate,  add the appropriate constraints.
*)
processSingleCandidate[ self_, st_, ty_, candidate_] :=
	With[ {
		type = candidate["type"],
		monomorphic1 = self["monomorphic"],
		isGeneralize = self["hasProperty", "generalize"] && TrueQ[self["getProperty", "generalize"]]
	},
	With[ {
		monomorphic = If[ KeyExistsQ[candidate, "recurse"], Join[ monomorphic1, candidate["monomorphic"]], monomorphic1]
	},
	Module[{
		constraints = If[isGeneralize,
	                    st["prependImplicitInstanceConstraint", ty, type, monomorphic],
	                    {st["prependEqualConstraint", ty, type]}]
	},
		self["initial"]["setProperty", "resolvedType" -> type];
		If[ KeyExistsQ[candidate, "recurse"],
			constraints = Prepend[constraints,st["prependLookupConstraint", candidate["symbol"], candidate["recurse"], monomorphic]]];
		Map[#["setProperty", "source" -> self]&, constraints];
		constraints
    ]]]


(*
Save the $ContextPath at package load time, and use it when calling ToString
This will allow ToString to stringify symbols as, e.g., "AbstractType" instead of the full-qualified "TypeFramework`AbstractType"
*)
$contextPathAtLoadTime = $ContextPath


makeFailure[self_, tag_, txtIn_, argsIn_] :=
	Module[ {lookupName = self["getProperty", "lookupName", Null], txt = txtIn, args = argsIn},
		If[ lookupName =!= Null,
			txt = StringJoin[ "Function `", ToString[Length[args]+1], "`: ", txt];
			args = Append[args, lookupName]
		];
        Block[{$ContextPath = $contextPathAtLoadTime}, args = ToString /@ args];
		TypeFailure[
            tag,
            txt,
            Sequence @@ args
        ]
	]


monitorForm[ self_, sub_, rest_] :=
    ConstraintSolveForm[<|
       "name0" -> ("Alternative"&),
       "type" -> self["type"],
       "alternatives" -> self["alternatives"],
       "monomorphic" -> self["monomorphic"],
       "unify" -> sub,
       "rest" -> rest
    |>]

icon := icon =  Graphics[Text[
    Style["ALT\nCONS",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
];

toBoxes[self_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "AlternativeConstraint",
        self,
        icon,
        {
            BoxForm`SummaryItem[{Pane["type: ",        {90, Automatic}], self["type"]}],
            BoxForm`SummaryItem[{Pane["alternatives: ",{90, Automatic}], Framed[
                Column[
                    self["alternatives"]
                ],
                Background -> White
            ]}],
            BoxForm`SummaryItem[{Pane["monomorphic: ", {90, Automatic}], self["monomorphic"]}]
        },
        {

        },
        fmt
    ]


toString[self_] :=
    StringJoin[
       "Alternative[",
       self["type"]["toString"],
       ", ",
       Riffle[#["toString"]& /@ self["alternatives"], " | "],
       "]"
    ];

unresolve[self_] :=
    AlternativeConstraint[<|
        "type" -> self["type"]["unresolve"],
        "monomorphic" -> (#["unresolve"]& /@ self["monomorphic"]["unresolve"]),
        "initial" -> self["initial"]["unresolve"],
        "alternatives" -> (#["unresolve"]& /@ self["alternatives"])
    |>]

format[self_, shortQ_:True] :=
    With[{
        alts = #["format", shortQ]& /@ self["alternatives"]
    },
	    Row[{
	       "Alternative[",
	       self["type"]["format", shortQ],
	       ", ",
	       Riffle[alts, " | "],
	       "]"
	    }]
    ]

End[]

EndPackage[]
