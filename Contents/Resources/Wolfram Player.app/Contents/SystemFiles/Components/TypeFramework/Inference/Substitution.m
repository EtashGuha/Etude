
BeginPackage["TypeFramework`Inference`Substitution`"]


CreateTypeSubstitution
TypeSubstitutionQ
SequenceSubstitution

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`TypeObjects`TypeLiteral`"]
Needs["TypeFramework`TypeObjects`TypeEvaluate`"]
Needs["TypeFramework`TypeObjects`TypeQualified`"]
Needs["TypeFramework`TypeObjects`TypePredicate`"]
Needs["TypeFramework`TypeObjects`TypeProjection`"]
Needs["TypeFramework`TypeObjects`TypeRecurse`"]
Needs["TypeFramework`TypeObjects`TypeAssumption`"]
Needs["TypeFramework`TypeObjects`TypeSequence`"]
Needs["TypeFramework`ConstraintObjects`EqualConstraint`"]
Needs["TypeFramework`ConstraintObjects`GeneralizeConstraint`"]
Needs["TypeFramework`ConstraintObjects`InstantiateConstraint`"]
Needs["TypeFramework`ConstraintObjects`LookupConstraint`"]
Needs["TypeFramework`ConstraintObjects`ProveConstraint`"]
Needs["TypeFramework`ConstraintObjects`SkolemConstraint`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["TypeFramework`ConstraintObjects`AlternativeConstraint`"]
Needs["TypeFramework`ConstraintObjects`AssumeConstraint`"]
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`Inference`Unify`"]



(*
  Code to work with Type substitutions.  This creates, applies and manipulates substitutions.
  Substitutions have a lhs of a variable and a rhs of a type.
  Fields include 
  	varMap:  varId to variable
  	typeMapFree:  varId to Type, type has free variables
  	typeMapNoFree:  varId to Type, type has no free variables
*)

$NextId

If[!ValueQ[$NextId],
	$NextId = 1
]

GetNextId[] :=
	$NextId++



$MaxIterationCount = 1024


RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeSubstitutionClass = DeclareClass[
	TypeSubstitution,
	<|
	    "isEmpty" -> (isEmpty[Self]&),
		"agree" -> Function[{cons}, agree[Self, cons]],
		"applySelf" -> (applySelf[Self]&),
		"apply" -> (applyTop[Self, ##]&),
		"add" -> Function[{var, ty}, add[Self, var, ty]],
		"drop" -> (drop[Self, ##]&),
		"lookupType" -> Function[{var, def}, lookupType[Self, var, def]],
		"clone" -> (clone[Self]&),
		"compose" -> Function[{subs}, compose[Self, subs]],
		"composeIterated" -> Function[{subs}, composeIterated[Self, subs]],
		"free" -> Function[{}, free[Self]],
		"format" -> (format[Self,##]&),
        "toAssociation" -> Function[{}, toAssociation[Self]],
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"id",
		"typeMapFree",
		"typeMapNoFree",
		"varMap",
        "properties"
	},
	Predicate -> TypeSubstitutionQ,
    Extends -> {
        ClassPropertiesTrait
    }
]
]]


defaultOptions = <|
    "TypeEnvironment" -> Undefined
|>;

Options[CreateTypeSubstitution] = Normal[defaultOptions];

CreateTypeSubstitution[] :=
    CreateTypeSubstitution["TypeEnvironment" -> Undefined];

CreateTypeSubstitution["TypeEnvironment" -> val_] :=
    CreateTypeSubstitution[<||>, "TypeEnvironment" -> val];

CreateTypeSubstitution[var_?TypeVariableQ -> ty_, opts:OptionsPattern[]] :=
	CreateTypeSubstitution[<| var -> ty |>, opts];

CreateTypeSubstitution[assoc_?AssociationQ, iopts:OptionsPattern[]] :=
    With[{
        opts = Association[iopts],
        grps = GroupBy[ assoc, Length[#["free"]] > 0&]
    },
    With[{
        tenv = Lookup[opts, "TypeEnvironment", Undefined],
        free = Lookup[ grps, True, <||>],
        nofree = Lookup[ grps, False, <||>]
    },
        If[tenv =!= Undefined && !TypeEnvironmentQ[tenv],
            TypeFailure[
		        "CreateTypeSubstitution",
		        "the envrionment `1` passed while creating the substitution is not a valid type environment.",
		        tenv
		    ]
        ];
	    CreateObject[TypeSubstitution,
	        <|
	        	"id" -> GetNextId[],
	            "typeMapFree" -> CreateReference[Association[KeyValueMap[Function[{var, ty}, var["id"] -> ty], free]]],
	            "typeMapNoFree" -> CreateReference[Association[KeyValueMap[Function[{var, ty}, var["id"] -> ty], nofree]]],
	            "varMap" -> CreateReference[Association[KeyValueMap[Function[{var, ty}, var["id"] -> var], assoc]]],
	            "properties" -> CreateReference[<|
	                "TypeEnvironment" -> tenv
	            |>]
	        |>
	    ]
    ]];


isEmpty[self_] :=
    self["varMap"]["keys"] === {};

add[ self_, var_, ty_] := 
	(
	If[Length[ty["free"]] > 0,
		self["typeMapFree"]["associateTo", var["id"] -> ty],
		self["typeMapNoFree"]["associateTo", var["id"] -> ty]];
	self["varMap"]["associateTo", var["id"] -> var];
	);

drop[self_, var_?TypeObjectQ] := (
    self["typeMapFree"]["keyDropFrom", var["id"]];
    self["typeMapNoFree"]["keyDropFrom", var["id"]];
    self["varMap"]["keyDropFrom", var["id"]];
);

drop[self_, vars_?ListQ] :=
    drop[self, #]& /@ vars;

(*
  Use this for apply on TypeVariable
*)
lookupType[self_, varId_, def_] :=
Module[{res =  self["typeMapFree"]["lookup", varId, Null]},
		If[res === Null,
			self["typeMapNoFree"]["lookup", varId, def]
			,
			res]
	]

clone[self_] :=
    With[{
        sub = CreateTypeSubstitution[]
    },
        sub["setTypeMapFree", self["typeMapFree"]["clone"]];
        sub["setTypeMapNoFree", self["typeMapNoFree"]["clone"]];
        sub["setVarMap", self["varMap"]["clone"]];
        sub["setProperties", self["properties"]["clone"]];
        sub
    ]

toAssociation[self_] :=
    Association[
        Table[
            self["varMap"]["lookup", key] -> self["lookupType", key, Null],
            {key, self["varMap"]["keys"]}
        ]
    ]




(*
  Note this uses getFree because SequenceSubsitution is an Expr,  if it was an
  object,  like a type variable this would be neater.
*)
free[self_] :=
	Join @@ Map[
	   getFree[#]&,
	   self["typeMapFree"]["values"]
	]


applyTop[ self_, arg_, opts_:<||>] :=
	Which[
		self["typeMapFree"]["isEmpty"] && self["typeMapNoFree"]["isEmpty"],
			arg,
		Lookup[ opts, "Iterated", False],
        	applyIterated[self, arg, opts],
        ListQ[arg],
        	Map[apply[self,#,opts]&, arg],
        True,
        	apply[self, arg, opts]
   ]

apply[ self_, args_?ReferenceQ, opts_:<||>] := (
    (* args is a list reference *)
    Do[
        args["setPart", ii, apply[self, args["getPart", ii]]],
        {ii, args["length"]}
    ];
    args
);

apply[ self_, arg_, opts_:<||>] :=
	If[ Length[arg["free"]] === 0,
		arg,
	With[{
	   s = apply0[self, arg]
	},
	If[ s["id"] === arg["id"],
		s,
		applyExtra[self,s]]
	]]


(*
  The extra steps for apply,  these are split into a separate function 
  so that they can be invoked separately.
*)
applyExtra[ self_, arg_] :=
		With[{
	   		v = tryToDestructTypeLiteral[self, arg]
		},
		With[{
	   		a = resolveAlias[self, v]
		},
	   		executeTypeEvaluate[self, a]
		]]


(*
  apply1 is a worker for apply0.  It calls apply0 on the argument.
  If there was a change it calls applyExtra on the new result. It
  returns {True, newResult} if there was a change, or {init, arg} 
  if there was not.  init is passed in as an argument in case another 
  argument in the construct calling apply had a change in an argument.
*)
apply1[ self_, init_, arg_] :=
	Module[ {tmp},
		tmp = apply0[self, arg];
		If[ tmp["id"] =!= arg["id"],
			{True, applyExtra[self, tmp]},
			{init, arg}]
		]

apply1[ self_, init_, args_List] :=
	Module[ {changed = init, tmp, res},
		res = 
			Map[
				(
				tmp = apply0[self, #];
				If[tmp["id"] =!= #["id"],
					changed = True;
					tmp = applyExtra[self, tmp]];
				tmp
				)&, args];
				
		{changed, res}
	]


executeTypeEvaluate[self_, a_?TypeEvaluateQ] :=
	a["execute", self["getProperty", "TypeEnvironment", Undefined]]
	
executeTypeEvaluate[self_, a_] :=
	a
	

(* example: replace TypeLiteral[Anything, "Integer"] with "Integer" *)
tryToDestructTypeLiteral[self_, ty_?TypeLiteralQ] :=
    If[TypeObjectQ[ty["value"]] && TypeTopQ[ty["value"]],
        ty["type"],
        ty
    ];

tryToDestructTypeLiteral[self_, ty_] := ty;

resolveAlias[self_, ty_?TypeObjectQ] :=
    Which[
        Length[ty["free"]] =!= 0, (* Maybe too strict *)
            ty,
        !self["hasProperty", "TypeEnvironment"] ||
        self["getProperty", "TypeEnvironment"] === Undefined,
            ty,
        True,
            With[{
                uty = ty["unresolve"],
                tenv = self["getProperty", "TypeEnvironment"]
            },
                (*If[tenv["typeAliases"]["keyExistsQ", uty],
                    Print[ty["unresolve"] -> tenv["typeAliases"]["lookup", uty]];
                ];*)
                tenv["typeAliases"]["lookup", uty, ty]
            ]
    ];

resolveAlias[self_, ty_] :=
    ty

(*
  Iterate the application of substitution until
  the free variables don't change.  Also have a count
  limit to throw an error in case the substitutions don't
  converge (but this shouldn't happen).
*)

getFree[SequenceSubstitution[vars_]] :=
	Join @@ Map[
	   #["free"]&,
	   vars
	]

getFree[arg_] :=
	arg["free"]

applyIterated[ self_, arg_, _] :=
	Module[ {oldArg, oldFree, newArg, newFree, cnt = 0},
		oldArg = arg;
		oldFree = getFree[oldArg];
		newArg = apply[self, oldArg];
		newFree = getFree[newArg];
		While[ Length[ KeyComplement[ {oldFree, newFree}]] > 0,
			oldFree = newFree;
			newArg = apply[ self, newArg];
			newFree = oldFree;
			If[ cnt++ > $MaxIterationCount,
				TypeFailure[
				    "ApplyRepeatedSubstitution",
				    "Substitution iteration limit exceeded while processing `1` with `2`",
				    self, arg
				]
			];
		];
		newArg
	]


applySelf[ self_] :=
	Module[ {cnt = 0, loop = True},
		While[ loop,
			loop = applySelf1[ self];
			If[ cnt++ > $MaxIterationCount,
				TypeFailure[
				    "ApplyRepeatedSubstitution",
				    "Substitution iteration limit exceeded while processing `1`",
				    self
				]
			];
		];
		self
	]


applySelf1[ self_] :=
	Module[ {var, type, key, changed, updatedQ = False},
		Do[
			type = self["typeMapFree"]["lookup", key];
			{changed, type} = apply1[ self, False, type];
			If[changed,
				updatedQ = True;
				var = self["varMap"]["lookup", key];
				self["drop", var];
				self["add", var, type]],
			{key, self["typeMapFree"]["keys"]}
		];
		updatedQ
	]


$applyClassHandlers = 
<|
	TypeSubstitution -> applyTypeSubstitution,
    EqualConstraint -> applyEqualConstraint,
	LookupConstraint -> applyLookupConstraint,
	InstantiateConstraint -> applyInstantiateConstraint,
	GeneralizeConstraint -> applyGeneralizeConstraint,
	ProveConstraint -> applyProveConstraint,
	AssumeConstraint -> applyAssumeConstraint,
	AlternativeConstraint -> applyAlternativeConstraint,
	SkolemConstraint -> applySkolemConstraint,
	TypeVariableClassName -> applyTypeVariable,
	TypeConstructorObject -> applyTypeConstructor,
	TypeArrow -> applyTypeArrow,
	TypeApplication -> applyTypeApplication,
	TypeLiteralObject -> applyTypeLiteral,
	TypeProjectionObject -> applyTypeProjection,
	TypeEvaluateObject -> applyTypeEvaluate,
	TypeSequence -> applyTypeSequence,
	TypeForAll -> applyTypeForAll,
	TypeQualified -> applyTypeQualified,
	TypePredicate -> applyTypePredicate,
	TypeAssumption -> applyTypeAssumption,
	TypeRecurseObject -> applyTypeRecurse
|>


applySequenceSubstitution

apply0[ self_, arg_] :=
	Module[{fun},
		fun =
			Which[
				ObjectInstanceQ[arg],
					Lookup[$applyClassHandlers, arg["_class"], applyUnknown]
				,
				MatchQ[arg, SequenceSubstitution[_]],
					applySequenceSubstitution
				,
				True,
					applyUnknown
					];
		fun[self,arg]
	]

applyUnknown[ self_, arg_] :=
	ThrowException[{"Unknown argument to apply", arg}]


(*
  TODO revisit this
*)
applyTypeSubstitution[ self_, arg_] :=
	Module[ {ef, var, type, key},
		ef = CreateTypeSubstitution[
		  "TypeEnvironment" -> self["getProperty", "TypeEnvironment", Undefined]
		];
		Do[
			var = arg["varMap"]["lookup", key];
			type = arg["typeMapFree"]["lookup", key];
			type = self["apply", type];
			ef["add", var, type],
			{key, arg["typeMapFree"]["keys"]}
		];
		Do[
			var = arg["varMap"]["lookup", key];
			type = arg["typeMapNoFree"]["lookup", key];
			ef["add", var, type],
			{key, arg["typeMapNoFree"]["keys"]}
		];
		ef
	]


applyEqualConstraint[self_, c_] :=
	Module[ {changed, nLhs, nRhs},
		{changed, nLhs} = apply1[ self, False, c["lhs"]];
		{changed, nRhs} = apply1[ self, changed, c["rhs"]];
		If[changed,
    		With[{
        		new = CreateEqualConstraint[nLhs, nRhs]
    		},
        		new["setProperties", c["properties"]["clone"]];
        		new
    		],
    		c
		]]

applyLookupConstraint[self_, c_] :=
	Module[ {changed, nType, nMono},
		{changed, nType} = apply1[ self, False, c["type"]];
		{changed, nMono} = apply1[ self, changed, c["monomorphic"]];
		If[changed,
			With[{
        		    new = CreateLookupConstraint[ c["name"], nType, nMono]
			},
				new["setInitial", c["initial"]];
	        		new["setProperties", c["properties"]["clone"]];
	        		new
			],
			c
	   ]
  	]
	
	
applyInstantiateConstraint[self_, c_] :=
	Module[{changed, nTau, nRho},
		{changed, nTau} = apply1[self, False, c["tau"]];
		{changed, nRho} = apply1[self, changed, c["rho"]];
		If[changed,		
    		With[{
        		new = CreateInstantiateConstraint[nTau, nRho]
    		},
        		new["setProperties", c["properties"]["clone"]];
        		new
    		],
    		c
		]
	]

applyGeneralizeConstraint[self_, c_] :=
	Module[ {changed, nSigma, nTau, nMono},
		{changed, nSigma} = apply1[self, False, c["sigma"]];
		{changed, nTau} = apply1[self, changed, c["tau"]];
		{changed, nMono} = apply1[self, changed, c["monomorphic"]];
		If[changed,	
    		With[{
        		new = CreateGeneralizeConstraint[nSigma, nTau, nMono]
    		},
        		new["setProperties", c["properties"]["clone"]];
        		new
    		],
    		c]
	]

applyProveConstraint[self_, c_] :=
	Module[ {changed, nPred},
		{changed, nPred} = apply1[self, False, c["predicate"]];
		If[changed,
    		With[{
        		new = CreateAssumeConstraint[nPred]

   			 },
        		new["setProperties", c["properties"]["clone"]];
        		new
    		]
    		,
    		c]
	]

applyAssumeConstraint[self_, c_] :=
	Module[ {changed, nPred},
		{changed, nPred} = apply1[self, False, c["predicate"]];
		If[changed,
    		With[{
        		new = CreateAssumeConstraint[nPred]

   			 },
        		new["setProperties", c["properties"]["clone"]];
        		new
    		]
    		,
    		c]
	]


applyAlternativeConstraint[self_, c_] :=
	Module[ {changed, nType, nAlts, nMono},
		{changed, nType} = apply1[self, False, c["type"]];
		{changed, nAlts} = apply1[self, changed, c["alternatives"]];
		{changed, nMono} = apply1[self, changed, c["monomorphic"]];
		If[changed,
    		With[{
        		new = CreateAlternativeConstraint[nType, c["initial"], nAlts, nMono]
    		},
        		new["setProperties", c["properties"]["clone"]];
        		new
    		],
    		c]
    ]

applySkolemConstraint[self_, c_] :=
    todoapply0[self, c]

todoapply0[self_, c_] :=
    TypeFailure[
        "SubstitutionApplicationUndefined",
        "the apply substitution method for the class `1` has not been defined",
        SymbolName[c["_class"]]
    ]


applyTypeVariable[self_, ty_] :=
	Module[{res =  self["typeMapFree"]["lookup", ty["id"], Null]},
		If[res === Null,
			self["typeMapNoFree"]["lookup", ty["id"], ty]
			,
			res]
	]

applyTypeConstructor[self_, ty_] :=
    ty



(*
 When we apply a substitution to the arguments of a TypeArrow we have
 check if the incoming values are carrying in a sequence (from TypeSequence).
 If we are then we need to flatten out the values into the arguments.  For example
 take,  {Integer, var1} -> Boolean with the substitution var -> Sequence[Integer, Integer].
 The result should be {Integer, Integer, Integer} -> Boolean
*)

isSequence[ SequenceSubstitution[arg_]] :=
	True

isSequence[ arg_] :=
	False

flattenSequence[ args_] :=
	Fold[Join[#1, If[isSequence[#2], First[#2],{#2}]]&, {}, args]


applyTypeArrow[self_, ty_] :=
	Module[ {changed, newArgs, newRes, ef},
		{changed, newArgs} = apply1[ self, False, ty["arguments"]];
		{changed, newRes} = apply1[ self, changed, ty["result"]];
		newArgs = Map[
					If[ isSequence[#],
						changed = True;
						First[#],
						#]&, newArgs];
		If[ changed,
			ef = CreateTypeArrow[Flatten[newArgs], newRes];
			ef["setProperties", ty["properties"]["clone"]];
			ef
			,
			ty]
	]




applyTypeApplication[self_, ty_] :=
	Module[ {changed, newArgs, nType, ef},
		{changed, newArgs} = apply1[ self, False, ty["arguments"]];
		{changed, nType} = apply1[ self, changed, ty["type"]];
		If[ changed,
			ef = CreateTypeApplication[nType, newArgs];
			ef["setProperties", ty["properties"]["clone"]];
			ef
			,
			ty]
	]



(*
  Perhaps this flatten out the sequence or it should be flattened out in the
  CreateArrow this came from.
*)
applySequence[ self_, ty_?TypeSequenceQ] :=
	Module[ {changed, nType, nBinding},
		{changed, nType} = apply1[ self, False, ty["type"]];
		If[ty["binding"] === None,
			nBinding = None,
			{changed, nBinding} = apply1[ self, changed, ty["binding"]]];
		If[changed,
			With[ {
				new = CreateTypeSequence[nType, nBinding, {ty["min"], ty["max"]}]
			},
				new["setProperties", ty["properties"]["clone"]];
        		new
			]
			,
			ty
		]
	]

applyTypeLiteral[ self_, ty_] :=
	Module[{changed, nType},
		{changed, nType} = apply1[self, False, ty["type"]];
		If[changed,
           With[{
                res = CreateTypeLiteral[ ty["value"], nType]
             },
                res["setProperties", ty["properties"]["clone"]];
                res
         	]
         	,
         	ty]
    ]
    
applyTypeProjection[ self_, ty_] :=
	Module[{changed, nType, nValue, res},
		{changed, nType} = apply1[self, False, ty["type"]];
		{changed, nValue} = apply1[self, changed, ty["value"]];
		res = If[changed,
			With[{nproj = CreateTypeProjection[ nType, nValue]},
				nproj["cloneProperties", ty];
				nproj
			]
         	,
         	ty
         ];
        If[!TypeVariableQ[nType],
			res = res["project"];
		];
		res
    ]



    
 applyTypeEvaluate[ self_, ty_] :=
 	Module[ {changed, nArgs},
 		{changed, nArgs} = apply1[ self, False, ty["arguments"]];
 		If[
 			changed,
    		With[ {
        		new = CreateTypeEvaluate[ ty["function"], nArgs]
    		},
				new["cloneProperties", ty];
				new
    		]
    		,
    		ty]
    ]
   
    
applyTypeSequence[ self_, ty_] :=
	If[ty["hasBinding"] && 
			(self["typeMapFree"]["keyExistsQ", ty["binding"]["id"]] || 
			 self["typeMapNoFree"]["keyExistsQ", ty["binding"]["id"]]),
		   apply[self, ty["binding"]],
		   applySequence[self, ty]
    ]


applyTypeForAll[self_, ty_] :=
    With[{
        newSubst = self["clone"]
    },
	newSubst["drop", ty["variables"]];
	Module[ {changed, nType},
		{changed, nType} = apply1[ newSubst, False, ty["type"]];
		If[changed,
			With[{
        		new = CreateTypeForAll[  ty["variables"], nType]
    		},
        		new["setProperties", ty["properties"]["clone"]];
        		new
    		],
    		ty
		]
	]]

applyTypeQualified[self_, ty_] :=
	Module[ {changed, nPreds, nType}, 
		{changed, nPreds} = apply1[self, False, ty["predicates"]];
		{changed, nType} = apply1[self, changed, ty["type"]];
		If[changed,
			With[{
        		new = CreateTypeQualified[nPreds, nType]
		  	},
        		new["setProperties", ty["properties"]["clone"]];
        		new
			]
			,
        	ty]
	]

 
applyTypePredicate[self_, ty_] :=
	Module[ {changed, nTypes}, 
		{changed, nTypes} = apply1[self, False, ty["types"]];
		If[changed,
			With[{
        		new = CreateTypePredicate[nTypes, ty["test"]]
		  	},
        		new["setProperties", ty["properties"]["clone"]];
        		new
			]
			,
        	ty]
	]


applyTypeAssumption[self_, ty_] :=
    With[{
        new = CreateTypeAssumption[
	        ty["name"],
	        apply[self, ty["type"]]
	    ]
    },
        new["setProperties", ty["properties"]["clone"]];
        new
    ]


(*
 Everything in a TypeRecurse is local and gets expanded out 
 later,  so this is a no-op.
*)
applyTypeRecurse[self_, ty_] :=
    ty




(*
  We don't care about a change for SequenceSubstitution because this 
  is always flattened into the TypeArrow which forces a new type 
  object.
*)
applySequenceSubstitution[self_, SequenceSubstitution[args_]] :=
	SequenceSubstitution[ Map[ apply[self,#]&, args]]



(*
 TODO this looks wrong
*)
agree[self_, sub_?TypeSubstitutionQ] :=
	AllTrue[
	        Intersection[
	           self["varMap"]["values"],
	           sub["varMap"]["values"],
	           SameTest -> sameQ
	        ],
	        TypeUnifiableQ[self["apply", #], sub["apply", #], "TypeEnvironment" -> self["getProperty", "TypeEnvironment", Undefined]]&
	    ]

sameQ[a_, b_] := a["sameQ", b];


composeIterated[self_, sub_?TypeSubstitutionQ] :=
	Module[ {newSub},
		self["compose", sub];
		newSub = self["applySelf"];
		newSub
	]


(* : TypeSubstitution -> TypeSubstitution -> TypeSubstitution *)
compose[self_, sub_?TypeSubstitutionQ] :=
	Module[{oldVar, oldType},
		If[!self["agree", sub],
			TypeFailure[
		        "TypeSubstitutionAgree",
		        "the two substitutions `1` and `2` do not agree",
		        self,
		        sub
		    ]
		];
		Do[
			oldVar = sub["varMap"]["lookup", key];
			oldType = sub["typeMapFree"]["lookup", key];
			addSubstitution[self, oldVar, oldType],
			{key, sub["typeMapFree"]["keys"]}
		];
		Do[
			oldVar = sub["varMap"]["lookup", key];
			oldType = sub["typeMapNoFree"]["lookup", key];
			addSubstitution[self, oldVar, oldType],
			{key, sub["typeMapNoFree"]["keys"]}
		];
		self
	]



(* : TypeSubstitution -> TypeVariable -> TypeObject -> TypeSubstitution *)
addSubstitution[ self_, var_, type_] :=
	Module[ {newVar, newType},
		(*newVar = self["apply", var];*)
		newVar = var;
		newType = self["apply", type];
		self["add", newVar, newType];
	]


(**************************************************)

icon := icon = Graphics[Text[
	Style["TYP\nSUB",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]

getTypes[self_] :=
	Join[self["typeMapNoFree"]["values"], self["typeMapFree"]["values"]]

toBoxes[self_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"TypeSubstitution",
		self,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["length: ", {90, Automatic}], Length[getTypes[self]]}]
  		},
  		{
  		    With[{
  		        typeMapKeys = self["typeMapFree"]["keys"]
  		    },
	  		    If[Length[typeMapKeys] === 0,
	  		        Nothing,
		  		    CompileInformationPanel[
		  		    	"TypeSubstitution"
		  		    	,
			  		    (*Association[
			  		        Map[
			  		            (self["varMap"]["lookup", #]["format"] -> self["typeMap"]["lookup", #]["format"])&,
			                    typeMapKeys
			  		        ]
			            ]*)
                        Map[
                            (self["varMap"]["lookup", #] -> self["lookupType", #, Null])&,
                            typeMapKeys
                        ]
			  		]
	  		    ]
  		    ]
  		},
  		fmt
  	];


toStringImpl[ SequenceSubstitution[ args_]] :=
	StringJoin[ "SequenceSubstitution[", Riffle[Map[#["toString"]&, args], ","],  "]"]

toStringImpl[ x_] :=
	x["toString"]

toStringImpl[ x_String] :=
	x


toString[self_] := "TypeSubstitution[" <>
	Riffle[Map[ {self["varMap"]["lookup", #]["toString"], "->",
		     toStringImpl[self["lookupType", #, ""]]}&,
		 self["varMap"]["keys"]], ","]  <>
	"]";

format[self_, shortQ_:True] :=
	Map[ (self["varMap"]["lookup", #]["format", shortQ] ->
		     self["lookupType", #, Null]["format", shortQ])&,
		 self["varMap"]["keys"]];

End[];

EndPackage[]
