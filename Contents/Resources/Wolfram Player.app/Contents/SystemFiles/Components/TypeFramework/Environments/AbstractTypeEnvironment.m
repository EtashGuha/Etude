
BeginPackage["TypeFramework`Environments`AbstractTypeEnvironment`"]

AbstractTypeEnvironmentQ
CreateAbstractTypeEnvironment

Begin["`Private`"]



(*

The AbstractTypeEnvironment provides support for abstract types.

classes      is a map from name to abstract type object
             it is created a abstract types are defined and created

instanceMap  is a map from abstract type name to a list of concrete type instances
             it is created at type initialization time.

HNFCache     is a cache that maps from predicates to their HNF form
             is it created as the inferencer runs


A type qualified type is defined as

Qualified[{predicates}, someType]


A type instance is similar and is stored as

Qualified[{predicates}, predicate]

a type instance records that a type is an instance of an abstract type this is used for 
resolving predicates in context reduction.

type instances are stored in each abstract type object, but these are only those instances directly
mentioned,  not those reachable by inheritance.  A map of all type instances for each abstract type
is kept in the instanceMap.

For a concrete type eg the Integer64 type the instance has the following form

For example the Integer64 type instance is stored in the AbstractType["Integers"] as

Qualified[
   {}, 
   Predicate[
       type -> "Integer64", 
       test -> MemberQ[AbstractType["Integers"]]
   ]
]

the initial list for the Qualified is empty in this case.  But if the instance has arguments which 
have constraints these might appear.

For example if you define a type such as "Native" as

Native["a"] is in AbstractType["Integers"] iff Element["a", AbstractType["Integers"]]

Then the instance is defined as the following Qualified type

Qualified[
  {
    Predicate[type -> "a", test -> MemberQ[AbstractType["Integers"]]]
  }, 
  Predicate[
     type -> "Native"["a"], 
     test -> MemberQ[AbstractType["Integers"]]
]


If I define a Pair type as


Pair["a", "b"] is in AbstractType["Integers"] iff Element["a", AbstractType["Integers"]] and Element["b", AbstractType["Integers"]]

Then the instance is defined as the following Qualified type

Qualified[
  {
    Predicate[type -> "a", test -> MemberQ[AbstractType["Integers"]]], 
    Predicate[type -> "b", test -> MemberQ[AbstractType["Integers"]]]
  }, 
  Predicate[
      type -> "Pair"["a", "b"], 
      test -> MemberQ[AbstractType["Integers"]]
]


*)

Needs["TypeFramework`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`Inference`Unify`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`TypeObjects`TypePredicate`"]
Needs["TypeFramework`TypeObjects`TypeLiteral`"]
Needs["TypeFramework`TypeObjects`TypeEvaluate`"]
Needs["TypeFramework`TypeObjects`TypeQualified`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]


RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
AbstractTypeEnvironmentClass = DeclareClass[
	AbstractTypeEnvironment,
	<|
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"scanAll" -> Function[ {data, fun}, scanAll[Self, data, fun]],
		"hasClass" -> (hasClass[Self, #]&),
        "getClass" -> (getClass[Self, ##]&),
        "addClass" -> (addClass[Self, #]&),
        "hasDefault" -> (hasDefault[Self, #]&),
        "getDefault" -> (getDefault[Self, ##]&),
        "addDefault" -> (addDefault[Self, #]&),
        "predicatesEntailQ" -> (predicatesEntailQ[Self, ##]&),
        "predicatesEntailBySuperQ" -> (predicatesEntailBySuperQ[Self, ##]&),
        "finalize" -> (finalize[Self]&),
       	"contextReduce" -> (contextReduce[Self, ##]&),
        "dispose" -> (dispose[Self]&),
        "graphForm" -> (graphForm[Self]&)
	|>,
	{
		"HNFCache",
		"instanceMap",
		"classes" ,
		"defaults"
	},
	Predicate -> AbstractTypeEnvironmentQ
]
]]

dispose[self_] :=
	(
	self["setHNFCache", Null];
	self["setInstanceMap", Null];
	self["setClasses", Null];
	self["setDefaults", Null];
	)

CreateAbstractTypeEnvironment[] :=
	CreateObject[AbstractTypeEnvironment, <|
			"classes" -> CreateReference[<||>],
            "defaults" -> CreateReference[<||>]
		|>]




scanAll[self_, data_, fun_] :=
	Scan[ fun[data, #]&, self["classes"]["values"]]




(**************************************************)

hasClass[self_, cls_?AbstractTypeQ] :=
    hasClass[self, AbstractType[cls["typename"]]];
hasClass[self_, cls_?StringQ] :=
    hasClass[self, AbstractType[cls]];
hasClass[self_, AbstractType[name_]] :=
    self["classes"]["keyExistsQ", AbstractType[name]];

getClass[self_, cls_] :=
    getClass[self, cls, cls];
getClass[self_, cls_?AbstractTypeQ, default_] :=
    If[self["hasClass", cls], cls, default];
getClass[self_, cls_?StringQ, default_] :=
    getClass[self, AbstractType[cls], default];
getClass[self_, AbstractType[name_], default_] :=
    self["classes"]["lookup", AbstractType[name], default];

addClass[self_, cls_?AbstractTypeQ] :=
    If[self["hasClass", AbstractType[cls["typename"]]],
        TypeFailure[
            "AbstractTypeEnvironmentAddClass",
            "the abstract type `1` already exists within " <>
            "the abstract type environment `2`",
            cls["typename"],
            self
        ],
        self["classes"]["associateTo",
            AbstractType[cls["typename"]] -> cls
        ]
    ];

(**************************************************)

hasDefault[self_, cls_?AbstractTypeQ] :=
    hasDefault[self, AbstractType[cls["typename"]]];
hasDefault[self_, cls_?StringQ] :=
    hasDefault[self, AbstractType[cls]];
hasDefault[self_, AbstractType[name_]] :=
    self["defaults"]["keyExistsQ", AbstractType[name]];

getDefault[self_, cls_] :=
    getDefault[self, cls, Undefined];
getDefault[self_, cls_?AbstractTypeQ, default_] :=
    If[self["hasDefault", cls],
        getDefault[AbstractType[cls["typename"]]],
        default
    ];
getDefault[self_, cls_?StringQ, default_] :=
    getDefault[self, AbstractType[cls], default];
getDefault[self_, AbstractType[name_], default_] :=
    self["defaults"]["lookup", AbstractType[name], default];


addDefault[self_, cls_?AbstractTypeQ, ty_?TypeObjectQ] :=
    addDefault[self, cls, ty];
addDefault[self_, cls_?AbstractTypeQ -> ty_?TypeObjectQ] :=
    addDefault[self, AbstractType[cls["typename"]] -> ty];
addDefault[self_, AbstractType[name_], ty_?TypeObjectQ] :=
    addDefault[self, AbstractType[name], ty];
addDefault[self_, AbstractType[name_] -> ty_?TypeObjectQ] :=
    If[self["hasDefault", AbstractType[name]],
        TypeFailure[
            "AbstractTypeEnvironmentAddDefault",
            "the default type for the abstract type `1` already exists within " <>
            "the abstract type environment `2`",
            AbstractType[name],
            self
        ],
        self["defaults"]["associateTo",
            AbstractType[name] -> ty
        ]
    ];
    
    
   
   
finalize[ self_] :=
	(
	self["setHNFCache", CreateReference[<||>]];
	createInstanceMap[self];
	)
    
 (*
  The instanceMap is a map from abstract class name to every instance that is attached 
  to the class or to classes that derived from this class (ie that have this class as a 
  super class).
 *)
createInstanceMap[self_] :=
	Module[{
		forwardMap = CreateReference[<||>],
		backwardMap = CreateReference[<||>],
		instanceMap = CreateReference[<||>]
		},
		(*
		  Build a forward map
		*)
		Scan[
			(
			backwardMap["associateTo", #["typename"] -> CreateReference[<||>]];
			addSupers[self, forwardMap, #["typename"]]
			)&, self["classes"]["values"]];
		(*
		  Fill the reverse map
		*)
		forwardMap["associationMap",
			Module[{sub = First[#], supers = Last[#]["keys"], subObj},
				subObj = self["getClass", sub];
				Scan[
					Function[super,
						Module[{table = backwardMap["lookup", super, Null]},
							(*  table should not be Null *)
							table["associateTo", sub -> subObj]
						]], supers]
			]&];
		(*
		  Now fill the instanceMap
		*)
		backwardMap["associationMap",
			Module[ {absClassName = First[#], derivedMap = Last[#], absObj, listRef},
				listRef = CreateReference[{}];
				(*  table should not be Null *)
				absObj = self["getClass", absClassName];
				joinRef[ listRef, absObj["instances"]];
				Scan[
					Function[value,
						joinRef[ listRef, value["instances"]]],  derivedMap["values"]];
				instanceMap["associateTo", absClassName -> listRef["get"]];
			]&];
		
		self["setInstanceMap", instanceMap]
	]

joinRef[ listRef1_, listRef2_] :=
	Scan[ listRef1["appendTo", #]&, listRef2["get"]]


addSupers[ self_, classMap_, className_] :=
	Module[ {classObj, ref, table, val},
		If[!classMap["keyExistsQ", className],
			classObj = self["getClass", className];
			ref = CreateReference[<||>];
			Scan[
				(
				val = #;
				addSupers[self, classMap, #];
				table = classMap["lookup", #, Null];
				(*
				  table should not be Null
				*)
				ref["join", table];
				ref["associateTo", # -> True];
				)&, classObj["supers"]["get"]];
			classMap["associateTo", className -> ref]];
	]	
	

(**************************************************)

(* Checks whether p can be deduced from ps using
 * superclasses
 *)
predicatesEntailBySuperQ[self_, preds:{___?TypePredicateQ}, pred_?TypePredicateQ] :=
    superclassEntailsQ[self, preds, pred];

(* Checks whether p can be deduced from ps using
 * superclasses and existing instances.
 *)
predicatesEntailQ[self_, preds:{___?TypePredicateQ}, pred_?TypePredicateQ] :=
    superclassEntailsQ[self, preds, pred] ||
    With[{insts = CatchTypeFailure[byInstances[self, pred], TypeError, $Failed&]},
        If[FailureQ[insts],
            False,
            entailList[self, preds, insts]
        ]
    ];
predicatesEntailQ[self_, args__] :=
    TypeFailure[
        "AbstractTypePredicateEntailsQ",
        "invalid arguments `1` passed during the predicate entails call to `2`",
        {args},
        self
    ];

entailList[self_, ps_, qs_] :=
    AllTrue[qs, self["predicatesEntailQ", ps, #]&];

superclassEntailsQ[self_, preds_, pred_] :=
  With[{
    preconditions = Catenate[Map[bySuper[self, #]&, preds]]
  },
    AnyTrue[preconditions, #["sameQ", pred]&]
  ];

(**************************************************)

(* Gets a list of predicates that must hold true for p
 * based on superclass information.
 *******************************************************
 * Given predicate `p = t \[Element] c`, find all super classes of `c` where if c' is a superclass
 * then c \[Subset] c', thus p' = t \[Element] c'
 *)
bySuper[self_, pred_?TypePredicateQ] :=
    With[{class = pred["underlyingAbstractType"]},
        DeleteDuplicates[
	        Prepend[
	            Catenate[Map[
	                bySuper[self, CreateTypePredicate[pred["types"], MemberQ[#]]]&,
	                If[self["hasClass", class],
	                    self["getClass", class]["supers"]["toList"],
	                    {}
	                ]
	            ]],
	            pred
	        ],
	        sameQ
        ]
    ];


(* Gets a list of predicates that must hold true for p based on
 * the instance for p in the class environment.
 ********************************************************
 * Given predicate `p = t \[Element] c`, find type `t` in instance store of class `c` and collect all predicates of `t`
 * If `t` has no more predicates, output []. If `t` Does not exist, then output Nothing
 *
 * In fact we only take the first one that matches,  I presume this is fine because we check that 
 * instances don't overlap.  A potential optimization would be to stop when we get one that matches.
 *)
byInstances[self_?AbstractTypeEnvironmentQ, pred_?TypePredicateQ] :=
         With[{
            insts = instancesOfClass[self, pred["underlyingAbstractType"]]
        },
        With[{
            preds = Map[try[#,pred]&, insts]
        },
            If[preds === {},
                (*
                TypeFailure[
                    "byInstances",
                    "unable to find instances for predicate `1`",
                    pred
                ];
                *)
                $Failed,
                First[preds]
            ]
        ]]



try[qual_?TypeQualifiedQ, pred_] := 
	With[{
            ps   = qual["predicates"],
            h    = qual["type"]
        },
            If[instanceMatchQ[h, pred],
            	(* 
            	  We can ignore the predicate test -- it must match that in the instance 
            	*)
                With[{
                    sub = TypeUnifyMany[h["types"], pred["types"]]
                },
                (*
                  Now with a substitution we can substitute back into the predicates of the instance.  
                  These will be the result. 
                *)
                With[{
                    r = Map[sub["apply", #]&, ps]
                },
                    r
                ]],
                Nothing
            ]
        ];



instancesOfClass[self_?AbstractTypeEnvironmentQ, class_?AbstractTypeQ] :=
	instancesOfClass[self, AbstractType[class["typename"]]]
  
  
instancesOfClass[self_?AbstractTypeEnvironmentQ, class_?StringQ] :=
  instancesOfClass[self, AbstractType[class]];

instancesOfClass[self_?AbstractTypeEnvironmentQ, abs:AbstractType[class_?StringQ]] :=
	If[self["instanceMap"]["keyExistsQ", class],
		self["instanceMap"]["lookup", class],
    	TypeFailure[
        	"AbstractTypeEnvironmentGetClassInstances",
        	"unable to find class `1` within the type environment `2`",
        	class,
        	self
    	]
  	]
		
  

sameQ[a_, b_] := a["sameQ", b];



instanceMatchQ[insts_?ListQ, qual_?TypeQualifiedQ] :=
    instanceMatchQ[insts, qual["type"]]
    
instanceMatchQ[insts_?ListQ, pred_?TypePredicateQ] :=
    If[insts === {},
        False,
        AnyTrue[
            Map[#["type"]&, insts],
            TypeUnifiableQ[#["type"], pred["type"]]&
        ]
    ]

instanceMatchQ[pred1_?TypePredicateQ, pred2_?TypePredicateQ] :=
    AllTrue[Transpose[{pred1["types"], pred2["types"]}], (Apply[TypeUnifiableQ, #]&)]



(**************************************************
 ** Context Reduction for membership of abstract classes
 **************************************************)


contextReduce0[tenv_?AbstractTypeEnvironmentQ, {}] := {};
contextReduce0[tenv_?AbstractTypeEnvironmentQ, preds:{__?TypePredicateQ}] :=
    Module[{loop, res},
        loop[rs_, {}] := rs;
        loop[rs_, {p_, ps___}] :=
          If[superclassEntailsQ[tenv, Join[rs, {ps}], p],
            loop[rs, {ps}],
            loop[Prepend[rs, p], {ps}]
          ];
        res = loop[{}, preds];
        ClearAll[loop];
        res
    ];

contextReduceAbstract[tenv_?AbstractTypeEnvironmentQ, preds_?ListQ] :=
    With[{
        qs = toHnfs[tenv, preds]
    },
        contextReduce0[tenv, qs]
    ];



(*
  Context reduction for Literal predicates
*)


(*
   Apply the test to the literal value.
*)
evaluateLiteral[ self_, test_, ty_?TypeLiteralQ] :=
	Which[
		test[ty["value"]] === True, 
			True,
		test[ty["value"]] === False, 
			TypeFailure[
            	"ContextReduction",
            	"failed to evaluate for TypeLiteral `1`",
            	ty
          	],
		True, ty
	]

evaluateLiteral[ self_, test_, ty_?TypeEvaluateQ] :=
	ty

evaluateLiteral[ self_, test_, ty_?TypeVariableQ] :=
	ty


(*
  Anything else in a TypePredicate like this cannot be handled.
*)
evaluateLiteral[ args___] :=
   ThrowException[TypeInferenceException[{"Unknown argument to evaluateLiteral", {args}}]]
 
(*
 For each predicate apply the test to the literal value,  if the test returns True we will drop
 this predicate,  if it returns False we will error out, and otherwise we keep the predicate.
*)
contextEvaluate[ tenv_?AbstractTypeEnvironmentQ, pred_?TypePredicateQ] :=
	With[ {
		vals = Map[ evaluateLiteral[tenv, pred["test"], #]&, pred["types"]]
	},
	With[ {
		cleaned = DeleteCases[ vals, True]
	},
		If[cleaned === {}, 
			{}, 
			CreateTypePredicate[ cleaned, pred["test"]]]
	]]

contextEvaluate[ args___] :=
   ThrowException[TypeInferenceException[{"Unknown argument to contextEvaluate", {args}}]]



contextReduceEvaluate[ tenv_?AbstractTypeEnvironmentQ, preds_?ListQ] :=
	Flatten[Map[contextEvaluate[tenv, #]&, preds]]



contextReduce[ tenv_?AbstractTypeEnvironmentQ, preds_?ListQ] :=
	With[ {
		groups = GroupBy[ preds, #["hasAbstractType"]&]
	},
		Join[ contextReduceAbstract[ tenv,Lookup[groups, True, {}]],
			contextReduceEvaluate[ tenv,Lookup[groups, False, {}]]]
	]




(**************************************************)


ClearAll[toHnf]
toHnf[tenv_?AbstractTypeEnvironmentQ, pred_?TypePredicateQ] :=
	With[ {
		cacheVal = getFromHNFCache[tenv, pred]
	},
	Which[
		cacheVal =!= Null,
			cacheVal,
		pred["isHnf"],
        	{pred},
        True,
	        With[ {
	            insts = CatchTypeFailure[
	                byInstances[tenv, pred],
	                TypeError,
	                $Failed&
	            ]
	        },
	            If[FailureQ[insts],
	                {
	                  TypeFailure[
	                    "ContextReduction",
	                    "failed to perform context reduction for predicate `1`",
	                    pred
	                  ]
	                },
	                addToHNFCache[ tenv, pred, toHnfs[tenv, insts]]
	            ]
	        ]
		]
    ]
  
getFromHNFCache[ self_, pred_] :=
	self["HNFCache"]["lookup", pred["unresolve"], Null]

addToHNFCache[ self_, pred_, hnf_] :=
	(
	self["HNFCache"]["associateTo", pred["unresolve"] -> hnf];
	hnf
	)



ClearAll[toHnfs]
toHnfs[tenv_, preds_] :=
  With[{
    ps = CatchTypeFailure[
        Catenate[toHnf[tenv, #]& /@ preds],
        TypeError,
        $Failed&
    ]
  },
    If[FailureQ[ps],
      TypeFailure[
        "ContextReduction",
        "failed to perform context reduction for predicates `1`",
        preds
      ],
      ps
    ]
  ];


(**************************************************)

icon := icon = Graphics[Text[
    Style["TConsEnv",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
]

toBoxes[env_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "AbstractTypeEnvironment",
        env,
        icon,
        {
            BoxForm`SummaryItem[{Pane["length: ", {90, Automatic}], Length[env["classes"]["get"]]}]
        },
        {
            BoxForm`SummaryItem[{Pane["classes: ", {90, Automatic}], env["classes"]}]
        },
        fmt
    ]


toString[env_] := "AbstractTypeEnvironment[<>]"

(**************************************************)


graphForm[self_] :=
    Module[{
        classes = self["classes"]["get"],
        classValues = self["classes"]["values"]
    },
    With[{
        verts = #["typename"]& /@ classValues,
        edges = Flatten@Table[
            (cls["typename"] -> #)& /@ cls["supers"]["get"],
            {cls, classValues}
        ]
    },
        DynamicModule[{graph, makeInfo, selected = None},
			makeInfo[env_, None] :=
			    Nothing;
			makeInfo[env_, name_?StringQ] :=
			    If[!KeyExistsQ[env, AbstractType[name]],
			        "The class " <> name <> " was not found in abstract type environment",
			        With[{
			            cls = Lookup[env, AbstractType[name]]
			        },
			            CompileInformationPanel[
                            "AbstractType[" <> cls["typename"] <> "] Information"
                            ,
			                {
			                    "id" -> cls["id"],
			                    "name" -> cls["typename"],
			                    "arguments" -> cls["arguments"],
			                    "members" -> If[cls["members"] === {}, None, Panel[Column[cls["members"]]]],
			                    "supers" -> cls["supers"]["toList"],
			                    "instances" -> Column[cls["instances"]["toList"]],
			                    If[cls["default"] === Undefined,
			                        Nothing,
			                        "default type" -> cls["default"]
			                    ]
			                }
			            ]
			        ]
			    ];
            Dynamic[
                Framed[
	                Grid[{
	                        {Item[Text[Style["Abstract Type Environment", Bold, Larger]], Alignment -> {Center, Center}]},
	                        {
	                            Item[graph, Alignment -> {Center, Center}]
	                        },
	                        {
	                            With[{info = makeInfo[classes, selected]},
	                                If[info === Nothing,
	                                    Nothing,
	                                    Item[info, Alignment -> {Center, Center}]
	                                ]
	                            ]
	                        }
	                    },
	                    Frame -> True,
	                    RowLines -> True,
	                    FrameStyle -> GrayLevel[0.65],
	                    Spacings -> {"Columns" -> {{1}}},
	                    Alignment -> {"Columns" -> {{Left}}}
	                ],
	                RoundingRadius->5,
                    FrameStyle->Opacity[0.1],
                    Background -> GrayLevel[0.95]
                ],
		        TrackedSymbols :> {selected}
            ],
            Initialization :> (
                graph = Graph[
                    verts, Reverse[edges, 2],
                    VertexLabels -> Placed["Name", Before],
                    VertexStyle -> GrayLevel[0.65], EdgeStyle -> Gray,
                    GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Top},
                    VertexShapeFunction -> (
                         EventHandler[
                             Disk[#1, .04],
                             {
                                "MouseClicked" :> (selected = #2),
                                Method -> "Preemptive",
                                PassEventsDown -> Automatic,
                                PassEventsUp -> True
                            }
                         ]&
                    ),
                    ImageSize -> {600}
                ]
            )
        ]
    ]];

(**************************************************)
(**************************************************)

End[]

EndPackage[]

