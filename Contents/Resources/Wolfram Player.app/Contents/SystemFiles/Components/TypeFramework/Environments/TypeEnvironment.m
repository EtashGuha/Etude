
BeginPackage["TypeFramework`Environments`TypeEnvironment`"]

CreateTypeEnvironment

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`Environments`TypeConstructorEnvironment`"]
Needs["TypeFramework`Environments`AbstractTypeEnvironment`"]
Needs["TypeFramework`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`Language`DesugarType`"]
Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`TypeObjects`TypeLiteral`"]
Needs["TypeFramework`TypeObjects`TypeEvaluate`"]
Needs["TypeFramework`TypeObjects`TypePredicate`"]
Needs["TypeFramework`TypeObjects`TypeProjection`"]
Needs["TypeFramework`TypeObjects`TypeQualified`"]
Needs["TypeFramework`TypeObjects`TypeSequence`"]
Needs["TypeFramework`TypeObjects`TypeRecurse`"]
Needs["TypeFramework`ConstraintObjects`AlternativeConstraint`"]
Needs["TypeFramework`ConstraintObjects`AssumeConstraint`"]
Needs["TypeFramework`ConstraintObjects`ConstraintBase`"]
Needs["TypeFramework`ConstraintObjects`EqualConstraint`"]
Needs["TypeFramework`ConstraintObjects`FailureConstraint`"]
Needs["TypeFramework`ConstraintObjects`GeneralizeConstraint`"]
Needs["TypeFramework`ConstraintObjects`InstantiateConstraint`"]
Needs["TypeFramework`ConstraintObjects`LookupConstraint`"]
Needs["TypeFramework`ConstraintObjects`ProveConstraint`"]
Needs["TypeFramework`ConstraintObjects`SkolemConstraint`"]
Needs["TypeFramework`ConstraintObjects`SuccessConstraint`"]
Needs["TypeFramework`Environments`FunctionTypeLookup`"]
Needs["TypeFramework`Environments`TypeInitialization`"]
Needs["TypeFramework`Environments`FunctionDefinitionLookup`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Error`Suggestions`"]


RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeEnvironmentClass = DeclareClass[
	TypeEnvironment,
	<|
		"addInitialization" -> Function[ {data}, Self["initializationData"]["addInitialization", Self, data]],
		"reifiableQ" -> Function[ {ty}, reifiableQ[Self, ty]],
		"reify" -> Function[ {ty}, reify[Self, ty]],
		"resolvableQ" -> Function[ {ty}, resolvableQ[Self, ty]],
		"resolve" -> Function[ {ty}, resolveTop[Self, ty]],
		"resolvableWithVariablesQ" -> Function[ {ty}, resolvableWithVariablesQ[Self, ty]],
		"resolveWithVariables" -> Function[ {ty, vars}, resolveWithVariables[Self, ty, vars]],
		"declareFunction" -> (declareFunction[Self, ##]&),
		"declareAtom" -> (declareAtom[Self, ##]&),
		"declareType" -> (declareType[Self, ##]&),
		"finalize" -> (finalize[Self, ##]&),
		"reopen" -> (reopen[Self]&),
		"getLiteralType" -> (getLiteralType[Self, #]&),
		"dispose" -> (dispose[Self]&),
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"name",
		"abstracttypes",
		"typeconstructors",
		"functionTypeLookup",
		"functionDefinitionLookup",
		"initializationData",
		"literalProcessor",
		"typeAliases",
		"typeVariableAliases",
		"typeCache",
		"status" -> "Initial",
		"properties"
	},
	Extends -> {
		ClassPropertiesTrait
	},
	Predicate -> TypeEnvironmentQ
]
]]


(*
  declareType Method
*)

(*
  If the environment has been finalized,  this should have a different implementation
*)

declareType[ self_, arg_] :=
	Module[ {ty, handler},
		ty = DesugarType[arg];
		handler = getHandler[ty];
		self["addInitialization", <|"handler" -> handler, "type" -> ty|>]
	]

isTypeConstructor[ _TypeConstructor] := True
isTypeConstructor[ _] := False

isAbstractType[ _AbstractType] := True
isAbstractType[ _] := False

isTypeInstance[ _TypeInstance] := True
isTypeInstance[ _] := False

isTypeAlias[ _TypeAlias] := True
isTypeAlias[ _] := False

getHandler[ty_] :=
	Which[
		isTypeConstructor[ty], "ConcreteType",
		isAbstractType[ty], "AbstractType",
		isTypeInstance[ty], "TypeInstance",
		isTypeAlias[ty], "TypeAlias",
		True, 
		  ThrowException[TypeInferenceException[{"Unknown type object", ty}]]
	]

declareType[ args___] :=
	ThrowException[TypeInferenceException[{"Unknown call to declareType", {args}}]]



(*
  declareFunction Method
*)

(*
  If the environment has been finalized,  this should have a different implementation
*)

declareFunction[ self_, name_, arg_] :=
	self["addInitialization", <| "handler" -> "DeclareFunction", "argument" -> arg, "name" -> name|>]

declareFunction[ args___] :=
	ThrowException[TypeInferenceException[{"Unknown call to declareFunction", {args}}]]

declareAtom[ self_, name_, arg_] :=
	self["addInitialization", <| "handler" -> "DeclareAtom", "argument" -> arg, "name" -> name|>]

declareAtom[ args___] :=
	ThrowException[TypeInferenceException[{"Unknown call to declareAtom", {args}}]]


(*
  Look at arg to see if it could be turned into a literal type.
  Return the name of the type if so and Null otherwise.
*)

getLiteralType[self_, ty_] :=
	Module[ {litProcessor = self["literalProcessor"], ef},
		If[litProcessor === Null,
			ThrowException[TypeInferenceException[{"Invalid literal processor. The value cannot be resolved for a TypeLiteral", ty}]]
		];
		ef = litProcessor[ self, ty];
		If[Head[ef] === litProcessor || ef === Null,
			Null,
			self["resolve", ef]]
	]

(*
  Type Resolution
*)

reifiableQ[self_, t_] :=
	CatchTypeFailure[reify[self,t]; True, "ThrownExceptionTag", False &]
reifiableQ[___] := False

reify[self_, t_] :=
    With[{
        ty = self["resolve", t]["unresolve"]
    },
        self["resolve", ty]
    ];
    

resolvableQ[self_, t_] :=
	Catch[resolveTop[self,t]; True, _, False &]
resolvableQ[___] := False

resolveTop[ self_, t_TypeRecurse] :=
	resolveTypeRecurse[self, t]

resolveTop[ self_, Type[t_TypeRecurse]] :=
	resolveTypeRecurse[self, t]


resolveTop[ self_, ty_] :=
	Module[ {state},
		state = <|"variableStore" -> CreateReference[<||>] |>;
		resolve[self,state, ty]
	]
	
resolvableWithVariablesQ[self_, ty_, varDefs_] :=
	AssociationQ[varDefs] &&
	CatchTypeFailure[resolveWithVariables[self, ty, varDefs]; True, "ThrownExceptionTag", False &]
resolvableWithVariablesQ[___] := False

resolveWithVariables[ self_, ty_, varDefs_?AssociationQ] :=
	resolveWithVariables[self, ty, CreateReference[varDefs]]
	
associationReferenceQ[ref_] :=
	ReferenceQ[ref] && ref["head"] === Association

resolveWithVariables[ self_, ty_, varDefs_?associationReferenceQ] :=
	Module[ {state},
		state = <|"variableStore" -> varDefs |>;
		resolve[self,state, ty]
	]


resolve[self_, state_, Type[ ty_]] :=
	resolve[self, state, ty]

resolve[self_, state_, TypeSpecifier[ ty_]] :=
	resolve[self, state, ty]

resolve[self_, state_, ty_?TypeBaseClassQ] :=
	ty
resolve[self_, state_, ty_?BaseConstraintQ] :=
    ty

resolve[self_, state_, ty_] :=
	Module[{val},
		val = self["typeCache"]["lookup", ty, Null];
		If[ val =!= Null,
			Return[val]
		];
		val = resolveWorker[ self, state, ty];
		If[TypeConstructorQ[val],
			self["typeCache"]["associateTo", ty -> val]
		];
		val
	]

resolveWorker[ self_, state_, ty_Symbol] :=
(
    If[!MemberQ[$ContextPath, Context[ty]],
       ThrowException[{"Attempting to resolve a private symbol. This is most likely due to a missing package import.", Context[ty] <> SymbolName[ty]}];
    ];
	resolve[self, state, SymbolName[ty]]
)

resolveWorker[ self_, state_, ty_String] :=
	Module[ {tObj, typeConstructors, keys},
		tObj = state["variableStore"]["lookup", ty, Null];
		If[ tObj =!= Null,
			Return[ tObj]
		];
        tObj = self["typeAliases"]["lookup", TypeSpecifier[ty], Null];
        If[ tObj =!= Null,
            Return[resolveTop[self, tObj]];
        ];
        typeConstructors = self["typeconstructors"];
		tObj = typeConstructors["lookup", ty];
		If[ !TypeObjectQ[tObj],
			keys = typeConstructors["types"]["keys"];
			ThrowException[{"Cannot find type constructor for " <> ToString[ty] <> ". " <>  SuggestionsString[ty, keys]}]
		];
		tObj
	]

resolveWorker[self_, state_, AbstractType[ name_String]] :=
	Module[ {tObj},
		tObj = self["abstracttypes"]["getClass", name];
		If[ !AbstractTypeQ[tObj],
			ThrowException[{"Cannot find abstract type ", name, tObj}]
		];
		tObj
	]

(*
  Maybe it should be an error if the variable is not in the store, this means it is an 
  unbound variable. 
*)	
resolveWorker[ self_, state_, TypeVariable[ name_String]] :=
	Module[ {ty},
		ty = state["variableStore"]["lookup", name, Null];
		If[ ty === Null,
			ty = CreateTypeVariable[name];
			state["variableStore"]["associateTo", name -> ty]
		];
		ty
	]


resolveWorker[ self_, state_, TypeLiteral[ value_, tyR_]] :=
    Module[ {ty = resolve[self, state, TypeSpecifier[tyR]]},
        CreateTypeLiteral[
            value,
            ty
        ]
    ]
	
    
resolveWorker[ self_, state_, TypeProjection[ tyR_, value_?IntegerQ]] :=
	Module[{lit = getLiteralType[self, value]},
		If[lit === Null,
			ThrowException[TypeInferenceException[{"Unknown value cannot be resolved for a TypeLiteral", value}]]];
    	resolveWorker[self, state, 	TypeProjection[tyR, TypeLiteral[value, lit]]]
	]
    
resolveWorker[ self_, state_, TypeProjection[ tyR_, value_]] :=
    Module[ {ty = resolve[self, state, TypeSpecifier[tyR]]},
        CreateTypeProjection[
        	ty,
            resolve[self, state, value] 
        ]
    ]
		
resolveWorker[self_, state_, argsIn_List -> resIn_] :=
	Module[ {args, res},
		args = Map[ resolve[self, state, #]&, argsIn];
		res = resolve[self, state, resIn];
		CreateTypeArrow[args, res]
	]
	
(*AlternativeConstraint*)
resolveWorker[self_, state_, AlternativeConstraint[assoc_?AssociationQ]] :=
    CreateAlternativeConstraint[
        resolve[self, state, assoc["type"]],
        resolve[self, state, assoc["initial"]],
        resolve[self, state, #]& /@ assoc["alternatives"],
        resolve[self, state, #]& /@ assoc["monomorphic"]
    ]
(*AssumeConstraint*)
resolveWorker[self_, state_, AssumeConstraint[assoc_?AssociationQ]] :=
    CreateAssumeConstraint[resolve[self, state, assoc["predicate"]]];
(*EqualConstraint*)
resolveWorker[self_, state_, EqualConstraint[lhs0_, rhs0_]] :=
    Module[ {lhs, rhs},
        lhs = resolve[self, state, lhs0];
        rhs = resolve[self, state, rhs0];
        CreateEqualConstraint[lhs, rhs]
    ]
(*FailureConstraint*)
resolveWorker[self_, state_, FailureConstraint[]] :=
    CreateFailureConstraint[]
(*GeneralizeConstraint*)
resolveWorker[self_, state_, GeneralizeConstraint[assoc_?AssociationQ]] :=
    CreateGeneralizeConstraint[
        resolve[self, state, assoc["sigma"]],
        resolve[self, state, assoc["tau"]],
        resolve[self, state, #]& /@ assoc["monomorphic"]
    ]
(*LookupConstraint*)
resolveWorker[self_, state_, LookupConstraint[assoc_?AssociationQ]] :=
    Module[{c},
        c = CreateLookupConstraint[
	        assoc["name"],
	        resolve[self, state, assoc["type"]],
	        resolve[self, state, #]& /@ assoc["monomorphic"]
	    ];
        c["setInitial", resolve[self, state, assoc["initial"]]];
        c
    ];
(*InstantiateConstraint*)
resolveWorker[self_, state_, InstantiateConstraint[assoc_?AssociationQ]] :=
    CreateInstantiateConstraint[
        resolve[self, state, assoc["tau"]],
        resolve[self, state, assoc["rho"]]
    ]
(*ProveConstraint*)
resolveWorker[self_, state_, ProveConstraint[assoc_?AssociationQ]] :=
    CreateProveConstraint[
        resolve[self, state, assoc["predicate"]]
    ]
(*SkolemConstraint*)
resolveWorker[self_, state_, SkolemConstraint[assoc_?AssociationQ]] :=
    CreateSkolemConstraint[
        resolve[self, state, assoc["tau"]],
        resolve[self, state, assoc["rho"]],
        resolve[self, state, #]& /@ assoc["monomorphic"]
    ]
(*SuccessConstraint*)
resolveWorker[self_, state_, SuccessConstraint[]] :=
    CreateSuccessConstraint[]
    
resolveWorker[self_, state_, headIn_[argsIn__]] :=
	Module[ {args, head = headIn, tObj, headAlias},
		args = Map[ resolve[self, state, #]&, {argsIn}];
		If[ MatchQ[head, TypeSpecifier[_]],
			head = First[head]];
		head = resolve[self, state, head];
		headAlias = self["typeVariableAliases"]["lookup", head["unresolve"], Null];
		If[ headAlias === Null,
			tObj = CreateTypeApplication[head, args];
        	self["typeAliases"]["lookup",
            	tObj["unresolve"],
            	tObj
        	],
        	resolveHeadAlias[ self, state, head, args, headAlias]
        	]
	]

resolveHeadAlias[ self_, state_, head_, args_, {len_, fun_}] :=
	Module[ {ty1},
		If[ Length[args] =!= len,
			ThrowException[{"Type alias length does not match ", head["unresolve"]}]];
		ty1 = fun @@ args;
		resolve[ self, state, ty1]
	]



resolveWorker[self_, state_, TypeSequence[type_]] :=
	resolveWorker[self, state, TypeSequence[type, None, {0,Infinity}]]

resolveWorker[self_, state_, TypeSequence[type_, var_]] :=
	resolveWorker[self, state, TypeSequence[type, var, {0,Infinity}]]

resolveWorker[self_, state_, TypeSequence[type_, {min_, max_}]] :=
	resolveWorker[self, state, TypeSequence[type, None, {min, max}]]


resolveWorker[self_, state_, TypeSequence[typeIn_, varIn_, {min_, max_}]] :=
	Module[ {type, var},
		var = If[ varIn === None, None, resolve[self, state, varIn]];
		type = resolve[self, state, typeIn];
		CreateTypeSequence[type, var, {min, max}]
	]




(*
   For now we just use the store 
*)


getVariable[ self_, state_, TypeVariable[name_]] :=
	getVariable[self, state, name]
	
getVariable[ self_, state_, name_] :=
	Module[ {ty},
		ty = CreateTypeVariable[name];
		state["variableStore"]["associateTo", name -> ty];
		ty
	]




(*
  TypeForAll and Predicates
*)

$predicateHandler = 
<| 
	Element -> getElementPredicate,
	TypePredicate -> getTypePredicatePredicate
|> 


getPredicateVariable[self_, state_, var_] :=
	Module[{},
		resolve[self, state, var]
	]
	

getElementPredicate[self_, state_, Element[var_, absName_String]] :=
	getElementPredicate[self, state, Element[var, AbstractType[absName]]]

getElementPredicate[self_, state_, Element[var_, AbstractType[ absName_String]]] :=
	Module[{tyVar},
		tyVar = getPredicateVariable[self, state, var];
		CreateTypePredicate[ tyVar, MemberQ[ absName]]
	]

getTypePredicatePredicate[ self_, state_, TypePredicate[ HoldPattern[Element][args__]]] :=
	getElementPredicate[self, state, Element[args]]

getPredicate[self_, state_, TypePredicate[ tysIn:Except[_List], test_]] :=
	getPredicate[self, state, TypePredicate[ {tysIn}, test]]

getPredicate[self_, state_, TypePredicate[ tysIn_List, test_]] :=
	Module[ {tys},
		tys = Map[getPredicateVariable[self,state,#]&, tysIn];
		CreateTypePredicate[ tys, test]
	]

getPredicate[self_, state_, pred_] :=
	Module[{handler, ef},
		handler = Lookup[$predicateHandler, Head[pred], Null];
		ef = handler[self, state, pred];
		If[!TypePredicateQ[ef],
			ThrowException[TypeInferenceException[{"Cannot create TypePredicate ", pred}]]
		];
		ef
	]


getPredicates[ self_, state_, preds:Except[_List]] :=
	getPredicates[self, state, {preds}]

getPredicates[ self_, state_, preds_List] :=
	Map[ getPredicate[self, state,#]&, preds]

isPredicate[ a_TypePredicate] :=
	True

isPredicate[ a_Element] :=
	True

isPredicate[ a_] :=
	False

resolveWorker[self_, state_, TypePredicate[args__]] :=
	getPredicate[self, state, TypePredicate[args]]

resolveWorker[self_, state_, TypeQualified[predsIn_, typeIn_?isPredicate]] :=
	Module[ {type, preds},
		preds = getPredicates[self, state, predsIn];
		type = getPredicate[self, state, typeIn];
		CreateTypeQualified[ preds, type]
	]


resolveWorker[self_, state_, TypeQualified[predsIn_, typeIn_]] :=
	Module[ {type, preds},
		preds = getPredicates[self, state, predsIn];
		type = resolve[self, state, typeIn];
		CreateTypeQualified[ preds, type]
	]

resolveWorker[self_, state_, TypeForAll[ varsIn_, typeIn:Except[_TypeQualified]]] :=
	resolveWorker[self, state, TypeForAll[ varsIn, {}, typeIn]]

resolveWorker[self_, state_, TypeForAll[ varsIn_, predsIn_, typeIn_]] :=
	Module[ {vars, type, preds, qual},
		vars = Map[ getVariable[self, state, #]&, varsIn];
		preds = getPredicates[self, state, predsIn];
		type = resolve[self, state, typeIn];
		qual = CreateTypeQualified[ preds, type];
		CreateTypeForAll[vars, qual]
	]

resolveWorker[self_, state_, TypeForAll[ varsIn_, TypeQualified[preds_, type_]]] :=
	Module[ {vars, qual},
		vars = Map[ getVariable[self, state, #]&, varsIn];
		qual = resolve[self, state, TypeQualified[preds, type]];
		CreateTypeForAll[vars, qual]
	]

resolveWorker[self_, state_, TypeEvaluate[ fun_, argsIn_List]] :=
	With[ {
		args = Map[ resolve[self, state, #]&, argsIn]
	},
		CreateTypeEvaluate[fun, args]
	]


resolveWorker[self_, state_, arg_] :=
	Module[ {literalTy},
		literalTy = self["getLiteralType", arg];
		If[literalTy === Null,
			ThrowException[TypeInferenceException[{"Unknown type cannot be resolved ", arg}]]
			,
			resolveWorker[self, state, TypeLiteral[ arg, literalTy]]
			]
		]
	


resolveTypeRecurse[self_, TypeRecurse[sym_, vars_, t1_, t2_]] :=
	Module[ {},
		CreateTypeRecurse[sym, vars, t1, t2]
	]


resolveTypeRecurse[self_, arg_] :=
	ThrowException[TypeInferenceException[{"Unknown TypeRecurse cannot be resolved ", arg}]]


dispose[self_] :=
	(
	self["initializationData"]["dispose"];
	self["setInitializationData", Null];
	self["setLiteralProcessor", Null];
	self["setStatus", "Disposed"];
	self["setTypeCache", Null];
	self["abstracttypes"]["dispose"];
	self["functionTypeLookup"]["dispose"];
	self["typeconstructors"]["dispose"];
	)


finalize[self_] := (
    self["setStatus", "Finalized"];
	self["initializationData"]["finalize", self]
	)

reopen[self_] :=
	If[self["getStatus"] =!= "Reopened",
	   self["setStatus", "Reopened"];
	   self["setInitializationData", CreateTypeInitialization[]];
	]

CreateTypeEnvironment[] :=
	CreateObject[TypeEnvironment, <|
			"name"->"Main",
			"literalProcessor" -> Null,
			"initializationData" -> CreateTypeInitialization[],
			"abstracttypes" -> CreateAbstractTypeEnvironment[],
			"typeconstructors" -> CreateTypeConstructorEnvironment[],
			"functionTypeLookup" -> CreateFunctionTypeLookup[],
			"functionDefinitionLookup" -> CreateFunctionDefinitionLookup[],
            "typeAliases" -> CreateReference[<||>],
            "typeVariableAliases" -> CreateReference[<||>],
			"typeCache" -> CreateReference[<||>],
			"properties" -> CreateReference[<||>]
		|>]


(**************************************************)

icon := Graphics[Text[
	Style["TDecEnv",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[env_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"TypeEnvironment",
		env,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["name: ", {90, Automatic}], env["name"]}],
			BoxForm`SummaryItem[{Pane["status: ", {90, Automatic}], env["status"]}]
  		},
  		{
  			BoxForm`SummaryItem[{Pane["constructors: ", {90, Automatic}], env["typeconstructors"]}]
  		}, 
  		fmt
  	]


toString[env_] := "TypeEnvironment[<>]"



End[]

EndPackage[]

