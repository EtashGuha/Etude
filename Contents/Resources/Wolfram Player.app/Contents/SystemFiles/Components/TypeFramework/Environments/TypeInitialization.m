
BeginPackage["TypeFramework`Environments`TypeInitialization`"]

TypeInitializationQ
CreateTypeInitialization

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`"]
Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`TypeObjects`AbstractType`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`TypeObjects`TypePredicate`"]
Needs["TypeFramework`TypeObjects`TypeQualified`"]
Needs["TypeFramework`TypeObjects`TypeRecurse`"]
Needs["TypeFramework`Utilities`PrenexNormalForm`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`Language`DesugarType`"]



RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeInitializationClass = DeclareClass[
	TypeInitialization,
	<|
		"addInitialization" -> Function[{tyEnv, data}, addInitialization[Self,tyEnv,data]],
		"dispose" -> (dispose[Self]&),
		"finalize" -> (finalize[Self, ##]&)
	|>,
	{
        "typeAliases",
		"abstractTypes",
		"concreteTypes",
		"declareAtoms",
		"declareFunctions",
		"typeInstances",
		"active" -> True
	},
	Predicate -> TypeInitializationQ
]
]]


CreateTypeInitialization[ ] :=
	CreateObject[TypeInitialization,
		<|
        "typeAliases" -> CreateReference[{}],
		"abstractTypes" -> CreateReference[{}],
		"concreteTypes" -> CreateReference[{}],
		"declareAtoms" -> CreateReference[{}],
		"declareFunctions" -> CreateReference[{}],
		"typeInstances" -> CreateReference[{}]
		|>
	]



$initializationHandlers =
<|
	"DeclareAtom" -> addDeclareAtom,
	"DeclareFunction" -> addDeclareFunction,
	"ConcreteType" -> addConcreteType,
	"AbstractType" -> addAbstractType,
    "TypeAlias" -> addTypeAlias,
	"TypeInstance" -> addTypeInstance
|>

dispose[self_] :=
	Module[ {},
		self["setActive", False];
		self["setAbstractTypes", Null];
		self["setConcreteTypes", Null];
		self["setDeclareAtoms", Null];
		self["setDeclareFunctions", Null];
		self["setTypeInstances", Null];
	]


checkActive[self_] :=
	If[ !TrueQ[self["active"]],
		ThrowException[{"TypeInitialization system is not active"}]
	]

getInitializationFunction[self_, data_] :=
	(
	checkActive[self];
	Lookup[$initializationHandlers, data["handler"], unknownInitialization]
	)

unknownInitialization[self_, data_] :=
	ThrowException[{"Cannot find initialization handler ", data}]

addInitialization[self_, tyEnv_, data_] :=
	getInitializationFunction[self, data][self, tyEnv, data]

addDeclareFunction[self_, tyEnv_, data_] :=
	self["declareFunctions"]["appendTo", data]

addDeclareAtom[self_, tyEnv_, data_] :=
	self["declareAtoms"]["appendTo", data]

addTypeAlias[self_, tyEnv_, data_] :=
    self["typeAliases"]["appendTo", data["type"]]

addTypeInstance[self_, tyEnv_, data_] :=
	self["typeInstances"]["appendTo", data["type"]]

addAbstractType[self_, tyEnv_, data_] :=
	With[{tyObj = CreateAbstractType[data["type"]]},
		tyEnv["abstracttypes"]["addClass", tyObj];
		self["abstractTypes"]["appendTo", tyObj]
	]

addConcreteType[self_, tyEnv_, data_] :=
	With[{
	    tyCons = CreateTypeConstructor[data["type"]]
    },
    With[{
        abstractTypes = tyCons["implements"]
    },
		tyEnv["typeconstructors"]["add", tyCons];
		self["concreteTypes"]["appendTo", tyCons];
		Do[
		  self["typeInstances"]["appendTo",
		      TypeInstance[abstractType, {}, tyCons, {}, "Constraints" -> {}]
		  ],
		  {abstractType, abstractTypes}
		];
		If[tyCons["hasProperty", "metadata"] && KeyExistsQ[tyCons["getProperty", "metadata"], "NameAlias"],
			self["typeAliases"]["appendTo",
				DesugarType[TypeAlias[tyCons["getProperty", "metadata"]["NameAlias"], TypeSpecifier[tyCons["name"]]]]
			]
		];
		tyCons
	]];

(*
  Type Environment Resolution

  Pass through
  	type constructors
  	abstract types
  	worklist
*)

finalize[self_, tyEnv_] :=
	CatchException[
		Module[ {duplicates, concreteTypes, typeAliases, abstractTypes, typeInstances, declaredFunctions, declaredAtoms},
			concreteTypes = self["concreteTypes"]["get"];
			duplicates = DeleteCases[Tally[#["name"]& /@ concreteTypes], {_, 1}];
			If[duplicates =!= {},
				ThrowException[TypeInferenceException[{"Duplicate concreteTypes are declared: ", duplicates}]]
			];
			Scan[ finalizeTypeConstructor[tyEnv,#]&, concreteTypes];

			typeAliases = self["typeAliases"]["get"];
			duplicates = DeleteCases[Tally[#[[1]]& /@ typeAliases], {_, 1}];
			If[duplicates =!= {},
				ThrowException[TypeInferenceException[{"Duplicate typeAliases are declared: ", duplicates}]]
			];
			Scan[ finalizeTypeAlias[tyEnv,#]&, typeAliases];

			abstractTypes = self["abstractTypes"]["get"];
			duplicates = DeleteCases[Tally[#["typename"]& /@ abstractTypes], {_, 1}];
			If[duplicates =!= {},
				ThrowException[TypeInferenceException[{"Duplicate abstractTypes are declared: ", duplicates}]]
			];
			Scan[ finalizeAbstractType[tyEnv,#]&, abstractTypes];

			typeInstances = self["typeInstances"]["get"];
			(*
			think of what to test for duplicate typeInstances
			*)
			Scan[ finalizeTypeInstance[tyEnv,#]&, typeInstances];

			declaredFunctions = self["declareFunctions"]["get"];
			duplicates = DeleteCases[Tally[declaredFunctions], {_, 1}];
			If[duplicates =!= {},
				ThrowException[TypeInferenceException[{"Duplicate functions are declared: ", duplicates}]]
			];
			Scan[ finalizeDeclareFunction[tyEnv,#]&, declaredFunctions];
            
            declaredAtoms = self["declareAtoms"]["get"];
            duplicates = DeleteCases[Tally[declaredAtoms], {_, 1}];
            If[duplicates =!= {},
				ThrowException[TypeInferenceException[{"Duplicate atoms are declared: ", duplicates}]]
			];
			Scan[ finalizeDeclareAtom[tyEnv,#]&, declaredAtoms];
            
            tyEnv["abstracttypes"]["finalize"];
			self["dispose"];
			tyEnv["setStatus", "Normal"];
		]
		,
		{{_, Function[
				tyEnv["dispose"];
				ThrowException[#]
		]}}
	]


(*
  Nothing to do.
*)
finalizeTypeConstructor[ tyEnv_, obj_] :=
	Null




(*
  Finalize each abstract type
*)


finalizeAbstractType[tyEnv_, class_] :=
	Module[ {args},
		finalizeClassDefaultType[tyEnv, class];
        args = class["arguments"];
		Switch[Length[args],
			0, finalizeClassZero[ tyEnv, class],
			_, finalizeClassNary[ tyEnv, class, args]
		];
	]

(* Finalize the default paramater for
 * each of the abstract classes
 *)

finalizeClassDefaultType[ tyEnv_, class_] :=
    Module[{ty},
        If[class["default"] === Undefined,
            Return[]
        ];
        ty = tyEnv["resolve", class["default"]];
        class["setDefault", ty];
        tyEnv["abstracttypes"]["addDefault", class -> class["default"]]
    ]


(*
  Check there are no functions defined.   Should this be an error?
*)
finalizeClassZero[ tyEnv_, class_] :=
	Null



finalizeClassNary[ tyEnv_, class_, args_] :=
	Module[{argVars, preds},
		argVars = MapIndexed[genAbstractTypeArg, args];
		preds = Map[ CreateTypePredicate[#, MemberQ[class["typename"]]]&, argVars];
		Scan[ processMember[ tyEnv, AssociationThread[args -> argVars], preds, #]&, class["members"]];
	]


(*
  This is called for declareFunction, declareAtom and processing member functions of 
  abstract classes.
*)
canonicalizeData[ data_?AssociationQ] :=
	Module[ {name, arg},
		name = data["name"];
		arg = data["argument"];
		If[  MissingQ[name],
			ThrowException[TypeInferenceException[{"Missing a function name value", name, data}]]
		];
		If[  MissingQ[arg],
			ThrowException[TypeInferenceException[{"Missing an argument value", name, data}]]
		];
		canonicalizeData0[name, arg]
	]

canonicalizeData0[ name_, TypeSpecifier[type_]] :=
	FunctionData[<|"Name" -> name, "Type" -> type|>]

canonicalizeData0[ name_, Type[type_]] :=
	FunctionData[<|"Name" -> name, "Type" -> type|>]

canonicalizeData0[ name_, Typed[type_][body_]] :=
	canonicalizeData0[ name, Typed[body, type]]

canonicalizeData0[ name_, Typed[ body_, type_]] :=
	FunctionData[<|"Name" -> name, "Type" -> type, "Implementation" -> body|>]

canonicalizeData0[ name_, MetaData[data_?AssociationQ]] :=
	FunctionData[Join[ data, <|"Name" -> name|>]]
	
canonicalizeData0[name_, MetaData[data_][r_]] :=
	canonicalizeMetaData[name, MetaData[data][r]]

canonicalizeMetaData[  name_, MetaData[data1_?AssociationQ][MetaData[data2_?AssociationQ][r_]]] :=
	canonicalizeData0[name, MetaData[Join[ data1, data2]][r]]
	
canonicalizeMetaData[  name_, MetaData[data_?AssociationQ][Type[type_]]] :=
	FunctionData[Join[ data, <|"Name" -> name, "Type" -> type|>]]

canonicalizeMetaData[  name_, MetaData[data_?AssociationQ][TypeSpecifier[type_]]] :=
	FunctionData[Join[ data, <|"Name" -> name, "Type" -> type|>]]

canonicalizeMetaData[ name_, MetaData[data_?AssociationQ][Typed[body_, type_]]] :=
	FunctionData[Join[ data, <|"Name" -> name, "Type" -> type, "Implementation" -> body|>]]

canonicalizeMetaData[ name_, MetaData[data_?AssociationQ][Typed[type_][body_]]] :=
	FunctionData[Join[ data, <|"Name" -> name, "Type" -> type, "Implementation" -> body|>]]

canonicalizeMetaData[ name_, MetaData[data_?AssociationQ][f_Function]] :=
	FunctionData[Join[ data, <|"Name" -> name, "Implementation" -> f|>]]

canonicalizeMetaData[ args___] :=
	ThrowException[TypeInferenceException[{"Unknown call to metadata canonicalisation", {args}}]]
	
canonicalizeData0[ args___] :=
	ThrowException[TypeInferenceException[{"Unknown call to function canonicalisation", {args}}]]






genAbstractTypeArg[a_, {idx_}] :=
    Switch[a,
        _?StringQ,
            CreateTypeVariable[a],
        _?TypeVariableQ,
            CreateTypeVariable[a["name"]],
        _,
            TypeFailure[
                "CreateAbstractType",
                "one or more of the args `1` used to create a type class is invalid",
                a
            ]
    ];

extractTypeInstanceArguments[tys_?ListQ] :=
    With[{
        as = Merge[extractTypeInstanceArgument /@ tys, Join]
    },
        {
            Lookup[as, "Variable", {}],
            Lookup[as, "Constraint", {}]
        }
    ];
extractTypeInstanceArgument[ty_?TypeBaseClassQ] :=
    (* Todo:: can perform more syntax sugar here *)
    <| "Variable" -> ty, "Constraint" -> Nothing |>;
extractTypeInstanceArgument[name_?StringQ] :=
    <| "Variable" -> name, "Constraint" -> Nothing |>;
extractTypeInstanceArgument[e:Element[var_, abstractType_]] :=
    <| "Variable" -> var, "Constraint" -> e |>;

finalizeTypeInstance[tyEnv_, TypeInstance[abs_, vars0_, typeIn_, funs_?ListQ, opts_:<||>]] :=
	Module[ {argVars, varDefs, type, preds, class, pred, qual, vars, cons0, cons},
		cons0 = Flatten[ {Lookup[opts, "Constraints", {}]}];
        {vars, cons} = extractTypeInstanceArguments[vars0];
        cons = Join[cons, cons0];
        argVars = MapIndexed[genAbstractTypeArg, vars];
		varDefs = AssociationThread[vars -> argVars];
		type = tyEnv["resolveWithVariables", typeIn, varDefs];
		preds = Map[ tyEnv["resolveWithVariables", TypePredicate[#], varDefs]&, cons];
		class = tyEnv["abstracttypes"]["getClass", abs, Null];
		If[ class === Null,
			ThrowException[TypeInferenceException[{"Cannot find abstract class", {abs}}]]
		];
		pred = CreateTypePredicate[ type, MemberQ[abs]];
		qual = CreateTypeQualified[preds, pred];
		class["addInstance", qual];
        Scan[ processMember[ tyEnv, AssociationThread[vars -> argVars], pred, #]&, funs];

	]

processMember[ tyEnv_, varMap_, preds_,  nameIn_ -> arg_] :=
	Module[ {tyObj, qual, forall, data, name, type, defs},
		data = canonicalizeData[<|"argument" -> arg, "name" -> nameIn|>];
		If[ !MatchQ[ data, FunctionData[_?AssociationQ]],
			ThrowException[TypeInferenceException[{"DeclareFunction missing a valid data field", arg}]]
		];
		data = First[data];
		name = data["Name"];
		type = data["Type"];
		If[  MissingQ[name],
			ThrowException[TypeInferenceException[{"Function declaration missing a function name value", name, arg}]]
		];
		If[  MissingQ[type],
			ThrowException[TypeInferenceException[{"Function declaration missing a type value", name, arg}]]
		];	
		tyObj = tyEnv["resolveWithVariables", TypeSpecifier[type], varMap];
		If[ !TypeObjectQ[tyObj] && !TypeRecurseQ[tyObj],
			ThrowException[TypeInferenceException[{"Function declaration cannot resolve its type", name, type}]]
		];
		qual = CreateTypeQualified[ preds, tyObj];
		forall = CreateTypeForAll[Values[varMap], qual];
		forall = ToPrenexNormalForm[forall];
		defs = KeyDrop[data, {"Name", "Type"}];
		defs = tyEnv["functionDefinitionLookup"]["finalizeDefinition", tyEnv, name, defs, forall];
		If[ defs === Null,
			tyEnv["functionTypeLookup"]["addExcludedType", name, forall],
			forall["setProperty", "definition" -> defs];
			tyEnv["functionTypeLookup"]["addType", name, forall]
		];
	]


finalizeDeclareFunction[tyEnv_, arg_] :=
	Module[ {data, ty, name, type, defs},
		data = canonicalizeData[arg];
		If[ !MatchQ[ data, FunctionData[_?AssociationQ]],
			ThrowException[TypeInferenceException[{"DeclareFunction missing a valid data field", arg}]]
		];
		data = First[data];
		name = data["Name"];
		type = data["Type"];
		If[  MissingQ[name],
			ThrowException[TypeInferenceException[{"Function declaration missing a function name value", name, arg}]]
		];
		If[  MissingQ[type],
			ThrowException[TypeInferenceException[{"Function declaration missing a type value", name, arg}]]
		];
		ty = tyEnv["resolve", type];
		If[ !TypeObjectQ[ty] && !TypeRecurseQ[ty],
			ThrowException[TypeInferenceException[{"Encountered an error when finalizing function declarations. Function declaration cannot resolve its type", name, type}]]
		];
		defs = KeyDrop[data, {"Name", "Type"}];
		defs = tyEnv["functionDefinitionLookup"]["finalizeDefinition", tyEnv, name, defs, ty];
		ty["setProperty", "definition" -> defs];
		tyEnv["functionTypeLookup"]["addType", name, ty];
	]

finalizeDeclareAtom[tyEnv_, arg_] :=
	Module[ {data, ty, name, type, defs},
		data = canonicalizeData[arg];
		If[ !MatchQ[ data, FunctionData[_?AssociationQ]],
			ThrowException[TypeInferenceException[{"DeclareFunction missing a valid data field", arg}]]
		];
		data = First[data];
		name = data["Name"];
		type = data["Type"];
		If[  MissingQ[name],
			ThrowException[TypeInferenceException[{"Atom declaration missing a name value", name, arg}]]
		];
		If[  MissingQ[type],
			ThrowException[TypeInferenceException[{"Atom declaration missing a type value", name, arg}]]
		];
		ty = tyEnv["resolve", type];
		If[ !TypeObjectQ[ty],
			ThrowException[TypeInferenceException[{"Atom declaration cannot resolve its type", type}]]
		];
		defs = KeyDrop[data, {"Name", "Type"}];
		defs = tyEnv["functionDefinitionLookup"]["finalizeAtomDefinition", tyEnv, name, defs, ty];
		ty["setProperty", "definition" -> defs];
		tyEnv["functionTypeLookup"]["addType", name, ty];
	]


finalizeTypeAlias[tyEnv_, TypeAlias[ty0_TypeConstructor, val_, opts_:<||>]] :=
	finalizeTypeAlias[tyEnv, TypeAlias[CreateTypeConstructor[ty0], val, opts]]

finalizeTypeAlias[tyEnv_, TypeAlias[ty0_, val_, opts_:<||>]] :=
    If[ Lookup[ opts, "VariableAlias", False], 
    	finalizeTypeVariableAlias[tyEnv, TypeAlias[ty0, val, opts]]
    	,
		finalizeTypeAliasBase[tyEnv, TypeAlias[ty0, val, opts]]
    ]


finalizeTypeVariableAlias[tyEnv_, TypeAlias[TypeSpecifier[ty0_], tyVal_, opts_:<||>]] :=
    finalizeTypeVariableAlias[tyEnv, TypeAlias[ty0, tyVal, opts]]

finalizeTypeVariableAlias[tyEnv_, TypeAlias[Type[ty0_], tyVal_, opts_:<||>]] :=
    finalizeTypeVariableAlias[tyEnv, TypeAlias[ty0, tyVal, opts]]


makeFunction[ args_, body_] :=
	With[ {
		vars =  Map[ Unique, args]
	},
	With[ {
		rules = MapThread[ Rule, {args, vars}]	
	},
	With[ {
		nBody = ReplaceAll[body, rules]
	},
		Function @@ {vars, nBody}
	]]]
   
finalizeTypeVariableAlias[tyEnv_, TypeAlias[ty0_[tyArgs__], tyVal_, opts_:<||>]] :=
    With[{
        ty = tyEnv["resolve", ty0],
        fun =  makeFunction[ {tyArgs}, tyVal]
    },
    With[{
        uty = ty["unresolve"]
    },
        tyEnv["typeVariableAliases"]["associateTo", uty -> {Length[ {tyArgs}], fun}];
    ]];


constraintPass[ tyEnv_, opts_] :=
	With[ {meta = Lookup[opts, MetaData, Null]},
		If[meta === Null,
			True,
			TrueQ[Lookup[meta,"Constraint", True&][tyEnv]]
		]
	]


finalizeTypeAliasBase[tyEnv_, TypeAlias[ty0_, val_, opts_:<||>]] :=
	If[ constraintPass[tyEnv, opts],
    	With[{
        	ty = tyEnv["resolve", ty0]
    	},
    	With[{
        	trgt = tyEnv["resolve", val],
        	uty = ty["unresolve"]
    	},
        	tyEnv["typeAliases"]["associateTo", uty -> trgt];
    	]]
	]



End[]

EndPackage[]

