

BeginPackage["CompileUtilities`ClassSystem`"]

Self;
Class;
DeclareClass;
ObjectInstance;
ObjectInstanceQ;
CreateObject;
ClassTrait;
Extends;
Predicate;
SetData;
$Classes;
ClassPropertiesTrait;

Begin["`Private`"] 

SetAttributes[Self, Protected]

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)


CreateObject::classnodef = "A definition for the class `1` cannot be found."

If[!ListQ[$Classes],
	$Classes = {}
]

camelCase[str_String] :=
	ToUpperCase[StringTake[str, 1]] <> StringDrop[str, 1]
camelCase[s_] := camelCase[ToString[s]]


$baseMethods = <|
	"initialize" -> Function[{}, Null],
	"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
	"toString" -> Function[{}, toString[Self]],
	"fullform" -> Function[{}, fullform[Self]]
|>


getExtendMethods[ClassTrait[x_]] := 
	x

(*
  Don't include the predicate,  these are not extensible.
*)
getExtendMethods[Class[_, methods_, _, _]] := 
	Delete[methods, Key["_predicate"]]
	
getExtendMethods[x_List] := 
	getExtendMethods /@ x
	
getExtendFields[x_ClassTrait] := 
	<||>
	
getExtendFields[Class[_, _, _, fields_]] := 
	fields
	
getExtendFields[x_List] := 
	Apply[Join, getExtendFields /@ x]

DeclareClass[name_, methodsIn_?AssociationQ, fields_?AssociationQ, opts0___] :=
	DeclareClass[name, methodsIn, Normal[fields], opts0]

DeclareClass[name_, methodsIn_?AssociationQ, fields0_?ListQ, opts0___] :=
    Module[{bindSelf, cls, methods, opts = Association[opts0], extends, extendedClasses, accessors, predicateName, fields},

    	(* static analyze methods to see if they call down to valid functions *)
        checkDeclaredMethods[name, methodsIn];
        checkDeclaredFields[name, fields0];

        bindSelf[key_ -> val_] := key-> val;
        methods = Join[methodsIn];

        extends = Lookup[opts, Extends, {ClassTrait[<||>]}];
        If[MatchQ[extends, _ClassTrait | _Class],
        		extends = {extends}
        ];
		If[!AllTrue[extends, MatchQ[#, _ClassTrait | _Class]&],
			With[{e = Select[extends, !MatchQ[#, _ClassTrait]&]},
				ThrowException[{"Invalid trait passed in while creating class ", name, "  ", e}]
			]
		];
		
		extendedClasses = Cases[extends, _Class];

        fields = Join[
	        	getExtendFields[extends],
	        	Association[
	        		Replace[fields0, field:Except[_Rule] :> field -> Undefined, 1]
	        	]
        ];
        
        accessors = AssociationMap[makeAccessor, fields];
        
        (*
         Now create the predicate. Only do this if the setting is a non-system context symbol.
        *)
        predicateName = Lookup[opts, Predicate, Undefined];
        If[Head[predicateName] === Symbol && Context @@ {predicateName} =!= "System`",
        		makePredicate[predicateName, name]
        ];
        
        cls = Class[name,
	        AssociationMap[
	        	bindSelf,
	        	Apply[
	        		Join,
	        		Join[
	        			{$baseMethods}, (**< This needs to be first, since the traits and methods override it *)
	        			{accessors},
	        			getExtendMethods[extends],
						If[predicateName =!= Undefined,
	        				With[ {predName = predicateName},
	        					{<| "_predicate" -> Function[{}, predName] |>}],
	        				{}
	        			],
	        			{methods}
	        		]
	        	]
	        ],
	        methods,
	        fields
        ];
        AppendTo[$Classes, name];
        setupClassLookup[name, cls, extendedClasses];
        cls
    ]
DeclareClass[args___] :=
	ThrowException[{"Unrecognized call to DeclareClass", args}]


(*
Check the methods and make sure that they are not calling undefined functions,

Returns True if all checked methods are ok
*)
checkDeclaredMethods[name_, methods_] :=
Module[{callees},
	callees = KeyValueMap[

		Replace[#2, {
		(*
		These are all just common patterns for methods
		More valid patterns may be used in the future, and this list would need to be updated.

		We inspect each method definition, taking care to not evaluate anything.
		If the definition is another function call, then we examine it further.

		*)

		(* get the more specific patterns out of the way first *)
		HoldPattern[Function][{___}|PatternSequence[], Self[___][___]] -> Nothing,
		HoldPattern[Function][{___}|PatternSequence[], Self[___]] -> Nothing,
		HoldPattern[Function][{___}|PatternSequence[], Self] -> Nothing,
		HoldPattern[Function][{___}|PatternSequence[], Verbatim[Slot][1][___]] -> Nothing,
		HoldPattern[Function][{___}|PatternSequence[], True|False|Null|None] -> Nothing,
		HoldPattern[Function][{___}|PatternSequence[], _String] -> Nothing,
		HoldPattern[Function][{___}|PatternSequence[], _Integer] -> Nothing,

		(* anything left with this structure is further examined *)
		HoldPattern[Function][{___}|PatternSequence[], callee_[___][___]] :> (#1 -> Hold[callee]),
		HoldPattern[Function][{___}|PatternSequence[], callee_[___]] :> (#1 -> Hold[callee]),

		_ :> ThrowException[{"Unrecognized pattern in checkMethods", name, #1 -> #2}]

		}]&, methods];

	AllTrue[callees, examine[name, #]&]
]

(*
examine whether this is valid function call.
There are some cases recognized:
1. A System` function call (we assume calling a System` function is correct)
2. A call to a symbol that has DownValues (we assume calling a symbol with DownValues is correct)
3. A call to a symbol that has OwnValues (we could dig down further to see if the OwnValues needs examining, etc.)
4. A Internal`WithLocalSettings call (probably from static analysis tools)
*)
examine[className_, methodName_ -> heldCallee_] :=
Module[{heldHead},
	heldHead = Extract[heldCallee, {1, 0}, Hold];
	Which[
		heldHead === Hold[Symbol] && (Context @@ heldCallee === "System`"),
			True
		,
		heldHead === Hold[Symbol] &&
				(DownValues @@ heldCallee =!= {}),
			True
		,
		heldHead === Hold[Symbol] &&
				(OwnValues @@ heldCallee =!= {}),
			True
		,
		heldCallee === Hold[Internal`WithLocalSettings] || heldCallee === Hold[RuntimeTools`ProfileDataWrapper],
			True
		,
		True,
			ThrowException[{"Cannot find function definition", className, methodName -> heldCallee}]
	]
]

checkDeclaredFields[name_, fields_] :=
Module[{},
	If[!DuplicateFreeQ[fields],
		ThrowException[{"Duplicate fields found", name, fields}]
	]
]





CreateObject[ name_] :=
		CreateObject[ name, <||>]
		
CreateObject[ name_, st_Association] :=
	Module[ {obj},
		obj = Compile`Utilities`Class`Impl`CreateObject[ name, Normal[st]];
		If[FailureQ[obj],
			obj
		,
			obj["initialize"];
			obj]
	]



createObjectInstance[ name_, st0_Association, fields_] :=
	Module[{st, obj},
		st = checkFields[st0, fields];
		st = CreateReference[st];
		obj = ObjectInstance[name,st];
		obj["initialize"];
		obj
	];


setupClassLookup[ name_, cls_, extendedClasses_] :=
	Module[ {methods, fields},
		methods = Part[ cls, 2];
		fields = Part[ cls, 4];
		Compile`Utilities`Class`Impl`CreateClass[ name, Normal[methods], Normal[fields]];
	]



Unprotect[ Compile`Utilities`Class`Impl`ClassErrorHandler]

Compile`Utilities`Class`Impl`ClassErrorHandler["noname", key_, className_] :=
	ThrowException[{"Cannot access ", key, " for class ", SymbolName[className]}]


 makePredicate[ predicateName_, name__] :=
	Module[{},
		predicateName[obj_ /; (ObjectInstanceQ[obj] && obj["_class"] === name)] := True;
		predicateName[___] := False
	]

   

(*
  If there are fields in obj that don't appear in class then throw an exception.
  Otherwise use the class fields to give defaults.
*)
checkFields[objF_, classF_] :=
	Module[ {fields},
		fields = Join[classF, objF];
		If[ Complement[ Keys[fields], Keys[classF]] =!= {},
			ThrowException[{"Object with incompatible fields", {"accessing" -> Complement[ Keys[fields], Keys[classF]], "objectFields" -> Keys[objF], "classFields" -> Keys[classF]}}]
		];
		fields
	]


makeAccessor[key_ -> val_] :=
	If[ StringQ[key],
        <|
            "get" <> camelCase[key] -> Compile`Utilities`Class`Impl`GetData[key],
            "set" <> camelCase[key] -> Compile`Utilities`Class`Impl`SetData[key]
        |>,
        <||>	     
	]



SetAttributes[SetData, HoldFirst];

SetData[inst_[key_], val_] := 
	Compile`Utilities`Class`Impl`SetObjectField[inst, key, val]


SetData1[ObjectInstance[nm_, st_], key_, val_] :=
	(
		st["associateTo", key -> val];
		val
	)



(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)

classIcon := Graphics[
	Text[
		Style["CLS", GrayLevel[0.7], Bold, 1.1*CurrentValue["FontCapHeight"]/ AbsoluteCurrentValue[Magnification]]
	],
	$FormatingGraphicsOptions
];    
Class /: MakeBoxes[cls:Class[name_, boundMethods_, methods_], fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"Class",
		cls,
  		classIcon,
  		{
  		    BoxForm`SummaryItem[{"name: ", ToString[name]}],
  		    BoxForm`SummaryItem[{"methods: ", methods}]
  		},
  		{}, 
  		fmt,
		"Interpretable" -> False
  	]
  	
objectInstanceIcon := Graphics[Text[
  Style["CI*", GrayLevel[0.7], Bold, 
   1.2*CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];   

Unprotect[ObjectInstance];      
ObjectInstance /: MakeBoxes[obj_ObjectInstance, fmt_] :=
	If[ ObjectInstanceQ[obj], 
		obj["toBoxes", fmt], 
		With[ {obj1 = Apply[ "ObjectInstance", obj]}, MakeBoxes[ obj1, fmt]]]


ObjectInstanceQ[args__] := Compile`Utilities`Class`Impl`ObjectInstanceQ[args];


toBoxes[inst_?ObjectInstanceQ, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"ObjectInstance",
		inst,
  		objectInstanceIcon,
  		{
  		    CompileInformationPanel[inst["_class"], Normal[inst["_state"]]]
  		},
  		{
  		    BoxForm`SummaryItem[{"methods: ", inst["_methods"]}]
  		}, 
  		fmt,
		"Interpretable" -> False
  	]




toString[inst_?ObjectInstanceQ] :=
	StringJoin[
		ToString[inst["_class"]],
		"[",
		ToString[inst["_state"]],
		"]"
	]

fullform[inst_?ObjectInstanceQ] :=
	inst["_class"][]

	
Unprotect[ Compile`Utilities`ClassSystem`ObjectInstance]
Format[ r_Compile`Utilities`ClassSystem`ObjectInstance, OutputForm] := OutputForm[r["toString"]]
Format[ r_Compile`Utilities`ClassSystem`ObjectInstance, InputForm] := OutputForm[r["toString"]]
Format[ r_Compile`Utilities`ClassSystem`ObjectInstance, ScriptForm] := OutputForm[r["toString"]]

Unprotect[ Compile`Utilities`Class`Impl`ObjectInstance]
Format[ r_Compile`Utilities`Class`Impl`ObjectInstance, OutputForm] := OutputForm[r["toString"]]
Format[ r_Compile`Utilities`Class`Impl`ObjectInstance, InputForm] := OutputForm[r["toString"]]
Format[ r_Compile`Utilities`Class`Impl`ObjectInstance, ScriptForm] := OutputForm[r["toString"]]


Protect[ObjectInstance];	

(*********************)
(*********************)
(*********************)

(*
  Slightly strange name to avoid a collision with a System` context symbol called
  getProperty introduced into the Cloud.
*)
getProperty1[self_, prop_] := self["properties"]["lookup", prop]
getProperty1[self_, prop_, default_] := self["properties"]["lookup", prop, default]

cloneProperties[self_, obj_] :=
	self["setProperties", obj["properties"]["clone"]];

joinProperties[self_, other_] :=
	self["setProperties", self["properties"]["join", other]];

ClassPropertiesTrait = ClassTrait[<|
        "getProperties" -> Function[{}, Self["properties"]["get"]], (**< overload the default accessor *)
        "getProperty" -> (getProperty1[Self, ##]&),
        "removeProperty" -> Function[{prop}, Self["properties"]["keyDropFrom", prop]],
        "setProperty" -> Function[{kv}, Self["properties"]["associateTo", kv]],
        "joinProperties" -> Function[{other}, joinProperties[ Self, other]],
        "cloneProperties" -> Function[{obj}, cloneProperties[Self, obj]],
        "hasProperty" -> Function[{key}, Self["properties"]["keyExistsQ", key]],
        "clonedProperties" -> Function[{},
                CreateReference[
                        Map[
                                Function[{val},
                                        If[ReferenceQ[val],
                                                val["clone"],
                                                val
                                        ]
                                ],
                                Self["properties"]["get"]
                        ]
                ]
        ]
|>]



End[]

EndPackage[]
