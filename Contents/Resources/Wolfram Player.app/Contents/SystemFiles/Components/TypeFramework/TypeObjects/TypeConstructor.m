
BeginPackage["TypeFramework`TypeObjects`TypeConstructor`"]

TypeConstructorQ
CreateTypeConstructor
TypeBottomQ
TypeTopQ
TypeConstructorObject

Begin["`Private`"]

Needs["TypeFramework`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["TypeFramework`TypeObjects`Kind`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]



sameQ[self_, other_?TypeConstructorQ] :=
	self["id"] === other["id"] ||
	(
		self["name"] === other["name"] &&
		self["kind"]["sameQ", other["kind"]]
	)
		
sameQ[___] := 
	Module[{},
		False
	]

TypeBottomQ[ty_?TypeConstructorQ] := ty["getProperty", "Bottom"];
TypeBottomQ[___] := False;

TypeTopQ[ty_?TypeConstructorQ] := ty["getProperty", "Top"];
TypeTopQ[___] := False;

format[ self_, shortQ_:True] :=
	Module[ {over},
		StringJoin[
			If[shortQ, 
				over = self["shortnameoverride"];
				If[ StringQ[over],
					over,
					StringTake[self["typename"], 1]
			    ], 
				self["typename"]
		    ]
		    (*,
		    If[hasFields[self],
		    	Echo@StringJoin[
		    		"<|",
		    		StringRiffle[
		    			KeyValueMap[
			    			Function[{k, v},
			    				{ToString[k], "\[Rule]", ToString[v]} 
			    			],
			    			Lookup[self["getProperty", "metadata"], "Fields"]
			    		],
			    		", "
		    		],
		    		"|>"
		    	],
		    	""
		    ]*)
		]
	]

accept[ self_, vst_] :=
	vst["visitConstructor", self]

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeConstructorClass = DeclareClass[
	TypeConstructorObject,
	<|
        "computeFree" -> (computeFree[Self]&),
        "toScheme" -> (toScheme[Self]&),
		"sameQ" -> (sameQ[Self, ##]&),
		"clone" -> (clone[Self, ##]&),
		"unresolve" -> Function[ {}, unresolve[Self]],
		"accept" -> Function[{vst}, accept[Self, vst]],
		"isConstructor" -> (isConstructor[Self, ##]&),
		"implementsQ" -> (implementsQ[Self, ##]&),
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	{
		"implements",
		"kind",
		"typename"
	},
	Predicate -> TypeConstructorQ,
	Extends -> TypeBaseClass
];
RegisterTypeObject[TypeConstructorObject];
]]


CreateTypeConstructor[typename_Symbol, kind_?KindQ, opts_:<||>] :=
	CreateTypeConstructor[SymbolName[typename], kind, opts]
	
CreateTypeConstructor[typename_String, kind_?KindQ, opts_:<||>] :=
	CreateTypeConstructor[TypeConstructor[typename, kind, opts]]

CreateTypeConstructor[ TypeConstructor[ typename_String, kind_?KindQ, opts_:<||>]] :=
	With[{
        shortnameoverride = Lookup[opts, "ShortName", Null],
        implements = Flatten[{Lookup[opts, "Implements", {}]}],
        metadata = Lookup[ opts, MetaData, Null]
	},
	With[{
		tycon = CreateObject[TypeConstructorObject, <|
			"id" -> GetNextTypeId[],
			"implements" -> implements,
			"typename" -> typename,
			"kind" -> kind,
			"shortnameoverride" ->  shortnameoverride
		|>]
	},
		tycon["setProperties", CreateReference[opts]];
		If[ metadata =!= Null,
			tycon["setProperty", "metadata" -> metadata]
		];
		tycon["setProperty", "Bottom" -> Lookup[opts, "Bottom", False]];
		tycon["setProperty", "Top" -> Lookup[opts, "Top", False]];
		If[KeyExistsQ[opts, "ByteCount"],
			tycon["setProperty", "ByteCount" -> Lookup[opts, "ByteCount"]]
		];
		tycon
	]]

CreateTypeConstructor[args___] :=
	ThrowException[{"Unrecognized call to CreateTypeConstructor", {args}}]



unresolve[ self_] :=
	TypeSpecifier[self["name"]]

clone[self_] :=
	clone[self, CreateReference[<||>]];

clone[self_, varmap_] :=
	With[{
		ty = CreateTypeConstructor[
			self["typename"],
			self["kind"],
			<|
				"shortnameoverride" -> self["shortnameoverride"]
			|>
		]
	},
		ty["setProperties", self["properties"]["clone"]];
		ty
	];

isConstructor[self_, name_] :=
	self["typename"] === name



implementsQ[self_, AbstractType[ name_]] :=
	implementsQ[self, name]

implementsQ[self_, name_String] :=
	MemberQ[ self["implements"], name]

implementsQ[args___] :=
    ThrowException[TypeInferenceException[{"Unknown argument to implementsQ", {args}}]]


(**************************************************)


computeFree[self_] :=
	<||>

toScheme[self_] :=
    CreateTypeForAll[{}, self];
    
(**************************************************)

hasFields[typ_] :=
	AssociationQ[typ["getProperty", "metadata"]] &&
  	AssociationQ[Lookup[typ["getProperty", "metadata"], "Fields"]]

icon := Graphics[Text[
	Style["TCon",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
      
toBoxes[typ_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"TypeConstructor",
		typ,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["name: ", {90, Automatic}], typ["name"]}],
  		    BoxForm`SummaryItem[{Pane["kind: ", {90, Automatic}], typ["kind"]["toString"]}]
  		},
  		{
  		BoxForm`SummaryItem[{Pane["implements: ", {90, Automatic}], typ["implements"]}],
  		(* If the constructor is a struct like object, then show its fields *)
  		If[hasFields[typ],
  			BoxForm`SummaryItem[{Pane["fields: ", {90, Automatic}], 
  				CompileInformationPanel["Fields", Normal@Lookup[typ["getProperty", "metadata"], "Fields"]]}
  			]
  			, (* Else *)
  			Nothing
  		]
  		}, 
  		fmt
  	]


toString[typ_] := typ["name"]

End[]

EndPackage[]

