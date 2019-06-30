
BeginPackage["TypeFramework`TypeObjects`TypeBase`"]

TypeBaseClassQ
TypeBaseClass
GetNextTypeId
RegisterTypeObject

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]


$NextTypeId

If[!ValueQ[$NextTypeId],
	$NextTypeId = 1
]

GetNextTypeId[] :=
	$NextTypeId++

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeBaseClass = DeclareClass[
	TypeBase,
	<|
		"initialize" -> Function[{},
			If[Self["properties"] === Undefined,
				Self["setProperties", CreateReference[<||>]]
			]
		],
		"sameQ" -> (sameQ[Self, ##]&),
		"clone" -> (clone[Self, ##]&),
		"unresolve" -> Function[ {}, unresolve[Self]],
		"implementsType" -> (implementsType[Self, ##]&),
		"name" -> (Self["format", False]&),
		"reify" -> (Self&),
        "canonicalize" -> (canonicalize[Self, ##]&),
		"shortname" -> (Self["format", True]&),
		"isNamedApplication" -> (isNamedApplication[Self, ##]&),
		"isConstructor" -> (isConstructor[Self, ##]&),
		"implementsQ" -> (implementsQ[Self, ##]&),
		"addImplementation" -> Function[{imp},
			Self["setImplements", Append[Self["implements"], imp]]
		],
        "toScheme" -> (unimplemented[Self, "toScheme"]&),
		"computeFree" -> (computeFree[Self]&),
		"free" -> (free[Self]&),
		"variableCount" -> (variableCount[Self]&),
		"kindOf" -> (unimplemented[Self, "kind"]&),
		"generalize" -> (generalize[Self,##]&),
		"accept" -> Function[{vst}, accept[Self, vst]]
	|>,
	{
		"id" -> -1,
		"implements" -> {},
		"baseImplements" -> {},
		"shortnameoverride" -> Null,
		"properties" -> Undefined,
		"freeCache" -> Null
	},
(*
 Set a predicate with an internal name that will not be called.
 This class is not designed to be instantiated.
*)
	Predicate -> nullTypeBaseClassQ,
	Extends -> {
		ClassPropertiesTrait
	}
]
]]


(*
  Add functionality for TypeBaseClassQ
*)
If[!AssociationQ[$typeBaseObjects],
    $typeBaseObjects = <||>
];

RegisterTypeObject[ name_] :=
	AssociateTo[$typeBaseObjects, name -> True]
	
TypeBaseClassQ[ obj_] :=
	ObjectInstanceQ[obj] && KeyExistsQ[$typeBaseObjects, obj["_class"]]


throwError[ self_, txt_] :=
	ThrowException[{txt <> self["_class"]}]


unresolve[ self_] :=
	throwError[ self, "Unimplemented unresolve method for "]

unimplemented[self_, method_] :=
	ThrowException[{"the class " <> ToString[self["_class"]] <> " does not implement " <> method <> " make sure the method is overloaded"}]

free[self_] :=
	(
	If[self["freeCache"] === Null,
		self["setFreeCache", self["computeFree"]]];
	self["freeCache"]
	)

computeFree[self_] :=
	<||>


variableCount[self_] :=
	0

generalize[self_] :=
	generalize[self, {}]
	
	
generalize[self_, fvs_] :=
    With[{args = Complement[Values[self["free"]], fvs]},
        If[args === {},
            self,
            CreateTypeForAll[args, self]
        ]
    ]


canonicalize[self_] :=
    self;
canonicalize[self_, idx_] :=
    self;


accept[self_, ___] :=
	throwError[ self, "Unimplemented accept method for "]
	
sameQ[self_, ___] :=
	throwError[ self, "Unimplemented sameQ method for "]
	
clone[self_, ___] :=
	throwError[ self, "Unimplemented clone method for "]

implementsType[self_, ty_] :=
	MemberQ[self["reify"]["implements"], ty]
	
isNamedApplication[self_, name_] :=
	False

isConstructor[self_, name_] :=
	False

implementsQ[self_, name_] :=
	False

End[]

EndPackage[]

