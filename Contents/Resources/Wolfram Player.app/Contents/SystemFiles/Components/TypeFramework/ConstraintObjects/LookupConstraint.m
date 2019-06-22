
BeginPackage["TypeFramework`ConstraintObjects`LookupConstraint`"]

CreateLookupConstraint
LookupConstraintQ

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`"]
Needs["TypeFramework`Inference`Unify`"]
Needs["TypeFramework`TypeObjects`TypeQualified`"]
Needs["TypeFramework`ConstraintObjects`ConstraintBase`"]
Needs["TypeFramework`Inference`Substitution`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]


RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
LookupConstraintClass = DeclareClass[
	LookupConstraint,
	<|
        "active" -> (active[Self]&),
        "computeFree" -> Function[{}, computeFree[Self]],
        "hasShape" -> (hasShape[Self, ##]&),
        "solve" -> (solve[Self, ##]&),
		"monitorForm" -> (lookupMonitorForm[Self, ##]&),
		"judgmentForm" -> (judgmentForm[Self]&),
		"format" -> (format[Self,##]&),
        "unresolve" -> (unresolve[Self]&),
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"id",
        "monomorphic",
		"name",
		"type",
		"initial"
	},
	Predicate -> LookupConstraintQ,
    Extends -> BaseConstraintClass
];
RegisterConstraint[ LookupConstraint];
]]


CreateLookupConstraint[name_, type_] :=
    CreateLookupConstraint[name, type, {}]
    
CreateLookupConstraint[name_, type_, m_?ListQ] :=
	CreateObject[LookupConstraint, <|
		"id" -> GetNextConstraintId[],
        "monomorphic" -> m,
        "name" -> name,
        "type" -> type,
        "initial" -> type
    |>]



computeFree[self_] :=
	self["type"]["free"]

active[self_] :=
    0;


sameQ[a_, b_] :=
    a["sameQ", b];

 
solve[self_, st_] :=
	Module[ {monoDefs, polyDefs, defs, def, constraint, name, ty},
		name = self["name"];
		ty = self["type"];
		def = st["typeEnvironment"]["functionTypeLookup"]["getMonomorphic", name, ty];
		If[ def === Null,
			polyDefs = st["typeEnvironment"]["functionTypeLookup"]["getPolymorphicList", name];
			monoDefs = st["typeEnvironment"]["functionTypeLookup"]["getMonomorphicList", name];
			defs = Join[ monoDefs, polyDefs];
			If[defs === {},
	    		TypeFailure[
               	 	"LookupNotFound",
                	"Cannot find a definition for `1`.",
                	name
		    	]
			];
	   		constraint = st["prependAlternativeConstraint", ty, self["initial"], defs, self["monomorphic"]];
	   		constraint["setProperty", "generalize" -> True];
	   		, (* Else *)
	   		constraint = st["appendEqualConstraint", ty, def];
	   		self["initial"]["setProperty", "resolvedType" -> def];
	   	];
	    constraint["setProperty", "source" -> self];
		constraint["setProperty", "lookupName" -> name];
	    {CreateTypeSubstitution["TypeEnvironment" -> st["typeEnvironment"]], {constraint}}
	]


(*
 Strip out TypeQualified.  This is a utility for hasShape which doesn't guarantee that 
 the types will match,  just that some have the right shape.
*)
stripQualified[ ty_?TypeQualifiedQ] :=
	ty["type"]

stripQualified[ty_] :=
	ty

(*
  Return True if type has the same shape of any of the definitions (determined by being unifiable) and False otherwise.
  At present we reject if it there any polymorphic elements,  but maybe that should be allowed as well.
*)
hasShape[self_, tyEnv_] :=
	Module[ {name, ty, polyDefs, monoDefs, defs},
		name = self["name"];
		ty = self["type"];
		polyDefs = tyEnv["functionTypeLookup"]["getPolymorphicList", name];
		monoDefs = tyEnv["functionTypeLookup"]["getMonomorphicList", name];
		If[
			AnyTrue[monoDefs,
				TypeUnifiableQ[ty, #, "TypeEnvironment" -> tyEnv]&],
			True
			,
			defs = Map[ #["instantiate", "TypeEnvironment" -> tyEnv]&, polyDefs];
			defs = Map[ stripQualified, defs];
			AnyTrue[defs,
				TypeUnifiableQ[ty, #, "TypeEnvironment" -> tyEnv]&]]
	]


lookupMonitorForm[ self_, sub_, rest_] :=
	ConstraintSolveForm[<|
	   "name0" -> ("Lookup"&),
	   "name" -> self["name"],
	   "type" -> self["type"],
       "monomorphic" -> self["monomorphic"],
	   "unify" -> sub,
	   "rest" -> rest
	|>]

judgmentForm[self_] :=
	StyleBox[
		GridBox[{
			{RowBox[{
				ToString[self["name"]], "\[RightPointer]", ToString[self["type"]["id"]]}]},
			{StyleBox[
				RowBox[{"(" <>
					ToString[self["properties"]["lookup", "source"]] <>
					")"}],
				FontSize -> Small]}
		}],
		FontFamily -> "Verdana"
	];

icon := icon =  Graphics[Text[
    Style["LK\nCONS",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
];

toBoxes[self_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"LookupConstraint",
		self,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["name: ",        {90, Automatic}], self["name"]}],
			BoxForm`SummaryItem[{Pane["type: ",        {90, Automatic}], self["type"]}],
            BoxForm`SummaryItem[{Pane["monomorphic: ", {90, Automatic}], self["monomorphic"]}]
  		},
  		{

  		},
  		fmt
  	]


toString[self_] :=
    StringJoin[
       "Lookup[",
       ToString[self["name"]],
       ",",
       self["type"]["toString"],
       "]"
    ];

unresolve[self_] :=
    LookupConstraint[<|
        "name" -> self["name"],
        "type" -> self["type"]["unresolve"],
        "initial" -> self["initial"]["unresolve"],
        "monomorphic" -> (#["unresolve"]& /@ self["monomorphic"])
    |>]

format[self_, shortQ_:True] :=
	Row[{
	   "Lookup[",
	   self["name"],
	   ",",
	   self["type"]["format", shortQ],
	   "]"
	}]

End[]

EndPackage[]
