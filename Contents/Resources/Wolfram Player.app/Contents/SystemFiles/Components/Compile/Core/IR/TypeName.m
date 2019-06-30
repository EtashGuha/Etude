
BeginPackage["Compile`Core`IR`TypeName`"]

TypeName;
TypeNameQ;
CreateTypeName;
TypeNameClass;

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Internal`Utilities`"]


If[!IntegerQ[nextId],
	nextId = 1
]

RegisterCallback["DeclareCompileClass", Function[{st},
TypeNameClass = DeclareClass[
	TypeName,
	<|
		"initialize" -> Function[{},
			Self["setId", nextId++];
			Self["setProperties", CreateReference[<||>]];
		],
		"sameQ" -> Function[{other}, TypeNameQ[other] && Self["id"] === other["id"]],
		"rename" -> Function[{name}, Self["setName", name]; Self],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt},
			toBoxes[Self, fmt]
		]
	|>,
	{
		"id" -> 0,
		"name" -> 0,
		"type" -> Undefined,
		"mexpr",
		"properties"
	},
	Predicate -> TypeNameQ,
	Extends -> {
		ClassPropertiesTrait
	}
]
]]


IRFormatType[t_?TypeNameQ] :=
	t["name"]

TypeInformation[t_?TypeNameQ] :=
	TypeInformation[t["type"]]
	
CreateTypeName[name_, type_, mexpr_:None] :=
	CreateObject[
		TypeName,
		<|
			"name" -> name,
			"type" -> type,
			"mexpr" -> mexpr
		|>
	]
	
toString[var_] := StringJoin[
	"T[",
	var["name"],
	"]",
	"(",
	ToString[var["id"]],
	")"
]


(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)

icon := Graphics[Text[
  Style["TN", GrayLevel[0.7], Bold, 
   1.2*CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];  

toBoxes[var_?TypeNameQ, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"TypeName",
		var,
  		icon,
  		{
  		    BoxForm`SummaryItem[{"id: ", var["id"]}],
  		    BoxForm`SummaryItem[{"name: ", var["name"]}],
  		    BoxForm`SummaryItem[{"info: ", TypeInformation[var]}]
  		},
  		{}, 
  		fmt
  	]
End[]

EndPackage[]
