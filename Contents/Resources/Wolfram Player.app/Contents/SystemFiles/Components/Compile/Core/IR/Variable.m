BeginPackage["Compile`Core`IR`Variable`"]

VariableQ;
CreateVariable;
VariableClass
DeserializeVariable

Begin["`Private`"] 

Needs["CompileAST`Create`Construct`"]
Needs["Compile`Core`IR`Internal`Utilities`"]
Needs["CompileAST`Class`Base`"]
Needs["Compile`Core`IR`Internal`VariableTrait`"]
Needs["Compile`Core`IR`Internal`Show`"]
Needs["Compile`Utilities`Serialization`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Markup`"]



If[!IntegerQ[$NextVariableId],
	$NextVariableId = 1
]

sameQ[self_, other_] :=
	VariableQ[other] && self["id"] === other["id"]



RegisterCallback["DeclareCompileClass", Function[{st},
VariableClass = DeclareClass[
	Variable,
	<|
		"initialize" -> Function[{},
			Self["setProperties", CreateReference[<||>]];
		],
		"serialize" -> (serialize[Self, #]&),
		"clone" -> (clone[Self, ##]&),
		"sameQ" -> Function[{other}, sameQ[Self, other]],
		"rename" -> Function[{name}, Self["setId", name]; Self],
		"typename" -> Function[{}, Self["type"]["toString"]],
		"dispose" -> Function[{}, dispose[Self]],
		"makePrettyPrintBoxes" -> Function[{}, makePrettyPrintBoxes[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"lhsToString" -> Function[{}, lhsToString[Self]],
		"toBoxes" -> Function[{fmt},
			toBoxes[Self, fmt]
		]
	|>,
	{
		"id" -> 0,
		"name",
		"type" -> Undefined,
		"mexpr" -> None,
		"_defs" -> {}, (**< Instruction ids that define the variable.
				            We use Ids to allow for self refernence.
				            We also make this a list to allow us to handle.
				            Non-SSA form. Syntax sugar is added to pretend there is
				            only one definition
				         *)
		"uses" -> {} , (**< Instruction ids that use the variable *)
		"properties"
	},
	Predicate -> VariableQ,
	Extends -> {
		ClassPropertiesTrait,
		VariableTrait
	}
]
]]


sameQ[self_, other_] := VariableQ[other] &&
						self["id"] === other["id"]

CreateVariable[] :=
        CreateVariable[$NextVariableId, None]

CreateVariable[name_String] :=
	Module[{var},
		var = CreateObject[
			Variable,
			<|
				"id" -> $NextVariableId,
				"name" -> name,
				"type" -> Undefined
			|>
		];
		$NextVariableId++;
		var
	]

CreateVariable[s_Symbol] :=
	With[{mexpr = CreateMExpr[s]},
		CreateVariable[mexpr]
	]
CreateVariable[mexpr_?MExprQ] :=
	CreateVariable[$NextVariableId, mexpr]
	
CreateVariable[id_?IntegerQ, mexpr_:None] :=
	CreateVariable[id, Undefined, mexpr]
	
CreateVariable[type_?TypeObjectQ, mexpr_:None] :=
	CreateVariable[$NextVariableId, type, mexpr]
	
CreateVariable[id_, type_, mexpr_] :=
	Module[{var},
		var = CreateObject[
			Variable,
			<|
				"id" -> id,
				"name" -> "var" <> ToString[id],
				"type" -> type,
				"mexpr" -> mexpr
			|>
		];
		$NextVariableId = Max[$NextVariableId + 1, id + 1];
		var
	]

CreateVariable[args___]:=
	ThrowException[{"Unrecognized call to CreateVariable", {args}}]



DeserializeVariable[ env_, "Variable"[ data_]] :=
	deserialize[ env, data]

deserialize[env_, data_]  :=
	Module[ {var, id, newId, type},
		id = data["id"];
		var = env["getVariable", id];
		If[ var === Null,
			var = CreateVariable[ data["name"]];
			type = env["getType", data["type"]];
			var["setType", type];
			newId = If[ env["uniqueID"], $NextVariableId++, id];
			var["setId", newId];
			env["setVariable", id, var]
			, (* Else *)
			If[data["name"] =!= var["name"] || !sameDeserializedType[env, data["type"], var["type"]],
				ThrowException[{"Cached variable does not match", data, var}]
			]
		];
		var
	]

ClearAll[sameDeserializedType]
sameDeserializedType[env_, Undefined, Undefined] := True 
sameDeserializedType[env_, t1_, t2_] :=
	env["typeSameQ", t1, t2]

serialize[ self_, env_] :=
	Module[ {ty},
		ty = self["type"];
		"Variable"[<|"name" -> self["name"], "id" -> self["id"], "type" -> env["serializeType", ty]|>]
	]


clone[self_, env_] :=
	deserialize[ env, self]
	
clone[self_] :=
	deserialize[ CreateCloneEnvironment[], self]



dispose[self_] :=
	Module[{},
		self["properties"]["set", <||>];
		self["clearDef"];
		self["clearUses"];
	]



toString[var_] := BoldGreenText[StringJoin[
	"%",
	ToString[var["id"]]
]]

lhsToString[var_] := StringJoin[
	toString[var],
	":",
	BoldRedText[IRFormatType[var["type"]]]
]
	
(**************************************************)
(**************************************************)



(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)

icon := Graphics[Text[
  Style["Var", GrayLevel[0.7], Bold, 
   1.2*CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];  

typeName[Undefined] = Undefined
typeName[TypeSpecifier[Undefined]] = Undefined
typeName[TypeSpecifier[t__]] = t
typeName[Type[Undefined]] = Undefined
typeName[Type[t__]] = t
typeName[t_] := t["name"]

(*
  Maybe add scope info here.
*)       
toBoxes[var_?VariableQ, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"Variable",
		var["toString"],
  		icon,
  		{
  		    BoxForm`SummaryItem[{"id: ", var["id"]}],
  		    If[var["name"] === Undefined,
  		    	Nothing,
  		    	BoxForm`SummaryItem[{"name: ", var["name"]}]
  		    ],
  		    BoxForm`SummaryItem[{"type: ", typeName[var["type"]]}],
  		    If[var["mexpr"] === None,
  		    		Nothing,
  		    		BoxForm`SummaryItem[{"mexpr: ", var["mexpr"]["toString"]}]
  		    ]
  		},
		{
  		    If[var["uses"] === {},
  		    	Nothing,
  		    	BoxForm`SummaryItem[{"uses: ", #["toString"]& /@ var["uses"]}]
  		    ],
  		    If[var["defs"] === {},
  		    	Nothing,
  				BoxForm`SummaryItem[{"defs: ", #["toString"]& /@ var["defs"]}]
  		    ]
		},
  		fmt
  	]


makeInformationPanel[var_] :=
	ToBoxes[
		CompileInformationPanel[
			"Variable"
			, {
			"id: " -> var["id"],
	  		"type: " -> typeName[var["type"]],
	  		"mexpr: " -> If[var["mexpr"] === None,
	  			Nothing,
				var["mexpr"]["toString"]
	  		],
	  		"uses: " -> StringRiffle[#["toString"]& /@ var["uses"], "\n"],
	  		"defs: " -> StringRiffle[#["toString"]& /@ var["defs"], "\n"]
		}]
	]
makePrettyPrintBoxes[self_] :=
	With[{box = StyleBox["%" <> ToString[self["id"]] <> "", Bold, $VariableColor]},
	    TooltipBox[box, makeInformationPanel[self]]
	]

End[]

EndPackage[]
