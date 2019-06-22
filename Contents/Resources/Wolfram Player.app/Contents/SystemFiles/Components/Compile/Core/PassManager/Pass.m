
BeginPackage["Compile`Core`PassManager`Pass`"]

Pass;
PassQ;
CreatePass;

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



requiredFields = {
	"information"
}

optionalFields = {
	"traversalOrder",
	"requires",
	"postPasses",
	"preserves",
	"passClass"
}

requiredMethods = {
	"runPass"
}

optionalMethods = {
	"initializePass",
	"finalizePass",
	"verifyPass",
	"logger",
	"constraint"
}

allFields = Union[requiredFields, optionalFields];
allMethods = Union[requiredMethods, optionalMethods];

RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[
	Pass,
	<|
		"toString" -> Function[{}, Self["information"]["toString"]],
		"initialize" -> Function[{},
			Self["setProperties", CreateReference[<||>]];
		],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	Join[
		{
			"name",
			"description",
			"properties",
			"_passtype"
		},
		allFields,
		allMethods
	],
	Predicate -> PassQ,
	Extends -> {
		ClassPropertiesTrait   (* Needed for properties *)
	}
]
]]

CreatePass[passType_, data0_?AssociationQ] :=
	Module[{allMethods, methods, allFields, fields, info, data = data0},
		If[Complement[
				Keys[data],
				Union[requiredFields, optionalFields, requiredMethods, optionalMethods]
			] =!= {},
			With[{
			   extra = Complement[
					Keys[data],
					Union[requiredFields, optionalFields, requiredMethods, optionalMethods]
				]
			},
			ThrowException[{"The input fields ", data, " do not conform to the Pass arguments. Extra = ", extra}]
			]
		];
		allFields = Union[requiredFields, optionalFields];
		allMethods = Union[requiredMethods, optionalMethods];
		fields = Intersection[Keys[data], allFields];
		methods = Intersection[Keys[data], allMethods];
		data["traversalOrder"] = Lookup[data, "traversalOrder", "anyOrder"];
		data["requires"] = Lookup[data, "requires", {}];
		data["postPasses"] = Lookup[data, "postPasses", {}];
		data["preserves"] = Lookup[data, "preserves", {}];
		data["verifyPass"] = Lookup[data, "verifyPass", Function[{}, True]];
		data["passClass"] = Lookup[data, "passClass", "Transform"];
		data["constraint"] = Lookup[data, "constraint", True&];
		info = data["information"];
		CreateObject[
			Pass,
			Join[
				<|
					"name" -> info["name"],
					"description" -> info["description"],
					"_passtype" -> passType
				|>,
				data
			]
		]
	]


(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)

icon[name_] := Graphics[Text[
  Style[name, GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];  


getPassType[obj_] :=
	Module[{name},
		name = obj["_passtype"];
		If[Head[name] === Symbol,
			SymbolName[name],
			ToString[name]
		]
	]

 
toBoxes[var_?PassQ, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		getPassType[var],
		var,
  		icon[getPassType[var]],
  		{
  		    BoxForm`SummaryItem[{"name: ", var["name"]}],
  		    BoxForm`SummaryItem[{"information: ", var["information"]["information"]}]
  		},
  		{
  		    BoxForm`SummaryItem[{"description: ", var["description"]}]
  		}, 
  		fmt,
		"Interpretable" -> False
  	]
End[]

EndPackage[]
