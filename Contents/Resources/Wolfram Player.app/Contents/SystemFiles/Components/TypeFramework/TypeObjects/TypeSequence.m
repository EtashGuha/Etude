
BeginPackage["TypeFramework`TypeObjects`TypeSequence`"]

TypeSequenceQ
CreateTypeSequence


Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`"] (* TypeInferenceException *)
Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`Utilities`Error`"]


sameQ[self_, other_?TypeSequenceQ] :=
	self["id"] === other["id"] ||
	(
		self["min"] === other["min"] &&
		self["max"] === other["max"] &&
		self["type"]["sameQ", other["type"]]
	)
		
sameQ[___] := 
	Module[{},
		False
	]

format[ self_, shortQ_:True] :=
	Module[ {},
		If[ self["min"] === 0 && self["max"] === Infinity, 
			self["type"]["format", shortQ] <> "..."
			, 
			"TypeSequence[" <> self["type"]["format", shortQ] <> ",{" <> ToString[self["min"]] <> ",", ToString[self["max"]] <> "}]"]
	]

accept[ self_, vst_] :=
	vst["visitSequence", self]

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeSequenceClass = DeclareClass[
	TypeSequence,
	<|
        "computeFree" -> (computeFree[Self]&),
        "toScheme" -> (toScheme[Self]&),
		"sameQ" -> (sameQ[Self, ##]&),
		"clone" -> (clone[Self, ##]&),
		"unresolve" -> Function[ {}, unresolve[Self]],
		"accept" -> Function[{vst}, accept[Self, vst]],
		"hasBinding" -> Function[{}, Self["binding"] =!= None],
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	{
		"type",
		"min",
		"max",
		"binding"
	},
	Predicate -> TypeSequenceQ,
	Extends -> TypeBaseClass
];
RegisterTypeObject[TypeSequence];
]]

CreateTypeSequence[type_?TypeObjectQ] :=
	CreateTypeSequence[type, None, {0,Infinity}]
	
CreateTypeSequence[type_?TypeObjectQ, {min_, max_}] :=
	CreateTypeSequence[type, None, {min,max}]
	
	
isBinding[var_?TypeVariableQ] := True
isBinding[None] := True
isBinding[var_] := False


CreateTypeSequence[type_?TypeObjectQ, var_?isBinding, {min_?IntegerQ, max:(_?IntegerQ | Infinity)}] :=
	Module[ {},
		CreateObject[TypeSequence, <|
			"id" -> GetNextTypeId[],
			"type" -> type,
			"min" -> min,
			"max" -> max,
			"binding" -> var
		|>]
	]

CreateTypeSequence[args___] :=
	TypeFailure["TypeSequence", "Unrecognized call to CreateTypeSequence", args]



unresolve[ self_] :=
	If[ self["binding"] === None,
		TypeSpecifier[TypeSequence[ self["type"]["unresolve"], 
			{self["min"], self["max"]}]],
			TypeSpecifier[TypeSequence[ self["type"]["unresolve"], 
			self["binding"]["unresolve"],
			{self["min"], self["max"]}]]]
			


clone[self_] :=
	clone[self, CreateReference[<||>]];

clone[self_, varmap_] :=
	If[varmap["keyExistsQ", self["id"]],
		varmap["lookup", self["id"]],
		With[{
			ty = CreateTypeSequence[
				self["type"]["clone", varmap],
				If[self["binding"] === None, None, self["binding"]["clone", varmap]],
				{self["min"], self["max"]}
			]
		},
			ty["setProperties", self["properties"]["clone"]];
			ty
		]
	];



(**************************************************)


computeFree[self_] := 
	self["type"]["free"]

    
toScheme[self_] :=
    CreateTypeForAll[{}, self];
    
(**************************************************)

icon := Graphics[Text[
	Style["TSeq",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[typ_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"TypeSequence",
		typ,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["type: ", {90, Automatic}], typ["type"]}],
  		    BoxForm`SummaryItem[{Pane["binding: ", {90, Automatic}], typ["binding"]}],
  		    BoxForm`SummaryItem[{Pane["min: ", {90, Automatic}], typ["min"]}],
  		    BoxForm`SummaryItem[{Pane["max: ", {90, Automatic}], typ["max"]}]
  		},
  		{

  		}, 
  		fmt
  	]

toStringLimits[self_] :=
	If[ self["min"] === 0 && self["max"] === Infinity, {}, {"{", ToString[self["min"]], ",", ToString[self["max"]], "}"}]

toStringBinding[self_] :=
	If[ self["binding"] === None, {}, {self["binding"]["toString"]}]

toString[self_] := 
	Module[ {},
		StringJoin[
			"TypeSequence[",
				Riffle[  {
					self["type"]["toString"],
					toStringBinding[self],
					toStringLimits[self]
				}, ","],
			"]"
		]
	]
	(*	
		
		If[ self["min"] === 0 && self["max"] === Infinity, 
			self["type"]["toString"] <> ".."
			, 
			"TypeSequence[" <> self["type"]["toString"] <> ",{" <> ToString[self["min"]] <> ",", ToString[self["max"]] <> "}]"]
	]
*)
End[]

EndPackage[]

