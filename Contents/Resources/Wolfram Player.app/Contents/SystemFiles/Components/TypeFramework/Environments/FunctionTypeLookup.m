
BeginPackage["TypeFramework`Environments`FunctionTypeLookup`"]

FunctionTypeLookupQ
CreateFunctionTypeLookup

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]



RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
FunctionTypeLookupClass = DeclareClass[
	FunctionTypeLookup,
	<|
		"addType" -> Function[{name, ty}, addType[Self, name, ty]],
		"addExcludedType" -> Function[{name, ty}, addExcludedType[Self, name, ty]],
		"addMonomorphic" -> Function[{name, ty}, addMonomorphic[Self, name, ty]],
		"hasTypes" -> Function[{name}, hasTypes[Self, name]],
		"getPolymorphicList" -> Function[{name}, getPolymorphicList[Self, name]],
		"getMonomorphic" -> Function[{name, args}, getMonomorphic[Self, name, args]],
		"getMonomorphicList" -> Function[{name}, getMonomorphicList[Self, name]],
		"dispose" -> Function[{}, dispose[Self]],
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"excluded",   (*  Types that have been excluded by the FunctionDefinition mechanism,  eg no implementation *)
		"polymorphic",   (*  Polymorphic types *)
		"monomorphic"    (*  Monomorphic types *)
	},
	Predicate -> FunctionTypeLookupQ
]
]]


addMonomorphic[self_, name_, ty_?TypeArrowQ] :=
	Module[ {vals, args, key},
		vals = self["monomorphic"]["lookup", name, Null];
		If[vals === Null,
			vals = CreateReference[<||>];
			self["monomorphic"]["associateTo", name -> vals]];
		args = ty["arguments"];
		key = Map[#["unresolve"]&, args];
		vals["associateTo", key -> ty]
	]
	
addMonomorphic[self_, name_, ty_] :=
	Module[ {vals, key},
		vals = self["monomorphic"]["lookup", name, Null];
		If[vals === Null,
			vals = CreateReference[<||>];
			self["monomorphic"]["associateTo", name -> vals]];
		key = ty["unresolve"];
		vals["associateTo", key -> ty]
	]
	
getMonomorphic[self_, name_, ty_?TypeArrowQ] :=
	Module[ {vals, key},
		vals = self["monomorphic"]["lookup", name, Null];
		If[vals =!= Null,
			key = Map[#["unresolve"]&, ty["arguments"]];
			vals["lookup", key, Null]
			,
			Null]
	]

getMonomorphic[self_, name_, ty_] =
	Null

getMonomorphicList[self_, name_] :=
	Module[ {vals},
		vals = self["monomorphic"]["lookup", name, Null];
		If[vals === Null,
			{},
			vals["values"]
			,
			Null]
	]


isMonomorphic[ ty_?TypeArrowQ] :=
	Module[{free = ty["free"]},
		Length[free] === 0
	]

isMonomorphic[ ty_?TypeConstructorQ] :=
	True

isMonomorphic[ ty_?TypeApplicationQ] :=
	Module[{free = ty["free"]},
		Length[free] === 0
	]

isMonomorphic[ ty_] :=
	False

addType[ self_, name_, ty_] :=
	Module[ {vals},
		If[ isMonomorphic[ty],
			addMonomorphic[self, name, ty]
			,
			vals = self["polymorphic"]["lookup", name, Null];
			If[vals === Null,
				vals = CreateReference[{}];
				self["polymorphic"]["associateTo", name -> vals]];
			vals["appendTo", ty]];
	]

hasTypes[ self_, name_] :=
	self["polymorphic"]["keyExistsQ", name]

getPolymorphicList[ self_, name_] :=
	Module[ {vals},
		vals = self["polymorphic"]["lookup", name, Null];
		If[ vals === Null,
			{},
			vals["get"]]
	]

addExcludedType[ self_, name_, ty_] :=
	Module[ {vals},
		vals = self["excluded"]["lookup", name, Null];
		If[vals === Null,
			vals = CreateReference[{}];
			self["excluded"]["associateTo", name -> vals]];
		vals["appendTo", ty];
	]


dispose[self_] :=
	(
	self["setExcluded", Null];
	self["setPolymorphic", Null];
	self["setMonomorphic", Null];
	)

CreateFunctionTypeLookup[] :=
	CreateObject[FunctionTypeLookup, <|
			"excluded" -> CreateReference[<||>],
			"polymorphic" -> CreateReference[<||>],
			"monomorphic" -> CreateReference[<||>]
		|>]



(**************************************************)

icon := Graphics[Text[
	Style["FunTy",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[self_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"FunctionTypeLookup",
		self,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["polymorphicTypes: ", {90, Automatic}], Length[self["polymorphic"]["get"]]}],
			BoxForm`SummaryItem[{Pane["monomorphicTypes: ", {90, Automatic}], Length[self["monomorphic"]["get"]]}]
  		},
  		Join[
  			Map[
  		   	BoxForm`SummaryItem[{Pane[First[#], {90, Automatic}], Column[Last[#]["get"]]}]&,
  		   	Normal[self["polymorphic"]["get"]]	
  			], 
  			Map[
  		   	BoxForm`SummaryItem[{Pane[First[#], {90, Automatic}], Column[Normal[Last[#]["get"]]]}]&,
  		   	Normal[self["monomorphic"]["get"]]	
  			]
  		],
  		fmt
  	]


toString[env_] := "FunctionTypeLookup[<>]"


End[]

EndPackage[]


