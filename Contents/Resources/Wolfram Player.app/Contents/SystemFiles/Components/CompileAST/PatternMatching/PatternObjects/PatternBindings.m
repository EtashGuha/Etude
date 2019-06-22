BeginPackage["CompileAST`PatternMatching`PatternObjects`PatternBindings`"]

PatternBindings

Begin["`Private`"] 

Needs["CompileAST`Utilities`MExprVisitor`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`PatternMatching`Matcher`"]
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareCompileASTClass", Function[{st},
PatternBindingsClass = DeclareClass[
	PatternBindings,
	<|
		"search" -> Function[{index}, search[Self, index]],
		"add" -> Function[{index, mexpr}, add[Self, index, mexpr]],
		"getBindings" -> Function[{}, getBindings[Self]],
		"initialize" -> Function[{}, initialize[Self]],
		"getNameValueMap" -> Function[{}, getNameValueMap[Self]],
		"substitute" -> Function[ {mexpr}, substitute[Self, mexpr]],
		"reset" -> Function[{}, reset[Self]],
		"toString" -> Function[{},
			toString[Self]
		],
		"toBoxes" -> Function[{fmt},
			toBoxes[Self, fmt]
		]
	|>,
	{
	   "number",
	   "nameMap",
	   "bindingList"
	},
    Predicate -> PatternBindingsQ
]
]]

initialize[ self_] :=
	reset[self]

reset[self_] :=
	Module[ {bindings, list},
		list = Table[Null, {self["number"]["get"]}];
		bindings = CreateReference[list];
		self["setBindingList", bindings]
	]


search[ self_, index_] :=
	Module[ {},
		self["bindingList"]["getPart", index]
	]

add[ self_, index_, mexpr_] :=
	Module[ {},
		self["bindingList"]["setPart", index, mexpr]	
	]

getElem[ bindingList_, _[ name_, index_]] :=
	name -> bindingList[[index]]

getBindings[ self_] :=
	Module[ {nameMap, bindingList},
		nameMap = self["nameMap"]["get"];
		bindingList = self["bindingList"]["get"];
		AssociationMap[ getElem[bindingList, #]&, nameMap]
	]

getNameValueMap[ self_] :=
	Module[ {nameMap, bindingList},
		nameMap = self["nameMap"]["get"];
		bindingList = self["bindingList"]["get"];
		AssociationMap[ getElem[bindingList, #]&, nameMap]
	]



(*
 Substitution
*)

RegisterCallback["DeclareCompileASTClass", Function[{st},
DeclareClass[
	MExprSubstituteBindingsVisitor,
	<|
		"visitSymbol" -> Function[{mexpr}, visitSymbol[Self, mexpr]],
		"visitNormal" -> Function[{mexpr}, visitNormal[Self, mexpr]],
		"initialize" -> Function[ {}, initializeVisitor[Self]]
	|>,
	{
  		"nameValueMap",
  		"bindings"
  	},
  	Extends -> {MExprMapVisitorClass}
 ]
]]


initializeVisitor[ self_] :=
	Module[ {nameValueMap},
		self["setResult", Null];
		nameValueMap = self["bindings"]["getNameValueMap"];
		self["setNameValueMap", nameValueMap];
	]


visitNormal[ self_, mexpr_] :=
	Module[ { res, head, args, newArgs, ef},
		self["processNormal", mexpr];
		(*
		  Now look for any sequences and flatten if found.
		*)
		res = self["result"];
		args = res["arguments"];
		head = res["head"];
		newArgs =
			Map[
				(
				If[ #["head"]["sameQ", MatcherSequence],
					#["arguments"]
					,
					{#}]
				)&, args];
		newArgs = Flatten[ newArgs];
		ef = CreateMExprNormal[ head, newArgs];		
		self["setResult", ef];
		False
	]

(*
  TODO add an assert that mexpr is a symbol
*)
visitSymbol[ self_, mexpr_] :=
	With[{
		fullName = mexpr["fullName"]
	},
	With[{
		ef = Lookup[self["nameValueMap"], fullName, mexpr]
	},
		self["setResult", ef]
	]];
	
substitute[self_, mexpr_] :=
	With[{
		vst = CreateObject[
			MExprSubstituteBindingsVisitor,
			<|
				"bindings" -> self
			|>
		]
	},
		mexpr["accept", vst];
		vst["result"]
	]

(*
Formatting 
*)

toString[self_] := 
	Module[ {bindings},
		bindings = getBindings[self];
		bindings = AssociationMap[ (#[[1]] -> #[[2]]["toString"])&, bindings];
		StringJoin[ "MExprPatternBindings[", ToString[bindings], "]"]
	]
		


icon := icon = Graphics[
	Text[
		Style["Bind", GrayLevel[0.7], Bold, 0.9*CurrentValue["FontCapHeight"] / AbsoluteCurrentValue[Magnification]]
	],
	$FormatingGraphicsOptions
];   


toBoxes[self_, fmt_] :=
	Module[ {bindings, len, visible, sometimes},
		bindings = getBindings[self];
		bindings = AssociationMap[ (#[[1]] -> #[[2]]["toString"])&, bindings];
		len = Length[bindings];
		visible = {BoxForm`SummaryItem[{"Length: ", len}]};
		sometimes = Map[ BoxForm`SummaryItem[{"", #}]&, Normal[bindings]];
		BoxForm`ArrangeSummaryBox[
			"MExprPatternBindings",
			self,
  			icon,
			visible
	  		,
			sometimes
			, 
  			fmt
  		]
	]
	







End[]

EndPackage[]
