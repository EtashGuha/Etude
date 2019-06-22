
BeginPackage["TypeFramework`TypeObjects`TypeRecurse`"]

TypeRecurseQ
CreateTypeRecurse
TypeRecurseObject


Begin["`Private`"]

Needs["TypeFramework`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`TypeObjects`TypeBase`"]



format[ self_, shortQ_:True] :=
	"TypeRecurse[" <> self["variable"]["format", shortQ] <> "," <> self["type"]["format", shortQ] <> "]"

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeRecurseClass = DeclareClass[
	TypeRecurseObject,
	<|
		"instantiate" -> Function[ {env}, instantiate[Self, env]],
		"unresolve" -> Function[ {}, unresolve[Self]],
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	{
		"id" -> -1,
		"symbol",
		"variables",
		"type",
		"recurse",
		"properties"
	},
	Predicate -> TypeRecurseQ,
	Extends -> {
		ClassPropertiesTrait
	}
]
]]

CreateTypeRecurse[sym_, vars_, type_, recurse_] :=
	CreateObject[TypeRecurseObject, <|
				"id" -> GetNextTypeId[],
				"symbol" -> sym,
				"variables" -> vars,
				"type" -> type,
				"recurse" -> recurse,
				"properties" -> CreateReference[<||>]
			|>]



instantiate[self_, env_] :=
	Module[ {vars, type, recurse},
		vars = <|Map[ (# -> CreateTypeVariable[#]) &, self["variables"]]|>;
		type = env["resolveWithVariables", self["type"], vars];
		recurse = env["resolveWithVariables", self["recurse"], vars];
		<|"type" -> type, "recurse" -> recurse, "monomorphic" -> Values[vars]|>
	]

unresolve[ self_] :=
	TypeRecurse[ self["symbol"], self["variables"], self["type"], self["recurse"]]


format[ self_, shortQ_:True] :=
	self["toString"]


    
(**************************************************)

icon := Graphics[Text[
	Style["TRec",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[typ_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"TypeRecurse",
		typ,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["symbol: ", {90, Automatic}], typ["symbol"]}],
			BoxForm`SummaryItem[{Pane["variables: ", {90, Automatic}], typ["variables"]}],
  		    BoxForm`SummaryItem[{Pane["type: ", {90, Automatic}], typ["type"]}],
  		    BoxForm`SummaryItem[{Pane["recurse: ", {90, Automatic}], typ["recurse"]}]
  		},
  		{
  		}, 
  		fmt
  	]


toString[typ_] := StringJoin["TypeRecurse[",
							Riffle[{
								ToString[typ["symbol"]],
								ToString[typ["variables"]],
								ToString[typ["type"]],
								ToString[typ["recurse"]]								
								}, ","],
							"]"]
					
End[]

EndPackage[]

