
BeginPackage["Compile`Core`IR`ExternalDeclaration`"]

(**
a forward declaration
*)
ExternalDeclaration
CreateExternalDeclaration
ExternalDeclarationQ
DeserializeExternalDeclaration

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareCompileClass", Function[{st},
ExternalDeclarationClass = DeclareClass[
	ExternalDeclaration,
	<|
		"scan" -> (scan[Self, #]&),
		"addFunction" -> (addFunction[Self, ##]&),
		"addRawData" -> (addRawData[Self, ##]&),
		"serialize" -> (serialize[Self, ##]&),
		"lookupUpdateFunction" -> Function[ {env, name}, lookupUpdateFunction[Self, env, name]],
		"lookupFunction" -> Function[ {name}, lookupFunction[Self, name]],
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"declarations",
		"rawData"
	},
	Predicate -> ExternalDeclarationQ
]
]]


CreateExternalDeclaration[decls_?ExternalDeclarationQ] :=
	decls


CreateExternalDeclaration[decls_?ReferenceQ] :=
	Module[ {lookup},
		lookup = CreateObject[ExternalDeclaration, <|
			"declarations" -> decls["clone"],
			"rawData" -> CreateReference[{}]
			|>];
		lookup
	]
	
CreateExternalDeclaration[decls_?AssociationQ] :=
	Module[ {lookup},
		lookup = CreateObject[ExternalDeclaration, <|
			"declarations" -> CreateReference[decls],
			"rawData" -> CreateReference[{}]
			|>];
		lookup
	]
	
CreateExternalDeclaration[args___] :=
	ThrowException[{"Unknown call to CreateExternalDeclaration", {args}}]


scan[self_, fun_] :=
	Scan[ fun, Normal[ self["declarations"]["get"]]]


addFunction[ self_, funName_, data_] :=
	Module[ {},
		self["declarations"]["associateTo", funName -> Prepend[data, "class" -> "Function"]]
	]


lookupFunction[self_, funName_] :=
	Module[ {data},
		data = self["declarations"]["lookup", funName, <||>];
		If[Lookup[data, "class", Null] === "Function",
			data,
			Missing["unable to find function in external declarations", funName]
		]
	]


(*
  Looks up a function definition,  if it can't be found look in the environment.
*)
lookupUpdateFunction[self_, env_, funName_] :=
	Module[ {data, funType, funDef, defs, def},
		data = lookupFunction[self, funName];
		If[!MissingQ[data],
			Return[data]];
		(*
		   Now look in cache
		*)
		funDef = env["functionDefinitionLookup"];
		data = funDef["cache"]["lookup", funName, Null];
		If[data =!= Null,
			addFunction[self, funName, data];
			data = lookupFunction[self, funName];
			Return[data]];
		(*
		    Now look in type lookup,  we look in the PolymorphicList, maybe 
		    we also could in the MonomorphicList.  We only accept one definition
		    if it is Linked.
		*)
		funType = env["functionTypeLookup"];
		defs = funType["getPolymorphicList", funName];
		data = searchAddDefs[self, funName, defs];
		If[data =!= Null,
			Return[data]];
		defs = funType["getMonomorphicList", funName];
		data = searchAddDefs[self, funName, defs];
		data
	]

searchAddDefs[ self_, funName_, defs_] :=
	Module[ {def, data = Null},
		If[Length[defs] =!= 1,
			Return[Null]];
		def = First[defs]["getProperty", "definition"];
		If[AssociationQ[def] && Lookup[def, "Class"] === "Linked",
			addFunction[self, funName, def];
			data = lookupFunction[self, funName]];
		data
	]

addRawData[ self_, data_] :=
	Module[ {},
		self["rawData"]["appendTo", data]
	]



serialize[self_, env_] :=
	"ExternalDeclarations"[ Map[ serializeEntry[self, env, #]&, self["declarations"]["get"]]]

serializeEntry[self_, env_, elem_] :=
	If[ KeyExistsQ[ elem, "type"],
		MapAt[env["serializeType", #]&, elem, "type"],
		elem]
	

DeserializeExternalDeclaration[env_, "ExternalDeclarations"[data_]] :=
	CreateReference[Map[ deserializeElement[env, #]&, data]]

deserializeElement[env_, assoc_] :=
	If[ KeyExistsQ[assoc, "type"],
		MapAt[ env["getType", #]&, assoc, "type"],
		assoc
	]


(**************************************************)






icon := Graphics[Text[
	Style["ED",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[env_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"ExternalDeclaration",
		env,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["declarations: ", {90, Automatic}], env["declarations"]}]
  		},
  		{
			BoxForm`SummaryItem[{Pane["rawData: ", {90, Automatic}], env["rawData"]}]

  		}, 
  		fmt,
		"Interpretable" -> False
  	]


toString[env_] := "ExternalDeclaration[<>]"






End[]
EndPackage[]
