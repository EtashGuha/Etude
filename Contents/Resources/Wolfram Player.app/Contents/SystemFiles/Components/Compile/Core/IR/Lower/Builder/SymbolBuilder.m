BeginPackage["Compile`Core`IR`Lower`Builder`SymbolBuilder`"]

(*


*)


SymbolBuilder;
SymbolBuilderQ;
SymbolBuilderClass;
CreateSymbolBuilder;

Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileAST`Class`Symbol`"]; (* For MExprSymbolQ *)
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]


(*
	The SymbolBuilderClass holds global properties about scoped symbols, such 
	as whether they are writable, and the current variable.
	
	It is a global property so held by the ProgramModuleBuilder
*)

RegisterCallback["DeclareCompileClass", Function[{st},
SymbolBuilderClass = DeclareClass[
	SymbolBuilder,
	<|
		"initialize" -> Function[{},
			Self["setAssignAliases", CreateReference[<||>]];
			Self["setTable", CreateReference[<||>]];
		],
		"add" -> (add[Self, ##]&),
		"getVariable" -> (getVariable[Self, ##]&),
		"readVariable" -> (readVariable[Self, ##]&),
		"updateVariable" -> (updateVariable[Self, ##]&),
		"exists" -> (exists[Self, ##]&),
		"getCurrentFunctionBuilder" -> Function[{}, getCurrentFunctionBuilder[Self]],
		"isWritable" -> (isWritable[Self, ##]&),
		"pushAssignAlias" -> (pushAssignAlias[Self, ##]&),
		"popAssignAlias" -> (popAssignAlias[Self, ##]&),
		"peekAssignAlias" -> (peekAssignAlias[Self, ##]&),
		"fullform" -> Function[{}, fullform[Self]],
		"dispose" -> Function[{}, dispose[Self]],
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt},
			toBoxes[Self, fmt]
		]

	|>,
	{
		"assignAliases",
		"table",
		"programBuilder"
	},
	Predicate -> SymbolBuilderQ
]
]]


(*

*)
ClearAll[add]
instOrNothing[x_] := MatchQ[x, Nothing | _?InstructionQ] 
add[self_, name_String, var_?VariableQ] := add[self, name, var, Nothing, <||>]
add[self_, name_String, var_?VariableQ, props_?AssociationQ] := add[self, name, var, Nothing, props]
add[self_, name_String, var_?VariableQ, inst_?instOrNothing, props_:<||>] :=
	Module[ {table = self["table"], writable},
		var["setProperty", "variableValue" -> name];
		If[inst =!= Nothing,
			self["getCurrentFunctionBuilder"]["writeVariable", var, inst];
		];
		writable = Lookup[ props, "Writable", True];
		(*
		self["getCurrentFunctionBuilder"]["writeVariable", var, inst];
		*)
		table["associateTo", name -> <|
			"var" -> var,
			"writable" -> writable
		|>]
	]

add[ ___] :=
	ThrowException[{"Error in calling SymbolBuilder add."}]

getVariable[self_, name_String] :=
	Module[ {data},
		data = getData[self, name];
		If[ MissingQ[data],
			data,
			Lookup[data, "var"]
		]
	]
	
readVariable[self_, name_String, mexpr_?MExprSymbolQ] :=
	Module[ {data},
		data = getData[self, name];
		If[ MissingQ[data],
			data,
			self["getCurrentFunctionBuilder"]["readVariable", name, mexpr]
		]
	]
readVariable[args___] :=
        Throw["SymbolBuilder/readVariable called with bad args: ", {args}];
	
updateVariable[self_, name_String, inst_?InstructionQ] :=
	Module[ {data},
		data = getData[self, name];
		self["getCurrentFunctionBuilder"]["writeVariable", name, inst];
		If[ MissingQ[data],
			ThrowException[{"variable cannot be found", name}]
		];
		If[inst["definedVariable"]["type"] === Undefined,
			inst["definedVariable"]["setType", data["var"]["type"]]
		];
		data["var"] = inst["definedVariable"];
		setData[self, name, data]
	]

updateVariable[self_, args___] :=
	ThrowException[{"Bad arguments to SymbolBuilder/updateVariable: ", {args}}]

exists[ self_, name_] :=
	!MissingQ[getData[self, name]]
	
isWritable[ self_, name_] :=
	Module[ {data},
		data = getData[self, name];
		If[ MissingQ[data],
			ThrowException[{"variable cannot be found", name}]
		];
		Lookup[data, "writable", True]		
	]


pushAssignAlias[ self_, sym_, var_] :=
	Module[ {val = self["assignAliases"]["lookup", sym, {}]},
		self["assignAliases"]["associateTo", sym -> {var, val}]
	]
	
popAssignAlias[ self_, sym_] :=
	Module[{val = self["assignAliases"]["lookup", sym, {}]},
		If[ !MatchQ[val, {_,_List}],
			ThrowException[{"malformed assignAlias"}]];
		val = Last[val];
		If[ val === {},
			self["assignAliases"]["keyDropFrom", sym],
			self["assignAliases"]["associateTo", sym -> val]]
	]

peekAssignAlias[ self_, sym_] :=
	Module[{val = self["assignAliases"]["lookup", sym, Null]},
		Which[
			val === Null,
				Null
			,
			!MatchQ[val, {_,_List}],
				ThrowException[{"malformed assignAlias"}]
			,
			True,
				First[ val]]
	]



(*
  Get the data for name,  can return Missing
*)
getData[ self_, name_] :=
	Module[ {table = self["table"]},
		table["lookup", name]
	]

setData[ self_, name_, data_] :=
	Module[ {table = self["table"]},
		table["associateTo", name -> data]
	]


getCurrentFunctionBuilder[self_] :=
	self["programBuilder"]["currentFunctionModuleBuilder"]

CreateSymbolBuilder[pmb_] :=
	CreateObject[
		SymbolBuilder,
		<|
		"programBuilder" -> pmb
		|>
	]

dispose[self_] :=
	Module[{},
		self["setProgramBuilder", Null];
		self["setAssignAliases", Null];
		self["setTable", Null];
	]
	
(*********************************************************************)
(*********************************************************************)
	
(**
  * # Formating code
  *)
icon := Graphics[Text[
  Style["SYM\nBLD", GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];   
     

toBoxes[builder_, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"SymbolBuilder",
		builder,
  		icon,
  		{
  			BoxForm`SummaryItem[{"table: ", builder["table"]}]
  		},
		{}, 
  		fmt,
		"Interpretable" -> False
  	]

toString[self_] := (
	StringJoin[
		"SymbolBuilder[",
		"\n",
		self["table"]["toString"],
		"\n]"
	]
)
	
fullform[self_] := 
		SymbolBuilder[ self["table"]]
	
	
End[]
EndPackage[]
