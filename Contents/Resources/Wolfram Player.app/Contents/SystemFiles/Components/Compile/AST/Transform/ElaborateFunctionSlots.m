BeginPackage["Compile`AST`Transform`ElaborateFunctionSlots`"]


ElaborateFunctionSlots
ElaborateFunctionSlotsPass

Begin["`Private`"] 


Needs["CompileAST`Utilities`MExprVisitor`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileAST`Create`Construct`"]
Needs["Compile`Core`PassManager`MExprPass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["CompileAST`Class`Literal`"]
Needs["CompileAST`Class`Symbol`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]

(*
 Scoping rules.
 
 Each nested slot function starts a new scoping environment.
 The inner function #1/#2 are not bound to the args.

In[15]:= Function[#1 + Function[#1 + #2]][1, 2]

Out[15]= 1 + (#1 + #2 &)


Notice that the lack of #2 is not a problem.

In[16]:= Function[#1][1, 2]

Out[16]= 1

But here the prescence of #3 is a problem since it can't be filled.

In[17]:= Function[#1 + #3][1, 2]

During evaluation of In[17]:= Function::slotn: Slot number 3 in #1+#3& cannot be filled from (#1+#3&)[1,2].

Out[17]= 1 + #3

There are argument sequences:

In[18]:= Function[{#1, f[##]}][1, 2]

Out[18]= {1, f[1, 2]}

and a binding for the function itself

In[19]:= Function[f[#0]][1, 2]

Out[19]= f[f[#0] &]

and bindings via Association

In[23]:= {#foo, #1} &[<|"foo" -> 1|>]

Out[23]= {1, <|"foo" -> 1|>}

These latter three features might not supported just yet.

*)

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"ElaborateFunctionSlots",
	"Transforms all functions with nameless slots into names argument "<>
	"paramaters to simplify later transformations"
];

ElaborateFunctionSlotsPass = CreateMExprPass[<|
	"information" -> info,
	"runPass" -> ElaborateFunctionSlots
|>];

RegisterPass[ElaborateFunctionSlotsPass]
]]


ElaborateFunctionSlots[mexpr_, opts:OptionsPattern[]] :=
	ElaborateFunctionSlots[mexpr, <| opts |>]
ElaborateFunctionSlots[mexpr_, opts_?AssociationQ] :=
	Module[ {visitor},
		visitor = CreateObject[ SlotFunctionReplaceVisitor, <||>];
		mexpr[ "accept", visitor];
		visitor["result"]
	]
	
	
slot[visitor_, mexpr_] :=
	Module[ {slot, val, sym},
		If[ mexpr["length"] =!= 1,
			Return[ visitor["processNormal", mexpr]]
		];
		slot = mexpr["part", 1];
		If[ !(MExprLiteralQ[slot] && slot["hasHead", Integer]),
			Return[ visitor["processNormal", mexpr]]];
		val = slot["data"];
		If[val < 0,
			Return[ visitor["processNormal", mexpr]]];
		sym = visitor["slotData"]["lookup", val, Null];
		If[ sym === Null,
			sym = CreateMExprSymbol[];
			visitor["slotData"]["associateTo", val -> sym]];
		visitor["setResult", sym];
	]
	
function[visitor_, mexpr_] :=
	Module[ {args, body, max, val, data, keys},
		If[ mexpr["length"] =!= 1,
			Return[visitor["processNormal", mexpr]]
		];
		data = visitor["slotData"];
		visitor["setSlotData", CreateReference[ <||>]];
		mexpr["part", 1]["accept", visitor];
		body = visitor["result"];
		keys = Sort[visitor["slotData"]["keys"]];
		max = If[ Length[keys] === 0, 0, Max[ keys]];
		args = Table[
			      val = visitor["slotData"]["lookup", i, Null];
			      If[ val === Null,
			      	     CreateMExprSymbol[],
			      	     val]
			      ,
			      {i,max}];
		args = CreateMExprNormal[ CreateMExprSymbol[List], args];
		mexpr["setArguments", {args, body}];
		mexpr["setProperty", "SlotFunction" -> True];
		visitor["setResult", mexpr];
		visitor["setSlotData", data]
	]

$dispatchNormal = <|
	"System`Function" -> function,
	"System`Slot" -> slot
|>

normal[visitor_, mexpr_] :=
	visitor["processNormal", mexpr]

visitNormal[ visitor_, mexpr_] :=
	With[{
	   h = mexpr["head"]
	},
	With[{
	   dispatch = If[MExprSymbolQ[h],
			Lookup[
				$dispatchNormal,
				h["fullName"],
				normal
			],
			normal
		]
	},
		dispatch[visitor, mexpr];
		False
	]]


initializeVisitor[ self_] :=
	Module[ {},
		self["setSlotData", Null];
	]

RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[
	SlotFunctionReplaceVisitor,
	<|
		"visitNormal" -> Function[{mexpr}, visitNormal[Self, mexpr]],
		"initialize" -> Function[ {}, initializeVisitor[Self]]
	|>,
	{
  		"slotData"
  	},
  	Extends -> {MExprMapVisitorClass}
 ]
]]


End[]

EndPackage[]
