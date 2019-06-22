
BeginPackage["Compile`Core`IR`Lower`Utilities`Fresh`"]

CreateFreshId
CreateFreshVariable

Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]

nextVarId := Compile`Core`IR`Variable`Private`$NextVariableId

updateVarId[ value_] :=
	(
	Compile`Core`IR`Variable`Private`$NextVariableId = value +1;
	value
	)

CreateFreshId[state_] :=
	updateVarId[Max[state["nextVariable"]["increment"], nextVarId]]

CreateFreshVariable[state_] :=
	With[{id = CreateFreshId[state]},
	   	state["nextVariable"]["set", id];
		CreateVariable[id]
	]
	
CreateFreshVariable[state_, mexpr_] :=
	With[{id = CreateFreshId[state]},
	   	state["nextVariable"]["set", id];
		CreateVariable[id, mexpr]
	]

End[] 

EndPackage[]