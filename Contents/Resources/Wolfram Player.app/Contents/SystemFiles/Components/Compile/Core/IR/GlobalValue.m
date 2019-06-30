
BeginPackage["Compile`Core`IR`GlobalValue`"]

(**
a set of arguments and basic blocks
*)
GlobalValue;
CreateGlobalValue;
GlobalValueQ;

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]


	
CreateGlobalValue[args___] :=
	GlobalValue[CreateReference[createGlobalValue[args]]]
	
createGlobalValue[] :=
	<|
		"id" -> 0,
		"variable" -> CreateReference[],
		"value" -> CreateReference[],
		"mexpr" -> None,
		"mexprId" -> -1,
		"programModule" -> CreateReference[],
		"properties" -> CreateReference[<||>]
	|>


End[]
EndPackage[]