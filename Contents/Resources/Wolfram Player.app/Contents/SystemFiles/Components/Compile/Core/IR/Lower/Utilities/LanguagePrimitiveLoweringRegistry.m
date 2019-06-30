BeginPackage["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]

$LanguagePrimitiveLoweringRegistry;

RegisterLanguagePrimitiveLowering;

Begin["`Private`"] 

Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]


If[!AssociationQ[$LanguagePrimitiveLoweringRegistry],
	$LanguagePrimitiveLoweringRegistry = <||>
]


RegisterLanguagePrimitiveLowering[info_?LanguagePrimitiveQ, lower_] := 
	With[{fullName = info["fullName"]},
		Assert[KeyExistsQ[$LanguagePrimitiveLoweringRegistry, fullName] === False];
		$LanguagePrimitiveLoweringRegistry[fullName] = <|
			"info" -> info,
			"lower" -> lower
		|>;
	]
	
RegisterLanguagePrimitiveLowering[args___] :=
	ThrowException[{"Invalid primitive registration ", args}]


End[] 

EndPackage[]
