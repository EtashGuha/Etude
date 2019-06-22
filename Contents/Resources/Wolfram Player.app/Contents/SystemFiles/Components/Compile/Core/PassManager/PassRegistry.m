BeginPackage["Compile`Core`PassManager`PassRegistry`"]

$Passes;
$DebugPasses;
RegisterPass;

Begin["`Private`"] 

Needs["Compile`Core`PassManager`Pass`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Asserter`Expect`"]


If[!AssociationQ[$Passes],
	$Passes = <||>
]


RegisterPass[pass_?PassQ] := (
	If[!TrueQ[$DebugPasses],
		ExpectThat["The pass has been registered before.",
			pass["name"]]["named", pass["name"]]["isNotAMemberOf", $Passes]
	];
	$Passes[pass["name"]] = pass;
	pass
)
RegisterPass[args___] :=
	ThrowException[{"Invalid pass registration ", args}]


End[] 

EndPackage[]
