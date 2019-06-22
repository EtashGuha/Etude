
BeginPackage["Compile`Core`IR`Lower`Utilities`TypeEnvironment`"]

(*
LoweringEnvironment
LoweringEnvironmentQ
LoweringEnvironmentClass
*)
CreateLoweringTypeEnvironment

Begin["`Private`"] 

Needs["TypeFramework`"]
Needs["Compile`"]
Needs["CompileUtilities`Error`Exceptions`"]

Options[CreateLoweringTypeEnvironment] = {
	"OptimizationLevel" -> Automatic,
	"EnvironmentOptions" -> <||>
}

CreateLoweringTypeEnvironment[opts:OptionsPattern[]] :=
	CreateLoweringTypeEnvironment[<| opts |>]
	
CreateLoweringTypeEnvironment[opts_?AssociationQ] :=
	Module[ {envOpt, env},
		envOpt = Lookup[opts, "EnvironmentOptions", <||>];
		env = Lookup[envOpt, "TypeEnvironment", Null];
		If[TypeEnvironmentQ[env],
			env,
			$DefaultTypeEnvironment
		]
	]

CreateLoweringTypeEnvironment[args___] :=
	ThrowException[{"Unrecognized call to CreateLoweringTypeEnvironment", {args}}]

End[] 

EndPackage[]
