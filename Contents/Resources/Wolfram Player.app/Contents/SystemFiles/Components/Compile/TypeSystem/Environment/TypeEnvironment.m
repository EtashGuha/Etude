
BeginPackage["Compile`TypeSystem`Environment`TypeEnvironment`"]

CreateCompileTypeEnvironment
GetTypeEnvironmentCache
SetTypeEnvironmentCache
InstantiateTypeEnvironment

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`TypeSystem`Environment`FunctionDefinitionLookup`"]
Needs["TypeFramework`"]
Needs["TypeFramework`Environments`TypeEnvironment`"]
Needs["Compile`"]
Needs["CompileUtilities`Error`Exceptions`"]


CreateCompileTypeEnvironment[] :=
	Module[ {tyEnv, defLookup},
		tyEnv = CreateTypeEnvironment[];
		defLookup = CreateCompileFunctionDefinitionLookup[];
		tyEnv["setFunctionDefinitionLookup", defLookup];
		tyEnv["setLiteralProcessor", literalProcessor];
		tyEnv
	]

(*
  So Type[1] will resolve to TypeLiteral[ 1, "MachineInteger"]
*)

literalProcessor[ tyEnv_, arg_] :=
	Which[
		IntegerQ[arg], "MachineInteger",
		arg === True, "Boolean",
		arg === False, "Boolean",
		True, Null
	]


finalize[st_] :=
	(
	st["typeEnvironment"]["finalize"]
	)

RegisterCallback["FinalizeTypeSystem", finalize]

GetTypeEnvironmentCache[ env_?TypeEnvironmentQ] :=
	Module[ {cacheData},
		cacheData = env["functionDefinitionLookup"]["cache"]["get"];
		cacheData
	]

GetTypeEnvironmentCache[args___] :=
	ThrowException[{"Unrecognized call to GetTypeEnvironmentCache", {args}}]

SetTypeEnvironmentCache[ env_?TypeEnvironmentQ, cacheData_?AssociationQ] :=
	Module[ {},
		env["functionDefinitionLookup"]["updateCache", cacheData];
		cacheData
	]

SetTypeEnvironmentCache[args___] :=
	ThrowException[{"Unrecognized call to SetTypeEnvironmentCache", {args}}]

(*
  Carry out many compilations in order to populate the type environment
  with a full cache so that it can be saved.  Do this by running the 
  "InstantiateFunctions" callback.  These are spread throughout the 
  Bootstrip dirs.
*)


InstantiateTypeEnvironment[env_] :=
	Module[ {state},
		state = <|"create" -> (CompileExpr[ #, "TypeEnvironment" -> env]&)|>;
		RunCallback[ "InstantiateFunctions", state];
	]

End[]

EndPackage[]
