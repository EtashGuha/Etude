BeginPackage["Compile`TypeSystem`Bootstrap`Debugging`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]


"StaticAnalysisIgnore"[

setup[st_] :=
	Module[ {env = st["typeEnvironment"]},

		(*env["declareFunction", Native`PrimitiveFunction["DebugDeclare"],*)
						(*MetaData[*)
							(*<|"Linkage" -> "LLVMDebug"|>*)
							(*]@TypeSpecifier[TypeForAll[ {"a"},*)
								(*{"a"} -> "Void"]]];*)

		env["declareFunction", Native`PrimitiveFunction["DebugDeclare"],
			MetaData[
				<|"Linkage" -> "LLVMDebug"|>
			]@TypeSpecifier[TypeSpecifier[{} -> "Void"]]];

		env["declareFunction", Native`PrimitiveFunction["SetBreakpoint"],
						MetaData[
							<|"Linkage" -> "LLVMCompileTools"|>
							]@TypeSpecifier[{} -> "Void"]];

	]

] (* StaticAnalysisIgnore *)


RegisterCallback["SetupTypeSystem", setup]


End[]

EndPackage[]
