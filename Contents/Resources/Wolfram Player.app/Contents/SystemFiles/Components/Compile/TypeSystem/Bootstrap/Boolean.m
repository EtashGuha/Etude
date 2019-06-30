
BeginPackage["Compile`TypeSystem`Bootstrap`Boolean`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]


"StaticAnalysisIgnore"[

setupTypes[st_] :=
	Module[{env = st["typeEnvironment"],
			inline = MetaData[<|"Inline" -> "Hint"|>],
			llvmLinkage = MetaData[<|"Linkage" -> "LLVMCompileTools"|>]},

		env["declareFunction", Native`PrimitiveFunction["BitAnd"], 
			llvmLinkage@TypeSpecifier[{"Boolean", "Boolean"} -> "Boolean"]];

		env["declareFunction", Native`PrimitiveFunction["BitOr"], 
			llvmLinkage@TypeSpecifier[{"Boolean", "Boolean"} -> "Boolean"]];


		env["declareFunction", Compile`EagerAnd, 
			inline@Typed[
				 {"Boolean", "Boolean"} -> "Boolean"
			]@Function[{x, y},
 				Native`PrimitiveFunction["BitAnd"][x,y]
				]];

		env["declareFunction", Compile`EagerOr, 
			inline@Typed[
				 {"Boolean", "Boolean"} -> "Boolean"
			]@Function[{x, y},
 				Native`PrimitiveFunction["BitOr"][x,y]
				]];


	]

] (* StaticAnalysisIgnore *)

RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]
