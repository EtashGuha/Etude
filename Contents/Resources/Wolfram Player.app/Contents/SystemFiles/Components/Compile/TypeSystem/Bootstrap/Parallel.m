
BeginPackage["Compile`TypeSystem`Bootstrap`Parallel`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]



"StaticAnalysisIgnore"[

setupTypes[st_] :=
	Module[ {env = st["typeEnvironment"]},
	
		env["declareFunction", Native`PrimitiveFunction["NonParallelDo_Closure"], 
			MetaData[<|"Linkage" -> "Runtime"|>
				]@TypeSpecifier[{"VoidHandle", "VoidHandle","MachineInteger", "MachineInteger"} -> "UnsignedInteger32"] 
						];
	
		env["declareFunction", Native`PrimitiveFunction["NonParallelDo"], 
			MetaData[<|"Linkage" -> "Runtime", "ClosureForward" -> Native`PrimitiveFunction["NonParallelDo_Closure"]|>
				]@TypeSpecifier[{{"MachineInteger"} -> "UnsignedInteger32","MachineInteger", "MachineInteger"} -> "UnsignedInteger32"] 
						];

	]
	
] (* StaticAnalysisIgnore *)


RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]
