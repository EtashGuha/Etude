
BeginPackage["Compile`TypeSystem`Bootstrap`MObject`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]


(*
   Base String implementation.

*)

"StaticAnalysisIgnore"[

setupTypes[st_] :=
	Module[{env = st["typeEnvironment"]},

		env["declareType", TypeConstructor["MObject"]];
		env["declareType", AbstractType["MObjectManaged", {}]];
		env["declareFunction", SameQ,
			Typed[
				TypeForAll[ {"a"}, 
					{Element["a", "MObjectManaged"], Element["a", "Equal"]}, 
						{"a", "a"} -> "Boolean"]
			]@Function[{arg1, arg2},
				Module[ {obj1, obj2},
					obj1 = Native`BitCast[ arg1, TypeSpecifier["MObject"]];
					obj2 = Native`BitCast[ arg2, TypeSpecifier["MObject"]];
					Native`PrimitiveFunction["SameQ_MObject_MObject_B"][obj1,obj2]
				]
			]];
		env["declareFunction", Native`PrimitiveFunction["SameQ_MObject_MObject_B"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"MObject", "MObject"} -> "Boolean"]];

		env["declareFunction", Native`PrimitiveFunction["Runtime_PushStackFrame"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"CArray"["MObject"], "MachineInteger"} -> "Void"]];
				
		env["declareFunction", Native`PrimitiveFunction["Runtime_PopStackFrame"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{} -> "Void"]];

		env["declareFunction", Native`PrimitiveFunction["Runtime_CheckGarbageCollect"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{} -> "Void"]];

		env["declareFunction", Native`PrimitiveFunction["Runtime_RunGarbageCollect"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{} -> "Void"]];

		]

] (* StaticAnalysisIgnore *)


RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]
