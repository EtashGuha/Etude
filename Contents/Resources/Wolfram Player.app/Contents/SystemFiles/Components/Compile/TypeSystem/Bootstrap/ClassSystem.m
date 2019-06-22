
BeginPackage["Compile`TypeSystem`Bootstrap`ClassSystem`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]


"StaticAnalysisIgnore"[

setupTypes[st_] :=
	Module[{env = st["typeEnvironment"],
			inline = MetaData[<|"Inline" -> "Hint"|>]},


		env["declareType", 
			MetaData[<|"Fields" -> <|"f1" -> 1, "f2" -> 2|>|>
      			]@TypeConstructor["MClassStruct", {"*", "*", "*", "*", "*", "*"} -> "*"]];

		env["declareType", TypeAlias["MClassData", 
								       "Handle"["MClassStruct"["CString", "VoidHandle", "VoidHandle", "VoidHandle", "VoidHandle", "VoidHandle"]]]];

		env["declareType", 
			MetaData[<|"Fields" -> <|"f1" -> 1, "f2" -> 2|>|>
      			]@TypeConstructor["MObjectStruct", {"*", "*"} -> "*"]];

		env["declareType", TypeAlias["MObjectData", 
								       "Handle"["MObjectStruct"["MClassData", "VoidHandle"]]]];


		env["declareFunction", CompileClassSystem`ObjectInstanceQ, 
				Typed[{"Expression"} -> "Boolean"
				]@Function[{e},
					Native`PrimitiveFunction["Expr`Type"][e] === Typed[9, "Integer16"] &&
					   Native`PrimitiveFunction["Expr`RawType"][e] === Typed[26, "Integer16"]
				]];

		env["declareFunction", CompileClassSystem`MObject, 
				inline@Typed[{"Expression"} -> "MObjectData"
				]@Function[{e},
					Module[{data},
						data = Native`PrimitiveFunction["Expr`RawContents"][e];
						Native`BitCast[data, "MObjectData"]
					]
				]];
		
		env["declareFunction", CompileClassSystem`MClass, 
				inline@Typed[{"MObjectData"} -> "MClassData"
				]@Function[{e},
					e[[Native`Field[0]]]
				]];

		env["declareFunction", CompileClassSystem`MClass, 
				inline@Typed[{"Expression"} -> "MClassData"
				]@Function[{e},
					Module[{obj = CompileClassSystem`MObject[e]},
						CompileClassSystem`MClass[obj]
					]
				]];

		env["declareFunction", CompileClassSystem`MClassName, 
				inline@Typed[{"MClassData"} -> "CString"
				]@Function[{e},
					e[[Native`Field[0]]]
				]];

		env["declareFunction", CompileClassSystem`MClassName, 
				inline@Typed[{"Expression"} -> "CString"
				]@Function[{e},
					Module[{classData = CompileClassSystem`MClass[e]},
						CompileClassSystem`MClassName[classData]
					]
				]];


		

	]

] (* StaticAnalysisIgnore *)

RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]
