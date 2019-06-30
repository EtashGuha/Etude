
BeginPackage["Compile`TypeSystem`Bootstrap`String`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileAST`Create`Construct`"]

(*
   Base String implementation.

*)

"StaticAnalysisIgnore"[

setupTypes[st_] :=
	With[{env = st["typeEnvironment"],
	        inline = MetaData[<|"Inline" -> "Hint"|>]},

		env["declareType", AbstractType["StringSerializable", {}]];
		env["declareType", TypeConstructor["String", "Implements" -> {"StringSerializable", "MObjectManaged", "Equal"}]];



		env["declareFunction", StringJoin, 
						Typed[
							{"String", "String"} -> "String"
							]@Function[{s1, s2}, 
								Native`PrimitiveFunction["StringJoin_MString_MString_MString"][s1, s2]
							]];

		env["declareFunction", StringLength, 
						Typed[
							{"String"} -> "MachineInteger"
							]@Function[{s1}, 
								Native`PrimitiveFunction["StringLength_MString_I"][s1]
							]];

		env["declareFunction", Native`SubString, 
						Typed[
							{"String", "MachineInteger", "MachineInteger"} -> "String"
							]@Function[{s1, i1, i2}, 
								Module[{len},
									If[ i1 > i2,
										len = StringLength[s1];
										If[i1 === i2+1 && i2 <= len,
											Return[""]
											,
											Native`ThrowWolframException[Typed[Native`ErrorCode["StringExtract"], "Integer32"]]]];
									Native`PrimitiveFunction["StringTake_MString_I_I_MString"][s1, i1, i2]
									]
							]];

		env["declareFunction", ToCharacterCode, 
						Typed[
							{"String"} -> "PackedArray"["MachineInteger", 1]
							]@Function[{s1}, 
								Native`PrimitiveFunction["ToCharacterCode_MString_VectorI"][s1]
							]];

		env["declareFunction", ToString,
						inline@Typed[
							{"String"} -> "String"
							]@Function[{s1}, 
								s1
							]];

        env["declareFunction", Print,
            inline@Typed[{"Boolean"} -> "Void"]@
            Function[{a}, 
                If[a === True,
                	Print["True"],
                	Print["False"]
                ]
            ]
        ];
                            
		env["declareFunction", Print,
						inline@Typed[
							TypeForAll[ {"a"}, {Element["a", "StringSerializable"]}, {"a"} -> "Void"]
							]@Function[{s1}, 
								Native`PrimitiveFunction["Print_MString_Void"][ToString[s1]]
							]];

		env["declareFunction", Native`PrimitiveFunction["StringTake_MString_I_I_MString"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"String", "MachineInteger", "MachineInteger"} -> "String"]];

		env["declareFunction", Native`PrimitiveFunction["StringLength_MString_I"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"String"} -> "MachineInteger"]];

		env["declareFunction", Native`PrimitiveFunction["ToCharacterCode_MString_VectorI"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"String"} -> "PackedArray"["MachineInteger", 1]]];


		env["declareFunction", Native`PrimitiveFunction["StringJoin_MString_MString_MString"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"String", "String"} -> "String"]];


		env["declareFunction", Native`PrimitiveFunction["NewMString_UI8_MString"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"CArray"["UnsignedInteger8"]} -> "String"]];


		env["declareFunction", Native`PrimitiveFunction["Print_MString_Void"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"String"} -> "Void"]];
				
		env["declareType", TypeAlias["Character", "UnsignedInteger32"]];


		env["declareFunction", Native`PrimitiveFunction["AddMStringToCString"], 
						MetaData[
							<|"Linkage" -> "LLVMCompileTools"|>
							]@TypeSpecifier[  
								{"String"} -> "CString"]];

		env["declareFunction", Compile`Cast, 
				inline@Typed[
					{"String", "CString"} -> "CString"
				]@Function[{arg1, arg2}, 
					Native`PrimitiveFunction["AddMStringToCString"][arg1]]];


		]

] (* StaticAnalysisIgnore *)

setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		RegisterMacro[env, Native`Character,
			Native`Character[a_] -> Compile`Internal`MacroEvaluate[ makeCharacter[a]]
		];

	]

makeCharacter[a_] :=
	Module[ {str, code},
		If[ !a["literalQ"],
			ThrowException[ {"The argument of Native`Character should be a string of one character.", a}]];
		str = a["data"];
		If[ !StringQ[str],
			ThrowException[ {"The argument of Native`Character should be a string of one character.", a}]];
		code = ToCharacterCode[str];
		If[ Length[code] =!= 1,
			ThrowException[ {"The argument of Native`Character should be a string of one character.", a}]];
		With[{arg = First[code]},
			CreateMExpr[ Typed, {arg, "Character"}]]
	]


RegisterCallback["SetupTypeSystem", setupTypes]
RegisterCallback["SetupMacros", setupMacros]

End[]

EndPackage[]
