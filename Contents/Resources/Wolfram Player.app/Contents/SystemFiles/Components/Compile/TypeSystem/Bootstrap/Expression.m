
BeginPackage["Compile`TypeSystem`Bootstrap`Expression`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]
Needs["Compile`API`RuntimeErrors`"]

(*
   Base String implementation.

*)

"StaticAnalysisIgnore"[

setupTypes[st_] :=
	With[{env = st["typeEnvironment"],
	      inline = MetaData[<|"Inline" -> "Hint"|>]},

		env["declareType", TypeConstructor["Expression"]];

		env["declareAtom", Expr`EFAIL, TypeSpecifier["Expression"]];

		env["declareFunction", Plus, 
						inline@Typed[
							TypeForAll[ {"a"},
								{"Expression", "a"} -> "Expression"]
							]@Function[{arg1, arg2},
								Module[ {s2 = Compile`Cast[arg2, TypeSpecifier["Expression"]]},
									Native`PrimitiveFunction["Plus_E_E_E"][arg1, s2]
								]
							]];

		env["declareFunction", Plus, 
						inline@Typed[
							TypeForAll[ {"a"},
								{"a", "Expression"} -> "Expression"]
							]@Function[{arg1, arg2},
								Module[ {s1 = Compile`Cast[arg2, TypeSpecifier["Expression"]]},
									Native`PrimitiveFunction["Plus_E_E_E"][s1, arg2]
								]
							]];

		env["declareFunction", Times, 
						inline@Typed[
							TypeForAll[ {"a"},
								{"Expression", "a"} -> "Expression"]
							]@Function[{arg1, arg2},
								Module[ {s2 = Compile`Cast[arg2, TypeSpecifier["Expression"]]},
									Native`PrimitiveFunction["Times_E_E_E"][arg1, s2]
								]
							]];

		env["declareFunction", Times, 
						inline@Typed[
							TypeForAll[ {"a"},
								{"a", "Expression"} -> "Expression"]
							]@Function[{arg1, arg2},
								Module[ {s1 = Compile`Cast[arg1, TypeSpecifier["Expression"]]},
									Native`PrimitiveFunction["Times_E_E_E"][s1, arg2]
								]
							]];

		env["declareFunction", Native`GetPartUnary, 
						inline@Typed[
							{"Expression", "Expression"} -> "Expression"
							]@Function[{arg1, arg2}, 
								Native`PrimitiveFunction["Part_E_E_E"][arg1, arg2]
							]];

		env["declareFunction", Native`GetPartUnary, 
						inline@Typed[
							{"Expression", "MachineInteger"} -> "Expression"
							]@Function[{arg1, arg2}, 
								Native`PrimitiveFunction["Part_E_I_E"][arg1, arg2]
							]];

		env["declareFunction", Head, 
						inline@Typed[
							{"Expression"} -> "Expression"
							]@Function[{arg1}, 
								Native`PrimitiveFunction["Part_E_I_E"][arg1, 0]
							]];


		env["declareFunction", Length, 
						inline@Typed[
							{"Expression"} -> "MachineInteger"
							]@Function[{arg1},
								If[ Native`PrimitiveFunction["Expr`Type"][arg1] === Typed[6, "Integer16"],
									Native`PrimitiveFunction["Expr`NormalLength"][arg1],
									0]
							]];

		env["declareFunction", Native`PrimitiveFunction["Expr`NormalLength"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression"} -> "MachineInteger"]];



		Module[ {compList = {
			{Less, "Less_E_E_E"}, {LessEqual, "LessEqual_E_E_E"}, 
			{Greater, "Greater_E_E_E"}, {GreaterEqual, "GreaterEqual_E_E_E"},
			{Equal, "Equal_E_E_E"}, {Unequal, "Unequal_E_E_E"}}
		},
			Scan[
				With[ {fun = First[#], name = Last[#]},
					env["declareFunction", fun, 
						inline@Typed[
							TypeForAll[{"a"},
							{"Expression", "a"} -> "Expression"]
							]@Function[{arg1, arg2},
								Native`PrimitiveFunction[name][arg1, Compile`Cast[arg2, TypeSpecifier["Expression"]]]
							]];
					env["declareFunction", fun, 
						inline@Typed[
							TypeForAll[{"a"},
							{"a", "Expression"} -> "Expression"]
							]@Function[{arg1, arg2},
								Native`PrimitiveFunction[name][Compile`Cast[arg1, TypeSpecifier["Expression"]], arg2]
							]];
					env["declareFunction", Native`PrimitiveFunction[name], 
						MetaData[
							<|"Linkage" -> "External"|>
						]@TypeSpecifier[{"Expression", "Expression"} -> "Expression"]];
				]&,
				compList
			];
		];

		env["declareFunction", SameQ, 
			inline@Typed[
				{"Expression", "Expression"} -> "Expression"
			]@Function[{arg1, arg2},
				Native`PrimitiveFunction["SameQ_E_E_E"][arg1, arg2]
		]];
		
		env["declareFunction", Native`PrimitiveFunction["SameQ_E_E_E"], 
			MetaData[
				<|"Linkage" -> "External"|>
		]@TypeSpecifier[{"Expression", "Expression"} -> "Expression"]];

		env["declareFunction", UnsameQ, 
			inline@Typed[
				{"Expression", "Expression"} -> "Expression"
			]@Function[{arg1, arg2},
				Native`PrimitiveFunction["UnsameQ_E_E_E"][arg1, arg2]
		]];

		env["declareFunction", Native`PrimitiveFunction["UnsameQ_E_E_E"], 
			MetaData[
				<|"Linkage" -> "External"|>
		]@TypeSpecifier[{"Expression", "Expression"} -> "Expression"]];

		env["declareFunction", Compile`Cast, 
				inline@Typed[
					{"Expression", "Expression"} -> "Expression"
				]@Function[{arg1, arg2}, 
					arg1]];

		env["declareFunction", Compile`Cast, 
				inline@Typed[
					{"Integer32", "Expression"} -> "Expression"
				]@Function[{arg1, arg2}, 
					Native`PrimitiveFunction["Integer32ToExpr"][arg1]]];

		env["declareFunction", Compile`Cast, 
				inline@Typed[
					{"Boolean", "Expression"} -> "Expression"
				]@Function[{arg1, arg2}, 
					If[arg1, Native`ConstantSymbol["System`True"],Native`ConstantSymbol["System`False"]]]];

		env["declareFunction", Compile`Cast, 
				inline@Typed[
					{"Integer64", "Expression"} -> "Expression"
				]@Function[{arg1, arg2}, 
					Native`PrimitiveFunction["Integer64ToExpr"][arg1]]];

		env["declareFunction", Compile`Cast, 
				inline@Typed[
					{"Real64", "Expression"} -> "Expression"
				]@Function[{arg1, arg2}, 
					Native`PrimitiveFunction["Real64ToExpr"][arg1]]];

		env["declareFunction", Compile`Cast, 
				inline@Typed[
					{"String", "Expression"} -> "Expression"
				]@Function[{arg1, arg2}, 
					Native`PrimitiveFunction["StringToExpr"][arg1]]];
		
		Scan[			
			With[ {
					fun = Part[#,1],
					ty = Part[#,2]},			
				env["declareFunction", Compile`Cast, 
						inline@Typed[
							{"Expression", ty} -> ty
						]@Function[{arg1, arg2},
							Module[ {hand = Native`Handle[]},
								If[ !Native`PrimitiveFunction[fun][hand, arg1],
									Native`ThrowWolframException[Typed[Native`ErrorCode["ExpressionConversion"], "Integer32"]]];
								Native`Load[hand]
								]]];							
				env["declareFunction", Native`PrimitiveFunction[fun], 
					MetaData[
						<|"Linkage" -> "LLVMCompileTools"|>
					]@TypeSpecifier[{"Handle"[ty], "Expression"} -> "Boolean"]];
			]&,
		{
			{"ExprToBoolean", "Boolean"},
			{"ExprToInteger32", "Integer32"},
			{"ExprToInteger64", "Integer64"},
			{"ExprToReal64", "Real64"},
			{"ExprToString", "String"}
			}];

		env["declareFunction", Compile`Cast, 
				inline@Typed[
					{"Expression", "Complex"["Real64"]} -> "Complex"["Real64"]
				]@Function[{arg1, arg2},
					Module[ {handRe = Native`Handle[], handIm = Native`Handle[]},
						If[ !Native`PrimitiveFunction["ExprToReal64ReIm"][handRe, handIm, arg1],
							Native`ThrowWolframException[Typed[Native`ErrorCode["ExpressionConversion"], "Integer32"]]];
						Complex[ Native`Load[handRe], Native`Load[handIm]]
						]]];

		env["declareFunction", Native`PrimitiveFunction["CreateComplex_RR_E"], 
			MetaData[
				<|"Linkage" -> "External"|>
			]@TypeSpecifier[{"Real64", "Real64"} -> "Expression"]];
					
		env["declareFunction", Compile`Cast, 
				inline@Typed[
					{"Complex"["Real64"], "Expression"} -> "Expression"
				]@Function[{arg1, arg2},
					Native`PrimitiveFunction["CreateComplex_RR_E"][Re[arg1], Im[arg1]]
					]];
						
		env["declareFunction", Compile`Cast, 
				inline@Typed[
					TypeForAll[ {"a", "b"},
								{"Expression", "PackedArray"["a", "b"]} -> "PackedArray"["a", "b"]]
				]@Function[{arg1, arg2},
					Module[ {hand = Native`Handle[], rank, elemType},
						elemType = Native`MTensorElementType[arg2];
						rank = TensorRank[arg2];
						If[ !Native`PrimitiveFunction["ExprToMTensor"][hand, arg1, elemType, rank],
							Native`ThrowWolframException[Typed[Native`ErrorCode["ExpressionConversion"], "Integer32"]]];
						Native`BitCast[ Native`Load[hand], Compile`TypeOf[arg2]]
						]]];
													
		env["declareFunction", Native`PrimitiveFunction["ExprToReal64ReIm"], 
			MetaData[
				<|"Linkage" -> "LLVMCompileTools"|>
			]@TypeSpecifier[{"Handle"["Real64"], "Handle"["Real64"], "Expression"} -> "Boolean"]];

		env["declareFunction", Native`PrimitiveFunction["ExprToMTensor"], 
			MetaData[
				<|"Linkage" -> "LLVMCompileTools"|>
			]@TypeSpecifier[{"Handle"["MTensor"], "Expression", "Integer32", "MachineInteger"} -> "Boolean"]];

		env["declareFunction", Compile`Cast, 
				inline@Typed[
					TypeForAll[{"elem", "rank"},
						{"PackedArray"["elem", "rank"], "Expression"} -> "Expression"]
				]@Function[{arg1, arg2},
					Native`PrimitiveFunction["CreateMTensorExpr"][Native`BitCast[arg1, "MTensor"]]]];
	
					
		env["declareFunction", Compile`Cast, 
				inline@Typed[
					{"Expression", "CString"} -> "CString"
				]@Function[{arg1, arg2},
					Module[{hand = Native`Handle[]},
						If[
							!Native`PrimitiveFunction["TestGet_CString"][arg1, hand],
							Native`ThrowWolframException[Typed[Native`ErrorCode["ExpressionConversion"], "Integer32"]]];
						Native`Load[hand]
					]
				]];


		env["declareFunction", Native`PrimitiveFunction["TestGet_CString"], 
			MetaData[
				<|"Linkage" -> "External"|>
			]@TypeSpecifier[{"Expression", "Handle"["CString"]} -> "Boolean"]];



		env["declareFunction", Native`PrimitiveFunction["Integer32ToExpr"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Integer32"} -> "Expression"]];

		env["declareFunction", Native`PrimitiveFunction["Integer64ToExpr"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Integer64"} -> "Expression"]];


		env["declareFunction", Native`PrimitiveFunction["ExprToCString"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression"} -> "CString"]];

		env["declareFunction", Native`PrimitiveFunction["Real64ToExpr"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Real64"} -> "Expression"]];

		env["declareFunction", Native`PrimitiveFunction["StringToExpr"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"String"} -> "Expression"]];


		env["declareFunction", Native`PrimitiveFunction["Length_E_I"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression"} -> "MachineInteger"]];

		env["declareFunction", Native`PrimitiveFunction["Flags_E_UI16"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression"} -> "UnsignedInteger16"]];

		env["declareFunction", Native`PrimitiveFunction["TestGet_Integer"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression", "UnsignedInteger32", "UnsignedInteger32", "Handle"["Integer64"]} -> "UnsignedInteger16"]];

		env["declareFunction", Native`PrimitiveFunction["TestGet_ComplexFloat"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression", "UnsignedInteger32", "Handle"["Real64"], "Handle"["Real64"]} -> "UnsignedInteger16"]];

		env["declareFunction", Native`PrimitiveFunction["TestGet_Float"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression", "UnsignedInteger32", "Handle"["Real64"]} -> "UnsignedInteger16"]];

		env["declareFunction", Native`PrimitiveFunction["TestGet_MTensor"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression", "Integer32", "MachineInteger", "Handle"["MTensor"]} -> "UnsignedInteger16"]];

		env["declareFunction", Native`PrimitiveFunction["TestGet_MNumericArray"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression", "Integer32", "MachineInteger", "Handle"["MTensor"]} -> "UnsignedInteger16"]];

		env["declareFunction", Native`PrimitiveFunction["CreateMTensorExpr"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"MTensor"} -> "Expression"]];

		env["declareFunction", Native`PrimitiveFunction["CreateMNumericArrayExpr"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"MNumericArray"} -> "Expression"]];


		env["declareFunction", Native`PrimitiveFunction["TestGet_Function"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression", "CString", "Handle"["VoidHandle"]} -> "UnsignedInteger16"]];


		env["declareFunction", Print,
						Typed[
							{"Expression"} -> "Expression"
							]@Function[{e}, 
								Module[ {print, ef, args},
									print = Native`LookupSymbol["System`Print"];
									args = Native`StackArray[1];
									Native`SetElement[args, 0,  e];
									ef = Native`CreateExpression[print, 1, args];
									ef = Native`Evaluate[ef];
									ef
								]
							]];

		env["declareFunction", Native`PrimitiveFunction["StringFromExpr"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression"} -> "String"]];

		env["declareFunction", Not, 
						inline@Typed[
							{"Expression"} -> "Expression"
							]@Function[{arg1},
								Which[
									Native`SameInstanceQ[arg1, Native`ConstantSymbol["System`True"]],
									    Native`ConstantSymbol["System`False"],
									Native`SameInstanceQ[arg1, Native`ConstantSymbol["System`False"]],
									    Native`ConstantSymbol["System`True"],
									True,
									    arg1]
							]];

		env["declareFunction", TrueQ, 
				inline@Typed[
					{"Expression"} -> "Boolean"
					]@Function[{arg1}, 
						Native`SameInstanceQ[arg1, Native`ConstantSymbol["System`True"]]
					]];
					


		env["declareFunction", Native`SameInstanceQ, 
				inline@Typed[
					{"Expression", "Expression"} -> "Boolean"
					]@Function[{arg1, arg2}, 
						Native`PrimitiveFunction["Native`SameInstanceQ"][arg1, arg2]]];

		env["declareFunction", Native`PrimitiveFunction["Native`SameInstanceQ"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression", "Expression"} -> "Boolean"]];



		env["declareFunction", Native`LookupSymbol, 
				inline@Typed[
					{"String"} -> "Expression"
					]@Function[{arg1},
						Module[ {eName},
							eName = Compile`Cast[arg1, TypeSpecifier["Expression"]];
							Native`PrimitiveFunction["LookupSymbol_E_E"][eName]
						]]];
		
		env["declareFunction", Native`PrimitiveFunction["LookupSymbol_E_E"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"Expression"} -> "Expression"]];

		env["declareFunction", Native`PrimitiveFunction["CreateGeneralExpr"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"CArray"["UnsignedInteger8"]} -> "Expression"]];


		env["declareFunction", Native`PrimitiveFunction["CreateHeaded_IE_E"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"MachineInteger", "Expression"} -> "Expression"]];

		env["declareFunction", Native`PrimitiveFunction["SetElement_EIE_Void"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"Expression", "MachineInteger", "Expression"} -> "Void"]];


		env["declareFunction", Native`CreateExpression, 
				Typed[{"Expression", "MachineInteger", "CArray"["Expression"]} -> "Expression"
				]@Function[{head, len, args},
					Module[ {ef, i, ei},
						ef = Native`PrimitiveFunction["CreateHeaded_IE_E"][len, head];
				     	i = 0;
				     	While[ i < len,
				     		ei = Native`GetElement[args,i];
				      		Native`PrimitiveFunction["SetElement_EIE_Void"][ef, i+1, ei];
				      		i = i + 1];
				     	ef
					]
				]];


		Quiet[
			env["declareFunction", Native`CreateExpression, 
				Typed[{TypeSequence["Expression", {1, Infinity}]} -> "Expression"
				]@Function[{Compile`ArgumentSequence[a]},
					Module[ {len, array, index = -1, head, ei},
						len = Compile`SequenceLength[a]-1;
						array = Native`StackArray[len];
						head = Compile`SequenceElement[a, 0];
						Compile`SequenceIterate[
							If[index >= 0,
								ei = Compile`SequenceSlot[];
								Native`SetElement[array, index, ei]];
							index = index+1;
							, a];
						Native`CreateExpression[head, len, array]
					]
				]],
		(* this is a benign message warning about Function[{ArgumentSequence[a]}, ...] *)
		{Function::flpar}];

		Quiet[
			env["declareFunction", Native`CreateEvaluateExpression, 
				Typed[{TypeSequence["Expression", {1, Infinity}]} -> "Expression"
				]@Function[{Compile`ArgumentSequence[a]},
					Module[ {len, array, index = -1, head, ei, ef},
						len = Compile`SequenceLength[a]-1;
						array = Native`StackArray[len];
						head = Compile`SequenceElement[a, 0];
						Compile`SequenceIterate[
							If[index >= 0,
								ei = Compile`SequenceSlot[];
								Native`SetElement[array, index, ei]];
							index = index+1;
							, a];
						ef = Native`CreateExpression[head, len, array];
						Native`Evaluate[ef]
					]
				]],
		(* this is a benign message warning about Function[{ArgumentSequence[a]}, ...] *)
		{Function::flpar}];



		env["declareFunction", Native`Evaluate, 
				inline@Typed[
					{"Expression"} -> "Expression"
					]@Function[{arg1}, 
						Native`PrimitiveFunction["Evaluate_E_E"][arg1]]];

		env["declareFunction", Native`PrimitiveFunction["Evaluate_E_E"],
		    inline@
		    MetaData[<|"Linkage" -> "External"|>]@
			TypeSpecifier[{"Expression"} -> "Expression"]];



		env["declareFunction", Native`PrimitiveFunction["Expr`Type"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression"} -> "Integer16"]];

		env["declareFunction", Native`PrimitiveFunction["Expr`RawType"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression"} -> "Integer16"]];

		env["declareFunction", Native`PrimitiveFunction["Plus_E_E_E"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"Expression", "Expression"} -> "Expression"]];

		env["declareFunction", Native`PrimitiveFunction["Times_E_E_E"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"Expression", "Expression"} -> "Expression"]];

		env["declareFunction", Native`PrimitiveFunction["Part_E_E_E"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"Expression", "Expression"} -> "Expression"]];

		env["declareFunction", Native`PrimitiveFunction["Part_E_I_E"], 
				MetaData[
					<|"Linkage" -> "External"|>
				]@TypeSpecifier[{"Expression", "MachineInteger"} -> "Expression"]];

		env["declareFunction", Native`PrimitiveFunction["DecrementReferenceCount"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression"} -> "Integer64"]];

		env["declareFunction", Native`PrimitiveFunction["IncrementReferenceCount"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression"} -> "Integer64"]];

		env["declareFunction", Native`PrimitiveFunction["Expr`RawContents"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"Expression"} -> "VoidHandle"]];

		]
] (* StaticAnalysisIgnore *)

RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]
