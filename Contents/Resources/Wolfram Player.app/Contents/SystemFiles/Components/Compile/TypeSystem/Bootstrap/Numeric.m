
BeginPackage["Compile`TypeSystem`Bootstrap`Numeric`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]

resolveIntegerSize[ tyEnv_, size_] :=
	Module[{envSize = tyEnv["getProperty", "MachineIntegerSize", Null]},
		envSize === size
	]


"StaticAnalysisIgnore"[

setupTypes[st_] :=
	Module[{env = st["typeEnvironment"],
	        inline = MetaData[<|"Inline" -> "Hint"|>],
	        llvmLinkage = MetaData[<|"Linkage" -> "LLVMCompileTools"|>]},
		env["declareType", TypeConstructor["Bit", "ShortName" -> "Bit", "Implements" -> "Integral"]];
		env["declareType", TypeConstructor["Integer8",              "ShortName" -> "I8",   "ByteCount" -> 1,  "Implements" -> "SignedIntegral"]];
		env["declareType", TypeConstructor["Integer16",             "ShortName" -> "I16",  "ByteCount" -> 2,  "Implements" -> "SignedIntegral"]];
		env["declareType", TypeConstructor["Integer32",             "ShortName" -> "I32",  "ByteCount" -> 4,  "Implements" -> "SignedIntegral"]];
		env["declareType", TypeConstructor["Integer64",             "ShortName" -> "I64",  "ByteCount" -> 8,  "Implements" -> "SignedIntegral"]];
		env["declareType", TypeConstructor["UnsignedInteger8",      "ShortName" -> "UI8",  "ByteCount" -> 1,  "Implements" -> "UnsignedIntegral"]];
		env["declareType", TypeConstructor["UnsignedInteger16",     "ShortName" -> "UI16", "ByteCount" -> 2,  "Implements" -> "UnsignedIntegral"]];
		env["declareType", TypeConstructor["UnsignedInteger32",     "ShortName" -> "UI32", "ByteCount" -> 4,  "Implements" -> "UnsignedIntegral"]];
		env["declareType", TypeConstructor["UnsignedInteger64",     "ShortName" -> "UI64", "ByteCount" -> 8,  "Implements" -> "UnsignedIntegral"]];
		env["declareType", TypeConstructor["Real16",                "ShortName" -> "R16",  "ByteCount" -> 2,  "Implements" -> "RealFloatingPoint"]];
		env["declareType", TypeConstructor["Real32",                "ShortName" -> "R32",  "ByteCount" -> 4,  "Implements" -> "RealFloatingPoint"]];
		env["declareType", TypeConstructor["Real64",                "ShortName" -> "R64",  "ByteCount" -> 8,  "Implements" -> "RealFloatingPoint"]];
	
	
		env["declareType", AbstractType["SignedIntegral", "ShortName" -> "I", "Deriving" -> "Integral"]];
		env["declareType", AbstractType["UnsignedIntegral", "ShortName" -> "UI", "Deriving" -> "Integral"]];

		env["declareType", MetaData["Constraint"-> (resolveIntegerSize[#, 64]&)]@TypeAlias["MachineInteger", "Integer64"]];
		env["declareType", MetaData["Constraint"-> (resolveIntegerSize[#, 32]&)]@TypeAlias["MachineInteger", "Integer32"]];
      	env["declareType", MetaData["Constraint"-> (resolveIntegerSize[#, 64]&)]@TypeAlias["UnsignedMachineInteger", "UnsignedInteger64"]];
		env["declareType", MetaData["Constraint"-> (resolveIntegerSize[#, 32]&)]@TypeAlias["UnsignedMachineInteger", "UnsignedInteger32"]];

		createUnaryIdentityFunction[] :=
			inline@Typed[
				Function[{arg1}, arg1],
 						{"a"} -> "a"];
 						
		createUnaryFunction[ fun_] :=
			Typed[
				Function[{arg1},
  							Native`PrimitiveFunction[fun][arg1]
   							],
 						{"a"} -> "a"];

		createUnaryFunction[ fun_, outTy_] :=
			Typed[
				Function[{arg1},
  							Native`PrimitiveFunction[fun][arg1]],
 						{"a"} -> outTy];
 						
		createBinaryFunction[ fun_, ty1_, ty2_] :=
			Typed[
				Function[{x, y},
					Module[ {
						arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
						arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
						Native`PrimitiveFunction[fun][arg1, arg2]
				]],
				{ty1, ty2} -> "TypeJoin"[ty1, ty2]];

        env["declareType", AbstractType["Ordered", {"a", "b"},
            {
                 Less -> 
                	inline@Typed[{"a", "b"} -> "Boolean"
                	]@Function[ {x, y},
						Module[ {
							arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
							arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
                 	     	Native`PrimitiveFunction["binary_less"][arg1, arg2]
						]],

                 LessEqual -> 
                	inline@Typed[{"a", "b"} -> "Boolean"
                	]@Function[ {x, y},
						Module[ {
							arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
							arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
                 	     	Native`PrimitiveFunction["binary_lessequal"][arg1, arg2]
						]],

                 Greater -> 
                	inline@Typed[{"a", "b"} -> "Boolean"
                	]@Function[ {x, y},
						Module[ {
							arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
							arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
                 	     	Native`PrimitiveFunction["binary_greater"][arg1, arg2]
						]],

                 GreaterEqual -> 
                	inline@Typed[{"a", "b"} -> "Boolean"
                	]@Function[ {x, y},
						Module[ {
							arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
							arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
                 	     	Native`PrimitiveFunction["binary_greaterequal"][arg1, arg2]
						]]
	        },
	        "Deriving" -> "Equal"
        ]];

        env["declareType", AbstractType["Number", {"a", "b"},
            {
            	Plus -> MetaData[<|"Inline"->"Minimal"|>]@createBinaryFunction[ "binary_plus", "a", "b"],
 						
            	Subtract -> MetaData[<|"Inline"->"Minimal"|>]@createBinaryFunction[ "binary_subtract", "a", "b"],

            	Times -> MetaData[<|"Inline"->"Minimal"|>]@createBinaryFunction[ "binary_times", "a", "b"],

(*
  Should really be a compile error for Unsigned types.
*)
            	Minus -> createUnaryFunction[ "unary_minus"],
	            
            	ToString -> 
	            	Typed[
 						Function[{x},
 							Module[{
 								chars = Native`PrimitiveFunction["to_string"][x]
 							},
 								Native`PrimitiveFunction["NewMString_UI8_MString"][chars]
 							]
   						],
 						{"a"} -> "String"],

        		Mod -> createBinaryFunction[ "binary_mod", "a", "b"],

        		FractionalPart -> createUnaryFunction["unary_fracpart"],
        		IntegerPart -> createUnaryFunction["unary_intpart", "MachineInteger"],
        		
       			Fibonacci -> createUnaryFunction["unary_fibonacci"],
               	LucasL -> createUnaryFunction[ "unary_lucasl"],
       			Gamma -> createUnaryFunction["unary_gamma"]

            },
            "Deriving" -> {"Equal", "StringSerializable"},
            "Default" -> "MachineInteger"
        ]];


		env["declareFunction", SameQ, 
			inline@Typed[
				TypeForAll[ {"a"}, {Element["a", "Integral"]}, {"a", "a"} -> "Boolean"]
			]@Function[{x, y},
 				Native`PrimitiveFunction["binary_sameq"][x,y]
				]];

		inline@env["declareFunction", SameQ, 
			Typed[
				TypeForAll[ {"a", "a"}, {Element["a", "RealFloatingPoint"]}, {"a", "a"} -> "Boolean"]
			]@Function[{x, y},
 				Native`PrimitiveFunction["binary_sameq"][x,y]
				]];

		env["declareFunction", SameQ, 
			inline@Typed[
				TypeForAll[ {"a"}, {"Complex"["a"], "Complex"["a"]} -> "Boolean"]
			]@Function[{x, y},
 				Compile`EagerAnd[Re[x]===Re[y], Im[x]===Im[y]]
				]];


		env["declareFunction", Equal, 
			inline@Typed[
				TypeForAll[ {"a", "b"}, {Element["a", "Real"], Element["b", "Real"]}, {"a", "b"} -> "Boolean"]
			]@Function[{x, y},
				Module[ {
						x1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
						y1 = Compile`Cast[y, Compile`TypeJoin[x, y]]
					},
 					Native`PrimitiveFunction["binary_equal"][x1,y1]
				]
				]];


		env["declareFunction", Equal, 
			inline@Typed[
				TypeForAll[ {"a"}, {"Complex"["a"], "Complex"["a"]} -> "Boolean"]
			]@Function[{x, y},
 				Compile`EagerAnd[Re[x] == Re[y], Im[x] == Im[y]]
				]];

		env["declareFunction", Equal, 
			inline@Typed[
				TypeForAll[ {"a", "b"}, {Element["a", "Real"]}, {"a", "Complex"["b"]} -> "Boolean"]
			]@Function[{x, y},
				Compile`EagerAnd[x == Re[y], 0 == Im[y]]
				]];

		env["declareFunction", Equal, 
			inline@Typed[
				TypeForAll[ {"a", "b"}, {Element["a", "Real"]}, {"Complex"["b"], "a"} -> "Boolean"]
			]@Function[{x, y},
				Compile`EagerAnd[Re[x] == y, 0 == Im[x]]
				]];

        env["declareType", AbstractType["Enumerable", {"a"},
            {
	            Increment -> TypeSpecifier[{"a"} -> "a"],
	            Decrement -> TypeSpecifier[{"a"} -> "a"],
	            PreIncrement -> TypeSpecifier[{"a"} -> "a"],
	            PreDecrement -> TypeSpecifier[{"a"} -> "a"]
            }
        ]];

        env["declareType", AbstractType["Real", {"a", "b"},
        	{
        	},
            "Deriving" -> {"Ordered", "Number"}
        ]];

        env["declareType", AbstractType["Integral", {"a"},
            {        	
				Ceiling -> createUnaryIdentityFunction[],
				Floor -> createUnaryIdentityFunction[],
				Round -> createUnaryIdentityFunction[],
				LucasL -> createUnaryFunction["unary_lucasl"],
				UnitStep -> createUnaryFunction["unary_nneg", "MachineInteger"],
				BitLength -> createUnaryFunction["unary_bit_length"]
            },
            "Deriving" -> {"Real"}
        ]];
        
        
		Scan[
        	With[ {name = Part[#, 1], primName1 = Part[#,2], primName2 = Part[#,3]},
         		env["declareFunction", name,
        			Typed[ TypeForAll[ {"a", "b"}, 
        				{Element["a", "Integral"], Element["b", "SignedIntegral"]}, 
        					{"a", "b"} -> "TypeJoin"["a", "b"]]
        			]@Function[ {x,y},
        				Module[ {
        					arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
							arg2, arg3
						},
        					If[ y < 0,
        						arg2 = Compile`Cast[-y, Compile`TypeJoin[x, y]];
        						Native`PrimitiveFunction[primName2][arg1, arg2]
        						,
        						arg3 = Compile`Cast[y, Compile`TypeJoin[x, y]];
        						Native`PrimitiveFunction[primName1][arg1, arg3]]
        				]
        			]];
       		env["declareFunction", name,
        			Typed[ TypeForAll[ {"a", "b"}, 
        				{Element["a", "Integral"], Element["b", "UnsignedIntegral"]}, 
        					{"a", "b"} -> "a"]
        			]@Function[ {x,y},
          					Native`PrimitiveFunction[primName1][x, y]
        			]];
  				env["declareFunction", Native`PrimitiveFunction[primName1], 
					MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
					]@TypeSpecifier[TypeForAll[ {"a", "b"},{Element["a", "Integral"], Element["b", "Integral"]}, {"a", "b"} -> "a"]]];
        		]&,
        		{
       				{BitShiftLeft, "BitShiftLeft","BitShiftRight"},
       				{BitShiftRight,"BitShiftRight","BitShiftLeft"}
        		}];

		With[{
			fun = Function[{x, y},
					Module[{
						arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
						arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]
					},
					Native`PrimitiveFunction["binary_quotient"][arg1, arg2]
					]]
			},
			env["declareFunction", Quotient,
					Typed[ TypeForAll[ {"a","b"},
						{Element["a", "Integral"], Element["b", "Integral"]}, 
        					{"a", "b"} -> "TypeJoin"["a", "b"]]
					]@fun];
			env["declareFunction", Quotient,
					Typed[ TypeForAll[ {"a","b"},
						{Element["a", "FloatingPoint"], Element["b", "Integral"]}, 
        					{"a", "b"} -> "MachineInteger"]
					]@fun];
			env["declareFunction", Quotient,
					Typed[ TypeForAll[ {"a","b"},
						{Element["a", "Integral"], Element["b", "FloatingPoint"]}, 
        					{"a", "b"} -> "MachineInteger"]
					]@fun];
			env["declareFunction", Quotient,
					Typed[ TypeForAll[ {"a","b"},
						{Element["a", "FloatingPoint"], Element["b", "FloatingPoint"]}, 
        					{"a", "b"} -> "MachineInteger"]
					]@fun];
			];


        Scan[
        	With[ {name = First[#], primName = Last[#]},
        		env["declareFunction", name,
        			Typed[ TypeForAll[ {"a", "b"}, 
        				{Element["a", "Integral"], Element["b", "Integral"]}, 
        					{"a", "b"} -> "TypeJoin"["a", "b"]]
        			]@Function[ {x,y},
        				Module[ {
   							arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
   							arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]
        				},
        					Native`PrimitiveFunction[primName][arg1, arg2]
        				]
        			]];
  				env["declareFunction", Native`PrimitiveFunction[primName], 
					MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
					]@TypeSpecifier[TypeForAll[ {"a"},{Element["a", "Integral"]}, {"a", "a"} -> "a"]]];
        		]&,
        		{
        			{BitAnd, "BitAnd"},
       				{BitOr, "BitOr"},
       				{BitXor, "BitXor"}
        		}];
        env["declareFunction", BitNot,
        			Typed[ TypeForAll[ {"a"}, {Element["a", "Integral"]}, {"a"} -> "a"]
        			]@Function[ {x},
        				Module[ {cons},
        					cons = Compile`Cast[-1, Compile`TypeOf[x]];
        					Native`PrimitiveFunction["BitXor"][x, cons]
        				]
        			]];       			
 
 		env["declareFunction", EvenQ,
			Typed[TypeForAll[{"a"}, {Element["a", "Integral"]}, {"a"} -> "Boolean"]
			]@Function[ {arg},               
 				BitAnd[arg, Typed[1, Compile`TypeOf[arg]]] === Typed[0, Compile`TypeOf[arg]]
 				]];
 				
		env["declareFunction", OddQ,
			Typed[TypeForAll[{"a"}, {Element["a", "Integral"]}, {"a"} -> "Boolean"]
			]@Function[ {arg},               
 				BitAnd[arg, Typed[1, Compile`TypeOf[arg]]] === Typed[1, Compile`TypeOf[arg]]
 				]];

		env["declareFunction", Unitize,
					Typed[ TypeForAll[ {"num"},
						{Element["num", "Number"]}, 
        					{"num"} -> "MachineInteger"]
					]@Function[{arg},
						If[arg == 0, 0, 1]]
			];

        			
        env["declareType", AbstractType["RealFractional", {"a"},
            "Deriving" -> {"Real"}
        ]];

        env["declareType", AbstractType["Fractional", {"a"},
            "Deriving" -> {"Number"}
        ]];


        env["declareType", AbstractType["FloatingPoint", {"a"},
            {
            	Sin -> createUnaryFunction[ "unary_sin"],
            	Cos -> createUnaryFunction[ "unary_cos"],
                Tan -> createUnaryFunction[ "unary_tan"],
                Sec -> createUnaryFunction[ "unary_sec"],
                Csc -> createUnaryFunction[ "unary_csc"],
                Cot -> createUnaryFunction[ "unary_cot"],
                ArcSin -> createUnaryFunction[ "unary_asin"],
                ArcCos -> createUnaryFunction[ "unary_acos"],
                ArcTan -> createUnaryFunction[ "unary_atan"],
                ArcSec  -> createUnaryFunction[ "unary_asec"],
                ArcCsc -> createUnaryFunction[ "unary_acsc"],
               	ArcCot -> createUnaryFunction[ "unary_acot"],
                Sinh -> createUnaryFunction[ "unary_sinh"],
                Cosh  -> createUnaryFunction[ "unary_cosh"],
              	Tanh -> createUnaryFunction[ "unary_tanh"],
                Sech  -> createUnaryFunction[ "unary_sech"],
                Csch -> createUnaryFunction[ "unary_csch"],
                Coth -> createUnaryFunction[ "unary_coth"],
               	ArcSinh -> createUnaryFunction[ "unary_asinh"],
                ArcCosh -> createUnaryFunction[ "unary_acosh"],
                ArcTanh -> createUnaryFunction[ "unary_atanh"],
                ArcSech  -> createUnaryFunction[ "unary_asech"],
                ArcCsch -> createUnaryFunction[ "unary_acsch"],
                ArcCoth -> createUnaryFunction[ "unary_acoth"],
                Exp -> createUnaryFunction[ "unary_exp"],
                Internal`Expm1 -> createUnaryFunction[ "unary_expm1"],
                Log -> createUnaryFunction[ "unary_log"],
                Internal`Log1p -> createUnaryFunction[ "unary_log1p"],
                Log2 -> createUnaryFunction[ "unary_log2"],
                Log10 -> createUnaryFunction[ "unary_log10"],
                Sqrt -> createUnaryFunction[ "unary_sqrt"],
                Internal`ReciprocalSqrt -> createUnaryFunction[ "unary_rsqrt"],
                Floor -> createUnaryFunction["unary_floor", "MachineInteger"],
                Round -> createUnaryFunction["unary_round", "MachineInteger"],
                Ceiling -> createUnaryFunction["unary_ceiling", "MachineInteger"],
        		IntegerPart -> createUnaryFunction["unary_intpart", "MachineInteger"],
                CubeRoot -> createUnaryFunction[ "unary_cbrt"],
               	Sinc -> createUnaryFunction[ "unary_sinc"],
                Erf -> createUnaryFunction[ "unary_erf"],
                Erfc -> createUnaryFunction[ "unary_erfc"],
                LogGamma -> createUnaryFunction[ "unary_loggamma"],
                Gudermannian -> createUnaryFunction[ "unary_gudermannian"],
                Haversine -> createUnaryFunction[ "unary_haversine"],
                InverseGudermannian -> createUnaryFunction[ "unary_inversegudermannian"],
                InverseHaversine -> createUnaryFunction[ "unary_inversehaversine"],
                UnitStep -> createUnaryFunction["unary_nneg", "MachineInteger"],
               	Ramp -> createUnaryFunction["unary_ramp"]
            },
            "Deriving" -> {"Fractional"}
        ]];

		Scan[
			With[ {sym = First[#], primFun = Last[#]},
				env["declareFunction", sym, Typed[
					TypeForAll[ {"a", "b"}, {Element["a", "FloatingPoint"], Element["b", "Number"]}, {"a", "b"} -> "TypeJoin"["a", "b"]]
					]@Function[{x, y},
	 					Module[ {
		   					arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
		   					arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
		   					Native`PrimitiveFunction[primFun][arg1, arg2]
		  				]]
					];
				env["declareFunction", sym, Typed[
					TypeForAll[ {"a", "b"}, {Element["a", "Number"], Element["b", "FloatingPoint"]}, {"a", "b"} -> "TypeJoin"["a", "b"]]
					]@Function[{x, y},
	 					Module[ {
		   					arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
		   					arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
		   					Native`PrimitiveFunction[primFun][arg1, arg2]
		  				]]
					];
			]&, {{Log, "binary_log"}, {ArcTan, "binary_atan2"}}
		];


        env["declareType", AbstractType["RealFloatingPoint", {"a"},
            "Deriving" -> {"RealFractional", "FloatingPoint"},
            "Default" -> "Real64"
        ]];

		env["declareFunction", Power, Typed[TypeForAll[ {"a", "b"}, {Element["a", "FloatingPoint"], Element["b", "Real"]}, {"a", "b"} -> "a"]
			]@Function[{arg1, arg2},
				Native`PrimitiveFunction["binary_pow"][arg1, arg2]
				]
			];

		env["declareFunction", Power, Typed[TypeForAll[ {"a", "b"}, {Element["a", "Real"], Element["b", "FloatingPoint"]}, {"a", "b"} -> "b"]
			]@Function[{arg1, arg2},
				Module[ {
					cast = Compile`Cast[arg1, Compile`TypeOf[arg2]]
				},
					Native`PrimitiveFunction["binary_pow"][cast, arg2]
				]]
			];

		env["declareFunction", Power, Typed[TypeForAll[ {"a", "b"}, {Element["a", "Integral"], Element["b", "Integral"]}, {"a", "b"} -> "TypeJoin"["a", "b"]]
			]@Function[{arg1, arg2},
				Module[ {
					cast1 = Compile`Cast[arg1, Compile`TypeJoin[arg1, arg2]],
					cast2 = Compile`Cast[arg2, Compile`TypeJoin[arg1, arg2]]
				},
					Native`PrimitiveFunction["binary_pow"][cast1, cast2]
				]]
			];

		env["declareFunction", Power, Typed[TypeForAll[ {"a", "b"}, {Element["a", "FloatingPoint"], Element["b", "FloatingPoint"]}, {"a", "b"} -> "TypeJoin"["a", "b"]]
			]@Function[{arg1, arg2},
				Module[ {
					cast1 = Compile`Cast[arg1, Compile`TypeJoin[arg1, arg2]],
					cast2 = Compile`Cast[arg2, Compile`TypeJoin[arg1, arg2]]
				},
					Native`PrimitiveFunction["binary_pow"][cast1, cast2]
				]]
			];


		env["declareFunction", Divide, Typed[TypeForAll[ {"a", "b"}, {Element["a", "FloatingPoint"], Element["b", "FloatingPoint"]}, {"a", "b"} -> "TypeJoin"["a", "b"]]
			]@Function[{arg1, arg2},
				Module[ {
					cast1 = Compile`Cast[arg1, Compile`TypeJoin[arg1, arg2]],
					cast2 = Compile`Cast[arg2, Compile`TypeJoin[arg1, arg2]]
				},
					Native`PrimitiveFunction["binary_divide"][cast1, cast2]
				]]
			];

		env["declareFunction", Divide, Typed[TypeForAll[ {"a", "b"}, {Element["a", "FloatingPoint"], Element["b", "Number"]}, {"a", "b"} -> "TypeJoin"["a", "b"]]
			]@Function[{arg1, arg2},
				Module[ {
					cast1 = Compile`Cast[arg1, Compile`TypeJoin[arg1, arg2]],
					cast2 = Compile`Cast[arg2, Compile`TypeJoin[arg1, arg2]]
				},
					Native`PrimitiveFunction["binary_divide"][cast1, cast2]
				]]
			];

		env["declareFunction", Divide, Typed[TypeForAll[ {"a", "b"}, {Element["a", "Number"], Element["b", "FloatingPoint"]}, {"a", "b"} -> "TypeJoin"["a", "b"]]
			]@Function[{arg1, arg2},
				Module[ {
					cast1 = Compile`Cast[arg1, Compile`TypeJoin[arg1, arg2]],
					cast2 = Compile`Cast[arg2, Compile`TypeJoin[arg1, arg2]]
				},
					Native`PrimitiveFunction["binary_divide"][cast1, cast2]
				]]
			];


		env["declareType", TypeConstructor["Rational", {"*"} -> "*"]];
		env["declareType", TypeInstance["Integral", {Element["a", "Integral"]}, "Rational"["a"]]];

		env["declareType", TypeConstructor["Complex", {"*"} -> "*", "ByteCountFunction" -> (2*Total[#]&)]];
		env["declareType", TypeInstance["FloatingPoint", {Element["a", "FloatingPoint"]}, "Complex"["a"]]];
		env["declareType", TypeInstance["Number", {Element["a", "Number"]}, "Complex"["a"]]];
		env["declareType", TypeAlias["ComplexReal64", "Complex"["Real64"]]];
		env["declareType", TypeAlias["Complex128", "Complex"["Real64"]]];
		env["declareType", TypeAlias["ComplexReal32", "Complex"["Real32"]]];
		env["declareType", TypeAlias["Complex64", "Complex"["Real32"]]];
 
   		env["declareFunction", Re, 
					inline@Typed[TypeForAll[ {"a"},{Element["a", "Real"]}, {"a"} -> "a"]
	            	]@Function[{arg},
	            		arg
	            	]];


    	env["declareFunction", Im, 
					inline@Typed[TypeForAll[ {"a"},{Element["a", "Integral"]}, {"a"} -> "a"]
	            	]@Function[{arg},
	            		Compile`ConstantZero[arg]
	            	]];
  
    	env["declareFunction", Im, 
					inline@Typed[TypeForAll[ {"a"},{Element["a", "RealFloatingPoint"]}, {"a"} -> "MachineInteger"]
	            	]@Function[{arg},
	            		0
	            	]];
  
    	env["declareFunction", Arg, 
					inline@Typed[TypeForAll[ {"a"},{Element["a", "Integral"]}, {"a"} -> "a"]
	            	]@Function[{arg},
	            		Native`PrimitiveFunction["unary_arg"][arg]
	            	]];
  
    	env["declareFunction", Arg, 
					inline@Typed[TypeForAll[ {"a"},{Element["a", "RealFloatingPoint"]}, {"a"} -> "MachineInteger"]
	            	]@Function[{arg},
	            		Native`PrimitiveFunction["unary_arg"][arg]
	            	]];
	            	
   		env["declareFunction", Arg, 
					inline@Typed[TypeForAll[ {"a"}, {Element["a", "RealFloatingPoint"]}, {"Complex"["a"]} -> "a"]
	            	]@Function[{arg},
	            		Native`PrimitiveFunction["unary_arg"][arg]
	            	]];

    	env["declareFunction", Sign, 
					inline@Typed[TypeForAll[ {"a"},{Element["a", "Integral"]}, {"a"} -> "a"]
	            	]@Function[{arg},
	            		Native`PrimitiveFunction["unary_sign"][arg]
	            	]];
  
    	env["declareFunction", Sign, 
					inline@Typed[TypeForAll[ {"a"},{Element["a", "RealFloatingPoint"]}, {"a"} -> "MachineInteger"]
	            	]@Function[{arg},
	            		Native`PrimitiveFunction["unary_sign"][arg]
	            	]];
	            	
   		env["declareFunction", Sign, 
					inline@Typed[TypeForAll[ {"a"}, {Element["a", "RealFloatingPoint"]}, {"Complex"["a"]} -> "Complex"["a"]]
	            	]@Function[{arg},
	            		Native`PrimitiveFunction["unary_sign"][arg]
	            	]];

 		env["declareFunction", Conjugate, 
					inline@Typed[TypeForAll[ {"a"},{Element["a", "Real"]}, {"a"} -> "a"]
	            	]@Function[{arg},
	            		arg
	            	]];
  
 		env["declareFunction", Abs, 
					inline@Typed[TypeForAll[ {"a"},{Element["a", "Real"]}, {"a"} -> "a"]
	            	]@Function[{arg},
	            		Native`PrimitiveFunction["unary_abs"][arg]
	            	]];
  
   
  		env["declareFunction", Re, 
					inline@Typed[TypeForAll[ {"a"}, {"Complex"["a"]} -> "a"]
	            	]@Function[{arg},
	            		Native`PrimitiveFunction["ExtractElement"][arg, 0]
	            	]];

 		env["declareFunction", Im, 
					inline@Typed[TypeForAll[ {"a"}, {"Complex"["a"]} -> "a"]
	            	]@Function[{arg},
	            		Native`PrimitiveFunction["ExtractElement"][arg, 1]
	            	]];

		env["declareFunction", Abs, 
					inline@Typed[TypeForAll[ {"a"}, {Element["a", "RealFloatingPoint"]}, {"Complex"["a"]} -> "a"]
	            	]@Function[{arg},
	            		Native`PrimitiveFunction["unary_abs"][arg]
	            	]];
	            	
		env["declareFunction", Conjugate, 
					inline@Typed[TypeForAll[ {"a"}, {"Complex"["a"]} -> "Complex"["a"]]
	            	]@Function[{arg},
	            		Complex[ Re[arg], -Im[arg]]
	            	]];

		env["declareFunction", Native`PrimitiveFunction["ExtractElement"], 
						llvmLinkage@TypeSpecifier[TypeForAll[{"a", "b"}, {"a"["b"], "MachineInteger"} -> "b"]]];

		env["declareFunction", Complex, 
					inline@Typed[TypeForAll[ {"a", "b"}, {"a", "b"} -> "Complex"[ "TypeJoin"[ "a", "b"]]]
	            	]@Function[{arg1, arg2},
	            		Module[ {
							cast1 = Compile`Cast[arg1, Compile`TypeJoin[arg1, arg2]],
							cast2 = Compile`Cast[arg2, Compile`TypeJoin[arg1, arg2]]
						},
	            		Native`PrimitiveFunction["CreateComplex"][cast1, cast2]
	            	]]];

(*
  Later should have a constraint on the type argument,  not needed now.
*)
		env["declareFunction", Native`PrimitiveFunction["CreateComplex"], 
						llvmLinkage@TypeSpecifier[TypeForAll[{"a"}, {"a", "a"} -> "Complex"["a"]]]];

       env["declareFunction",
            N, 
      		inline@Typed[TypeForAll[{"a"}, {Element["a", "Integral"]}, {"a"} -> "Real64"]
      		]@Function[{arg}, Compile`Cast[arg, TypeSpecifier["Real64"]]]];
 
        env["declareFunction",
            N, 
      		inline@Typed[TypeForAll[{"a"}, {Element["a", "FloatingPoint"]}, {"a"} -> "a"]
      		]@Function[{arg}, arg]];
 
         env["declareFunction",
            N, 
      		inline@Typed[TypeForAll[{"a"}, {Element["a", "FloatingPoint"]}, {"Complex"["a"]} -> "Complex"["a"]]
      		]@Function[{arg}, arg]];
      		   		

		env["declareType", TypeConstructor["RealExact",  "Implements" -> {"Real"}]];
		
		addTypeJoin[ env, "RealExact", "Real64", "Real64"];
		addTypeJoin[ env, "RealExact", "Integer64", "RealExact"];
		addTypeJoin[ env, "RealExact", "Complex"["Real64"], "Complex"["Real64"]];
		addTypeJoin[ env, "Complex"["RealExact"], "Real64", "Complex"["Real64"]];
		addTypeJoin[ env, "Complex"["RealExact"], "Integer64", "Complex"["RealExact"]];
		addTypeJoin[ env, "Complex"["RealExact"], "Complex"["Real64"], "Complex"["Real64"]];


         env["declareFunction",
            Positive, 
      		inline@Typed[TypeForAll[{"a"}, {Element["a", "Real"]}, {"a"} -> "Boolean"]
      		]@Function[{arg}, 
      			arg > 0]];
      			
         env["declareFunction",
            Positive, 
      		inline@Typed[TypeForAll[{"a"},  {"Complex"["a"]} -> "Boolean"]
      		]@Function[{arg}, 
      			False]];


        env["declareFunction",
            NonPositive, 
      		inline@Typed[TypeForAll[{"a"}, {Element["a", "Real"]}, {"a"} -> "Boolean"]
      		]@Function[{arg}, 
      			arg <= 0]];
      			
         env["declareFunction",
            NonPositive, 
      		inline@Typed[TypeForAll[{"a"},  {"Complex"["a"]} -> "Boolean"]
      		]@Function[{arg}, 
      			False]];

         env["declareFunction",
            Negative, 
      		inline@Typed[TypeForAll[{"a"}, {Element["a", "Real"]}, {"a"} -> "Boolean"]
      		]@Function[{arg}, 
      			arg < 0]];
      			
         env["declareFunction",
            Negative, 
      		inline@Typed[TypeForAll[{"a"},  {"Complex"["a"]} -> "Boolean"]
      		]@Function[{arg}, 
      			False]];

        env["declareFunction",
            NonNegative, 
      		inline@Typed[TypeForAll[{"a"}, {Element["a", "Real"]}, {"a"} -> "Boolean"]
      		]@Function[{arg}, 
      			arg >= 0]];
      			
         env["declareFunction",
            NonNegative, 
      		inline@Typed[TypeForAll[{"a"},  {"Complex"["a"]} -> "Boolean"]
      		]@Function[{arg}, 
      			False]];


    ]

] (* StaticAnalysisIgnore *)



addTypeJoin[ env_, t1_, t2_, tf_] :=
	Module[ {},
		env["declareType", TypeAlias["TypeJoin"[t1, t2], tf]];
		env["declareType", TypeAlias["TypeJoin"[t2, t1], tf]];
	]

RegisterCallback["SetupTypeSystem", setupTypes]

RegisterCallback["InstantiateFunctions", createFunctions]

	createFunctions[ state_] :=
		Module[{complexList, realList, signedList, unsignedList, complexBinaryFuns, binaryFuns, binaryRealFuns, binaryIntegerFuns, unaryIntegerFuns},
			complexList = {"Complex"["Real64"]};
			realList = {"Real32", "Real64"};
			signedList = { "Integer8", "Integer16", "Integer32", "Integer64"};
			unsignedList = {"UnsignedInteger8", "UnsignedInteger16", "UnsignedInteger32","UnsignedInteger64"};
			binaryFuns = {Plus, Times, Power, Subtract, SameQ, UnsameQ,  Less, LessEqual, Greater, GreaterEqual};
			complexBinaryFuns = {Plus, Times, Power, Subtract};
			binaryRealFuns = {};
			binaryIntegerFuns = {Quotient, Mod, BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight};
			unaryIntegerFuns = {};
			
			Print["Instantiate Numeric"];
			
			Scan[
				Function[ {fun},
					Print[ fun];
					Scan[
						Function[ {ty},
              Print[fun, " - ", ty];
							state["create"][ Function[ {Typed[ arg1,ty], Typed[ arg2,ty]}, fun[arg1, arg2]]]],
						Join[ realList, signedList, unsignedList]]],
						binaryFuns
				];

			Scan[
				Function[ {fun},
					Print[ fun];
					Scan[
						Function[ {ty},
              Print[fun, " - ", ty];
							state["create"][ Function[ {Typed[ arg1,ty], Typed[ arg2,ty]}, fun[arg1, arg2]]]],
						complexList]],
						complexBinaryFuns
				];

			Scan[
				Function[ {fun},
					Print[ fun];
					Scan[
						Function[ {ty},
              Print[fun, " - ", ty];
							state["create"][ Function[ {Typed[ arg1,ty], Typed[ arg2,ty]}, fun[arg1, arg2]]]],
						Join[ signedList, unsignedList]]],
						binaryIntegerFuns
				];

			Scan[
				Function[ {fun},
					Print[ fun];
					Scan[
						Function[ {ty},
              Print[fun, " - ", ty];
							state["create"][ Function[ {Typed[ arg1,ty]}, fun[arg1]]]],
						Join[ signedList, unsignedList]]],
						unaryIntegerFuns
				];

		]




End[]

EndPackage[]
