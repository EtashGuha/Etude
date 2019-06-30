BeginPackage["TypeFramework`Basic`Types`"]

InitializeTypes
ClearTypes

Begin["`Private`"]

Needs["TypeFramework`"]
Needs["TypeFramework`Utilities`TypeEnvironmentUtilities`"]

$TypesInitialized

If[!System`ValueQ[$TypesInitialized],
    $TypesInitialized = False
]

ClearTypes[] := (
    ClearTypeEnvironment[];
    $TypesInitialized = False;
);


If[!TrueQ[$TypesInitialized],
    ClearTypes[]
]


Options[InitializeTypes] = {"Finalize" -> True}

InitializeTypes[opts:OptionsPattern[]] :=
(
	If[ !$TypesInitialized,
		InitializeTypeEnvironment[];
		$TypeEnvironment["declareType", TypeConstructor["Error"]];
		$TypeEnvironment["declareType", TypeConstructor["Empty", "Bottom" -> True]];
		$TypeEnvironment["declareType", TypeConstructor["Anything", "Top" -> True]];
		$TypeEnvironment["declareType", TypeConstructor["Integer"]];
		$TypeEnvironment["declareType", TypeConstructor["Real"]];
		$TypeEnvironment["declareType", TypeConstructor["Boolean"]];

		$TypeEnvironment["declareType", TypeConstructor["Uninitialized"]];
		$TypeEnvironment["declareFunction", Compile`Uninitialized, TypeSpecifier["Uninitialized"]];

		$TypeEnvironment["declareFunction", Floor, TypeSpecifier[{"Integer"} -> "Integer"]];
		$TypeEnvironment["declareFunction", Floor, TypeSpecifier[{"Real"} -> "Integer"]];

		$TypeEnvironment["declareFunction", Plus, TypeSpecifier[{"Integer", "Integer"} -> "Integer"]];
		$TypeEnvironment["declareFunction", Plus, TypeSpecifier[{"Real", "Real"} -> "Real"]];

		$TypeEnvironment["declareFunction", Greater, TypeSpecifier[{"Integer", "Integer"} -> "Boolean"]];
		$TypeEnvironment["declareFunction", Greater, TypeSpecifier[{"Real", "Real"} -> "Boolean"]];

		$TypeEnvironment["declareFunction", Less, TypeSpecifier[{"Integer", "Integer"} -> "Boolean"]];
		$TypeEnvironment["declareFunction", Less, TypeSpecifier[{"Real", "Real"} -> "Boolean"]];

		$TypeEnvironment["declareType", TypeConstructor["Handle", {"*"} -> "*"]];

		$TypeEnvironment["declareFunction", Identity, TypeSpecifier[TypeForAll[ {"a"}, {"a"} -> "a"]]];

		$TypeEnvironment["setLiteralProcessor", Function[{tyEnv, arg},
				Which[
					IntegerQ[arg], "Integer",
					arg === True, "Boolean",
					arg === False, "Boolean",
					True, Null
				]
		]];
		If[ OptionValue["Finalize"],
			$TypeEnvironment["finalize"]
	    ];
		$TypesInitialized = True
	];
)

InitializeTypes["AbstractBasic", opts:OptionsPattern[]] :=
	If[ !$TypesInitialized,
		InitializeTypeEnvironment[];
		$TypeEnvironment["declareType", TypeConstructor["Error"]];
		$TypeEnvironment["declareType", TypeConstructor["Empty", "Bottom" -> True]];
		$TypeEnvironment["declareType", TypeConstructor["Anything", "Top" -> True]];
		
		$TypeEnvironment["declareType", TypeConstructor["Uninitialized"]];
		$TypeEnvironment["declareFunction", Compile`Uninitialized, TypeSpecifier["Uninitialized"]];
		

        $TypeEnvironment["declareType", TypeConstructor["Boolean", "Implements" -> "Equal"]];
		$TypeEnvironment["declareType", TypeConstructor["Integer", "Implements" -> "Ordered"]];
		$TypeEnvironment["declareType", AbstractType["Equal", {"a"},
            {
                Equal -> TypeSpecifier[{"a", "a"} -> "Boolean"]
            }
        ]];

        $TypeEnvironment["declareType", AbstractType["Ordered", {"a"},
            {
                Less -> TypeSpecifier[{"a", "a"} -> "Boolean"],
	            LessEqual -> TypeSpecifier[{"a", "a"} -> "Boolean"],
	            Greater -> TypeSpecifier[{"a", "a"} -> "Boolean"],
	            GreaterEqual -> TypeSpecifier[{"a", "a"} -> "Boolean"]
	        },
	        "Deriving" -> "Equal"
        ]];
		$TypeEnvironment["declareType", TypeConstructor["Native", {"*"} -> "*"]];
		
		$TypeEnvironment["setLiteralProcessor", Function[{tyEnv, arg},
				Which[
					IntegerQ[arg], "Integer",
					arg === True, "Boolean",
					arg === False, "Boolean",
					True, Null
				]
		]];
		
		If[ OptionValue["Finalize"],
			$TypeEnvironment["finalize"]
	    ];

		$TypesInitialized = True
	];

$integralTypes = {
    "Integer8",
    "Integer16",
    "Integer32",
    "Integer64",
    "UnsignedInteger8",
    "UnsignedInteger16",
    "UnsignedInteger32",
    "UnsignedInteger64"
};
$realFloatingTypes = {
    "Real32",
    "Real64"
};
InitializeTypes["Abstract", opts:OptionsPattern[]] :=
	If[ !$TypesInitialized,
		InitializeTypeEnvironment[];
		$TypeEnvironment["declareType", TypeConstructor["Error"]];
		$TypeEnvironment["declareType", TypeConstructor["Empty", "Bottom" -> True]];
		$TypeEnvironment["declareType", TypeConstructor["Anything", "Top" -> True]];
		
		$TypeEnvironment["declareType", TypeConstructor["Uninitialized"]];
		$TypeEnvironment["declareFunction", Compile`Uninitialized, TypeSpecifier["Uninitialized"]];

        $TypeEnvironment["declareType", TypeConstructor["Boolean", "Implements" -> "Equal"]];
		$TypeEnvironment["declareType", TypeConstructor["Integer", "Implements" -> "Integral"]];
		$TypeEnvironment["declareType", TypeConstructor["Real",    "Implements" -> "RealFloatingPoint"]];

        Do[
            $TypeEnvironment["declareType", TypeConstructor[integralType, "Implements" -> "Integral"]],
            {integralType, $integralTypes}
        ];
        Do[
            $TypeEnvironment["declareType", TypeConstructor[realFloatingType, "Implements" -> "RealFloatingPoint"]],
            {realFloatingType, $realFloatingTypes}
        ];

		$TypeEnvironment["declareType", AbstractType["Equal", {"a"},
            {
                Equal -> TypeSpecifier[{"a", "a"} -> "Boolean"],
                Unequal -> TypeSpecifier[{"a", "a"} -> "Boolean"],
                SameQ -> TypeSpecifier[{"a", "a"} -> "Boolean"],
                UnsameQ -> TypeSpecifier[{"a", "a"} -> "Boolean"]
            }
        ]];

        $TypeEnvironment["declareType", AbstractType["Ordered", {"a"},
            {
                Less -> TypeSpecifier[{"a", "a"} -> "Boolean"],
	            LessEqual -> TypeSpecifier[{"a", "a"} -> "Boolean"],
	            Greater -> TypeSpecifier[{"a", "a"} -> "Boolean"],
	            GreaterEqual -> TypeSpecifier[{"a", "a"} -> "Boolean"]
	        },
	        "Deriving" -> "Equal"
        ]];

        $TypeEnvironment["declareType", AbstractType["Number", {"a", "b"},
            {
	            Plus -> Typed[
 				   Function[{x, y},
  				      Module[{z = Compile`CreateHandle[]},
   				         Native`PrimitiveFunction["binary_plus"][z, x, y];
   						 Compile`Load[z]
   				   ]]
   				   ,
 				   {"a", "b"} -> "TypeJoin"["a", "b"]
 			    ],
	            Times -> TypeSpecifier[{"a", "a"} -> "a"],
	            Subtract -> TypeSpecifier[{"a", "a"} -> "a"],
	            Divide -> TypeSpecifier[{"a", "a"} -> "a"],
	            Abs -> TypeSpecifier[{"a"} -> "a"],
	            Power -> TypeSpecifier[{"a", "a"} -> "a"]
            },
            "Deriving" -> "Ordered",
            "Default" -> "Integer"
        ]];

        $TypeEnvironment["declareType", AbstractType["Enumerable", {"a"},
            {
	            Increment -> TypeSpecifier[{"a"} -> "a"],
	            Decrement -> TypeSpecifier[{"a"} -> "a"],
	            PreIncrement -> TypeSpecifier[{"a"} -> "a"],
	            PreDecrement -> TypeSpecifier[{"a"} -> "a"]
            }
        ]];


        $TypeEnvironment["declareType", AbstractType["Real", {"a"},
            "Deriving" -> {"Ordered", "Number"}
        ]];
        

        $TypeEnvironment["declareType", AbstractType["Integral", {"a"},
            {
                Ceiling -> TypeSpecifier[{"a"} -> "a"],
                Floor -> TypeSpecifier[{"a"} -> "a"],
                Quotient -> TypeSpecifier[{"a", "a"} -> "a"],
                Mod -> TypeSpecifier[{"a", "a"} -> "a"],
		        Mod -> TypeSpecifier[{"a", "a", "a"} -> "a"]
            },
            "Deriving" -> {"Real"}
        ]];

        $TypeEnvironment["declareType", AbstractType["RealFractional", {"a"},
            "Deriving" -> {"Real"}
        ]];

        $TypeEnvironment["declareType", AbstractType["Fractional", {"a"},
            "Deriving" -> {"Number"}
        ]];

        $TypeEnvironment["declareType", AbstractType["FloatingPoint", {"a"},
            {
                Sin -> TypeSpecifier[{"a"} -> "a"],
                Cos -> TypeSpecifier[{"a"} -> "a"],
                Tan -> TypeSpecifier[{"a"} -> "a"],
                Cot -> TypeSpecifier[{"a"} -> "a"],
                ArcSin -> TypeSpecifier[{"a"} -> "a"],
                ArcCos -> TypeSpecifier[{"a"} -> "a"],
                ArcTan -> TypeSpecifier[{"a"} -> "a"],
                ArcTan -> TypeSpecifier[{"a", "a"} -> "a"],
                ArcCot -> TypeSpecifier[{"a"} -> "a"],
                Sinh -> TypeSpecifier[{"a"} -> "a"],
                Tanh -> TypeSpecifier[{"a"} -> "a"],
                ArcSinh -> TypeSpecifier[{"a"} -> "a"],
                ArcCosh -> TypeSpecifier[{"a"} -> "a"],
                ArcCoth -> TypeSpecifier[{"a"} -> "a"],
                Sinc -> TypeSpecifier[{"a"} -> "a"],
                Exp -> TypeSpecifier[{"a"} -> "a"],
                Log -> TypeSpecifier[{"a"} -> "a"],
                Log -> TypeSpecifier[{"a", "a"} -> "a"],
                Sqrt -> TypeSpecifier[{"a"} -> "a"]
            },
            "Deriving" -> {"Fractional"}
        ]];

        
        $TypeEnvironment["declareType", AbstractType["RealFloatingPoint", {"a"},
            "Deriving" -> {"RealFractional", "Integral", "FloatingPoint"},
            "Default" -> "Real"
        ]];


        $TypeEnvironment["declareType", AbstractType["Coercible", {"a", "b"},
            {
                "coerceTo" -> TypeSpecifier[{"a"} -> "b"],
                "coerceFrom" -> TypeSpecifier[{"b"} -> "a"]
            }
        ]];

		$TypeEnvironment["declareType", TypeConstructor["Handle", {"*"} -> "*"]];
		$TypeEnvironment["declareType", TypeConstructor["PackedVector", {"*"} -> "*"]];

		$TypeEnvironment["declareFunction", Identity, TypeSpecifier[TypeForAll[ {"a"}, {"a"} -> "a"]]];

        $TypeEnvironment["declareFunction", "typeOf", TypeSpecifier[TypeForAll[ {"a"}, {"a"} -> "a"]]];

        $TypeEnvironment["declareFunction", "coerceTo", TypeSpecifier[ {"Integer", "Real"} -> "Real"]];
        $TypeEnvironment["declareFunction", "coerceTo", TypeSpecifier[ {"Real", "Integer"} -> "Real"]];
        $TypeEnvironment["declareFunction", "coerceTo", TypeSpecifier[TypeForAll[ {"a"}, {"a", "a"} -> "a"]]];

        (*
        $TypeEnvironment["declareFunction", Plus,
            TypeSpecifier[TypeForAll[
                {"a", "b"},
                {Element["a" , AbstractType["Number"]], Element["b" , AbstractType["Number"]]},
                {"a", "b"} -> TypeEvaluate[TypeApply[Plus, {TypeApply["coerceTo", {"a", "b"}], TypeApply["coerceTo", {"b", "a"}]}]]
            ]]
        ];
        *)


        (******************
         * Native
         ******************)
		$TypeEnvironment["declareType", TypeConstructor["Native", {"*"} -> "*"]];
        
        $TypeEnvironment["declareType", TypeInstance["Integral", {Element["a", "Number"]}, "Native"["a"] (*, {
                Plus -> Typed[
                   Function[{x, y},
                      Module[{z = Compile`CreateHandle[]},
                         Native`PrimitiveFunction["binary_plus"][z, x, y];
                         Compile`Load[z]
                   ]],
                   {"Native"["a"], "Native"["a"]} -> "Native"["a"]       
                ]
            }*)
        ]];
        
        $TypeEnvironment["declareFunction", Floor,   TypeSpecifier[ {"Real"}   -> "Integer"]];
        $TypeEnvironment["declareFunction", Floor,   TypeSpecifier[ {"Real32"} -> "Integer32"]];
        $TypeEnvironment["declareFunction", Floor,   TypeSpecifier[ {"Real64"} -> "Integer64"]];
        $TypeEnvironment["declareFunction", Ceiling, TypeSpecifier[ {"Real"}   -> "Integer"]];
        $TypeEnvironment["declareFunction", Ceiling, TypeSpecifier[ {"Real32"} -> "Integer32"]];
        $TypeEnvironment["declareFunction", Ceiling, TypeSpecifier[ {"Real64"} -> "Integer64"]];
        
        $TypeEnvironment["declareFunction", Floor,   TypeSpecifier[ {"Native"["Real"]}   -> "Native"["Integer"]]];
        $TypeEnvironment["declareFunction", Floor,   TypeSpecifier[ {"Native"["Real32"]} -> "Native"["Integer32"]]];
        $TypeEnvironment["declareFunction", Floor,   TypeSpecifier[ {"Native"["Real64"]} -> "Native"["Integer64"]]];
        $TypeEnvironment["declareFunction", Ceiling, TypeSpecifier[ {"Native"["Real"]}   -> "Native"["Integer"]]];
        $TypeEnvironment["declareFunction", Ceiling, TypeSpecifier[ {"Native"["Real32"]} -> "Native"["Integer32"]]];
        $TypeEnvironment["declareFunction", Ceiling, TypeSpecifier[ {"Native"["Real64"]} -> "Native"["Integer64"]]];
        
        (*
        $TypeEnvironment["declareType", TypeInstance["RealFloatingPoint", "Native"["a"],
            "Constraints" -> {
                Element["a", "RealFloatingPoint"]
            }
        ]];
        *)
        Do[
            $TypeEnvironment["declareFunction", "coerceTo", TypeSpecifier[ {"Integer", "Native"[integralType]} -> "Integer"]];
            $TypeEnvironment["declareFunction", "coerceTo", TypeSpecifier[ {"Native"[integralType], "Integer"} -> "Integer"]];
            ,
            {integralType, $integralTypes}
        ];


        (******************
         * Complex
         ******************)
        $TypeEnvironment["declareType", TypeConstructor["Complex", {"*"} -> "*"]];
        $TypeEnvironment["declareType", TypeInstance["RealFloatingPoint", {Element["a", "RealFloatingPoint"]}, "Complex"["a"]]];


        (*
         * MInteger
        **)
        
        If[$SystemWordLength === 64,
            $TypeEnvironment["declareType", TypeAlias["MInteger", "Integer64"]],
            $TypeEnvironment["declareType", TypeAlias["MInteger", "Integer32"]]
        ];
        
        (*
         * TypeSpecifier Common
        **)
        
        $TypeEnvironment["declareType", TypeConstructor["TypeJoin", {"*", "*"} -> "*"]];
        
        $TypeEnvironment["declareType", TypeAlias["TypeJoin"["Real", "Real"], "Real"]];
        $TypeEnvironment["declareType", TypeAlias["TypeJoin"["Integer", "Integer"], "Integer"]];
        $TypeEnvironment["declareType", TypeAlias["TypeJoin"["Integer", "Real"], "Real"]];
        $TypeEnvironment["declareType", TypeAlias["TypeJoin"["Real", "Integer"], "Real"]];
        
		$TypeEnvironment["declareType", TypeAlias["TypeJoin"["Native"["Real"], "Native"["Real"]], "Native"["Real"]]];
        $TypeEnvironment["declareType", TypeAlias["TypeJoin"["Native"["Integer"], "Native"["Integer"]], "Native"["Integer"]]];
        $TypeEnvironment["declareType", TypeAlias["TypeJoin"["Native"["Integer"], "Native"["Real"]], "Native"["Real"]]];
        $TypeEnvironment["declareType", TypeAlias["TypeJoin"["Native"["Real"], "Native"["Integer"]], "Native"["Real"]]];

		$TypeEnvironment["declareType", TypeAlias["TypeJoin"["Complex"["Real"], "Complex"["Real"]], "Complex"["Real"]]];
        $TypeEnvironment["declareType", TypeAlias["TypeJoin"["Complex"["Integer"], "Complex"["Integer"]], "Complex"["Integer"]]];
        $TypeEnvironment["declareType", TypeAlias["TypeJoin"["Complex"["Integer"], "Complex"["Real"]], "Complex"["Real"]]];
        $TypeEnvironment["declareType", TypeAlias["TypeJoin"["Complex"["Real"], "Complex"["Integer"]], "Complex"["Real"]]];

        (*$TypeEnvironment["declareType", TypeInstance["Number", {Element["a", "Number"], Element["b", "Number"]}, "TypeJoin"["a", "b"]]];*)
        
(*        
	This causes Plus of Complex[Real], Complex[Real] to fail.
	
		$TypeEnvironment["declareFunction", Plus,
            TypeSpecifier[TypeForAll[ {"a", "b"}, {Element["a", "Number"], Element["b", "Number"]},
                             {"a", "b"} -> "TypeJoin"["a", "b"]
            ]]
        ];
*)

		$TypeEnvironment["setLiteralProcessor", Function[{tyEnv, arg},
				Which[
					IntegerQ[arg], "Integer",
					arg === True, "Boolean",
					arg === False, "Boolean",
					True, Null
				]
		]];
		
		If[ OptionValue["Finalize"],
			$TypeEnvironment["finalize"]
	    ];

		$TypesInitialized = True
	];


End[]

EndPackage[]

