BeginPackage["Compile`Core`IR`Lower`Utilities`LoweringState`"]

LoweringState
LoweringStateQ
LoweringStateClass
CreateLoweringState


Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`Lower`Builder`ProgramModuleBuilder`"]
Needs["Compile`Core`IR`Lower`Utilities`Fresh`"]
Needs["Compile`Core`IR`Lower`Utilities`TypeEnvironment`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileAST`Export`FromMExpr`"]
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["TypeFramework`"] (* For TypeVariable *)
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileAST`Class`Literal`"]
Needs["CompileAST`Class`Normal`"]
Needs["CompileAST`Class`Base`"]
Needs["CompileAST`Class`Symbol`"]





(*
	dispatchKey is a complicated name for a simple function:
	dispatchKey[ f ] returns f
	dispatchKey[ f[1] ] returns f[]
	dispatchKey[ f[1][2] ] returns f[][]
	dispatchKey[ f[1][2][3] ] returns f[][] also
	
	sym[args1][args2] uses subvalues of sym but
	sym[args1][args2][args3] also uses subvalues of sym
	
	The actual subvalue code cannot distinguish between sym[args1][args2] and sym[args1][args2][args3]

	Each one of the calls f, f[1], f[1][2], are dispatched from the same symbol,
		but they are all calls to different definitions
*)
dispatchKey[mexpr_] :=
	Which[
		MExprSymbolQ[mexpr], (* symbol case *)
			mexpr["fullName"],
		MExprNormalQ[mexpr] && MExprSymbolQ[mexpr["head"]], (* Down code case *)
			mexpr["head"]["fullName"][],
		True, (* Sub code case *)
			getSymbolHead[mexpr][][]
	]

getSymbolHead[ mexpr_?MExprSymbolQ] :=
	mexpr["fullName"]

getSymbolHead[ mexpr_] :=
	getSymbolHead[ mexpr["head"]]

(*
  If the mexpr is a symbol which is known to be an Atom (as determined by $LanguagePrimitiveLoweringRegistry), 
  then lower this directly.  Otherwise the symbol will be a function.  If we get just the symbol then it needs
  to be treated as a Global.  Eg Compile[ {}, Module[ {y}, y = Take]]  and Compile[ {}, Module[ {y}, y = True]]
  are different.
*)
dispatch[state_, mexpr_] :=
	dispatch[state, mexpr, <||>]
dispatch[state_, mexpr_, opts_] :=
	Module[{fullName, lowering, key, default, res},
		
		If[state["builder"]["currentFunctionModuleBuilder"] =!= Undefined,
			If[TrueQ[state["builder"]["currentFunctionModuleBuilder"]["returnMode"]],
				Return[Null]
			];
		];
		
		(*
		  Figure out the fullName.  This will be the name of the symbol if it is an atom.
		  If it is a normal, then look for the head.
		*)
		fullName = Which[
			MExprLiteralQ[mexpr] && !MExprSymbolQ[mexpr],
				"Literal",
			True,
				dispatchKey[mexpr]
		];
		
		lowering = Lookup[$LanguagePrimitiveLoweringRegistry, fullName];
		If[MissingQ[lowering],
			key = Which[
					MExprSymbolQ[mexpr], 
						"System`Symbol"
					,
					MExprSymbolQ[mexpr["head"]];
					(*
					 Maybe match to name method here.
					*)
					name = state["processFunctionName", mexpr["head"]];
					state["isFunction", name],
						"Compile`Internal`FunctionCall"[]
					,
					True,
						"Compile`Internal`General"[]];
			default = $LanguagePrimitiveLoweringRegistry[key];
			Assert[MissingQ[default] === False];
			lowering = default;
		];
		
		Assert[LanguagePrimitiveQ[lowering["info"]]];

		res = lowering["lower"][state, mexpr, opts];
		
		res
	]


isTypedMarkup[mexpr_, opts_] :=
	mexpr["normalQ"] &&
    With[{hd = mexpr["head"]},
        hd["symbolQ"] && 
        hd["fullName"] === "System`Typed" && 
        mexpr["length"] === 2
    ]
	
isTypedMarkup[self_, mexpr_, opts_] :=
	isTypedMarkup[mexpr, opts]

isTypedMarkup[args___] :=
	ThrowException[ {"The parameters to isTypedMarkup are not valid.", {args}}]
	

unwrapType[mexpr_] :=
	If[isTypeMarkup[mexpr],
		mexpr["part", 1],
		mexpr
	]

isMExprInteger[mexpr_] :=
	mexpr["literalQ"] &&
	mexpr["hasHead", Integer]
	

isTypeMarkup[mexpr_] :=
	mexpr["normalQ"] &&
	mexpr["head"]["symbolQ"] &&
		(mexpr["head"]["fullName"] === "System`Type" ||  mexpr["head"]["fullName"] === "System`TypeSpecifier")&&
	mexpr["length"] === 1
	
isTypeMarkup[mexpr_, opts_] :=
	isTypeMarkup[mexpr]
isTypeMarkup[self_, mexpr_, opts_] :=
	isTypeMarkup[mexpr, opts]
	
isTypeMarkup[args___] :=
	ThrowException[ {"The parameters to isTypeMarkup are not valid.", {args}}]
	
isTypeLiteralMarkup[mexpr_] :=
	mexpr["normalQ"] &&
	mexpr["head"]["symbolQ"] &&
	mexpr["head"]["fullName"] === "TypeFramework`TypeLiteral" &&
	(mexpr["length"] === 1 || mexpr["length"] === 2) 

isTypeLiteralMarkup[mexpr_, opts_] :=
	isTypeLiteralMarkup[mexpr]
	
isTypeLiteralMarkup[self_, mexpr_, opts_] :=
	isTypeLiteralMarkup[mexpr, opts]

isTypeLiteralMarkup[args___] :=
	ThrowException[ {"The parameters to isTypeLiteralMarkup are not valid.", {args}}]
	
isTypeOf[mexpr_] :=
	isTypeOf[mexpr, <||>]
isTypeOf[mexpr_, opts_] :=
	(isTypeMarkup[mexpr, opts] && isTypeOf[unwrapType[mexpr], opts]) ||
	(
		mexpr["normalQ"] && 
		mexpr["head"]["symbolQ"] &&
		mexpr["head"]["fullName"] === "Compile`TypeOf"
	)

isTypeOf[args___] :=
	ThrowException[ {"The parameters to isTypeOf are not valid.", {args}}]
		
isResultOf[mexpr_, opts_] :=
	(
		mexpr["normalQ"] && 
		mexpr["head"]["symbolQ"] &&
		mexpr["head"]["fullName"] === "Compile`ResultOf"
	)

isResultOf[args___] :=
	ThrowException[ {"The parameters to isResultOf are not valid.", {args}}]
		
isTypeOfResultOf[mexpr_] :=
	isTypeOfResultOf[mexpr, <||>]
		
isTypeOfResultOf[mexpr_, opts_] :=
	isTypeOf[mexpr, opts] &&
	isResultOf[mexpr["part", 1], opts]
	
isTypeOfResultOf[args___] :=
	ThrowException[ {"The parameters to isTypeOfResultOf are not valid.", {args}}]
	
isTypeJoin[mexpr_] :=
	mexpr["normalQ"] &&
	mexpr["head"]["symbolQ"] &&
	mexpr["head"]["fullName"] === "Compile`TypeJoin"
	

isMExprRuleLike[mexpr_] :=
	mexpr["normalQ"] &&
	mexpr["head"]["symbolQ"] &&
	mexpr["length"] === 2 &&
	mexpr["part", 1]["normalQ"] &&
	mexpr["part", 1]["isList"] &&
	(
	 mexpr["head"]["fullName"] === "System`Rule" ||
	 mexpr["head"]["fullName"] === "System`RuleDelayed"
	) 
	
isMExprString[mexpr_] :=
	mexpr["literalQ"] && mexpr["head"]["fullName"] === "System`String"
	
isMExprNormalWithStringHead[mexpr_] :=
	mexpr["normalQ"] && isMExprString[mexpr["head"]]

isTypeSpecHead[mexpr_] :=
	mexpr["normalQ"] &&
	mexpr["length"] === 1 &&
	mexpr["head"]["symbolQ"] &&
	(
	 mexpr["head"]["fullName"] === "System`Type" ||
	 mexpr["head"]["fullName"] === "System`TypeSpecifier"
	) 

isMExprNormalWithTypeSpecHead[mexpr_] :=
	mexpr["normalQ"] && isTypeSpecHead[mexpr["head"]]

ClearAll[withTypeHead]
withTypeHead[t_TypeSpecifier] := t
withTypeHead[t_] := TypeSpecifier[t]	
withTypeHead[Type[t_]] := TypeSpecifier[t]	

ClearAll[stripTypeHead]
stripTypeHead[Type[t___]] := t
stripTypeHead[TypeSpecifier[t___]] := t
stripTypeHead[t_] := t	

ClearAll[parseTypeMarkup]
parseTypeMarkup[self_, str_?StringQ, opts_] :=    
    <|
   		"type" -> withTypeHead[str]
   	|>
    
      
    
parseTypeMarkup[self_, mexpr_?isTypeLiteralMarkup, opts_] :=
	With[ {
        ty = ReleaseHold[FromMExpr[mexpr]]
    },
   		<|
   			"type" -> withTypeHead[ty]
   		|>
    ]
    
parseTypeMarkup[self_, mexpr_?isMExprInteger, opts_] :=
	With[ {
        ty = ReleaseHold[FromMExpr[mexpr]]
    },
   		<|
   			"type" -> withTypeHead[ty]
   		|>
    ]
parseTypeMarkup[self_, mexpr_?isMExprString, opts_] :=
	With[ {
        ty = ReleaseHold[FromMExpr[mexpr]]
    },
   		<|
   			"type" -> withTypeHead[ty]
   		|>
    ]
parseTypeMarkup[self_, mexpr_?isMExprNormalWithStringHead, opts_] :=
	With[ {
        hdTy = parseTypeMarkup[self, mexpr["head"], opts],
        argsTy = parseTypeMarkup[self, #, opts]& /@ mexpr["arguments"]
    },
    With[{
    	hd = stripTypeHead[Lookup[hdTy, "type"]],
    	args = stripTypeHead /@ Lookup[argsTy, "type"] 
    },
   		<|
   			"type" -> withTypeHead[Apply[hd, args]]
   		|>
    ]]

parseTypeMarkup[self_, mexpr_?isMExprNormalWithTypeSpecHead, opts_] :=
	With[ {
        hdTy = parseTypeMarkup[self, mexpr["head"]["part",1], opts],
        argsTy = parseTypeMarkup[self, #, opts]& /@ mexpr["arguments"]
    },
    With[{
    	hd = stripTypeHead[Lookup[hdTy, "type"]],
    	args = stripTypeHead /@ Lookup[argsTy, "type"] 
    },
   		<|
   			"type" -> withTypeHead[Apply[hd, args]]
   		|>
    ]]

parseTypeMarkup[self_, mexpr_?isMExprRuleLike, opts_] :=
	With[ {
        args = parseTypeMarkup[self, #, opts]& /@ mexpr["part", 1]["arguments"],
        res = parseTypeMarkup[self, mexpr["part", 2], opts]
    },
		AssertThat["None of the arguments should be have a variable (this is not currently supported)",
			Append[args, res]]["named", mexpr["toString"]]["elementsSatisfy", (MissingQ[Lookup[#, "variable"]])&
		];
   		<|
   			"type" -> withTypeHead[(stripTypeHead[Lookup[#, "type"]]& /@ args) -> stripTypeHead[Lookup[res, "type"]]]
   		|>
    ];

parseTypeMarkup[self_, mexpr_?isTypeMarkup, opts_] :=
	parseTypeMarkup[self, unwrapType[mexpr], opts];


(*
ResultOf and TypeOf actually need to lower their arguments, so 
we switch to parseTypeMarkupLower.
*)
parseTypeMarkup[self_, mexpr_?isTypeOfResultOf, opts_] :=
	parseTypeMarkupLower[self, mexpr, opts];

parseTypeMarkup[self_, mexpr_?isTypeOf, opts_] :=
	parseTypeMarkupLower[self, mexpr, opts];

parseTypeMarkup[self_, mexpr_?isTypeJoin, opts_] :=
	parseTypeMarkupLower[self, mexpr, opts];



(*
  Unknown,  just make this the type,  it'll probably be an error later.
*)    
parseTypeMarkup[self_, mexpr_, opts_] :=
   	<|
   			"type" -> withTypeHead[ stripTypeHead[mexpr]]
   	|>


(*
  parseTypeMarkupLower is called by lowering code that takes arguments that 
  need lowering but don't have a wrapper.  Eg TypeJoin[ a,b],  both a and b need to 
  be lowered,  but they 
*)  	
   		
(* We do not want to evaluate the first argument of 
 * result of because that causes an infinite recursion
 *)
parseTypeMarkupLower[self_, mexpr_?isTypeOfResultOf, opts_] := 
	Module[{trgt, resultOfExpr, inst, hd, args},
	    AssertThat["A TypeOf has only one argument.", mexpr["length"]
			]["named", mexpr
			]["isEqualTo", 1
		];
		
		resultOfExpr = mexpr["part", 1];
		
		hd = CreateConstantValue[resultOfExpr["part", 1]];
		args = self["lower", #, Append[opts, "Erasure" -> True]]& /@ Rest[resultOfExpr["arguments"]];
		
		trgt = self["createFreshVariable", resultOfExpr];
		inst = self["builder"]["createCallInstruction",
			trgt,
			CreateConstantValue[Compile`ResultOf],
			Join[{hd}, args],
			resultOfExpr
		];
		inst["setProperty", "Erasure" -> True];
		
	    If[trgt["type"] === Undefined,
	    	trgt["setType", withTypeHead[TypeVariable[ToString[trgt["id"]]]]]
	    ];
	    
   		<|
   			"variable" -> trgt,
   			"type" -> trgt["type"]
   		|>
	];
	
parseTypeMarkupLower[self_, mexpr_?isTypeOf, opts_] := 
	Module[{var},
	    AssertThat["A TypeOf has only one argument.", mexpr["length"]
			]["named", mexpr
			]["isEqualTo", 1
		];
	    var = self["lower", mexpr["part", 1]];
	    If[var["type"] === Undefined,
	    	var["setType", withTypeHead[TypeVariable[ToString[var["id"]]]]]
	    ];
   		<|
   			"variable" -> var,
   			"type" -> var["type"]
   		|>
	];

parseTypeMarkupLower[self_, mexpr_?isTypeSpecHead, opts_] :=
	parseTypeMarkup[self, mexpr, opts]


parseTypeMarkupLower[self_, mexpr_, opts_] :=
   	Module[{var},
   		var = self["lower", mexpr];
   		If[var["type"] === Undefined,
	    	var["setType", withTypeHead[TypeVariable[ToString[var["id"]]]]]
	    ];
   		<|
   			"variable" -> var,
   			"type" -> var["type"]
   		|>
   	]
     
	
parseTypedMarkup[self_, mexpr_?MExprQ] :=
	parseTypedMarkup[self, mexpr, <||>];
	
parseTypedMarkup[self_, mexpr_?MExprQ, opts_] :=
	(
	    AssertThat["Typed argument is expected to have 2 arguments", mexpr["length"]
	    		]["named", "arguments"
	    		]["isEqualTo", 2
	   ];
	   parseTypeMarkup[self, mexpr["part",2], opts]
	)
	

RegisterCallback["DeclareCompileClass", Function[{st},
LoweringStateClass = DeclareClass[
	LoweringState,
	<|
		"createFreshId" -> Function[{},
			CreateFreshId[Self]
		],
		"createFreshVariable" -> (
			CreateFreshVariable[Self, ##]&
		),
		"lower" -> (dispatch[Self, ##]&),
		"isTypeMarkup" -> (isTypeMarkup[Self,##]&),
        "isTypedMarkup" -> (isTypedMarkup[Self,##]&),
		"parseTypeMarkup" -> (parseTypeMarkup[Self, ##]&),
		"parseTypeMarkupLower" -> (parseTypeMarkupLower[Self, ##]&),
        "parseTypedMarkup" -> (parseTypedMarkup[Self, ##]&),
        "addFunction" -> (addFunction[Self, ##]&),
        "isFunction" -> (isFunction[Self, ##]&),
        "processFunctionName" -> (processFunctionName[Self, ##]&),
        "dispose" -> (dispose[Self]&)
	|>,
	{
		"id",
		"builder",
		"nextVariable",
		"loweringFunction",
		"basicBlockStates",
		"typeEnvironment",
		"returnMode" -> False,
		"mainName" -> "Main",  (* Used for each FM in the builder *)
		"entryFunctionName" -> "Main",  (* Used for one FM in the builder *)
		"functionNames",
		"properties"
	},
	Predicate -> LoweringStateQ,
	Extends -> {
		ClassPropertiesTrait
	}
]
]]


currentId = 0;
makeId[] := currentId++;

Options[CreateLoweringState] = {
}


	
CreateLoweringState[mexpr_, opts_?AssociationQ] :=
	CreateLoweringState[makeId[], mexpr, opts]


setMetaInformation[builder_, envOptions_, key_] :=
	Module[ {value = Lookup[envOptions, key, Null]},
		If[value =!= Null,
			builder["setMetaInformation", key -> value]]
	]

CreateLoweringState[id_, mexpr_, opts_?AssociationQ] :=
	Module[{builder, tenv},
		If[ TrueQ[Lookup[ opts, "ResetVariableID"]],
			Compile`Core`IR`Variable`Private`$NextVariableId = 1
		];
		tenv = CreateLoweringTypeEnvironment[opts];
		builder = CreateProgramModuleBuilder[mexpr, opts];
		builder["setTypeEnvironment", tenv];
		builder["metaInformation"]["associateTo", "TargetSystemID" -> tenv["getProperty", "TargetSystemID", $SystemID]];
		CreateObject[
			LoweringState,
			<|
				"id" -> id,
				"builder" -> builder,
				"nextVariable" -> CreateReference[1],
				"basicBlockStates" -> CreateReference[<||>],
				"typeEnvironment" -> tenv,
				"functionNames" -> CreateReference[<||>],
				"properties" -> CreateReference[<||>]
			|>
		]
	]


addFunction[self_, name_] :=
	Module[{},
		If[self["functionNames"]["keyExistsQ", name],
			ThrowException[ {"Duplicate function name found.", name}]];
		self["functionNames"]["associateTo", name -> True]
	]

isFunction[self_, name_] :=
	self["functionNames"]["keyExistsQ", name]

processFunctionName[self_, name_Symbol] :=
	StringReplace[ Context[name] <> SymbolName[name], "`" -> "_"]

processFunctionName[self_, name_?MExprSymbolQ] :=
	StringReplace[ name["fullName"], "`" -> "_"]

processFunctionName[self_, name_] :=
	name


dispose[self_] :=
	Module[{},
		self["builder"]["dispose"]
	]
	
End[] 

EndPackage[]
