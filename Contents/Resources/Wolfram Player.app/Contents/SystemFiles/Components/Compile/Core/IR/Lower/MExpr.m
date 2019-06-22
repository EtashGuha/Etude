BeginPackage["Compile`Core`IR`Lower`MExpr`"]


MExprLower
MExprLowerPass

Begin["`Private`"]

Needs["CompileAST`Export`FromMExpr`"]
Needs["Compile`Core`IR`Lower`Utilities`LoweringState`"]
Needs["Compile`Core`PassManager`CompiledProgramPass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Debug`Logger`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`"]
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`Language`DesugarType`"]
Needs["Compile`Core`IR`TypeDeclaration`"]
Needs["Compile`Core`IR`FunctionDeclaration`"]


ClearAll[MExprLower]
MExprLower[cp_, opts:OptionsPattern[]] :=
	MExprLower[cp, <| opts |>]
	
MExprLower[cp_, opts_?AssociationQ] :=
	With[{st = CreateLoweringState[cp["getMExpr"], opts]},
		MExprLower[st, cp, opts]
	]

MExprLower[st_?LoweringStateQ, cp_, opts:OptionsPattern[]] :=
	MExprLower[st, cp, <| opts |>]
	
MExprLower[st_?LoweringStateQ, cp_, opts_?AssociationQ] :=
	Module[{pm, tys, funDecls, tyEnv = st["typeEnvironment"], rawData},
		tyEnv["reopen"];
		tys = cp["typeDataList"];
		Scan[
			Module[{ty, tyDecl},
				ty = DesugarType[#];
				tyEnv["declareType", #];
				tyDecl = CreateTypeDeclaration[ty];
				pm["typeDeclarations"]["appendTo", tyDecl];
			]&,
			tys
		];
		funDecls = cp["functionDeclarationList"];
		Scan[
			Module[{name, arg, funDecl},
				name = Part[#, 1];
				arg = Part[#, 2];
				tyEnv["declareFunction", name, arg];
				funDecl = CreateFunctionDeclaration[name, arg];
				pm["functionDeclarations"]["appendTo", funDecl];
			]&,
			funDecls
		];
		tyEnv["finalize"];
		lowerTopCompiledProgram[st, cp, opts];
		If[st["getProperty", "needsTypeEnvironmentFinalize", False],
			tyEnv["finalize"]
		];
		pm = st["builder"]["getProgramModule"];
		st["dispose"];
		pm["setEntryFunctionName", st["entryFunctionName"]];
		rawData = cp["rawDataList"];
		Scan[
			pm["externalDeclarations"]["addRawData",#]&,
			rawData
		];
		pm
	]



lowerTopCompiledProgram[ state_, cp_, opts_] :=
	Module[ {funs, ind = 0, entryFunctionName},
		funs = cp["functionDataList"];
		Scan[
			Function[{fun},
				Module[ {meta, name},
					meta = Part[fun,1];
					name = meta["getData", "Name", Null];
					If[ name =!= Null,
						name = state["processFunctionName", name];
						state["addFunction", name]];
				]
			], 
			funs
			];
		
		Scan[
			Function[{fun0},
				Module[{fun = fun0, meta, ty, name, exported},
					meta = Part[fun,1];
					ty = Part[fun,2];
					fun = Part[fun,3];
					name = meta["getData", "Name", Null];
					If[ name === Null,
						name = If[ ind === 0, "Main", entryFunctionName <> "_Auxiliary" <> ToString[ind]]];
					If[ SymbolQ[name],
						name = state["processFunctionName", name]];
					If[ ind === 0,
							entryFunctionName = name;
							state["setEntryFunctionName", entryFunctionName]];
					(*
					  Maybe an error here if the name is not a string?
					*)
					state["setMainName", name];
					state["builder"]["setCurrentFunctionModuleBuilder", Undefined];
					lowerTop[state, fun, opts];
					If[ ty =!= Undefined,
						ty = TypeSpecifier[ty];
						state["builder"]["currentFunctionModuleBuilder"]["setType", ty]];
					exported = meta["getData", "Exported", Null];
					If[exported === Null,
						exported = If[ ind === 0, True, False],
						exported = TrueQ[exported]];
					state["builder"]["currentFunctionModuleBuilder"]["setProperty", "exported" -> exported];
					state["builder"]["currentFunctionModuleBuilder"]["setMetaData", meta];
					ind++;
				]
			],
			funs
		];
	]



lowerTop[state_, mexpr_, opts_] :=
	Module[{hd, res, builder, lastInst},
		builder = state["builder"];
		hd = mexpr["getHead"];
		res = Switch[hd,
			Function,
				state["lower", mexpr, opts],
			Typed,
				lowerTopTyped[state, mexpr, opts],
			_,
				builder["createFunctionModule", "Main", mexpr];
				res = state["lower", mexpr, opts];
				lastInst = builder["currentFunctionModuleBuilder"]["lastInstruction"];
				If[!TrueQ[lastInst["isA", "ReturnInstruction"]],
					res = builder["createReturnInstruction", res, res["mexpr"]];
					res = res["value"];
				];
				builder["currentFunctionModuleBuilder"]["setResult", res];
				res
		];
		res		
	]


lowerTopTyped[ state_, mexpr_, opts_] :=
	Module[ {arg, ty, trgt},
		If[ mexpr["length"] =!= 2,
			ThrowException[LanguageException[{"Typed is expected to have 2 arguments ", mexpr["toString"]}]]
		];
		arg = mexpr["part", 1];
		ty =  mexpr["part", 2];
		ty = ReleaseHold[ FromMExpr[ty]];
		ty = wrapType[ty];
		trgt = state["lower", arg, opts];
		state["builder"]["currentFunctionModuleBuilder"]["setType", ty];
		trgt
	]

wrapType[Type[t_]] := TypeSpecifier[t]
wrapType[t_TypeSpecifier] := t
wrapType[t_] := TypeSpecifier[t]

RegisterCallback["RegisterPass", Function[{st},
logger = CreateLogger["MExpr", "ERROR"];

info = CreatePassInformation[
	"MExprLower",
	"The actual step of lowering the MExpr AST into Wolfram IR."
];

MExprLowerPass = CreateCompiledProgramPass[<|
	"information" -> info,
	"runPass" -> MExprLower
|>];

RegisterPass[MExprLowerPass]
]]

	
End[]

EndPackage[]
