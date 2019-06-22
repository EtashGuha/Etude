BeginPackage["Compile`Core`IR`Lower`Primitives`DeclareType`"]


Begin["`Private`"]


Needs["TypeFramework`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["CompileAST`Export`FromMExpr`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]



fromMExpr[mexpr_] := ReleaseHold[FromMExpr[mexpr]]

lowerDeclareStructure[state_, mexpr_, opts_] :=
    Print["TODO::: lowerDeclare structure ", mexpr]
    
lowerAliasType[state_, mexpr_, opts_] :=
	Module[{ty, fst, name, typeEnv, builder, decl, varAndty},
		builder = state["builder"];
		typeEnv = state["typeEnvironment"];
		fst = mexpr["part", 1];
		If[fst["hasHead", "Type"] || fst["hasHead", "TypeSpecifier"],
			fst = fst["part", 1]
		];
        name = fromMExpr[fst];
		varAndty = state["parseTypeMarkup", mexpr["part", 2], opts];
		ty = varAndty["type"];
        If[typeEnv["getStatus"] =!= "Reopened",
        	   typeEnv["reopen"]
        ];
		typeEnv["declareType", TypeAlias[name, ty]];
		decl = builder["createTypeDeclaration", TypeAlias[ name, ty]];
		state["setProperty", "needsTypeEnvironmentFinalize" -> True];
		ty
	]	

lower[state_, mexpr_, opts_] := (
	Which[
		mexpr["part", 2]["hasHead", Association],
			lowerDeclareStructure[state, mexpr, opts],
        state["isTypeMarkup", mexpr["part", 2], opts] ||
        (mexpr["part", 2]["literalQ"] && mexpr["part", 2]["hasHead", String]),
            lowerAliasType[state, mexpr, opts],
		True,
			ThrowException[{"Invalid usage of DeclareType :: ", mexpr}]
	];
	Null
)


RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`DeclareType], lower]
]]


End[]

EndPackage[]
