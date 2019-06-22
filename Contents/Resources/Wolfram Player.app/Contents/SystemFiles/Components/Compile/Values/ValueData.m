BeginPackage["Compile`Values`ValueData`"]

ValueData
CreateValueData

Begin["`Private`"]

Needs["CompileAST`PatternMatching`Matcher`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`Export`FromMExpr`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Callback`"]



lhsTemplate ="Left-hand side of DownValues, `LHS`, does not have the expected form."

processLHS[self_] :=
	Module[ {lhs, argTys, funTy, defRule, proms, resTy},
		lhs = CreateMExpr @@ {self["lhs"]};
		If[ lhs["normalQ"] && lhs["hasHead", HoldPattern] && lhs["length"] === 1,
			lhs = lhs["part", 1]];
		If[ !lhs["normalQ"],
			self["setError", True];
			Return[]];
		self["processArguments", lhs["arguments"]];
		If[ self["hasError"],
			Return[Null]];
		resTy = First[ self["arguments"]]["type"];
		self["setResultType", resTy];
		argTys = Map[ #["type"]&, self["arguments"]];	
		funTy = TypeSpecifier[argTys -> self["resultType"]];
		self["setType", funTy];
		defRule = With[ {args = argTys, fun = self["symbol"], res = self["resultType"], linkName = self["linkName"]},
			fun :> Primitive`ExternalFunction[ "LocalLink", linkName, args -> res]];
		self["setDefinition", defRule];
		proms = MapIndexed[If[#1 === "Expr", {First[#2], TypeSpecifier["Expr"]}, Null]&, argTys];
		proms = DeleteCases[proms, Null];
		self["setPromotion", self["symbol"] -> TypePromotionData["CoerceArguments", proms]];
	]


fixValues[self_] :=
	Module[ {lhs, vars, rule, fun},
		lhs = self["lhs"];
		vars = Map[ First[#["name"]["data"]]&, self["arguments"]];
		rule = With[ {rhs = Apply[ fun, vars]},
			lhs :> rhs];
		rule[[2,0]] = self["function"];
		rule
	]

createFunction[self_] :=
	Module[ {vars, fun},
		vars = Map[getVariable, self["arguments"]];
		vars = CreateMExprNormal[ List, vars];
		fun = CreateMExprNormal[ Function, {vars, self["body"]}];
		self["setFunction", fun];
	]
	
getVariable[ data_] :=
	Module[ {type},
		type = Lookup[data, "type", "Expr"];
		type = CreateMExpr @@ {type};
		CreateMExprNormal[Typed, {data["name"], type}]
	]	


processArguments[self_, args_] :=
	Module[ {opers},
		self["setLength", Length[args]];
		opers = MapIndexed[ processArgument[self, #1, #2]&, args];
		self["setArguments", opers];
	]

processArgument[ self_, arg_, {ind_}] :=
	Module[ {ef, symbol = Null, type = "Expr"},
		ef = MExprMatchGetBindings[arg, self["namedBlankPatt"]];
		If[ Length[ef] === 1,
			symbol = First[ ef],
			ef = MExprMatchGetBindings[arg, self["namedBlankPatt1"]];
			If[ Length[ef] === 1,
				symbol = First[ ef];
				type = arg["part",2];
				type = ReleaseHold[FromMExpr[type]];
				];
			];
		If[symbol === Null || !symbol["symbolQ"],
			self["errorList"]["appendTo",
					Failure["CompileValues",
						<|"MessageParameters" -> <|"LHS" -> self["lhs"]|>,  "MessageTemplate" -> lhsTemplate|>]];
			Null,
			<|"index" -> ind, "name" -> symbol, "type" -> type|>
		]
	]
	
RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[
	ValueData,
	<|
		"initialize" -> (initialize[Self, ##]&),
		"processLHS" -> (processLHS[Self,##]&),
		"processArguments" -> (processArguments[Self,##]&),
		"createFunction" -> (createFunction[Self,##]&),
		"fixValues" -> (fixValues[Self,##]&),
		"hasError" -> Function[{}, Self["errorList"]["length"] > 0]
	|>,
	{
		"namedBlankPatt",
		"namedBlankPatt1",
		"length",
		"arguments",
		"resultType",
		"type",
		"lhs",
		"body",
		"symbol",
		"linkName",
		"function",
		"definition",
		"promotion",
		"errorList" 			
	}
]		
]]


CreateValueData[ lhs_ :> rhs_, sym_, linkName_] :=
	CreateObject[ValueData, 
		<|
			"lhs" -> lhs, 
			"body" -> CreateMExpr[rhs], 
		  	"symbol" -> sym,
		  	"linkName" -> linkName,
		  	"errorList" -> CreateReference[{}]|>];

initialize[self_] :=
	Module[ {},
		self["setNamedBlankPatt", CreateMExpr[Verbatim[Pattern][x_, Verbatim[_]]]];
		self["setNamedBlankPatt1", CreateMExpr[Typed[Verbatim[Pattern][x_, Verbatim[_]], _]]];
	]


	
	
	
	


End[]


EndPackage[]
