BeginPackage["Compile`Core`IR`CompiledProgram`"]


CompiledProgramQ
CreateCompiledProgram

Begin["`Private`"] 

Needs["Compile`"]
Needs["TypeFramework`"] (* for MetaData *)
Needs["CompileAST`Create`Construct`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`MetaData`"]



RegisterCallback["DeclareCompileClass", Function[{st},
CompiledProgramClass = DeclareClass[
	CompiledProgram,
	<|
		"rawDataList" -> (Self["rawData"]["get"]&), 
		"functionDataList" -> (Self["functionData"]["get"]&), 
		"typeDataList" -> (Self["typeData"]["get"]&), 
		"functionDeclarationList" -> (Self["functionDeclarationData"]["get"]&), 
		"functionList" -> (Self["functionData"]["get"][[All,3]]&), 
		"getMExpr" -> Function[{}, getMExpr[Self]], 
		"mapFunctionData" -> (mapFunctionData[Self, #]&), 
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt},
			toBoxes[Self, fmt]
		]
	|>,
	{
		"rawData",
		"typeData",
		"functionData",
		"functionDeclarationData",
		"properties"
	},
	Predicate -> CompiledProgramQ,
	Extends -> {
		ClassPropertiesTrait
	}
]
]]

(*
 TODO,  fill this out into a full MExpr rendition of the CompiledProgram.
*)
getMExpr[self_] :=
	Module[{funList = self["functionList"], propHolders, ef},
		If[Length[funList] === 0,
			funList = {CreateMExpr[Function[{}, 1]]}];
		propHolders = Fold[#1["join", #2["getProperty", "propertyHolders", CreateReference[<||>]]]&,  
									CreateReference[<||>], funList];
		ef = CreateMExprNormal[Compile`Program, funList];
		ef["setProperty", "propertyHolders" -> propHolders];
		ef
	]

addFunction[ funs_, f_Function] :=
	addFunction[funs, MetaData[<||>][Typed[Undefined][f]]]

addFunction[ funs_, Typed[f_Function, ty_]] :=
	addFunction[funs, MetaData[<||>][Typed[ty][f]]]

addFunction[ funs_, Typed[ty_][f_Function]] :=
	addFunction[funs, MetaData[<||>][Typed[ty][f]]]

addFunction[ funs_, MetaData[data_][f_Function]] :=
	addFunction[funs, MetaData[data][Typed[Undefined][f]]]

addFunction[ funs_, MetaData[data_][Typed[f_Function, ty_]]] :=
	addFunction[funs, MetaData[data][Typed[ty][f]]]

addFunction[MetaData[d_][MetaData[e_][r___]]] :=
	addFunction[MetaData[Join[d, e]][r]]

addFunction[ funs_, MetaData[data_][Typed[ty_][f_Function]]] :=
	Module[{mexpr = CreateMExpr[f], metaData = CreateMetaData[data]},
		funs["appendTo", {metaData, ty, mexpr}];
	]

addFunction[funs_, _] :=
	Null



addTypes[ tys_, DeclareType[arg_]] :=
	Module[{},
		tys["appendTo", arg];
	]
	
addTypes[funs_, _] :=
	Null



addFunctionDecls[ decls_, DeclareFunction[name_, fun_]] :=
	Module[{},
		decls["appendTo", {name, fun}];
	]
	
addFunctionDecls[funs_, _] :=
	Null


addRawData[ decls_, LLVMString[ str_]] :=
	Module[{},
		decls["appendTo", LLVMString[str]];
	]
	
addRawData[funs_, _] :=
	Null


iCreateCompiledProgram[Program[arg_List]] :=
	Module[{compProg, 
			funs = CreateReference[{}], 
			types = CreateReference[{}],
			funDecls = CreateReference[{}],
			rawData = CreateReference[{}]},
		Scan[ addFunction[funs, #]&, arg];
		Scan[ addTypes[types, #]&, arg];
		Scan[ addFunctionDecls[funDecls, #]&, arg];
		Scan[ addRawData[rawData, #]&, arg];
		compProg = CreateObject[
			CompiledProgram,
			<|
				"rawData" -> rawData,
				"typeData" -> types,
				"functionData" -> funs,
				"functionDeclarationData" -> funDecls,
				"properties" -> CreateReference[<||>]
			|>
		];
		compProg
	]
	
CreateCompiledProgram[args___] :=
	With[{
		cp = iCreateCompiledProgram[args],
		mexpr = CreateMExpr[args]
	},
		cp["setProperty", "mexpr" -> mexpr];
		cp
	]

iCreateCompiledProgram[MetaData[d_][f_Function]] :=
	iCreateCompiledProgram[Program[{MetaData[d][f]}]]

iCreateCompiledProgram[MetaData[d_][Typed[ty_][f_Function]]] :=
	iCreateCompiledProgram[Program[{MetaData[d][Typed[ty][f]]}]]

iCreateCompiledProgram[ Typed[f_Function, ty_]] :=
	iCreateCompiledProgram[Program[{Typed[ty][f]}]]

iCreateCompiledProgram[Typed[ty_][f_Function]] :=
	iCreateCompiledProgram[Program[{Typed[ty][f]}]]

iCreateCompiledProgram[MetaData[d_][Typed[f_Function, ty_]]] :=
	iCreateCompiledProgram[Program[{MetaData[d][Typed[ty][f]]}]]

iCreateCompiledProgram[MetaData[d_][MetaData[e_][r___]]] :=
	iCreateCompiledProgram[MetaData[Join[d, e]][r]]
	
iCreateCompiledProgram[f_Function] :=
	iCreateCompiledProgram[Program[{f}]]

iCreateCompiledProgram[args___] :=
    ThrowException[{"Invalid call to CreateCompiledProgram.", {args}}]


mapFunctionData[self_, fun_] :=
	Module[{data = self["functionData"]["get"], ndata},
		ndata = Map[{Part[#,1], Part[#,2], fun[Part[#,3]]}&, data];
		self["functionData"]["set", ndata]
	]


toString[obj_?CompiledProgramQ] :=
	Module[{funs},
		funs = Map[ {"    ",#["toString"],"\n"}&, obj["functionList"]];
		StringJoin[
			"CompiledProgram[\n"
			,
				funs
			,
			"]"
		]
	]
	
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)
(**************************************************)

icon := Graphics[Text[
  Style["COM\nPROG", GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];  
      
toBoxes[obj_?CompiledProgramQ, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"CompiledProgram",
		"",
  		icon,
  		Map[
  			BoxForm`SummaryItem[{"", #}]&,
  			obj["functionList"]
  		],
  		{}, 
  		fmt
  	]
 

End[]

EndPackage[]
