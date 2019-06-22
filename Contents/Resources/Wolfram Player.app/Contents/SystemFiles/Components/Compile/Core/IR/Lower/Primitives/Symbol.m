BeginPackage["Compile`Core`IR`Lower`Primitives`Symbol`"]

Begin["`Private`"]

Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileAST`Export`FromMExpr`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`"]
Needs["CompileUtilities`Callback`"]




getGlobal[ mexpr_] :=
	ReleaseHold[FromMExpr[mexpr]]

lower[state_, mexpr_, opts_] :=
	Module[{builder, inst, var},
	    builder = state["builder"];
	    
	    (*
	      If the symbol is known to the symbol builder (ie it was a formal symbol from
	      Function, Module, With, etc...   then read it.  Even things like Table/Do localize.
	      
	      If it is not found then it is a global symbol and we should add a LoadGlobalInstruction. 
	      Note that we have to do this each time the symbol is found.
	      
	      If the symbol exists, perhaps have a check that the variableValue has been set 
	      correctly by the symbolBuilder.
	    *)
	    If[builder["symbolBuilder"]["exists", mexpr["lexicalName"]],
			var = builder["symbolBuilder"]["readVariable", mexpr["lexicalName"], mexpr];
			, (* Else *)
	        var = state["createFreshVariable", mexpr];
			inst = builder["createLoadGlobalInstruction", var, CreateConstantValue[getGlobal[mexpr]], mexpr];
			var["setProperty", "Writable" -> False];
		];
		If[var =!= Undefined,
			var["setProperty", "variableValue" -> mexpr["lexicalName"]];
		];
		var
	]



(*
  TODO
    deprecate the use of Native`ConstantSymbol,  Native`ConstantExpression is more general.
*)
isString[ mexpr_] :=
	mexpr["literalQ"] && mexpr["head"]["symbolQ"] && mexpr["head"]["data"] === HoldComplete[String]

lowerGlobalConstant[state_, mexpr_, opts_] :=
	Module[{builder, sym, name, trgt, cons, ty, inst},
	    builder = state["builder"];
		If[ mexpr["length"] =!= 1,
			ThrowException[LanguageException[{"GlobalSymbol is expected to have one argument ", mexpr["toString"]}]]
		];
		sym = mexpr["part", 1];
		name = Which[
			sym["symbolQ"],
				sym["fullName"],
			isString[sym],
				sym["data"],
			True,
				ThrowException[LanguageException[{"GlobalSymbol is expected to have one symbol or string argument ", mexpr["toString"]}]]
		];
		cons = CreateConstantValue[Primitive`GlobalSymbol[name]];
		ty = TypeSpecifier["Expression"];
		cons["setType", ty];
		trgt = state["createFreshVariable", mexpr];
		inst = builder["createCopyInstruction", trgt, cons, mexpr];
	    trgt
	]


RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitiveAtom[Symbol], lower]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`ConstantSymbol], lowerGlobalConstant]
]]

End[]

EndPackage[]
