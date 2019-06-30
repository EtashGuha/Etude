BeginPackage["Compile`Values`CompileValues`"]

CompileSymbolValues
ToProgramExpr
FixDownValues

Begin["`Private`"]

Needs["Compile`Values`ValueData`"]
Needs["CompileAST`Create`Construct`"]


$storedDownValues = <||>

FixDownValues[ {sym_, fun_}] :=
	Module[ {},
		AssociateTo[ $storedDownValues, sym -> DownValues[sym]];
		sym = fun;
		sym
	]

(*
  I don't think this is used.
  TODO remove it.
*)
CompileSymbolValues[ sym_] :=
	Module[ {lhsData, newValues, failures},
		lhsData = Map[ processRule[sym, #]&, DownValues[sym]];
		failures = getFailures[sym, lhsData];
		If[ FailureQ[failures],
			Return[failures]];
		Scan[ #["createFunction"]&, lhsData];
		newValues = Map[ #["fixValues"]&, lhsData];
		DownValues[sym] = newValues;
		sym
	]
	

ToProgramExpr[ sym_, opts_:<||>] :=
	Module[ {linkDataList, progExpr, failures},
		linkDataList = MapIndexed[ processRule[sym, First[#2], #1]&, DownValues[sym]];
		failures = getFailures[sym, linkDataList];
		If[ FailureQ[failures],
			Return[failures]];
		progExpr = makeProgramExpr[linkDataList];
		progExpr
	]
	
	

processRule[ sym_, index_, rule_] :=
	Module[ {argObj, linkName},
		linkName = Context[ sym] <> SymbolName[sym] <> "$" <> ToString[index];
		argObj = CreateValueData[ rule, sym, linkName];
		argObj["processLHS"];
		If[ !argObj["hasError"],
			argObj["createFunction"]];
		argObj
	]
		
makeProgramExpr[linkDataList_] :=
	Module[ {funs, defs, proms},
		(*
		  Make funs be a MNormal MExpr,  with the form { {name, fun}, ...}
		*)
		funs = Map[ {#["linkName"], #["function"]}&, linkDataList];
		funs = Map[ CreateMExprNormal[List,#]&, funs];
		funs = With[ {arg = funs},
			CreateMExprNormal[ List, arg]];			
		defs = Map[ #["definition"]&, linkDataList];
		proms = Map[#["promotion"]&, linkDataList];
		defs = With[ {arg = defs},
			CreateMExprNormal[ Rule, {"TypeDefinitions", arg}]];
		proms = With[ {arg = proms},
			CreateMExprNormal[ Rule, {"TypePromotions", arg}]];
		funs = With[ {arg = funs},
			CreateMExprNormal[ Rule, {"Functions", arg}]];
		With[ {arg = {defs, proms, funs}},
			CreateMExprNormal[ Compile`ProgramExpr, arg]]
	]
	
	
failTemplate ="Errors found processing DownValues of symbol `symbol`."


getFailures[symb_,list_] :=
	Module[ {failures, failArg},
		failures = Flatten[ Map[ #["errorList"]["toList"]&, list]];
		If[ Length[failures] > 0,
			failArg = Association[MapIndexed[ "fail" <> ToString[First[#2]] -> #1 &, failures]];
			failArg = Join[
					<| "MessageParameters" -> <|"symbol" -> symb|>,  "MessageTemplate" -> failTemplate|>, failArg];
			Failure["CompileValues", failArg]
			,
			Null]
	]


End[]


EndPackage[]
