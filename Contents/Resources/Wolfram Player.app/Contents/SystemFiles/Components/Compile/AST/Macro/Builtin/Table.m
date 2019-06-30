BeginPackage["Compile`AST`Macro`Builtin`Table`"]

Begin["`Internal`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)



(*
  As with Do,  it might be better to use a function for the body, 
  but this is hard because we might want to assign to variables that come 
  from outside the variable and the closure support is still too weak. 
  The ii is given as a local var to the Module so it is registered.
  Also,  the local variable `Internali can collide with the body, 
  a closure implementation would avoid this.
*)
setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		RegisterMacro[env, Table,
			Table[body_, end:Except[_List]] ->
				(
				Compile`AssertType[end, "MachineInteger"];
				Table[body, {end}]
				),
			Table[body_, {end_}] ->
				Table[body, {ii, 1, end, 1}],
				
			Table[body_, {ii_, end_}] ->
				Table[body, {ii, 1, end, 1}],
				
            Table[body_, {ii_, start_, end_}] -> 
                Table[body, {ii, start, end, 1}],
                
            Table[body_, {ii_, start_, end_, inc_}] -> 
                Native`UncheckedBlock@
                Module[ {ii, `Internal`start,`Internal`start1, `Internal`end, `Internal`end1, `Internal`inc, `Internal`inc1, `Internal`strmarry, `Internal`len, `Internal`res, `Internal`idx, pos},
                    `Internal`start1 = Native`CheckedBlockRestore[start];
                    `Internal`inc1 = Native`CheckedBlockRestore[inc];
                   	`Internal`start = Compile`Cast[`Internal`start1, Compile`TypeJoin[`Internal`start1, `Internal`inc1]];
					`Internal`inc = Compile`Cast[`Internal`inc1, Compile`TypeJoin[`Internal`start1, `Internal`inc1]];
                    `Internal`end1 = Native`CheckedBlockRestore[end];
                    `Internal`end = Native`IteratorConvertEnd[ `Internal`end1, `Internal`start];
                    `Internal`len = Native`IteratorCount[`Internal`start, `Internal`end, `Internal`inc];
                    ii = Native`IteratorValue[`Internal`start, `Internal`inc, 0];   
                    `Internal`res = Native`CheckedBlockRestore[body]; 
                    `Internal`strmarry = Native`CreatePackedArray[`Internal`len, `Internal`res]; 
                    Native`SetArrayElementUnary[`Internal`strmarry, 0, `Internal`res];   
                    `Internal`idx = 1;       
                    While[ `Internal`idx < `Internal`len,
                    	pos = `Internal`idx;
                        ii = Native`IteratorValue[`Internal`start, `Internal`inc, `Internal`idx]; 
                        `Internal`res = Native`CheckedBlockRestore[body];              
                        Native`SetArrayElementUnary[`Internal`strmarry, pos, `Internal`res];
                        `Internal`idx = `Internal`idx + 1;
                    ];
                    `Internal`strmarry
                ],
                                
			Table[body_, iter1_, iters__] ->
                Compile`Internal`MacroEvaluate[ makeMultiTable[body, iter1, {iters}]]
		
		];

		RegisterMacro[env, Compile`MultiTable,
			Compile`MultiTable[body_, iters__] ->
				Compile`Internal`MacroEvaluate[ makeMultiTable[body, {iters}]]
				
		];


		
	]

(*

Implementation of code to generate multi-iterator Table.
An implementation for two iterators is shown,  this is what the program
below generates.

Table[body_, {ii1_, startIn1_, endIn1_, incIn1_}, {ii2_, startIn2_, endIn2_, incIn2_}] -> 
    Native`UncheckedBlock@
    Module[ {
    		ii1, startE1, incE1, start1, inc1, end1, len1, idx1, 
   			ii2, startE2, incE2, start2, inc2, end2, len2, idx2, 
    		idxInner = 1, rank, dimsArray, array, res, pos, lenres},
    	
    	rank = 2;
    	dimsArray = Native`StackArray[rank];
    	
        startE1 = Native`CheckedBlockRestore[startIn1];
        end1 = Native`CheckedBlockRestore[endIn1];
        incE1 = Native`CheckedBlockRestore[incIn1];
        start1 = Compile`Cast[startE1, Compile`TypeJoin[startE1, incE1]];
        inc1 = Compile`Cast[incE1, Compile`TypeJoin[startE1, incE1]];
        end1 = Native`IteratorConvertEnd[ end1, start1];
        len1 = Native`IteratorCount[start1,  end1, inc1];
        ii1 = Native`IteratorValue[ start1, inc1, 0];
        Native`SetElement[dimsArray, 0, len1];
        
        startE2 = Native`CheckedBlockRestore[startIn2];
        end2 = Native`CheckedBlockRestore[endIn2];
        incE2 = Native`CheckedBlockRestore[incIn2];
        start2 = Compile`Cast[startE2, Compile`TypeJoin[startE2, incE2]];
        inc2 = Compile`Cast[incE2, Compile`TypeJoin[startE2, incE2]];                   
        end2 = Native`IteratorConvertEnd[ end2, start2];
        len2 = Native`IteratorCount[start2,  end2, inc2];
        ii2 = Native`IteratorValue[ start2, inc2, 0];
        Native`SetElement[dimsArray, 1, len2];

        res = Native`CheckedBlockRestore[body];
        array = Native`CreatePackedArray[2, dimsArray, Typed[2,TypeSpecifier[ 2]], res];
        lenres = Native`ArrayNumberOfElements[res];
        Native`SetArrayElementNary[array, lenres, 0, res];
        
        pos = 1;
        idx1 = 0;
        While[ idx1 < len1,
        	ii1 = Native`IteratorValue[ start1, inc1, idx1];
        	idx2 = idxInner;
        	idxInner = 0;
        	While[ idx2 < len2,
            	ii2 = Native`IteratorValue[start2, inc2, idx2];
            	res = Native`CheckedBlockRestore[body];
            	Native`SetArrayElementNary[array, lenres, pos, res];
            	pos++;
            	idx2++];
             idx1++;
        ];
        array
    ]
  
*)



(*
  Process each iterator,  this corrects for correct number of arguments.
  It does the following:
   corrects for correct number of arguments in the iterator
   evaluate the arguments 
   casts start and increment to a unified type.
   computes the number of iterations
   assigns the initial value to the iterator variable (needed for the first eval)
   stores the length in the dimensions array
   
   The result is all of the variables created along with the evaluations needed to init them.
  
*)

makeIter[ iterIn_, dimsArray_, indexNum_] :=
	Module[{iter = iterIn, evals = {}, sym, 
			start, end, inc, startE, incE, startSym, endSym, incSym, lenSym, index, indexHold},
		If[ !iter["hasHead", List],
			evals = Append[evals, buildMExpr[ Compile`AssertType, {iter, buildLiteral["MachineInteger"]}]];
			iter = buildMExpr[List, iter]];
		iter = iter["arguments"];
		
		If[ Length[iter] === 1,
			iter = Insert[iter, buildSymbol["iter"], 1]];
		If[ Length[iter] === 2,
			iter = Insert[ iter, buildLiteral[1], 2]];
		If[ Length[iter] === 3,
			iter = Insert[ iter, buildLiteral[1], 4]];
		If[ Length[iter] =!= 4,
			ThrowException[CompilerException["Table iterator has wrong number of arguments, between 1 and 4 are expected.", iterIn]]];
		{sym, start, end, inc} = iter;
		If[!sym["symbolQ"],
			ThrowException[CompilerException["Iterator must be a symbol.", iterIn]]];
		{evals, index} = addAssignCreateSymbolEvalsAdd[ evals, "index", buildLiteral[1]];		
		{evals, indexHold} = addAssignCreateSymbolEvalsAdd[ evals, "indexHold", buildLiteral[1]];		
		{evals, startE} = addAssignCreateSymbolEvalsAdd[ evals, "startE", buildMExpr[Native`CheckedBlockRestore, start]];		
		{evals, endSym} = addAssignCreateSymbolEvalsAdd[ evals, "endSym", buildMExpr[Native`CheckedBlockRestore, end]];
		{evals, incE} = addAssignCreateSymbolEvalsAdd[ evals, "incE", buildMExpr[Native`CheckedBlockRestore, inc]];

		{evals, startSym} = addAssignCreateSymbolEvalsAdd[ evals, "startSym", 
						buildMExpr[Compile`Cast, {startE, buildMExpr[Compile`TypeJoin, {startE, incE}]}]];		
		{evals, incSym} = addAssignCreateSymbolEvalsAdd[ evals, "incSym", 
						buildMExpr[Compile`Cast, {incE, buildMExpr[Compile`TypeJoin, {startE, incE}]}]];		
		evals = addAssignSymbolEvalsAdd[ evals, endSym, 
						buildMExpr[Native`IteratorConvertEnd, {endSym, startSym}]];		
		{evals, lenSym} = addAssignCreateSymbolEvalsAdd[ evals, "lenSym", 
						buildMExpr[Native`IteratorCount, {startSym, endSym, incSym}]];		
		evals = addAssignSymbolEvalsAdd[ evals, sym, 
						buildMExpr[Native`IteratorValue, {startSym, incSym, buildLiteral[0]}]];		
		evals = Append[evals, 
						buildMExpr[Native`SetElement, {dimsArray, buildLiteral[indexNum], lenSym}]];
		<|
		"extraSyms" -> {startE, incE},
		"start" -> startSym, "end" -> endSym, "inc" -> incSym, "len" -> lenSym, 
			"indexHold" -> indexHold, "index" -> index, "symbol" -> sym, "evaluations" -> evals
		|>
	]

(*
   Main function called for Table with 2 or more iters.
   creates the dimensions array
   creates the position pointer
   creates the iterator data
   evals the body for the initial value
   creates the packed array result
   creates the length of the result
   sets the result array with the body eval
   creates while loops for the iterators
   returns the packed array result
   creates a big module with everything in it

*)

makeMultiTable[ body_, iter1_, itersIn_] :=
	Module[{
			evals = {},
			iters = Prepend[itersIn["arguments"], iter1],
			dimsArray, numIters, iterData, pos, res, array, lenres,
			globalData, evalRes, modVars
		},
		numIters = Length[iters];
		{evals, dimsArray} = addAssignCreateSymbolEvalsAdd[ evals, "dimsArray", 
						buildMExpr[Native`StackArray, buildLiteral[numIters]]];		
		{evals, pos} = addAssignCreateSymbolEvalsAdd[ evals, "pos", buildLiteral[0]];
		
		iterData = MapIndexed[ makeIter[#1, dimsArray, First[#2]-1]&, iters];
		evals = Join[evals, Flatten[ Map[#["evaluations"]&, iterData]]];
		
		{evals, res} = addAssignCreateSymbolEvalsAdd[ evals, "res", 
							buildMExpr[Native`CheckedBlockRestore, body]];
		{evals, array} = addAssignCreateSymbolEvalsAdd[ evals, "array", 
							buildMExpr[Native`CreatePackedArray, {buildLiteral[numIters], dimsArray, makeTyped[numIters], res}]];
		{evals, lenres} = addAssignCreateSymbolEvalsAdd[ evals, "lenres", 
							buildMExpr[Native`ArrayNumberOfElements, res]];
		evals = Append[evals, buildMExpr[Native`SetArrayElementNary, {array, lenres, buildLiteral[0], res}]];
		evals = addAssignSymbolEvalsAdd[ evals, pos, buildLiteral[1]];

		globalData = <|"body" -> body, "array" -> array, "lenres" -> lenres, "pos" -> pos, "res" -> res|>;
		
		evalRes = Fold[ addEvalStage[ globalData, #1, #2]&, Null, Reverse[iterData]];
		evals = Join[ evals, evalRes];
		evals = Append[evals, array];
		modVars = Map[{#["symbol"],#["start"], #["end"], #["inc"], #["start"], #["len"], #["indexHold"], #["index"], #["extraSyms"]}&, iterData];
		modVars = Join[ modVars, {dimsArray, array, lenres, pos, res}];
		buildMExpr[ Module, {buildMExpr[ List, Flatten[modVars]], buildMExpr[CompoundExpression, evals]}]
	]


(*
  creates the evaluation stage for all except the inner iterator
  
  index = 0;
  While[ index < len,
    sym = Native`IteratorValue[ start, inc, index];
    innerEval[];
    ++index;
  ];

*)
addEvalStage[ globalData_, evalsInner_List, iterData_] :=
	Module[{evals, cond, body, while},
		
		evals = addAssignSymbolEvalsAdd[ {}, iterData["symbol"], 
						buildMExpr[Native`IteratorValue, {iterData["start"], iterData["inc"], iterData["index"]}]];
		evals = Join[evals, evalsInner];
		evals = Append[evals, buildMExpr[PreIncrement, {iterData["index"]}]];
						
		body = buildMExpr[CompoundExpression, evals];
		cond = buildMExpr[ Less, {iterData["index"], iterData["len"]}];
		while = buildMExpr[ While, {cond, body}];
		evals = addAssignSymbolEvalsAdd[ {}, iterData["index"], buildLiteral[0]];
		evals = Append[evals, while];
		evals				
	]

(*
  creates the evaluation stage for the inner iterator
  
 	index = inner;
	inner = 0;
	While[ index < len,
    	sym = Native`IteratorValue[start, inc, index];
    	res = Native`CheckedBlockRestore[body];
    	Native`SetArrayElementNary[array, lenres, pos, res];
    	++pos++;
    	++index];

*)
addEvalStage[ globalData_, Null, iterData_] :=
	Module[{evals, cond, body, while},
		evals = addAssignSymbolEvalsAdd[ {}, iterData["symbol"], 
						buildMExpr[Native`IteratorValue, {iterData["start"], iterData["inc"], iterData["index"]}]];
		evals = addAssignSymbolEvalsAdd[ evals, globalData["res"], 
						buildMExpr[Native`CheckedBlockRestore, {globalData["body"]}]];
		evals = Append[evals, 
						buildMExpr[Native`SetArrayElementNary, {globalData["array"], globalData["lenres"], 
								globalData["pos"], globalData["res"]}]];
		evals = Append[evals, 
						buildMExpr[PreIncrement, {globalData["pos"]}]];
		evals = Append[evals, 
						buildMExpr[PreIncrement, {iterData["index"]}]];
		body = buildMExpr[CompoundExpression, evals];
		cond = buildMExpr[ Less, {iterData["index"], iterData["len"]}];
		while = buildMExpr[ While, {cond, body}];
		evals = addAssignSymbolEvalsAdd[ {}, iterData["index"], iterData["indexHold"]];
		evals = addAssignSymbolEvalsAdd[ evals, iterData["indexHold"], buildLiteral[0]];
		evals = Append[evals, while];
		evals				
	]
	
	
(*
  Create the argument for the TypeLiteral to give the rank of the components outside the
  element (ie the number of iterators).  This is Typed[ num, TypeSpecifier[num]]
*)
makeTyped[num_] :=
	buildMExpr[Typed, {buildLiteral[num], buildMExpr[TypeSpecifier, buildLiteral[num]]}]



(*
  Various utility functions for collecting evaluations and making MExprs
*)

addAssignSymbolEvalsAdd[ evals_, sym_, value_] :=
	With[ {assign = buildMExpr[Set, {sym, value}]},
		Append[evals, assign]
	]

addAssignCreateSymbolEvalsAdd[ evals_, name_, value_] :=
	With[ {sym = buildSymbol[name]},
	With[ {assign = buildMExpr[Set, {sym, value}]},
		{Append[evals, assign], sym}
	]]


buildLiteral[ x_] :=
	CreateMExprLiteral[x]
	
buildSymbol[ base_] :=
	With[{name = Unique[base]},
		CreateMExprSymbol[name]
	]
	
buildMExpr[ h_, args_List] :=
	CreateMExpr[h, args]

buildMExpr[ h_, arg_] :=
	buildMExpr[h, {arg}]


RegisterCallback["SetupMacros", setupMacros]

End[]
EndPackage[]

