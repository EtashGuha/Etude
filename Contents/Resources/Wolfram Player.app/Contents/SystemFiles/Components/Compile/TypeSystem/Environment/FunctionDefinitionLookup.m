
BeginPackage["Compile`TypeSystem`Environment`FunctionDefinitionLookup`"]

CreateCompileFunctionDefinitionLookup
CompileFunctionDefinitionLookupQ

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeLiteral`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["Compile`Utilities`Serialization`"]
Needs["Compile`Core`IR`Lower`Utilities`LoweringTools`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["Compile`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`ConstantValue`"]

(*
  specific holds non-overloaded functions,  ie each function name only has one definition
  monomorphic holds monomorphic functions ...  might be overloaded though
  polymorphic holds polymorphic functions ... might be overloaded and written with generic type variables
*)

RegisterCallback["DeclareCompileClass", Function[{st},
CompileFunctionDefinitionLookupClass = DeclareClass[
	CompileFunctionDefinitionLookup,
	<|
		"clear" -> Function[{}, clear[Self]],
		"finalizeDefinition" -> Function[{tyEnv, name, def, ty}, finalizeDefinition[Self, tyEnv, name, def, ty]],
		"finalizeAtomDefinition" -> Function[{tyEnv, name, def, ty}, finalizeAtomDefinition[Self, tyEnv, name, def, ty]],
		"process" -> Function[{pm, funName, funTy, def}, process[Self, pm, funName, funTy, def]],
		"processAtom" -> Function[{tyEnv, name}, processAtom[Self, tyEnv, name]],
		"updateCache" -> Function[{newValue}, updateCache[Self, newValue]],
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"cache"
	},
	Predicate -> CompileFunctionDefinitionLookupQ
]
]]

CreateCompileFunctionDefinitionLookup[] :=
	CreateObject[CompileFunctionDefinitionLookup, <|
		"cache" -> CreateReference[<||>]
	|>]

clear[self_] :=
	self["setCache", CreateReference[<||>]]



(*
  If there are any definitions then add this as a general lowering.
  AddLowerGeneral should not add any lowering rule if there already is one.
  
  If this is linked or redirect add to the cache,  this means that the 
  functions cannot be overloaded.  Not quite sure if that's good.
*)
finalizeDefinition[ self_, tyEnv_, name_, def_, ty_] :=
	Module[ {data, ef},
		If[Length[def] > 0,
			AddLowerGeneral[name]
		];
		data = Lookup[def, "Linkage", None];
		If[ data =!= None,
			ef = Join[ def, <| "Class" -> "Linked"|>];
			addToCacheMangle[ self, tyEnv, name, ty, ef];
			Return[ef]
		];
		data = Lookup[def, "Redirect", None];
		If[ data =!= None,
			ef = Join[ def, <| "Class" -> "Redirect"|>];
			addToCacheMangle[ self, tyEnv, name, ty, ef];
			Return[ef]
		];
		Which[ 
			Length[def] === 0,
				Null,
			KeyExistsQ[def, "Class"], 
				def,
			True,
				Join[def, <| "Class" -> "General"|>]
		]
	]





finalizeAtomDefinition[ self_, tyEnv_, name_, def_, ty_] :=
	Module[ {},
		AddLowerGeneralAtom[name];
		Null
	]



(*
   Function Processing
*)


(*
 Fix a function that has sequence arguments.  
 For now only deal with one sequence, but this can be fixed later.
 
 Generate Typed[ var, ty] for each type in funTy where var is a new variable.
 Replace calls to Compile`SequenceLength with the number of arguments.
 
*)



createSequenceArgument[ ty_] :=
	Module[ {var},
		var = Unique["arg"];
	   <| "variable" -> var, "argument" -> Typed[var, ty] |>
	]


fixSequenceIterate[ pos_ -> Compile`SequenceSlot[], arg_] :=
	Module[ {},
		pos -> arg["variable"]
	]

(*
Must be HoldAll
*)
SetAttributes[ Compile`SequenceIterate, HoldAll]

(*
  Fix SequenceIterate.
  An example:
  SequenceIterate[
      sum = sum + SequenceSlot[],
      seqName]
      
  We turn this into
  
      sum = sum + arg1;
      sum = sum + arg2;
      sum = sum + arg3;
      
  where  arg1, ...   are the arguments of the function.
      
  We wrap the body in Hold, and search for nested SequenceSlot.
  Then we create a list of replacements for the SequenceSlot for each function argument.
  Each one of these is substituted into the SequenceIterate body, making a list of new bodys.
      {Hold[ newBod1], Hold[ newBod2], ...}
  The list of these is turned into a CompoundExpression wrapped in Hold.
      Hold[ Hold[ newBod1];Hold[ newBod2]; ...]
  The inner Holds are dropped.
      Hold[ newBod1;newBod2; ...]
  A RuleDelayed is created
      pos :> Hold[ newBod1;newBod2; ...]
  The Hold is dropped
      pos :> newBod1;newBod2; ...
  This is returned.
*)
fixSequenceFeatures[  pos_ -> Compile`SequenceIterate[body_, _], args_, argLen_] :=
	Module[ {posN, elems, holdBody, newBodys, ef},
		holdBody = Hold[ body];
		posN = Position[ holdBody, Compile`SequenceSlot[]];
		elems = Map[ (# -> Extract[holdBody,#])&, posN];
		elems = Map[ Function[ {arg}, Map[ fixSequenceIterate[#, arg]&, elems]], args];
		newBodys = Map[ ReplacePart[ holdBody, #]&, elems];
		newBodys = Hold @@ {newBodys};
		newBodys = ReplacePart[ newBodys, {{1,0} -> CompoundExpression}]; 
		newBodys = Delete[ newBodys, Table[{1, i, 0}, {i, argLen}]];
		ef = Apply[ RuleDelayed, {pos, newBodys}];
		ef = Delete[ ef, {2,0}];
		ef
	]


(*
  Fix SequenceApply,  apply the function to the arguments.
*)
fixSequenceFeatures[  pos_ -> Compile`SequenceApply[fun_, _], args_, argLen_] :=
	Module[ {vars = Part[args, All, "variable"], ef},
		ef = Apply[ RuleDelayed, {pos, vars}];
		ef = ReplacePart[ef, {2,0} -> fun];
		ef
	]

(*
  Fix SequenceLength,  just return the argument length.
*)
fixSequenceFeatures[  pos_ -> Compile`SequenceLength[_], args_, argLen_] :=
	pos -> argLen

	
(*
  Fix SequenceElement,  this just returns the variable (and throws an 
  exception if it doesn't exist).
*)
fixSequenceFeatures[ pos_ -> Compile`SequenceElement[_, index_], args_, argLen_] :=
	Module[ {},
		If[ index < 0 || index >= argLen,
			ThrowException[{"Function argument sequence element does not exist", index, argLen}]
		];
		pos -> Part[args, index+1, "variable"]
	]



(*
  Fix the body of a function that has Sequence arguments.   We are looking for sequence macros.
  The body comes wrapped in Hold and is returned wrapped in Hold.
  
  We find the positions of the sequence macros.
  Extract the macros exprs and process them.
  
  Substitute them back into the body.
  
  We have to use Rule to allow evaluation of the macros processing code. 
  Note that the SequenceIterate macro is HoldAll to prevent evaluation of its 
  body -- this is code.
*)
fixSequenceFunctionBody[ body_, args_, argLen_] :=
	Module[ {posN, elems},
		posN = Position[ body, Compile`SequenceLength[_] | Compile`SequenceElement[_, _] | 
					Compile`SequenceIterate[_, _] | Compile`SequenceApply[_, _]];
		elems = Map[ (# -> Extract[body,#])&, posN];
		elems = Map[ fixSequenceFeatures[#, args, argLen]&, elems];
		ReplacePart[ body, elems]
	]


(*
  Take a function type and a Function definition where the function variables use 
  a Sequence argument.   Fix the sequence argument so that it matches the types of 
  the corresponding function arguments and then expand any Sequence macros.
*)


"StaticAnalysisIgnore"[

fixSequenceFunction[funTy_?TypeArrowQ, fun:HoldPattern[Function[ vars_?hasSequence, body_]]] :=
	Module[ {args, argLen, newBody, newFun},
		If[ Length[vars] =!= 1,
			ThrowException[{"Function argument sequence ony supports one sequence", vars, funTy["unresolve"]}]
		];
		args = Map[ createSequenceArgument[stripType[#["unresolve"]]]&, funTy["arguments"]];
		argLen = Length[args];
		newBody = fixSequenceFunctionBody[ Hold[body], args, argLen];
		newFun = Function @@ {Part[args, All, "argument"], newBody};
		Delete[ newFun, {2,0}]
	]

]; (* StaticAnalysisIgnore *)



stripType[ TypeSpecifier[a_]] :=
	stripType[a]

stripType[ Type[a_]] :=
	stripType[a]

stripType[ a_] :=
	a

getVar[ Typed[ var_, ty_]] :=
	var
	
getVar[ var_] :=
	var


getFunctionArguments[vars_, funTy_?TypeArrowQ] :=
	(
	If[ Length[vars] =!= Length[funTy["arguments"]],
		ThrowException[{"Function type length does not match definition", vars, funTy["unresolve"]}]
	];
	MapThread[ Typed, {Map[ getVar, vars], Map[stripType[#["unresolve"]]&, funTy["arguments"]]}]
	)
	
getFunctionArguments[vars_, funTy_] :=
	ThrowException[{"Type not a valid function type", funTy["unresolve"]}]


hasSequence[ vars_] :=
	AnyTrue[vars, MatchQ[#, Compile`ArgumentSequence[_]]&]



(*
  TODO,  probably remove the legacy sequence functionality.
*)
"StaticAnalysisIgnore"[

fixFunction[ funTy_?TypeArrowQ, fun:HoldPattern[Function[ vars_?hasSequence, body_]]] :=
	fixSequenceFunction[ funTy, fun]

]; (* StaticAnalysisIgnore *)




"StaticAnalysisIgnore"[


(*
 Just needs to strip out Typed, this is because type info is added 
 with a Typed applied to the entire function.
*)
fixFunction[ funTy_?TypeArrowQ, HoldPattern[Function[ vars_, body_]]] :=
	Module[ {},
		ReplaceRepeated[Function[ vars, body],
			HoldPattern[Function][{a1___, Typed[v_,_], a2___}, b_] :> Function[{a1,v,a2},b]]
	]

]; (* StaticAnalysisIgnore *)




(*
  Mangle based on type
*)

mangleType[tyEnv_, ty_Type] :=
	mangleType[tyEnv, tyEnv["resolve", ty]] 

mangleType[tyEnv_, ty_?TypeLiteralQ] :=
	ToString[ty["value"]]

mangleType[tyEnv_, ty_?TypeConstructorQ] :=
	ty["name"]

mangleType[tyEnv_, ty_?TypeApplicationQ] :=
	StringJoin[
		Riffle[Prepend[ Map[ mangleType[tyEnv, #]&, ty["arguments"]], mangleType[tyEnv, ty["type"]]], "_"]
	]

mangleType[tyEnv_, ty_?TypeArrowQ] :=
	StringJoin[
		Riffle[Append[ Map[ mangleType[tyEnv, #]&, ty["arguments"]], mangleType[tyEnv, ty["result"]]], "_"]
	]

mangleType[tyEnv_, Undefined] :=
	ThrowException[{"Undefined type not supported by mangling "}]
	
mangleType[tyEnv_, ty_] :=
	ThrowException[{"Type not supported by mangling ", ty["unresolve"]}]
	
mangleType[args___] :=
	ThrowException[{"Invalid arguments to mangle type ", {args}}]

getName[tyEnv_, name_Symbol] :=
	StringReplace[Context[name], "`" -> "__"] <> SymbolName[name]
	
getName[tyEnv_, name_String] :=
	name
	
getName[tyEnv_, h_[ name_]] :=
	getName[tyEnv, h] <> "_" <> getName[tyEnv, name]

mangleName[tyEnv_, name_, funTy_] :=
	"_" <> getName[tyEnv, name] <> "_" <> mangleType[tyEnv, funTy]


processDefinition[ self_, pm_, funNameIn_, funTy_, def_] :=
	Module[ {fun, funName, newFun, progExpr, pmNew, fm, fmSer, localFuns, localFunsSer, inline, alias},
		funName = mangleName[pm["typeEnvironment"], funNameIn, funTy];
		(*
		  Lookup the implementation and inline setting from the def.
		  The inline setting if there will have come from MetaData 
		  around the function definition.
		*)
		Quiet[
			fun = Lookup[ def, "Implementation", Null];
			inline = Lookup[def, "Inline", Automatic];
			alias = Lookup[def, "ArgumentAlias", False];
			,
			(* this is a benign message warning about Function[{ArgumentSequence[a]}, ...] *)
			{Function::flpar}
		];
		If[ fun === Null,
			ThrowException[{"Cannot find function body", funNameIn, def}]
		];
		(*
		  We add the cache definition here in case in compiling this function we reach 
		  the name of this function again.
		*)
		addToCache[self,  funName, Join[ def, <|"type" -> funTy, "Class" -> "Redirect", "Redirect" -> funName|>]];
		newFun = fixFunction[funTy, fun];
		newFun = Typed[ newFun, funTy["unresolve"]];
		progExpr = MetaData[<|
			"Name" -> funName,
			"UnmangledName" -> funNameIn,
			"Inline" -> inline,
			"CompilerDefinition" -> True,
			"ArgumentAlias" -> alias,
			"LLVMFunctionAttributes" -> {"dso_local", "weak"} 
		|>]@newFun;
		(**
		 * TODO:
		 * I don't think CompileExprRecurse ever throws an exception to be caught by CatchTypeFailure or CatchException.
		 * It only returns a Failure object (after issuing a message).  This should be revisited, maybe CompileExprRecurse
		 * should throw exceptions that are caught here.
		 * 
		 * Need to set "OptimizationLevel" so that the resulting function is properly filled out, eg with types.
		 * Perhaps could restrict to non calling ResolveFunctionCall.
		 **)
		pmNew = CatchTypeFailure[
			Module[ {logger = pm["getProperty", "PassLogger", Null]},
				If[ logger === Null,
					CompileExprRecurse[progExpr, "OptimizationLevel" -> 1, Sequence@@Normal[pm["getProperty", "environmentOptions", {}]], "TypeEnvironment" -> pm["typeEnvironment"]],
					CompileExprRecurse[progExpr, "OptimizationLevel" -> 1, Sequence@@Normal[pm["getProperty", "environmentOptions", {}]], "TypeEnvironment" -> pm["typeEnvironment"], "PassLogger" -> logger]]
			]
			,
			All, processTypeFailure[self, funNameIn, funTy, newFun, #1]&];
		If[FailureQ[pmNew],
			removeFromCache[ self, funName];
			ThrowException[pmNew[[2]]]
		];
		fm = pmNew["getFunctionModule", funName];
		If[ MissingQ[ fm],
			ThrowException[{"Could not find function module ", funName}]
		];
		fmSer = WIRSerialize[ pm["typeEnvironment"], fm];
		(*
		  Need to take care of any local functions,  these need to be passed on.
		  Probably should think about the name -- maybe should use a UUID?
		  
		  And we should not be adding the definitions of the local functions to the value stored in
		  the cache. We should be adding just the names here, and make sure the definitions are also
		  in the cache. 
		*)
		localFuns = Select[pmNew["functionModules"]["get"], #["getProperty", "localFunction", False]&];
		localFunsSer = Map[ WIRSerialize[ pm["typeEnvironment"], #]&, localFuns];
		pmNew["externalDeclarations"]["scan",
			addToCache[self, First[#], Last[#]]&];
		pmNew["dispose"];
		addToCache[self,  funName, Join[ def, <| "type" -> funTy["unresolve"], 
					"Class" -> "Definition", "Definition" -> fmSer,
					"LocalDefinitions" -> localFunsSer|>]];
		<|"Class" -> "Redirect", "Redirect" -> funName|>
	]



Compile::typefailure = "Compiling the function `Name` encountered a type failure `Tag`."

processTypeFailure[ self_, funName_, funTy_, body_, typeFailure_] :=
	Module[ {failure, ty = funTy["unresolve"]},
		failure =
			Failure["CompileTypeFailure", 
				<|"MessageTemplate" -> Compile::typefailure, 
  				  "MessageParameters" -> <|"Name" -> funName, "Tag" -> typeFailure[[1]]|>, 
                  "Name" -> funName,
                  "Type" -> ty, 
                  "Body" -> body, 
                  "TypeFailure" -> typeFailure
            	|>];
         Throw[failure, TypeError]
	]
	
	


(*
  The TypeArrow check here is to make sure we only add monomorphic definitions 
  to the cache.
  
  The cache is more of a store than an optimization cache.
  It is required to load functions eg the targets of a redirect, ie 
  that are not resolved via type inference.
  
  Note the cache is loaded from the compiler state.
*)
addToCacheMangle[ self_, tyEnv_, funName_, ty_, def_] :=
	Which[
		Head[ty] === Type || Head[ty] === TypeSpecifier,
		(*
		  I think this should be addToCacheMangle
		*)
			addToCache[self, tyEnv, funName, tyEnv["resolve", ty], def],
		TypeArrowQ[ty],
			addToCache[self, mangleName[tyEnv, funName, ty], def],
		True,
			Null
	]



addToCache[ self_, funName_, def_] :=
	self["cache"]["associateTo", funName -> def]

removeFromCache[ self_, funName_] :=
	self["cache"]["keyDropFrom", funName]

cacheSearch[self_, pm_, funName_, funTy_] :=
	Module[ {res},
		res = self["cache"]["lookup", funName, Null];
		If[ res === Null,
			(* We tried the normal name and found nothing, so try the mangled name *)
			self["cache"]["lookup", mangleName[pm["typeEnvironment"], funName, funTy], Null]
			,
			res
		]
	]

$classFunction =
<|
	"Linked" -> returnDef,
	"Redirect" -> returnDef,
	"Erasure" -> returnDef,
	"General" -> processDefinition
|>



throwError[self_, pm_, funName_, funTy_, def_] :=
	ThrowException[{"Unknown function resolution class", funName, def}]

returnDef[self_, tyEnv_, funName_, funTy_, def_] :=
	Module[{},
		def
	]
	
process[ self_, pm_, funName_, funTy_, def_] :=
	Module[ {workFun, res},
		res = cacheSearch[self, pm, funName, funTy];
		Which[
			res =!= Null || MissingQ[ def],
				res
			,
			AssociationQ[def],
				workFun = Lookup[$classFunction, def["Class"], throwError];
				workFun[ self, pm, funName, funTy, def]
			,
			True,
				ThrowException[{"Unknown function definition", funName, def}]
		]
	]


(*
  Typically called from LoadGlobalInstruction
*)
processAtom[self_, tyEnv_, name_] :=
	Module[ {},
		<| "type" -> tyEnv["resolve", "Expression"], "value" -> Primitive`GlobalSymbol[name], "class" -> "Constant"|>
	]



(*
 Cache operations.
*)
updateCache[self_, newCacheData_?AssociationQ] :=
	Module[ {cache = self["cache"], name, val},
		Scan[
			Function[{rule},
				name = First[rule];
				val = Last[rule];
				cache["associateTo", name -> val]]
				, Normal[newCacheData]];
	]


(**************************************************)

icon := Graphics[Text[
	Style["CFunDef",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[env_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"CompileFunctionDefinitionLookup",
		env,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["specific: ", {90, Automatic}], env["cache"]}]
  		},
  		{

  		}, 
  		fmt,
		"Interpretable" -> False
  	]


toString[env_] := "CompileFunctionDefinitionLookup[<>]"





End[]

EndPackage[]
