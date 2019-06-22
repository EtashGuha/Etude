
BeginPackage["Compile`TypeSystem`Inference`InferencePassState`"]

CreateInferencePassState
InferencePassStateQ

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["TypeFramework`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`Inference`TypeInferenceState`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]
Needs["CompileAST`Class`Base`"]

RegisterCallback["DeclareCompileClass", Function[{st},
InferencePassStateClass = DeclareClass[
	InferencePassState,
	<|
		
		"solve" -> Function[{}, solve[Self]],
		"addAssumption" -> (addAssumption[Self, ##]&),
		"propagateAssumptions" -> (propagateAssumptions[Self, ##]&),
		"dropAssumption" -> Function[{var}, dropAssumption[Self, var]],
		"hasAssumption" -> Function[{var}, hasAssumption[Self, var]],
		"lookupAssumptions" -> (lookupAssumptions[Self, ##]&),
		"appendEqualConstraint" -> (appendEqualConstraint[Self, ##]&),
		"appendLookupConstraint" -> (appendLookupConstraint[Self, ##]&),
		"addData" -> Function[{var, type}, addData[Self, var, type]],
		"resolve" -> Function[{ty}, resolve[Self, ty]],
		"processSource" -> (processSource[Self, ##]&),
		"processTarget" -> (processTarget[Self, ##]&)
		
	|>,
	{
		"type",
		"programModule",
		"typeEnvironment",
		"typeInferenceState",
		"patternState" -> Null,
		"typeMap",
		"dataMap",
		"typeVarMap"
	},
	Predicate -> InferencePassStateQ
]
]]

CreateInferencePassState[programModule_, generateFun_] :=
	Module[{var, val, typeEnv = programModule["typeEnvironment"]},
		val = getValue[mexpr];
		var = CreateObject[
			InferencePassState,
			<|
				"programModule" -> programModule,
				"type" -> CreateReference[],
				"typeEnvironment" -> typeEnv,
				"dataMap" -> CreateReference[<||>],
				"typeMap" -> CreateReference[<||>],
				"typeVarMap" -> CreateReference[<||>],
				"typeInferenceState" -> CreateTypeInferenceState[ typeEnv, generateFun]
			|>
		];
		var
	]


processSource[ self_, arg_] :=
	processSource[self, arg, None]

(*
 processSource gets the type from the value.
 Directly from a constant or a type variable from a variable (and adding an assumption).
 Set the type for the result (might be void).
*)

processSource[self_, var_?VariableQ, inst_] :=
	Module[ {ty},
		ty = var["type"];
		self["addAssumption", var, ty, inst]
	]


getLiteral[self_, Complex[arg1_, arg2_], inst_] :=
	Module[ {t1, t2, tJoin},
		t1 = getLiteral[ self, arg1, inst];
		t2 = getLiteral[ self, arg2, inst];
		tJoin = self["typeEnvironment"]["resolve", TypeSpecifier["Complex"[ "TypeJoin"[t1, t2]]]];
		tJoin
	]

getLiteral[self_, val_Rational, _] :=
	self["typeEnvironment"]["resolve", TypeSpecifier["Rational"["MachineInteger"]]]

getLiteral[ self_, val_Integer, _] :=
	self["typeEnvironment"]["resolve", TypeSpecifier["MachineInteger"]]

getLiteral[ self_, val_Real, _] :=
	self["typeEnvironment"]["resolve", TypeSpecifier["Real64"]]

getLiteral[ self_, val:(True|False), _] :=
	self["typeEnvironment"]["resolve", TypeSpecifier["Boolean"]]

getLiteral[ self_, val_String, _] :=
	self["typeEnvironment"]["resolve", TypeSpecifier["String"]]

getLiteral[ self_, val_?NumericQ, inst_] :=
	Module[{test = N[val]},
		Which[ 
			Head[test] === Real,
				self["typeEnvironment"]["resolve", TypeSpecifier["RealExact"]]
			,
			Head[test] === Complex,
				self["typeEnvironment"]["resolve", TypeSpecifier["Complex"["RealExact"]]]
			,
			True,
				getLiteralFallThrough[self, val, inst]
		]	
	]


(*
 Looks like a PackedArray?  If so fill out appropriately.
*)
getLiteral[ self_, val_List, inst_] :=
	Module[{valPacked = Developer`ToPackedArray[val], rank, elem},
		If[Developer`PackedArrayQ[valPacked],
			elem = getBaseType[self, valPacked];
			If[ elem === Null,
				getLiteralFallThrough[self, val, inst]
				,
				rank = TensorRank[valPacked];
				self["typeEnvironment"]["resolve", TypeSpecifier["PackedArray"[elem, rank]]]]
			,
			getLiteralFallThrough[self, val, inst]]
	]
			

getBaseType[self_, {e1_, ___}] :=
	getBaseType[self, e1]
	
getBaseType[self_, e_Integer] :=
	"MachineInteger"

getBaseType[self_, e_Real] :=
	"Real64"

getBaseType[self_, Complex[arg1_, arg2_]] :=
	"Complex"[getBaseType[self, arg1]]
	
getBaseType[self_, _] :=
	Null

(*
 Fall Through
*)

getLiteral[self_, val_, inst_] :=
	getLiteralFallThrough[self, val, inst]



	
(*
  TODO fix this,  perhaps should be an error?
*)
getLiteralFallThrough[self_, val_, inst_] :=
	Module[ {fm},
		fm = self["programModule"]["getFunctionModule", val];
		If[ !MissingQ[fm] && fm["type"] =!= Undefined,
			fm["type"],
			addLookup[self, val, inst]
		]
	]

attachSource[ self_, cons_, inst_] :=
	Module[ {mexpr},
		If[cons === Null || inst === None || !MExprQ[inst["mexpr"]], 
			Return[]];
		mexpr = HoldForm @@ inst["mexpr"]["toExpression"];
		cons["setProperty", "sourceInstruction" -> mexpr]

	]

addLookup[ self_, val_, inst_] :=
	Module[ {ty, inferSt = self["typeInferenceState"], cons},
		ty = CreateTypeVariable["Lookup"];
		cons = inferSt["appendLookupConstraint", val, ty, inferSt["monomorphicSet"]["get"]];
		attachSource[self, cons, inst];
		ty
	]
	

processSource[self_, cons_?ConstantValueQ, inst_] :=
	Module[ {ty},
		ty = cons["type"];
		If[ty === Undefined,
			ty = getLiteral[self, cons["value"], inst];
			self["addData", cons, ty]
			,
			If[ ty["variableCount"] > 0,
				self["addData", cons, ty]
			]
		];
		ty
	]




(*
  Coming here this must be an external function, so 
  add a LookupConstraint.
*)	
processSource[ self_, val_Symbol, inst_] :=
	addLookup[self, val, inst]
	

processSource[ self_, args_List, inst_] :=
	Map[ processSource[self,#, inst]&, args]

processSource[args___] :=
	ThrowException[{"Bad arguments to processSource", {args}}]

(*
  processTarget gets any assumptions from the trgt (these relate to the uses) and 
  adds equality constraints for these with the type of the source.

  lookupAssumptions returns all the types of the uses of trgt, we add Equality constraints
  for these with ty.   If trgt has no assumptions, this is because trgt is not used. Really, 
  this is dead code, but we add an assumption to make sure that the trgt gets a type.
  At the end we drop the assumptions for trgt because it is passing out of scope.
  
  If there are multiple types,  which might only come from Phi instructions then add 
  equality constraints for all of these.
*)


processTarget[ self_, trgt_, tys:{ty1_, tyr___}, inst_] :=
	Module[ {},
		If[ !self["hasAssumption", trgt], 
			processSource[self, trgt, inst]
		];
		Scan[ 
			addEqualityConstraints[self, tys,#, inst]&,
			self["lookupAssumptions", trgt]
		];
		Scan[
			self["appendEqualConstraint", #, ty1, inst]&,
			{tyr}
		];
		self["dropAssumption", trgt];
	]

addEqualityConstraints[ self_, tys_, assumpTy_, inst_] :=
	Scan[
		self["appendEqualConstraint", #, assumpTy, inst]&,
		tys
	]

processTarget[ self_, trgt_, ty_, inst_] :=
	Module[{},
		If[ !self["hasAssumption", trgt], 
			processSource[self, trgt, inst]
		];
		Scan[
			self["appendEqualConstraint", ty, #, inst]&
			, 
			self["lookupAssumptions", trgt]
		];
		self["dropAssumption", trgt];
	]


(*
 propagateAssumptions takes assumptions from the trgt and adds them to the src.
 processTarget gets any assumptions from the trgt (these relate to the uses) and 
 adds equality constraints for these with the type of the source.
*)
propagateAssumptions[self_, trgt_, src_?VariableQ, inst_:None] :=
	Module[ {assumps},
		assumps = self["lookupAssumptions", trgt];
		Scan[
			self["addAssumption", src, #, inst]&,
			assumps
		];
	]

propagateAssumptions[ self_, trgt_, src_List, inst_:None] :=	
	Scan[propagateAssumptions[self,trgt,#, inst]&, src]

propagateAssumptions[ self_, trgt_, src_, inst_:None] :=	
	Null
	


getAdornment[ data_] :=
	Which[
		VariableQ[data],  "Variable"[data["id"]],
		ConstantValueQ[data],  "Constant"[data["id"]],
		FunctionModuleQ[data],  "Function"[data["id"]],
		Head[data] === Symbol,  data,
		StringQ[data],  data,		
		True, ThrowException[{"Unknown data element", data}]
	]
		

(*
  Add an assumption.  
  
  If the type is Undefined look in the typeMap,  if something is found the data has already been added.
  If something is not found then create a type variable and add this to the typeMap and as an assumption
  in the typeInferenceState.
  Return whatever is in the typeMap.
  
  If the type is not Undefined, then add this as an assumption in the typeInferenceState.
  Return the type.
  
  This makes sure we don't create too many type variables (and hence too many constraints) for the same object.
*)	
addAssumption[self_, data_, typeIn_, inst_:None] :=
	Module[ {adorn = getAdornment[data], type = typeIn},
		If[ type === Undefined, 
			type = self["typeMap"]["lookup", adorn, Null];
			If[ type === Null,
				type = CreateTypeVariable[ToString[adorn]];
				self["dataMap"]["associateTo", adorn -> data];
				self["typeMap"]["associateTo", adorn -> type];
				With[{
					cons = self["typeInferenceState"]["addAssumption", adorn -> {type}]
				},
					attachSource[self, cons, inst];
				]
			];
			, (* Else *)
			self["dataMap"]["associateTo", adorn -> data];
			With[{
				cons = self["typeInferenceState"]["addAssumption", adorn -> {type}]
			},
				attachSource[self, cons, inst];
			]
		];
		type
	]


dropAssumption[self_, data_] :=
	Module[ {adorn = getAdornment[data]},
		self["typeInferenceState"]["dropAssumption", adorn];
	]

lookupAssumptions[self_, data_] :=
	Module[ {adorn = getAdornment[data]},
		self["typeInferenceState"]["lookupAssumptions", adorn]
	]

hasAssumption[self_, data_] :=
	Module[ {adorn = getAdornment[data]},
		self["typeInferenceState"]["hasAssumption", adorn]
	]


(*
 This is needed so that the type for data will get substituted in solve.
*)
addData[self_, data_, type_] :=
	Module[ {adorn = getAdornment[data]},
		self["dataMap"]["associateTo", adorn -> data];
		self["typeMap"]["associateTo", adorn -> type];
		If[TypeVariableQ[type],
			self["typeVarMap"]["associateTo", type["id"] -> data]
		];
	]


resolve[self_, type_] :=
	self["typeEnvironment"]["resolve", type]


(*
  If there are no variables in either type and they are equal then don't add.
  This would be a tautology.
*)
appendEqualConstraint[self_, t1_, t2_, inst_:None] :=
	Module[ {},
		If[ t1["variableCount"] > 0 || t2["variableCount"] > 0 || 
			!t1["sameQ", t2],
			With[{
				cons = self["typeInferenceState"]["appendEqualConstraint", t1, t2]
			},
				attachSource[self, cons, inst];
				cons
			]
		]
	]

appendLookupConstraint[self_, t1_, t2_, t3_, inst_:None] :=
	With[{
		cons = self["typeInferenceState"]["appendLookupConstraint", t1, t2, t3]
	},
		attachSource[self, cons, inst];
		cons
	]


solve[self_] :=
	CatchTypeFailure[
		Module[ {subs, consState, inferState},
			inferState = self["typeInferenceState"];
			consState = inferState[ "constraints"];
			(*inferState["processAssumptions"];*)
			subs = consState["solve"];
			(*
			   Now pass through dataMap
			*)
			Scan[
				applySubstitution[self, subs, #]&
				,
				self["dataMap"]["values"]
			];
		]
	    ,
	    TypeError
	    ,
	    (* convert this TypeError into an exception that will be caught *)
	    Function[
	    	ThrowException[#1[[2]]]
	    ]
	]


(*
  If the element does not have an entry in the type map,  this is valid
  if the element has a type which is not Undefined.
*)
applySubstitution[self_, subs_, elem_] :=
	Module[{adorn = getAdornment[elem], elemTy, resolved, definition},
		elemTy = elem["type"];
		If[ elemTy === Undefined,
			elemTy = self["typeMap"]["lookup", adorn, Null];
			If[ elemTy === Null,
				ThrowException[{"Cannot find type entry", elem}]]
			];
		If[ elemTy["hasProperty", "resolvedType"],
			resolved = elemTy["getProperty", "resolvedType"];
			definition = resolved["getProperty", "definition", Null];
			If[ definition =!= Null,
				elem["setProperty", "definition" -> definition]]];
		elemTy = subs["apply", elemTy];
			(*
			  The test can't just be for ty being a type variable,  because it might be a 
			  compound type with a variable.
			*)
		If[Length[elemTy["free"]] =!= 0,
			elemTy = subs["apply", elemTy]];
		elem["setType", elemTy];
	]


processSource[self_, args___] :=
	ThrowException[{"Invalid arguments to processSource ", args}]
processTarget[self_, args___] :=
	ThrowException[{"Invalid arguments to processTarget ", args}]
propagateAssumptions[self_, args___] :=
	ThrowException[{"Invalid arguments to propagateAssumptions ", args}]
lookupAssumptions[self_, args___] :=
	ThrowException[{"Invalid arguments to lookupAssumptions ", args}]
appendEqualConstraint[self_, args___] :=
	ThrowException[{"Invalid arguments to appendEqualConstraint ", args}]
appendLookupConstraint[self_, args___] :=
	ThrowException[{"Invalid arguments to appendLookupConstraint ", args}]
addAssumption[self_, args___] :=
	ThrowException[{"Invalid arguments to addAssumption ", args}]
addLookup[self_, args___] :=
	ThrowException[{"Invalid arguments to addLookup ", args}]


End[]

EndPackage[]


