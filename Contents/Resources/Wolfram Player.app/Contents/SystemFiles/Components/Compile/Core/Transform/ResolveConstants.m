BeginPackage["Compile`Core`Transform`ResolveConstants`"]

ResolveConstantsPass

Begin["`Private`"] 

Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`TypeSystem`Inference`InferencePassState`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`IR`Instruction`BinaryInstruction`"]
Needs["Compile`Core`IR`Instruction`CompareInstruction`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]

(*
 Process constants with types Rational[Integer], Complex[Integer], Complex[Rational[Integer]] 
 when combined with a Real or Complex[Real]
*)

(*
  We've recomputed the instruction with new arguments.  Now we should 
  reinfer the type,  to get the function marked with the type object 
  (needed for implementation metadata).  We also make the output undefined, 
  this should not strictly be necessary,  but is needed because of insufficient 
  TypeJoin definitions, eg for Function[{Typed[arg, "Complex"["Real64"]]}, I + arg].
*)
fixFunctionType[ data_, inst_] :=
	Module[ {inferSt, oper, operTy, argTys, resTy, funTy},
		inferSt = CreateInferencePassState[data["programModule"], Null];
		oper = inst["operator"];
		If[ !ConstantValueQ[oper],
			Return[ False]
		];
		operTy = inferSt["processSource", oper, inst];
		argTys = inferSt["processSource", inst["operands"], inst];
		resTy = inferSt["resolve", TypeSpecifier[TypeVariable["res$" <> ToString[inst["id"]]]]];
		inst["target"]["setType", Undefined];
		inferSt["processTarget", inst["target"], resTy, inst];
        funTy = inferSt["resolve", TypeSpecifier[ argTys -> resTy]];
        inferSt["appendEqualConstraint", operTy, funTy, inst];
		CatchException[
			inferSt["solve"];
			True
			,
			{{_, False&}}
		]
	]




fixDefinitionInstruction[data_, inst_?CopyInstructionQ] :=
	Module[{src, newSrc, newInst, newTrgt},
		src = inst["source"];
		newSrc = fixDefinitionValue[data, src];
		If[ newSrc === None,
			newSrc,
			newInst = CreateCopyInstruction[ "rational", newSrc, inst["mexpr"]];
			newTrgt = newInst["target"];
			newTrgt["setType", newSrc["type"]];
			newInst["moveAfter", inst];
			newTrgt]
	]

fixDefinitionInstruction[data_, _] :=
	None


fixDefinitionValue[data_, arg_?VariableQ, complex_] :=
	Module[ {def},
		def = arg["def"];
		fixDefinitionInstruction[ data, def]
	]


fixDefinitionValue[data_, arg_?ConstantValueQ, complexQ_] :=
	Module[ {val, consNew},
		val = arg["value"];
		consNew = CreateConstantValue[ N[val]];
		consNew["setType", If[ complexQ, data["complexreal64"], data["real64"]]];
		consNew
	]

fixDefinitionValue[data_, _, _] :=
	None


isIntegerRationalExact[ arg_] :=
	Module[ {type = arg["type"]},
		Which[ 
			type["isNamedApplication", "Rational"] && Length[type["arguments"]] === 1,
				True
			,
			type["isConstructor", "RealExact"],
			  True
			,
			True,
				False
		]
	]

isComplexIntegerRationalExact[ arg_] :=
	Module[ {tyArg, type = arg["type"]},
		If[ !type["isNamedApplication", "Complex"] || Length[type["arguments"]] =!= 1,
			Return[False]
		];
		tyArg = Part[ type["arguments"],1];
		tyArg["isConstructor", "Integer32"] || tyArg["isConstructor", "Integer64"] || tyArg["isConstructor", "RealExact"] || tyArg["isNamedApplication", "Rational"]
	]

isComplexRealOrReal[ type_] :=
	Module[ {tyArg},
		If[ type["isNamedApplication", "PackedArray"],
			Return[isComplexRealOrReal[First[type["getArguments"]]]]];
		If[type["isConstructor", "Real64"],
			Return[True]
		];
		If[ !type["isNamedApplication", "Complex"] || Length[type["arguments"]] =!= 1,
			Return[False]
		];
		tyArg = Part[ type["arguments"],1];
		tyArg["isConstructor", "Real64"]
	]




(*
  If one argument of the binary instruction is a Rational and the other a Real, 
  then we should change the Rational to a Real.
*)

checkRationalComplexInstruction[data_, inst_] :=
	Module[ {opers, pos, realTest, argRat, newArg, oper, operNew, newInst, complexQ = False},
		oper = inst["operator"];
		If[ !ConstantValueQ[oper],
			Return[False]
		];
		opers = inst["operands"];
		pos = Position[ opers, _?isIntegerRationalExact, {1}, Heads->False];
		If[ !MatchQ[ pos, {{_Integer}}],
			complexQ = True;
			pos = Position[ opers, _?isComplexIntegerRationalExact, {1}, Heads->False]
		];
		If[ !MatchQ[ pos, {{_Integer}}],
			Return[False]
		];
		pos = Part[pos,1,1];
		argRat = Part[opers, pos];
		realTest = Select[ opers, isComplexRealOrReal[#["type"]]&];
		If[Length[realTest] =!= 1,
			Return[False]
		];
		newArg = fixDefinitionValue[ data, argRat, complexQ];
		If[ newArg === None,
			Return[False]
		];
		opers = ReplacePart[opers, pos -> newArg];
		operNew = CreateConstantValue[oper["value"]];
		newInst = Which[
			BinaryInstructionQ[inst],
				CreateBinaryInstruction[inst["target"], operNew, opers]
			,
			CompareInstructionQ[inst],
				CreateCompareInstruction[inst["target"], operNew, opers]
			,
			True,
				ThrowException[{"Unexpected instruction", inst}]
		];
		If[fixFunctionType[ data, newInst],
			newInst["moveAfter", inst];
			inst["unlink"]
		];
		True
	]




isIntegerZero[ data_, val_] :=
	ConstantValueQ[val] && val["value"] === 0


(*
  Move these predicates into the TypeEnvironment
*)
$integerTypes = 
	<|
	"Integer8" -> True, "UnsignedInteger8" -> True,
	"Integer16" -> True, "UnsignedInteger16" -> True,
	"Integer32" -> True, "UnsignedInteger32" -> True,
	"Integer64" -> True, "UnsignedInteger64" -> True
	|>

isIntegerScalar[data_, ty_] :=
	TypeConstructorQ[ty] && Lookup[$integerTypes, ty["typename"], False]

$realTypes = 
	<|
	"Real16" -> True, "Real32" -> True, "Real64" -> True
	|>

isIntegerScalar[data_, ty_] :=
	TypeConstructorQ[ty] && Lookup[$integerTypes, ty["typename"], False]
	
isRealScalar[data_, ty_] :=
	TypeConstructorQ[ty] && Lookup[$realTypes, ty["typename"], False]

isComplexRealScalar[data_, ty_] :=
	ty["isNamedApplication", "Complex"] && isRealScalar[data, First[ty["arguments"]]]

(*
  See if we have constant integer zero times something.  If so if the something is a
  scalar replace with a zero of that type.
*)
checkZeroTimes[data_, inst_] :=
	Module[{oper, arg1, arg2, zeroArg, other, otherTy, val, source, newInst},
		oper = inst["operator"];
		If[ !(ConstantValueQ[oper] && oper["value"] === Times),
			Return[False]
		];
		arg1 = inst["getOperand", 1];
		arg2 = inst["getOperand", 2];
		(*
		   Pick out if one of args is an Integer Zero
		*)
		Which[ 
			isIntegerZero[data, arg1],
				zeroArg = arg1;
				other = arg2
			,
			isIntegerZero[data, arg2],
				zeroArg = arg2;
				other = arg1
			,
			True,
				Return[False]];
		(*
		   Determine the value to use if we have a scalar.
		*)
		otherTy = other["type"];
		Which[
			isIntegerScalar[data, otherTy],
				val = 0
			,
			isRealScalar[data, otherTy],
				val = 0.
			,
			isComplexRealScalar[data, otherTy],
				val = 0. + 0. I
			,
			True,
				Return[False]];
		source = CreateConstantValue[val];
		source["setType", otherTy];
		newInst = CreateCopyInstruction[inst["target"], source, inst["mexpr"]];
		newInst["moveAfter", inst];
		inst["unlink"];
		True		
	]
	
	

checkInstruction[data_, inst_] :=
	Module[ {},
		If[
			checkRationalComplexInstruction[data, inst]
			,
			Null
			,
			checkZeroTimes[data, inst]]
	]




run[fm_, opts_] :=
	Module[{data, visitor, tyEnv, pm = fm["programModule"]},
		tyEnv = pm["typeEnvironment"];
		data = <|
			"programModule" -> pm,
			"typeEnvironment" -> tyEnv,
			"complexreal64" -> TypeSpecifier["Complex"["Real64"]],
			"real64" -> TypeSpecifier["Real64"]
		|>;
		visitor =  CreateInstructionVisitor[
			data,
			<|
				"visitBinaryInstruction" -> checkInstruction,
				"visitCompareInstruction" -> checkInstruction
			|>,
			"IgnoreRequiredInstructions" -> True
		];
		visitor["traverse", fm];
	    fm
	]


RegisterCallback["RegisterPass", Function[{st},
resolveConstantsInfo = CreatePassInformation[
	"ResolveConstants",
	"The pass resolves constants such as rationals, complex of integers and complex of rationals " <>
	"which are combined with floating point variables.",
	"Rationals and Gaussian integers are supported in the type system, but there is no implementation for them."<>
	" Consequently, any function that combines a rational/complex number with a float should convert " <>
	"the rational number into a float or complex into a complex float."
];

ResolveConstantsPass = CreateFunctionModulePass[<|
	"information" -> resolveConstantsInfo,
	"runPass" -> run
|>];

RegisterPass[ResolveConstantsPass]
]]

End[]
	
EndPackage[]
