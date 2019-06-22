
BeginPackage["Compile`Core`Analysis`Function`FunctionIsTrivialCall`"]

FunctionIsTrivialCallPass

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`InvokeInstruction`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileUtilities`Callback`"]
    
callOrInvokeInstructionQ[inst_] := CallInstructionQ[inst] || InvokeInstruction[inst]

(* TODO: we should probably put this information as meta data on the function itself *)
isTrivialCall[Native`PrimitiveFunction[s_?StringQ]] := 
    isTrivialCall[s]
    
isTrivialCall[s_?StringQ] :=
    Which[
        MemberQ[{"AddressShift", "BitCast"}, s],
            True,
        StringStartsQ[s, "Unchecked"],
            True,
        StringStartsQ[s, "binary_sameq_"],
            True,
        StringStartsQ[s, "binary_unsameq_"],
            True,
        StringStartsQ[s, "binary_equal_"],
            True,
        StringStartsQ[s, "binary_unequal_"],
            True,
        StringStartsQ[s, "binary_greater_"],
            True,
        StringStartsQ[s, "binary_greaterequal_"],
            True,
        StringStartsQ[s, "binary_less_"],
            True,
        StringStartsQ[s, "binary_lessequal_"],
            True,
        StringStartsQ[s, "not_"],
            True,
        True,
            False
    ]
    
isTrivialCall[___] :=
    False

isTrivialCallInstructionQ[inst_?callOrInvokeInstructionQ] :=
    Module[{
        fun = inst["function"],
        funName
    },
        If[!ConstantValueQ[fun],
            Return[False]
        ];
        funName = fun["value"];
        (* Some calls are not really calls, such as Unchecked ones or bit casts *)
        isTrivialCall[funName]
    ]

isTrivialCallInstructionQ[___] := False


isTrivialFunction[fm_] :=
	False




$veryTrivialSkip =
 <|
    "LabelInstruction" -> False,
    "LoadArgumentInstruction" -> False,
    "ReturnInstruction" -> False
 |>

(*
 Return True if a function has only one basic block and only one instruction
 which is not a Label, LoadArgument or Return instruction. 
 This function really should be inlined.
*)
isSimpleFunction[ fm_] :=
	Module[{bbs = fm["getBasicBlocks"], inst, cnt = 0},
        If[Length[bbs] > 1,
            Return[False]
        ];
		inst = First[bbs]["firstInstruction"];
		While[ cnt < 2 && inst =!= None,
			If[Lookup[$veryTrivialSkip, inst["_instructionName"], True],
				cnt = cnt + 1];
			inst = inst["next"];
		];
		cnt < 2
	]

run[fm_?FunctionModuleQ, opts_] :=
    Module[{trivial = False},
        Which[
        	isSimpleFunction[fm],
        		trivial = True
        	,
        	isTrivialFunction[fm],
        		trivial = True
        	,
        	True,
        		trivial = False];
        fm["information"]["inlineInformation"]["setIsTrivial", trivial];
        fm
    ]

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
        "FunctionIsTrivialCall",
        "This pass computes whether the function is trivial call.",
        "A function is trivial call if it has only one basic block and only one call instruction."
];

FunctionIsTrivialCallPass = CreateFunctionModulePass[<|
    "information" -> info,
    "runPass" -> run,
    "passClass" -> "Analysis"
|>];

RegisterPass[FunctionIsTrivialCallPass]
]]

End[]

EndPackage[]
