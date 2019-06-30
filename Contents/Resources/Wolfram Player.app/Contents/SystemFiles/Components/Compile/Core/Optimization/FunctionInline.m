BeginPackage["Compile`Core`Transform`FunctionInline`"]


FunctionInlinePass

Begin["`Private`"] 

Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Utilities`Serialization`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`ProgramModule`"]
Needs["Compile`TypeSystem`Inference`InferencePass`"]
Needs["Compile`Core`Transform`ResolveConstants`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`IR`Instruction`LambdaInstruction`"]
Needs["Compile`Core`IR`Instruction`LoadGlobalInstruction`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`Transform`ResolveTypes`"]
Needs["Compile`Core`PassManager`PassRunner`"]
Needs["Compile`Core`Analysis`Function`FunctionInlineInformation`"]




(*
 Inline the function fmCall in place of the call instruction.
 
 Split the current BB after the instruction, get prevBB and afterBB.
 
 Go through the new BBs,  convert LoadArgument to Copy from the arguments.
 Change the return instruction to a Copy to the target.
 Add a branch to the afterBB.
 Remove the Call Instruction.
 Change the branch in the prevBB to point to the start BB.
*)
inlineFunction[ pm_, inst_, fun_, fmCall_] :=
    Module[ {fm, inline, desSer, funFM, prevBB, afterBB, firstBB, args, trgt, branchInst},
        fm = inst["basicBlock"]["functionModule"];
        inline = fm["getProperty", "inlineCall", 0];
        fm["setProperty", "inlineCall" -> inline +1];
        desSer = WIRDeserialize[ pm["typeEnvironment"], fmCall, "UniqueID" -> True];
        Which[
            FunctionModuleQ[desSer],
                funFM = desSer,
            ProgramModuleQ[desSer] && Length[desSer["functionModules"]] === 1,
                funFM = First[desSer["functionModules"]],
            True,
                ThrowException[CompilerException["Cannot deserialize function ", fun, fmCall]]
        ];
        (*
          Need to process the FM being inlined to make sure that other functions are added 
          to the PM.   Not sure this comment is relevant.
        *)
        funFM["setProgramModule", pm];
        funFM["setTypeEnvironment", pm["typeEnvironment"]];
        resolveFunctionModule[funFM, Null]; 
            
        prevBB = inst["basicBlock"];
        afterBB = prevBB[ "splitAfter", inst];
        firstBB = funFM["firstBasicBlock"];
        args = inst["operands"];
        trgt = inst["target"];
        funFM["scanInstructions", 
                Function[{inst1},
                    inst1["setMexpr", inst["mexpr"]];
                    Switch[inst1["_instructionName"],
                        "LoadArgumentInstruction", fixLoadArgument[inst1, args],
                        "ReturnInstruction", fixReturn[inst1, trgt, afterBB]]]];
        branchInst = inst["next"];
        Assert[ BranchInstructionQ[branchInst]];
        inst["unlink"];
        branchInst["setOperand", 1, firstBB];
        prevBB["removeChild", afterBB];
        prevBB["addChild", firstBB];
        linkBasicBlocks[prevBB["functionModule"], firstBB];
    ]


linkBasicBlocks[ fm_, bb_] :=
    Module[ {children},
        If[ fm["id"] === bb["functionModule"]["id"],
            Return[]];
        fm["linkBasicBlock", bb];   
        children = bb["getChildren"];   
        Scan[linkBasicBlocks[fm, #]&, children]
    ]



(*
  replace the LoadArgumentInstruction with a CopyInstruction.
*)
fixLoadArgument[ inst_, args_] :=
    Module[ {index, copyInst},
        index = inst["index"]["data"];
        copyInst = CreateCopyInstruction[inst["target"], Part[args, index], inst["mexpr"]];
        copyInst["moveAfter", inst];
        inst["unlink"]
    ]

(*
  replace the ReturnInstruction with a CopyInstruction and add a Branch
*)
fixReturn[ inst_, trgt_, afterBB_] :=
    Module[ {copyInst, branchInst},
        copyInst = CreateCopyInstruction[trgt, inst["value"], inst["mexpr"]];
        copyInst["moveAfter", inst];
        inst["unlink"];
        branchInst = CreateBranchInstruction[ {afterBB}, inst["mexpr"]];
        branchInst["moveAfter", copyInst];
        branchInst["basicBlock"]["addChild", afterBB];
    ]


(*
  If we inline any number of calls then it is useful to run the FuseBasicBlocks pass immediately.
  The mechanism for determining the number is done with a property on the FM.  It would be better 
  to pass a state around, but this would be more intrusive on the code, which I want to keep stable.
  
  It might also be good to have the running of the FuseBasicBlock pass controlled by the PassRunner 
  that invoked this.  But that is definitely more work.
  
  TODO,  improve this to use a state to pass around so that the number of inline calls can be kept.
*)
run[fm_, opts_] :=
    Module[{logger = Lookup[opts, "PassLogger", Null], ef},
        ef = resolveFunctionModule[fm, logger];
        ef
    ]
    
RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
    "FunctionInline",
    "The pass replaces CallInstructions which create lists with a sequence of instructions " <>
    "that actually do the work."
];

FunctionInlinePass = CreateFunctionModulePass[<|
    "information" -> info,
    "runPass" -> run,
    "requires" -> {
        FunctionInlineInformationPass
    }
|>];

RegisterPass[FunctionInlinePass]
]]

End[] 

EndPackage[]
