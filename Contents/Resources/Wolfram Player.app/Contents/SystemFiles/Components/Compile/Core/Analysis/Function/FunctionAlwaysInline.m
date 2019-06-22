

BeginPackage["Compile`Core`Analysis`Function`FunctionAlwaysInline`"]
(* 

Cases where the function must be inlined, for example if it allocates values 
on the stack and then returns the stack variable
 *)
 
 
FunctionAlwaysInlinePass

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`Analysis`DataFlow`Def`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`IR`Instruction`StackAllocateInstruction`"]
Needs["CompileUtilities`Callback`"]
 


visitDef[st_, inst_?CopyInstructionQ] :=
	Module[{val = inst["source"]},
		If[!ConstantValueQ[val],
			visitDef[st, val["def"]]]
	]

visitDef[st_, inst_?StackAllocateInstructionQ] :=
	st["stackFound"]["set", True]
	

visitReturn[ st_, inst_] :=
	Module[{val},
		If[st["stackFound"]["get"],
			Return[]];
		If[!inst["hasValue"],
			Return[]];
		val = inst["value"];
		If[!ConstantValueQ[val],
			visitDef[st, val["def"]]]
	]

(*
 Return True if a function returns a variable that was defined through a stack allocate instruction.
*)
returnsStackAllocate[fm_, opts_] :=
    Module[{
        st = <|"stackFound" -> CreateReference[]|>, visitor
    },
    	st["stackFound"]["set", False];
	    visitor =
	    	CreateInstructionVisitor[
	    	st,
	        <|
	            "visitReturnInstruction" -> visitReturn
	        |>,
	        "IgnoreRequiredInstructions" -> True
	    ];
	    visitor["visit", fm];
	    st["stackFound"]["get"]
    ];

run[fm_?FunctionModuleQ, opts_] :=
    Module[{alwaysInline = returnsStackAllocate[fm, opts]},
    	If[ 
    		alwaysInline,
    		fm["information"]["inlineInformation"]["setInlineValue", "Always"]];
        fm
    ];

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
        "FunctionAlwaysInline",
        "This pass computes whether the function is must be inlined.",
        "A function must be inlined if any of the returned values are defined using a stack allocate (there might be other conditions)."
];

FunctionAlwaysInlinePass = CreateFunctionModulePass[<|
    "information" -> info,
    "runPass" -> run,
    "requires" -> {
        DefPass
    },
    "passClass" -> "Analysis"
|>];

RegisterPass[FunctionAlwaysInlinePass]
]]

End[]

EndPackage[]