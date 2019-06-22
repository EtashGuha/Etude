

BeginPackage["Compile`Core`Analysis`Utilities`VariableList`"]

VariableList;

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]


VariableList[fm_?FunctionModuleQ] :=
    Module[{
    	   ref = CreateReference[{}]
    	},
    	   iVaraibleList[ref, fm];
       union[ref["get"]]
    ];
VariableList[bbs:{__?BasicBlockQ}] :=
	union[Flatten[VariableList /@ bbs]]
VariableList[bb_?BasicBlockQ] :=
    Module[{
       ref = CreateReference[{}]
    },
       iVaraibleList[ref, bb];
       union[ref["get"]]
    ]
   
iVariableList[ref_, fm_?FunctionModuleQ] :=
    fm["scanBasicBlocks", iVariableList[ref, #]&]
iVariableList[ref_, bb_?BasicBlockQ] :=
    fm["scanInstructions", iVariableList[ref, #]&]
iVariableList[ref_, inst_?InstructionQ] :=
    Module[{},
        If[inst["definesVariableQ"],
            ref["appendTo", inst["definedVariable"]]
       ];
       If[inst["usedVariables"],
            Scan[ref["appendTo", #]&, inst["usedVariables"]]
       ]
    ]
    
union[lst_?ListQ] :=
    SortBy[
    	   DeleteDuplicatesBy[lst, idOf],
    	   idOf
    ]
    
idOf[e_] := e["id"]

End[]

EndPackage[]
