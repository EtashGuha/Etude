BeginPackage["Compile`Core`Analysis`DataFlow`LiveInterval`"]

LiveIntervalPass;
LiveInterval;

Begin["`Private`"]

Needs["Compile`Core`Analysis`DataFlow`LiveVariables`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]



makeRange[var_, lst_List] :=
	If[Length[lst] >= 2,
		LiveInterval[var, <| "start" -> First[lst], "end" -> Last[lst] |>],
		LiveInterval[var, <| "start" -> First[lst], "end" -> First[lst]["next"] |>]
	]

LiveInterval[var_, _]["variable"] := var
LiveInterval[_, s_]["start"] := s["start"]
LiveInterval[_, s_]["end"] := s["end"]
LiveInterval[_, s_]["length"] := s["end"] - s["start"]

run[fm_, opts_] :=
	Module[{stmts, lv, currentRange, key},
		stmts = <||>;
		lv = {};
		CreateInstructionVisitor[
			<|
				"visitInstruction" -> Function[{st, instr},
					Do[
						AssociateTo[stmts,
							var -> DeleteDuplicates[Join[Lookup[stmts, var, {}], {instr}]]
						],
						{var, instr["getProperty", "live[in]"]}
					];
					Do[
						AssociateTo[stmts,
							var -> DeleteDuplicates[Join[Lookup[stmts, var, {}], {instr}]]
						],
						{var, instr["getProperty", "live[out]"]}
					];
				]
			|>,
			fm
		];
		Table[
			currentRange = {};
			Table[
				AppendTo[currentRange, instr],
				{instr, stmts[key]}
			];
			AppendTo[lv, makeRange[key, currentRange]]
			,
			{key, Keys[stmts]}
		];
		fm["setProperty", "liveIntervals" -> lv]
	]



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"LiveInterval",
	"A live interval is the span of instructions where a variable is alive.",
	"Live intervals are used to construct the interference graph to perform register allocation. " <>
	"This pass mainly aggregates information computed in LiveVariablesPass into a form to be consumed by " <>
	"the InterferenceGraphPass. The data is stored as a property of the function module."
];

LiveIntervalPass =  CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		LiveVariablesPass
	}
|>];

RegisterPass[LiveIntervalPass]
]]


End[]
EndPackage[]
