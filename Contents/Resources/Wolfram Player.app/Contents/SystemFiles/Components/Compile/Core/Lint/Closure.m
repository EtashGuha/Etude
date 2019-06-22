BeginPackage["Compile`Core`Lint`Closure`"]

LintClosurePass

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]


(* private imports *)
Needs["Compile`Core`Lint`Utilities`"]



(* ::Subsection:: *)
(* Utilities *)


(* Get some internal helper functions from Compile`Core`Lint`Utilities` *)
toString := toString = Compile`Core`Lint`Utilities`Private`toString
toFailure := toFailure = Compile`Core`Lint`Utilities`Private`toFailure
printIfNotQuiet := printIfNotQuiet = Compile`Core`Lint`Utilities`Private`printIfNotQuiet
error := error = Compile`Core`Lint`Utilities`Private`error
warn := warn = Compile`Core`Lint`Utilities`Private`warn
(***********************************************************************)

(* ::Subsection:: *)
(* Error Message *)


errorInvalidClosureUsage[st_, inst_, ty_] :=
	error[
		"ClosureUsage",
		ty,
		TemplateApply[
			StringTemplate[
				"The type `var` was not resolved in `inst` make sure to run the ResolveTypesPass.",
				InsertionFunction -> toString
			],
			<|
				"type" -> ty,
				"inst" -> inst
			|>
		]
	]


(* ::Subsection:: *)
(* Lint Rules *)

visitPhiInstruction[_, inst_] :=
	(
		0
	);
	
run[fm_, opts_] :=
	Module[{},
		CreateInstructionVisitor[
			<|
				"visitPhiInstruction" -> visitPhiInstruction
			|>,
			fm
		];
		fm
	]
	

(* ::Subsection:: *)
(* Register *)


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"LintClosure",
	"Checks for common errors for the closure usage in the SSA IR. " <>
	"These are restrictions to make the implementation of V1.0 of the compiler simpler. " <>
	"Printout can be suppressed using Quiet."
];

LintClosurePass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"passClass" -> "Analysis"
|>];

RegisterPass[LintClosurePass]
]]



End[]

EndPackage[]
