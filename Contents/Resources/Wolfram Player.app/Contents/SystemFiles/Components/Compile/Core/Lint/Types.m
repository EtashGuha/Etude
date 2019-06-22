BeginPackage["Compile`Core`Lint`Types`"]

LintTypesPass

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`Variable`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`Lint`Utilities`"] (* For LintFailure *)


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

errorTypeUnresolved[st_, inst_, ty_] :=
	error[
		"TypeUnresolved",
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

needsResolve[_Type] := True
needsResolve[___] := False

isResolved[inst_, ty_] :=
	If[needsResolve[ty] === True,
		errorTypeUnresolved[Null, inst, ty]
	];

visitInstruction[_, inst_] :=
	(
		If[inst["definesVariableQ"],
	    		isResolved[inst, inst["definedVariable"]]
	    ];
	    If[inst["hasOperands"],
			isResolved[inst, #]& /@ Select[inst["operands"], ConstantValueQ[#] || VariableQ[#]&];
		];
	);
	
run[fm_, opts_] :=
	Module[{},
		CreateInstructionVisitor[
			<|
				"visitInstruction" -> visitInstruction
			|>,
			fm
		];
		fm
	]
	

(* ::Subsection:: *)
(* Register *)


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"LintTypes",
	"Checks for common errors for the types in the SSA IR. " <>
	"If errors or warning do occur, then they are printed. " <>
	"Printout can be suppressed using Quiet."
];

LintTypesPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"passClass" -> "Analysis"
|>];

RegisterPass[LintTypesPass]
]]



End[]

EndPackage[]
