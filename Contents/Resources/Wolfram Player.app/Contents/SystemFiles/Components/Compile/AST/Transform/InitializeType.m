BeginPackage["Compile`AST`Transform`InitializeType`"]

MExprInitializeType;
MExprInitializeTypePass;

Begin["`Private`"]

Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`PassManager`MExprPass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["CompileAST`Class`Normal`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileAST`Class`MExprAtomQ`"]



initializeType[mexpr_, opts___] :=
	initializeTypeWork[mexpr]

initializeTypeWork[mexpr_?MExprAtomQ] := (
	mexpr["setType", Undefined];
	mexpr
)

initializeTypeWork[mexpr_?MExprNormalQ] := (
	mexpr["setType", Undefined];
	initializeTypeWork[mexpr["head"]];
	initializeTypeWork[#]& /@ mexpr["arguments"];
	mexpr
)

initializeTypeWork[mexpr_] :=
	ThrowException[{"Unknown arguments for MExprInitializeType", mexpr}]




RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"MExprInitializeType",
	"Initalizes the type field for all parts of the MExpr tree."
];

MExprInitializeTypePass = CreateMExprPass[<|
	"information" -> info,
	"runPass" -> initializeType
|>];

RegisterPass[MExprInitializeTypePass]
]]


End[]
EndPackage[]
