Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


If[TrueQ @ $SetupUpValues, 

(******************************************************************************)

MXSymbol /: Part[symbol_MXSymbol, index_] := MXSymbolOutputPart[symbol, index]

(******************************************************************************)

MXSymbol /: Plus[x_MXSymbol, y_MXSymbol] := MXNet`Plus[x, y]
MXSymbol /: Plus[x_MXSymbol, y_] := MXNet`PlusScalar[x, "scalar" -> y]
MXSymbol /: Plus[x_, y_MXSymbol] := Plus[y, x]

(******************************************************************************)

MXSymbol /: Subtract[x_MXSymbol, y_MXSymbol] := MXNet`Minus[x, y]
MXSymbol /: Subtract[x_MXSymbol, y_] := MXNet`MinusScalar[x, "scalar" -> y]
MXSymbol /: Subtract[x_, y_MXSymbol] := MXNet`RMinusScalar[y, "scalar" -> x]

(******************************************************************************)

MXSymbol /: Divide[x_MXSymbol, y_MXSymbol] := MXNet`Div[x, y]
MXSymbol /: Divide[x_MXSymbol, y_] := MXNet`DivScalar[x, "scalar" -> y]
MXSymbol /: Divide[x_, y_MXSymbol] := MXNet`RDivScalar[y, "scalar" -> x]

(******************************************************************************)

MXSymbol /: Times[x_MXSymbol, y_MXSymbol] := MXNet`Mul[x, y]
MXSymbol /: Times[x_MXSymbol, y_] := MXNet`MulScalar[x, "scalar" -> x]
MXSymbol /: Times[x_, y_MXSymbol] := Times[y, x]

(******************************************************************************)

MXSymbol /: Power[x_MXSymbol, y_MXSymbol] := MXNet`Power[x, y]
MXSymbol /: Power[x_MXSymbol, y_] := MXNet`PowerScalar[x, "scalar" -> y]
MXSymbol /: Power[x_, y_MXSymbol] := MXNet`RPowerScalar[y, "scalar" -> x]

(******************************************************************************)

MXSymbol /: Max[x_MXSymbol, y_MXSymbol] := MXNet`Maximum[x, y]
MXSymbol /: Max[x_MXSymbol, y_] := MXNet`MaximumScalar[x, "scalar" -> y]
MXSymbol /: Max[x_, y_MXSymbol] := Max[y, x]

(******************************************************************************)

MXSymbol /: Min[x_MXSymbol, y_MXSymbol] := MXNet`Minimum[x, y]
MXSymbol /: Min[x_MXSymbol, y_] := MXNet`MinimumScalar[x, "scalar" -> y]
MXSymbol /: Min[x_, y_MXSymbol] := Min[y, x]

(******************************************************************************)

MXSymbol /: Derivative[1][sym_MXSymbol] := MXGradSymbol[sym];

];

(******************************************************************************)

PackageExport["MXGradSymbol"]

MXGradSymbol[sym_MXSymbol][inputs__] := Scope[
	{exec, arrays} = bindWithInputs[sym, {inputs}];
	MXExecutorForward[exec, True];
	outGrad = NDArrayCloneShape[#, 1.0]& /@ Values[arrays["Outputs"]];
	MXExecutorBackward[exec, outGrad];
	Map[Normal, arrays["Gradients"]]
];

(******************************************************************************)

(* This is the subvalue that allows one to quickly and dirtily evaluate
an mxsymbol on a simple numeric input. There needs to be a numeric check here,
as well as an association input form for when there are multiple inputs, along
with a vararg form as well *)

bindWithInputs[sym_, inputs_List] := Scope[
	inames = Take[MXSymbolArguments[sym], Length[inputs]];
	iarrays = AssociationThread[inames, NDArrayCreate /@ inputs];
	MXSymbolBind[sym, iarrays]
];

sym_MXSymbol[inputs__] := Scope[
	exec = bindWithInputs[sym, {inputs}];
	MXExecutorForward[exec];
	Map[Normal, exec["OutputArrays"]]
];