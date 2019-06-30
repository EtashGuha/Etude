Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["LayerGradientNumerical"]

SetUsage @ "
LayerGradientNumerical[func$, input$, varKey$, outGrad$] returns the input gradient \
given the output gradient outGrad$, a function func$ which takes an association of inputs input$, \
the key of the parameter that will be varied varKey$, and the part of the input association. \
Input arrays can be NDArrays, RawArrays, or normal WL arrays. Output is always a normal WL array.
"

Options[LayerGradientNumerical] = {
	"Epsilon" -> 10^-5
}

LayerGradientNumerical[f_, input_Association, varKey_String, outGrad_, opts:OptionsPattern[]] := Scope[
	(* convert NDArrays to normal form *)
	inputNormal = ArrayNormal /@ input;
	outGradNormal = ArrayNormal @ outGrad;
	
	$eps = OptionValue@"Epsilon";
	
	(* We will assume a single output *)
	out = f@inputNormal;
	$dimIn = Dimensions@inputNormal[varKey];
	$dimOut = Dimensions@out;
	
	indices = Tuples[Range/@ $dimIn];
	totalDer = singleGrad[f, inputNormal, varKey, #]& /@ indices;
	totalDer = ArrayReshape[totalDer, Join[$dimIn, $dimOut]];
	prod = TensorProduct[totalDer, outGradNormal];
	
	contractRange = Range@Length@$dimOut + Length@$dimIn;
	contract = Transpose[{contractRange, contractRange + Length@$dimOut}];
	TensorContract[prod, contract]
]

singleGrad[f_, input_Association, key_String, index_] := Scope[
	maskEx = mask[index];
	(f@Join[input, <|key -> input[key] + maskEx|>] - f@Join[input, <|key -> input[key] - maskEx|>])/(2*$eps)
]

mask[index_] := Scope[
	a = ConstantArray[0, $dimIn];
	a[[Sequence@@index]] = $eps;
	a
]

(******************************************************************************)

PackageExport["TestOpRandom"]

SetUsage @ "
TestOpRandom['name', idims$1, $$, 'param$1' -> val$1, $$] creates a single-op executor \
and then feeds it random data, calls MXExecutorForward, and then returns the result and the \
input data together in a single association.
* idims$i are dimension lists of the successive inputs. 
* It is assumed there is precisely one output.
* This is for debugging purposes and playing around with single layers.
* See MakeOpExecutor.
"

TestOpRandom[name_String, idims___List, params___Rule] := Scope[
	exec = MakeOpExecutor[name, idims, params];
	If[FailureQ[exec], Return[exec]];
	inArrays = exec["InputArrays"];
	inNames = Keys[inArrays];
	inData = RandomReal[1, #]& /@ AssociationThread[inNames -> {idims}];
	NDArraySet[inArrays, inData];
	MXExecutorForward[exec, False];
	outData = Map[NDArrayGet, exec["OutputArrays"]];
	Join[inData, outData]
];

(******************************************************************************)

PackageExport["MXOpBackward"]

SetUsage @ "
MXOpBackward['name', idims$1, $$, 'param$1' -> val$1, $$] creates a single-op executor \
and then feeds it random data, calls MXExecutorForward, and then returns the result and the \
input data together in a single association.
* idims$i are dimension lists of the successive inputs. 
* It is assumed there is precisely one output.
* This is for debugging purposes and playing around with single layers.
* See MakeOpExecutor.
"

MXOpBackward[name_String, idata___List, params___Rule] := Scope[
	params2 = Map[hyperstr, <|params|>];
	method = Lookup[params2, Method, "Exact"];
	KeyDropFrom[params2, Method];
	eps = Lookup[params2, "Epsilon", 10^-5];
	KeyDropFrom[params2, "Epsilon"];

	idims = Dimensions /@ {idata};
	nargs = Length@idims;
	exec = MakeOpExecutor[name, Sequence @@ idims, Sequence @@ Normal[params2]];
	If[FailureQ[exec], Return[exec]];

	outND = First @ exec["OutputArrays"];
	context = NDArrayContext[outND];
	outDim = Dimensions[NDArrayGet[outND]];
	outgradND = NDArrayCreate[tensorIntervalGenerate[outDim], context];

	(* Fill in weights etc *)
	argArrays = exec["ArgumentArrays"];
	Map[
		NDArraySet[#, tensorIntervalGenerate[NDArrayDimensions[#]]]&,
		argArrays
	];
	inArrays = exec["InputArrays"];
	inNames = Keys[inArrays];
	inDataAssoc = AssociationThread[inNames -> {idata}];
	NDArraySet[inArrays, inDataAssoc];

	(* find int arrays *)
	realArrays = (Floor[#] =!= Ceiling[#])& /@ {idata};
	realArrays = Join[realArrays, ConstantArray[True, Length@argArrays]];

	If[method === "Exact",
		exactGradCalc[exec, outgradND, realArrays],
		numericGradCalc[exec, outgradND, realArrays]
	]
];

exactGradCalc[exec_, outgrad_NDArray, realArrays_] := Scope[
	MXExecutorForward[exec, True];
	MXExecutorBackward[exec, {outgrad}];
	gradArrays = exec["GradientArrays"];
	gradArrays = KeySelect[gradArrays, AssociationThread[Keys@gradArrays -> realArrays]];
	Map[NDArrayGet, gradArrays]
]

numericGradCalc[exec_, outgrad_NDArray, realArrays_List] := Scope[
	f = executorEval[exec];
	in = NDArrayGet /@ Join[exec["InputArrays"], exec["ArgumentArrays"]];
	gradNames = Pick[Keys@exec["GradientArrays"], realArrays];
	argNames = Pick[Keys@in, realArrays];
	grads = LayerGradientNumerical[f, in, #, outgrad]& /@ argNames;
	AssociationThread[gradNames -> grads]
]

executorEval[exec_][input_Association] := Scope[
	args = Join[exec["InputArrays"], exec["ArgumentArrays"]];
	NDArraySet[args, input];
	MXExecutorForward[exec, True];
	NDArrayGet[First @ exec["OutputArrays"]]
]

tensorIntervalGenerate[dim_] := N@ArrayReshape[Subdivide[-1, 1, Times @@ dim], dim]

(******************************************************************************)

PackageExport["MakeOpExecutor"]

SetUsage @ "
MakeOpExecutor['name', idims$1, $$, 'param$1' -> val$1, $$] creates a single-op executor.
* idims$i are dimension lists of the successive inputs. 
* It is assumed there is precisely one output.
* An MXExecutorData[$$] is returned.
* This is for debugging purposes and playing around with single layers.
* See TestOpRandom.
"

MakeOpExecutor[name_String, idims___List, params___Rule] := Scope[
	ninputs = Length[{idims}];
	inNames = Table["in" <> IntegerString[i], {i, ninputs}];
	nodes = <|"op" -> "null", "name" -> #, "param" -> <||>, "inputs" -> {}|>& /@ inNames;
	inputs = Range[ninputs] - 1;
	params2 = Map[hyperstr, <|params|>];
	device = Lookup[params2, TargetDevice, "CPU"];
	KeyDropFrom[params2, TargetDevice];
	AppendTo[nodes, 
		<|"op" -> name, "name" -> "layer", "param" -> params2, "inputs" -> Thread[{inputs, 0, 0}]|>
	];
	assoc = <|
		"nodes" -> nodes,
		"arg_nodes" -> inputs,
		"heads" -> {{ninputs, 0}}
	|>;
	res = Quiet @ CatchFailure @ MXSymbolFromJSON[assoc];
	If[FailureQ[res], Message[MakeOpExecutor::mxerr, MXGetLastError[]]; Return[$Failed, Block]];
	exec = MXSymbolBind[res, 
		AssociationThread[inNames -> {idims}], 
		"Context" -> device
	]
];

MakeOpExecutor::mxerr = "MXNet complained: ``.";
MakeOpExecutor::badparam = "`` is a bad parameter value."

hyperstr[e_] := Match[e,
	_String :> e,
	True :> "true",
	False :> "false",
	i_Integer :> intString[i],
	i:{__Integer} :> writeIntList[i],
	r_Real /; Developer`MachineRealQ[r] && Abs[r] < 3.402823466*^38 :> HighPrecisionDoubleString[r], 
	Message[MakeOpExecutor::badparam, e]; 
	Return[$Failed, Block];
];

writeIntList[{a_, b_}] := StringJoin["(", intString[a], ", ", intString[b], ")"];
writeIntList[{a_, b_, c_}] := StringJoin["(", intString[a], ", ", intString[b], ", ",  intString[c], ")"];
writeIntList[list_] := StringRiffle[list, {"(", ", ", ")"}];
intString[i_Integer] := If[Negative[i], "-" <> IntegerString[i], IntegerString[i]];
intString[_] := $Unreachable;


