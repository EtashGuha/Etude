Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]	


(******************************************************************************)

(* this is a el-cheapo way to communicate the learning rate
back to NetTrain for error messages and reporting. Hacky but 
the alternative is much more complex for little benefit *)

PackageExport["$LastInitialLearningRate"]
PackageExport["$LastOptimizerMethod"]
PackageExport["$LastGlobalLearningRate"]


(******************************************************************************)

$OptimizerMethodTable = <||>;

(* this table should have pairs of the form:
{{name, mxOperator, numStates, defaultLR, globalParams, mxParams}, userTypeSpec}

userTypeSpec is the spec given to ParseMethod 
mxParams is used in the ArrayUpdater

*)

$OptimizerMethodTable["ADAM"] = {
	{"ADAM", "adam_update", 2, 0.001, #, {"beta1" -> #Beta1, "beta2" -> #Beta2, "epsilon" -> #Epsilon}}&,
	<|
		"Beta1" -> {toUnitInterval, 0.9},
		"Beta2" -> {toUnitInterval, 0.999},
		"Epsilon" -> {toUnitInterval, 10^-5}
	|>
}

$OptimizerMethodTable["SGD"] = {
	If[#Momentum =!= None && #Momentum != 0, 
		{"SGD", "sgd_mom_update", 1, 0.01, #, {"momentum" -> #Momentum}},
		{"SGD", "sgd_update", 0, 0.01, #, {}}
	]&,
	<|
		"Momentum" -> {toUnitIntervalOrNone -> 0., 0.93},
		"LearningRateSchedule" -> {Identity -> None, "Polynomial"}
	|>
}

$OptimizerMethodTable["SignSGD"] = {
	If[#Momentum =!= None && #Momentum != 0, 
		{"SignSGD", "signum_update", 1, 0.01, #, {"momentum" -> #Momentum}},
		{"SignSGD", "signsgd_udpate", 0, 0.01, #, {}}
	]&,
	<|
		"Momentum" -> {toUnitIntervalOrNone -> 0., 0.93}
	|>
}

(* need to find way of putting 't' inside.
$OptimizerMethodTable["FTML"] = {
	{"FTML", "ftml_update", 3, 0.01, #, {"beta1" -> 0.5, "beta2" -> 1.0, "epsilon" -> 10^-5}},
	<||>
};
*)

$OptimizerMethodTable["RMSProp"] = {
	{"RMSProp", "rmspropalex_update", 3, 0.001, #, {"gamma1" -> #Beta, "gamma2" -> ReplaceAll[#Momentum, None -> 0.], "epsilon" -> #Epsilon}}&,
	<|
		"Beta" -> {toUnitInterval, 0.95},
		"Momentum" -> {toUnitIntervalOrNone -> 0., 0.9},
		"Epsilon" -> {toUnitInterval, 10^-6}
	|>
}

$OptimizerMethodTable["RMSProp"] = {
	{"RMSProp", "rmspropalex_update", 3, 0.001, #, {"gamma1" -> #Beta, "gamma2" -> ReplaceAll[#Momentum, None -> 0.], "epsilon" -> #Epsilon}}&,
	<|
		"Beta" -> {toUnitInterval, 0.95},
		"Momentum" -> {toUnitIntervalOrNone -> 0., 0.9},
		"Epsilon" -> {toUnitInterval, 10^-6}
	|>
}

$OptimizerMethodTable["$CommonParameters"] = <|
	"L2Regularization" -> {toNumberOrNoneVec -> 0., None},
	"GradientClipping" -> {toNumberOrNoneVec -> -1., None},
	"WeightClipping" -> {toNumberOrNoneVec -> -1., None},
	"LearningRate" -> {toNumber, Automatic},
	"LearningRateSchedule" -> {Identity, None}
|>

parseSpec[key_, spec_, checker_, default_] := 
	parseSpec[key, spec, checker -> default, default];

parseSpec[key_, spec_, checker_ -> noneVal_, default_] :=
	If[$resolver =!= None && RuleVectorQ[spec],
		$resolver[key, spec, checker, noneVal, default],
		Replace[checker @ Replace[spec, Automatic -> default], None -> noneVal]
	];

toNumber = MatchValues[
	n_ ? NumericQ := N[n];
	_ := $Failed["a number, or Automatic"]
];

toNumberOrNoneVec = MatchValues[
	n_ ? NumericQ := N[n];
	None := None;
	e_ := $Failed["a number, a list, or None"]
]

toUnitInterval := MatchValues[
	n_ ? NumericQ /; 0 <= n < 1 := N[n];
	e_ := $Failed["a number between 0 and 1"]
]

toUnitIntervalOrNone := MatchValues[
	n_?NumericQ /; 0 <= n < 1 := N[n];
	None := None;
	e_ := $Failed["a number between 0 and 1"]
]

(******************************************************************************)

PackageExport["CreateArrayOptimizer"]

SetUsage @ "
CreateArrayOptimizer[methodspec$, weights$, gradients$, opts$$] creates an ArrayOptimizer[$$] object to optimize weights$ via gradients$.
* methodspec$ is etiher a string like 'ADAM' or a spec like {'SGD', 'Momentum' -> 0.5}.
* The weights$ and gradients$ should be lists of NDArrays (or NDReplicaArrays and NDTotaledArrays respectively).
* The resulting ArrayOptimizer can be called like a function with no arguments to perform an update.
* The following options are supported:
| 'KVStore' | a KVStore[$$] object to bind to |
| 'GradientKeys' | a list of keys to use to interpret association specs for per-weight parameters |
| 'LearningRateMultipliers' | optional association of per-layer learning rates |
| 'GradientScale' | pre-multiply all gradients by this quantity |
| 'MaxIterations' | maximum value to pass to learning rate schedule function |
* If a KVStore[$$] is provided, the weights should be NDReplicaArray[$$] objects and the gradients should be NDTotaledArray[$$] objects."

Options[CreateArrayOptimizer] = {
	"KVStore" -> None,
	"GradientKeys" -> None,
	"LearningRateMultipliers" -> 1,
	"GradientScale" -> 1,
	"MaxIterations" -> 1000,
	"Resolver" -> None
}

CreateArrayOptimizer[method_, weights_List, gradients_List, opts:OptionsPattern[]] := ModuleScope @ Block[
	{$len, $gradientKeys},

	UnpackOptions[
		kVStore, $gradientKeys, learningRateMultipliers, gradientScale, maxIterations, resolver
	];

	Block[{$resolver = resolver},
	{baseMethod, opName, numStateArrays, defaultLR, allParams, mxParams} = 
		ParseMethod[method, $OptimizerMethodTable, parseSpec]];

	$LastOptimizerMethod ^= baseMethod;
	
	UnpackAssociation[allParams,
		learningRate, learningRateSchedule,
		gradientClipping, l2Regularization, weightClipping
	];

	schedule = parseSchedule[learningRateSchedule, maxIterations];

	$len = Length[weights];
	If[kVStore =!= None,
		arrays = ConstantArray[$NullNDArray, {2 + numStateArrays, $len}];
		initSpec = Range[0, $len-1] -> weights[[All, 1, 1]];
		MXKVStoreInit[kVStore, initSpec];
		(* ^ initialize kvstore with the first array from each NDReplicaArray.
		if we wanted to be more efficient we'd actually use a Pull to initialize
		those subarrays in the first place, but it doesn't matter much relative to training time *)
	,
		stateArrays = Table[NDArrayCloneShape[weights, 0.0], {numStateArrays}];
		arrays = Join[{weights, gradients}, stateArrays];
	];

	NDArraySetConstant[gradients, 0.0];

	If[baseMethod === "ADAM", 
		b1 = allParams["Beta1"]; b2 = allParams["Beta2"];
		b1 = First[b1, b1]; b2 = First[b2, b2];
		schedule = injectADAMCorrection[schedule, b1, b2];
	];

	SetAutomatic[learningRate, defaultLR];
	$LastInitialLearningRate ^= learningRate;

	learningRateMultipliers = listify[learningRateMultipliers];

	params = {
		"lr" -> learningRateMultipliers, 
		"wd" -> ReplaceAll[l2Regularization, None -> 0.],
		"clip_gradient" -> ReplaceAll[gradientClipping, None -> -1.0],
		"rescale_grad" -> (1.0 / gradientScale)
	};

	params = Association[params, mxParams];
	optimizer = ArrayUpdaterCreate[opName, arrays, listify /@ params];

	$time = 0;
	updateFunction = Function[
		$time++; optimizer; arrays; (* keep arrays and optimizer object alive via refcounts *)
		$LastGlobalLearningRate ^= learningRate * schedule[$time];
		ArrayUpdaterSetParamsColumn[optimizer, 0, learningRateMultipliers * $LastGlobalLearningRate];
		ArrayUpdaterApply[optimizer]
	];

	If[kVStore =!= None,
		ArrayUpdaterBindKVStore[optimizer, kVStore];
		{pushKeys, pushValues} = toPushPullSpecList[gradients];
		{pullKeys, pullValues} = toPushPullSpecList[weights];
		updateFunction = ReplacePart[
			updateFunction, 
			{1, -1} :> (
				MXKVStorePush[kVStore, pushKeys -> pushValues]; 
				MXKVStorePull[kVStore, pullKeys -> pullValues]
			)
		];
		(* ^ replace the ArrayUpdaterApply with calls to KV store, which will induce
		calls to ArrayUpdaterApply::KVUpdate *)
		Assert[FreeQ[updateFunction, ArrayUpdaterApply]];
	];

	If[weightClipping =!= None,
		If[kVStore =!= None, ThrowFailure["wcmultidev"]];
		(* because it would require changing this approach and its not worth supporting *)
		assoc = DeleteCases[None|-1.] @ AssociationThread[weights, listify @ weightClipping]; 
		If[assoc =!= <||>, 
			{weights2, bounds} = KeysValues[assoc];
			clipper = ArrayUpdaterCreate["clip", {weights2}, {"a_min" -> -bounds, "a_max" -> bounds}];
			updateFunction = Insert[
				updateFunction, 
				Unevaluated[ArrayUpdaterApply[clipper]], {1, -1}
			];
		];
	];

	ArrayOptimizer @ Association[
		"Method" -> baseMethod,
		"Operator" -> opName,
		"Updates" :> $time,
		"Arrays" -> weights,
		"Gradients" -> gradients,
		"StateArrays" -> arrays,
		"Gradients" -> gradients,
		"UpdateFunction" -> updateFunction
	]
]

toPushPullSpecList[list_] := Transpose[Join @@ MapIndexed[toPushPullSpec, list]]
toPushPullSpec[(NDTotaledArray|NDReplicaArray)[subs_], {i_}] := Thread[{i-1, subs}]

DeclarePostloadCode[
General::wcmultidev = "Weight clipping is not supported for multiple-device training."
]

(******************************************************************************)

listify[value_] := 
	ConstantArray[N @ value, $len]

listify[list_List] := 
	Developer`ToPackedArray @ N @ list

listify[assoc_Association] := 
	Developer`ToPackedArray @ N @ Lookup[assoc, $gradientKeys, Panic["MissingOptimizerAssocVals"]]

(* because for NetTrain we need to start with a dummy value of maxIterations and then fix it later
after we've timed one upate *)
unwrap[n_Integer] := n
unwrap[Hold[s_]] := s

checkUnitReal[r_] := If[Developer`MachineRealQ[r] && 0. < r <= 1., r, ThrowFailure["badlrs", r]]

injectADAMCorrection[HoldPattern @ Function[v_, body_], beta1_, beta2_] := 
	Function[v, body * Quiet[Sqrt[1. - beta2^v] / (1. - beta1^v)]]

parseSchedule[_Missing, _] := Function[iter, 1.0]

parseSchedule[None, _] := Function[iter, 1.0]

parseSchedule[Automatic|"Polynomial", maxiters_] := 
	Function[iter, Block[{max = unwrap[maxiters]},
		Re@Sqrt[1. - Min[iter, max]/(max + 1.)]
	]]

parseSchedule[f_, maxiters_] := 
	parseSchedule2[
		(* try upgrade from old convention *)
		If[Internal`UnsafeQuietCheck[!NumberQ[f[1, 100]] && NumberQ[f[1,100,1.0]], False], Function[f[#1,#2,1.0]], f], 
		maxiters
	]

parseSchedule2[f_, maxiters_] := Scope[ 
	checkUnitReal[N@f[1, 100]];
	Function[iter, Block[
		{max = unwrap[maxiters]},
		checkUnitReal @ N @ f[Min[iter-1, max], max]
	]]
]

DeclarePostloadCode[
General::badlrs = "Learning rate scheduler returned ``, which is not a real value between 0 and 1."
]