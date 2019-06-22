(*******************************************************************************

barnes-hut tsne

*******************************************************************************)

Package["NumericArrayUtilities`"]

PackageImport["GeneralUtilities`"]

(******************************************************************************)
DeclareLibraryFunction[barneshutTSNE, "barnes_hut_tsne_train", 
	{
		{"NumericArray", "Constant"},	(* input data *)
		Integer,					(* target_dims *)
		{"NumericArray", "Constant"},   (* distanceMatrix *)
		Real, 						(* perplexity *)
		Real,						(* theta *)
		Integer,					(* num_threads *)
		Integer,					(* max_iter *)
		Real,						(* tolerance *)
		Integer,					(* random_seed *)
		{"NumericArray", "Constant"},   (* reduced data (output) *)
		{"NumericArray", "Constant"},    (* beta (output)*)	
		{"NumericArray", "Constant"}    (* history (optional output)*)		
	}, 
	"Void"						
]

(******************************************************************************)
PackageExport["BarnesHutTSNE"]

SetUsage[BarnesHutTSNE, "
BarnesHutTSNE[input$, dims$, perp$, angle$] reduces the dimension of an input matrix of size {num examples, feature size} \
to {num examples, dims$} using perplexity perp$ and Barnes-Hut angle angle$. 

Options:
| \"ThreadNumber\" | Automatic | Number of parallel threads |
| \"Randomness\" | True | Whether to seed the internal random generator with a random value |
| \"Iterations\" | Automatic | Number of optimization steps |
| \"ReturnHistory\" | False | Return training history along with result |

Option \"Randomness\" is only meant to be used by NAU tests. Proper freezing of randomness in DimensionReduce is handled in \
MachineLearning and works correctly only if the option is set to True. 
Option \"Iterations\" is also there for testing purposes only. Proper stopping criteria are sorted out using \
$PerformanceGoal (set by DimensionReduce)
When Option \"ReturnHistory\" is set to True, the final association will contain an additional key, \"History\", containing the \
embedding, gradients, training updates and loss value at every iteration (excluding the final one)."]

Options[BarnesHutTSNE] = {
	"ThreadNumber" -> Automatic,
	"Randomness" -> True,
	"Iterations" -> Automatic,
	"ReturnHistory" -> False 	
}

BarnesHutTSNE[input_, targetDims_, perplexity_, angle_, OptionsPattern[]] := Scope[

	UnpackOptions[threadNumber, randomness, iterations, returnHistory];
	perfGoal = Developer`ToList@$PerformanceGoal;

	(* will never happen if called from DimensionReduce *)
	If[perplexity <= 0, 
		Panic["BadPerplexity", "Perplexity must be a positive machine-sized real."]];
	If[!Internal`PositiveMachineIntegerQ[targetDims], 
		Panic["BadDims", "Final dimension must be a positive machine integer."]];

	maxSteps = iterations;
	tolerance = 0;
	If[maxSteps === Automatic,
		Switch[perfGoal,
			{Automatic},
				tolerance = 0.0001; maxSteps = 1000,
			{___, "Quality", ___},
				tolerance = 0.00001; maxSteps = 2000,
			{___, "Speed", ___},
				tolerance = 0.0001; maxSteps = 800,
			_,
				tolerance = 0.0001; maxSteps = 1000
		]
	];
	If[threadNumber  === Automatic, threadNumber = $ProcessorCount];

	If[TrueQ@randomness,
		seed = RandomInteger[2^31 - 1],
		seed = 1
	];

	distanceMatrix = {};
	history = {};
	If[Length[input]*Length[input]*8 < $SystemMemory/3 && !MemberQ[perfGoal, "Memory"],
		distanceMatrix = DistanceMatrix[input, DistanceFunction -> SquaredEuclideanDistance, PerformanceGoal -> "Speed"];
	]; 
	If[TrueQ@returnHistory, 
		history = ConstantArray[-1., (3*Length[input]*targetDims + 1) * maxSteps]
	];

	inputNumeric = NumericArray[input, "Real32"];
	reducedDataNumeric = NumericArray[ConstantArray[0, {Length@input, targetDims}], "Real32"];
	betasNumeric = NumericArray[ConstantArray[0., {Length@input}], "Real32"];
	distanceMatrixNumeric = NumericArray[distanceMatrix, "Real32"];
	historyNumeric = NumericArray[history, "Real32"];

	distanceMatrix = Null;

	(* Writes onto reducedDataNumeric, betasNumeric, possibily historyNumeric*)
	return = barneshutTSNE[inputNumeric, targetDims, distanceMatrixNumeric, perplexity, angle, threadNumber, maxSteps, 
		tolerance, seed, reducedDataNumeric, betasNumeric, historyNumeric];

	If[Head@return === LibraryFunctionError, Return@$Failed];

	output = <|
		"ReducedData" -> Normal@reducedDataNumeric,
		"Perplexity" -> perplexity,
		"Betas" -> Normal@betasNumeric
	|>;

	If[TrueQ@returnHistory,
		history = TakeList[Normal[historyNumeric], Append[Table[Length[input]*targetDims*maxSteps, 3], maxSteps]];
		history = <|"History" -> <|
			"Embedding" -> ArrayReshape[history[[1]], {maxSteps, Length[input], targetDims}],
			"Gradients" -> ArrayReshape[history[[2]], {maxSteps, Length[input], targetDims}],
			"Updates"   -> ArrayReshape[history[[3]], {maxSteps, Length[input], targetDims}],
			"Loss"      -> history[[4]]
		|>|>;
		Join[output, history],
		output
	]
]

PackageExport["TSNELoss"]

SetUsage[TSNELoss, "
TSNELoss[inputData$, reductionOutput$] evaluates the t-SNE loss. inputData$ is the plain input data, reductionOutput$ is the \
output of BarnesHutTSNE[inputData$, ...].
"
]

TSNELoss[inputData_, reductionOutput_] := Scope[
	dX = DistanceMatrix[inputData, DistanceFunction -> SquaredEuclideanDistance, PerformanceGoal -> "Speed"];
	conditionalsX = Exp[-dX*reductionOutput["Betas"]];
	conditionalsX = conditionalsX - DiagonalMatrix[Diagonal@conditionalsX];
	conditionalsX = conditionalsX / Total[conditionalsX, {2}];
	jointX = (conditionalsX + Transpose[conditionalsX])/(2*Length[conditionalsX]);

	dY = DistanceMatrix[reductionOutput["ReducedData"], DistanceFunction -> SquaredEuclideanDistance, PerformanceGoal -> "Speed"];
	jointY = 1/(1 + dY);
	jointY = jointY - DiagonalMatrix[Diagonal@jointY];
	jointY = jointY / Total[jointY, 2];

	Total[jointX * Log[(jointX + $MinMachineNumber) / (jointY + $MinMachineNumber)], 2]
]