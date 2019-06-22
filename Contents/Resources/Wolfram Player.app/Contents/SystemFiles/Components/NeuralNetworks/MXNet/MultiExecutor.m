Package["NeuralNetworks`"]


PackageExport["MultiExecutor"]

MultiExecutor /: Normal[MultiExecutor[exec_]] := exec;

MultiExecutor[exec_][sub___] := exec[sub];


PackageScope["MultiExecutorCreate"]

divideBatchSize[virtualBatchSize_, n_] := Scope[
	Which[
		Divisible[virtualBatchSize, n], 
			batchSizesList = CTable[virtualBatchSize / n, n],
		virtualBatchSize < n,  
			n = virtualBatchSize;
			batchSizesList = CTable[1, n],
		True,
			b1 = Floor[virtualBatchSize / n];
			b2 = virtualBatchSize - b1*n;
			batchSizesList = CTable[b1, n];
			batchSizesList[[;;b2]]++;
	];
	batchSizesList
]

(* TODO: we will need to use several memory limits here, one per device. or 
when calcaluating the memory limit in the first place, using the Min across 
the devices *)
MultiExecutorCreate[contextsList_List, virtualBatchSize_Integer, args___] := Scope[
	n = Length[contextsList];
	batchSizesList = divideBatchSize[virtualBatchSize, n];
	If[n === 1, Return @ NetExecutorCreate[First @ contextsList, virtualBatchSize, args]];
	executors = Table[
		NetExecutorCreate[contextsList[[i]], batchSizesList[[i]], args],
		{i, n}
	];
	firstAuxArrays = executors[[1, 1, "MXAuxArrays"]];
	(* ^ the philosophy here is to just use the first as a representative,
	see https://github.com/tensorflow/tensorflow/issues/7439 for discussion *)
	executorArrays = AssociationTranspose @ executors[[All, 1, "Arrays"]];
	gluedArrays =  MapIndexed[glueArraysAssoc, executorArrays];
	MultiExecutor @ Association[
		"Executors" -> executors,
		"Arrays" -> gluedArrays,
		"ContextsList" -> Take[contextsList, n],
		"BatchSize" -> virtualBatchSize,
		"BatchSizesList" -> batchSizesList,
		"MXAuxArrays" -> firstAuxArrays
	]
];

Clear[glueArraysAssoc];
glueArraysAssoc[{None...}, _] := None;
glueArraysAssoc[assocs:{___Association}, {Key[k_]}] := 
	Map[
		Replace[k, $arrayGluers], 
		AssociationTranspose[assocs]
	];

$arrayGluers = {
	"Weights" -> NDReplicaArray,
	"WeightGradients" -> toNDTotaledArray,
	_ -> (NDCatenatedArray[#, batchSizesList]&)
};

toNDTotaledArray[nd_List] := 
	If[First[nd] === $NullNDArray, None, NDTotaledArray[nd]];


(******************************************************************************)
(* API functions shared with NetExecutor                                      *)
(******************************************************************************)

NetExecutorForward[MultiExecutor[assoc_], trainMode_:False] := 
	Scan[NetExecutorForward[#, trainMode]&, assoc["Executors"]];

NetExecutorBackward[MultiExecutor[assoc_]] := 
	Scan[NetExecutorBackward, assoc["Executors"]];

toBatchReplaceSpec[exec_] := Thread @ List[exec["BatchedKeys"], 1];

NetExecutorInherit[symbol_, MultiExecutor[assoc_], dims_] := Scope[
	batchSizePos = toBatchReplaceSpec @ First @ assoc["Executors"];
	MultiExecutorMap[
		NetExecutorInherit[symbol, #1, ReplacePart[dims, batchSizePos -> #2]]&,
		assoc, 
		assoc["BatchSizesList"]
	]
]

NetExecutorReshape[MultiExecutor[assoc_], shape_] := Scope[
	If[AssociationQ[shape],
		virtualBatchSize = shape[[1, 1]],
		virtualBatchSize = shape
	];
	UnpackAssociation[assoc, batchSizesList, executors];
	n = Length[executors];
	If[virtualBatchSize === 1,
		Return @ NetExecutorReshape[First @ executors, shape]];
	If[virtualBatchSize < n, 
		n = virtualBatchSize;
		assoc = MapAt[Take[#, n]&, assoc, Thread @ List @ {"BatchSizesList", "ContextsList", "Executors"}]
	];
	batchSizesList = divideBatchSize[virtualBatchSize, n];
	MultiExecutorMap[
		If[AssociationQ[shape],
			batchSizePos = toBatchReplaceSpec @ First @ executors;
			NetExecutorReshape[#1, ReplacePart[shape, batchSizePos -> #2]]&,
			NetExecutorReshape[#1, #2]&
		],
		assoc,
		batchSizesList
	]
];


NetExecutorMemoryInformation[MultiExecutor[<|"Executors" -> execs:{_NetExecutor, __}, __|>]] := Scope[
	Map[NetExecutorMemoryInformation, execs]
];

MultiExecutorMap[f_, assoc_Association, args___] := Scope[
	UnpackAssociation[assoc, executors, arrays, contextsList, batchSizesList];
	executors = MapThread[f, {executors, args}];
	batchSizesList = #["BatchSize"]& /@ executors;
	firstAuxArrays = executors[[1, 1, "MXAuxArrays"]];
	executorArrays = AssociationTranspose @ executors[[All, 1, "Arrays"]];
	gluedArrays =  MapIndexed[glueArraysAssoc, executorArrays];
	MultiExecutor @ Association[
		"Executors" -> executors,
		"Arrays" -> gluedArrays,
		"ContextsList" -> contextsList,
		"BatchSize" -> Total[batchSizesList],
		"BatchSizesList" -> batchSizesList,
		"MXAuxArrays" -> firstAuxArrays
	]
];