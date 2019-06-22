Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

(* Logarithmic Memory Cost planner
Based on: https://arxiv.org/abs/1604.06174
https://github.com/dmlc/mxnet-memonger
*)

isParam[name_String] := Which[
	StringContainsQ[name, "data", IgnoreCase -> True],
		False,
	StringContainsQ[name, 
		"bias" | "biases" | "weight" | "weights" | "beta" | "gamma", IgnoreCase -> True],
		True,
	True, (* default *)
		False
]

PackageScope["makeMirrorPlan"]

makeMirrorPlan[symbol_MXSymbol, thresh_Integer, inShapes_Association] := Scope[
	threshold = BitShiftLeft[thresh, 20];
	sym = MXSymbolCopy[symbol];
	internals = MXSymbolGetInternals[sym];
	outShapes = MXSymbolInferShape[internals, inShapes]["OutputArrays"];
	(* put outShapesinto form {{key1, val1}, ....} *)
	outShapes = List@@#& /@ Normal@outShapes;
	{totalSize, paramSize, localSize, saveSize, maxSize, lastLocal}  = ConstantArray[0, 6];
	lastSb = None;
	period = 1;
	lastStage = "";
	stageDecision = "";
	Do[
		sb = internals[[idx]];
		{name, shape} = outShapes[[idx]];
		If[isParam[name], 
			paramSize += Times @@ shape;
			Continue[];
			,
			totalSize += 4 * Times @@ shape;
			localSize += 4 * Times @@ shape;
			MXSymbolSetAttribute[sb, "force_mirroring", "True"];
		];

		stage = MXSymbolGetAttribute[sb, "mirror_stage"];
		If[stage =!= None,
			Which[
				(stage === "True") || (stage =!= lastStage),
					If[localSize > threshold,
						saveSize += 4 * Times @@ shape;
						maxSize = Max[maxSize, localSize];
						localSize = 0;
						stageDecision = "False";
						MXSymbolSetAttribute[sb, "force_mirroring", stageDecision];
						,
						stageDecision = "True"
					];
					lastStage = stage,
				stage === lastStage && stageDecision === "False",
					saveSize += 4 * Times @@ shape;
					MXSymbolSetAttribute[sb, "force_mirroring", stageDecision];
			]
		]
		,	
		{idx, Length@outShapes}
	];
	{sym, BitShiftRight[maxSize, 20], BitShiftRight[saveSize, 20]}
]

getCost[sym_MXSymbol, shapes_] := Scope[
	exec = MXSymbolBind[sym, shapes];
	MXExecutorRequiredMemory[exec]
]

PackageExport["SearchPlan"]

SetUsage["
SearchPlan[symbol$, ntrial$, inShapes$] takes in a symbol$, the number of trials ntrial$, \
and the input shapes to the symbol inShapes$. Returns an association of the form \
<|'NewCost' -> ..., 'NewSymbol' -> ..., 'OldCost' -> ..., 'Threshold' -> ...|>, \
which contains the cost and symbol of the most memory efficient configuration."
];

SearchPlan[symbol_MXSymbol, ntrial_Integer, inShapes_Association] := Scope[
	history = {};
	threshold = 0;
	minThreshold = None;
	minCost = None;
	nbegin = 3;
	Do[
		{sym, localSize, saveSize} = makeMirrorPlan[symbol, threshold, inShapes];
		cost = getCost[sym, inShapes];
		guess = Floor@Sqrt[saveSize * localSize / 2];
		If[minCost === None || minCost > cost,
			minCost = cost
		];
		If[minThreshold === None || localSize < minThreshold,
			minThreshold = localSize
		];
		AppendTo[history, {cost, threshold, sym}];
		threshold = guess;
	,
	{k, nbegin}	
	];
    maxThreshold = threshold * Sqrt[2];
    step = Floor[(maxThreshold - minThreshold) / ntrial];
    threshold = minThreshold + step;
    
	If[step > 0,
		Do[
			sym = First@makeMirrorPlan[symbol, threshold, inShapes];
			cost = getCost[sym, inShapes];
			AppendTo[history, {cost, threshold, sym}];
			threshold += step;

			,
			{k, ntrial}		
		]
	];
	(* Return best symbol *)
	history = First@MinimalBy[history, First];
	best = AssociationThread[{"NewCost", "Threshold", "NewSymbol"}, history];
	best["OldCost"] = getCost[symbol, inShapes];
	best
]