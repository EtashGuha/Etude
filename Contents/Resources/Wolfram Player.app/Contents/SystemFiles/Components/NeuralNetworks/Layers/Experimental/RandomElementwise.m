Inputs: 
	$Multiport: RealTensorT

Output: RealTensorT

Parameters:
	$Distribution: ValidatedParameterT[toUnivariateExpression]

ShapeFunction: EqualityShape

RankFunction: EqualityRank

(* TODO: Support MixtureDistribution *)

PostConstructionFunction: Function @ Scope[
	count = FirstCase[$Distribution, UnivariateDistributionObject[_, n_] :> n, Infinity];
	PCFExpandMultiport[count]
]

toUnivariateExpression[f_] := Scope[
	count = FunctionArgumentCount[f];
	If[FailureQ[count],
		UnsafeQuietCheck[f[]; count = 0, 
		UnsafeQuietCheck[f[1]; count = 1,
		UnsafeQuietCheck[f[1,2]; count = 2,
		ReturnFailed[]]]];
	];
	res = toSymbolicDist @ Apply[f, Table[UParam[i], {i, count}]];
	If[FailureQ[res], ReturnFailed[]];
	UnivariateDistributionObject[res, count]
];

DefineCustomBoxes[UnivariateDistributionObject, UnivariateDistributionObject[dist_, _] :> MakeBoxes[Function[dist]]];
DefineCustomBoxes[UDist, UDist[dist_, args___] :> MakeBoxes[dist[args]]];
DefineCustomBoxes[UParam, UParam[n_] :> MakeBoxes[Slot[n]]];
DefineCustomBoxes[CUParam, CUParam[n_] :> MakeBoxes[n]];

Writer: Function @ Scope[
	dims = GetOutputDims["Output"];
	shape = Prepend[dims, 0];
	inputs = GetInput[All];
	out = write @@ #Distribution[[1, 1]];
	If[inputs === {}, 
		out = SowSourceFixup[out, dims];
	];
	SetOutput["Output", out];
]


toNode[i_] := inputs[[i]];

Clear[write];
SetHoldFirst[write];

SowSampleNormal[mean_Real, sd_Real] := SowNode["random_normal", {}, "loc" -> mean, "scale" -> sd];
SowSampleNormal[mean_MXNode, sd_MXNode] := SowNode["sample_normal", {mean, sd}];
write[NormalDistribution, CUParam[mean_], CUParam[sd_]] := SowSampleNormal[mean, sd];
write[NormalDistribution, mean_, sd_] := SowSampleNormal[asNode @ mean, asNode @ sd];

SowSampleUniform[a_Real, b_Real] :=  SowNode["random_uniform", {}, "low" -> a, "high" -> b];
SowSampleUniform[a_MXNode, b_MXNode] :=  SowNode["sample_uniform", {a, b}];
write[UniformDistribution, {CUParam[a_], CUParam[b_]}] := SowSampleUniform[a, b];
write[UniformDistribution, {a_, b_}] := SowSampleUniform[asNode @ a, asNode @ b];

SowSampleGamma[shape_Real, scale_Real] :=  SowNode["random_gamma", {}, "alpha" -> shape, "beta" -> scale];
SowSampleGamma[shape_MXNode, scale_MXNode] := SowNode["sample_gamma", {shape, scale}];
write[GammaDistribution, CUParam[shape_ ? Positive], CUParam[scale_ ? Positive]] := SowSampleGamma[shape, scale];
write[GammaDistribution, shape_, scale_] := SowSampleGamma[asNode @ shape, asNode @ scale];

SowSampleExponential[lambda_Real] := SowNode["random_exponential", {}, "lam" -> lambda];
SowSampleExponential[lambda_MXNode] := SowNode["sample_exponential", lambda];
write[ExponentialDistribution, CUParam[lambda_ ? Positive]] := SowSampleExponential[lambda];
write[ExponentialDistribution, lamba_] := SowSampleExponential[asNode @ lambda];

SowSamplePoisson[lambda_Real] := SowNode["random_poisson", {}, "lam" -> lambda];
SowSamplePoisson[lambda_MXNode] := SowNode["sample_poisson", lambda];
write[PoissonDistribution, CUParam[lambda_ ? Positive]] := SowSamplePoisson[lambda];
write[PoissonDistribution, lambda_] := SowSamplePoisson[asNode @ lambda];

SowSampleNegativeBinomial[k_Real, p_MXNode] := SowNode["random_negative_binomial", {}, "k" -> k, "p" -> p];
SowSampleNegativeBinomial[k_MXNode, p_MXNode] := SowNode["sample_negative_binomial", {k, p}];
write[NegativeBinomialDistribution, CUParam[k_ ? Positive], CUParam[p_ ? Positive]] := SowSampleNegativeBinomial[k, p];
write[NegativeBinomialDistribution, k_, p_] := SowSampleNegativeBinomial[asNode @ k, asNode @ p];

asNode[UParam[i_]] := toNode[i];
asNode[CUParam[n_]] := SowConstantArray[n, dims];

write[dist_, args___] := FailValidation[RandomElementwiseLayer, "specified use of distribution `` is not currently supported.", dist]; 

$ValidDistributions = Hold[UniformDistribution, NormalDistribution, GammaDistribution, ExponentialDistribution, PoissonDistribution, NegativeBinomialDistribution];

Clear[toSymbolicDist];
Scan[
	Function[
		distSymbol,
		toSymbolicDist[HoldPattern @ distSymbol[args___]] := UDist[distSymbol, MapSequence[procDistArg, args]],
		HoldAll
	],
	$ValidDistributions
];

toSymbolicDist[___] := $Failed;

procDistArg[e_List] := Map[procDistArg, e];
procDistArg[u_UParam] := u;
procDistArg[n_ ? NumericQ] := CUParam[N[n]];
procDistArg[_] := Return[$Failed, Block];
