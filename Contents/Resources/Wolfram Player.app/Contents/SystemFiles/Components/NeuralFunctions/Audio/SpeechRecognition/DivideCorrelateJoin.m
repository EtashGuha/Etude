(*partition input tensor in chunks of given length with overlaping
as input windows is 10ms, recommended values are
length = 1000 (10 sec)
overlapIn = 100 (1 sec)
*)
PackageExport["partitionFeatures"]

partitionFeatures[feat_, length_, overlapIn_] :=
Block[
	{part},

	part = Partition[feat, UpTo[length], length - overlapIn, {1, overlapIn}, {}];
	If[Length[Last[part]] <= overlapIn,
		Most[part]
		,
		part
	]
]

(* compute the output length given the input sequence one*)
netOutLength[din_] := (1 + Floor[din/3])
(* Floor[((Floor[(din + 2 * 5 - (11 - 1)) / 3] + 1) + 2 * 5 - (11 - 1)) / 1] *)

(*join processed chunks in one single output*)
PackageExport["joinProcessedChunks"]

joinProcessedChunks[matrixList_, overlap_, transitionFunction_:Identity] :=
Block[
	{mask},
	Fold[mergeWithMask[#1, #2, overlap, transitionFunction]&, matrixList]
];

(* merge two successive chunks using transfun to weigth overlaping parts combination*)
mergeWithMask[mat1_, mat2_, overlap_, transfun_] :=
Block[
	{offset, mask, dim},
	offset = offSetFind[mat1[[- overlap ;; -1, 1 ;; 28]], mat2[[1 ;; overlap, 1 ;; 28]]];
	o = Floor[(1.5 + Sign[offset]*0.5)*offset];
	dim = Last[Dimensions[mat1]];
	mask = Array[
		ConstantArray[
			transfun[#],
			dim
		]&,
		overlap + o,
		{0., 1.}
	];
	Join[
		mat1[[1 ;; - (overlap + o) - 1]],
		mat1[[- (overlap + o) ;; - 1]]*(1. - mask) + mat2[[1 ;; overlap + o]]*mask,
		mat2[[overlap + o + 1 ;; - 1]]
	]
];

(*maximum correlation between the central third of overlaping parts is used to
evaluate best -4 < offset < 4 between successive chunks*)
offSetFind[left_, right_ ]:=
Block[
	{window, offset},
	(*change Ceiling[(Length[left])/3];*)
	window = Ceiling[(Length[left])/3];
	offset = MaximalBy[
		ReplacePart[
			#,
			(*there's a better way, look a statistics*)
			Ceiling[Length[#]/2] -> {0, #[[Ceiling[Length[#]/2], 2]] + 10.^-3}
		]&[
			Table[
				{
					offset,
					(*look at the components*)
					Mean@Flatten[left[[window + 1 ;; 2*window]]*right[[window + 1 + offset ;; 2*window + offset]]]
				},
				{offset, - Min[4, window - 1], Min[4, window - 1]}
			]
		],
		Last
	];
	MinimalBy[offset, Abs@*First][[1, 1]]
]

(* transition function to cumpute mask :
strength controls the transition speed.
strength = 0.5 is linear (default)
strength = 0.2 seems the best in KL div test
*)
PackageExport["trans"]

trans[x_, strength_] := If[x != 1, If[x > 0., .5*(Tanh[invLogSigm[x]*strength] + 1.), 0.], 1.]
invLogSigm[x_] := Log[- x/(x - 1)];