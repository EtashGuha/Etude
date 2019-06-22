Inputs:
	$Input: SequenceT[SizeT, AnyTensorT]
	$Position: TensorT[$$PosShape, IndexIntegerT[All]]

Output: AnyTensorT

Parameters:
	$$PosShape: SizeListT[]

ShapeFunction: ExtractShape

TypeFunction: Take[#, 1]&

AllowDynamicDimensions: True

Upgraders: {
	"12.0.5" ->
		DropParam["$InputLength"] /*
		Function[ (* bug 367798: change of representation for unbounded integers *)
			MapAt[
				ReplaceAll[IndexIntegerT[Infinity] -> IndexIntegerT[All]],
				#, {{"Inputs", "Position"}}]
		]
}

MaxArgCount: 0

PostInferenceFunction: Function @ If[
	ListQ[$$PosShape] && Length[$$PosShape] > 2,
	FailValidation["input to port Position cannot exceed rank 2."]
]

Writer: Function @ Scope[

	data = GetInput["Input"];
	pos = GetInput["Position"];

	inShape = GetInputDims["Input"];
	posShape = GetInputDims["Position"];
	posRank = Length[posShape];

	(* Handle case of scalar pos pos *)
	If[posRank === 0, 
		posShape = {1};
		posRank = 1;
		pos = SowReshape[pos, posShape];
	];

	inLenNode = GetDynamicLengthNode @ First[GetInputDims["Input"]];
	
	(* Clip position values up to max for each axis *)
	(* limitNode holds the maximum values*)
	If[inLenNode === None,
		(* In this case all sizes are fixed and we take them from
		   inhape *)
		limit = Take[inShape, {1, Last[posShape]}];
		limitNode = SowFixedArray["Limit", toNumericArray[limit]];
 		limitNode = Nest[SowInsertDim[#, 0]&, limitNode, posRank];
   		,
		If[Last[posShape] === 1,
			(* In this case we only need the variable sizes from inLenNode *)
			limitNode = SowInsertDim[inLenNode, 1]
			,
			(* In this case we must join the variable sizes from 
			   inLenNode with the fixed sized from inShape *)
			limit = Take[inShape, {2, Last[posShape]}];
			limitNode = SowBatchBroadcast @ SowFixedArray["Limit", toNumericArray[limit]];
 			limitNode = SowJoin[SowInsertDim[inLenNode, 1], limitNode, -1]		
		];
 		limitNode = Nest[SowInsertDim[#, 1]&, limitNode, posRank - 1]
   	];

   	(* Use modulo in order to match -1, -2, etc. to length, length-1, etc. *)
	pos = SowNode["broadcast_mod", {
		SowMinus[pos, SowRamp @ SowNode["sign", pos]], (* convert positive indices to 0-index *)
		limitNode
	}];

	(* Add batch indices to position pos *)
	range = SowUReshape[GetBatchIndexArray[], Prepend[CTable[1, posRank], 0]];
	If[posRank > 1,
		baxes = Range[posRank - 1];
		range = SowNode["broadcast_like", {range, pos},
			"lhs_axes" -> baxes, "rhs_axes" -> baxes
		];
	];
	pos = SowJoin[range, pos, -1];

	(* finally transpose and extract parts *)
	pos = SowTranspose[pos, RotateRight @ Range[0, posRank]];
	out = SowNode["gather_nd", {data, pos}]; 

	SetOutput["Output", out];
]

Tests: {
	(* Fixed-length *)

	{"Input" -> {5}, "Position" -> Hold @ Restricted["Integer", 5]} -> "_ZynO6newRQU_MS62CCOO6FQ=4.430442e-1",
	{"Input" -> {5, 2, 3}, "Position" -> Hold @ Restricted["Integer", 5]} -> "2*3_atJMCK9GbNE_Mn89DUWDZpw=3.320395e+0",
	{"Input" -> {5}, "Position" -> Hold @ Restricted["Integer", {-5, -1}]} -> "Specification Restricted[Integer, {-5, -1}] is not compatible with port \"Position\", which must be an array of integers.",
	{"Input" -> {5, 2, 3}, "Position" -> Hold @ Restricted["Integer", {-5, -1}]} -> "Specification Restricted[Integer, {-5, -1}] is not compatible with port \"Position\", which must be an array of integers.",
	{"Input" -> {3}, "Position" -> "Integer"} -> "_akQbzL2Wru4_I9oZHmP8HN4=1.119578e-1",
	{"Input" -> {3, 4}, "Position" -> "Integer"} -> "4_N6dBguLAOZ8_G34yBK7zG6w=1.674342e+0",
	{"Input" -> {3}, "Position" -> {}} -> "_akQbzL2Wru4_I9oZHmP8HN4=1.119578e-1",
	{"Input" -> {3}, "Position" -> {1}} -> "_akQbzL2Wru4_I9oZHmP8HN4=1.119578e-1",
	{"Input" -> {3}, "Position" -> {2, 1}} -> "2_MGw11w8jhSg_VG9H818WiKU=4.617735e-1",
	{"Input" -> {2, 3, 4}, "Position" -> {1}} -> "3*4_PaJdASXmEEg_DmXGzKrUl5M=5.231537e+0",
	{"Input" -> {2, 3, 4}, "Position" -> {5, 1}} -> "5*3*4_b8baJVsCOFc_PacgmCck40M=2.700870e+1",
	{"Input" -> {2, 3, 4}, "Position" -> {5, 2}} -> "5*4_b0G4+xdQK7w_cfyePly8QIg=9.342773e+0",
	{"Input" -> {2, 3, 4}, "Position" -> {5, 3}} -> "5_Xmfvi1BJUxE_B8jour/0kdc=2.770946e+0",
	{"Input" -> {2, 3, 4}, "Position" -> {5, 4}} -> "Type inconsistency in ExtractLayer: the final dimension 4 of the \"Position\" port must not exceed the rank 3 of the \"Input\" port.",
	{"Input" -> {2, 3, 4}, "Position" -> {5, 5, 3}} -> "Validation failed for ExtractLayer: input to port Position cannot exceed rank 2." ,
	(* Var-Length Input *)
	{"Input" -> {"x"}, "Position" -> "Integer"} -> "_RxyREFFKyFM_FkPp2KBBPl8=7.288513e-1",
	{"Input" -> {"x"}, "Position" -> {2, 1}} -> "2_dj932Ws77WA_b3uv++4frZk=1.457703e+0",
	{"Input" -> {"x", 3}, "Position" -> {5, 1}} -> "5*3_GhK/Ia9cqow_W6gP4PRm/EQ=6.019565e+0",
	{"Input" -> {"x", 3}, "Position" -> {5, 2}} -> "5_fEBoNnxVt2k_LjIbZ/2T0AA=2.358319e+0",
	{"Input" -> {"x", 3, Restricted["Integer", 10]}, "Position" -> {5, 2}} -> "5_YjkSXc7gUVs_FkxcQ0hjgT8=4.400000e+1",
	{"Input" -> {"x", 3}, "Position" -> {5, 3}} -> "Type inconsistency in ExtractLayer: the final dimension 3 of the \"Position\" port must not exceed the rank 2 of the \"Input\" port.",
	(* Var-Length Position *)
	{"Input" -> {10}, "Position" -> {"x", 1}} -> "10_NHfZyqDeXdk_UQzdJE2jqzM=3.560469e+0",
	{"Input" -> {2, 2}, "Position" -> {"x", 2}} -> "10_XQGTSAWkKts_KxL1XpO99A4=2.402096e+0",
	{"Input" -> {2, 2}, "Position" -> {"x", 3}} -> "Type inconsistency in ExtractLayer: the final dimension 3 of the \"Position\" port must not exceed the rank 2 of the \"Input\" port.",
	{"Input" -> {10}, "Position" -> {"x"}} -> "Type inconsistency in ExtractLayer: the final dimension of the \"Position\" port must be fixed.",
	(* Both Varying *)
	{"Input" -> {"y", 2}, "Position" -> {"x", 1}} -> "10*2_IsXfzAJmons_EETgxHXVikA=9.977647e+0",
	{"Input" -> {"y", 2}, "Position" -> {"x", 2}} -> "10_SNthgfCPxMw_FZAJN7BANxA=4.112255e+0",
	{"Input" -> {"y", 2}, "Position" -> {"x", 3}} -> "Type inconsistency in ExtractLayer: the final dimension 3 of the \"Position\" port must not exceed the rank 2 of the \"Input\" port.",
	(* Not possible *)
	{"Input" -> {}, "Position" -> {}} -> "Specification {} is not compatible with port \"Input\", which must be an array of rank \[GreaterEqual] 1."

}