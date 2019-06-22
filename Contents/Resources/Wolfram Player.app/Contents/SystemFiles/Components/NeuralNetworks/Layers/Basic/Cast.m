Input: AnyTensorT

Output: AnyTensorT

Parameters:
	$Type: EnumT[{"Integer", "Real"}]

TypeFunction: Function[{If[$Type ==="Integer", IndexIntegerT[All], RealT]}]

ShapeFunction: Identity

RankFunction: Identity

AllowDynamicDimensions: True

Upgraders: {
	"12.0.5" ->
		Function[ (* bug 367798: change of representation for unbounded integers *)
			MapAt[
				ReplaceAll[IndexIntegerT[Infinity] -> IndexIntegerT[All]], #,
				{{"Inputs", "Input"}, {"Outputs", "Output"}}
			]
		]
}

Writer: Function @ Scope[
	input = GetInput["Input"];
	If[#Type === "Integer",
		out = SowNode["round", input],
		out = input
	];
	SetOutput["Output", out]
]

Tests: {
	{"Real", "Input" -> {"Real"}} -> "_UlS0Sd7vNAo_N6dBguLAOZ8=3.498157e-1",
	{"Real", "Input" -> {2, "Real"}} -> "2_IMH4E1wGKP4_Liad5oMW3eI=7.928598e-1",
	{"Real", "Input" -> {2, 4, "Real"}} -> "2*4_FnrrjcbdMmI_QyAjiCVpwpg=3.109017e+0",
	{"Real", "Input" -> {2, 4, 5, "Real"}} -> "2*4*5_Z+TEggDS104_I6SP2qjvWEA=1.726360e+1",
	{"Real", "Input" -> {"Varying", "Real"}} -> "3_AIvAa0t3efE_fKqKMrZrk00=9.048177e-1",
	{"Real", "Input" -> {"Varying", 2, "Real"}} -> "3*2_IpNREqX6Prs_YG3AqZEo2ls=1.911142e+0",
	{"Real", "Input" -> {"Varying", 2, 4, "Real"}} -> "3*2*4_TpJLU17yRgY_AnBieG0TBcw=1.067583e+1",
	{"Real", "Input" -> {"Integer"}} -> "_KQ3bsMwIpN0_UW0NKEjwpUg=1.000000e+0",
	{"Real", "Input" -> {2, "Integer"}} -> "2_E73vzcvaDB4_XSd2R1Ic3n4=7.000000e+0",
	{"Real", "Input" -> {2, 4, "Integer"}} -> "2*4_LOzDEjjWmZo_Xi7Gxkii+1k=1.031000e+3",
	{"Real", "Input" -> {2, 4, 5, "Integer"}} -> "2*4*5_WHcQtVmOZVE_J4YEgpeQMHw=1.201000e+3",
	{"Real", "Input" -> {"Varying", "Integer"}} -> "3_UGQfaaJJFNI_OFMzlNITTPU=1.300000e+1",
	{"Real", "Input" -> {"Varying", 2, "Integer"}} -> "3*2_GcXQfNc7dWE_VlkAa9zNDRA=1.028000e+3",
	{"Real", "Input" -> {"Varying", 2, 4, "Integer"}} -> "3*2*4_N1cn37/vFSU_PfNa3uyo5Os=1.104000e+3"
}


