InheritsFrom: "SequenceMost"

Writer: Function @ Scope[
	input = GetInputMetaNode["Input"];
	rest = SowMetaDrop[input, #$OutputLength, True];
	SetOutput["Output", rest];
]

Tests: {
	{"Input" -> 4} -> "3_cwgaM9SDvGc_fcjOSo+tB00=1.324527e+0",
	{"Input" -> {3, 5}} -> "2*5_GiF+/cMRWSs_RhCCsyun4m8=4.582082e+0",
	{"Input" -> "x"} -> "9_Av5qBiAQoY4_BARaLMeKn0w=3.590024e+0",
	{"Input" -> {"x", 2}} -> "9*2_XNd32+p2Tso_TTPZX3vsNJU=7.780909e+0",
	{"Input" -> {"x", 2, 2}} -> "9*2*2_QIJ6v73E8EY_fauS+f6AEKE=1.558926e+1",
	{"Input" -> {"x", 2, 2, Restricted["Integer", 100]}} -> "9*2*2_Tv2YqZ/GAtQ_cADalGIrmBQ=1.761000e+3"
}