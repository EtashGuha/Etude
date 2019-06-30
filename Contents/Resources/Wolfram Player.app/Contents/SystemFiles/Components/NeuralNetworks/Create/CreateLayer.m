Package["NeuralNetworks`"]



PackageScope["ConstructNet"]

ConstructNet[assoc_Association] := 
	ConstructNet[InheritAmbientSharedArrays @ assoc, $StandardMetadata];

ConstructNet[assoc_Association, meta_Association] := 
	System`Private`ConstructNoEntry[NSymbol[assoc], assoc, meta];


PackageScope["ConstructUpgradedNet"]

ConstructUpgradedNet[net_, meta_] :=
	ConstructNet[net, Append[meta, $StandardMetadata]];


PackageScope["ValidNetAssocQ"]

ValidNetAssocQ[assoc_ ? AssociationQ] := And[
	KeyExistsQ[$TypeToSymbol, assoc["Type"]],
	KeyExistsQ[assoc, "Inputs"],
	KeyExistsQ[assoc, "Outputs"]
];

ValidNetAssocQ[___] := False;


PackageScope["CreateNet"]
PackageScope["$strictLayerArgs"]

$strictLayerArgs = True;

SetHoldRest[CreateNet];

CreateNet[name_String, net:(_Symbol[data_Association /; ValidNetAssocQ[data]])] := 
	UpgradeAndSeal11V0Net[net];

CreateNet[name_String, net:(_Symbol[data_Association /; ValidNetAssocQ[data], _Association])] := 
	SealNet[net];

CreateNet[name_String, head_Symbol[args___]] := iCreateNet[name, head, args];

iCreateNet[name_String, head_Symbol, Verbatim[Blank][___] | Verbatim[Pattern][_, _]] := 
	Fail;

General::netnargs = "`` arguments were provided, expected ``."

iCreateNet[name_, head_, args___] := Scope[
	
	data = $LayerData[name];

	$interiorStates = <||>;

	args = {args};

	{arrays, sarrays, params, inputs, outputs, states} = ParseArguments[head, True, data, args];

	subnets = data["SubNets"];
	istates = <||>;
	If[subnets =!= {},
		If[!data["StateExpanding"],
			(* this is only non-trivial for operators containing nets with their own unattached states *)
			istates = Association[interiorStateRules[#1][#2, params[#2]]& @@@ subnets];
		];
		(* strip shared arrays from our subnets and hoist them up to our level *)
		{params, sarrays2} = HoistSharedArrays[params, subnets[[All, 2]]];
		JoinTo[sarrays, sarrays2];
	];

	assoc = Association[{
		"Type" -> name,
		"Arrays" -> arrays,
		"Parameters" -> params,
		"Inputs" -> inputs,
		"Outputs" -> outputs,
		If[sarrays === <||>, Nothing, 
			sauxarray = FirstPosition[KeyTake[arrays, data["AuxArrays"]], _NetSharedArray];
			If[!MissingQ[sauxarray], 
				General::noauxsa = "Array `` cannot be shared.";
				ThrowFailure["noauxsa", sauxarray[[1,1]]]
			];
			"SharedArrays" -> sarrays
		],
		If[states === <||>, Nothing, "States" -> states],
		If[istates === <||>, Nothing, "InteriorStates" -> istates]
	}];

	{irules, pfuncs} = ReapInferenceRules[assoc];
	pcf = data["PostConstructionFunction"];
	If[pcf =!= None, AppendTo[pfuncs, pcf]];
	
	assoc = DoInference[assoc, irules, pfuncs];	

	CheckPortsForBadLengthVars[assoc];

	System`Private`ConstructNoEntry[head, assoc, $StandardMetadata]
]


PackageScope["$StandardMetadata"]

$StandardMetadata := Association[
	"Version" -> $NeuralNetworksVersionNumber,
	"Unstable" -> $NeuralNetworkFormatIsUnstable
];


PackageExport["NetSetMetadata"]

NetSetMetadata[NetP[data, meta], newmeta_] := 
	ConstructNet[data, Association[meta, newmeta]]


PackageExport["NetGetMetadata"]

NetGetMetadata[NetP[data, meta]] := meta;
