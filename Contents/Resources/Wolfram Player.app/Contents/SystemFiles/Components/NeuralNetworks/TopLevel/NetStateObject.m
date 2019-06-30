Package["NeuralNetworks`"]


PackageScope["RecurrentStateContainer"]


PackageExport["NetStateObject"]

(ns_NetStateObject ? System`Private`HoldEntryQ) := 
	UseMacros @ RuleCondition @ CatchFailureAsMessage[NetStateObject, make[ns]];

NetStateObject::arg1 = "First argument to NetStateObject should be a valid net."
NetStateObject::nostate = "Cannot create a NetStateObject for a net that has no recurrent state."
NetStateObject::invistate = "``, which was provided as a key for an interior state, is not a valid ."
NetStateObject::arg2 = "Second argument to NetStateObject should be Automatic or an association of initial states."

SetHoldAllComplete[make];
make[NetStateObject[net_, seed_:Automatic]] := ModuleScope[
	If[!ValidNetQ[net], ThrowFailure["arg1"]];
	If[!FullySpecifiedNetQ[net], ThrowNotSpecifiedFailure[net, "create states for", "Initialized"]];

	evals = 0;
	netData = NData[net];

	istates = GetInteriorStates[net];

	Switch[seed,
	Automatic,
		state = Map[
			CreateConstantNumericArray[TDimensions[netData @@ #], 0.]&,
			Values @ istates
		],
	_Association,
		extra = Complement[Keys[seed], Keys[istates]];
		If[extra =!= {}, ThrowFailure["invrsinkey", First[extra]]];
		state = KeyValueMap[
			CoerceParam[StringForm["the initial state of ``", #1], Lookup[seed, Key[#1], 0], netData @@ #2]&,
			istates
		],
	_,
		ThrowFailure["arg2"]
	];

	If[state === {}, ThrowFailure["nostate"]];
	state = RecurrentStateContainer[state];

	assoc = Association[
		"Net" -> net,
		"SessionID" -> $SessionID,
		"RecurrentStateKeys" -> Keys[istates],
		"RecurrentStateArrays" -> Hold[state],
		"EvaluationCount" -> Hold[evals]
	];

	System`Private`ConstructNoEntry[NetStateObject, assoc] 
];

NetStateObject::oldsess = "NetStateObject expressions cannot persist between kernel sessions. Please create a new NetStateObject from the original net.";

make[ns:NetStateObject[assoc_Association]] := Scope[
	If[assoc["SessionID"] =!= $SessionID, ThrowFailure["oldsess"]];
	System`Private`HoldSetNoEntry[ns]
];

make[_] := Fail;


PackageScope["GetNetStateObjectCurrentStates"]

GetNetStateObjectCurrentStates[HoldPattern[NetStateObject[assoc_Association]] ? System`Private`HoldNoEntryQ] :=
	AssociationThread[
		assoc["RecurrentStateKeys"],
		First @ ReleaseHold @ assoc["RecurrentStateArrays"]
	];

(ns_NetStateObject ? System`Private`HoldNoEntryQ)[args___] := CatchFailureAsMessage @ Block[
	{recurrentStateArrays, net, evaluationCount, $RSData},
	UnpackAssociation[getAssoc[ns], net, recurrentStateArrays, evaluationCount];
	$RSData = First[recurrentStateArrays];
	res = NetApply[net, args];
	(* TODO: Optimize to use NetApplyFast type technique, also support properties and
	arguments *)
	Replace[recurrentStateArrays, Hold[s_] :> (s = $RSData)];
	Replace[evaluationCount, Hold[s_] :> s++];
	res
];

SetHoldAll[getAssoc];
getAssoc[NetStateObject[assoc_]] := assoc;

DefineCustomBoxes[NetStateObject, ns_NetStateObject ? System`Private`HoldNoEntryQ :> MakeNetStateObjectBoxes[ns]];

$NetStateObjectIconBoxes := $NetStateObjectIconBoxes = Uncompress @ "
1:eJxTTMoPSmNhYGAo5gYS7kWJBRmZycVO+RVpEiBBkIxPZnFJGiOIxwkkQoDy2XmpxcVF1SLr3B9WLb
GHaIerZAbxOIBEkLuTc35OflGRdrvYzXPfn9tjMiDGsgMJ/4LE5MySyiIGMPgAlQEZ45WfmeeWX5QbzA
oyM780LwViBR9ULjXFubSoLBXkZEYUhzBhOgvGywRZkglSAGZhk2SESzLiNgfVDqjjBRygDDEH/PIWWO
UbDLHJB5XmpIKjCOxZ55z84tQUVJdBHEsoLha5bvv898oTe0zGsIwLCVhY8mCPC7i8Gfa4MMYmT2xcoD
oPHoiuKempoEAkGFcny/bNl9J/aI/JIBRXODIqRjyi+hkcq76ZJalFUONUHBBa3BKTUzG1YHE1zCWYDD
yuZoKlIrfMnBycqYgZN4+kVERCEkONI2xJyAyWRBiwJzEFYuVNsMubIcmzEgh78ktakIxzYgFa5qY84+
OJQPQow1diYAsYNVjAiKAH3KmFwDLNVg+bPLF5d+iF9KDMHPA40MKeuI2JlVfCLq+HTZ7YOEZNYqSXdK
/UDDnWyNyyL1r18ZJvkgCwLvXiYQImg5PDtKTThAW2OvbIUCZWXhe7vCYueTLiJmiHXOvrwBv2RfVZe0
omS7ywh9aCb4Zp3MDDVpVA2BKSlyMQd3KjtRD2EsiQQAlmOAAl1EjLBdqwwJbC3pw3IFYeR3dBG5c8GX
EjGGG55UTZK/uiSW9r7E3jgNXIAin9uypsB4dn3CC6tdhLoAYTYuWxl1AI89HlyYibx0tnH1HY8Mgemo
Ee2Bfxgir2m6+GZ9wg0jWOsDcgVh5H7aFNxbgZqfmmwQh/viAkz6BPIN+gy4/mG0JxA28V4Qh7eKuKkD
yOuEG0uvRHW13Y+/44Wl1YW2WjrS7iJMmtPXCVQAZEyuPKBdqjuYBA3wPH6AjWEbLR0RHiJMkeHcHRd1
AmVh573wRhvtRoLoAGzKTT9R77sY/zguWfiMiefPrfaCBzwQhrESFakzh62CbEymPPJQjzeUZzAfaynt
BIudYA5IIR1mdDtFiwj3QjWjyE5HGMpGsjyY/mAqSAwTraik1+IHLBiK0LcKRyE2LlsecChPnq2CLTsb
ggNbkkKLEkMx+8HsKxtCQ/F8hLRlMLkvPMTUxPDc6sSsWaaCwd4AwsegNy8oG25KWnEpE8GRiwmIRdHg
DMDGPP";

SetHoldAll[MakeNetStateObjectBoxes];
MakeNetStateObjectBoxes[ns:NetStateObject[assoc_]] := Scope[
	UnpackAssociation[assoc, evaluationCount, recurrentStateArrays, recurrentStateKeys];
	count = Length @ First @ ReleaseHold[recurrentStateArrays];
	info = Association[
		"Evaluation count" -> With[ec = evaluationCount, Replace[ReleaseHold @ ec, Except[_Integer] :> ""]],
		"Recurrent state" -> Row[{count, If[count === 1, " array", " arrays"]}]
	];
	extInfo = Association[
		"Recurrent state positions" -> recurrentStateKeys
	];
	icon = PaneBox[StyleBox["\:267a", 20], ImageSize -> {20, 24}, Alignment -> {Center, Center}];
	OptimizedArrangeSummaryBox[
		NetStateObject, $NetStateObjectIconBoxes, 
		fmtEntries[info], fmtEntries[extInfo], False
	]
];


NetStateObject /: Normal[HoldPattern[NetStateObject[assoc_] ? System`Private`HoldNoEntryQ]] :=
	assoc["Net"];