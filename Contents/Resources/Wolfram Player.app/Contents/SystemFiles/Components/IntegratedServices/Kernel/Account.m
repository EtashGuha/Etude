(* Mathematica Package *)

Unprotect[System`$ServiceCreditsAvailable];
Clear[System`$ServiceCreditsAvailable];

BeginPackage["IntegratedServices`"]

System`$ServiceCreditsAvailable::error = "There was a problem while checking your current Service Credits balance. Please try again.";
System`$ServiceCreditsAvailable::notauth = "You need to be authenticated with Wolfram Cloud server. Please try again after CloudConnect[].";
System`$ServiceCreditsAvailable::offline = "The Wolfram Language is currently configured not to use the Internet. To allow Internet use, check the \"Allow the Wolfram Language to use the Internet\" box in the Help \[FilledRightTriangle] Internet Connectivity dialog.";

Begin["`Private`"]

ClearAttributes[{$serviceCreditExpirationTime,$serviceCreditUpdateTime}, {Protected, ReadProtected}]
ClearAll[$serviceCreditExpirationTime,$serviceCreditUpdateTime];

System`$ServiceCreditsAvailable:= IntegratedServices`ServiceCreditsAvailable[]

$serviceCreditExpirationTime = 600;
$serviceCreditUpdateTime = 90;

serviceCreditTimestamp[_] = Missing[];
serviceCredits[_] = Missing[];

IntegratedServices`ServiceCreditsAvailable[] := (Message[System`$ServiceCreditsAvailable::offline]; $Failed)/;(!PacletManager`$AllowInternet)

IntegratedServices`ServiceCreditsAvailable[]:= With[{id = $WolframUUID},
	Catch[serviceCreditsAvailable[id]]
] /; $CloudConnected

IntegratedServices`ServiceCreditsAvailable[args___]:= With[{connected = CloudConnect[]},
	If[$CloudConnected,
		IntegratedServices`ServiceCreditsAvailable[]
		,
		Message[System`$ServiceCreditsAvailable::notauth];
		connected
	]
]

serviceCreditsAvailable[userUUID_]:=
	If[getnewservicecreditsQ[userUUID],
		getServiceCreditsSynchronous[userUUID]
		,
		If[updatecreditsasynchQ[userUUID],
			getServiceCreditsAsynchronous[userUUID]
			];
		serviceCredits[userUUID]
	]

getnewservicecreditsQ[userUUID_] := !IntegerQ[serviceCreditTimestamp[userUUID]] || (UnixTime[] > serviceCreditTimestamp[userUUID] + $serviceCreditExpirationTime)

getServiceCreditsSynchronous[userUUID_]:= With[{resp = Quiet[URLExecute[CloudObject[$ServiceCreditsRemainingEndpoint],"WL",VerifySecurityCertificates->$CloudPRDBaseQ]]},
	If[AssociationQ[resp] && TrueQ[resp["success"]],
		storeServiceCredits[userUUID, resp, True];
		serviceCredits[userUUID]
		,
		Message[System`$ServiceCreditsAvailable::error];
		Throw[$Failed]
	]
]

updatecreditsasynchQ[userUUID_]:= (UnixTime[] > serviceCreditTimestamp[userUUID] + $serviceCreditUpdateTime)

getServiceCreditsAsynchronous[userUUID_String]:= URLSubmit[
	CloudObject[$ServiceCreditsRemainingEndpoint],
	HandlerFunctions -> <|"TaskFinished" -> (storeServiceCreditsAsynchronous[userUUID,#]&)|>,
	HandlerFunctionsKeys -> {"StatusCode", "StatusCodeDescription", "Body"},
	VerifySecurityCertificates -> False
]

storeServiceCreditsAsynchronous[userUUID_,data_]:= With[{resp = Quiet[ToExpression[data["Body"][[1]]]]},
	If[AssociationQ[resp] && TrueQ[resp["success"]],
		storeServiceCredits[userUUID, resp]
	]
]

storeServiceCredits[userUUID_, response_, initialQ : (True | False) : False] := With[{new = parseCreditLevel[response]},
	If[!FailureQ[new],
		serviceCreditTimestamp[userUUID] = UnixTime[];
		serviceCredits[userUUID] = new,
		If[initialQ,
			Message[System`$ServiceCreditsAvailable::error];
			Throw[$Failed]
		]
	];
]

parseCreditLevel[response_]:= With[{res = parseCreditLevel0[response]},
	res /; IntegerQ[res]
]

parseCreditLevel0[response_]:= response["data"]["creditSummary"]["serviceCredits"]

parseCreditLevel[___]:= $Failed


updateAccountInfo[as_Association]:= updateaccountInfo[Lookup[as,"AccountInformation"],Lookup[as,"User"]]

updateaccountInfo[info_Association, userUUID_String]:= updateaccountinfo[Lookup[info,"ServiceCreditsRemaining"], userUUID]

updateaccountinfo[balance_?IntegerQ,userUUID_String]:= (serviceCredits[userUUID] = balance)

updateaccountinfo[balance0_String,userUUID_String]:= With[{balance = Quiet[ToExpression[balance0]]},
	If[IntegerQ[balance],
		serviceCredits[userUUID] = balance
	]
]


SetAttributes[{$serviceCreditExpirationTime,$serviceCreditUpdateTime}, {Protected, ReadProtected}]
SetAttributes[System`$ServiceCreditsAvailable, {Protected, ReadProtected}]

End[]

EndPackage[]
